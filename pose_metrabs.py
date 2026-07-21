#!/usr/bin/env python3
"""
pose_metrabs.py
BioVision gAIt – MeTRAbs 3D Pose Estimation Wrapper
Optimized for Apple Silicon (M4) – CPU inference

MeTRAbs (Metric-scale Truncation-robust Estimation of 3D Human Body Poses)
https://github.com/isarandi/metrabs

Install:
  pip install tensorflow tensorflow_hub opencv-python

Usage via TensorFlow Hub (no git clone needed):
  import tensorflow_hub as hub
  model = hub.load('https://bit.ly/metrabs_l')  # EfficientNetV2-L
  # or: https://bit.ly/metrabs_mob3  # MobileNetV3, fastest
  pred = model.detect_poses(image_bgr, skeleton='h36m17')
  # pred['poses3d'] -> (n_people, 17, 3) in mm
  # pred['poses2d'] -> (n_people, 17, 2) in px

H36M 17-joint skeleton (MeTRAbs output):
  0: pelvis      6: ankle_l     12: elbow_l
  1: hip_r       7: spine       13: wrist_l
  2: knee_r      8: neck        14: shoulder_r
  3: ankle_r     9: nose        15: elbow_r
  4: hip_l       10: head       16: wrist_r
  5: knee_l      11: shoulder_l
"""

import json
import warnings
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError:
    raise ImportError("opencv-python required: pip install opencv-python")

# ---------------------------------------------------------------------------
# MeTRAbs via TensorFlow Hub – with RTMPose fallback
# ---------------------------------------------------------------------------

METRABS_AVAILABLE = False
RTMLIB_AVAILABLE = False
_tfhub_model = None

try:
    import tensorflow as tf
    import tensorflow_hub as hub
    METRABS_AVAILABLE = True
except ImportError:
    warnings.warn(
        "TensorFlow / tfhub not installed. "
        "Install with: pip install tensorflow tensorflow_hub\n"
        "Falling back to RTMPose/rtmlib if available."
    )
    try:
        from rtmlib import RTMPose, YOLOX
        RTMLIB_AVAILABLE = True
    except ImportError:
        pass

# ---------------------------------------------------------------------------
# Joint definitions – H36M 17
# ---------------------------------------------------------------------------

H36M_JOINT_NAMES = [
    "pelvis", "hip_r", "knee_r", "ankle_r",
    "hip_l", "knee_l", "ankle_l",
    "spine", "neck", "nose", "head",
    "shoulder_l", "elbow_l", "wrist_l",
    "shoulder_r", "elbow_r", "wrist_r",
]

H36M_IDX = {name: i for i, name in enumerate(H36M_JOINT_NAMES)}

CameraView = Literal["sagittal", "frontal", "rear"]

# MeTRAbs TF-Hub model URLs
METRABS_MODELS = {
    "efficientnetv2_l": "https://bit.ly/metrabs_l",      # best accuracy
    "efficientnetv2_s": "https://bit.ly/metrabs_s",     # faster, good for M4
    "mobilenetv3": "https://bit.ly/metrabs_mob3",       # fastest
    # fallbacks – use _l if _s / mob3 404s
}

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_metrabs_model(
    backend: str = "efficientnetv2_s",
    device: str = "auto"
) -> object:
    """
    Load MeTRAbs via TensorFlow Hub.

    Args:
        backend: "efficientnetv2_l" | "efficientnetv2_s" | "mobilenetv3"
        device: ignored – TF handles CPU/MPS automatically

    Returns:
        TF-Hub model with .detect_poses() method
    """
    global _tfhub_model
    if _tfhub_model is not None:
        return _tfhub_model

    if not METRABS_AVAILABLE:
        raise RuntimeError(
            "TensorFlow / tfhub not installed.\n"
            "  pip install tensorflow tensorflow_hub"
        )

    # Map backend names to TF-Hub URLs
    url_map = {
        "efficientnetv2_l": "https://bit.ly/metrabs_l",
        "efficientnetv2_s": "https://bit.ly/metrabs_s",
        "mobilenetv3": "https://bit.ly/metrabs_mob3",
        "resnet50": "https://bit.ly/metrabs_l",  # fallback
    }
    model_url = url_map.get(backend, "https://bit.ly/metrabs_eff2s_y4")

    print(f"[MeTRAbs] Loading from TF-Hub: {model_url}")
    print("  (first run downloads ~80-300 MB – one time only)")
    try:
        model = hub.load(model_url)
    except Exception as e:
        # Try fallbacks
        for fallback_url in [
            "https://bit.ly/metrabs_l",
            "https://bit.ly/metrabs_eff2s_y4",
        ]:
            if fallback_url == model_url:
                continue
            try:
                print(f"  Trying fallback: {fallback_url}")
                model = hub.load(fallback_url)
                break
            except Exception:
                continue
        else:
            raise RuntimeError(f"Could not load MeTRAbs model from TF-Hub: {e}")

    _tfhub_model = model
    print("[MeTRAbs] Model loaded ✓")
    return model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_metrabs_inference(
    video_path: str,
    camera_view: CameraView = "sagittal",
    model: Optional[object] = None,
    backend: str = "efficientnetv2_s",
    conf_threshold: float = 0.3,
    output_dir: Optional[str] = None,
    draw_overlay: bool = True,
) -> Dict:
    """
    Run MeTRAbs 3D pose estimation on a gait video.

    Returns dict with keypoints_3d (n_frames, 17, 3) in mm,
    keypoints_2d (n_frames, 17, 2) in px, confidences, fps, etc.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    if output_dir is None:
        output_dir = video_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load model / fallback ---
    use_metrabs = METRABS_AVAILABLE
    if model is None and use_metrabs:
        try:
            model = load_metrabs_model(backend=backend)
        except Exception as e:
            print(f"[MeTRAbs] Load failed: {e}")
            use_metrabs = False

    if not use_metrabs:
        if RTMLIB_AVAILABLE:
            print("[pose_metrabs] MeTRAbs unavailable – using RTMPose fallback")
            return _run_rtmpose_fallback(
                str(video_path), camera_view, conf_threshold,
                output_dir, draw_overlay
            )
        raise RuntimeError(
            "No pose backend available.\n"
            "Install MeTRAbs: pip install tensorflow tensorflow_hub\n"
            "Or RTMPose fallback: pip install rtmlib onnxruntime"
        )

    # --- Video I/O ---
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[{camera_view}] {video_path.name} – {n_frames_total} frames @ {fps:.1f} FPS, {width}x{height}")

    overlay_writer = None
    overlay_path = None
    if draw_overlay:
        overlay_path = Path(output_dir) / f"overlay_{video_path.stem}_{camera_view}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        overlay_writer = cv2.VideoWriter(str(overlay_path), fourcc, fps, (width, height))

    # --- Inference loop ---
    keypoints_3d_all = []
    keypoints_2d_all = []
    confidences_all = []

    frame_idx = 0
    import tensorflow as tf

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        # MeTRAbs TF-Hub: model.detect_poses(image, skeleton='h36m17')
        # image: uint8 tensor [H, W, 3], BGR or RGB? – try BGR first, TF-Hub models usually expect RGB
        # Docs say the model handles color internally, try RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img_tensor = tf.convert_to_tensor(frame_rgb, dtype=tf.uint8)

        try:
            pred = model.detect_poses(img_tensor, skeleton='h36m17')
            # pred is dict: {'boxes', 'poses3d', 'poses2d'}
            poses3d = pred['poses3d'].numpy()  # (n_people, 17, 3) mm
            poses2d = pred['poses2d'].numpy()  # (n_people, 17, 2) px

            if poses3d.shape[0] > 0:
                kpt_3d = poses3d[0].astype(np.float32)
                kpt_2d = poses2d[0].astype(np.float32)
                conf = np.ones(17, dtype=np.float32)
            else:
                kpt_3d = np.zeros((17, 3), dtype=np.float32)
                kpt_2d = np.zeros((17, 2), dtype=np.float32)
                conf = np.zeros(17, dtype=np.float32)
        except Exception as e:
            if frame_idx == 0:
                print(f"[MeTRAbs] Inference error: {e}")
                print("  Check: pip install tensorflow tensorflow_hub")
            kpt_3d = np.zeros((17, 3), dtype=np.float32)
            kpt_2d = np.zeros((17, 2), dtype=np.float32)
            conf = np.zeros(17, dtype=np.float32)

        keypoints_3d_all.append(kpt_3d)
        keypoints_2d_all.append(kpt_2d)
        confidences_all.append(conf)

        # Overlay
        if overlay_writer is not None:
            vis = draw_skeleton_overlay(frame_bgr, kpt_2d, kpt_3d, camera_view=camera_view)
            overlay_writer.write(vis)

        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"  ... frame {frame_idx}/{n_frames_total}")

    cap.release()
    if overlay_writer is not None:
        overlay_writer.release()
        print(f"Overlay saved: {overlay_path}")

    keypoints_3d = np.stack(keypoints_3d_all, axis=0).astype(np.float32) if keypoints_3d_all else np.zeros((0, 17, 3), np.float32)
    keypoints_2d = np.stack(keypoints_2d_all, axis=0).astype(np.float32) if keypoints_2d_all else np.zeros((0, 17, 2), np.float32)
    confidences = np.stack(confidences_all, axis=0).astype(np.float32) if confidences_all else np.zeros((0, 17), np.float32)
    n_frames = len(keypoints_3d_all)

    # Save JSON
    json_path = Path(output_dir) / f"keypoints_{video_path.stem}_{camera_view}.json"
    with open(json_path, "w") as f:
        json.dump({
            "video": str(video_path),
            "camera_view": camera_view,
            "fps": float(fps),
            "n_frames": int(n_frames),
            "width": int(width),
            "height": int(height),
            "joint_names": H36M_JOINT_NAMES,
            "keypoints_3d_mm": keypoints_3d.tolist(),
            "keypoints_2d_px": keypoints_2d.tolist(),
            "confidences": confidences.tolist(),
            "backend": "metrabs_tfhub",
        }, f)
    print(f"Keypoints saved: {json_path}  ({n_frames} frames)")

    return {
        "keypoints_3d": keypoints_3d,
        "keypoints_2d": keypoints_2d,
        "confidences": confidences,
        "fps": fps,
        "n_frames": n_frames,
        "camera_view": camera_view,
        "overlay_path": str(overlay_path) if overlay_path else None,
        "json_path": str(json_path),
    }


# ---------------------------------------------------------------------------
# Skeleton drawing
# ---------------------------------------------------------------------------

H36M_SKELETON_EDGES = [
    (H36M_IDX["pelvis"], H36M_IDX["hip_r"]),
    (H36M_IDX["hip_r"], H36M_IDX["knee_r"]),
    (H36M_IDX["knee_r"], H36M_IDX["ankle_r"]),
    (H36M_IDX["pelvis"], H36M_IDX["hip_l"]),
    (H36M_IDX["hip_l"], H36M_IDX["knee_l"]),
    (H36M_IDX["knee_l"], H36M_IDX["ankle_l"]),
    (H36M_IDX["pelvis"], H36M_IDX["spine"]),
    (H36M_IDX["spine"], H36M_IDX["neck"]),
    (H36M_IDX["neck"], H36M_IDX["nose"]),
    (H36M_IDX["nose"], H36M_IDX["head"]),
    (H36M_IDX["neck"], H36M_IDX["shoulder_l"]),
    (H36M_IDX["shoulder_l"], H36M_IDX["elbow_l"]),
    (H36M_IDX["elbow_l"], H36M_IDX["wrist_l"]),
    (H36M_IDX["neck"], H36M_IDX["shoulder_r"]),
    (H36M_IDX["shoulder_r"], H36M_IDX["elbow_r"]),
    (H36M_IDX["elbow_r"], H36M_IDX["wrist_r"]),
]

def draw_skeleton_overlay(
    frame_bgr: np.ndarray,
    kpts_2d: np.ndarray,
    kpts_3d: np.ndarray,
    camera_view: CameraView = "sagittal",
) -> np.ndarray:
    """Draw 3D skeleton overlay on a video frame."""
    vis = frame_bgr.copy()
    h, w = vis.shape[:2]

    # If 2D is all zeros, project 3D orthographically
    if np.all(kpts_2d == 0) and np.any(kpts_3d != 0):
        pts = kpts_3d.copy().astype(np.float32)
        valid = np.linalg.norm(pts, axis=1) > 1
        if np.any(valid):
            pts_valid = pts[valid]
            # MeTRAbs 3D: X=right, Y=up, Z=away
            # Project X/Y to image
            x_min, x_max = pts_valid[:, 0].min(), pts_valid[:, 0].max()
            y_min, y_max = pts_valid[:, 1].min(), pts_valid[:, 1].max()
            scale = min(w * 0.6 / max(x_max - x_min, 1),
                        h * 0.6 / max(y_max - y_min, 1))
            cx, cy = w // 2, h // 2
            kpts_2d = np.zeros((17, 2), dtype=np.float32)
            kpts_2d[valid, 0] = cx + (pts[valid, 0] - pts_valid[:, 0].mean()) * scale
            kpts_2d[valid, 1] = cy - (pts[valid, 1] - pts_valid[:, 1].mean()) * scale

    # Draw bones
    for a, b in H36M_SKELETON_EDGES:
        xa, ya = kpts_2d[a]
        xb, yb = kpts_2d[b]
        if xa == 0 and ya == 0: continue
        if xb == 0 and yb == 0: continue
        cv2.line(vis, (int(xa), int(ya)), (int(xb), int(yb)), (0, 255, 0), 2)

    # Draw joints
    for i, (x, y) in enumerate(kpts_2d):
        if x == 0 and y == 0: continue
        color = (0, 200, 255) if i in [1,2,3,14,15,16] else (255, 180, 0)
        cv2.circle(vis, (int(x), int(y)), 4, color, -1)

    # View tag
    view_colors = {"sagittal": (255, 255, 0), "frontal": (0, 255, 255), "rear": (255, 0, 255)}
    cv2.rectangle(vis, (10, 10), (150, 45), (0, 0, 0), -1)
    cv2.putText(vis, camera_view.upper(), (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, view_colors.get(camera_view, (255,255,255)), 2)
    return vis


# ---------------------------------------------------------------------------
# RTMPose / rtmlib fallback
# ---------------------------------------------------------------------------

def _run_rtmpose_fallback(
    video_path: str,
    camera_view: CameraView,
    conf_threshold: float,
    output_dir: Path,
    draw_overlay: bool,
) -> Dict:
    """Fallback pose estimation using RTMPose via rtmlib.
    Returns 2D keypoints only – Z=0. Compatible with metrics_gait.py
    (angles computed in 2D, with view-appropriate plane selection).
    """
    try:
        from rtmlib import PoseTracker, Body, draw_skeleton
    except ImportError:
        raise RuntimeError("rtmlib not installed – pip install rtmlib onnxruntime")

    print(f"[RTMPose fallback] {video_path}  view={camera_view}")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    pose_tracker = PoseTracker(
        Body, mode='balanced',
        backend='onnxruntime', device='cpu'
    )

    overlay_writer = None
    overlay_path = None
    if draw_overlay:
        overlay_path = output_dir / f"overlay_{Path(video_path).stem}_{camera_view}_rtmpose.mp4"
        overlay_writer = cv2.VideoWriter(str(overlay_path),
            cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    # RTMPose COCO-17 → H36M 17 mapping
    # COCO: 0 nose,1 L_eye,2 R_eye,3 L_ear,4 R_ear,
    # 5 L_shoulder, 6 R_shoulder, 7 L_elbow, 8 R_elbow,
    # 9 L_wrist, 10 R_wrist, 11 L_hip, 12 R_hip,
    # 13 L_knee, 14 R_knee, 15 L_ankle, 16 R_ankle
    # H36M: 0 pelvis, 1 hip_r, 2 knee_r, 3 ankle_r,
    #  4 hip_l, 5 knee_l, 6 ankle_l, 7 spine, 8 neck,
    #  9 nose, 10 head, 11 shoulder_l, 12 elbow_l, 13 wrist_l,
    #  14 shoulder_r, 15 elbow_r, 16 wrist_r
    coco_to_h36m = [
        11,  # 0 pelvis <- avg L/R hip
        12,  # 1 hip_r <- R_hip (12)
        14,  # 2 knee_r <- R_knee (14)
        16,  # 3 ankle_r <- R_ankle (16)
        11,  # 4 hip_l <- L_hip (11)
        13,  # 5 knee_l <- L_knee (13)
        15,  # 6 ankle_l <- L_ankle (15)
        5,   # 7 spine <- avg shoulders
        5,   # 8 neck <- avg shoulders
        0,   # 9 nose <- nose
        0,   # 10 head <- nose
        5,   # 11 shoulder_l <- L_shoulder
        7,   # 12 elbow_l <- L_elbow
        9,   # 13 wrist_l <- L_wrist
        6,   # 14 shoulder_r <- R_shoulder
        8,   # 15 elbow_r <- R_elbow
        10,  # 16 wrist_r <- R_wrist
    ]

    kpts_2d_all = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        keypoints, scores = pose_tracker(frame)
        if keypoints is not None and len(keypoints) > 0:
            kp_coco = keypoints[0]  # (17, 2)
            # Map COCO → H36M
            h36m_2d = np.zeros((17, 2), dtype=np.float32)
            # Direct 1:1 mappings
            mapping = {
                1: 12, 2: 14, 3: 16,   # R hip/knee/ankle
                4: 11, 5: 13, 6: 15,   # L hip/knee/ankle
                11: 5, 12: 7, 13: 9,   # L shoulder/elbow/wrist
                14: 6, 15: 8, 16: 10,  # R shoulder/elbow/wrist
            }
            for h36m_i, coco_i in mapping.items():
                if coco_i < kp_coco.shape[0]:
                    h36m_2d[h36m_i] = kp_coco[coco_i]
            # pelvis = avg hips
            h36m_2d[0] = (h36m_2d[1] + h36m_2d[4]) / 2
            # spine/neck = avg shoulders
            shoulder_mid = (h36m_2d[11] + h36m_2d[14]) / 2
            h36m_2d[7] = shoulder_mid
            h36m_2d[8] = shoulder_mid
            # nose/head
            if kp_coco.shape[0] > 0:
                h36m_2d[9] = kp_coco[0]
                h36m_2d[10] = kp_coco[0]
        else:
            h36m_2d = np.zeros((17, 2), dtype=np.float32)

        kpts_2d_all.append(h36m_2d)

        if overlay_writer is not None:
            vis = frame.copy()
            if keypoints is not None and len(keypoints) > 0:
                try:
                    vis = draw_skeleton(vis, keypoints, scores, kpt_thr=conf_threshold)
                except Exception:
                    for x, y in keypoints[0]:
                        if x > 0 and y > 0:
                            cv2.circle(vis, (int(x), int(y)), 3, (0, 255, 0), -1)
            cv2.putText(vis, f"{camera_view.upper()} – RTMPose", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            overlay_writer.write(vis)
        frame_idx += 1

    cap.release()
    if overlay_writer:
        overlay_writer.release()

    kpts_2d = np.stack(kpts_2d_all, axis=0) if kpts_2d_all else np.zeros((0, 17, 2), np.float32)
    n_frames = kpts_2d.shape[0]
    kpts_3d = np.zeros((n_frames, 17, 3), dtype=np.float32)
    kpts_3d[:, :, :2] = kpts_2d
    conf = np.ones((n_frames, 17), dtype=np.float32)

    json_path = output_dir / f"keypoints_{Path(video_path).stem}_{camera_view}_rtmpose.json"
    with open(json_path, "w") as f:
        json.dump({
            "video": video_path,
            "camera_view": camera_view,
            "fps": float(fps),
            "n_frames": int(n_frames),
            "width": int(width),
            "height": int(height),
            "joint_names": H36M_JOINT_NAMES,
            "keypoints_3d_mm": kpts_3d.tolist(),
            "keypoints_2d_px": kpts_2d.tolist(),
            "confidences": conf.tolist(),
            "backend": "rtmpose",
        }, f)

    return {
        "keypoints_3d": kpts_3d,
        "keypoints_2d": kpts_2d,
        "confidences": conf,
        "fps": fps,
        "n_frames": n_frames,
        "camera_view": camera_view,
        "overlay_path": str(overlay_path) if overlay_path else None,
        "json_path": str(json_path),
    }


# ---------------------------------------------------------------------------
# Multi-view batch
# ---------------------------------------------------------------------------

def run_multiview_inference(
    videos: Dict[CameraView, str],
    **kwargs
) -> Dict[CameraView, Dict]:
    """Process 1-3 videos from different camera views."""
    results = {}
    for view, video_path in videos.items():
        print(f"\n{'='*60}\nProcessing {view} view: {video_path}\n{'='*60}")
        results[view] = run_metrabs_inference(
            video_path, camera_view=view, **kwargs
        )
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="BioVision gAIt – MeTRAbs 3D Pose")
    parser.add_argument("video", help="Input video file")
    parser.add_argument("--view", choices=["sagittal", "frontal", "rear"],
                        default="sagittal", help="Camera view")
    parser.add_argument("--backend", default="efficientnetv2_s",
                        help="efficientnetv2_l | efficientnetv2_s | mobilenetv3")
    parser.add_argument("--output-dir", "-o", default=None)
    parser.add_argument("--no-overlay", action="store_true")
    args = parser.parse_args()

    run_metrabs_inference(
        args.video,
        camera_view=args.view,
        backend=args.backend,
        output_dir=args.output_dir,
        draw_overlay=not args.no_overlay,
    )
