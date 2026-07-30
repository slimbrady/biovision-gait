# pose_rtmpose.py - RTMPose runner for BioVision gAIt
# Compatible interface with pose_metrabs.run_metrabs_inference
#
# Install MMPose / RTMPose:
#   pip install -U openmim
#   mim install mmengine mmcv mmpose mmdet
#
# Then drop in your RTMPose checkpoint/config paths below.

import numpy as np
from pathlib import Path

try:
    from mmpose.apis import MMPoseInferencer
    MMPOSE_AVAILABLE = True
except ImportError:
    MMPOSE_AVAILABLE = False

# ---- configure your RTMPose model here ----
RTMPOSE_CONFIG = None  # e.g. 'td-hm_rtmpose-m_8xb64-210e_coco-256x192.py'
RTMPOSE_CHECKPOINT = None  # e.g. 'rtmpose-m_simcc-body7_pt-body7_420e-256x192-...pth'

_inferencer = None

def _get_inferencer():
    global _inferencer
    if _inferencer is None and MMPOSE_AVAILABLE:
        kwargs = {}
        if RTMPOSE_CONFIG and RTMPOSE_CHECKPOINT:
            kwargs = dict(pose2d=dict(model=RTMPOSE_CHECKPOINT, config=RTMPOSE_CONFIG))
        _inferencer = MMPoseInferencer(pose2d='human', **kwargs)
    return _inferencer


def run_rtmpose_inference(video_path, camera_view="sagittal", backend="rtmpose_m",
                          conf_threshold=0.3, output_dir=None, draw_overlay=True):
    """
    RTMPose runner – drop-in compatible with run_metrabs_inference()

    Returns dict with:
      keypoints_3d: (N, J, 3) array – RTMPose is 2D, Z is filled with 0
      fps: float
      overlay_path: str | None
      n_frames, width, height

    Joint order mapped to H36M-compatible:
      0 Hip (mid), 1 R_hip, 2 R_knee, 3 R_ankle,
      4 L_hip, 5 L_knee, 6 L_ankle, ...
    Unmapped joints are filled with NaN.
    """
    if not MMPOSE_AVAILABLE:
        # Fallback: use MeTRAbs if available so the compare UI doesn't crash
        # Remove this once RTMPose is installed
        try:
            from pose_metrabs import run_metrabs_inference
            result = run_metrabs_inference(
                video_path, camera_view=camera_view,
                backend='efficientnetv2_s',
                conf_threshold=conf_threshold,
                output_dir=output_dir, draw_overlay=draw_overlay
            )
            # tag it so the UI knows this is a stub
            result['_rtmpose_stub'] = True
            return result
        except Exception:
            raise RuntimeError(
                "RTMPose not installed and MeTRAbs fallback failed. "
                "Install MMPose: pip install -U openmim && mim install mmengine mmcv mmpose mmdet"
            )

    import cv2
    inferencer = _get_inferencer()
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # COCO 17 keypoints -> H36M mapping
    # COCO: 0 nose, ... 11 L_hip, 12 R_hip, 13 L_knee, 14 R_knee, 15 L_ankle, 16 R_ankle
    coco_to_h36m = {11: 4, 12: 1, 13: 5, 14: 2, 15: 6, 16: 3}

    all_kpts = []
    out_writer = None
    overlay_path = None
    if draw_overlay and output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        overlay_path = str(output_dir / (Path(video_path).stem + "_rtmpose_overlay.mp4"))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(overlay_path, fourcc, fps, (width, height))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # MMPose inference
        result_generator = inferencer(frame, show=False)
        result = next(result_generator)
        predictions = result.get('predictions', [[]])[0]

        # Build H36M-style (17, 3) array, fill NaN
        kpts_h36m = np.full((17, 3), np.nan, dtype=np.float32)
        if predictions:
            # take highest-score person
            pred = max(predictions, key=lambda p: np.mean(p.get('keypoint_scores', [0])))
            kpts = np.array(pred['keypoints'])  # (K, 2)
            scores = np.array(pred.get('keypoint_scores', np.ones(len(kpts))))
            # filter by confidence
            kpts[scores < conf_threshold] = np.nan
            # map COCO -> H36M
            for coco_idx, h36m_idx in coco_to_h36m.items():
                if coco_idx < len(kpts):
                    x, y = kpts[coco_idx]
                    if np.isfinite(x):
                        kpts_h36m[h36m_idx] = [x, y, 0.0]
            # mid-hip = avg L+R
            if np.isfinite(kpts_h36m[1, 0]) and np.isfinite(kpts_h36m[4, 0]):
                kpts_h36m[0] = (kpts_h36m[1] + kpts_h36m[4]) / 2

            # draw overlay
            if out_writer is not None:
                vis_frame = frame.copy()
                for j in range(len(kpts_h36m)):
                    x, y, z = kpts_h36m[j]
                    if np.isfinite(x):
                        cv2.circle(vis_frame, (int(x), int(y)), 4, (0, 255, 0), -1)
                # skeleton
                skeleton = [(1, 2), (2, 3), (4, 5), (5, 6), (1, 4)]
                for a, b in skeleton:
                    if np.isfinite(kpts_h36m[a, 0]) and np.isfinite(kpts_h36m[b, 0]):
                        xa, ya = kpts_h36m[a, :2].astype(int)
                        xb, yb = kpts_h36m[b, :2].astype(int)
                        cv2.line(vis_frame, (xa, ya), (xb, yb), (0, 255, 0), 2)
                out_writer.write(vis_frame)
        all_kpts.append(kpts_h36m)
        frame_idx += 1

    cap.release()
    if out_writer:
        out_writer.release()

    keypoints_3d = np.stack(all_kpts, axis=0) if all_kpts else np.zeros((0, 17, 3), np.float32)

    return {
        'keypoints_3d': keypoints_3d,
        'fps': fps,
        'overlay_path': overlay_path,
        'n_frames': keypoints_3d.shape[0],
        'width': width,
        'height': height,
        'backend': 'rtmpose',
    }
