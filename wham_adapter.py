# wham_adapter.py
# BioVision gAIt – WHAM (World-grounded Human with Accurate Motion) Integration
#
# WHAM = "World-grounded Human with Accurate Motion"
# Paper: https://arxiv.org/abs/2312.07531
# Repo:  https://github.com/yufu-wang/wham
#
# ============ WHAM vs MeTRAbs Comparison ============
#
# | Feature              | MeTRAbs (default)          | WHAM                          |
# |----------------------|----------------------------|---------------------------------|
# | Output               | 3D joints (17 H36M)        | SMPL mesh (6890 vertices) +   |
# |                      | metric-scale, mm           | 24 body joints + global traj  |
# | Temporal smoothing   | Per-frame (post-filter)    | Built-in (temporal model)     |
# | Global trajectory    | Camera-relative            | World-grounded                |
# | Foot-ground contact  | Estimated from ankle Y     | Predicted explicitly          |
# | Speed (M4)           | ~15-30 FPS CPU             | ~2-5 FPS CPU, ~15+ FPS GPU    |
# | GPU required?        | No (CPU / MPS OK)          | Recommended (CUDA)            |
# | Model files          | Auto-download              | Requires SMPL model download  |
# | Memory               | ~1-2 GB                    | ~4-8 GB                       |
# | Clinical gait metrics| Excellent (joint angles)   | Excellent (+ mesh contact)    |
# | Best for             | M4 / clinical gait ROM     | Research / global trajectory  |
#
# Recommendation:
#   • Use MeTRAbs (default) for fast clinical gait analysis on Apple Silicon.
#   • Use WHAM when you need: world-grounded global trajectory,
#     SMPL body mesh, foot-ground contact prediction, or temporal smoothness
#     for longer sequences – and you have a CUDA GPU available.
#
# ========================================================

from pathlib import Path
from typing import Dict, Optional, Tuple
import warnings

import numpy as np

WHAM_AVAILABLE = False
try:
    import torch
    # WHAM is not pip-installable from PyPI – must install from git
    # pip install git+https://github.com/yufu-wang/wham.git
    try:
        from wham import WHAM  # type: ignore
        WHAM_AVAILABLE = True
    except ImportError:
        pass
except ImportError:
    pass


def install_wham_instructions() -> str:
    """
    Print installation instructions for WHAM.

    WHAM is GPU-recommended and requires SMPL model files.
    On Apple Silicon M4, MeTRAbs is the recommended default backend.
    """
    return """
=== WHAM Installation (Experimental / GPU Recommended) ===

1. Install WHAM from source:
   pip install git+https://github.com/yufu-wang/wham.git

2. Install PyTorch with appropriate backend:
   # Apple Silicon (MPS):
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   # CUDA (Linux/NVIDIA):
   pip install torch torchvision

3. Download SMPL model files:
   - Register at http://smpl.is.tue.mpg.de/
   - Download SMPL v1.0 (neutral / male / female)
   - Place in: ./data/smpl/

   Required files:
     data/smpl/SMPL_NEUTRAL.pkl
     data/smpl/SMPL_MALE.pkl
     data/smpl/SMPL_FEMALE.pkl

4. Download WHAM pretrained weights:
   Follow instructions at https://github.com/yufu-wang/wham

5. Run inference:
   from wham_adapter import run_wham_inference
   result = run_wham_inference("video.mp4")

NOTE: WHAM is ~5-10x slower than MeTRAbs on CPU.
      GPU (CUDA) is strongly recommended for WHAM.
      On MacBook M4, stick with MeTRAbs for clinical gait work.
"""


def load_wham_model(
    smpl_model_path: str = "./data/smpl/SMPL_NEUTRAL.pkl",
    checkpoint_path: Optional[str] = None,
    device: str = "auto",
):
    """
    Load WHAM model for SMPL-based 3D human motion capture.

    Args:
        smpl_model_path: Path to SMPL model .pkl file
        checkpoint_path: Path to WHAM pretrained weights (.ckpt)
        device: "auto", "cuda", "mps", "cpu"
            auto → cuda → mps → cpu

    Returns:
        WHAM model object

    Raises:
        RuntimeError if WHAM is not installed.
    """
    if not WHAM_AVAILABLE:
        raise RuntimeError(
            "WHAM is not installed.\n\n" + install_wham_instructions()
        )

    # Auto-select device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    print(f"[WHAM] Loading model on device={device}")
    print(f"[WHAM] SMPL model: {smpl_model_path}")

    if not Path(smpl_model_path).exists():
        raise FileNotFoundError(
            f"SMPL model not found at {smpl_model_path}\n"
            f"Download from http://smpl.is.tue.mpg.de/\n\n"
            f"{install_wham_instructions()}"
        )

    # WHAM API (pseudo – actual API may differ, check upstream repo)
    # model = WHAM(smpl_model_path=smpl_model_path, device=device)
    # if checkpoint_path:
    #     model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    warnings.warn(
        "WHAM model loading is a stub – see wham_adapter.py for integration notes. "
        "Full WHAM integration requires installing from "
        "https://github.com/yufu-wang/wham and downloading SMPL model files."
    )

    return {"device": device, "smpl_path": smpl_model_path, "_stub": True}


def run_wham_inference(
    video_path: str,
    output_dir: Optional[str] = None,
    smpl_model_path: str = "./data/smpl/SMPL_NEUTRAL.pkl",
    fps: float = 30.0,
) -> Dict:
    """
    Run WHAM inference on a gait video.

    Outputs:
      - SMPL body pose parameters (72-D: 24 joints × 3 axis-angle)
      - SMPL shape parameters (10-D betas)
      - Global translation / root trajectory (world-grounded)
      - 3D joint positions (24 joints, SMPL skeleton)
      - Foot-ground contact labels
      - Body mesh vertices (6890, 3)

    Args:
        video_path: Input video
        output_dir: Output directory
        smpl_model_path: Path to SMPL .pkl
        fps: Output FPS

    Returns:
        {
            "smpl_pose": np.ndarray, shape (n_frames, 72),
            "smpl_betas": np.ndarray, shape (10,),
            "transl": np.ndarray, shape (n_frames, 3),  # global translation (m)
            "joints_3d": np.ndarray, shape (n_frames, 24, 3),  # meters
            "vertices": np.ndarray, shape (n_frames, 6890, 3),
            "contacts": np.ndarray, shape (n_frames, 4),  # L/R heel/toe contact
            "fps": float,
            "n_frames": int,
            "smpl_params_path": str,  # .npz file
        }

    NOTE: This is a documented STUB for v0.1.
    Full WHAM integration requires:
      1. pip install git+https://github.com/yufu-wang/wham.git
      2. Download SMPL model files from http://smpl.is.tue.mpg.de/
      3. GPU recommended (CUDA) – slow on CPU / M4

    See install_wham_instructions() for details.

    For production gait analysis on Apple Silicon M4, use MeTRAbs
    via pose_metrabs.py – it's faster, simpler, and gives excellent
    joint angle accuracy for clinical ROM measurements.
    """
    if not WHAM_AVAILABLE:
        raise RuntimeError(
            "WHAM is not installed – cannot run WHAM inference.\n\n"
            + install_wham_instructions() + "\n\n"
            "For M4 / Apple Silicon gait analysis, use MeTRAbs instead:\n"
            "  from pose_metrabs import run_metrabs_inference\n"
            "  result = run_metrabs_inference(video_path)"
        )

    video_path = Path(video_path)
    if output_dir is None:
        output_dir = video_path.parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- STUB: Real WHAM integration goes here ---
    # Pseudo-code for when WHAM is installed:
    #
    # model = load_wham_model(smpl_model_path)
    #
    # # WHAM processes full video sequences
    # # Input: RGB frames, shape (T, H, W, 3)
    # # Output: SMPL parameters per frame
    #
    # import cv2
    # cap = cv2.VideoCapture(str(video_path))
    # frames = []
    # while True:
    #     ret, frame = cap.read()
    #     if not ret: break
    #     frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    # cap.release()
    # frames = np.stack(frames, axis=0)
    #
    # with torch.no_grad():
    #     wham_output = model(frames)  # actual API TBD
    #
    # smpl_pose = wham_output["pose"]      # (T, 72)
    # smpl_betas = wham_output["betas"]    # (10,)
    # transl = wham_output["transl"]       # (T, 3)
    # joints_3d = wham_output["joints"]    # (T, 24, 3)
    # vertices = wham_output["vertices"]  # (T, 6890, 3)
    # contacts = wham_output["contacts"]  # (T, 4)
    #
    # # Save SMPL parameters
    # np.savez(output_dir / f"{video_path.stem}_wham.npz",
    #          pose=smpl_pose, betas=smpl_betas, transl=transl,
    #          joints=joints_3d, vertices=vertices, contacts=contacts)
    #
    # return { ... }

    warnings.warn(
        "run_wham_inference is a STUB in v0.1 – WHAM not installed. "
        "Use MeTRAbs via pose_metrabs.run_metrabs_inference() for production gait analysis."
    )

    # Return empty stub result so callers don't crash
    return {
        "smpl_pose": np.zeros((0, 72), dtype=np.float32),
        "smpl_betas": np.zeros(10, dtype=np.float32),
        "transl": np.zeros((0, 3), dtype=np.float32),
        "joints_3d": np.zeros((0, 24, 3), dtype=np.float32),
        "vertices": np.zeros((0, 6890, 3), dtype=np.float32),
        "contacts": np.zeros((0, 4), dtype=np.float32),
        "fps": fps,
        "n_frames": 0,
        "smpl_params_path": None,
        "_stub": True,
    }


def convert_wham_to_gait_metrics(
    wham_result: Dict,
    fps: float = 30.0,
    camera_view: str = "sagittal",
) -> Dict:
    """
    Convert WHAM SMPL output to BioVision gAIt joint angle format.

    WHAM outputs SMPL pose parameters (axis-angle, 24 joints).
    This function extracts anatomical joint angles compatible with
    metrics_gait.py so WHAM results can feed into the same
    gait analysis pipeline.

    SMPL joint hierarchy (24 joints):
      0: pelvis, 1: L_hip, 2: R_hip, 3: spine1, 4: L_knee, 5: R_knee,
      6: spine2, 7: L_ankle, 8: R_ankle, 9: spine3, 10: L_foot, 11: R_foot,
      12: neck, 13: L_collar, 14: R_collar, 15: head,
      16: L_shoulder, 17: R_shoulder, 18: L_elbow, 19: R_elbow,
      20: L_wrist, 21: R_wrist, 22: L_hand, 23: R_hand

    Mapping SMPL → H36M (for metrics_gait.py compatibility):
      SMPL pelvis (0) → H36M pelvis
      SMPL R_hip (2)  → H36M hip_r
      SMPL R_knee (5) → H36M knee_r
      SMPL R_ankle (8) → H36M ankle_r
      ... (etc.)

    Args:
        wham_result: Output dict from run_wham_inference()
        fps: Frames per second
        camera_view: "sagittal" | "frontal" | "rear"

    Returns:
        Dict compatible with metrics_gait.analyze_gait() input:
        {
            "keypoints_3d_mm": np.ndarray, shape (n_frames, 17, 3),
            "fps": float,
            "camera_view": str,
            "source": "wham",
        }

    Then feed into:
        from metrics_gait import analyze_gait
        analyze_gait(keypoints_3d_mm, fps, camera_view)

    NOTE: STUB in v0.1 – returns empty arrays.
    Full implementation requires WHAM to be installed and run first.
    """
    if wham_result.get("_stub"):
        warnings.warn("convert_wham_to_gait_metrics: WHAM result is a stub – returning empty metrics")
        return {
            "keypoints_3d_mm": np.zeros((0, 17, 3), dtype=np.float32),
            "fps": fps,
            "camera_view": camera_view,
            "source": "wham_stub",
        }

    joints_smpl = wham_result["joints_3d"]  # (T, 24, 3) in meters
    n_frames = joints_smpl.shape[0]

    # SMPL → H36M joint mapping
    # H36M indices: see pose_metrabs.H36M_JOINT_NAMES
    smpl_to_h36m = {
        0: 0,   # pelvis → pelvis
        2: 1,   # R_hip → hip_r
        5: 2,   # R_knee → knee_r
        8: 3,   # R_ankle → ankle_r
        1: 4,   # L_hip → hip_l
        4: 5,   # L_knee → knee_l
        7: 6,   # L_ankle → ankle_l
        3: 7,   # spine1 → spine
        12: 8,  # neck → neck
        15: 9,  # head → nose/head
        # ... fill remaining shoulder/elbow/wrist mappings
        16: 11, # L_shoulder
        18: 12, # L_elbow
        20: 13, # L_wrist
        17: 14, # R_shoulder
        19: 15, # R_elbow
        21: 16, # R_wrist
    }

    keypoints_h36m = np.zeros((n_frames, 17, 3), dtype=np.float32)
    for smpl_idx, h36m_idx in smpl_to_h36m.items():
        if smpl_idx < joints_smpl.shape[1]:
            # SMPL joints are in meters → convert to mm for metrics_gait.py
            keypoints_h36m[:, h36m_idx, :] = joints_smpl[:, smpl_idx, :] * 1000.0

    return {
        "keypoints_3d_mm": keypoints_h36m,
        "fps": fps,
        "camera_view": camera_view,
        "source": "wham",
        "smpl_pose": wham_result.get("smpl_pose"),
        "contacts": wham_result.get("contacts"),  # foot-ground contact – WHAM advantage!
    }


# ---------------------------------------------------------------------------
# CLI helper
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(install_wham_instructions())
    print("\n" + "="*70)
    print("WHAM adapter status:")
    print(f"  WHAM_AVAILABLE = {WHAM_AVAILABLE}")
    if not WHAM_AVAILABLE:
        print("\n  → WHAM is not installed. This is expected for v0.1.")
        print("  → Use MeTRAbs (pose_metrabs.py) for production gait analysis on M4.")
        print("\n  To use WHAM (GPU recommended):")
        print("    pip install git+https://github.com/yufu-wang/wham.git")
        print("    # + download SMPL model from http://smpl.is.tue.mpg.de/")
