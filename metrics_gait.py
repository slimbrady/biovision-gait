# metrics_gait.py
# BioVision gAIt – Gait Biomechanics Calculator
# Joint angles, ROM, spatiotemporal gait parameters
# Optimized for Apple Silicon / M4 – pure numpy / scipy
#
# All angles in DEGREES.
# All distances in METERS (MeTRAbs outputs mm → converted).
# All times in SECONDS.
#
# Clinical ROM reference ranges are included as comments at each function.
# These are typical adult ranges during gait – not maximum passive ROM.

import json
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.signal import find_peaks, savgol_filter
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from pose_metrabs import H36M_JOINT_NAMES, H36M_IDX, CameraView

# ---------------------------------------------------------------------------
# Joint index shortcuts (H36M format)
# ---------------------------------------------------------------------------

PELVIS = H36M_IDX["pelvis"]
HIP_R = H36M_IDX["hip_r"] ; HIP_L = H36M_IDX["hip_l"]
KNEE_R = H36M_IDX["knee_r"] ; KNEE_L = H36M_IDX["knee_l"]
ANKLE_R = H36M_IDX["ankle_r"] ; ANKLE_L = H36M_IDX["ankle_l"]
SPINE = H36M_IDX["spine"]
NECK = H36M_IDX["neck"]
NOSE = H36M_IDX["nose"]
SHOULDER_L = H36M_IDX["shoulder_l"] ; SHOULDER_R = H36M_IDX["shoulder_r"]
ELBOW_L = H36M_IDX["elbow_l"] ; ELBOW_R = H36M_IDX["elbow_r"]
WRIST_L = H36M_IDX["wrist_l"] ; WRIST_R = H36M_IDX["wrist_r"]

# ---------------------------------------------------------------------------
# Angle math helpers
# ---------------------------------------------------------------------------

def angle_3d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Joint angle at vertex b, formed by points a–b–c, in 3D space.

    Args:
        a, b, c: 3D points, shape (..., 3)

    Returns:
        Angle in degrees, range [0, 180].
        Returns np.nan if any vector is degenerate (< 1e-6).

    Anatomical meaning: the interior angle between segments ba and bc.
    For flexion/extension joints, 180° = full anatomical extension,
    <180° = flexion.
    """
    ba = a - b
    bc = c - b
    ba_norm = np.linalg.norm(ba, axis=-1)
    bc_norm = np.linalg.norm(bc, axis=-1)
    valid = (ba_norm > 1e-6) & (bc_norm > 1e-6)
    cos_angle = np.zeros_like(ba_norm)
    if np.any(valid):
        dot = np.sum(ba[valid] * bc[valid], axis=-1)
        cos_angle[valid] = np.clip(dot / (ba_norm[valid] * bc_norm[valid]), -1.0, 1.0)
    angle = np.degrees(np.arccos(cos_angle))
    if angle.ndim == 0:
        return float(angle) if valid else np.nan
    angle[~valid] = np.nan
    return angle


def angle_sagittal(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Joint angle projected into the sagittal plane (Y-Z).

    Sagittal plane = side view.
    X (mediolateral) component is ignored.
    Use for: hip/knee/ankle flexion-extension, trunk flexion.

    Returns degrees [0, 180].
    """
    a2 = a.copy(); a2[..., 0] = 0
    b2 = b.copy(); b2[..., 0] = 0
    c2 = c.copy(); c2[..., 0] = 0
    return angle_3d(a2, b2, c2)


def angle_frontal(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Joint angle projected into the frontal/coronal plane (X-Y).

    Frontal plane = front/back view.
    Z (anteroposterior) component is ignored.
    Use for: hip abduction/adduction, trunk lateral lean,
             calcaneal inversion/eversion.

    Returns degrees [0, 180].
    """
    a2 = a.copy(); a2[..., 2] = 0
    b2 = b.copy(); b2[..., 2] = 0
    c2 = c.copy(); c2[..., 2] = 0
    return angle_3d(a2, b2, c2)


def angle_transverse(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Joint angle projected into the transverse/horizontal plane (X-Z).

    Transverse plane = top-down view.
    Y (vertical) component is ignored.
    Use for: foot progression angle, hip rotation.

    Returns degrees [0, 180].
    """
    a2 = a.copy(); a2[..., 1] = 0
    b2 = b.copy(); b2[..., 1] = 0
    c2 = c.copy(); c2[..., 1] = 0
    return angle_3d(a2, b2, c2)


def vector_angle_against_vertical(v: np.ndarray, plane: str = "sagittal") -> np.ndarray:
    """
    Angle between a vector and the vertical (Y) axis, in a given plane.

    Useful for trunk lean, pelvic obliquity, etc.

    Args:
        v: vector(s), shape (..., 3)
        plane: "sagittal" (Y-Z), "frontal" (X-Y), "transverse" (X-Z)

    Returns:
        Signed angle in degrees. Positive = forward/right depending on plane.
    """
    v = np.asarray(v)
    if plane == "sagittal":
        # Angle in Y-Z plane from vertical
        vy, vz = v[..., 1], v[..., 2]
        angle = np.degrees(np.arctan2(vz, vy))
    elif plane == "frontal":
        vx, vy = v[..., 0], v[..., 1]
        angle = np.degrees(np.arctan2(vx, vy))
    else:  # transverse
        vx, vz = v[..., 0], v[..., 2]
        angle = np.degrees(np.arctan2(vx, vz))
    return angle


def smooth_signal(x: np.ndarray, window: int = 7, poly: int = 2) -> np.ndarray:
    """Savitzky-Golay smoothing. Falls back to moving average if scipy unavailable."""
    x = np.asarray(x, dtype=np.float64)
    if SCIPY_AVAILABLE and len(x) > window:
        if window % 2 == 0:
            window += 1
        window = min(window, len(x) if len(x) % 2 == 1 else len(x) - 1)
        if window >= 5:
            return savgol_filter(x, window, poly, mode="interp")
    # Moving average fallback
    if len(x) < 3:
        return x
    w = min(5, len(x) // 2 * 2 + 1)
    kernel = np.ones(w) / w
    return np.convolve(x, kernel, mode="same")

# ---------------------------------------------------------------------------
# Joint angle calculator – per frame
# Input: keypoints_3d shape (n_frames, 17, 3) in MILLIMETERS
# Output: dict of angle time series, shape (n_frames,)
# ---------------------------------------------------------------------------

def compute_all_joint_angles(
    keypoints_3d_mm: np.ndarray,
    camera_view: CameraView = "sagittal",
    smooth: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Compute all gait joint angles from 3D keypoints.

    Args:
        keypoints_3d_mm: (n_frames, 17, 3) – MeTRAbs output in mm
        camera_view: "sagittal" | "frontal" | "rear"
            Determines which angles are computable / reliable.
            Angles requiring an unavailable view return all-NaN arrays.
        smooth: Apply Savitzky-Golay smoothing to angle traces

    Returns:
        Dict mapping angle_name → np.ndarray (n_frames,)
        Missing/uncomputable angles = all NaN, with clear naming.

    All angles in DEGREES.
    """
    kpts = np.asarray(keypoints_3d_mm, dtype=np.float64)
    n_frames = kpts.shape[0]

    def nan_series(): return np.full(n_frames, np.nan, dtype=np.float64)

    angles = {}

    # ==================== SAGITTAL PLANE ANGLES ====================
    # These are best/most reliable from a sagittal (side) camera view.
    # If camera_view != sagittal, results may be less accurate but still computed
    # from 3D keypoints.

    # 1. HIP FLEXION / EXTENSION (L/R)
    # Anatomical: angle between trunk (spine–pelvis) and thigh (hip–knee)
    # Keypoints: spine – hip – knee
    # Normal gait ROM: ~40° flexion to ~10° extension (total ~50°)
    #   Running: ~50-60° flexion, ~20° extension
    # Camera view REQUIRED: sagittal (primary)
    #
    for side, hip, knee in [("r", HIP_R, KNEE_R), ("l", HIP_L, KNEE_L)]:
        spine_pts = kpts[:, SPINE, :]
        hip_pts = kpts[:, hip, :]
        knee_pts = kpts[:, knee, :]
        # Hip flexion = angle between trunk vector and thigh vector
        # Use pelvis as trunk reference
        pelvis_pts = kpts[:, PELVIS, :]
        # Trunk vector: pelvis → spine (proximal)
        # Thigh vector: hip → knee (distal)
        trunk_vec = spine_pts - pelvis_pts
        thigh_vec = knee_pts - hip_pts
        # Angle between trunk and thigh in sagittal plane
        # Simplified: use 3-point angle spine-hip-knee
        hip_angle = np.array([
            angle_sagittal(spine_pts[i], hip_pts[i], knee_pts[i])
            for i in range(n_frames)
        ])
        # Convert to flexion/extension: 180° = neutral/extension, <180° = flexion
        hip_flexion = 180.0 - hip_angle
        angles[f"hip_flexion_{side}"] = smooth_signal(hip_flexion) if smooth else hip_flexion

    # 2. KNEE FLEXION / EXTENSION (L/R)
    # Anatomical: angle between thigh (hip–knee) and shank (knee–ankle)
    # Keypoints: hip – knee – ankle
    # Normal gait ROM: ~0° extension (stance) to ~60-70° flexion (swing)
    #   Running: ~20° at foot strike, ~90-120° peak swing flexion
    # Camera view REQUIRED: sagittal
    #
    for side, hip, knee, ankle in [
        ("r", HIP_R, KNEE_R, ANKLE_R),
        ("l", HIP_L, KNEE_L, ANKLE_L),
    ]:
        hip_pts = kpts[:, hip, :]
        knee_pts = kpts[:, knee, :]
        ankle_pts = kpts[:, ankle, :]
        knee_angle = np.array([
            angle_sagittal(hip_pts[i], knee_pts[i], ankle_pts[i])
            for i in range(n_frames)
        ])
        knee_flexion = 180.0 - knee_angle
        angles[f"knee_flexion_{side}"] = smooth_signal(knee_flexion) if smooth else knee_flexion

    # 3. ANKLE DORSIFLEXION / PLANTARFLEXION (L/R)
    # Anatomical: angle between shank (knee–ankle) and foot
    # Keypoints: knee – ankle – toe (missing in H36M, approximate)
    # Since MeTRAbs H36M has no toe keypoint, we approximate foot
    # vector from ankle position + forward direction.
    # Normal gait ROM: ~10° dorsiflexion to ~20° plantarflexion
    #   Running: ~25° DF to ~30° PF
    # Camera view REQUIRED: sagittal
    # ACCURACY CAVEAT: Without toe/foot keypoints, ankle angle is estimated.
    # Use RTMPose/WholeBody backend for better foot detail.
    #
    for side, knee, ankle in [("r", KNEE_R, ANKLE_R), ("l", KNEE_L, ANKLE_L)]:
        knee_pts = kpts[:, knee, :]
        ankle_pts = kpts[:, ankle, :]
        # Approximate foot vector: project ankle forward in Z
        # (assumes subject is walking/running along Z axis)
        foot_tip = ankle_pts.copy()
        foot_tip[:, 2] += 150.0  # ~15 cm foot length forward
        foot_tip[:, 1] -= 30.0   # ~3 cm below ankle
        ankle_angle = np.array([
            angle_sagittal(knee_pts[i], ankle_pts[i], foot_tip[i])
            for i in range(n_frames)
        ])
        # 90° = neutral, >90° = dorsiflexion, <90° = plantarflexion
        ankle_df = ankle_angle - 90.0
        angles[f"ankle_dorsiflexion_{side}"] = smooth_signal(ankle_df) if smooth else ankle_df

    # 4. TRUNK FLEXION / EXTENSION
    # Anatomical: trunk lean angle from vertical in sagittal plane
    # Keypoints: pelvis – spine – neck
    # Normal gait: ~5° forward lean (walking), ~10-15° (running)
    # Camera view REQUIRED: sagittal
    #
    pelvis_pts = kpts[:, PELVIS, :]
    spine_pts = kpts[:, SPINE, :]
    trunk_vec = spine_pts - pelvis_pts
    trunk_flexion = vector_angle_against_vertical(trunk_vec, plane="sagittal")
    angles["trunk_flexion"] = smooth_signal(trunk_flexion) if smooth else trunk_flexion

    # 5. NECK FLEXION / EXTENSION
    # Anatomical: head/neck angle from vertical
    # Keypoints: neck – nose
    # Normal gait: ~0-10° flexion
    # Camera view REQUIRED: sagittal
    #
    neck_pts = kpts[:, NECK, :]
    nose_pts = kpts[:, NOSE, :]
    neck_vec = nose_pts - neck_pts
    neck_flexion = vector_angle_against_vertical(neck_vec, plane="sagittal")
    angles["neck_flexion"] = smooth_signal(neck_flexion) if smooth else neck_flexion

    # 6. SHOULDER FLEXION / EXTENSION (L/R)
    # Anatomical: arm angle relative to trunk
    # Keypoints: elbow – shoulder – hip
    # Normal gait arm swing: ~20-45° total ROM
    #   Running: ~60-90° total ROM
    # Camera view REQUIRED: sagittal
    #
    for side, shoulder, elbow, hip in [
        ("r", SHOULDER_R, ELBOW_R, HIP_R),
        ("l", SHOULDER_L, ELBOW_L, HIP_L),
    ]:
        shoulder_pts = kpts[:, shoulder, :]
        elbow_pts = kpts[:, elbow, :]
        hip_pts = kpts[:, hip, :]
        shoulder_angle = np.array([
            angle_sagittal(hip_pts[i], shoulder_pts[i], elbow_pts[i])
            for i in range(n_frames)
        ])
        shoulder_flexion = 180.0 - shoulder_angle
        angles[f"shoulder_flexion_{side}"] = smooth_signal(shoulder_flexion) if smooth else shoulder_flexion

    # 7. ELBOW FLEXION / EXTENSION (L/R)
    # Anatomical: angle between upper arm and forearm
    # Keypoints: shoulder – elbow – wrist
    # Normal gait: ~70-120° (running: more flexed, ~80-100°)
    # Camera view REQUIRED: sagittal
    #
    for side, shoulder, elbow, wrist in [
        ("r", SHOULDER_R, ELBOW_R, WRIST_R),
        ("l", SHOULDER_L, ELBOW_L, WRIST_L),
    ]:
        shoulder_pts = kpts[:, shoulder, :]
        elbow_pts = kpts[:, elbow, :]
        wrist_pts = kpts[:, wrist, :]
        elbow_angle = np.array([
            angle_sagittal(shoulder_pts[i], elbow_pts[i], wrist_pts[i])
            for i in range(n_frames)
        ])
        elbow_flexion = 180.0 - elbow_angle
        angles[f"elbow_flexion_{side}"] = smooth_signal(elbow_flexion) if smooth else elbow_flexion

    # ==================== FRONTAL PLANE ANGLES ====================
    # These REQUIRE frontal or rear camera view for accuracy.
    # If camera_view == "sagittal", these return NaN with clear documentation.

    frontal_available = camera_view in ("frontal", "rear")

    # 8. HIP ABDUCTION / ADDUCTION (L/R)
    # Anatomical: lateral hip angle in frontal plane
    # Keypoints: knee – hip – pelvis_center
    # Normal gait: ~5-10° adduction/abduction total
    # Clinical flag: >10° adduction = genu valgum risk,
    #                excessive abduction = Trendelenburg pattern
    # Camera view REQUIRED: frontal or rear
    # Returns NaN if camera_view == "sagittal"
    #
    if frontal_available:
        for side, hip in [("r", HIP_R), ("l", HIP_L)]:
            # Hip abduction = lateral deviation of thigh from vertical
            hip_pts = kpts[:, hip, :]
            # Use opposite hip to define pelvic horizontal
            opp_hip = HIP_L if side == "r" else HIP_R
            opp_hip_pts = kpts[:, opp_hip, :]
            # Pelvic drop vector
            pelvic_vec = opp_hip_pts - hip_pts
            # Thigh vector
            knee_idx = KNEE_R if side == "r" else KNEE_L
            knee_pts = kpts[:, knee_idx, :]
            thigh_vec = knee_pts - hip_pts
            # Angle in frontal plane
            abd = np.array([
                angle_frontal(
                    hip_pts[i] + np.array([0, -100, 0]),  # vertical reference
                    hip_pts[i],
                    knee_pts[i]
                ) for i in range(n_frames)
            ])
            # Convert to signed abd/add: positive = abduction
            hip_x = hip_pts[:, 0]
            knee_x = knee_pts[:, 0]
            lateral_sign = np.sign(knee_x - hip_x)
            if side == "l":
                lateral_sign *= -1  # flip for left side consistency
            hip_abd_add = (90.0 - abd) * lateral_sign
            angles[f"hip_abduction_{side}"] = smooth_signal(hip_abd_add) if smooth else hip_abd_add
    else:
        angles["hip_abduction_r"] = nan_series()
        angles["hip_abduction_l"] = nan_series()

    # 9. CALCANEAL INVERSION / EVERSION (L/R)
    # Anatomical: rearfoot angle – heel tilt in frontal plane
    # Keypoints REQUIRED: heel, ankle, (big toe / small toe for reference)
    #   MeTRAbs H36M skeleton DOES NOT include foot keypoints.
    #   → This is an ESTIMATE from ankle mediolateral sway if using MeTRAbs alone.
    #   → For accurate rearfoot angle, use RTMPose/WholeBody backend
    #     which provides heel + toe keypoints.
    #
    # With WholeBody foot keypoints:
    #   Rearfoot angle = angle between shank (knee–ankle) and
    #                    calcaneal line (ankle–heel) in frontal plane
    #   Inversion = heel tilts medially (positive)
    #   Eversion  = heel tilts laterally (negative, pronation)
    #
    # Normal gait: ~2-5° inversion at foot strike,
    #              ~5-10° eversion during midstance (pronation),
    #              return to neutral/inversion at toe-off
    # Clinical flag: >10° eversion = excessive pronation
    #                >5° inversion at foot strike = supination pattern
    #
    # Camera view REQUIRED: rear (best) or frontal
    # MeTRAbs-only ESTIMATE: ankle mediolateral velocity as proxy
    #   – clearly marked as ESTIMATE in output
    #
    # Returns NaN if camera_view == "sagittal"
    #
    if frontal_available:
        for side, ankle_idx in [("r", ANKLE_R), ("l", ANKLE_L)]:
            ankle_pts = kpts[:, ankle_idx, :]
            # Estimate: mediolateral ankle position relative to hip
            # (crude proxy for rearfoot eversion)
            hip_idx = HIP_R if side == "r" else HIP_L
            hip_pts = kpts[:, hip_idx, :]
            ml_offset = ankle_pts[:, 0] - hip_pts[:, 0]  # mm
            # Convert to rough "eversion angle" estimate
            # This is NOT a true calcaneal angle – marked as estimate
            # Scale: ~10mm lateral offset ≈ 5° eversion (rough)
            calc_eversion_estimate = ml_offset / 2.0
            if side == "l":
                calc_eversion_estimate *= -1
            angles[f"calc_eversion_{side}_ESTIMATE"] = (
                smooth_signal(calc_eversion_estimate) if smooth else calc_eversion_estimate
            )
    else:
        angles["calc_eversion_r_ESTIMATE"] = nan_series()
        angles["calc_eversion_l_ESTIMATE"] = nan_series()

    # 10. TRUNK LATERAL LEAN / OBLIQUITY
    # Anatomical: trunk tilt in frontal plane
    # Keypoints: pelvis – spine
    # Normal gait: <5° lateral lean
    # Clinical flag: >10° = Trendelenburg / compensation pattern
    # Camera view REQUIRED: frontal or rear
    #
    if frontal_available:
        trunk_vec = spine_pts - pelvis_pts
        trunk_lean = vector_angle_against_vertical(trunk_vec, plane="frontal")
        angles["trunk_lean_frontal"] = smooth_signal(trunk_lean) if smooth else trunk_lean
    else:
        angles["trunk_lean_frontal"] = nan_series()

    # 11. PELVIC DROP / HIP HIKE
    # Anatomical: pelvic obliquity in frontal plane
    # Keypoints: hip_r – hip_l
    # Normal gait: ~5-10° pelvic drop during single-leg stance
    # Clinical flag: >15° drop = Trendelenburg sign (weak gluteus medius)
    # Camera view REQUIRED: frontal or rear
    #
    if frontal_available:
        hip_r_pts = kpts[:, HIP_R, :]
        hip_l_pts = kpts[:, HIP_L, :]
        pelvic_vec = hip_l_pts - hip_r_pts
        pelvic_obliquity = vector_angle_against_vertical(
            np.stack([pelvic_vec[:, 0], pelvic_vec[:, 1], np.zeros(n_frames)], axis=1),
            plane="frontal"
        )
        # Pelvic obliquity: positive = left hip higher
        angles["pelvic_obliquity"] = smooth_signal(pelvic_obliquity) if smooth else pelvic_obliquity
    else:
        angles["pelvic_obliquity"] = nan_series()

    # 12. FOOT PROGRESSION ANGLE
    # Anatomical: foot angle relative to direction of travel (transverse plane)
    # Keypoints: heel – toe
    # Normal: ~5-15° external rotation (toe-out)
    # Clinical: >20° out-toeing / in-toeing
    # Camera view: any (computed in transverse plane from 3D)
    # CAVEAT: MeTRAbs H36M has no foot keypoints → returns NaN
    #   Use RTMPose/WholeBody for foot progression angle
    angles["foot_progression_r"] = nan_series()
    angles["foot_progression_l"] = nan_series()

    return angles


# ---------------------------------------------------------------------------
# Gait event detection
# ---------------------------------------------------------------------------

def detect_gait_events(
    keypoints_3d_mm: np.ndarray,
    fps: float,
    camera_view: CameraView = "sagittal",
) -> Dict[str, np.ndarray]:
    """
    Detect foot strike and toe-off events from ankle kinematics.

    Method:
      - Foot strike: local minima in ankle vertical (Y) position + 
                     local maxima in downward vertical velocity
      - Toe-off: local maxima in ankle vertical velocity (push-off)

    Args:
        keypoints_3d_mm: (n_frames, 17, 3) in mm
        fps: frames per second
        camera_view: for documentation / view-dependent tuning

    Returns:
        {
            "foot_strike_r": np.ndarray of frame indices,
            "foot_strike_l": np.ndarray,
            "toe_off_r": np.ndarray,
            "toe_off_l": np.ndarray,
        }

    Clinical relevance:
        Foot strike / toe-off timing defines stance phase vs swing phase.
        Stance = foot on ground (~60% of gait cycle walking, ~40% running)
        Swing  = foot in air   (~40% walking, ~60% running)
    """
    kpts = keypoints_3d_mm
    n_frames = kpts.shape[0]

    events = {}

    for side, ankle_idx in [("r", ANKLE_R), ("l", ANKLE_L)]:
        ankle_y = kpts[:, ankle_idx, 1]  # vertical position, mm
        ankle_y = smooth_signal(ankle_y, window=7)

        # Vertical velocity
        dt = 1.0 / fps
        ankle_vy = np.gradient(ankle_y, dt)

        # Foot strike ≈ local minimum in ankle Y (lowest point)
        # Find peaks in -ankle_y
        if SCIPY_AVAILABLE and len(ankle_y) > 10:
            # Minimum distance between foot strikes ≈ 0.3s (running) to 1.0s (walking)
            min_distance = int(fps * 0.3)
            strikes, _ = find_peaks(-ankle_y, distance=min_distance, prominence=5)
            # Toe-off ≈ local maxima in upward velocity after strike
            toe_offs, _ = find_peaks(ankle_vy, distance=min_distance, prominence=10)
        else:
            # Fallback: simple thresholding
            strikes = np.array([], dtype=int)
            toe_offs = np.array([], dtype=int)

        events[f"foot_strike_{side}"] = strikes
        events[f"toe_off_{side}"] = toe_offs

    return events


# ---------------------------------------------------------------------------
# Spatiotemporal gait parameters
# ---------------------------------------------------------------------------

def compute_spatiotemporal_params(
    keypoints_3d_mm: np.ndarray,
    fps: float,
    events: Dict[str, np.ndarray],
    camera_view: CameraView = "sagittal",
    subject_height_m: float = 1.75,
) -> Dict[str, float]:
    """
    Compute spatiotemporal gait parameters.

    All distances converted from mm → meters.
    All times in seconds.

    Returns:
        speed_m_s, speed_km_h,
        cadence_steps_per_min,
        step_length_r_m, step_length_l_m,
        stride_length_m,
        stride_time_s,
        stance_time_r_s, stance_time_l_s,
        swing_time_r_s, swing_time_l_s,
        double_support_time_s,
        step_width_m  (frontal/rear view only, NaN if sagittal)

    Clinical normal ranges (walking, adult):
        Speed: 1.2 – 1.4 m/s
        Cadence: 100 – 120 steps/min
        Step length: ~0.6 – 0.8 m
        Stride length: ~1.2 – 1.6 m
        Stride time: ~1.0 – 1.2 s
        Stance: ~60% of gait cycle
        Swing: ~40% of gait cycle
        Double support: ~20% of gait cycle
        Step width: 0.08 – 0.15 m

    Running (typical recreational):
        Speed: 2.5 – 4.5 m/s
        Cadence: 160 – 190 steps/min
        Stride length: ~1.2 – 2.0 m
        Stance: ~30-40% of gait cycle
        Double support: ~0% (flight phase instead)
    """
    kpts = keypoints_3d_mm

    # --- Gait events ---
    fs_r = events.get("foot_strike_r", np.array([]))
    fs_l = events.get("foot_strike_l", np.array([]))
    to_r = events.get("toe_off_r", np.array([]))
    to_l = events.get("toe_off_l", np.array([]))

    def safe_mean(x, default=np.nan):
        x = np.asarray(x)
        return float(np.nanmean(x)) if len(x) > 0 and np.any(np.isfinite(x)) else default

    # --- Stride time ---
    def stride_times(fs):
        if len(fs) < 2:
            return np.array([])
        return np.diff(fs) / fps

    stride_time_r = safe_mean(stride_times(fs_r))
    stride_time_l = safe_mean(stride_times(fs_l))
    stride_time = np.nanmean([stride_time_r, stride_time_l])

    # --- Cadence (steps/min) ---
    # Cadence = 60 / step_time
    # step_time ≈ stride_time / 2
    if np.isfinite(stride_time) and stride_time > 0:
        cadence = 60.0 / (stride_time / 2.0)
    else:
        # Fallback: count foot strikes
        n_strikes = len(fs_r) + len(fs_l)
        duration_s = kpts.shape[0] / fps
        cadence = (n_strikes / duration_s * 60.0) if duration_s > 0 else np.nan

    # --- Step / stride length ---
    # From ankle AP displacement between consecutive foot strikes
    # MeTRAbs gives metric 3D – use Z (forward) displacement
    def step_lengths_mm(fs_events, ankle_idx):
        lengths = []
        for i in range(len(fs_events) - 1):
            f0, f1 = fs_events[i], fs_events[i + 1]
            if f0 >= kpts.shape[0] or f1 >= kpts.shape[0]:
                continue
            z0 = kpts[f0, ankle_idx, 2]
            z1 = kpts[f1, ankle_idx, 2]
            lengths.append(abs(z1 - z0))
        return np.array(lengths)

    step_len_r_mm = step_lengths_mm(fs_r, ANKLE_R)
    step_len_l_mm = step_lengths_mm(fs_l, ANKLE_L)

    step_length_r_m = safe_mean(step_len_r_mm) / 1000.0
    step_length_l_m = safe_mean(step_len_l_mm) / 1000.0
    stride_length_m = safe_mean([step_length_r_m, step_length_l_m]) * 2.0

    # --- Speed ---
    if np.isfinite(stride_time) and stride_time > 0 and np.isfinite(stride_length_m):
        speed_m_s = stride_length_m / stride_time
    else:
        # Fallback: pelvis displacement over time
        pelvis_z = kpts[:, PELVIS, 2]
        valid = np.isfinite(pelvis_z)
        if np.sum(valid) > 10:
            dz = pelvis_z[valid][-1] - pelvis_z[valid][0]
            dt = np.sum(valid) / fps
            speed_m_s = abs(dz / 1000.0) / dt if dt > 0 else np.nan
        else:
            speed_m_s = np.nan

    speed_km_h = speed_m_s * 3.6 if np.isfinite(speed_m_s) else np.nan

    # --- Stance / swing time ---
    # Stance = foot_strike → next toe_off (same side)
    # Swing  = toe_off → next foot_strike
    def stance_swing_times(fs, to):
        stances, swings = [], []
        for f in fs:
            # find next toe-off after foot strike
            to_after = to[to > f]
            if len(to_after) == 0:
                continue
            t_off = to_after[0]
            stances.append((t_off - f) / fps)
            # find next foot strike after toe-off
            fs_after = fs[fs > t_off]
            if len(fs_after) > 0:
                swings.append((fs_after[0] - t_off) / fps)
        return np.array(stances), np.array(swings)

    stance_r, swing_r = stance_swing_times(fs_r, to_r)
    stance_l, swing_l = stance_swing_times(fs_l, to_l)

    stance_time_r_s = safe_mean(stance_r)
    stance_time_l_s = safe_mean(stance_l)
    swing_time_r_s = safe_mean(swing_r)
    swing_time_l_s = safe_mean(swing_l)

    # --- Double support time ---
    # Time when both feet are on ground
    # Approximate: overlap of R_stance and L_stance intervals
    double_support_s = np.nan  # TODO: compute from event timelines

    # --- Step width / stance width (frontal/rear view only) ---
    # Mediolateral distance between ankles at foot strike
    # Camera view REQUIRED: frontal or rear
    # Returns NaN if camera_view == "sagittal"
    #
    if camera_view in ("frontal", "rear"):
        widths = []
        # Pair R and L foot strikes that are close in time
        for fr in fs_r:
            # find nearest L foot strike
            if len(fs_l) == 0:
                continue
            fl_nearest = fs_l[np.argmin(np.abs(fs_l - fr))]
            if abs(fl_nearest - fr) > fps * 0.5:  # >0.5s apart, skip
                continue
            # Use the later of the two frames
            f = max(fr, fl_nearest)
            if f >= kpts.shape[0]:
                continue
            ankle_r_x = kpts[f, ANKLE_R, 0]
            ankle_l_x = kpts[f, ANKLE_L, 0]
            width_mm = abs(ankle_r_x - ankle_l_x)
            widths.append(width_mm / 1000.0)
        step_width_m = safe_mean(widths)
    else:
        step_width_m = np.nan

    return {
        "speed_m_s": float(speed_m_s) if np.isfinite(speed_m_s) else np.nan,
        "speed_km_h": float(speed_km_h) if np.isfinite(speed_km_h) else np.nan,
        "cadence_steps_per_min": float(cadence) if np.isfinite(cadence) else np.nan,
        "step_length_r_m": float(step_length_r_m),
        "step_length_l_m": float(step_length_l_m),
        "stride_length_m": float(stride_length_m),
        "stride_time_s": float(stride_time),
        "stance_time_r_s": float(stance_time_r_s),
        "stance_time_l_s": float(stance_time_l_s),
        "swing_time_r_s": float(swing_time_r_s),
        "swing_time_l_s": float(swing_time_l_s),
        "double_support_time_s": float(double_support_s),
        "step_width_m": float(step_width_m),
    }


# ---------------------------------------------------------------------------
# ROM summary per joint
# ---------------------------------------------------------------------------

def compute_rom_summary(angles: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
    """
    Compute ROM (range of motion) statistics per joint.

    Returns:
        { joint_name: {"rom": ..., "min": ..., "max": ..., "mean": ..., "std": ...} }
    """
    summary = {}
    for name, series in angles.items():
        series = np.asarray(series, dtype=np.float64)
        valid = np.isfinite(series)
        if not np.any(valid):
            summary[name] = {
                "rom": np.nan, "min": np.nan, "max": np.nan,
                "mean": np.nan, "std": np.nan,
            }
            continue
        v = series[valid]
        summary[name] = {
            "rom": float(np.ptp(v)),
            "min": float(np.min(v)),
            "max": float(np.max(v)),
            "mean": float(np.mean(v)),
            "std": float(np.std(v)),
        }
    return summary


# ---------------------------------------------------------------------------
# Full pipeline: keypoints → angles → gait params → CSV/JSON
# ---------------------------------------------------------------------------

def analyze_gait(
    keypoints_3d_mm: np.ndarray,
    fps: float,
    camera_view: CameraView = "sagittal",
    subject_height_m: float = 1.75,
    output_prefix: str = "metrics",
) -> Dict:
    """
    Full gait analysis pipeline.

    Returns dict with angles, events, spatiotemporal params, ROM summary.
    Also writes:
      - {output_prefix}_{camera_view}.csv  – per-frame angles
      - {output_prefix}_summary.json       – aggregate metrics
    """
    # 1. Joint angles
    angles = compute_all_joint_angles(keypoints_3d_mm, camera_view=camera_view)

    # 2. Gait events
    events = detect_gait_events(keypoints_3d_mm, fps, camera_view=camera_view)

    # 3. Spatiotemporal
    st_params = compute_spatiotemporal_params(
        keypoints_3d_mm, fps, events, camera_view, subject_height_m
    )

    # 4. ROM summary
    rom_summary = compute_rom_summary(angles)

    # --- Export per-frame CSV ---
    df = pd.DataFrame(angles)
    df.insert(0, "frame", np.arange(len(df)))
    df.insert(1, "time_s", df["frame"] / fps)
    csv_path = f"{output_prefix}_{camera_view}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Angles CSV saved: {csv_path}")

    # --- Export summary JSON ---
    def json_safe(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.floating):
            return float(x) if np.isfinite(x) else None
        if isinstance(x, np.integer):
            return int(x)
        return x

    summary = {
        "camera_view": camera_view,
        "fps": fps,
        "n_frames": int(keypoints_3d_mm.shape[0]),
        "spatiotemporal": {k: json_safe(v) for k, v in st_params.items()},
        "rom_summary": rom_summary,
        "gait_events": {k: v.tolist() for k, v in events.items()},
    }
    json_path = f"{output_prefix}_summary_{camera_view}.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary JSON saved: {json_path}")

    return {
        "angles": angles,
        "events": events,
        "spatiotemporal": st_params,
        "rom_summary": rom_summary,
        "csv_path": csv_path,
        "json_path": json_path,
    }


# ---------------------------------------------------------------------------
# Multi-view fusion
# ---------------------------------------------------------------------------

def fuse_multiview_metrics(
    results_by_view: Dict[CameraView, Dict],
) -> Dict:
    """
    Fuse gait metrics from multiple camera views.

    Strategy:
      - Sagittal-plane angles → prefer sagittal view
      - Frontal-plane angles → prefer frontal/rear view (average if both)
      - Spatiotemporal (speed, cadence, stride length) → average across views,
        weighted by number of detected gait cycles

    Args:
        results_by_view: { "sagittal": analyze_gait(...),
                           "frontal": analyze_gait(...), ... }

    Returns:
        Fused metrics dict (same structure as analyze_gait output)
    """
    if len(results_by_view) == 1:
        return next(iter(results_by_view.values()))

    # Collect all angle names
    all_angle_names = set()
    for r in results_by_view.values():
        all_angle_names.update(r["angles"].keys())

    fused_angles = {}
    angle_sources = {}

    # Frontal-plane angle name patterns
    frontal_patterns = ("abduction", "adduction", "lean_frontal",
                        "pelvic_obliquity", "calc_eversion", "foot_progression")

    for angle_name in all_angle_names:
        # Determine preferred view
        is_frontal = any(p in angle_name for p in frontal_patterns)

        if is_frontal:
            # Prefer frontal/rear
            preferred_order = ["frontal", "rear", "sagittal"]
        else:
            # Sagittal-plane → prefer sagittal
            preferred_order = ["sagittal", "frontal", "rear"]

        series = None
        source = None
        for view in preferred_order:
            if view in results_by_view:
                s = results_by_view[view]["angles"].get(angle_name)
                if s is not None and np.any(np.isfinite(s)):
                    series = s
                    source = view
                    break

        if series is None:
            # Fall back to NaN series of matching length
            n = len(next(iter(results_by_view.values()))["angles"].values().__iter__().__next__())
            series = np.full(n, np.nan)
            source = "none"

        fused_angles[angle_name] = series
        angle_sources[angle_name] = source

    # Spatiotemporal: average across views
    st_keys = results_by_view[next(iter(results_by_view))]["spatiotemporal"].keys()
    fused_st = {}
    for k in st_keys:
        vals = [
            r["spatiotemporal"].get(k, np.nan)
            for r in results_by_view.values()
        ]
        vals = [v for v in vals if v is not None and np.isfinite(v)]
        fused_st[k] = float(np.nanmean(vals)) if vals else np.nan

    fused_rom = compute_rom_summary(fused_angles)

    return {
        "angles": fused_angles,
        "angle_sources": angle_sources,
        "spatiotemporal": fused_st,
        "rom_summary": fused_rom,
        "views_fused": list(results_by_view.keys()),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="BioVision gAIt – Gait Metrics Calculator")
    parser.add_argument("keypoints_json", help="keypoints JSON from pose_metrabs.py")
    parser.add_argument("--view", choices=["sagittal", "frontal", "rear"],
                        default="sagittal")
    parser.add_argument("--height", type=float, default=1.75,
                        help="Subject height in meters")
    parser.add_argument("--output-prefix", default="metrics")
    args = parser.parse_args()

    with open(args.keypoints_json) as f:
        kp_data = json.load(f)

    keypoints_3d = np.array(kp_data["keypoints_3d_mm"], dtype=np.float32)
    fps = kp_data.get("fps", 30.0)

    analyze_gait(
        keypoints_3d, fps,
        camera_view=args.view,
        subject_height_m=args.height,
        output_prefix=args.output_prefix,
    )
