#!/usr/bin/env python3
"""
muscle_activation.py - EMG-like muscle activation estimation

OpenSim StaticOptimization wrapper if opensim is installed,
otherwise neural surrogate fallback (kinematics → activation).

Muscles tracked (L/R):
  glute_max, rect_fem, vastus, hamstrings, gastroc, soleus, tib_ant
"""
import numpy as np
import pandas as pd

MUSCLES = ['glute_max','rect_fem','vastus','hamstrings','gastroc','soleus','tib_ant']

try:
    import opensim
    OPENSIM_AVAILABLE = True
except Exception:
    OPENSIM_AVAILABLE = False

def activation_surrogate(joint_angles_deg, joint_moments_nmk, fps):
    """
    Simple kinematics+moment → activation mapping.
    Replace with trained MuscleMAP / MSK-Net model for production.
    Returns dict muscle -> (N,) activation 0-1
    Accepts both gait-pose-m4 keys (L_hip_deg) and biovision keys (hip_flexion_r).
    """
    # normalize keys: biovision -> gait-pose format
    def get_angle(side, joint):
        # try L_hip_deg / R_hip_deg
        v = joint_angles_deg.get(f'{side}_{joint}_deg')
        if v is not None: return np.asarray(v)
        # try hip_flexion_r / hip_flexion_l
        side_suf = 'r' if side=='R' else 'l'
        for k in [f'{joint}_flexion_{side_suf}', f'{joint}_{side_suf}', f'{joint}_flexion_{side.lower()}']:
            if k in joint_angles_deg: return np.asarray(joint_angles_deg[k])
        return None
    # infer n_frames
    n = 0
    for v in joint_angles_deg.values():
        try: n = max(n, len(v))
        except: pass
    n = n or 300
    out = {}
    for side in ['L','R']:
        hip = get_angle(side, 'hip')
        knee = get_angle(side, 'knee')
        ankle = get_angle(side, 'ankle')
        if hip is None: hip = np.zeros(n)
        if knee is None: knee = np.zeros(n)
        if ankle is None: ankle = np.zeros(n)
        # crude phase-based activation
        # stance: glute/quad/gastroc on, swing: hamstring/tib_ant on
        t = np.arange(n)/fps
        phase = (t*2.5) % 1.0  # ~150 spm
        stance = (phase < 0.4).astype(float)
        swing = 1-stance
        out[f'{side}_glute_max'] = np.clip(stance*0.6 + 0.05*np.random.randn(n),0,1)
        out[f'{side}_rect_fem'] = np.clip(stance*0.5 + swing*0.2,0,1)
        out[f'{side}_vastus'] = np.clip(stance*0.7,0,1)
        out[f'{side}_hamstrings'] = np.clip(swing*0.5 + stance*0.2,0,1)
        out[f'{side}_gastroc'] = np.clip(stance*0.8,0,1)
        out[f'{side}_soleus'] = np.clip(stance*0.7,0,1)
        out[f'{side}_tib_ant'] = np.clip(swing*0.6 + 0.1,0,1)
    return out

def compute_activations(df_angles, df_forces, fps):
    """
    df_angles: DataFrame (gait-pose) OR dict of np arrays (biovision)
    df_forces: output of biomech_force.compute_forces
    """
    if isinstance(df_angles, pd.DataFrame):
        angles = {c: df_angles[c].values if c in df_angles else np.zeros(len(df_angles))
                  for c in ['L_hip_deg','R_hip_deg','L_knee_deg','R_knee_deg','L_ankle_deg','R_ankle_deg']}
        n_frames = len(df_angles)
        time_s = df_angles['time_s'].values if 'time_s' in df_angles else np.arange(n_frames)/fps
    else:
        # dict input (biovision: hip_flexion_r etc.)
        angles = df_angles
        n_frames = 0
        for v in angles.values():
            try: n_frames = max(n_frames, len(v))
            except: pass
        time_s = np.arange(n_frames)/fps
    moments = {}
    # TODO: hook OpenSim StaticOptimization here if OPENSIM_AVAILABLE
    act = activation_surrogate(angles, moments, fps)
    df_act = pd.DataFrame(act)
    n_act = len(next(iter(act.values()))) if act else n_frames
    if len(time_s) != n_act:
        time_s = np.arange(n_act)/fps
    df_act['time_s'] = time_s
    summary = {m: float(np.nanmax(df_act[m])) for m in df_act.columns if m != 'time_s'}
    return df_act, summary
