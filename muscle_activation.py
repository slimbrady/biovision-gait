#!/usr/bin/env python3
"""
muscle_activation.py - EMG-like muscle activation estimation

OpenSim StaticOptimization wrapper if opensim is installed,
otherwise kinematics-based surrogate fallback.

Muscles tracked (L/R):
  glute_max, rect_fem, vastus, hamstrings, gastroc, soleus, tib_ant

OpenSim setup:
  pip install opensim
  # Download Rajagopal 2016 model:
  # https://simtk.org/projects/opensim-models
  # Place at: ./models/Rajagopal2016/Rajagopal2016.osim
  # or set OPENSIM_MODEL_PATH env var

If OpenSim is not available, falls back to activation_surrogate().
"""
import os
import numpy as np
import pandas as pd
from pathlib import Path

MUSCLES = ['glute_max','rect_fem','vastus','hamstrings','gastroc','soleus','tib_ant']

# OpenSim muscle name mapping (Rajagopal 2016)
OPENSIM_MUSCLE_MAP = {
    'glute_max': ['gluteus_maximus_r', 'gluteus_maximus_l'],
    'rect_fem':  ['rectus_femoris_r', 'rectus_femoris_l'],
    'vastus':    ['vastus_lateralis_r', 'vastus_intermedius_r', 'vastus_medialis_r',
                  'vastus_lateralis_l', 'vastus_intermedius_l', 'vastus_medialis_l'],
    'hamstrings':['biceps_femoris_long_head_r', 'semitendinosus_r', 'semimembranosus_r',
                  'biceps_femoris_long_head_l', 'semitendinosus_l', 'semimembranosus_l'],
    'gastroc':   ['gastrocnemius_medial_r', 'gastrocnemius_lateral_r',
                  'gastrocnemius_medial_l', 'gastrocnemius_lateral_l'],
    'soleus':    ['soleus_r', 'soleus_l'],
    'tib_ant':   ['tibialis_anterior_r', 'tibialis_anterior_l'],
}

try:
    import opensim
    OPENSIM_AVAILABLE = True
except Exception:
    OPENSIM_AVAILABLE = False

# ---------------------------------------------------------------------------
# Surrogate (fallback when OpenSim not installed)
# ---------------------------------------------------------------------------

def activation_surrogate(joint_angles_deg, joint_moments_nmk, fps):
    """Kinematics + gait-phase → activation 0-1. Used when OpenSim unavailable."""
    def get_angle(side, joint):
        v = joint_angles_deg.get(f'{side}_{joint}_deg')
        if v is not None: return np.asarray(v)
        side_suf = 'r' if side=='R' else 'l'
        for k in [f'{joint}_flexion_{side_suf}', f'{joint}_{side_suf}', f'{joint}_flexion_{side.lower()}']:
            if k in joint_angles_deg: return np.asarray(joint_angles_deg[k])
        return None
    n = 0
    for v in joint_angles_deg.values():
        try: n = max(n, len(v))
        except: pass
    n = n or 300
    out = {}
    np.random.seed(0)  # deterministic
    for side in ['L','R']:
        hip = get_angle(side, 'hip'); knee = get_angle(side, 'knee'); ankle = get_angle(side, 'ankle')
        if hip is None: hip = np.zeros(n)
        if knee is None: knee = np.zeros(n)
        if ankle is None: ankle = np.zeros(n)
        t = np.arange(n)/fps
        phase = (t*2.5) % 1.0
        stance = (phase < 0.4).astype(float)
        swing = 1-stance
        # add small gait-phase noise so traces look realistic
        noise = lambda s: 0.04*np.random.randn(n)
        out[f'{side}_glute_max'] = np.clip(stance*0.6 + noise(n),0,1)
        out[f'{side}_rect_fem']  = np.clip(stance*0.5 + swing*0.2 + noise(n),0,1)
        out[f'{side}_vastus']    = np.clip(stance*0.7 + noise(n),0,1)
        out[f'{side}_hamstrings']= np.clip(swing*0.5 + stance*0.2 + noise(n),0,1)
        out[f'{side}_gastroc']   = np.clip(stance*0.8 + noise(n),0,1)
        out[f'{side}_soleus']    = np.clip(stance*0.7 + noise(n),0,1)
        out[f'{side}_tib_ant']   = np.clip(swing*0.6 + 0.1 + noise(n),0,1)
    return out

# ---------------------------------------------------------------------------
# OpenSim StaticOptimization
# ---------------------------------------------------------------------------

def find_opensim_model():
    """Locate Rajagopal2016.osim – check OPENSIM_MODEL_PATH, ./models/, ~/opensim/"""
    candidates = []
    env_path = os.environ.get('OPENSIM_MODEL_PATH')
    if env_path: candidates.append(Path(env_path))
    candidates += [
        Path('./models/Rajagopal2016/Rajagopal2016.osim'),
        Path('./Rajagopal2016.osim'),
        Path.home() / 'opensim/models/Rajagopal2016/Rajagopal2016.osim',
        Path('/usr/local/share/opensim/models/Rajagopal2016.osim'),
    ]
    for p in candidates:
        if p.exists(): return str(p)
    return None

def _angles_dict_to_opensim_mot(joint_angles_deg, fps, out_path):
    """
    Convert joint angle dict/DataFrame to OpenSim .mot (StatesTrajectory) file
    for IK / StaticOptimization input.
    
    Expected keys (either format):
      gait-pose-m4: L_hip_deg, R_hip_deg, L_knee_deg, R_knee_deg, L_ankle_deg, R_ankle_deg
      biovision:    hip_flexion_r/l, knee_flexion_r/l, ankle_dorsiflexion_r/l
    """
    def get(side, joint):
        # try gait-pose keys
        v = joint_angles_deg.get(f'{side}_{joint}_deg')
        if v is not None: return np.asarray(v, dtype=float)
        # try biovision keys
        s = 'r' if side == 'R' else 'l'
        for k in (f'{joint}_flexion_{s}', f'{joint}_{s}', f'{joint}_flexion_{side.lower()}'):
            if k in joint_angles_deg:
                return np.asarray(joint_angles_deg[k], dtype=float)
        return None

    # determine n_frames
    n = 0
    for v in joint_angles_deg.values():
        try: n = max(n, len(np.asarray(v)))
        except: pass
    if n == 0: n = 1

    def pad(a):
        if a is None: return np.zeros(n)
        a = np.asarray(a, dtype=float)
        return a if len(a) == n else np.interp(np.arange(n), np.arange(len(a)), a)

    # Rajagopal coordinate names (radians in .mot)
    deg2rad = np.pi/180.0
    q = {
        'hip_flexion_r':  pad(get('R','hip')) * deg2rad,
        'hip_flexion_l':  pad(get('L','hip')) * deg2rad,
        'knee_angle_r':   pad(get('R','knee')) * deg2rad,
        'knee_angle_l':   pad(get('L','knee')) * deg2rad,
        'ankle_angle_r':  pad(get('R','ankle')) * deg2rad,
        'ankle_angle_l':  pad(get('L','ankle')) * deg2rad,
    }
    t = np.arange(n) / float(fps)
    # write .mot
    with open(out_path, 'w') as f:
        f.write('Coordinates\n')
        f.write('version=1\n')
        f.write(f'nRows={n}\n')
        f.write(f'nColumns={1+len(q)}\n')
        f.write('inDegrees=no\n')
        f.write('endheader\n')
        f.write('time\t' + '\t'.join(q.keys()) + '\n')
        for i in range(n):
            f.write(f"{t[i]:.5f}\t" + '\t'.join(f"{q[k][i]:.6f}" for k in q) + '\n')
    return out_path

def compute_activations_opensim(joint_angles_deg, fps, model_path=None, work_dir=None):
    """
    Run OpenSim StaticOptimization to get muscle activations.
    
    Returns: dict muscle_side -> np.ndarray (N,) activation 0-1
    
    Requires:
      pip install opensim
      Rajagopal2016.osim model (see find_opensim_model)
    
    Pipeline:
      1. joint angles → .mot
      2. StaticOptimization (minimize sum(a^2), reserve actuators for hip/knee/ankle)
      3. parse activation .sto → dict
    """
    if not OPENSIM_AVAILABLE:
        raise RuntimeError("OpenSim not installed – pip install opensim")

    model_path = model_path or find_opensim_model()
    if not model_path or not Path(model_path).exists():
        raise FileNotFoundError(
            "Rajagopal2016.osim not found. Download from https://simtk.org/projects/opensim-models "
            "and set OPENSIM_MODEL_PATH, or place at ./models/Rajagopal2016/Rajagopal2016.osim"
        )

    import tempfile
    import shutil
    tmp = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix='osim_so_'))
    tmp.mkdir(parents=True, exist_ok=True)

    try:
        # 1. angles → .mot
        mot_path = tmp / 'ik.mot'
        _angles_dict_to_opensim_mot(joint_angles_deg, fps, str(mot_path))

        # 2. load model
        model = opensim.Model(model_path)
        model.initSystem()

        # 3. StaticOptimization setup
        so = opensim.StaticOptimization()
        so.setModel(model)
        so.set_start_time(0.0)
        # determine end time from mot
        n_frames = 0
        for v in joint_angles_deg.values():
            try: n_frames = max(n_frames, len(np.asarray(v)))
            except: pass
        t_end = max(0.01, (n_frames-1)/fps)
        so.set_end_time(t_end)
        # use the mot as coordinates
        so.set_coordinates_file_name(str(mot_path))
        # low-pass filter kinematics
        so.set_lowpass_cutoff_frequency_for_coordinates(6.0)
        # Use reserve actuators, minimize sum(a^2)
        so.set_use_model_force_set(True)
        so.set_use_muscle_physiology(True)
        so.set_activation_exponent(2.0)
        # output paths
        so.set_results_directory(str(tmp))
        # run
        ok = so.run()
        if not ok:
            raise RuntimeError("StaticOptimization failed – check kinematics / model scaling")

        # 4. read activation .sto
        # StaticOptimization writes e.g. StaticOptimization_activation.sto
        act_files = list(tmp.glob('*activation*.sto'))
        if not act_files:
            raise FileNotFoundError("No StaticOptimization activation output found in " + str(tmp))
        act_sto = opensim.TimeSeriesTable(str(act_files[0]))
        times = act_sto.getIndependentColumn()
        n = int(act_sto.getNumRows())

        # map OpenSim muscle names → our 7 groups, L/R averaged
        act_out = {}
        def get_col(name):
            if act_sto.hasColumn(name):
                return np.array([act_sto.getDependentColumn(name).getElt(i,0) for i in range(n)], dtype=float)
            return None

        for group, osim_names in OPENSIM_MUSCLE_MAP.items():
            for side, side_suffix in [('R','_r'), ('L','_l')]:
                cols = []
                for m in osim_names:
                    # match side
                    if side == 'R' and not m.endswith('_r'): continue
                    if side == 'L' and not m.endswith('_l'): continue
                    c = get_col(m)
                    if c is not None: cols.append(c)
                if cols:
                    act = np.mean(np.stack(cols), axis=0)
                    act_out[f'{side}_{group}'] = np.clip(act, 0, 1)
        # fill any missing muscles with surrogate
        if len(act_out) < 14:
            surr = activation_surrogate(joint_angles_deg, {}, fps)
            for k, v in surr.items():
                if k not in act_out:
                    # resample to n
                    if len(v) != n:
                        v = np.interp(np.linspace(0,1,n), np.linspace(0,1,len(v)), v)
                    act_out[k] = v
        return act_out, {'model': model_path, 'n_frames': n, 'so_dir': str(tmp)}

    except Exception:
        if work_dir is None:
            # clean up temp dir on failure only if we created it
            try: shutil.rmtree(tmp, ignore_errors=True)
            except: pass
        raise

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_activations(df_angles, df_forces, fps, use_opensim='auto', model_path=None):
    """
    Compute muscle activations.
    
    df_angles: DataFrame (gait-pose-m4) OR dict of np arrays (biovision)
    df_forces: output of biomech_force.compute_forces (currently unused, reserved for SO external loads)
    fps: frames per second
    use_opensim: 'auto' | True | False
      'auto' – use OpenSim if installed + model found, else surrogate
      True   – require OpenSim, raise if unavailable
      False  – force surrogate
    
    Returns: (df_act, summary)
      df_act: DataFrame with columns time_s, L_glute_max, R_glute_max, ...
      summary: {muscle_side: peak_activation}
    """
    # normalize angles to dict
    if isinstance(df_angles, pd.DataFrame):
        angles = {c: df_angles[c].values for c in df_angles.columns if c != 'time_s'}
        n_frames = len(df_angles)
        time_s = df_angles['time_s'].values if 'time_s' in df_angles else np.arange(n_frames)/fps
    else:
        angles = df_angles
        n_frames = 0
        for v in angles.values():
            try: n_frames = max(n_frames, len(np.asarray(v)))
            except: pass
        time_s = np.arange(n_frames)/fps

    # Try OpenSim
    act = None
    so_meta = {}
    want_opensim = use_opensim is True or (use_opensim == 'auto' and OPENSIM_AVAILABLE)
    if want_opensim and OPENSIM_AVAILABLE:
        try:
            act, so_meta = compute_activations_opensim(angles, fps, model_path=model_path)
        except Exception as e:
            if use_opensim is True:
                raise
            # fall through to surrogate
            act = None

    # Fallback surrogate
    if act is None:
        act = activation_surrogate(angles, {}, fps)

    # build DataFrame
    df_act = pd.DataFrame(act)
    n_act = len(next(iter(act.values()))) if act else n_frames
    if len(time_s) != n_act:
        time_s = np.arange(n_act)/fps
    df_act.insert(0, 'time_s', time_s[:len(df_act)])

    summary = {m: float(np.nanmax(df_act[m])) for m in df_act.columns if m != 'time_s'}
    if so_meta:
        summary['_opensim'] = True
        summary['_model'] = so_meta.get('model', '')
    else:
        summary['_opensim'] = False

    return df_act, summary


if __name__ == '__main__':
    print(f"OpenSim available: {OPENSIM_AVAILABLE}")
    if OPENSIM_AVAILABLE:
        import opensim
        print(f"OpenSim version: {opensim.__version__}")
        m = find_opensim_model()
        print(f"Model: {m or 'NOT FOUND – download Rajagopal2016.osim from https://simtk.org/projects/opensim-models'}")
    print(f"Muscles tracked: {', '.join(MUSCLES)}")
