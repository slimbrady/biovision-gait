#!/usr/bin/env python3
"""
sheets_logger.py - Google Sheets telemetry for gait analysis

pip install gspread google-auth

Setup:
1. Create a Google Cloud Service Account, enable Sheets API
2. Share your Sheet with the service account email (Editor)
3. Put service_account.json in secrets, or set GOOGLE_SHEETS_CREDENTIALS env var
4. In Streamlit: st.secrets["gspread_service_account"]

Sheet: https://docs.google.com/spreadsheets/d/1o4aA07t5ODfsXtLl5M0j6SudLRbGovgxpDUzHKFFOk8/
Tab: Analysis
"""
import json, os, hashlib
from datetime import datetime, timezone

try:
    import gspread
    from google.oauth2.service_account import Credentials
    GSHEETS_AVAILABLE = True
except ImportError:
    GSHEETS_AVAILABLE = False

SPREADSHEET_ID = "1o4aA07t5ODfsXtLl5M0j6SudLRbGovgxpDUzHKFFOk8"
WORKSHEET_NAME = "Analysis"

# Column order – add these to your Sheet header row if missing
COLUMNS = [
"timestamp","engine","model_version","subject_id","video_hash",
"mass_kg","mass_lb","height_m","sex","age",
# ROM - MeTRAbs / BioVision
"rom_hip_l_metrabs","rom_hip_r_metrabs",
"rom_knee_l_metrabs","rom_knee_r_metrabs",
"rom_ankle_l_metrabs","rom_ankle_r_metrabs",
# ROM - RTMPose
"rom_hip_l_rtmpose","rom_hip_r_rtmpose",
"rom_knee_l_rtmpose","rom_knee_r_rtmpose",
"rom_ankle_l_rtmpose","rom_ankle_r_rtmpose",
# Legacy single-backend ROM (kept for back-compat)
"rom_hip_l","rom_hip_r","rom_knee_l","rom_knee_r","rom_ankle_l","rom_ankle_r",
"speed_ms","cadence_spm","step_length_m",
"peak_grf_l_bw","peak_grf_r_bw",
"peak_hip_moment","peak_knee_moment","peak_ankle_moment",
"act_glutemax","act_quad","act_hamstring","act_gastroc","act_soleus","act_tibant",
# Perf / accuracy – per backend
"inference_fps","processing_time_s",
"inference_fps_metrabs","processing_time_s_metrabs",
"inference_fps_rtmpose","processing_time_s_rtmpose",
"accuracy_metrabs","accuracy_rtmpose",
"f1_metrabs","f1_rtmpose",
"compare_run_id","notes"
]

def get_client(creds_dict=None):
    if not GSHEETS_AVAILABLE: raise RuntimeError("pip install gspread google-auth")
    if creds_dict is None:
        # try streamlit secrets
        try:
            import streamlit as st
            creds_dict = st.secrets["gspread_service_account"]
        except Exception:
            pass
    if creds_dict is None:
        # env var path or JSON string
        p = os.environ.get("GOOGLE_SHEETS_CREDENTIALS", "service_account.json")
        if os.path.exists(p):
            with open(p) as f: creds_dict = json.load(f)
        else:
            try: creds_dict = json.loads(p)
            except: raise RuntimeError("No Google Sheets credentials found")
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
    return gspread.authorize(creds)

def video_hash(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""): h.update(chunk)
    return h.hexdigest()[:12]

def _extract_rom(summary):
    rd = summary.get("rom_deg", {})
    return {
        "hip_l": rd.get("L_hip_rom"),
        "hip_r": rd.get("R_hip_rom"),
        "knee_l": rd.get("L_knee_rom"),
        "knee_r": rd.get("R_knee_rom"),
        "ankle_l": rd.get("L_ankle_rom"),
        "ankle_r": rd.get("R_ankle_rom"),
    }

def log_run(summary, forces_summary=None, muscle_summary=None,
            engine="metrabs", model_version="efficientnetv2_s",
            subject_meta=None, perf=None, video_path=None,
            accuracy=None, f1=None, compare_run_id=None, notes=""):
    """
    summary: from metrics.compute_metrics
    forces_summary: from biomech_force.compute_forces
    muscle_summary: from muscle_activation.compute_activations
    subject_meta: dict with mass_kg, height_m, sex, age, subject_id
    perf: dict with inference_fps, processing_time_s
    accuracy/f1: optional pose accuracy metrics (None until GT is available)
    """
    subject_meta = subject_meta or {}
    mass_kg = subject_meta.get("mass_kg")
    rom = _extract_rom(summary)
    backend_suffix = "_metrabs" if "metra" in engine.lower() else "_rtmpose" if "rtm" in engine.lower() else ""

    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "engine": engine,
        "model_version": model_version,
        "subject_id": subject_meta.get("subject_id",""),
        "video_hash": video_hash(video_path) if video_path and os.path.exists(video_path) else "",
        "mass_kg": mass_kg,
        "mass_lb": round(mass_kg*2.20462,1) if mass_kg else "",
        "height_m": subject_meta.get("height_m",""),
        "sex": subject_meta.get("sex",""),
        "age": subject_meta.get("age",""),
        # legacy ROM
        "rom_hip_l": rom["hip_l"],
        "rom_hip_r": rom["hip_r"],
        "rom_knee_l": rom["knee_l"],
        "rom_knee_r": rom["knee_r"],
        "rom_ankle_l": rom["ankle_l"],
        "rom_ankle_r": rom["ankle_r"],
        "speed_ms": summary.get("gait",{}).get("speed_ms"),
        "cadence_spm": summary.get("gait",{}).get("cadence_spm"),
        "step_length_m": summary.get("gait",{}).get("step_length_m"),
        "peak_grf_l_bw": forces_summary.get("peak_grf_L_bw") if forces_summary else None,
        "peak_grf_r_bw": forces_summary.get("peak_grf_R_bw") if forces_summary else None,
        "peak_hip_moment": forces_summary.get("peak_hip_moment_L") if forces_summary else None,
        "peak_knee_moment": forces_summary.get("peak_knee_moment_L") if forces_summary else None,
        "peak_ankle_moment": forces_summary.get("peak_ankle_moment_L") if forces_summary else None,
        "inference_fps": perf.get("inference_fps") if perf else None,
        "processing_time_s": perf.get("processing_time_s") if perf else None,
        "compare_run_id": compare_run_id or "",
        "notes": notes,
    }
    # backend-specific ROM + perf
    if backend_suffix:
        for joint in ["hip_l","hip_r","knee_l","knee_r","ankle_l","ankle_r"]:
            row[f"rom_{joint}{backend_suffix}"] = rom[joint.replace("_","_")]
        if perf:
            row[f"inference_fps{backend_suffix}"] = perf.get("inference_fps")
            row[f"processing_time_s{backend_suffix}"] = perf.get("processing_time_s")
        if accuracy is not None:
            row[f"accuracy{backend_suffix}"] = accuracy
        if f1 is not None:
            row[f"f1{backend_suffix}"] = f1

    if muscle_summary:
        def avg_muscle(key): 
            l = muscle_summary.get(f"L_{key}", muscle_summary.get(key))
            r = muscle_summary.get(f"R_{key}")
            vals = [v for v in [l,r] if v is not None]
            return sum(vals)/len(vals) if vals else None
        row.update({
            "act_glutemax": avg_muscle("glute_max"),
            "act_quad": avg_muscle("vastus"),
            "act_hamstring": avg_muscle("hamstrings"),
            "act_gastroc": avg_muscle("gastroc"),
            "act_soleus": avg_muscle("soleus"),
            "act_tibant": avg_muscle("tib_ant"),
        })
    # write
    client = get_client()
    ws = client.open_by_key(SPREADSHEET_ID).worksheet(WORKSHEET_NAME)
    # ensure header has all columns
    try:
        header = ws.row_values(1)
        missing = [c for c in COLUMNS if c not in header]
        if missing:
            ws.update([COLUMNS], 'A1')
    except Exception:
        pass
    ws.append_row([row.get(c,"") if row.get(c) is not None else "" for c in COLUMNS], value_input_option="USER_ENTERED")
    return row


def log_compare(metrabs_summary, rtmpose_summary,
                metrabs_perf=None, rtmpose_perf=None,
                forces_summary=None, muscle_summary=None,
                subject_meta=None, video_path=None,
                accuracy_metrabs=None, accuracy_rtmpose=None,
                f1_metrabs=None, f1_rtmpose=None, notes=""):
    """
    Run both backends on the same video, log a single comparison row to Sheets.
    ROM is stored side-by-side: rom_hip_l_metrabs / rom_hip_l_rtmpose etc.
    speed = processing_time_s (per backend)
    accuracy/f1: leave None until Label Studio GT is available.
    Returns the row dict that was written.
    """
    import uuid
    compare_run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6]
    subject_meta = subject_meta or {}
    mass_kg = subject_meta.get("mass_kg")

    rom_m = _extract_rom(metrabs_summary)
    rom_r = _extract_rom(rtmpose_summary)

    # use metrabs gait metrics as primary (usually 3D)
    gait = metrabs_summary.get("gait", {}) or rtmpose_summary.get("gait", {})

    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "engine": "compare_metrabs_rtmpose",
        "model_version": f"metrabs+rtmpose",
        "subject_id": subject_meta.get("subject_id",""),
        "video_hash": video_hash(video_path) if video_path and os.path.exists(video_path) else "",
        "mass_kg": mass_kg,
        "mass_lb": round(mass_kg*2.20462,1) if mass_kg else "",
        "height_m": subject_meta.get("height_m",""),
        "sex": subject_meta.get("sex",""),
        "age": subject_meta.get("age",""),
        # ROM MeTRAbs
        "rom_hip_l_metrabs": rom_m["hip_l"], "rom_hip_r_metrabs": rom_m["hip_r"],
        "rom_knee_l_metrabs": rom_m["knee_l"], "rom_knee_r_metrabs": rom_m["knee_r"],
        "rom_ankle_l_metrabs": rom_m["ankle_l"], "rom_ankle_r_metrabs": rom_m["ankle_r"],
        # ROM RTMPose
        "rom_hip_l_rtmpose": rom_r["hip_l"], "rom_hip_r_rtmpose": rom_r["hip_r"],
        "rom_knee_l_rtmpose": rom_r["knee_l"], "rom_knee_r_rtmpose": rom_r["knee_r"],
        "rom_ankle_l_rtmpose": rom_r["ankle_l"], "rom_ankle_r_rtmpose": rom_r["ankle_r"],
        # legacy ROM = metrabs
        "rom_hip_l": rom_m["hip_l"], "rom_hip_r": rom_m["hip_r"],
        "rom_knee_l": rom_m["knee_l"], "rom_knee_r": rom_m["knee_r"],
        "rom_ankle_l": rom_m["ankle_l"], "rom_ankle_r": rom_m["ankle_r"],
        "speed_ms": gait.get("speed_ms"),
        "cadence_spm": gait.get("cadence_spm"),
        "step_length_m": gait.get("step_length_m"),
        "peak_grf_l_bw": forces_summary.get("peak_grf_L_bw") if forces_summary else None,
        "peak_grf_r_bw": forces_summary.get("peak_grf_R_bw") if forces_summary else None,
        "peak_hip_moment": forces_summary.get("peak_hip_moment_L") if forces_summary else None,
        "peak_knee_moment": forces_summary.get("peak_knee_moment_L") if forces_summary else None,
        "peak_ankle_moment": forces_summary.get("peak_ankle_moment_L") if forces_summary else None,
        # perf - both backends
        "inference_fps_metrabs": metrabs_perf.get("inference_fps") if metrabs_perf else None,
        "processing_time_s_metrabs": metrabs_perf.get("processing_time_s") if metrabs_perf else None,
        "inference_fps_rtmpose": rtmpose_perf.get("inference_fps") if rtmpose_perf else None,
        "processing_time_s_rtmpose": rtmpose_perf.get("processing_time_s") if rtmpose_perf else None,
        # combined perf fields = metrabs
        "inference_fps": metrabs_perf.get("inference_fps") if metrabs_perf else None,
        "processing_time_s": metrabs_perf.get("processing_time_s") if metrabs_perf else None,
        "accuracy_metrabs": accuracy_metrabs,
        "accuracy_rtmpose": accuracy_rtmpose,
        "f1_metrabs": f1_metrabs,
        "f1_rtmpose": f1_rtmpose,
        "compare_run_id": compare_run_id,
        "notes": notes,
    }

    if muscle_summary:
        def avg_muscle(key): 
            l = muscle_summary.get(f"L_{key}", muscle_summary.get(key))
            r = muscle_summary.get(f"R_{key}")
            vals = [v for v in [l,r] if v is not None]
            return sum(vals)/len(vals) if vals else None
        row.update({
            "act_glutemax": avg_muscle("glute_max"),
            "act_quad": avg_muscle("vastus"),
            "act_hamstring": avg_muscle("hamstrings"),
            "act_gastroc": avg_muscle("gastroc"),
            "act_soleus": avg_muscle("soleus"),
            "act_tibant": avg_muscle("tib_ant"),
        })

    client = get_client()
    ws = client.open_by_key(SPREADSHEET_ID).worksheet(WORKSHEET_NAME)
    try:
        header = ws.row_values(1)
        missing = [c for c in COLUMNS if c not in header]
        if missing:
            ws.update([COLUMNS], 'A1')
    except Exception:
        pass
    ws.append_row([row.get(c,"") if row.get(c) is not None else "" for c in COLUMNS], value_input_option="USER_ENTERED")
    return row
