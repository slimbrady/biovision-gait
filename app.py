# app.py
# BioVision gAIt – Streamlit UI for Markerless Gait Analysis
# v0.3 – 3D forces + muscle activation + Google Sheets logging + MeTRAbs vs RTMPose compare
import json, tempfile, time
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from pose_metrabs import run_metrabs_inference
from metrics_gait import analyze_gait, fuse_multiview_metrics

try:
    from pose_rtmpose import run_rtmpose_inference, MMPOSE_AVAILABLE
    RTMPOSE_AVAILABLE = True
except Exception:
    RTMPOSE_AVAILABLE = False
    MMPOSE_AVAILABLE = False

try:
    from biomech_force import compute_forces_3d
    FORCE_AVAILABLE = True
except Exception: FORCE_AVAILABLE = False
try:
    from muscle_activation import compute_activations
    MUSCLE_AVAILABLE = True
except Exception: MUSCLE_AVAILABLE = False
try:
    from sheets_logger import log_run, log_compare
    SHEETS_AVAILABLE = True
except Exception: SHEETS_AVAILABLE = False

st.set_page_config(page_title="BioVision gAIt", page_icon="🏃", layout="wide")
st.title("BioVision gAIt – Markerless Gait Analysis")
st.caption("Apple Silicon / M4 · 3D forces · Muscle activation · Sheets logging · MeTRAbs vs RTMPose")

with st.sidebar:
    st.header("⚙️ Subject")
    mass_unit = st.radio("Mass unit", ["kg","lb"], horizontal=True, index=0, key="mass_unit_bv")
    mass_default = 75.0 if mass_unit=="kg" else 165.0
    mass_input = st.number_input(f"Body mass ({mass_unit})", 30.0 if mass_unit=="kg" else 66.0,
        200.0 if mass_unit=="kg" else 440.0, mass_default, 0.5, key="mass_bv")
    mass_kg = mass_input if mass_unit=="kg" else mass_input/2.20462
    subject_height_m = st.number_input("Subject height (m)", 1.0, 2.5, 1.75, 0.01)
    sex = st.selectbox("Sex", ["unspecified","F","M"])
    age = st.number_input("Age", 0, 100, 30)
    subject_id = st.text_input("Subject ID", "", key="subj_bv")
    log_to_sheets = st.checkbox("Log to Google Sheets", value=True) if SHEETS_AVAILABLE else False

    st.divider()
    st.header("⚙️ Analysis Settings")
    compare_mode = st.checkbox("🔬 Compare MeTRAbs vs RTMPose", value=True,
        help="Run both backends on the same video, log side-by-side ROM to Sheets")
    if compare_mode and not RTMPOSE_AVAILABLE:
        st.warning("RTMPose module not found – will use MeTRAbs fallback. Install MMPose for real comparison.")
    camera_mode = st.radio("Camera view mode", ["single","multi"],
        format_func=lambda x: "Single view" if x=="single" else "Multi-view (2-3 cameras)")
    if camera_mode == "single":
        camera_view = st.selectbox("Camera view", ["sagittal","frontal","rear"])
    else:
        st.info("Upload 1-3 videos, tag each with its view")
    conf_threshold = st.slider("Detection confidence", 0.0, 1.0, 0.3, 0.05)
    if not compare_mode:
        pose_backend = st.selectbox("Pose backend", ["MeTRAbs (3D, recommended)", "RTMPose / WholeBody (2D+feet)"])
        backend_key = "metrabs" if "MeTRAbs" in pose_backend else "rtmpose"
    else:
        backend_key = "compare"
    model_size = st.selectbox("MeTRAbs model size", [
        "efficientnetv2_s (fast, M4 default)", "mobilenetv3 (fastest)", "efficientnetv2_l (accurate)"])
    backend_map = {"efficientnetv2_s (fast, M4 default)":"efficientnetv2_s",
                   "mobilenetv3 (fastest)":"mobilenetv3",
                   "efficientnetv2_l (accurate)":"efficientnetv2_l"}
    model_backend = backend_map[model_size]

# Upload
st.subheader("📹 Upload Video(s)")
if camera_mode == "single":
    uploaded = st.file_uploader("Upload gait video", type=["mp4","mov","avi","m4v"], accept_multiple_files=False)
    uploads = [(camera_view, uploaded)] if uploaded else []
else:
    c1,c2,c3 = st.columns(3)
    with c1: up_sag = st.file_uploader("Sagittal (side)", type=["mp4","mov","avi","m4v"], key="sag")
    with c2: up_fro = st.file_uploader("Frontal", type=["mp4","mov","avi","m4v"], key="fro")
    with c3: up_rear = st.file_uploader("Rear", type=["mp4","mov","avi","m4v"], key="rear")
    uploads = []
    if up_sag: uploads.append(("sagittal", up_sag))
    if up_fro: uploads.append(("frontal", up_fro))
    if up_rear: uploads.append(("rear", up_rear))

if not uploads:
    st.info("👆 Upload at least one video."); st.stop()

run_btn = st.button("🚀 Run Gait Analysis", type="primary", use_container_width=True)
if not run_btn and "results" not in st.session_state: st.stop()
if run_btn: st.session_state.pop("results", None)

def run_single_backend(backend_name, video_path, view, tmpdir):
    """Run pose + gait for one backend. Returns (pose_result, gait_result, perf, forces_summary, act_summary, df_forces, df_act)"""
    t_start = time.time()
    if backend_name == "metrabs":
        pose_result = run_metrabs_inference(str(video_path), camera_view=view,
            backend=model_backend, conf_threshold=conf_threshold,
            output_dir=str(tmpdir), draw_overlay=True)
    elif backend_name == "rtmpose" and RTMPOSE_AVAILABLE:
        pose_result = run_rtmpose_inference(str(video_path), camera_view=view,
            backend="rtmpose_m", conf_threshold=conf_threshold,
            output_dir=str(tmpdir), draw_overlay=True)
    else:
        # fallback
        pose_result = run_metrabs_inference(str(video_path), camera_view=view,
            backend=model_backend, conf_threshold=conf_threshold,
            output_dir=str(tmpdir), draw_overlay=True)
        if backend_name == "rtmpose":
            pose_result["_rtmpose_stub"] = True

    gait_result = analyze_gait(pose_result["keypoints_3d"], pose_result["fps"],
        camera_view=view, subject_height_m=subject_height_m,
        output_prefix=str(tmpdir / f"metrics_{view}_{backend_name}"))

    # Force + Muscle (MeTRAbs 3D only – skip for RTMPose 2D)
    df_forces = pd.DataFrame(); forces_summary = {}
    df_act = pd.DataFrame(); act_summary = {}
    if FORCE_AVAILABLE and backend_name == "metrabs":
        try:
            kpts_3d = pose_result["keypoints_3d"]
            def j(idx): return kpts_3d[:, idx, :]
            try:
                joint_dict = {
                    'R_hip': j(1), 'R_knee': j(2), 'R_ankle': j(3),
                    'L_hip': j(4), 'L_knee': j(5), 'L_ankle': j(6),
                }
                df_forces, forces_summary = compute_forces_3d(joint_dict, pose_result["fps"], mass_kg)
            except Exception as e:
                pass
        except Exception:
            pass
    if MUSCLE_AVAILABLE and not df_forces.empty:
        try:
            angles = gait_result.get("angles", {})
            df_act, act_summary = compute_activations(angles, df_forces, pose_result["fps"])
        except Exception:
            pass

    perf = {"inference_fps": pose_result.get("fps", 0), "processing_time_s": time.time() - t_start}
    return pose_result, gait_result, perf, forces_summary, act_summary, df_forces, df_act

if run_btn:
    t0 = time.time()
    with st.status("Running pose estimation + gait analysis…", expanded=True) as status:
        tmpdir = Path(tempfile.mkdtemp(prefix="biovision_"))
        results_by_view = {}
        compare_results = {}  # view -> {metrabs: {...}, rtmpose: {...}}
        for view, up_file in uploads:
            st.write(f"**{view.capitalize()}:** {up_file.name}")
            tmp_video = tmpdir / up_file.name
            tmp_video.write_bytes(up_file.read())

            if compare_mode:
                # ---- MeTRAbs ----
                st.write("  → MeTRAbs pose…")
                pose_m, gait_m, perf_m, forces_m, act_m, df_forces_m, df_act_m = run_single_backend("metrabs", tmp_video, view, tmpdir)
                st.write(f"     ✓ {perf_m['processing_time_s']:.1f}s")
                # ---- RTMPose ----
                st.write("  → RTMPose pose…")
                pose_r, gait_r, perf_r, forces_r, act_r, df_forces_r, df_act_r = run_single_backend("rtmpose", tmp_video, view, tmpdir)
                if pose_r.get("_rtmpose_stub"):
                    st.warning("RTMPose not installed – using MeTRAbs fallback. pip install mmpose to get real comparison.")
                st.write(f"     ✓ {perf_r['processing_time_s']:.1f}s")

                # store both, display MeTRAbs as primary
                gait_m["pose_result"] = pose_m
                gait_m["view"] = view
                gait_m["video_name"] = up_file.name
                gait_m["df_forces"] = df_forces_m
                gait_m["forces_summary"] = forces_m
                gait_m["df_act"] = df_act_m
                gait_m["act_summary"] = act_m
                results_by_view[view] = gait_m

                compare_results[view] = {
                    "metrabs": {"gait": gait_m, "perf": perf_m, "pose": pose_m, "forces": forces_m, "act": act_m},
                    "rtmpose": {"gait": gait_r, "perf": perf_r, "pose": pose_r, "forces": forces_r, "act": act_r},
                }

                # Sheets compare log
                if log_to_sheets and SHEETS_AVAILABLE:
                    try:
                        row = log_compare(
                            gait_m, gait_r,
                            metrabs_perf=perf_m, rtmpose_perf=perf_r,
                            forces_summary=forces_m, muscle_summary=act_m,
                            subject_meta={"mass_kg": mass_kg, "height_m": subject_height_m, "sex": sex, "age": age, "subject_id": subject_id},
                            video_path=str(tmp_video),
                            notes=f"view={view}"
                        )
                        st.write(f"  📊 Logged to Sheets: {row['compare_run_id']}")
                    except Exception as e:
                        st.warning(f"Sheets log failed: {e}")
            else:
                # single-backend mode (original)
                st.write(f"  → {backend_key} pose…")
                pose_result, gait_result, perf, forces_summary, act_summary, df_forces, df_act = run_single_backend(backend_key, tmp_video, view, tmpdir)
                gait_result["pose_result"] = pose_result
                gait_result["view"] = view
                gait_result["video_name"] = up_file.name
                gait_result["df_forces"] = df_forces
                gait_result["forces_summary"] = forces_summary
                gait_result["df_act"] = df_act
                gait_result["act_summary"] = act_summary
                results_by_view[view] = gait_result
                if log_to_sheets and SHEETS_AVAILABLE:
                    try:
                        log_run(gait_result, forces_summary, act_summary,
                            engine=backend_key, model_version=model_backend,
                            subject_meta={"mass_kg": mass_kg, "height_m": subject_height_m, "sex": sex, "age": age, "subject_id": subject_id},
                            perf=perf, video_path=str(tmp_video))
                    except Exception as e:
                        st.warning(f"Sheets log failed: {e}")

        if len(results_by_view) > 1 and not compare_mode:
            st.write("  → Fusing multi-view…")
            fused = fuse_multiview_metrics(results_by_view)
            results_by_view["_fused"] = fused

        st.session_state["results"] = results_by_view
        st.session_state["compare_results"] = compare_results
        status.update(label="✅ Analysis complete!", state="complete")

results_by_view = st.session_state.get("results", {})
compare_results = st.session_state.get("compare_results", {})
if not results_by_view: st.stop()
display_key = "_fused" if "_fused" in results_by_view else next(k for k in results_by_view if not k.startswith("_"))
result = results_by_view[display_key]

# --- Compare ROM table (shown first when compare_mode was used) ---
if compare_results and display_key in compare_results:
    st.subheader("🔬 MeTRAbs vs RTMPose – ROM Comparison")
    cr = compare_results[display_key]
    gait_m = cr["metrabs"]["gait"]; perf_m = cr["metrabs"]["perf"]
    gait_r = cr["rtmpose"]["gait"]; perf_r = cr["rtmpose"]["perf"]

    def get_rom(g, joint):
        return g.get("rom_deg", {}).get(joint, np.nan)

    rom_rows = [
        ("Hip L", get_rom(gait_m, "L_hip_rom"), get_rom(gait_r, "L_hip_rom")),
        ("Hip R", get_rom(gait_m, "R_hip_rom"), get_rom(gait_r, "R_hip_rom")),
        ("Knee L", get_rom(gait_m, "L_knee_rom"), get_rom(gait_r, "L_knee_rom")),
        ("Knee R", get_rom(gait_m, "R_knee_rom"), get_rom(gait_r, "R_knee_rom")),
        ("Ankle L", get_rom(gait_m, "L_ankle_rom"), get_rom(gait_r, "L_ankle_rom")),
        ("Ankle R", get_rom(gait_m, "R_ankle_rom"), get_rom(gait_r, "R_ankle_rom")),
    ]
    df_cmp = pd.DataFrame([
        {"Joint": j, "MeTRAbs (°)": round(m,1) if np.isfinite(m) else None,
         "RTMPose (°)": round(r,1) if np.isfinite(r) else None,
         "Δ (°)": round(m-r,1) if np.isfinite(m) and np.isfinite(r) else None}
        for j, m, r in rom_rows
    ])
    st.dataframe(df_cmp, hide_index=True, use_container_width=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MeTRAbs time", f"{perf_m['processing_time_s']:.2f}s")
    c2.metric("RTMPose time", f"{perf_r['processing_time_s']:.2f}s")
    speedup = perf_m['processing_time_s'] / perf_r['processing_time_s'] if perf_r['processing_time_s'] > 0 else 0
    c3.metric("Speedup", f"{speedup:.1f}×" if speedup else "—")
    c4.metric("Δ avg ROM", f"{np.nanmean([abs(m-r) for _,m,r in rom_rows]):.1f}°")
    st.divider()

# Overlay videos
st.subheader("🎬 Pose Overlay")
view_keys = [k for k in results_by_view if not k.startswith("_")]
if compare_results and display_key in compare_results:
    # show both overlays side by side
    cr = compare_results[display_key]
    col_m, col_r = st.columns(2)
    with col_m:
        st.markdown("**MeTRAbs**")
        op = cr["metrabs"]["pose"].get("overlay_path")
        if op and Path(op).exists(): st.video(op)
    with col_r:
        st.markdown("**RTMPose**")
        op = cr["rtmpose"]["pose"].get("overlay_path")
        is_stub = cr["rtmpose"]["pose"].get("_rtmpose_stub", False)
        if is_stub: st.caption("⚠️ RTMPose not installed – showing MeTRAbs fallback")
        if op and Path(op).exists(): st.video(op)
else:
    cols = st.columns(len(view_keys))
    for col, vk in zip(cols, view_keys):
        with col:
            pr = results_by_view[vk].get("pose_result", {})
            overlay_path = pr.get("overlay_path")
            st.markdown(f"**{vk.capitalize()}**")
            if overlay_path and Path(overlay_path).exists(): st.video(overlay_path)

# Metrics dashboard
st.subheader("📊 Gait Metrics Dashboard")
st_params = result.get("spatiotemporal", {})
def metric_card(label, value, unit="", normal_range=None):
    if value is None or (isinstance(value, float) and np.isnan(value)): st.metric(label, "N/A"); return
    st.metric(label, f"{value:.2f} {unit}".strip())
m1,m2,m3,m4,m5 = st.columns(5)
with m1: metric_card("Speed", st_params.get("speed_m_s"), "m/s")
with m2: metric_card("Cadence", st_params.get("cadence_steps_per_min"), "spm")
with m3:
    sl = st_params.get("step_length_r_m", np.nan)
    if np.isnan(sl): sl = st_params.get("step_length_l_m", np.nan)
    metric_card("Step length", sl, "m")
with m4: metric_card("Stride time", st_params.get("stride_time_s"), "s")
with m5:
    sw = st_params.get("step_width_m", np.nan)
    metric_card("Stance width", sw*100 if np.isfinite(sw) else np.nan, "cm")

# Force summary if available
forces_summary = result.get("forces_summary", {})
if forces_summary:
    st.subheader("💪 Force Summary")
    f1,f2,f3,f4 = st.columns(4)
    f1.metric("Peak GRF L", f"{forces_summary.get('peak_grf_L_bw',0):.2f} ×BW")
    f2.metric("Peak GRF R", f"{forces_summary.get('peak_grf_R_bw',0):.2f} ×BW")
    f3.metric("Peak Knee Moment", f"{forces_summary.get('peak_knee_moment_L',0):.2f} Nm/kg")
    f4.metric("Peak Ankle Moment", f"{forces_summary.get('peak_ankle_moment_L',0):.2f} Nm/kg")

# Tabs: ROM / Forces / Muscles
tab_rom, tab_forces, tab_muscles = st.tabs(["📐 ROM", "💪 Forces", "⚡ Muscle Activation"])

with tab_rom:
    rom_summary = result.get("rom_summary", {})
    rows = []
    angle_sources = result.get("angle_sources", {})
    for joint, stats in sorted(rom_summary.items()):
        rom = stats.get("rom", np.nan)
        if np.isnan(rom): continue
        mean = stats.get("mean", np.nan); vmin = stats.get("min", np.nan); vmax = stats.get("max", np.nan)
        rows.append({"Joint": joint.replace("_"," ").title(), "ROM (°)": f"{rom:.1f}",
                     "Mean (°)": f"{mean:.1f}" if np.isfinite(mean) else "—",
                     "Min": f"{vmin:.1f}" if np.isfinite(vmin) else "—",
                     "Max": f"{vmax:.1f}" if np.isfinite(vmax) else "—",
                     "View": angle_sources.get(joint, display_key)})
    if rows: st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    # angle time series
    angles = result.get("angles", {})
    if angles:
        plot_keys = [k for k in angles if np.any(np.isfinite(angles[k]))][:10]
        if plot_keys:
            fig = go.Figure()
            n_frames = len(next(iter(angles.values())))
            fps_est = 30.0
            for vk in view_keys:
                if "pose_result" in results_by_view[vk]:
                    fps_est = results_by_view[vk]["pose_result"].get("fps", 30.0); break
            t = np.arange(n_frames)/fps_est
            for k in plot_keys:
                fig.add_trace(go.Scatter(x=t, y=angles[k], mode="lines", name=k))
            fig.update_layout(xaxis_title="Time (s)", yaxis_title="Angle (°)", height=400)
            st.plotly_chart(fig, use_container_width=True)

with tab_forces:
    df_forces = result.get("df_forces", pd.DataFrame())
    if df_forces.empty:
        st.info("Force estimation unavailable – check biomech_force.py")
    else:
        grf_cols = [c for c in df_forces.columns if 'grf_bw' in c.lower()]
        if grf_cols: st.plotly_chart(px.line(df_forces, x='time_s', y=grf_cols, title="GRF (%BW)"), use_container_width=True)
        mom_cols = [c for c in df_forces.columns if 'moment_nmk' in c]
        if mom_cols: st.plotly_chart(px.line(df_forces, x='time_s', y=mom_cols[:6], title="Joint Moments (Nm/kg)"), use_container_width=True)
        st.json(forces_summary)

with tab_muscles:
    df_act = result.get("df_act", pd.DataFrame())
    act_summary = result.get("act_summary", {})
    if df_act.empty:
        st.info("Muscle activation unavailable – check muscle_activation.py")
    else:
        muscle_cols = [c for c in df_act.columns if c != 'time_s']
        fig_a = px.line(df_act, x='time_s', y=muscle_cols, title="Muscle Activation (0-1)")
        fig_a.update_yaxes(range=[0,1.05])
        st.plotly_chart(fig_a, use_container_width=True)
        st.json(act_summary)

# Downloads
st.subheader("⬇️ Download Results")
for vk in view_keys:
    r = results_by_view[vk]
    csv_path = r.get("csv_path"); json_path = r.get("json_path")
    overlay = r.get("pose_result", {}).get("overlay_path")
    c1,c2,c3,c4,c5 = st.columns(5)
    if csv_path and Path(csv_path).exists():
        c1.download_button(f"📄 Angles ({vk})", Path(csv_path).read_bytes(), f"angles_{vk}.csv", "text/csv", key=f"csv_{vk}")
    if json_path and Path(json_path).exists():
        c2.download_button(f"📋 JSON ({vk})", Path(json_path).read_bytes(), f"summary_{vk}.json", "application/json", key=f"json_{vk}")
    if overlay and Path(overlay).exists():
        c3.download_button(f"🎬 MP4 ({vk})", Path(overlay).read_bytes(), f"overlay_{vk}.mp4", "video/mp4", key=f"mp4_{vk}")
    df_f = r.get("df_forces", pd.DataFrame())
    if not df_f.empty:
        c4.download_button(f"💪 Forces ({vk})", df_f.to_csv(index=False).encode(), f"forces_{vk}.csv", "text/csv", key=f"force_{vk}")
    df_a = r.get("df_act", pd.DataFrame())
    if not df_a.empty:
        c5.download_button(f"⚡ Muscles ({vk})", df_a.to_csv(index=False).encode(), f"muscles_{vk}.csv", "text/csv", key=f"mus_{vk}")

st.divider()
with st.expander("📚 Research & Citations"):
    st.markdown("""
**Validation papers – BioVision / MeTRAbs pipeline**

1. Chougule et al., 2026 – *Accuracy and Validity of 3D Markerless Motion Capture* – https://doi.org/10.3390/s26123956
2. Çabuk et al., 2025 – *Can OpenCap deliver valid kinematic data?* – Biocybernetics and Biomedical Engineering
3. D'Souza et al., 2024 – *Theia3D vs marker-based gait* – Sci Rep https://doi.org/10.1038/s41598-024-80499-8
4. Wren et al., 2023 – *Theia markerless vs Vicon in clinical patients* – Gait & Posture https://doi.org/10.1016/j.gaitpost.2023.05.029
5. Cao et al., 2026 – *Markerless gait validation in ankle-injury patients* – Sensors

**Validation papers – RTMPose / 2D pose pipeline**

1. Guo & Zhao, 2024 – *Gait analysis based on RTMPose using knee angle*
2. Menychtas et al., 2023 – *Gait analysis: 2D pose vs 3D marker-based* – Front. Rehabil. Sci. https://doi.org/10.3389/fresc.2023.1238134
3. Wade et al., 2022 – *Applications and limitations of markerless motion capture* – PeerJ https://doi.org/10.7717/peerj.12995
4. Tang et al., 2022 – *Joint Moment and Power: markerless vs marker-based running* – https://doi.org/10.3390/biomechanics9040574
5. Johnson et al., 2022 – *Foot and Tibia Angles During Running – markerless vs manual* – J Appl Biomech

---
**Full citation table:** [Google Sheet – Citations tab](https://docs.google.com/spreadsheets/d/1o4aA07t5ODfsXtLl5M0j6SudLRbGovgxpDUzHKFFOk8/edit#gid=1112676311)

**GitHub repos:**
- BioVision-gait: https://github.com/slimbrady/biovision-gait
- gait-pose-m4: https://github.com/slimbrady/gait-pose-m4
""")

st.divider()
st.caption(f"BioVision gAIt v0.3 · MeTRAbs vs RTMPose · mass: {mass_kg:.1f} kg · github.com/slimbrady/biovision-gait")
