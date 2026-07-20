# app.py
# BioVision gAIt – Streamlit UI for Markerless Gait Analysis
# Apple Silicon / M4 optimized

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pose_metrabs import run_multiview_inference, run_metrabs_inference
from metrics_gait import (
    analyze_gait, compute_all_joint_angles,
    fuse_multiview_metrics, compute_rom_summary
)

st.set_page_config(
    page_title="BioVision gAIt",
    page_icon="🏃",
    layout="wide",
)

st.title("BioVision gAIt – Markerless Gait Analysis (MeTRAbs 3D)")
st.caption("Apple Silicon / M4 optimized · 3D metric-scale pose · Multi-view support")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Analysis Settings")

    camera_mode = st.radio(
        "Camera view mode",
        ["single", "multi"],
        format_func=lambda x: "Single view" if x == "single" else "Multi-view (2-3 cameras)",
        index=0,
    )

    if camera_mode == "single":
        camera_view = st.selectbox(
            "Camera view",
            ["sagittal", "frontal", "rear"],
            help="Sagittal: side view – hip/knee/ankle flexion, speed, cadence\n"
                 "Frontal: front view – hip abd/add, stance width, trunk lean\n"
                 "Rear: back view – calcaneal inversion/eversion, foot strike"
        )
    else:
        st.info("Upload 1-3 videos, tag each with its view (sagittal/frontal/rear)")

    st.divider()
    subject_height_m = st.number_input(
        "Subject height (m)", min_value=1.0, max_value=2.5,
        value=1.75, step=0.01,
        help="Used for gait parameter scaling / calibration"
    )

    conf_threshold = st.slider("Detection confidence threshold", 0.0, 1.0, 0.3, 0.05)

    pose_backend = st.selectbox(
        "Pose backend",
        ["MeTRAbs (3D, recommended)", "RTMPose / WholeBody (2D+feet)"],
        help="MeTRAbs: 3D metric-scale, fast on M4 CPU\n"
             "RTMPose: COCO-WholeBody with foot keypoints for calcaneal angle"
    )
    backend_key = "metrabs" if "MeTRAbs" in pose_backend else "rtmpose"

    model_size = st.selectbox(
        "Model size",
        ["efficientnetv2_s (fast, M4 default)",
         "mobilenetv3 (fastest)",
         "efficientnetv2_l (accurate)"],
    )
    backend_map = {
        "efficientnetv2_s (fast, M4 default)": "efficientnetv2_s",
        "mobilenetv3 (fastest)": "mobilenetv3",
        "efficientnetv2_l (accurate)": "efficientnetv2_l",
    }
    model_backend = backend_map[model_size]

    st.divider()
    st.caption("💡 **Tip:** Sagittal view alone gives most gait metrics. "
               "Add frontal/rear for hip abduction, stance width, "
               "and calcaneal inversion/eversion.")

# ---------------------------------------------------------------------------
# File upload
# ---------------------------------------------------------------------------

st.subheader("📹 Upload Video(s)")

if camera_mode == "single":
    uploaded = st.file_uploader(
        "Upload gait video",
        type=["mp4", "mov", "avi", "m4v"],
        accept_multiple_files=False,
    )
    uploads = [(camera_view, uploaded)] if uploaded else []
else:
    c1, c2, c3 = st.columns(3)
    with c1:
        up_sag = st.file_uploader("Sagittal (side)", type=["mp4", "mov", "avi", "m4v"], key="sag")
    with c2:
        up_fro = st.file_uploader("Frontal", type=["mp4", "mov", "avi", "m4v"], key="fro")
    with c3:
        up_rear = st.file_uploader("Rear", type=["mp4", "mov", "avi", "m4v"], key="rear")
    uploads = []
    if up_sag: uploads.append(("sagittal", up_sag))
    if up_fro: uploads.append(("frontal", up_fro))
    if up_rear: uploads.append(("rear", up_rear))

if not uploads:
    st.info("👆 Upload at least one video to begin analysis.")
    st.stop()

# ---------------------------------------------------------------------------
# Run analysis
# ---------------------------------------------------------------------------

run_btn = st.button("🚀 Run Gait Analysis", type="primary", use_container_width=True)

if not run_btn and "results" not in st.session_state:
    st.stop()

if run_btn:
    st.session_state.pop("results", None)

    with st.status("Running pose estimation + gait analysis…", expanded=True) as status:
        tmpdir = Path(tempfile.mkdtemp(prefix="biovision_"))
        results_by_view = {}

        for view, up_file in uploads:
            st.write(f"**{view.capitalize()} view:** {up_file.name}")
            # Save uploaded file
            tmp_video = tmpdir / up_file.name
            tmp_video.write_bytes(up_file.read())

            # Pose inference
            st.write(f"  → Running {backend_key} pose estimation…")
            try:
                pose_result = run_metrabs_inference(
                    str(tmp_video),
                    camera_view=view,
                    backend=model_backend,
                    conf_threshold=conf_threshold,
                    output_dir=str(tmpdir),
                    draw_overlay=True,
                )
            except Exception as e:
                st.error(f"Pose estimation failed: {e}")
                st.stop()

            # Gait metrics
            st.write(f"  → Computing joint angles + gait parameters…")
            gait_result = analyze_gait(
                pose_result["keypoints_3d"],
                pose_result["fps"],
                camera_view=view,
                subject_height_m=subject_height_m,
                output_prefix=str(tmpdir / f"metrics_{view}"),
            )
            gait_result["pose_result"] = pose_result
            gait_result["view"] = view
            gait_result["video_name"] = up_file.name
            results_by_view[view] = gait_result

        # Multi-view fusion
        if len(results_by_view) > 1:
            st.write("  → Fusing multi-view metrics…")
            fused = fuse_multiview_metrics(results_by_view)
            results_by_view["_fused"] = fused

        st.session_state["results"] = results_by_view
        status.update(label="✅ Analysis complete!", state="complete")

results_by_view = st.session_state.get("results", {})
if not results_by_view:
    st.stop()

# Pick display result: fused if available, else single view
display_key = "_fused" if "_fused" in results_by_view else next(k for k in results_by_view if not k.startswith("_"))
result = results_by_view[display_key]
is_fused = display_key == "_fused"

# ---------------------------------------------------------------------------
# Results – video overlays
# ---------------------------------------------------------------------------

st.subheader("🎬 Pose Overlay")
view_keys = [k for k in results_by_view if not k.startswith("_")]
cols = st.columns(len(view_keys))
for col, vk in zip(cols, view_keys):
    with col:
        pr = results_by_view[vk].get("pose_result", {})
        overlay_path = pr.get("overlay_path")
        st.markdown(f"**{vk.capitalize()} view**")
        if overlay_path and Path(overlay_path).exists():
            st.video(overlay_path)
        else:
            st.caption("(overlay unavailable)")

# ---------------------------------------------------------------------------
# Metrics dashboard
# ---------------------------------------------------------------------------

st.subheader("📊 Gait Metrics Dashboard")

st_params = result.get("spatiotemporal", {})

def metric_card(label, value, unit="", normal_range=None):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        st.metric(label, "N/A", delta=None)
        return
    delta = None
    delta_color = "off"
    if normal_range:
        lo, hi = normal_range
        if value < lo:
            delta = "below normal"
            delta_color = "inverse"
        elif value > hi:
            delta = "above normal"
        else:
            delta = "normal"
    st.metric(label, f"{value:.2f} {unit}".strip(), delta=delta, delta_color=delta_color)

m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    metric_card("Speed", st_params.get("speed_m_s"), "m/s", (1.0, 1.6))
with m2:
    metric_card("Cadence", st_params.get("cadence_steps_per_min"), "spm", (100, 130))
with m3:
    sl = st_params.get("step_length_r_m", np.nan)
    if np.isnan(sl): sl = st_params.get("step_length_l_m", np.nan)
    metric_card("Step length", sl, "m", (0.5, 0.9))
with m4:
    metric_card("Stride time", st_params.get("stride_time_s"), "s", (0.9, 1.3))
with m5:
    sw = st_params.get("step_width_m", np.nan)
    metric_card("Stance width", sw * 100 if np.isfinite(sw) else np.nan, "cm", (8, 15))

# ---------------------------------------------------------------------------
# Joint ROM table
# ---------------------------------------------------------------------------

st.subheader("🦴 Joint Range of Motion")

rom_summary = result.get("rom_summary", {})

# Clinical normal ROM reference (gait, not max passive)
CLINICAL_NORM = {
    "hip_flexion_r": "40° flex / 10° ext", "hip_flexion_l": "40° flex / 10° ext",
    "knee_flexion_r": "0-70°", "knee_flexion_l": "0-70°",
    "ankle_dorsiflexion_r": "10° DF / 20° PF", "ankle_dorsiflexion_l": "10° DF / 20° PF",
    "trunk_flexion": "0-10°",
    "neck_flexion": "0-10°",
    "shoulder_flexion_r": "20-45°", "shoulder_flexion_l": "20-45°",
    "elbow_flexion_r": "70-120°", "elbow_flexion_l": "70-120°",
    "hip_abduction_r": "±5-10°", "hip_abduction_l": "±5-10°",
    "calc_eversion_r_ESTIMATE": "±5-10°", "calc_eversion_l_ESTIMATE": "±5-10°",
    "trunk_lean_frontal": "<5°",
    "pelvic_obliquity": "5-10°",
}

# Build table rows
rows = []
angle_sources = result.get("angle_sources", {})
for joint, stats in sorted(rom_summary.items()):
    rom = stats.get("rom", np.nan)
    mean = stats.get("mean", np.nan)
    vmin = stats.get("min", np.nan)
    vmax = stats.get("max", np.nan)
    if np.isnan(rom):
        continue
    source = angle_sources.get(joint, display_key if not is_fused else "fused")
    normal = CLINICAL_NORM.get(joint, "—")
    # Split joint / side
    if joint.endswith("_r") or joint.endswith("_l"):
        side = joint[-1].upper()
        joint_name = joint[:-2]
    elif "_r_" in joint or "_l_" in joint:
        # e.g. calc_eversion_r_ESTIMATE
        parts = joint.split("_")
        side = "R" if "r" in parts else "L" if "l" in parts else "—"
        joint_name = joint.replace("_r_", "_").replace("_l_", "_")
    else:
        side = "—"
        joint_name = joint

    rows.append({
        "Joint": joint_name.replace("_", " ").title(),
        "Side": side,
        "ROM (°)": f"{rom:.1f}",
        "Mean (°)": f"{mean:.1f}" if np.isfinite(mean) else "—",
        "Min": f"{vmin:.1f}" if np.isfinite(vmin) else "—",
        "Max": f"{vmax:.1f}" if np.isfinite(vmax) else "—",
        "Normal Range": normal,
        "View": source,
    })

if rows:
    df_rom = pd.DataFrame(rows)
    st.dataframe(df_rom, use_container_width=True, hide_index=True)
else:
    st.info("No ROM data available.")

# ---------------------------------------------------------------------------
# Time-series plots
# ---------------------------------------------------------------------------

st.subheader("📈 Joint Angle Time Series")

angles = result.get("angles", {})
if angles:
    # Group angles
    sagittal_keys = [k for k in angles if any(x in k for x in
        ["hip_flexion", "knee_flexion", "ankle_dorsiflexion", "trunk_flexion",
         "neck_flexion", "shoulder_flexion", "elbow_flexion"])]
    frontal_keys = [k for k in angles if any(x in k for x in
        ["abduction", "lean_frontal", "pelvic_obliquity", "calc_eversion", "foot_progression"])]

    plot_group = st.radio("Plot group", ["Sagittal", "Frontal", "Spatiotemporal"],
                          horizontal=True)

    if plot_group == "Sagittal":
        plot_keys = [k for k in sagittal_keys if np.any(np.isfinite(angles[k]))]
        y_title = "Angle (°)"
    elif plot_group == "Frontal":
        plot_keys = [k for k in frontal_keys if np.any(np.isfinite(angles[k]))]
        y_title = "Angle (°)"
    else:
        plot_keys = []
        y_title = ""

    if plot_keys:
        fig = go.Figure()
        n_frames = len(next(iter(angles.values())))
        # Approximate time axis
        fps_est = 30.0
        for vk in view_keys:
            if "pose_result" in results_by_view[vk]:
                fps_est = results_by_view[vk]["pose_result"].get("fps", 30.0)
                break
        t = np.arange(n_frames) / fps_est

        for k in plot_keys[:10]:  # limit to 10 traces
            y = angles[k]
            fig.add_trace(go.Scatter(x=t, y=y, mode="lines", name=k))
        fig.update_layout(
            xaxis_title="Time (s)",
            yaxis_title=y_title,
            height=420,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(l=20, r=20, t=30, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)
    elif plot_group == "Spatiotemporal":
        st.info("Spatiotemporal plots coming soon – see Metrics Dashboard above.")
    else:
        st.info(f"No {plot_group.lower()} angles available for the current camera view(s). "
                f"Upload a {plot_group.lower()} view video to compute these metrics.")
else:
    st.info("No angle data available.")

# ---------------------------------------------------------------------------
# Gait abnormality flags
# ---------------------------------------------------------------------------

st.subheader("🚩 Gait Abnormality Screening")

flags = []

def get_rom_stat(joint, field="mean"):
    s = rom_summary.get(joint, {})
    v = s.get(field, np.nan)
    return v if np.isfinite(v) else None

# Genu valgum / varum – excessive hip adduction
for side in ["r", "l"]:
    v = get_rom_stat(f"hip_abduction_{side}", "mean")
    if v is not None and abs(v) > 10:
        flags.append(f"⚠️ Hip adduction {side.upper()}: {v:.1f}° – "
                     f"possible genu valgum / Trendelenburg (>10° threshold)")

# Excessive pronation
for side in ["r", "l"]:
    v = get_rom_stat(f"calc_eversion_{side}_ESTIMATE", "max")
    if v is not None and v > 10:
        flags.append(f"⚠️ Calcaneal eversion {side.upper()}: {v:.1f}° – "
                     f"excessive pronation (>10°). "
                     f"Note: ESTIMATE only – use RTMPose/WholeBody for accurate rearfoot angle.")

# Trunk lean asymmetry
v = get_rom_stat("trunk_lean_frontal", "mean")
if v is not None and abs(v) > 10:
    flags.append(f"⚠️ Trunk lateral lean: {v:.1f}° – compensation pattern (>10°)")

# Pelvic drop
v = get_rom_stat("pelvic_obliquity", "rom")
if v is not None and v > 15:
    flags.append(f"⚠️ Pelvic obliquity ROM: {v:.1f}° – possible Trendelenburg sign (>15°)")

# Cadence
cad = st_params.get("cadence_steps_per_min", np.nan)
if np.isfinite(cad):
    if cad < 100:
        flags.append(f"ℹ️ Low cadence: {cad:.0f} spm – typical walking is 100-120, running 160-190")
    elif cad > 200:
        flags.append(f"ℹ️ Very high cadence: {cad:.0f} spm")

if flags:
    for f in flags:
        st.warning(f)
else:
    st.success("✅ No gait abnormalities detected above screening thresholds. "
               "Note: This is a simple rule-based screen, not a clinical diagnosis.")

st.caption("Clinical thresholds are general screening values. "
           "Consult a qualified clinician for diagnosis. "
           "Calcaneal eversion is ESTIMATED from ankle mediolateral sway "
           "when using MeTRAbs – use RTMPose/WholeBody for true rearfoot angle.")

# ---------------------------------------------------------------------------
# Downloads
# ---------------------------------------------------------------------------

st.subheader("⬇️ Download Results")

dl_cols = st.columns(3)

# Find CSV / JSON files
for vk in view_keys:
    r = results_by_view[vk]
    csv_path = r.get("csv_path")
    json_path = r.get("json_path")
    overlay = r.get("pose_result", {}).get("overlay_path")

    with dl_cols[0]:
        if csv_path and Path(csv_path).exists():
            st.download_button(
                f"📄 Angles CSV ({vk})",
                Path(csv_path).read_bytes(),
                file_name=f"angles_{vk}.csv",
                mime="text/csv",
                key=f"csv_{vk}"
            )
    with dl_cols[1]:
        if json_path and Path(json_path).exists():
            st.download_button(
                f"📋 Summary JSON ({vk})",
                Path(json_path).read_bytes(),
                file_name=f"summary_{vk}.json",
                mime="application/json",
                key=f"json_{vk}"
            )
    with dl_cols[2]:
        if overlay and Path(overlay).exists():
            st.download_button(
                f"🎬 Overlay MP4 ({vk})",
                Path(overlay).read_bytes(),
                file_name=f"overlay_{vk}.mp4",
                mime="video/mp4",
                key=f"mp4_{vk}"
            )

st.divider()
st.caption("BioVision gAIt v0.1 · MeTRAbs 3D · Apple Silicon optimized · "
           "github.com/slimbrady/biovision-gait")
