# BioVision gAIt – Markerless 3D Gait Analysis for Apple Silicon

Markerless running/walking gait analysis with **MeTRAbs 3D pose** + COCO keypoints, multi-view fusion (sagittal / frontal / rear), and comprehensive joint ROM metrics. Optimized for **MacBook M4 / Apple Silicon** – no CUDA, no GPU required.

![License](https://img.shields.io/badge/license-MIT-blue) ![Platform](https://img.shields.io/badge/platform-macOS%20%7C%20Linux-lightgrey) ![Python](https://img.shields.io/badge/python-3.9%2B-blue)

---

## ✨ Features

- **3D metric-scale pose** via [MeTRAbs](https://github.com/isarandi/metrabs) – joint positions in millimeters, zero-shot, no camera calibration
- **Multi-view support**: sagittal (side), frontal, rear – fuse measurements across views
- **12 joint angles** with per-frame time series + ROM (min/max/mean/std)
  - Sagittal: hip/knee/ankle flexion, trunk/neck flexion, shoulder/elbow flexion
  - Frontal: hip abduction/adduction, trunk lateral lean, pelvic obliquity, calcaneal inversion/eversion
- **Spatiotemporal gait parameters**: speed, cadence, step/stride length, stance/swing time, step width
- **Gait abnormality screening**: genu valgum/varum, excessive pronation, Trendelenburg pattern, trunk lean compensation
- **Streamlit UI** with pose overlay, Plotly time-series charts, downloadable CSV/JSON
- **Apple Silicon optimized**: EfficientNetV2-S / MobileNetV3 backbones, ~15-30 FPS on M4 CPU
- **Optional WHAM adapter** for SMPL mesh + global trajectory + foot-ground contact (GPU recommended)

---

## 🆚 Model Comparison

| Feature | **MeTRAbs** (default) | RTMPose | DeepLabCut | **WHAM** |
|---|:---:|:---:|:---:|:---:|
| 3D pose | ✅ metric-scale | 2D only¹ | 2D/3D² | ✅ SMPL mesh |
| Training required | ❌ zero-shot | ❌ zero-shot | ✅ per-project | ❌ zero-shot |
| Speed on M4 | ~15-30 FPS CPU | ~20-40 FPS | ~5-10 FPS | ~2-5 FPS CPU |
| ROM fidelity | Excellent | Good (2D) | Excellent | Excellent |
| Foot detail | Basic (ankle only) | ✅ WholeBody³ | Custom | ✅ mesh |
| Frontal plane | ✅ 3D | 2D | 2D | ✅ 3D |
| Camera calibration | ❌ not needed | ❌ | ✅ recommended | ❌ |
| Apple Silicon | ✅ native | ✅ | ✅ | ⚠️ slow (no CUDA) |
| Foot-ground contact | Estimated | ❌ | ❌ | ✅ predicted |

¹ RTMPose/WholeBody provides detailed 2D foot keypoints (heel + toes) – useful for calcaneal inversion/eversion  
² DeepLabCut requires training a project-specific model but achieves highest accuracy  
³ Enable via `pip install rtmlib onnxruntime` – see `requirements.txt`

**Recommendation:** MeTRAbs is the default for clinical gait ROM on M4. Use RTMPose/WholeBody backend for accurate rearfoot angle. Use WHAM (with GPU) for global trajectory + SMPL mesh + foot contact.

---

## 📷 Camera Setup Guide

### Sagittal view (side) – PRIMARY / REQUIRED
```
        Camera
          │
          │  distance: 3-5 m
          │
          ▼
    ◄── Subject ──►  (walking/running perpendicular to camera)
    
    ✅ Captures: hip/knee/ankle flexion-extension, trunk flexion,
                 shoulder/elbow flexion, speed, cadence, step length
```
- Camera height: ~1.0-1.3 m (hip height)
- Frame rate: ≥60 FPS recommended for running, ≥30 FPS for walking
- Capture ≥5-10 gait cycles in frame
- Treadmill or overground both work

### Frontal view
```
        Camera
          │
          │  distance: 3-5 m
          │
          ▼
       Subject  (walking/running toward camera)
    
    ✅ Captures: hip abduction/adduction, stance width,
                 trunk lateral lean, pelvic obliquity
```

### Rear view
```
       Subject  (walking/running away from camera)
          ▲
          │  distance: 3-5 m
          │
        Camera
    
    ✅ Captures: calcaneal inversion/eversion, hip abd/add,
                 stance width, foot strike pattern
    💡 Best view for rearfoot / pronation analysis
```

**Multi-view fusion:** Upload 2-3 videos simultaneously. Sagittal-plane angles use sagittal view, frontal-plane angles use frontal/rear view, spatiotemporal parameters are averaged.

---

## 📐 Joint Metrics Table

| # | Joint / Parameter | Plane | Required View | Normal Gait ROM | Units |
|---|---|---|---|---|---|
| 1 | Hip flexion/extension (L/R) | Sagittal | Sagittal | 40° flex / 10° ext (walk)<br>50-60° flex / 20° ext (run) | ° |
| 2 | Knee flexion/extension (L/R) | Sagittal | Sagittal | 0-70° (walk)<br>20-120° (run) | ° |
| 3 | Ankle dorsiflexion/plantarflexion (L/R) | Sagittal | Sagittal | 10° DF / 20° PF (walk)<br>25° DF / 30° PF (run) | ° |
| 4 | Trunk flexion/extension | Sagittal | Sagittal | 0-10° (walk)<br>10-15° (run) | ° |
| 5 | Neck flexion/extension | Sagittal | Sagittal | 0-10° | ° |
| 6 | Shoulder flexion/extension (L/R) | Sagittal | Sagittal | 20-45° (walk)<br>60-90° (run) | ° |
| 7 | Elbow flexion/extension (L/R) | Sagittal | Sagittal | 70-120° | ° |
| 8 | Hip abduction/adduction (L/R) | Frontal | Frontal/Rear | ±5-10° | ° |
| 9 | Calcaneal inversion/eversion (L/R) | Frontal | Rear | 2-5° inv FS<br>5-10° ever midstance | ° |
| 10 | Trunk lateral lean | Frontal | Frontal/Rear | <5° | ° |
| 11 | Pelvic obliquity / drop | Frontal | Frontal/Rear | 5-10° | ° |
| 12 | Foot progression angle (L/R) | Transverse | Any⁴ | 5-15° external | ° |

⁴ Foot progression requires WholeBody foot keypoints (RTMPose backend)

### Spatiotemporal Parameters

| Parameter | Normal Walking | Normal Running | Units |
|---|---|---|---|
| Speed | 1.2 – 1.4 | 2.5 – 4.5 | m/s |
| Cadence | 100 – 120 | 160 – 190 | steps/min |
| Step length | 0.6 – 0.8 | 0.8 – 1.4 | m |
| Stride length | 1.2 – 1.6 | 1.2 – 2.0 | m |
| Stride time | 1.0 – 1.2 | 0.6 – 0.8 | s |
| Stance time | ~60% GC | ~30-40% GC | % |
| Swing time | ~40% GC | ~60-70% GC | % |
| Step width | 0.08 – 0.15 | 0.05 – 0.10 | m |

GC = gait cycle

---

## 🚀 Install – MacBook M4

```bash
# 1. Create conda env (recommended)
conda create -n biovision python=3.11 -y
conda activate biovision

# 2. Install TensorFlow for Apple Silicon (optional, for MPS)
# pip install tensorflow-macos tensorflow-metal

# 3. Install BioVision gAIt
git clone https://github.com/slimbrady/biovision-gait.git
cd biovision-gait
pip install -r requirements.txt

# MeTRAbs auto-downloads model weights on first run (~50-100 MB)
```

**Linux / x86:**
```bash
pip install tensorflow  # instead of tensorflow-macos
# rest is identical
```

### Optional: Foot keypoint detail (calcaneal angle)

```bash
pip install rtmlib onnxruntime
# Apple Silicon: pip install onnxruntime-silicon
```

### Optional: WHAM (GPU recommended)

```bash
pip install torch torchvision
pip install git+https://github.com/yufu-wang/wham.git
# Download SMPL model from http://smpl.is.tue.mpg.de/
# Place in ./data/smpl/SMPL_NEUTRAL.pkl
```
See `wham_adapter.py` for full instructions.

---

## 🏃 Quick Start

### Streamlit UI (recommended)

```bash
streamlit run app.py
```

Then:
1. Select camera view mode (single / multi)
2. Upload 1-3 videos (tag with sagittal / frontal / rear)
3. Set subject height, confidence threshold, pose backend
4. Click **Run Gait Analysis**
5. View pose overlay, metrics dashboard, ROM table, time-series plots
6. Download angles CSV, summary JSON, overlay MP4

### CLI

```bash
# Pose estimation
python pose_metrabs.py run_side.mp4 --view sagittal --backend efficientnetv2_s

# → outputs: keypoints_run_side_sagittal.json, overlay_run_side_sagittal.mp4

# Gait metrics
python metrics_gait.py keypoints_run_side_sagittal.json --view sagittal --height 1.75

# → outputs: metrics_sagittal.csv, metrics_summary_sagittal.json
```

### Python API

```python
from pose_metrabs import run_metrabs_inference
from metrics_gait import analyze_gait

# 1. Pose estimation
pose = run_metrabs_inference("run.mp4", camera_view="sagittal")

# 2. Gait analysis
result = analyze_gait(
    pose["keypoints_3d"],
    pose["fps"],
    camera_view="sagittal",
    subject_height_m=1.75,
)

print(result["spatiotemporal"])
# {'speed_m_s': 3.2, 'cadence_steps_per_min': 172, ...}

print(result["rom_summary"]["knee_flexion_r"])
# {'rom': 95.3, 'min': 22.1, 'max': 117.4, 'mean': 58.2, 'std': 28.1}
```

---

## 🔬 WHAM Integration

**WHAM** = "World-grounded Human with Accurate Motion" – SMPL mesh-based 3D human motion capture.

| | MeTRAbs (default) | WHAM |
|---|---|---|
| Output | 17 joints, metric 3D | SMPL mesh (6890 vertices) + 24 joints + global trajectory |
| Foot-ground contact | Estimated | ✅ Predicted |
| Temporal smoothing | Post-filter | Built-in |
| Speed (M4) | ~15-30 FPS CPU | ~2-5 FPS CPU |
| GPU required | ❌ No | ✅ Recommended |
| Best for | Clinical gait ROM on M4 | Research / global trajectory / mesh |

**When to use WHAM:**
- You need world-grounded global trajectory (not camera-relative)
- You want SMPL body mesh output
- You need foot-ground contact prediction
- You have a CUDA GPU available
- Temporal smoothness for long sequences is critical

**Install:**
```bash
pip install git+https://github.com/yufu-wang/wham.git
# Download SMPL model from http://smpl.is.tue.mpg.de/
```

**Use:**
```python
from wham_adapter import run_wham_inference, convert_wham_to_gait_metrics

wham_result = run_wham_inference("run.mp4")
gait_input = convert_wham_to_gait_metrics(wham_result, fps=30, camera_view="sagittal")

from metrics_gait import analyze_gait
result = analyze_gait(gait_input["keypoints_3d_mm"], fps=30)
```

See `wham_adapter.py` for full documentation and SMPL → H36M joint mapping.

---

## 📁 File Map

```
biovision-gait/
├── pose_metrabs.py      # MeTRAbs 3D pose wrapper + RTMPose fallback
│                         # load_metrabs_model(), run_metrabs_inference()
├── metrics_gait.py      # Biomechanics calculator
│                         # compute_all_joint_angles(), compute_spatiotemporal_params(),
│                         # analyze_gait(), fuse_multiview_metrics()
├── app.py               # Streamlit UI
│                         # Upload → Analyze → Dashboard → Download
├── wham_adapter.py      # WHAM / SMPL integration (experimental)
│                         # run_wham_inference(), convert_wham_to_gait_metrics()
├── requirements.txt     # Core + optional [foot] + [wham] dependencies
└── README.md            # This file
```

---

## ⚠️ Accuracy Notes / Clinical Caveats

- **Ankle dorsiflexion/plantarflexion**: MeTRAbs H36M skeleton has no toe/foot keypoints – ankle angle is estimated from shank vector + assumed foot orientation. Use RTMPose/WholeBody backend for accurate foot angles.
- **Calcaneal inversion/eversion**: True rearfoot angle requires heel + toe keypoints (COCO-WholeBody). MeTRAbs-only output is an **estimate** from ankle mediolateral sway, clearly marked `_ESTIMATE` in all outputs.
- **Foot progression angle**: Requires foot keypoints – returns NaN with MeTRAbs alone, use RTMPose/WholeBody.
- **Gait event detection**: Foot strike / toe-off from ankle vertical kinematics – works best with sagittal view at ≥60 FPS. Treadmill videos give cleaner event detection than overground.
- **Speed / stride length**: MeTRAbs gives metric-scale 3D – accuracy depends on subject staying in frame, consistent camera distance, and sufficient gait cycles (≥5 recommended).
- **View-dependent metrics**: Hip abduction, trunk lean, pelvic obliquity, calcaneal angle, and step width **require frontal or rear camera view** – they return NaN with sagittal-only input, with clear messaging in the UI.
- **Clinical screening**: The abnormality flags in the Streamlit app are simple rule-based thresholds for screening only – **not a clinical diagnosis**. Consult a qualified physical therapist / biomechanist for clinical interpretation.

---

## 📚 References

- **MeTRAbs** – Sárándi et al. "MeTRAbs: Metric-Scale Truncation-Robust Estimation of 3D Human Body Poses". ECCV 2022. https://arxiv.org/abs/2207.08976
- **VisionMD-Gait** – Stenum et al. "Vision-based gait analysis: validation of a markerless approach". 2023.
- **Ali et al. 2024** – "Markerless motion capture for clinical gait analysis: a systematic review"
- **Washabaugh et al. 2022** – "Validity and repeatability of inertial measurement units for measuring running gait parameters"
- **WHAM** – Wang et al. "WHAM: Reconstructing World-Grounded Humans with Accurate 3D Motion". CVPR 2024. https://arxiv.org/abs/2312.07531

---

## 📄 License

MIT – see LICENSE file.

---

## 🙋 Citation

If you use BioVision gAIt in research, please cite MeTRAbs:

```bibtex
@inproceedings{sarandi2022metrabs,
  title={MeTRAbs: Metric-Scale Truncation-Robust Estimation of 3D Human Body Poses},
  author={S{\'a}r{\'a}ndi, Istv{\'a}n and Hermans, Alexander and Leibe, Bastian},
  booktitle={ECCV},
  year={2022}
}
```

---

*BioVision gAIt v0.1 · Built for Apple Silicon · github.com/slimbrady/biovision-gait*
