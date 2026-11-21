# GazeOS

**A modular, multi-agent eye-intelligence framework: core eye analytics, a synthetic digital twin, and an experimental assistive scene-description prototype.**

> [!WARNING]
> **GazeAssist is an experimental prototype, not a certified mobility aid.**
> It can miss obstacles, misjudge distances, speak late, or be wrong. It must
> never replace a white cane, guide dog, human assistance, or orientation and
> mobility (O&M) training. Do not use it near traffic, stairs, platforms,
> water, or any other hazardous environment. See [Safety](#safety).

GazeOS is three coordinated Python modules plus this README:

| File | Role |
|---|---|
| [`gaze_core.py`](#gaze_corepy--generic-eye-movement-intelligence) | Reusable, backend-agnostic eye-movement intelligence: landmarks → gaze/blinks/fixations/saccades → windowed features → baseline → anomalies → events. |
| [`eye_twin.py`](#eye_twinpy--synthetic-eye-digital-twin) | Seeded, controllable synthetic eye model, schematic renderer, labels/metadata, train/val/test datasets, domain randomization, optional Blender job export, and closed-loop validation through `gaze_core`. |
| [`gaze_assist.py`](#gaze_assistpy--experimental-assistive-scene-understanding) | Real-time scene-understanding prototype: pluggable object detection + depth, proximity/TTC risk, path score, spoken and haptic cues, graceful fallback, hardware-free demo. |
| `README.md` | This document. |

There are **no other repository files** required. Models, datasets, logs and
virtual environments are generated locally and are not part of the source
(see [Repository layout](#repository-layout)).

---

## Table of contents

1. [What this is (and is not)](#what-this-is-and-is-not)
2. [Capability status](#capability-status)
3. [Architecture](#architecture)
4. [Repository layout](#repository-layout)
5. [Requirements](#requirements)
6. [Installation](#installation)
7. [Model downloads](#model-downloads)
8. [Webcam setup](#webcam-setup)
9. [Quick start](#quick-start)
10. [GazeOS core](#gaze_corepy--generic-eye-movement-intelligence)
11. [Digital twin](#eye_twinpy--synthetic-eye-digital-twin)
12. [GazeAssist](#gaze_assistpy--experimental-assistive-scene-understanding)
13. [Configuration reference](#configuration-reference)
14. [Testing and quality](#testing-and-quality)
15. [Measured observations](#measured-observations)
16. [Troubleshooting](#troubleshooting)
17. [Limitations](#limitations)
18. [Privacy and ethics](#privacy-and-ethics)
19. [Safety](#safety)
20. [Roadmap](#roadmap)
21. [Extending GazeOS](#extending-gazeos)
22. [License and third-party components](#license-and-third-party-components)

---

## What this is (and is not)

### It is

* **A reusable analytics library.** `gaze_core` works on any backend that can
  produce five eye landmarks; MediaPipe Face Landmarker is the included real
  backend. It outputs structured, timestamped `GazeFrame` objects and
  edge-triggered events.
* **A digital twin.** `eye_twin` generates temporally coherent eye signals
  from explicit motor/appearance/lighting/pose parameters with reproducible
  seeds. It also produces schematic images, labels, metadata and splits, and
  can replay twin signals through the real analytics pipeline.
* **A legitimate real-time prototype stack.** `gaze_assist` runs capture and
  inference asynchronously, drops stale frames, rate-limits inference, ranks
  hazards by proximity and time-to-collision, and speaks short template
  messages. It is designed so that components are replaceable.
* **Local by default.** GazeOS itself makes no network calls. No video frames
  are sent to an LLM or to a GazeOS service. Optional logs are
  aggregate/structured data, and raw frame storage in `gaze_core` requires
  explicit consent flags. (Ultralytics may download weights and may send its
  own telemetry depending on its configuration; GazeOS does not rely on it.)

### It is not

* A medical device, diagnostic tool, certified driver-monitoring system, or
  safety-certified mobility aid.
* A calibrated eye tracker. Raw gaze is an eye-in-head geometric estimate in
  normalized units. Screen-point output requires a prototype calibration and
  is not guaranteed accurate.
* A photorealistic eye renderer. The twin renderer is schematic and
  deliberately labelled as such.
* An object-detection benchmark or a claim of detection accuracy. No accuracy,
  range, or latency figures are certified here.
* A replacement for a white cane, guide dog, human assistant, GPS navigation,
  SLAM, or O&M instruction.
* An LLM scene-reasoning system. Real-time frames intentionally never pass
  through a language model.

---

## Capability status

| Capability | Status | Needs |
|---|---|---|
| Core analytics + events + profiles | **Implemented** | `numpy` |
| Core demo and self-test | **Implemented** | `numpy` |
| Core webcam/video landmarks | **Implemented** | OpenCV, MediaPipe, `face_landmarker.task` |
| 9-point calibration GUI | **Implemented** | Camera, display, MediaPipe model |
| Headless CSV calibration | **Implemented** | `numpy` |
| Twin signal generation (seeded, labelled) | **Implemented** | `numpy` |
| Twin schematic PNG rendering | **Implemented** | `numpy` (PNG writer uses stdlib) |
| Twin dataset with splits + metadata | **Implemented** | `numpy` |
| Twin ↔ core closed-loop validation | **Implemented** | `numpy` |
| Blender photorealistic rendering | **Interface + export only** | Blender 3.6+, user-supplied rigged eye/face asset |
| Assist demo (mock scene) | **Implemented** | `numpy` |
| Assist self-test | **Implemented** | `numpy` |
| Assist live with Haar/motion fallback | **Implemented** | OpenCV |
| Assist live with YOLO semantics | **Implemented** | `ultralytics` + weights |
| Doors/stairs detection with COCO YOLO | **Not provided** | Custom detector/weights, or the demo |
| Metric depth | **Not provided** | Calibrated metric depth model (not bundled) |
| Relative depth (path score) | **Implemented** | User-supplied ONNX depth model |
| Speech output | **Implemented** | OS TTS (`say`, `espeak-ng`, `spd-say`, Windows SAPI) |
| Haptic output | **Interface implemented** | `pyserial` + compatible hardware; untested on specific devices |
| Optional eye analytics in assist | **Implemented, off by default** | MediaPipe model; research/operator use only |
| Human accuracy validation | **Not performed** | A study with consent, ground truth and O&M oversight |

---

## Architecture

GazeOS uses "multi-agent" in the software-architecture sense: three cooperating
modules with typed contracts and pluggable backends. They are ordinary Python
components, not LLM agents, and no video, landmarks or scene data are routed
through a language model.

```
                         gaze_core.py
 ┌──────────────────────────────────────────────────────────────────────────┐
 │  FrameSource (ABC)                 EyeLandmarkDetector (ABC)             │
 │   ├─ CameraSource (OpenCV)          └─ MediaPipeEyeLandmarkDetector      │
 │   ├─ VideoFileSource                (iris + lids + blendshape + pose)    │
 │   └─ custom / test sources                        │                      │
 │              │                                    ▼                      │
 │              └────────────► FaceObservation (typed contract)             │
 │                                   │                                      │
 │                                   ▼                                      │
 │                     GazeEstimator ──► GazeSample                         │
 │                     CalibrationModel ──► ScreenPoint (optional)          │
 │                                   │                                      │
 │        ┌──────────────────────────┼────────────────────────────┐         │
 │        ▼                          ▼                            ▼         │
 │  BlinkDetector            FixationDetector            TemporalFeature     │
 │  (durations, closure)     (I-DT, saccade speed)        Extractor           │
 │        └──────────────► GazePipeline ◄─────────────────────────┘         │
 │                            │  │  │                                       │
 │              BaselineTracker  AnomalyDetector  EventBus                  │
 │                            │                                             │
 │                            ▼                                             │
 │              GazeFrame → callbacks / JSONL / preview                    │
 └──────────────────────────────────────────────────────────────────────────┘
                             ▲
                             │ sample_to_face()
                   eye_twin.py  (digital twin)
 ┌──────────────────────────────────────────────────────────────────────────┐
 │ EyeTwinParams → EyeTwinGenerator → TwinSequence (samples + labels)       │
 │                        │                       │                         │
 │                        ▼                       ▼                         │
 │            ProceduralEyeRenderer      EyeDatasetBuilder                  │
 │            (schematic PNG frames)     (labels.csv, metadata, splits)     │
 │                        │                                                 │
 │                        └──► BlenderIntegration.export_job() [manual]     │
 └──────────────────────────────────────────────────────────────────────────┘

                    gaze_assist.py (scene understanding, not gaze navigation)
 ┌──────────────────────────────────────────────────────────────────────────┐
 │  CaptureWorker (newest-frame only)                                       │
 │       │                                                                  │
 │       ▼                                                                  │
 │  ObjectDetector (ABC)        DepthEstimator (ABC)                        │
 │   ├─ UltralyticsYOLO          ├─ GeometricDepthEstimator                 │
 │   ├─ Haar fallback            ├─ OnnxDepthEstimator (relative)           │
 │   ├─ Motion fallback          └─ NoDepthEstimator                        │
 │   └─ MockDetector                                                        │
 │       └──────────────┬───────────────┘                                  │
 │                      ▼                                                   │
 │           distance fusion → RiskEngine (proximity/TTC/direction)         │
 │                      │                                                   │
 │                      ▼                                                   │
 │           path_analysis → SceneReport                                     │
 │                      │                                                   │
 │                      ▼                                                   │
 │           AlertManager → SpeechOutput / ConsoleOutput / SerialHaptic     │
 │                                                                          │
 │  optional EyeAnalyticsMonitor (gaze_core; research only, NOT navigation) │
 └──────────────────────────────────────────────────────────────────────────┘
```

### Data contracts

All cross-module contracts are typed dataclasses:

| Contract | Module | Purpose |
|---|---|---|
| `FramePacket` | core | image + monotonic timestamp + frame index |
| `EyeGeometry` | core | five normalized landmarks per eye, derived openness/gaze ratios |
| `FaceObservation` | core | backend-agnostic face: eyes, confidence, head pose, blink blendshape |
| `GazeSample` | core | gaze X/Y (eye-space), confidence, head pose, raw values |
| `ScreenPoint` | core | calibrated normalized screen coordinates |
| `TemporalFeatures` | core | windowed blink/fixation/saccade/perclos/stability metrics |
| `Anomaly` | core | severity-ranked deviation with code/metric/baseline z |
| `GazeFrame` | core | complete per-frame output, JSON-serializable |
| `GazeEvent` | core | edge-triggered event with payload |
| `TwinSample` / `TwinSequence` | twin | synthetic per-frame state and sequence metadata |
| `DatasetManifest` | twin | dataset counts, splits, labels hash |
| `BBox`, `Detection` | assist | normalized geometry and detector-agnostic object |
| `DepthMap` | assist | relative inverse depth / optional metric depth |
| `Hazard`, `SceneReport` | assist | ranked risk and frame-level scene output |

### Event model

`gaze_core.EventBus` is synchronous and thread-safe. Subscribers can filter by
type and receive exceptions isolated from the pipeline.

| Event | Meaning |
|---|---|
| `SESSION_STARTED` / `SESSION_ENDED` | session lifecycle |
| `FACE_FOUND` / `FACE_LOST` | visibility transitions (loss after grace period) |
| `BLINK` | completed blink with duration |
| `PROLONGED_CLOSURE` | a long closure ended |
| `FIXATION` | completed fixation with duration (flushed at stream end) |
| `SACCADE` | completed saccade with peak approximate speed |
| `DROWSINESS` | prolonged closure / critical PERCLOS-like indicator crossed |
| `ANOMALY` | new anomaly code (edge-triggered) |
| `FRAME_ERROR` | per-frame detection failure |
| `DEGRADED` | reserved for application-level fallback reporting |

### Coordinate conventions

* Landmarks: MediaPipe normalized `(x, y)` in `[0, 1]`, `y` down.
* Gaze: normalized eye-space `[-1, 1]`, `gaze_x` toward image-right,
  `gaze_y` up. Not screen coordinates until calibrated.
* Screen points: normalized `[0, 1]`.
* Assist boxes: normalized `[0, 1]`, `y` down.
* Depth: relative inverse depth, larger = closer, unless a metric map is
  supplied by a custom backend.

---

## Repository layout

```
.
├── gaze_core.py      # core analytics (one file, documented classes + CLI)
├── eye_twin.py       # synthetic twin + renderer + dataset builder + CLI
├── gaze_assist.py    # assistive scene-understanding prototype + CLI
└── README.md
```

Everything else is **generated at runtime**:

```
models/face_landmarker.task      # downloaded MediaPipe model (optional)
yolo11n.pt                       # optional YOLO weights (auto-download)
artifacts/eye_preview/           # contact sheet from eye_twin.py preview
artifacts/eye_dataset/           # generated dataset (images, labels.csv, metadata/)
artifacts/blender_job/           # exported Blender job + starter script
calibration.json                 # fitted calibration (optional)
artifacts/*.jsonl                # opt-in structured logs
.venv/                           # local virtual environment
__pycache__/                     # Python bytecode
```

No extra required config, tests, or weights are shipped. The self-tests live
inside each module (`python <module>.py selftest`).

---

## Requirements

### Software

* Python **3.10–3.12** (validated here with 3.12.5).
* `numpy` — required.
* `opencv-python` — camera/video input, previews, Haar/motion fallbacks, ONNX depth.
* `mediapipe` — real eye landmarks (validated with 1.2.3).
* `ultralytics` — optional YOLO detector (pulls PyTorch).
* `pyserial` — optional serial haptics.
* OS text-to-speech — macOS `say`, Linux `espeak-ng`/`espeak`/`spd-say`,
  Windows PowerShell SAPI.

Validated environment: macOS arm64, Python 3.12.5, NumPy 2.5, OpenCV 5.1,
MediaPipe 1.2.3, current Ultralytics on CPU. Other versions may work but were
not exhaustively tested.

### Hardware

* CPU-only is sufficient for the demo, twin generation and self-tests.
* A webcam (or a video file) for live eye analytics and GazeAssist.
* Speakers/headphones for speech; optional serial actuator for haptics.
* No GPU is required. `--device cuda` is supported by the YOLO backend when
  available and falls back to CPU, but the repository does not guarantee a
  GPU speedup or support a particular GPU.
* Expected disk: ~10 MB for the face and YOLO-nano weights; installing
  `ultralytics` brings PyTorch and can require several GB. These are
  environment observations, not hard requirements.

---

## Installation

```bash
# 1. Get the repository
git clone <your-repository-url>
cd <repository-directory>

# 2. Virtual environment
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip

# 3. Required analytics / demo dependencies
pip install numpy

# 4. Live eye landmarks + camera/video/preview
pip install opencv-python mediapipe

# 5. Optional semantic object detector (large download: PyTorch)
pip install ultralytics

# 6. Optional serial haptics
pip install pyserial
```

There is intentionally no `requirements.txt`: the four deliverables are the
repository, and pinning versions during a hackathon tends to age badly. Use
the commands above or pin them in your own environment.

**Minimal / offline demo:** with only `numpy` installed, these work:

```bash
python gaze_core.py demo
python gaze_core.py selftest
python eye_twin.py preview --out artifacts/eye_preview
python eye_twin.py generate --out artifacts/eye_dataset \
  --sequences 6 --duration 3 --fps 10
python eye_twin.py validate
python gaze_assist.py demo --seconds 20
python gaze_assist.py selftest
```

No webcam, GPU, model weights, or network are needed for any command above.

---

## Model downloads

### 1. MediaPipe Face Landmarker (required for real eye landmarks)

```bash
mkdir -p models
curl -L -o models/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

Alternatively pass `--model /path/to/face_landmarker.task` or set
`GAZEOS_FACE_MODEL`. The model file is not bundled.

The MEDIAPIPE face landmarker provides 478 landmarks including iris points and
optional blendshapes. It runs on CPU in the default configuration.

### 2. YOLO object weights (optional, for semantic scene detection)

`gaze_assist.py --detector yolo --weights yolo11n.pt` asks Ultralytics to
download the nano COCO model on first use (network required). A local file can
be supplied instead. Check the [Ultralytics license](https://github.com/ultralytics/ultralytics)
(AGPL-3.0 by default) before redistribution or commercial use.

COCO classes mapped by GazeAssist: person, bicycle, motorcycle, car, bus,
truck, train, chair, bench, dog; potted plant, stop sign and traffic light are
mapped to the generic `obstacle` label. **COCO does not contain doors or
stairs.** The label mapping also accepts `door`/`doorway`, `stairs`/
`staircase`/`steps`, `curb`, `pole`, `traffic cone` and `wall`, so a **custom
YOLO model** trained on those classes can feed the same risk engine unchanged.
The mock demo exercises door/stairs labels; no such custom weights are bundled.

### 3. ONNX depth model (optional, relative depth only)

No depth weights are bundled and no accuracy is claimed. Export or obtain an
ONNX depth model (for example a MiDaS/Depth-Anything-style single-image model)
and pass:

```bash
python gaze_assist.py run --depth onnx --depth-model models/depth.onnx \
  --camera 0 --acknowledge-risks
```

The loader assumes an RGB `1×3×S×S` input with ImageNet mean/std and a
single-channel depth-like output. It normalizes output to relative inverse
depth (larger = closer). If your model instead outputs farther = larger, add
`--depth-invert`. Verify input size/preprocessing against your exported model;
`--depth geometric` is the safe default and needs no weights.

### 4. Blender (optional, manual photorealistic path)

Not automatic. See [Blender integration](#blender-integration).

---

## Webcam setup

The processing frame is **not mirrored**; mirroring is applied only to preview
windows. Camera coordinates therefore stay consistent with the saved frames.

### macOS

1. System Settings → Privacy & Security → Camera.
2. Enable access for Terminal, iTerm, your IDE, or whichever app launches
   Python.
3. Fully quit and reopen the terminal/IDE after changing permission.
4. Close apps that may be using the camera (Zoom, Teams, OBS).
5. Test:

```bash
python gaze_core.py run --camera 0 --model models/face_landmarker.task --display
```

### Linux

```bash
ls -l /dev/video*
groups
sudo usermod -aG video "$USER"    # log out/in after this
python - <<'PY'
import cv2
cap = cv2.VideoCapture(0)
print("opened:", cap.isOpened())
ok, frame = cap.read()
print("frame:", ok, None if frame is None else frame.shape)
cap.release()
PY
```

If `/dev/video0` is missing, check the kernel module/permissions or try another
index. For a virtual camera, indices can be higher (`--camera 2`, etc.).

### Windows

Settings → Privacy & Security → Camera → allow desktop apps. Close
applications that hold the camera. If index 0 fails, try `--camera 1`.

### Environment doctor

```bash
python gaze_assist.py doctor            # no permission prompt
python gaze_assist.py doctor --camera-check
```

`--camera-check` may trigger the OS camera permission dialog. It opens the
camera and reads one frame, then releases it.

---

## Quick start

### 1. Core demo (no hardware)

```bash
python gaze_core.py profiles
python gaze_core.py demo --seconds 20 --profile driver_monitoring --seed 3
python gaze_core.py selftest
```

`demo` runs a procedural synthetic face through the full pipeline, including
events, baseline, anomalies and a summary. It does not establish accuracy.

### 2. Digital twin

```bash
python eye_twin.py presets
python eye_twin.py preview --out artifacts/eye_preview --profile fatigued --seed 2
python eye_twin.py generate --out artifacts/eye_dataset \
  --sequences 6 --duration 3 --fps 10 --seed 42
python eye_twin.py validate --profile alert --seconds 30 --seed 7
python eye_twin.py selftest
```

Inspect `artifacts/eye_preview/contact_sheet.png` and
`artifacts/eye_dataset/labels.csv`. The images are schematic, not photos.

### 3. Assistive demo (no hardware)

```bash
python gaze_assist.py demo --seconds 30             # console + haptic log
python gaze_assist.py demo --seconds 30 --voice     # adds system speech
python gaze_assist.py demo --seconds 30 --display   # OpenCV overlay
python gaze_assist.py selftest
```

The demo uses a scripted walking scene. Timeline (loops every 30 s):

| Time | Event |
|---|---|
| 0–3 s | far ambient person only |
| 3–9 s | chair ahead, approaching from 5.0 m to 0.8 m |
| 9–14 s | car ahead, approaching from 10 m to 2 m |
| 14–19 s | person crossing from left to right at ~1.7 m |
| 19–24 s | stairs ahead at 1.6–1.1 m |
| 24–28 s | door on the left at 4.5 m |
| 28–30 s | clear center view |

Example (abridged, from a real run of this repository):

```
[INFO] No immediate obstacle detected ahead.
[WARN] Chair approaching ahead, about 3 seconds.
[STOP] Chair approaching ahead, 1.1 meters. Stop.
[WARN] Car approaching ahead, about 3 seconds.
[STOP] Car approaching ahead, 2.4 meters. Stop.
[STOP] Stairs very close ahead. Stop.
```

These messages come from a simulation, not a real walk.

### 4. Live run (experimental)

```bash
# Eye analytics only
python gaze_core.py run --camera 0 --profile research \
  --model models/face_landmarker.task --display

# Assistive prototype, console + speech
python gaze_assist.py run --camera 0 --detector auto --depth geometric \
  --voice --acknowledge-risks

# With YOLO semantics and an ONNX relative-depth map
python gaze_assist.py run --camera 0 --detector yolo --weights yolo11n.pt \
  --depth onnx --depth-model models/depth.onnx \
  --voice --jsonl artifacts/scenes.jsonl --acknowledge-risks
```

`--acknowledge-risks` is mandatory for live assistive mode. It is not a
certification; it only confirms that you have read the warning.

---

## `gaze_core.py` — generic eye-movement intelligence

### Pipeline responsibilities

1. Read frames from a `FrameSource` (`CameraSource`, `VideoFileSource`, or
   your own implementation).
2. Detect eye landmarks with a pluggable `EyeLandmarkDetector`.
3. Estimate normalized gaze from iris position/eyelid geometry and head pose.
4. Detect blinks (hysteresis + durations), fixations (dispersion + minimum
   duration) and saccades (angular speed).
5. Aggregate windowed features: blink rate/duration, fixation duration/ratio,
   saccade speed, PERCLOS-like closed ratio, gaze stability, on-target proxy.
6. Track a robust session baseline (median/MAD) and flag rule/z-score anomalies.
7. Emit a `GazeFrame` plus events and optionally calibrated `ScreenPoint`s.

### Application profiles

Profiles only change thresholds/rules; the analytics are the same.

```bash
python gaze_core.py profiles
```

| Profile | Window | PERCLOS warn/crit | Blink/min band | Notes |
|---|---:|---:|---:|---|
| `generic` | 30 s | 0.30 / 0.45 | 6–35 | balanced defaults |
| `driver_monitoring` | 15 s | 0.25 / 0.40 | 6–30 | shorter window, longer-closure emphasis, head-pose limits, camera-space attention zone; **not road truth** |
| `learning_analytics` | 45 s | 0.40 / 0.60 | 6–40 | descriptive fixation/blink metrics; avoids fatigue alarmism |
| `screen_fatigue` | 30 s | 0.30 / 0.45 | 8–30 | blink suppression and long-fixation emphasis |
| `research` | 60 s | 0.30 / 0.45 | 6–35 | more history, higher z threshold, fewer automatic conclusions |

Build a profile with overrides:

```python
from gaze_core import make_config

config = make_config(
    "driver_monitoring",
    perclos_warn=0.28,
    blink_open_threshold=0.30,
    window_s=20.0,
)
```

Unknown fields raise `ValueError`; invalid thresholds fail fast in
`__post_init__`.

### CLI examples

```bash
# List profiles
python gaze_core.py profiles

# Hardware-free pipeline demo
python gaze_core.py demo --profile learning_analytics --seconds 30 --seed 4

# Save per-frame aggregate JSONL (no images / no landmarks)
python gaze_core.py demo --seconds 20 --jsonl artifacts/gaze_demo.jsonl

# Real camera with preview and calibrated screen mapping
python gaze_core.py run --camera 0 --model models/face_landmarker.task \
  --profile screen_fatigue --calibration calibration.json --display

# Video file, headless, opt-in aggregate log
python gaze_core.py run --video session.mp4 --model models/face_landmarker.task \
  --jsonl artifacts/session.jsonl

# Interactive 9-point calibration (requires display + camera + model)
python gaze_core.py calibrate --camera 0 --model models/face_landmarker.task \
  --output calibration.json

# Headless calibration from a CSV of raw/target pairs
python gaze_core.py calibrate --points calibration_points.csv --output calibration.json
```

#### Calibration

The GUI workflow shows nine targets (3×3). Look at each target and press
**SPACE**; press **Q**/Esc to abort. It averages the most recent 12 reliable
frames per target and fits a ridge model from raw gaze features to normalized
screen coordinates. It prints an RMSE for the *fit points*. That number is not
a validated point-of-regard accuracy and should not be quoted as one.

Headless CSV format:

```csv
gaze_x,gaze_y,target_x,target_y,yaw_deg,pitch_deg
-0.60,-0.50,0.20,0.20,0.0,0.0
...
```

`target_x/target_y` are normalized screen coordinates. At least 6 points are
needed; 9–16 are recommended.

### Python API

```python
from gaze_core import (
    CameraSource,
    EventType,
    GazeSession,
    MediaPipeEyeLandmarkDetector,
    make_config,
)

config = make_config("research", window_s=45.0)
source = CameraSource(camera_index=0, width=640, height=480, fps=30)
detector = MediaPipeEyeLandmarkDetector(
    model_path="models/face_landmarker.task",
    num_faces=1,
)

session = GazeSession(source, config=config, detector=detector)
session.bus.subscribe(
    lambda event: print(event.type.value, event.message),
    [EventType.BLINK, EventType.ANOMALY, EventType.FACE_LOST],
)

summary = session.run(max_seconds=30, display=True)
print(summary.format_text())
```

For your own capture loop or transport, feed observations directly:

```python
from gaze_core import GazePipeline, make_config

pipeline = GazePipeline(make_config("driver_monitoring"))
# face is a FaceObservation from any backend, or None if no face was found
report = pipeline.process(face, timestamp=12.34)
print(report.sample.to_dict() if report.sample else "no gaze")
print(report.features.to_dict())
print([a.code for a in report.anomalies])
```

Useful APIs:

| API | Purpose |
|---|---|
| `make_config(profile, **overrides)` | validated config from a named profile |
| `GazeEstimator(config).estimate(face)` | raw gaze + heuristic confidence |
| `BlinkDetector(config).update(t, openness)` | blink state/duration/closure |
| `FixationDetector(config).update(t, sample, valid)` | fixation/saccade dynamics |
| `TemporalFeatureExtractor(config)` | windowed features |
| `BaselineTracker(config)` | per-session median/MAD baseline |
| `AnomalyDetector(config).detect(...)` | rule + z-score anomalies |
| `GazeCalibrator().add(...).fit()` | ridge calibration |
| `GazeSession.run(...)` | source + detector + pipeline orchestration |
| `render_overlay(...)` | OpenCV debug preview (optional) |

### Structured output

`GazeFrame.to_dict()` is JSON-safe and contains:

* frame index, timestamp, face-present flag, processing time, source;
* gaze (x/y, confidence, head pose) and optional calibrated screen point;
* windowed features (see `TemporalFeatures`);
* anomalies with code/severity/metric/value/baseline z;
* events emitted on that frame;
* baseline statistics once ready.

Aggregate JSONL logging never writes images or landmarks:

```json
{"record":"header","schema":"gazeos.frame.v1","privacy":"aggregate gaze features only; no images or landmarks","metadata":{...}}
{"frame":42,"t":1.4,"face":true,"sample":{"gx":0.12,"gy":-0.03,"conf":0.91,...},"features":{"perclos":0.04,"blink_rate":14.2,...},"anomalies":[],"events":[...]}
```

### What the numbers mean

* `perclos` here is the **fraction of recent face-present frames with eyes
  below the closed threshold**, not the standard 80%-eyelid PERCLOS metric.
  It is a prototype proxy, not a drowsiness diagnosis.
* Saccade speed uses normalized gaze × `degrees_per_norm_unit`; it is
  approximate and not eye-radius calibrated.
* `blink_open_threshold` is camera/user dependent. When MediaPipe blendshapes
  are available the blink blendshape is the primary openness signal; the
  geometric lid-gap fallback is used for backends without blendshapes.
* Baseline z-scores compare a user to their **own session**, not to a
  population.

### Privacy behaviour in core

* Frames stay in memory.
* The JSONL logger stores features/events only, never images or landmarks.
* Raw frame dumping in the CLI is refused unless **both**
  `--save-frames DIR` and `--store-identifiable-frames` are supplied. That
  second flag is an explicit consent gate; frames are personal data.
* Custom integrations can use `on_packet` to store frames, but they own the
  consent and retention obligations.

---

## `eye_twin.py` — synthetic eye digital twin

### What is simulated

The twin is a phenomenological state machine. It produces:

* fixation targets, drift, microsaccades, saccades (simplified main sequence),
  fixation durations;
* blinks with configurable rate, probability, duration and waveform;
* pupil size linked to lighting;
* slowly varying head yaw/pitch/roll;
* lighting intensity/color/azimuth and glare;
* appearance (skin/iris/sclera colors, iris/pupil size, lid opening, lashes);
* camera pose, distance, jitter and sensor noise;
* occlusion episodes;
* per-frame labels and sequence metadata.

The renderer draws an eye with ellipse/iris/pupil masks, shading, limbal ring,
specular highlight, lash band, blink closure, camera warp and occlusion. It is
**schematic**, not physically based: no corneal refraction, tear film,
volumetric tissue, real textures, lens distortion or photorealism. Every
metadata file records `simulated: true` and `photorealistic: false`.

### Presets

```bash
python eye_twin.py presets
```

| Preset | Blink/min | Blink mean | Fixation mean | Lid opening | Character |
|---|---:|---:|---:|---:|---|
| `alert` | 15 | 180 ms | 380 ms | 1.00 | baseline |
| `fatigued` | 28 | 300 ms | 550 ms | 0.78 | exaggerated demo pattern; **not a fatigue diagnosis** |
| `reading` | 20 | 180 ms | 260 ms | 0.92 | smaller saccades, tighter range |
| `driving` | 17 | 180 ms | 500 ms | 1.00 | wider gaze range, more head motion |
| `scanning` | 13 | 180 ms | 180 ms | 1.00 | large, frequent saccades |

All presets are adjustable `EyeTwinParams`; presets are not clinical or
population models.

### Reproducibility

```python
from eye_twin import EyeTwinGenerator, preset

params = preset("reading")
seq_a = EyeTwinGenerator(params, seed=123).sequence(duration_s=10, fps=30)
seq_b = EyeTwinGenerator(params, seed=123).sequence(duration_s=10, fps=30)
assert [s.gaze_x for s in seq_a.samples] == [s.gaze_x for s in seq_b.samples]

# Hold an explicit gaze angle while blinks/head pose/lighting still vary:
fixed = EyeTwinGenerator(params, seed=4).sequence(
    duration_s=2, fps=30, fixed_gaze_deg=(5.0, -3.0)
)
assert all(s.gaze_x_deg == 5.0 and s.gaze_y_deg == -3.0 for s in fixed.samples)
```

Same parameters + seed + fps + duration produce identical samples and label
content. Dataset split assignment uses a seeded permutation, so the same dataset
seed reproduces the split. Metadata `created_at` timestamps differ between runs.

### Generate a dataset

```bash
python eye_twin.py generate \
  --out artifacts/eye_dataset \
  --sequences 8 \
  --duration 6 \
  --fps 15 \
  --seed 42 \
  --profiles alert reading fatigued driving scanning \
  --splits 0.7,0.15,0.15 \
  --frames-per-second 10
```

`--sequences` must be at least 3 so all splits are populated. Use
`--no-images` for labels only, `--no-domain-randomization` for fixed
parameters, and `--image-height/--image-width` to change resolution.

Output layout:

```
artifacts/eye_dataset/
├── images/<split>/<sequence_id>/frame_000000.png
├── labels.csv
├── metadata/sequences.json
├── metadata/<sequence_id>.json
└── manifest.json
```

`labels.csv` columns:

`sequence_id, split, profile, seed, frame, filename, t, gaze_x, gaze_y,
gaze_x_deg, gaze_y_deg, state, blink, is_blink, pupil_mm, fixation_id,
saccade_id, blink_id, gaze_speed_deg_s, head_yaw_deg, head_pitch_deg,
head_roll_deg, lighting, occlusion, is_occluded`

`blink` is a continuous eyelid-closure amount in `[0, 1]`; `is_blink` uses
`blink > 0.5`. If you need a different ground-truth threshold, use the
continuous column (the analytics' default closed threshold corresponds to a
higher closure amount; see `blink_open_threshold`).

`manifest.json` records sequence/frame/image counts, split/profile counts,
renderer name, image size, labels SHA-256, `simulated: true` and
`photorealistic: false`.

Domain randomization varies skin/iris colors, iris/pupil size, lid opening,
lash strength, lighting intensity/azimuth/glare, camera yaw/pitch/roll/
distance/jitter, fixation durations/noise, saccade amplitude/velocity, blink
rate/duration, head motion, sensor noise and occlusion probability. These
ranges are defined in `_randomize_params` and are not calibrated to a
population.

### Preview

```bash
python eye_twin.py preview --out artifacts/eye_preview --profile alert --seed 2 --frames 5
```

This writes `contact_sheet.png`, individual frames and
`preview_labels.json`. Open the contact sheet to confirm exactly what the
renderer produces; do not assume photorealism.

### Closed-loop validation

```bash
python eye_twin.py validate --profile alert --seconds 30 --seed 7
```

This replays twin samples through the **real** `gaze_core` pipeline and compares
recovered gaze, blinks and fixations with twin labels. Example output (one
validation run):

```json
{
  "gaze": {
    "correlation_x": 0.999,
    "correlation_y": 0.9998,
    "rmse_normalized_raw": {"x": 0.129, "y": 0.0316},
    "rmse_normalized_after_affine_fit": {"x": 0.0154, "y": 0.0096}
  },
  "blinks": {"ground_truth_count": 9, "detected_count": 9,
             "ground_truth_mean_ms": 100.0, "detected_mean_ms": 100.0},
  "fixations": {"ground_truth_count": 61, "detected_count": 61}
}
```

**Interpretation:** this is a *closed-loop synthetic consistency check*. It
shows that the pipeline, geometry bridge, event thresholds and labels agree on
known signals. It is not human accuracy, not a degree/pixel error, and not a
claim that a model trained on these images will transfer to real eyes. The
report includes these caveats at runtime.

### Python API

```python
from eye_twin import EyeTwinGenerator, preset, sample_to_face
from gaze_core import GazePipeline, make_config

params = preset("fatigued")
params.seed = 7
sequence = EyeTwinGenerator(params, seed=7).sequence(duration_s=20, fps=30)
print(sequence.metadata())

pipeline = GazePipeline(make_config("screen_fatigue"))
for sample in sequence.samples:
    face = sample_to_face(sample, params)
    report = pipeline.process(face, timestamp=sample.t)
```

Or stream observations directly:

```python
from eye_twin import EyeTwinObservationSource

for face in EyeTwinObservationSource(profile="reading", seed=3, duration_s=10):
    ...
```

### Blender integration

`eye_twin.py blender-job` exports:

* `job.json` — per-frame gaze/blink/head-pose/lighting/occlusion states;
* `render_eye_sequence.py` — a starter bpy script with an `apply_state()` stub;
* `README_blender.txt` — setup notes.

```bash
python eye_twin.py blender-job --out artifacts/blender_job \
  --profile driving --seed 3 --seconds 4 \
  --template /path/to/your_rigged_eye.blend
```

It does **not** render and does not ship a 3D model. To produce plausible
photoreal images you need Blender 3.6+ (4.x recommended), a rigged eye/face
asset, materials, and adaptation of the script to your object/bone/shape-key
names. Then run manually:

```bash
blender -b your_rigged_eye.blend -P artifacts/blender_job/render_eye_sequence.py -- artifacts/blender_job/job.json
```

Only a real 3D engine with a suitable asset can approach physical realism.

---

## `gaze_assist.py` — experimental assistive scene understanding

### Design

* **Capture thread** continuously reads frames and keeps only the newest.
  Slow inference therefore drops stale frames instead of queuing them.
* **Detector rate** (`--detector-hz`) and **depth rate** (`--depth-hz`) are
  configurable and independent.
* **Risk engine** fuses distance, closing speed (short IoU-associated track
  history), time-to-collision, lateral zone and label weight.
* **Outputs** are short templates over OS TTS plus optional serial haptics.
* **Graceful degradation:** detector failure tries Haar → motion; depth failure
  falls back to geometric (`--depth`); eye analytics failures disable only the
  optional analytics. Capture errors end the session with a measured summary.
* **No gaze navigation.** Optional `--eye-analytics` only describes fatigue
  proxies for sighted researchers/operators; it never feeds obstacle alerts.

### Backends

#### Object detectors

| Backend | CLI | Labels | Notes |
|---|---|---|---|
| Ultralytics YOLO | `--detector yolo` | COCO hazard classes above | Learned detector; needs `ultralytics` + weights; CPU default |
| Haar fallback | `--detector haar` | person (faces/full bodies) | Ships with OpenCV; heuristic confidence 0.55; no vehicles/doors/stairs |
| Motion fallback | `--detector motion` | `obstacle` | Static-camera frame differencing; no semantics; false positives with camera motion |
| Scripted mock | demo only | person, chair, car, stairs, door | No pixels required; deterministic scenario |
| Custom | pass an `ObjectDetector` to `AssistPipeline` | any labels in the risk weights / defaults | Implement `load()` and `detect()` |

`--detector auto` chooses YOLO when `ultralytics` is importable, else Haar,
else motion. If a detector fails to load, the pipeline attempts the fallback
chain and records the degradation.

#### Depth / distance

| Backend | CLI | Output | Limits |
|---|---|---|---|
| Geometric | `--depth geometric` (default) | metric estimate per detection | assumes known object heights (`ASSUMED_HEIGHTS_M`) and flat ground; wrong for partial/odd objects |
| ONNX | `--depth onnx --depth-model ...` | relative inverse depth map | no metric distance unless your model supports it; used for path score |
| None | `--depth none` | no depth | detections need distances or size-based relative mode |

Geometric distance: `assumed_height × focal_px / box_height_px`, with focal
from vertical FOV 55° by default. Doors use an assumed 2.0 m height. Stairs
have no height model, so no distance is invented for them; the risk engine uses
a conservative size cue.

#### Haptics

`SerialHapticOutput` writes newline-delimited ASCII `direction,priority\n` at
115200 baud by default (e.g. `ahead,1`). A microcontroller can map priority 1
to a strong pulse, 2 medium, 3 light. The protocol and class are provided; no
specific actuator has been validated. Do not treat vibration as a safety
guarantee.

### Risk and path logic

* Direction zones: left `< 0.38`, ahead `0.38–0.62`, right `> 0.62` of frame
  width.
* Priority 1 (critical): distance ≤ 0.9 m, TTC ≤ 1.6 s, or stairs ≤ 1.6 m.
* Priority 2 (warning): distance ≤ 1.8 m or TTC ≤ 3.0 s.
* Priority 3 (caution): risk score ≥ 0.22.
* With no metric distance, a large low central detection can produce a
  warning without any distance claim.
* TTC assumes constant closing speed derived from a short track history and is
  easily wrong when the camera or object changes speed.
* `walkable_score` (0–1) combines center hazards and (when available) relative
  depth in the bottom-center region. `path_hint` describes the camera view; it
  is **not** a safe route instruction and never says "safe to cross".

### Alert wording

Templates are short and deterministic, e.g.:

* `"Stairs very close ahead. Stop."`
* `"Car approaching ahead, 2.4 meters. Stop."`
* `"Person, 1.7 meters on your left."`
* `"Possible obstacle ahead."`
* `"No immediate obstacle detected ahead."` (only after a hazard-free interval)

Critical messages bypass the general throttle but still have a minimum repeat
interval (2.5 s default). Warnings repeat at most every 2 s generally and 5 s
for the same (label, direction). The speech queue holds at most four items and
drops lower-priority stale ones. TTS duration is unknown in advance, so actual
audio timing is not guaranteed.

### CLI

```bash
python gaze_assist.py doctor [--camera-check]

python gaze_assist.py demo --seconds 30 [--voice] [--display] [--quiet] [--no-clear]

python gaze_assist.py run \
  [--camera 0 | --video path] \
  [--detector auto|yolo|haar|motion] [--weights path] [--device cpu] \
  [--conf 0.35] [--imgsz 416] \
  [--depth geometric|onnx|none] [--depth-model path] [--depth-invert] \
  [--detector-hz 6] [--depth-hz 2] [--seconds N] \
  [--voice] [--haptic-port /dev/ttyUSB0] [--display] \
  [--eye-analytics] [--eye-profile screen_fatigue] [--eye-model path] \
  [--jsonl artifacts/scenes.jsonl] [--no-clear] [--acknowledge-risks]

python gaze_assist.py eye-demo --profile fatigued --seconds 20 --seed 3
```

Measured rates and latencies are printed in the final summary. There is no
hidden "expected" performance number.

### Python API and custom backends

```python
from gaze_assist import (
    AssistConfig,
    AssistPipeline,
    BBox,
    ConsoleOutput,
    Detection,
    GeometricDepthEstimator,
    MockDetector,
    MockSceneSource,
    ObjectDetector,
)


class MyDetector(ObjectDetector):
    name = "my-model"

    def load(self) -> None:
        ...  # load weights; raise a clear error if unavailable

    def detect(self, frame, timestamp):
        # Return normalized boxes and optional metric distance.
        return [
            Detection(
                "door",
                0.8,
                BBox(0.05, 0.2, 0.3, 0.9),
                distance_m=None,
                source=self.name,
            )
        ]


pipeline = AssistPipeline(
    MockSceneSource(fps=30),
    AssistConfig(detector="mock", detector_hz=8, display=False),
    ConsoleOutput(),
    detector=MyDetector(),
    depth=GeometricDepthEstimator(),
)
stats = pipeline.run(max_seconds=5)
print(stats.format_text())
```

A custom depth backend subclasses `DepthEstimator` and implements `estimate()`
and/or `distance_for()`. A custom output channel subclasses `AssistOutput`
and implements `announce()` (plus optional `haptic()`).

### Optional eye analytics

`--eye-analytics` runs MediaPipe + `gaze_core` on the same frames at a reduced
rate and can announce fatigue-proxy anomalies. It is:

* **off by default**;
* for sighted operators/researchers only;
* never used to steer or choose actions for a blind user;
* dependent on the face model and a usable webcam view of the operator.

`eye-demo` shows the same analytics path on synthetic `eye_twin` data, with no
camera or model file.

---

## Configuration reference

### Core (`GazeConfig`)

Key fields and defaults:

| Field | Default | Meaning |
|---|---:|---|
| `max_faces` | 1 | faces requested from detector |
| `min_face_confidence` | 0.30 | minimum confidence for valid gaze |
| `detect_every_n_frames` | 1 | optional detection subsampling |
| `gaze_gain_x/y` | 1.35 / 1.15 | eye geometry → normalized gaze |
| `head_yaw/pitch_compensation` | 0.010 / 0.008 | heuristic head term per degree |
| `gaze_smoothing` | 0.65 | EMA alpha; saccade jump bypasses |
| `blink_open_threshold` | 0.24 | closed below this openness |
| `blink_close_hysteresis` | 0.04 | reopen above threshold + hysteresis |
| `blink_min_ms` / `blink_max_ms` | 40 / 800 | valid blink band |
| `prolonged_closure_ms` | 1200 | prolonged closure threshold |
| `saccade_threshold_deg_s` | 90 | saccade speed threshold |
| `fixation_dispersion_deg` | 2.5 | I-DT dispersion threshold |
| `fixation_min_ms` | 100 | minimum fixation duration |
| `degrees_per_norm_unit` | 30 | approximate angular scale |
| `window_s` | 30 | feature window |
| `baseline_warmup_s` | 5 | baseline warm-up |
| `baseline_min_samples` | 60 | per-metric baseline samples |
| `anomaly_z` | 3.5 | robust z threshold |
| `perclos_warn/critical` | 0.30 / 0.45 | profile-specific |
| `blink_rate_low/high` | 6 / 35 | profile-specific |
| `attention_zone` | `None` | camera-space attention proxy |
| `mirror_preview` | `True` | preview only; processing stays unmirrored |

Use `make_config(profile, **overrides)` or a JSON config via
`load_config(path, profile)` in Python. CLI supports profile + common overrides
(`--detect-every`); more advanced overrides are best done from Python.

### Assist (`AssistConfig`, `RiskConfig`)

Important defaults:

| Setting | Default | Notes |
|---|---:|---|
| detector | `auto` | YOLO → Haar → motion |
| detector rate | 6 Hz | independent of capture rate |
| depth rate | 2 Hz | 0 disables map updates |
| capture | 640×480 @ 30 fps | clips to the newest frame |
| confidence | 0.35 | pipeline filter |
| caution / warning / critical | 3.0 / 1.8 / 0.9 m | prototype thresholds |
| TTC warning / critical | 3.0 / 1.6 s | constant-speed assumption |
| lateral zones | 0.38 / 0.62 | frame-width fractions |
| alert general interval | 2.0 s | throttle |
| critical interval | 2.5 s | repeat limit |
| haptic interval | 1.0 s | repeat limit |

These are **not** validated safety margins. Tune for your environment, test
with supervision, and never treat a setting as certified.

---

## Testing and quality

### Self-tests (no webcam, GPU or weights)

```bash
python gaze_core.py selftest      # 14/14 checks
python eye_twin.py selftest       # 19/19 checks
python gaze_assist.py selftest    # 15/15 checks
```

They cover geometry signs, blink/fixation detection, calibration mapping,
JSONL logging, seed reproducibility, dataset splits/metadata/reproducibility,
PNG writing, core↔twin gaze signs, risk priority/TTC, path scoring, alert
throttling, mock scenario coverage, async pipeline stats and twin integration.

### Syntax / static checks

```bash
python -m py_compile gaze_core.py eye_twin.py gaze_assist.py

# If you use ruff
ruff check gaze_core.py eye_twin.py gaze_assist.py --select F,E9
```

### End-to-end checks used during development

* Downloaded the official MediaPipe `face_landmarker.task`, ran it on a
  public-domain portrait and on a short video generated from that portrait,
  and confirmed landmark extraction, gaze/confidence output and a JSONL
  session.
* Ran the Ultralytics YOLO backend on CPU with `yolo11n.pt`, confirming model
  load, COCO class mapping and inference.
* Built twin datasets and verified labels, metadata, splits, images and
  reproducibility.
* Ran the mock assist pipeline and the scripted demo end to end.

These checks validate wiring and basic behaviour, not accuracy or safety.

---

## Measured observations

Measured during validation on the development machine (**macOS arm64, Python
3.12.5, CPU-only, OpenCV 5.1, MediaPipe 1.2.3**). They are observations, not
guarantees, benchmarks, or safety claims. Measure on your own hardware.

| Observation | Value |
|---|---|
| YOLO11n @ 416, first prediction after load (includes warm-up) | ~1.55 s |
| YOLO11n @ 416, warm CPU predictions | ~21–26 ms per frame |
| MediaPipe session on a 640×799 portrait video | 40 frames processed over ~1.4 s with 30 fps video pacing (not a standalone latency benchmark) |
| Mock GazeAssist demo at `--detector-hz 6` | ~5–8 processed scene reports/s, ~30 fps capture (frame dropping active) |
| Core demo | 600 synthetic frames processed faster than real time on CPU |

No detection accuracy, distance accuracy, or human gaze accuracy is reported
because none has been validated.

---

## Troubleshooting

### Camera will not open

* Check OS permission (macOS/Windows) and `/dev/video*` + `video` group (Linux).
* Close other apps using the camera; try `--camera 1`.
* A camera indicator LED staying off usually means permission or index.
* `python gaze_assist.py doctor --camera-check` for a one-shot probe.

### `ModelNotFoundError: MediaPipe face model not found`

Download the model (see [Model downloads](#model-downloads)) or pass
`--model`. For a no-model run, use the core demo or `eye-demo`.

### `DependencyMissingError: mediapipe` / `opencv-python`

```bash
pip install mediapipe opencv-python
```

If MediaPipe wheels are unavailable for your Python version, use Python
3.10–3.12 or a supported platform.

### YOLO errors

* Install `ultralytics`; it downloads PyTorch the first time.
* If a local weights path is supplied, it must exist.
* A plain filename such as `yolo11n.pt` may be downloaded from the network.
* CUDA is optional; the backend falls back to CPU when unavailable.
* COCO YOLO will never output doors/stairs. Use a custom model/backend or the
  mock demo.

### Low processing rate

* Lower `--detector-hz`, use `--imgsz 320`, or use `yolo11n`/a smaller model.
* Use `--depth geometric` or `--depth none`; ONNX depth can be expensive.
* Frame dropping is intentional: capture may run at 30 fps while scene
  reports run at the detector rate.
* Close other applications; check thermal throttling.

### False or missing alerts

* Motion fallback triggers on camera movement — use a static mount or a
  learned detector.
* Geometric distance is wrong for partial objects, unusual camera pitch, and
  unknown classes; there is no metric depth without a model.
* Windows, mirrors and reflective floors can confuse any detector.
* Tune confidence/thresholds, but re-test with supervision. Lower thresholds
  mean more false positives, not more safety.

### No speech

```bash
python gaze_assist.py doctor
```

Install `espeak-ng` on Linux, or use macOS `say` / Windows SAPI. Console output
always works. Speech playback is asynchronous; long utterances can delay later
messages, and the queue intentionally drops stale low-priority items.

### Haptics do not fire

`--haptic-port` requires `pyserial` and a serial device. The protocol is
`direction,priority\n`. Verify permissions (`dialout` group on Linux) and the
baud rate. Hardware support is not validated.

### Blinks/fixations look wrong on a real camera

Eye openness and gaze geometry vary by person, camera and glasses. Tune
`blink_open_threshold`, gains, smoothing, fixation dispersion and confidence
thresholds on your own consented recordings. Use the debug preview. Do not
treat defaults as universal.

### Display issues on a headless machine

Use `--no-display`/omit `--display`; everything else runs headless. Calibration
GUI requires an actual display. Use CSV calibration otherwise.

---

## Limitations

### Gaze core

* Geometric gaze is a heuristic, not a calibrated eye tracker. Screen accuracy
  is unverified; head pose, glasses, makeup, lighting and camera angle all
  bias it.
* Blink thresholds are user/camera dependent. Blendshape availability changes
  behaviour. Blink durations have roughly a one-frame quantization error.
* Saccade speeds use an assumed degrees-per-normalized-unit scale; they are
  not measured angular velocities.
* Fixation detection uses a dispersion window and misses very short or
  drifting fixations; PERCLOS is a prototype proxy, not the standard metric.
* Baseline and anomaly detection describe deviations from this session, not
  diagnoses. Rules can misfire during calibration, blinks, or head motion.
* MediaPipe per-face detection scores are not surfaced by the Tasks API here;
  confidence is a geometric heuristic.
* No identity recognition, gaze-to-object mapping, reading analysis, or
  medical inference is implemented.
* Processing is single-stream and synchronous; high frame rates may require
  your own threading/queue around `GazePipeline`.

### Digital twin

* The renderer is schematic. No cornea, tear film, subsurface scattering, real
  iris texture, lens distortion, rolling shutter, or physically based lighting.
* The motor model is phenomenological. Values are plausible defaults, not
  fitted to measured human distributions.
* Dataset images will not make a model robust to real eyes by themselves.
  Closed-loop validation is circular by design: the same synthetic bridge is
  used for labels and analytics. It tests plumbing/consistency, not biology.
* Dataset metadata timestamps differ across builds; labels/splits are
  reproducible for the same seed. Old files are not deleted when rebuilding
  into the same directory.
* Blender export is a job specification plus a starter script; it requires a
  user-supplied rig and manual adaptation.
* `frames_per_second_saved` changes which frames appear in labels; the label
  `frame` is the original generated-frame index.

### GazeAssist

* **Not a certified or validated mobility aid.** No human-subject safety or
  efficacy study has been performed. Detection range, false-negative rate,
  distance error, alert latency and usability are unmeasured here.
* COCO YOLO detects only mapped classes; **doors and stairs are not detected
  from the camera** without a custom model/backend. Stairs/door labels exist
  for the API, risk engine and mock demo.
* The Haar/motion fallbacks are weak: people-only or generic motion blobs,
  heuristic confidence, no semantics. They are not substitutes for a learned
  detector.
* Geometric distance assumes object height and flat ground; a crouching
  person, partial view, child, or unusual camera pitch can be badly wrong.
  ONNX depth is relative unless the model is metric, and only influences the
  path score.
* `walkable_score`/`path_hint` are coarse descriptions of the camera view, not
  a traversability map. There is no SLAM, localization, map, GPS, IMU fusion,
  or route planning.
* TTC assumes constant closing speed and IoU-based association is not a
  production tracker.
* OS TTS latency and queueing are outside the vision loop's control. Speech
  can arrive after the situation changes. Haptic hardware is not validated.
* The optional eye analytics are research-only and require a sighted operator
  in view; blind users are not expected to control anything with gaze.
* No audio spatialization, bone conduction, obstacle-specific sonification, or
  mobile app is implemented.

### Cross-cutting

* No accuracy, range, latency or safety certification.
* No multi-camera fusion, active sensing, or sensor calibration.
* Python + OpenCV CPU stack is a prototype foundation, not a hardened
  embedded runtime.

---

## Privacy and ethics

### Data flow

* All GazeOS processing is **local**. GazeOS makes no cloud/API calls and
  sends no video to an LLM. The optional Ultralytics package may download
  weights and may have its own telemetry/update checks; review its settings and
  licenses. No frames are sent by GazeOS itself.
* `gaze_core` never stores frames implicitly. JSONL contains aggregate
  features/events only.
* Raw frame storage in the core CLI requires explicit consent flags
  (`--save-frames` **and** `--store-identifiable-frames`). Frames are
  identifiable personal data; obtain informed consent and follow retention
  rules.
* `gaze_assist` logs scene reports only (`--jsonl`), not images or audio.
  Detections and locations can still be sensitive; protect the files.
* `eye_twin` generates synthetic data. Do not label it as real or as
  de-identified human data.

### Eye analytics

Gaze, blink and fixation data are behavioural and potentially sensitive. They
can reveal attention, fatigue proxies, and cognitive load. Do not use GazeOS
for covert monitoring, employee surveillance, medical diagnosis, or any
high-stakes decision. The driver/learning/fatigue profiles are **descriptive
prototypes**, not validated assessments.

### Assistive use and consent

* Eye analytics in `gaze_assist` are off by default and must never be used to
  judge a blind or low-vision person's attention or capability.
* Test the assistive prototype only with informed participants, in a safe,
  closed area, with a sighted spotter and O&M oversight where appropriate.
* Do not record bystanders without consent. Prefer synthetic or consented
  data.

### Dataset licensing

The MediaPipe model and OpenCV are third-party components with their own
licenses. Ultralytics YOLO is AGPL-3.0 by default; check compatibility before
distributing derived work. Do not redistribute downloaded weights or
third-party assets unless their licenses allow it.

---

## Safety

> [!CAUTION]
> **This software is experimental and must not be relied on for navigation,
> obstacle avoidance, or personal safety.**

* Never use GazeAssist as the sole means of detecting obstacles.
* Always carry and use a white cane or guide dog if that is your established
  mobility method; seek human assistance in unfamiliar or hazardous places.
* Do not use it while crossing roads, on platforms, near stairs, water, fire,
  machinery, or in any situation where a missed detection could cause harm.
* Speech and haptics can be delayed, masked, misunderstood, or unavailable.
  Check headphones, battery and volume before any test.
* The system has no fail-safe certification, redundancy, watchdogs, or
  guaranteed alerting. A crash or exhausted battery means no alerts.
* Do not use it for driving or operating machinery. The driver profile is a
  research demo, not a safety system.
* Test with a sighted spotter. Get O&M training and user feedback before
  drawing any usability conclusions.
* If you are a researcher, obtain ethics approval and informed consent, and
  provide a clear stop procedure.

---

## Roadmap

1. **Real validation.** Human-factors and accuracy studies with consented
   participants, ground-truth annotations, O&M experts and pre-registered
   metrics. Report false negatives, distance error, alert latency and
   usability honestly.
2. **Metric depth and sensor fusion.** Calibrated stereo/ToF/IMU, camera
   intrinsics, ground-plane estimation, robust tracking and latency budgets.
3. **Better semantics.** Custom door/stairs/curb/tactile-paving models, open-
   vocabulary detection with on-device distillation, and scene segmentation
   for walkable paths.
4. **Embedded runtime.** ONNX/TFLite/Core ML export, quantized models,
   watchdogs, audio prioritization, mobile/edge deployment.
5. **Gaze improvements.** Per-user calibration, head-pose coupling,
   uncertainty estimates, more rigorous blink/Fixation algorithms and
   validation against eye trackers.
6. **Twin improvements.** Physically based Blender/Mitsuba asset, real iris
   textures with proper licensing, measured distributions, sim-to-real
   benchmarks, and clearly separated label versions.
7. **Accessibility research.** Co-design with blind and low-vision users,
   spatial audio, haptic language studies, and a proper safety case.

---

## Extending GazeOS

* **New eye backend:** implement `EyeLandmarkDetector.load/detect` and return
  `FaceObservation` objects. Nothing else changes.
* **New frame source:** implement `FrameSource.open/read/close`; return
  `FramePacket`.
* **New app profile:** add thresholds in `_PROFILE_OVERRIDES` and a name in
  `PROFILE_NAMES`.
* **New event consumer:** `EventBus.subscribe(handler, [EventType...])`.
* **New detector/depth backend:** subclass `ObjectDetector` or
  `DepthEstimator`; return normalized `Detection`s / `DepthMap`.
* **New output channel:** subclass `AssistOutput`; keep `announce` and
  `haptic` non-blocking or cheap.
* **New twin preset:** add a branch in `preset()` and a name in `PRESETS`.
* **Custom Blender rig:** adapt `apply_state()` in the exported starter script.

Keep the typed contracts intact; that is what makes the three modules
composable.

---

## License and third-party components

No license file is included in this four-file deliverable, so no license is
granted by default. Add an explicit license before distributing or accepting
contributions.

Third-party components are not vendored:

* MediaPipe — Apache-2.0.
* OpenCV — Apache-2.0.
* Ultralytics YOLO — AGPL-3.0 (or a commercial license).
* Blender — GPL; exported job scripts are yours to adapt.
* Any model weights or 3D assets you supply remain under their own licenses.

You are responsible for complying with those licenses and for the ethics and
safety of any deployment.

---

**GazeOS — build carefully, measure honestly, and never let a demo become a
safety claim.**
