# 👁️ GazeOS
 
### Eyes are the most information-rich signal your body gives off. We built the open toolkit to read them, simulate them, and point a camera at the world for people who can't see it.
 
**Three Python modules, plus an Android XR glasses skeleton. Zero cloud. No camera needed to try it.**
 
*Built for the Samsung PRISM hackathon.*
 
> [!WARNING]
> **GazeAssist is an experimental prototype, not a certified mobility aid.** It can miss obstacles, misjudge distances, and speak late. It must never replace a white cane, guide dog, human assistance, or O&M training. Never use it near traffic, stairs, platforms, or water.
 
🕶️ **Glasses demo (simulated):** https://claude.ai/artifact/6PhJrtMLHWvT7RAHSiJEPc &nbsp;|&nbsp; 🎥 **Demo video:** `<link>` &nbsp;|&nbsp; 🖼️ **Slides:** `<link>` &nbsp;|&nbsp; 👥 **Team:** `<names>`
 
---
 
## 💡 The problem
 
- **Eye data is powerful but locked away.** Fatigue, attention, and reading behavior all show up in blinks, fixations, and saccades. Most tools are closed, expensive, or tied to specialized hardware.
- **Eye datasets are hard to get.** Real recordings mean consent forms, privacy risk, and weeks of labeling.
- **Assistive scene description is hard to prototype.** Researchers need a clean, swappable pipeline (detect, estimate distance, rank risk, speak) that runs on a laptop.
## 🚀 What we built
 
| Module | What it does |
|---|---|
| 🧠 **`gaze_core.py`** | Any webcam or video becomes structured eye intelligence: gaze, blinks, fixations, saccades, fatigue-style anomalies, and live events. Works with any backend that supplies 5 landmarks per eye. |
| 🧬 **`eye_twin.py`** | A **synthetic eye digital twin**. Seeded, controllable, and fully labeled. It generates signals, schematic images, and train/val/test datasets with no real people involved. |
| 🦯 **`gaze_assist.py`** | A real-time prototype that detects objects, estimates distance, computes time-to-collision, and speaks short alerts like *"Car approaching ahead, 2.4 meters. Stop."* |
| 🕶️ **`GazeAssistGlasses.kt`** | An Android XR integration skeleton for Samsung's audio glasses: the risk engine ported to Kotlin, speech output, and a glasses-camera binding. **Untested on emulator or hardware.** |
 
> "Multi-agent" here means cooperating software components with typed contracts. They are **not** LLM agents. No video ever touches a language model.
 
## ✨ Why it's different
 
- 🔒 **Private by design.** Everything runs locally. Logs hold aggregate features only, never images. Saving raw frames takes two explicit consent flags.
- 🔌 **Fully pluggable.** Swap the eye backend, object detector, depth model, or output channel (console, TTS, serial haptics) without touching the rest.
- 🧪 **Closed-loop testing.** The twin feeds known ground-truth signals through the real `gaze_core` pipeline. In our run it recovered 9/9 blinks and 61/61 fixations, with gaze correlation above 0.99.
- ⚡ **Built for real time.** Capture and inference run asynchronously and stale frames are dropped, so slow inference never builds up lag.
- 🪂 **Graceful degradation.** If YOLO is missing it falls back to Haar, then to motion detection. A single failed component never takes down the whole app.
- 🧾 **Radically honest.** Every limitation is documented. See [What this is not](#-what-this-is-not).
## ⏱️ Try it in 60 seconds
 
Only `numpy` is required. No webcam, GPU, model weights, or internet.
 
```bash
git clone <repo-url> && cd <repo>
python3 -m venv .venv && source .venv/bin/activate
pip install numpy
 
python gaze_core.py demo --seconds 20          # full eye-analytics pipeline
python eye_twin.py preview --out artifacts/eye_preview   # see the synthetic eyes
python eye_twin.py validate                    # twin → core closed-loop check
python gaze_assist.py demo --seconds 30        # scripted walking scene with alerts
```
 
Run `python <module>.py selftest` on any module to verify it. All three pass with no extra hardware.
 
### Demo output (simulated scene)
 
```
[WARN] Chair approaching ahead, about 3 seconds.
[STOP] Chair approaching ahead, 1.1 meters. Stop.
[STOP] Car approaching ahead, 2.4 meters. Stop.
[STOP] Stairs very close ahead. Stop.
```
 
*These come from a scripted simulation, not a real walk.*
 
## 🎬 Go live (optional)
 
```bash
pip install opencv-python mediapipe
mkdir -p models && curl -L -o models/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
 
# Live eye analytics with preview
python gaze_core.py run --camera 0 --model models/face_landmarker.task --display
 
# Assistive prototype (console + speech)
pip install ultralytics   # optional: YOLO semantics (large download)
python gaze_assist.py run --camera 0 --detector auto --voice --acknowledge-risks
```
 
## 🕶️ Samsung Android XR glasses
 
GazeAssist is a natural fit for Samsung's Android XR "intelligent eyewear": an outward camera in, short spoken alerts out. We built the path to get there and are clear about how far along it is.
 
### How the platform works
 
Per the Android XR developer docs, the first glasses are audio-only and run no standalone apps. An ordinary Android phone app projects an activity to the glasses through **Jetpack Projected**, and reaches the glasses' outward camera and audio through a projected context. The glasses camera supports one active stream, so we bind only an image-analysis use case. The SDK is still a developer preview, so APIs may change.
 
### What fits and what doesn't
 
| Module | On the glasses? | Why |
|---|---|---|
| `gaze_assist.py` | **Yes, via a port** | Outward camera plus audio matches its detect → risk → speak loop. |
| `gaze_core.py` | **No, not as designed** | Eye analytics need a camera facing the eyes. The documented glasses camera points outward. |
| `eye_twin.py` | **Dev tooling only** | Synthetic data and test signals. |
 
### What's in the repo
 
| File | Purpose |
|---|---|
| `GazeAssistGlasses.kt` | Kotlin port of the risk engine (same thresholds and wording), speech throttling, a detector interface, and a CameraX analysis binding using the newest-frame-only strategy. |
| Glasses demo page | A browser simulation of the glasses-camera walk with live risk alerts and optional voice. Linked at the top of this README. |
 
### Verification status (read this)
 
| Item | Status |
|---|---|
| Risk logic in the browser demo | **Runs**, on a scripted scene |
| Kotlin risk engine | **Written, not compiled or unit-tested** |
| Projected-context camera binding | **Skeleton with TODOs**; API names need checking against Developer Preview 4 |
| Android XR emulator | **Not yet run** |
| Real Samsung glasses | **Not tested** |
| Detection accuracy, frame rate, link latency | **Unmeasured** |
 
**Next milestone:** compile the Kotlin, bind it to the emulator's camera, and record the result.
 
## 🏗️ How it works
 
```
 Camera/Video ─► Eye landmarks ─► Gaze + Blink + Fixation ─► Features ─► Baseline ─► Anomalies ─► Events
                                         ▲
   eye_twin (synthetic eyes + labels) ───┘   (closed-loop validation)
 
 Camera ─► Newest frame ─► Object detector + Depth ─► Risk engine (distance, TTC, direction)
                                                         └─► Short spoken alert / haptic pulse
```
 
**Risk levels in GazeAssist**
 
| Priority | Trigger |
|---|---|
| 🔴 Critical | ≤ 0.9 m, or TTC ≤ 1.6 s, or stairs ≤ 1.6 m |
| 🟠 Warning | ≤ 1.8 m, or TTC ≤ 3.0 s |
| 🟡 Caution | Risk score ≥ 0.22 |
 
**Application profiles** (same analytics, different thresholds): `generic`, `driver_monitoring`, `learning_analytics`, `screen_fatigue`, `research`.
 
## 🎯 Use cases
 
- 📚 **Learning analytics:** descriptive attention and reading metrics
- 💻 **Screen fatigue:** blink suppression and long-fixation tracking
- 🚗 **Driver-monitoring research:** fatigue-proxy prototyping (research only)
- 🤖 **ML data generation:** labeled synthetic eye data with reproducible seeds
- 🦯 **Assistive-tech research:** a modular testbed for scene-description ideas
<!-- ## 🧭 What this is not
 
We would rather be trusted than impressive.
 
- ❌ Not a medical device, diagnostic tool, or certified driver-monitoring system
- ❌ Not a calibrated eye tracker. Gaze is a geometric estimate until you calibrate it.
- ❌ Not a photorealistic eye renderer. Twin images are schematic.
- ❌ Not validated for accuracy. No detection, distance, or human gaze accuracy claims are made.
- ❌ Not yet verified on Samsung glasses. The glasses integration is an untested skeleton plus a browser simulation.
- ❌ Not a Gemini replacement. It does one narrow thing: deterministic, local, auditable hazard alerts.
- ❌ Standard COCO YOLO **cannot detect doors or stairs**. Those appear only in the mock demo unless you bring a custom model.
- ❌ The twin's validation is **circular by design**. It proves the plumbing is consistent, not that it works on real eyes. -->
## 🔐 Privacy and safety
 
- Local processing only. GazeOS makes no network calls.
- Do not use for covert monitoring, employee surveillance, or diagnosis.
- Never judge a blind or low-vision person's attention or capability with gaze data.
- A worn camera captures bystanders. The glasses path analyzes frames in memory only and stores no images. Keep it that way, and get consent before any recording.
- Test assistive features only with informed participants, in a safe area, with a sighted spotter.
## 🛣️ What's next
 
1. **Glasses bring-up:** compile the Kotlin port, run it on the Android XR emulator, then on real glasses, and publish measured frame rate and latency
2. **Real validation** with consented participants, ground truth, and O&M experts
3. **Metric depth and sensor fusion** (stereo/ToF/IMU)
4. **Custom door, stairs, and curb detectors**
5. **Embedded runtime** (ONNX/TFLite, quantization, watchdogs)
6. **Photoreal twin** via a Blender asset pipeline
7. **Co-design with blind and low-vision users**, including spatial audio and haptics
## 🧩 Built with
 
Python · NumPy · OpenCV · MediaPipe Face Landmarker · Ultralytics YOLO (optional) · ONNX depth (optional) · Kotlin · Android XR SDK (Jetpack Projected) · CameraX
 
<!-- ## 📜 License
 
No license is included yet. **Add one before distributing** (e.g. MIT or Apache-2.0). Note that Ultralytics YOLO is AGPL-3.0, so check compatibility for derived work. -->
 
---
 
**GazeOS: build carefully, measure honestly, and never let a demo become a safety claim.**
 