# 👁️ GazeOS — Assistive Gaze and Indoor Guidance Research Toolkit

> **GazeOS is the umbrella research repository. Its flagship implementation is GazeAssist: a multi-agent, memory-enabled indoor guidance prototype that interprets a camera view, assesses risk, remembers useful household information, and delivers short spoken alerts.**
>
> The repository also contains a separate artificial eye movement project for building a textured 3D face rig and generating controlled eye-movement renders with ground-truth pupil coordinates.

**Built for the Samsung PRISM hackathon.**

[Simulated glasses demo](https://claude.ai/artifact/6PhJrtMLHWvT7RAHSiJEPc) · **Demo video:** `<add link>` · **Slides:** `<add link>` · **Team:** `<add names>`

---

## Repository structure

The repository is organized into two independent project folders. Each folder has its own implementation, setup requirements, and deliverables.

```text
GazeOS/
│
├── GazeAssist_Multi_Agent_Guidance/
│   └── Main project: multi-agent indoor guidance system
│
├── Final_EyeMovemnt_AgentDemo/
│   └── Artificial eye rig, Python controls, and render-generation project
│
└── README.md
```

| Project folder | Role | Main purpose |
|---|---|---|
| **`GazeAssist_Multi_Agent_Guidance/`** | **Flagship / main project** | Multi-agent scene understanding, risk assessment, household memory, and spoken indoor guidance |
| `Final_EyeMovemnt_AgentDemo/` | Supporting research project | Textured artificial face rig, deterministic eye articulation, and synthetic eye-movement image/metadata generation |

> **Start here:** The primary application is in `GazeAssist_Multi_Agent_Guidance/` & `Final_EyeMovemnt_AgentDemo/`. The artificial eye movement project is a separate component of the broader GazeOS research toolkit; it is not a dependency of the GazeAssist multi-agent runtime.

---

# 🌟 1. Flagship project — GazeAssist Multi-Agent Guidance

GazeAssist is the main application in this repository. It extends the original **detect → risk → short spoken alert** workflow into a **LangGraph-based system with four cooperating agents and two memory layers**.

The system is designed around a deliberate separation of responsibilities:

- **Gemini Vision interprets the scene.** It receives a small, downscaled camera frame and returns structured detections, approximate left/right positions, distance estimates, room context, and visible personal items.
- **Deterministic Python logic assesses risk.** The Path and Risk agents do not use an LLM. They apply explicit rules to available-space information, distance, time-to-collision, and stair-related conditions.
- **A deterministic Output agent produces concise alerts.** It uses fixed wording, throttles repeated messages, and allows higher-priority escalations through.
- **Memory adds personal context.** A recent buffer remembers where items were last seen and which rooms were visited; permanent memory stores learned room-to-room directions and usual item locations.

Only the Obstacle agent uses an LLM. A language model is useful for visual interpretation, but it should not be responsible for making a time-critical decision such as whether the user should stop. The vision call is also the slowest stage, so detection runs at approximately **1 Hz by default**.

## 1.1 Architecture at a glance

```text
                    User query / command
                             |
                     Is this a memory query?
                       /             \
                     Yes              No
                      |                |
                 Memory read      Camera frame
                      |                |
                      |          Obstacle Agent (1)
                      |          Gemini Vision
                      |                |
                      +--------+-------+
                               |
              +----------------+----------------+
              |                |                |
         Path Agent (2)   Risk Agent (3)   Memory operations
          Rule-based       Rule-based      Recent + permanent
              |                |                |
              +----------------+----------------+
                               |
                        Output Agent (4)
                    Fixed wording + throttling
                               |
                         ElevenLabs TTS
                               |
                      Spoken guidance output
```

The Path agent, Risk agent, and relevant memory operations are designed to run in parallel after visual detection. Memory writes run in a background thread so that they do not intentionally delay an alert.

## 1.2 The four agents

| Agent | Responsibility | Uses an LLM? |
|---|---|---|
| **1. Obstacle** | Sends a small downscaled frame to Gemini and receives typed detections: object label, left/right position, approximate distance, room, and visible personal items. | **Yes — Gemini Vision** |
| **2. Path** | Estimates free space in left, ahead, and right zones; produces a coarse directional hint; retrieves a stored room-to-room route when `--destination` is supplied. | No |
| **3. Risk** | Assigns priority levels using distance, time-to-collision, and stair-related rules. It follows the GazeAssist risk thresholds. | No |
| **4. Output** | Builds short fixed-wording alerts, throttles repeated messages, allows escalations through, and sends speech to ElevenLabs. | No |

## 1.3 Two-layer memory

GazeAssist uses two bounded memory systems so that short-lived observations and stable household knowledge are handled differently.

| Memory | What it stores | Location | Limits |
|---|---|---|---|
| **Recent buffer** | Recently visited rooms and temporary item-location observations, e.g. “keys, on the kitchen counter, 10 min ago”. | RAM, snapshotted to Firebase every 5 minutes and on exit | 30 items, 20 rooms, 6-hour expiry |
| **Permanent memory** | Room-to-room directions and usual locations for personal items. | Firebase Firestore, with local JSON fallback | Text only |

### Memory router

1. **Rules first:** clear statements such as “I put my keys on the counter” are stored in the recent buffer. Explicit route instructions are stored in permanent memory.
2. **Gemini for ambiguous notes:** when rules cannot classify a note, Gemini performs structured classification. If no API key is configured, unclear notes are safely discarded.
3. **Promotion:** an item observed in the same place on three separate occasions, or a room-to-room move observed twice, can be promoted from recent to permanent memory.
4. **Reads:** “Where are my keys?” checks recent observations first and then usual locations. “How do I get to the kitchen?” searches permanent route memory from the current room.

## 1.4 Privacy and data handling

> **Important:** This multi-agent version changes the original GazeOS zero-cloud promise. Analysed camera frames are sent to Google's Gemini API. Text sent for speech generation goes to ElevenLabs. Firebase stores household knowledge as text, not camera images.

The implementation is designed to limit what it retains and transmits:

- Only the newest camera frame is kept; there is no accumulating frame queue.
- Frames are downscaled to 320 px and JPEG-compressed before visual analysis.
- The image is removed from graph state after detection.
- No LangGraph checkpointer is used, so graph history does not accumulate.
- Recent and permanent memories are bounded.
- The audio phrase cache is limited to 24 clips.
- Firebase writes are limited to promotions, explicitly taught notes, and the periodic recent-buffer snapshot.

A wearable camera may capture bystanders and private areas of a home. Obtain consent, avoid recording people who have not agreed, review Google's, ElevenLabs' and Firebase's current data terms, and protect credentials. Keep `.env` and `firebase-key.json` out of version control.

## 1.5 Multi-agent setup

Open a terminal in the multi-agent project directory:

```bash
cd GazeAssist_Multi_Agent_Guidance

python3 -m venv .venv
```

Activate the environment:

**Linux / macOS**
```bash
source .venv/bin/activate
```

**Windows PowerShell**
```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies and configure credentials:

```bash
pip install -r requirements.txt
cp .env.example .env
```

On Windows, copy the example file with:

```powershell
Copy-Item .env.example .env
```

Add the required API keys and configuration to `.env`, then run:

```bash
python -m gazeassist_agents doctor
```

Audio playback requires an MP3 player such as `ffplay` (FFmpeg), `mpg123`, or macOS `afplay`. If ElevenLabs or a player is unavailable, the application falls back to an operating-system voice such as `say` or `espeak-ng`. Messages are also printed to the console.

### Firebase configuration

1. Create a Firebase project and a Cloud Firestore database.
2. Create/download a service-account key.
3. Save it as `firebase-key.json` in the multi-agent project directory.
4. Set `FIREBASE_CREDENTIALS` in `.env`.
5. Set `FIREBASE_USER_ID` to the intended user identifier.

Data is stored below:

```text
users/{FIREBASE_USER_ID}/
    edges
    habits
    meta
```

Firebase is used for permanent household knowledge and recent-buffer snapshots. If Firebase credentials are unavailable, the system can use local JSON fallback where supported.

## 1.6 Run the multi-agent system

Run these commands from `GazeAssist_Multi_Agent_Guidance/`:

```bash
# Offline checks; no API keys required
python -m gazeassist_agents selftest

# Scripted walk; no camera or API keys required
python -m gazeassist_agents run --source mock --fast --seconds 40

# Live camera with Gemini and ElevenLabs
python -m gazeassist_agents run --source camera --interactive

# Live camera and a stored destination route
python -m gazeassist_agents run --source camera --destination kitchen
```

With `--interactive`, type notes and questions while the system runs:

```text
I put my keys on the kitchen counter
From the bedroom to the kitchen: go straight 6 steps, then turn left
where are my keys?
```

The first statement is a temporary item-location note; the route instruction is intended for permanent memory; the final question retrieves the remembered item location.

## 1.7 Current limitations

- Gemini distance estimates are approximate, rough, and not validated. The prompt asks the model to estimate closer when uncertain, but this does not make the estimates safe or reliable.
- Network and vision-call latency dominate response time. Detection runs at about 1 Hz by default.
- If vision fails twice consecutively, the system announces that obstacle detection is unavailable and tells the user to use a cane or ask for help.
- Object tracking matches by label and zone; it is not a full multi-object tracker.
- Room recognition can be wrong. A room must be detected twice consecutively before it is accepted.
- Live Gemini, ElevenLabs, and Firebase integrations have not yet been verified in the build environment. Offline tests cover logic, but live integrations still need testing with valid credentials.
- Voice input is not implemented. Questions are currently typed.
- Audio plays on the computer running Python. For Samsung glasses, the MP3 bytes need to be forwarded through a phone application by supplying an `on_audio` callback to `AudioOut`.

---

# 🧬 2. Artificial Eye Movement — 3D Face Rig

**Folder:** `Final_EyeMovemnt_AgentDemo/`

This is a separate artificial eye movement project within GazeOS. It uses Blender and Python to work with a textured, articulatable 3D face rig. Its goal is to provide deterministic control of eye direction and generate repeatable renders with ground-truth pupil-location metadata.

This project is intended for synthetic eye imagery, eye-movement experimentation, and eye-tracking dataset generation. It is **not a runtime dependency of the GazeAssist multi-agent guidance application**.

## 2.1 Objectives and specifications

- Textured, articulatable face rig based on the supplied MakeHuman / GNM / FLAME assets.
- Left and right eye bones for eye articulation.
- Python control interface designed around `eyes.move(h, v)` and `eyes.reset()`.
- Fixation target 35 cm forward.
- Locked camera at 40 cm.
- 1024 × 1024 PNG output.
- Ten clockwise angular positions, with \(\theta_k = 2\pi k/10\).
- Per-frame JSON metadata containing the 2D pupil centers.
- Contact sheet or a 1–2 minute demonstration video.

| Parameter | Specification |
|---|---|
| Face rig | Textured, articulatable 3D face |
| Rig foundation | MakeHuman / GNM / FLAME |
| Eye articulation | Left and right eye bones |
| Fixation distance | 35 cm forward |
| Camera distance | Locked at 40 cm |
| Render resolution | 1024 × 1024 |
| Image format | PNG |
| Movement sequence | 10 clockwise positions |
| Angular increment | 36° |
| Metadata | Per-frame JSON with 2D pupil centers |

## 2.2 Project files

```text
Final_EyeMovemnt_AgentDemo/
│
├── .gitkeep
├── base.obj
├── build_face_rig.py
├── default_weights.mhw
├── default.mhskel
├── eyes_wrapper.py
├── FaceRig.blend
├── FaceRig.blend1
├── make_skin_texture.py
└── skin_diffuse.jpg
```

| File | Purpose |
|---|---|
| `base.obj` | Base 3D facial geometry. |
| `build_face_rig.py` | Script associated with constructing/configuring the face rig. |
| `default_weights.mhw` | MakeHuman weight configuration. |
| `default.mhskel` | Skeleton definition. |
| `eyes_wrapper.py` | Python interface intended for controlling eye movement. |
| `FaceRig.blend` | Main Blender scene containing the face model and rig. |
| `FaceRig.blend1` | Blender backup file. |
| `make_skin_texture.py` | Script associated with preparing/generating the skin texture. |
| `skin_diffuse.jpg` | Diffuse skin texture. |

## 2.3 Setup

Prerequisites:

- Python 3.x.
- Blender with Python scripting support.
- MakeHuman, if required to regenerate the source face model.

Open Blender and load:

```text
FaceRig.blend
```

Check that the scene contains the face mesh, eye geometry, left/right eye bones, armature relationships, skin material, texture, and camera.

If the skin texture is missing, open Blender's Shading workspace, locate the skin material, verify its image texture node, and relink `skin_diffuse.jpg`.

Scripts that import Blender's `bpy` module must be run using Blender's bundled Python environment or from Blender's Scripting workspace. A regular system Python interpreter does not automatically provide `bpy`.

## 2.4 Eye control interface

The intended wrapper interface is:

```python
eyes.move(h, v)
eyes.reset()
```

- `h` represents horizontal eye movement.
- `v` represents vertical eye movement.
- `eyes.move(h, v)` applies the requested eye orientation.
- `eyes.reset()` returns the eyes to their default orientation.

The actual numerical ranges, coordinate conventions, initialization procedure, and supported control behavior must be confirmed in `eyes_wrapper.py`; they are not inferred here.

## 2.5 Ten-position clockwise sequence

The specified sequence is:

\[
\theta_k = \frac{2\pi k}{10}, \qquad k=0,1,\ldots,9
\]

The angular increment is:

\[
\Delta\theta = \frac{360^\circ}{10}=36^\circ
\]

The camera should remain locked throughout the sequence so that changes between frames come from the intended eye movement rather than camera motion.

## 2.6 Expected output structure

The following is the intended deliverable layout; create these directories/files as part of the rendering and metadata workflow:

```text
output/
│
├── renders/
│   ├── frame_00.png
│   ├── frame_01.png
│   ├── ...
│   └── frame_09.png
│
├── metadata/
│   ├── frame_00.json
│   ├── frame_01.json
│   ├── ...
│   └── frame_09.json
│
└── contact_sheet.png
```

Example metadata shape (the coordinates below are placeholders, not measured ground truth):

```json
{
  "frame_id": 0,
  "angle_degrees": 0,
  "movement_direction": "clockwise",
  "image_resolution": [1024, 1024],
  "pupil_centers": {
    "left": {"x": 0, "y": 0},
    "right": {"x": 0, "y": 0}
  }
}
```

The pupil coordinates must be calculated from the rendered scene or a defined projection procedure. Do not use the placeholder values as actual labels.

## 2.7 Deliverables and evaluation

Expected deliverables:

- Complete `.blend` scene with textured mesh, eye bones, and materials.
- Python wrapper and setup README.
- Ten clockwise PNG renders.
- Per-frame JSON metadata with 2D pupil centers.
- Contact sheet or 1–2 minute demo video.

The supplied evaluation breakdown is:

| Evaluation component | Weight |
|---|---:|
| Execution and Efficiency | 40% |
| Calibration | 20% |
| Automation | 20% |
| Quality and Movement | 60% |
| Direction pass/fail | 40% |
| Sequence and metadata | 20% |

---

# 🧩 3. Broader GazeOS toolkit and related components

The repository's broader research direction includes eye analytics, synthetic data generation, and assistive scene understanding. The following components are described in the project documentation; check the actual files present in the relevant project folder before running them.

| Module / component | Purpose |
|---|---|
| `gaze_core.py` | Converts webcam or video input into structured eye analytics: gaze estimates, blinks, fixations, saccades, fatigue-style anomalies, and live events. It works with a backend that supplies five landmarks per eye. |
| `eye_twin.py` | A seeded, controllable synthetic eye digital twin. It generates signals, schematic images, and train/validation/test datasets without recording real people. |
| `gaze_assist.py` | The original local assistive prototype: detects objects, estimates distance, computes time-to-collision, and speaks short alerts. It is separate from the newer LangGraph/Gemini multi-agent implementation. |
| `GazeAssistGlasses.kt` | Android XR integration skeleton: Kotlin risk-engine port, speech output, detector interface, and a CameraX analysis binding. It is not verified on an emulator or physical glasses. |

**Important distinction:** The original GazeOS/GazeAssist pipeline was designed for local processing. The newer multi-agent system uses external Gemini, ElevenLabs, and optionally Firebase services. Do not describe the two configurations as having identical privacy or network behavior.

## 3.1 Try the core toolkit

The basic eye-twin and analytics demos can be tried without a webcam, GPU, model weights, or internet connection. NumPy is the minimum dependency for the lightweight demo path.

Run these commands from the directory containing the corresponding Python modules:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy

# Eye analytics demo
python gaze_core.py demo --seconds 20

# Generate/preview synthetic eyes
python eye_twin.py preview --out artifacts/eye_preview

# Validate synthetic twin through the core pipeline
python eye_twin.py validate

# Original scripted assistive scene
python gaze_assist.py demo --seconds 30
```

Run a module's self-test independently:

```bash
python gaze_core.py selftest
python eye_twin.py selftest
python gaze_assist.py selftest
```

Example simulated alerts:

```text
[WARN] Chair approaching ahead, about 3 seconds.
[STOP] Chair approaching ahead, 1.1 meters. Stop.
[STOP] Car approaching ahead, 2.4 meters. Stop.
[STOP] Stairs very close ahead. Stop.
```

> These messages come from a scripted simulation, not a real walk or a validated obstacle-detection test.

## 3.2 Optional live core toolkit

The original local eye-analytics and assistive modules can be run with their optional dependencies:

```bash
pip install opencv-python mediapipe

mkdir -p models
curl -L -o models/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task

# Live eye analytics with preview
python gaze_core.py run --camera 0 --model models/face_landmarker.task --display

# Original assistive prototype (console + speech)
pip install ultralytics  # optional; large download
python gaze_assist.py run --camera 0 --detector auto --voice --acknowledge-risks
```

The face-landmarker model is downloaded separately. The original `gaze_core.py` camera workflow and the newer Gemini-based multi-agent workflow have different data paths; consult the relevant module and configuration before running either with a camera.

## 3.3 Original GazeAssist risk levels

| Priority | Trigger |
|---|---|
| 🔴 **Critical** | Distance ≤ 0.9 m, or TTC ≤ 1.6 s, or stairs ≤ 1.6 m |
| 🟠 **Warning** | Distance ≤ 1.8 m, or TTC ≤ 3.0 s |
| 🟡 **Caution** | Risk score ≥ 0.22 |

These are the documented thresholds of the original GazeAssist risk engine. They should not be interpreted as validated safety thresholds or assumed to be identical to every configuration in the newer multi-agent implementation.

---

# 🕶️ 4. Samsung Android XR glasses integration

GazeAssist is a natural fit for audio-first smart glasses: an outward-facing camera provides scene input, while short spoken alerts provide output. The repository includes an Android XR integration skeleton, but **the glasses integration is not yet a verified deployment**.

## 4.1 Platform concept

The Android XR approach described by the project uses a companion Android phone application and Jetpack Projected to project an activity to the glasses. The phone-side app is intended to access the outward-facing camera and glasses audio through a projected context. The glasses camera supports one active stream, so the skeleton binds an image-analysis use case. The relevant SDK was a developer preview when this integration was written; APIs and availability may change.

## 4.2 Component suitability

| Component | Glasses suitability | Reason |
|---|---|---|
| Original `gaze_assist.py` | Potentially, via a port | Its outward-camera → risk → spoken-alert loop matches an audio-glasses workflow. |
| `gaze_core.py` | Not as designed | Eye analytics require a camera facing the user's eyes; the documented glasses camera faces outward. |
| `eye_twin.py` | Development tooling | Generates synthetic signals and test data; it is not a glasses runtime component. |
| LangGraph multi-agent system | Phone/companion-side processing is the intended route | The current implementation is Python-based and relies on cloud APIs for vision and speech. |

## 4.3 Integration status

| File/component | Purpose / status |
|---|---|
| `GazeAssistGlasses.kt` | Kotlin port of the risk engine, alert throttling, detector interface, and CameraX newest-frame-only analysis binding. |
| Browser glasses demo | Simulated camera walk with risk alerts and optional voice. |
| Kotlin risk engine | Written, but not compiled or unit-tested according to the project notes. |
| Projected-context camera binding | Skeleton with TODOs; API names need verification against the relevant Android XR SDK version. |
| Android XR emulator | Not yet run. |
| Physical Samsung glasses | Not tested. |
| Detection accuracy, frame rate, and link latency | Not measured. |

**Next glasses milestone:** compile the Kotlin port, connect it to the emulator camera, and record measured performance before attempting a physical-device test.

For the newer multi-agent version, audio forwarding is a separate integration task: connect `AudioOut(on_audio=...)` to a phone application that can deliver the audio to the glasses.

---

# 🛣️ 5. Roadmap

1. **Multi-agent live integration:** verify Gemini, ElevenLabs, and Firebase end-to-end with test credentials and documented failure handling.
2. **Voice interaction:** add speech-to-text so users can ask questions and teach the memory system without typing.
3. **Distance and scene validation:** compare estimates with measured ground truth in controlled environments.
4. **Object tracking:** replace label-and-zone matching with a robust multi-object tracking method.
5. **Route and memory evaluation:** test room recognition, route retrieval, memory promotion, expiry, and correction workflows.
6. **Glasses bring-up:** compile the Kotlin port, run it on the Android XR emulator, then test on supported hardware and publish measured latency and frame rate.
7. **Metric depth and sensor fusion:** investigate stereo/ToF depth and IMU integration.
8. **Custom hazard detectors:** improve detection for doors, stairs, curbs, and other objects not reliably covered by generic models.
9. **Embedded runtime:** explore ONNX/TFLite, quantization, and watchdogs for lower-latency on-device execution.
10. **Human-centred validation:** co-design with blind and low-vision users, O&M experts, and accessibility researchers; evaluate spatial audio and haptics.
11. **Artificial eye dataset automation:** complete the deterministic ten-position render pipeline, pupil-center extraction, metadata validation, and contact-sheet generation.

---

# 🧰 6. Technologies

- **Core toolkit:** Python · NumPy · OpenCV · MediaPipe Face Landmarker · Ultralytics YOLO (optional) · ONNX depth (optional)
- **Multi-agent GazeAssist:** LangGraph · Google Gemini API · ElevenLabs · Firebase Firestore · Python
- **Artificial eye rig:** Blender · Python · MakeHuman-related assets · OBJ · PNG · JSON
- **Wearable integration:** Kotlin · Android XR SDK / Jetpack Projected · CameraX

---