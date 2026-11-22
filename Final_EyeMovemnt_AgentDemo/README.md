# Artificial Eye Movement & Procedural Face Rig

**Project:** `Final_EyeMovemnt_AgentDemo`

A Python-controlled artificial eye movement system built using Blender and a textured, articulatable 3D face rig. The project focuses on generating controlled eye movements, independently articulating the left and right eyes, and producing deterministic renders with corresponding ground-truth metadata.

The system is designed to support automated eye movement generation, reproducible camera positioning, and dataset generation for eye-tracking and gaze-related applications.

---

## 1. Project Overview

The project implements an artificial eye rig using a 3D facial model with articulated eye structures.

Python scripts provide programmatic control over eye movement, allowing the eyes to move horizontally and vertically while maintaining a fixed camera and viewing configuration.

The primary objectives are to:

- Develop a textured and articulatable artificial face rig.
- Implement left and right eye articulation.
- Provide deterministic Python-based eye movement controls.
- Maintain a fixed camera and fixation configuration.
- Generate a sequence of 10 clockwise eye movement renders.
- Produce ground-truth JSON metadata containing the 2D pupil centers for every rendered frame.

### Key Specifications

| Component | Specification |
|---|---|
| Face rig | Textured, articulatable 3D face |
| Rig foundation | MakeHuman / GNM / FLAME |
| Eye articulation | Left and right eye bones |
| Fixation distance | 35 cm forward |
| Camera distance | Locked at 40 cm |
| Render resolution | 1024 × 1024 |
| Image format | PNG |
| Eye control | Python |
| Movement | Horizontal and vertical |
| Rendering sequence | 10 clockwise positions |
| Angular positions | θₖ = 2πk/10 |
| Metadata | Per-frame JSON |
| Ground truth | 2D pupil centers |

---

## 2. Project Structure

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

### File Descriptions

| File | Purpose |
|---|---|
| `base.obj` | Base 3D facial geometry used for the artificial face model. |
| `build_face_rig.py` | Python script associated with constructing and configuring the face rig. |
| `default_weights.mhw` | MakeHuman weight configuration file. |
| `default.mhskel` | Skeleton definition used for the character rig. |
| `eyes_wrapper.py` | Python interface for controlling artificial eye movement. |
| `FaceRig.blend` | Main Blender project containing the face model, rig, and scene configuration. |
| `FaceRig.blend1` | Blender backup of the project file. |
| `make_skin_texture.py` | Python script for generating or preparing the skin texture. |
| `skin_diffuse.jpg` | Diffuse skin texture applied to the artificial face model. |

---

## 3. Technology Stack

| Technology | Application |
|---|---|
| Python | Automation and eye movement control |
| Blender | 3D modelling, rigging and rendering |
| MakeHuman | Character modelling and rig foundation |
| OBJ | 3D geometry representation |
| JSON | Ground-truth metadata storage |
| PNG | Rendered image output |

---

## 4. Installation and Setup

### Prerequisites

Install the following software before running the project:

1. Python 3.x
2. Blender (with Python scripting support)
3. MakeHuman, if required for regenerating the original face model.

Blender includes its own Python environment, which should be used when executing scripts that depend on the `bpy` module.

### Step 1: Clone the Repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd Final_EyeMovemnt_AgentDemo
```

### Step 2: Open the Blender Project

Open Blender and load:

```text
FaceRig.blend
```

Verify that the scene contains:

- The artificial face mesh.
- Eye geometry.
- Left and right eye bones.
- Appropriate armature and rig relationships.
- Skin material and diffuse texture.
- Camera configuration.

### Step 3: Verify Textures

Ensure that `skin_diffuse.jpg` is accessible to Blender.

If textures are missing:

1. Open the Shading workspace.
2. Locate the skin material.
3. Verify the image texture connection.
4. Relink `skin_diffuse.jpg` if necessary.

### Step 4: Configure Python Scripts

The project contains two primary Python scripts:

**`build_face_rig.py`**

Responsible for the face rig construction workflow.

**`eyes_wrapper.py`**

Provides the Python interface for artificial eye movement.

Execute Blender-dependent scripts through Blender's Python environment or its Scripting workspace.

---

## 5. Artificial Eye Movement System

The artificial eye system provides programmatic control over the direction of gaze.

The left and right eye structures are articulated through their respective eye bones, allowing controlled movement while preserving the overall facial structure.

### Movement Parameters

The eye movement interface is designed around two primary parameters:

- `h`: Horizontal eye movement.
- `v`: Vertical eye movement.

The wrapper is specified to expose the following control interface:

```python
eyes.move(h, v)
eyes.reset()
```

**`eyes.move(h, v)`**

Changes the eye orientation according to the supplied horizontal and vertical movement values.

**`eyes.reset()`**

Restores the eyes to their default orientation.

The exact numerical range and coordinate convention for `h` and `v` depend on the implementation in `eyes_wrapper.py`.

### Eye Movement Workflow

1. Load the artificial face model.
2. Initialize the eye rig and its control interface.
3. Set the required horizontal and vertical eye positions.
4. Apply the transformations to the eye bones.
5. Update the Blender scene.
6. Render the resulting eye position.
7. Extract the corresponding 2D pupil coordinates.

---

## 6. Deterministic Eye Movement and Rendering

A major objective of this project is reproducible eye movement generation.

The system uses a fixed camera configuration and predefined angular positions to generate a controlled sequence of artificial eye movements.

### Clockwise Angular Sequence

The rendering sequence consists of 10 positions:

\[
\theta_k = \frac{2\pi k}{10}
\]

where:

\[
k = 0,1,2,\ldots,9
\]

Each angular position represents a step in the clockwise movement sequence.

The corresponding angular increment is:

\[
\Delta\theta = \frac{360^\circ}{10}=36^\circ
\]

This provides 10 evenly distributed angular positions over one complete revolution.

### Rendering Configuration

| Parameter | Value |
|---|---|
| Number of frames | 10 |
| Movement direction | Clockwise |
| Angular increment | 36° |
| Camera position | Fixed |
| Camera distance | 40 cm |
| Fixation distance | 35 cm |
| Resolution | 1024 × 1024 |
| Output format | PNG |

The camera remains locked throughout the sequence to maintain consistent image geometry.

---

## 7. Ground-Truth Metadata Generation

Alongside each rendered image, the system is intended to generate a JSON metadata file.

The metadata records the corresponding 2D pupil centers.

This information can be used for eye localization, gaze analysis, tracking evaluation, and supervised dataset generation.

### Expected Output Structure

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

### Example JSON Metadata

```json
{
    "frame_id": 0,
    "angle_degrees": 0,
    "movement_direction": "clockwise",
    "image_resolution": [1024, 1024],
    "pupil_centers": {
        "left": {
            "x": 0,
            "y": 0
        },
        "right": {
            "x": 0,
            "y": 0
        }
    }
}
```

*Note: The pupil coordinates shown above are placeholders. Actual values must be calculated from the rendered scene.*

---

## 8. Expected Deliverables

| Deliverable | Description |
|---|---|
| Blender scene | Complete textured face with articulatable eye rig |
| Python wrapper | Deterministic eye movement interface |
| Setup documentation | Installation and execution instructions |
| 10 PNG renders | Clockwise eye movement sequence |
| JSON metadata | Per-frame 2D pupil center coordinates |
| Contact sheet | Combined visualization of the rendered sequence |
| Demo video | Optional 1–2 minute demonstration |

---

## 9. Evaluation Criteria

| Evaluation Component | Weight |
|---|---:|
| Execution and Efficiency | 40% |
| Calibration | 20% |
| Automation | 20% |
| Quality and Movement | 60% |
| Direction accuracy | 40% |
| Sequence and metadata | 20% |

---

## 10. License

The project includes a Creative Commons CC0 1.0 Universal license declaration.

CC0 allows works to be dedicated to the public domain to the extent permitted by applicable law.

Ensure that the declaration applies to the actual project assets before distributing the complete repository under CC0. Third-party models, textures, and other included assets may have separate licensing requirements.

---

## 11. Future Improvements

Potential extensions to the artificial eye movement system include:

- Independent calibration of left and right pupil positions.
- Automated gaze direction estimation.
- More granular eye movement trajectories.
- Automated pupil-center extraction and validation.
- Generation of larger synthetic eye-tracking datasets.
- Integration with computer vision-based eye-tracking models.

---

**Project Status:** Artificial face rig and Python-based eye articulation project. Final rendering automation, metadata validation, and demonstration should be verified against the completed implementation.
