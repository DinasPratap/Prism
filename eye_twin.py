"""EyeTwin: a controllable synthetic digital twin of human eye movement.

``eye_twin`` builds on :mod:`gaze_core` and has three jobs:

1. **Signal simulation** -- generate temporally coherent, reproducible
   sequences of gaze, blinks, saccades, fixations, pupil, lighting, head pose
   and occlusion from explicit motor parameters.
2. **Schematic rendering + datasets** -- rasterise those states into simple
   eye images (numpy only), and export images, per-frame labels (CSV) and
   metadata (JSON) with deterministic train/validation/test splits and domain
   randomization.
3. **Validation** -- replay twin observations through ``gaze_core`` and
   compare the recovered signals with the twin's ground truth, producing an
   honest closed-loop check of the analytics pipeline.

Honesty about realism
---------------------
The procedural renderer draws **schematic** eyes with geometric primitives,
shading, pose warping, blink occlusion and noise.  It is useful for pipeline
tests, label generation and demos.  It is **not** a physically based or
photorealistic eye renderer; cornea refraction, tear film, iris texture,
eyelid-skin deformation and camera optics are not modelled.  For
photorealistic data you need a 3D rendering engine; see
:class:`BlenderIntegration`, which exports a job specification and a starter
script for a user-supplied Blender eye rig.

Reproducibility
---------------
Every sequence is driven by an explicit seed.  Same parameters + seed + fps +
duration => identical samples and labels.  Dataset splits are assigned with a
seeded permutation, so rebuilding with the same seed reproduces the split.

Run ``python eye_twin.py --help`` for dataset generation, previews,
validation and self-tests.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import logging
import math
import shutil
import statistics
import struct
import sys
import time
import zlib
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from gaze_core import (
    EventType,
    EyeGeometry,
    FaceObservation,
    GazeConfig,
    GazePipeline,
    make_config,
)

LOG = logging.getLogger("gazeos.twin")

__all__ = [
    "BlenderIntegration",
    "CameraPose",
    "DatasetConfig",
    "DatasetManifest",
    "EyeAppearance",
    "EyeDatasetBuilder",
    "EyeMotionParams",
    "EyeTwinGenerator",
    "EyeTwinObservationSource",
    "EyeTwinParams",
    "Lighting",
    "ProceduralEyeRenderer",
    "TwinSample",
    "TwinSequence",
    "preset",
    "preset_names",
    "render_preview",
    "sample_to_face",
    "validate_against_core",
    "write_png",
]

PRESETS: tuple[str, ...] = ("alert", "fatigued", "reading", "driving", "scanning")


# --------------------------------------------------------------------------- #
# Parameters
# --------------------------------------------------------------------------- #
@dataclass
class EyeAppearance:
    """Static, per-sequence appearance (schematic, not biometric)."""

    iris_color: tuple[int, int, int] = (72, 108, 136)
    skin_color: tuple[int, int, int] = (206, 170, 142)
    sclera_color: tuple[int, int, int] = (238, 238, 232)
    iris_radius_ratio: float = 0.30  # of image half-width
    pupil_ratio: float = 0.38  # of iris radius
    lid_open: float = 1.0  # structural eyelid aperture (fatigue narrows it)
    lash_strength: float = 0.75
    iris_texture: float = 0.35

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Lighting:
    """Illumination parameters (used by the schematic renderer)."""

    intensity: float = 0.95
    ambient: float = 0.30
    azimuth_deg: float = -25.0
    glare: float = 0.12
    variation: float = 0.08  # slow temporal modulation amplitude
    flicker_hz: float = 0.0  # reserved for future rendering; not animated

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CameraPose:
    """Camera/head-relative pose with optional per-frame jitter in the twin.

    ``jitter_deg`` is applied by the dataset builder (domain randomization),
    not by the motor model.
    """

    yaw_deg: float = 0.0
    pitch_deg: float = 0.0
    roll_deg: float = 0.0
    distance: float = 1.0
    fov_deg: float = 60.0
    jitter_deg: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EyeMotionParams:
    """Oculomotor parameters.  Values are plausible, not subject-calibrated."""

    fixation_duration_mean_s: float = 0.38
    fixation_duration_sigma: float = 0.35  # lognormal sigma
    fixation_noise_deg: float = 0.30
    fixation_drift_gain: float = 0.12
    microsaccade_rate_per_s: float = 0.3
    microsaccade_amplitude_deg: float = 0.25
    saccade_amplitude_mean_deg: float = 8.0
    saccade_peak_velocity_deg_s: float = 350.0
    saccade_range_x_deg: tuple[float, float] = (-15.0, 15.0)
    saccade_range_y_deg: tuple[float, float] = (-9.0, 9.0)
    blink_rate_per_min: float = 15.0
    blink_duration_mean_s: float = 0.18
    blink_duration_sigma: float = 0.22
    blink_probability: float = 1.0
    head_yaw_amplitude_deg: float = 2.0
    head_pitch_amplitude_deg: float = 1.2
    head_roll_amplitude_deg: float = 0.8
    head_motion_period_s: float = 8.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EyeTwinParams:
    """Complete, serialisable configuration of one synthetic eye.

    The digital twin is explicitly a *simulation*.  ``renderer_note`` travels
    with exported metadata so consumers cannot mistake the images for real
    recordings.
    """

    profile: str = "alert"
    fps: float = 30.0
    seed: int = 0
    appearance: EyeAppearance = field(default_factory=EyeAppearance)
    lighting: Lighting = field(default_factory=Lighting)
    camera: CameraPose = field(default_factory=CameraPose)
    motion: EyeMotionParams = field(default_factory=EyeMotionParams)
    iris_excursion_ratio: float = 0.55  # iris travel as a fraction of half-width
    noise_sigma: float = 0.02  # image noise (fraction of 255)
    occlusion_probability: float = 0.0  # chance per second of an occluder episode
    occlusion_duration_s: tuple[float, float] = (0.3, 1.2)
    renderer: str = "procedural"
    renderer_note: str = (
        "Schematic geometric rendering, not physically based or photorealistic."
    )

    def __post_init__(self) -> None:
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if not 0.0 <= self.occlusion_probability <= 1.0:
            raise ValueError("occlusion_probability must be in [0, 1]")
        lo, hi = self.occlusion_duration_s
        if lo <= 0 or hi <= lo:
            raise ValueError("occlusion_duration_s must be (low, high) with 0 < low < high")
        if self.motion.blink_rate_per_min < 0:
            raise ValueError("blink_rate_per_min must be >= 0")
        if self.motion.fixation_duration_mean_s <= 0:
            raise ValueError("fixation_duration_mean_s must be > 0")

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "fps": self.fps,
            "seed": self.seed,
            "appearance": self.appearance.to_dict(),
            "lighting": self.lighting.to_dict(),
            "camera": self.camera.to_dict(),
            "motion": self.motion.to_dict(),
            "iris_excursion_ratio": self.iris_excursion_ratio,
            "noise_sigma": self.noise_sigma,
            "occlusion_probability": self.occlusion_probability,
            "occlusion_duration_s": list(self.occlusion_duration_s),
            "renderer": self.renderer,
            "renderer_note": self.renderer_note,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EyeTwinParams":
        """Rebuild parameters from :meth:`to_dict` output (extra keys ignored)."""

        def build(dc_type: Any, payload: Any) -> Any:
            if not isinstance(payload, Mapping):
                return dc_type()
            valid = {f.name for f in fields(dc_type)}
            clean = {k: v for k, v in payload.items() if k in valid}
            for key in ("occlusion_duration_s",):
                if key in clean and isinstance(clean[key], list):
                    clean[key] = tuple(clean[key])
            for key in ("iris_color", "skin_color", "sclera_color"):
                if key in clean and isinstance(clean[key], list):
                    clean[key] = tuple(clean[key])
            return dc_type(**clean)

        return cls(
            profile=str(data.get("profile", "alert")),
            fps=float(data.get("fps", 30.0)),
            seed=int(data.get("seed", 0)),
            appearance=build(EyeAppearance, data.get("appearance")),
            lighting=build(Lighting, data.get("lighting")),
            camera=build(CameraPose, data.get("camera")),
            motion=build(EyeMotionParams, data.get("motion")),
            iris_excursion_ratio=float(data.get("iris_excursion_ratio", 0.55)),
            noise_sigma=float(data.get("noise_sigma", 0.02)),
            occlusion_probability=float(data.get("occlusion_probability", 0.0)),
            occlusion_duration_s=tuple(data.get("occlusion_duration_s", (0.3, 1.2))),  # type: ignore[arg-type]
            renderer=str(data.get("renderer", "procedural")),
            renderer_note=str(data.get("renderer_note", "")),
        )


def preset_names() -> tuple[str, ...]:
    """Return the available eye-twin presets."""
    return PRESETS


def preset(name: str) -> EyeTwinParams:
    """Return a deep copy of a named preset.

    Presets only adjust parameters; they are not clinical or population
    models.  ``fatigued`` is an exaggerated, demo-friendly pattern (more
    blinks, longer closures, narrower lids, longer fixations), not a
    diagnostic of fatigue.
    """
    if name not in PRESETS:
        raise ValueError(f"Unknown preset {name!r}; choose from {PRESETS}")
    params = EyeTwinParams(profile=name)
    if name == "fatigued":
        params.appearance = replace(params.appearance, lid_open=0.78)
        params.motion = replace(
            params.motion,
            fixation_duration_mean_s=0.55,
            fixation_noise_deg=0.55,
            saccade_peak_velocity_deg_s=250.0,
            blink_rate_per_min=28.0,
            blink_duration_mean_s=0.30,
            blink_duration_sigma=0.45,
        )
        params.lighting = replace(params.lighting, intensity=0.82)
    elif name == "reading":
        params.appearance = replace(params.appearance, lid_open=0.92)
        params.motion = replace(
            params.motion,
            fixation_duration_mean_s=0.26,
            fixation_noise_deg=0.18,
            saccade_amplitude_mean_deg=4.0,
            saccade_peak_velocity_deg_s=220.0,
            saccade_range_x_deg=(-9.0, 9.0),
            saccade_range_y_deg=(-6.0, 6.0),
            blink_rate_per_min=20.0,
        )
    elif name == "driving":
        params.motion = replace(
            params.motion,
            fixation_duration_mean_s=0.50,
            saccade_amplitude_mean_deg=12.0,
            saccade_peak_velocity_deg_s=420.0,
            saccade_range_x_deg=(-30.0, 30.0),
            saccade_range_y_deg=(-12.0, 12.0),
            blink_rate_per_min=17.0,
            head_yaw_amplitude_deg=5.0,
        )
        params.camera = replace(params.camera, yaw_deg=8.0)
        params.occlusion_probability = 0.05
    elif name == "scanning":
        params.motion = replace(
            params.motion,
            fixation_duration_mean_s=0.18,
            saccade_amplitude_mean_deg=16.0,
            saccade_peak_velocity_deg_s=480.0,
            blink_rate_per_min=13.0,
        )
        params.lighting = replace(params.lighting, variation=0.18)
        params.camera = replace(params.camera, jitter_deg=1.5)
    return params


# --------------------------------------------------------------------------- #
# Sequence data contracts
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TwinSample:
    """One frame of simulated eye state and label.

    Attributes
    ----------
    state:
        ``"fixation"`` or ``"saccade"`` (motor state; blinks are orthogonal).
    blink:
        Eyelid closure amount in ``[0, 1]`` (0 open, 1 closed).
    gaze_x, gaze_y:
        Normalized gaze in ``[-1, 1]`` (eye-space), for compatibility with
        ``gaze_core``.  ``gaze_x_deg``/``gaze_y_deg`` are the simulated
        angular values.
    """

    t: float
    gaze_x: float
    gaze_y: float
    gaze_x_deg: float
    gaze_y_deg: float
    state: str
    blink: float
    pupil_mm: float
    fixation_id: int
    saccade_id: int
    blink_id: int
    gaze_speed_deg_s: float
    head_yaw_deg: float
    head_pitch_deg: float
    head_roll_deg: float
    lighting: float
    occlusion: float
    is_occluded: bool
    is_blink: bool

    def to_row(self) -> dict[str, Any]:
        return {
            "t": round(self.t, 5),
            "gaze_x": round(self.gaze_x, 5),
            "gaze_y": round(self.gaze_y, 5),
            "gaze_x_deg": round(self.gaze_x_deg, 3),
            "gaze_y_deg": round(self.gaze_y_deg, 3),
            "state": self.state,
            "blink": round(self.blink, 4),
            "is_blink": int(self.is_blink),
            "pupil_mm": round(self.pupil_mm, 3),
            "fixation_id": self.fixation_id,
            "saccade_id": self.saccade_id,
            "blink_id": self.blink_id,
            "gaze_speed_deg_s": round(self.gaze_speed_deg_s, 2),
            "head_yaw_deg": round(self.head_yaw_deg, 3),
            "head_pitch_deg": round(self.head_pitch_deg, 3),
            "head_roll_deg": round(self.head_roll_deg, 3),
            "lighting": round(self.lighting, 3),
            "occlusion": round(self.occlusion, 3),
            "is_occluded": int(self.is_occluded),
        }


@dataclass
class TwinSequence:
    """A generated sequence plus its split, labels and serialisable metadata."""

    sequence_id: str
    params: EyeTwinParams
    samples: list[TwinSample]
    split: str = "train"
    started_at: float = field(default_factory=time.time)

    @property
    def fps(self) -> float:
        return self.params.fps

    @property
    def duration_s(self) -> float:
        return len(self.samples) / self.params.fps if self.params.fps else 0.0

    def episodes(self) -> dict[str, int]:
        """Count motor events by segment transitions."""
        stats = {
            "blinks": 0,
            "fixations": 0,
            "saccades": 0,
            "occlusions": 0,
        }
        last_state = None
        last_blink = False
        last_occ = False
        for sample in self.samples:
            if sample.state != last_state:
                if sample.state == "fixation":
                    stats["fixations"] += 1
                elif sample.state == "saccade":
                    stats["saccades"] += 1
                last_state = sample.state
            if sample.is_blink and not last_blink:
                stats["blinks"] += 1
            if sample.is_occluded and not last_occ:
                stats["occlusions"] += 1
            last_blink = sample.is_blink
            last_occ = sample.is_occluded
        return stats

    def metadata(self, image_dir: str | None = None) -> dict[str, Any]:
        return {
            "schema": "gazeos.eye_twin.sequence.v1",
            "sequence_id": self.sequence_id,
            "split": self.split,
            "simulated": True,
            "photorealistic": False,
            "generator": "eye_twin.py",
            "created_at": self.started_at,
            "seed": self.params.seed,
            "profile": self.params.profile,
            "fps": self.params.fps,
            "duration_s": round(self.duration_s, 4),
            "n_frames": len(self.samples),
            "labels": self.episodes(),
            "renderer_note": self.params.renderer_note,
            "image_dir": image_dir,
            "params": self.params.to_dict(),
        }


# --------------------------------------------------------------------------- #
# Generator
# --------------------------------------------------------------------------- #
def _clip(value: float, low: float, high: float) -> float:
    return low if value < low else high if value > high else value


def _lognormal(rng: np.random.Generator, mean: float, sigma: float) -> float:
    mean = max(mean, 1e-4)
    return float(rng.lognormal(mean=math.log(mean), sigma=max(sigma, 1e-4)))


def _sample_gaze_target(
    rng: np.random.Generator, motion: EyeMotionParams
) -> np.ndarray:
    x_lo, x_hi = motion.saccade_range_x_deg
    y_lo, y_hi = motion.saccade_range_y_deg
    return np.array(
        [rng.uniform(x_lo, x_hi), rng.uniform(y_lo, y_hi)], dtype=np.float64
    )


def _saccade_peak_velocity(amplitude_deg: float, motion: EyeMotionParams) -> float:
    """Simplified main-sequence relation (capped, saturating)."""
    amplitude = max(amplitude_deg, 0.1)
    cap = max(motion.saccade_peak_velocity_deg_s, 30.0)
    return cap * amplitude / (amplitude + 5.0)


class EyeTwinGenerator:
    """Seeded state-machine generator for temporally coherent eye signals.

    Blinks are scheduled independently of fixations/saccades (as in reality),
    while gaze drift, microsaccades, head pose and lighting evolve smoothly.
    The model is phenomenological: amplitudes, velocities and durations are
    plausible defaults, not fitted to measured human data.
    """

    def __init__(self, params: EyeTwinParams | None = None, seed: int | None = None) -> None:
        self.params = params or preset("alert")
        self.seed = int(seed if seed is not None else self.params.seed)
        self.rng = np.random.default_rng(self.seed)

    def sequence(
        self,
        duration_s: float = 20.0,
        fps: float | None = None,
        start_time: float = 0.0,
        sequence_id: str = "sequence",
        split: str = "train",
        fixed_gaze_deg: tuple[float, float] | None = None,
    ) -> TwinSequence:
        """Generate one reproducible sequence of samples.

        ``fixed_gaze_deg`` holds gaze at an explicit angular position while
        blinks, head motion, lighting and noise continue.  This is useful for
        controlled appearance/rendering tests; normally gaze is generated by
        the saccade/fixation state machine.
        """
        if duration_s <= 0:
            raise ValueError("duration_s must be positive")
        p = self.params
        fps = float(fps or p.fps)
        n = max(1, int(round(duration_s * fps)))
        dt = 1.0 / fps
        rng = self.rng
        motion = p.motion

        gaze = np.zeros(2, dtype=np.float64)
        target = _sample_gaze_target(rng, motion)
        state = "fixation"
        fixation_end = start_time + _clip(
            _lognormal(rng, motion.fixation_duration_mean_s, motion.fixation_duration_sigma),
            0.08,
            3.0,
        )
        saccade_from = gaze.copy()
        saccade_to = target.copy()
        saccade_start = 0.0
        saccade_duration = 0.1
        fixation_id = 1
        saccade_id = 0
        blink_id = 0
        blink_start: float | None = None
        blink_duration = 0.0
        occlusion_start: float | None = None
        occlusion_duration = 0.0
        prev_gaze: np.ndarray | None = None
        head_phase = float(rng.uniform(0.0, 2.0 * math.pi))
        light_phase = float(rng.uniform(0.0, 2.0 * math.pi))
        samples: list[TwinSample] = []

        for i in range(n):
            t = start_time + i * dt

            # --- independent blink schedule -------------------------------- #
            blink = 0.0
            if blink_start is None:
                per_frame = (
                    max(motion.blink_rate_per_min, 0.0) / 60.0 * motion.blink_probability * dt
                )
                if rng.random() < per_frame:
                    blink_start = t
                    blink_duration = _clip(
                        _lognormal(rng, motion.blink_duration_mean_s, motion.blink_duration_sigma),
                        0.06,
                        0.9,
                    )
                    blink_id += 1
            if blink_start is not None:
                phase = (t - blink_start) / blink_duration
                if phase >= 1.0:
                    blink_start = None
                    blink = 0.0
                else:
                    blink = math.sin(math.pi * _clip(phase, 0.0, 1.0)) ** 0.8

            # --- occlusion schedule ---------------------------------------- #
            occlusion = 0.0
            if occlusion_start is None:
                if rng.random() < p.occlusion_probability * dt:
                    occlusion_start = t
                    lo, hi = p.occlusion_duration_s
                    occlusion_duration = float(rng.uniform(lo, hi))
            if occlusion_start is not None:
                phase = (t - occlusion_start) / max(occlusion_duration, 1e-3)
                if phase >= 1.0:
                    occlusion_start = None
                else:
                    ramp = _clip(min(phase, 1.0 - phase) * 8.0, 0.0, 1.0)
                    occlusion = 0.85 * ramp

            # --- motor state ------------------------------------------------ #
            if fixed_gaze_deg is not None:
                gaze = np.array(fixed_gaze_deg, dtype=np.float64)
                state = "fixation"
            elif state == "fixation":
                drift = rng.normal(0.0, motion.fixation_noise_deg * 0.12, size=2)
                gaze = gaze + drift + motion.fixation_drift_gain * (target - gaze)
                if rng.random() < motion.microsaccade_rate_per_s * dt:
                    gaze += rng.normal(0.0, motion.microsaccade_amplitude_deg, size=2)
                if t >= fixation_end:
                    saccade_from = gaze.copy()
                    saccade_to = _sample_gaze_target(rng, motion)
                    amplitude = float(np.linalg.norm(saccade_to - saccade_from))
                    peak = _saccade_peak_velocity(amplitude, motion)
                    # Triangular-ish velocity profile: duration = 1.5 A / v_peak.
                    saccade_duration = max(1.5 / fps, 1.5 * amplitude / max(peak, 1.0))
                    saccade_start = t
                    saccade_id += 1
                    state = "saccade"
            if state == "saccade":
                phase = (t - saccade_start) / max(saccade_duration, 1e-4)
                if phase >= 1.0:
                    gaze = saccade_to.copy()
                    target = saccade_to.copy()
                    fixation_id += 1
                    state = "fixation"
                    fixation_end = t + _clip(
                        _lognormal(
                            rng,
                            motion.fixation_duration_mean_s,
                            motion.fixation_duration_sigma,
                        ),
                        0.08,
                        3.0,
                    )
                else:
                    ease = phase * phase * (3.0 - 2.0 * phase)
                    gaze = saccade_from + (saccade_to - saccade_from) * ease

            gaze[0] = _clip(gaze[0], motion.saccade_range_x_deg[0], motion.saccade_range_x_deg[1])
            gaze[1] = _clip(gaze[1], motion.saccade_range_y_deg[0], motion.saccade_range_y_deg[1])

            speed = (
                0.0
                if prev_gaze is None
                else float(np.linalg.norm(gaze - prev_gaze) / dt)
            )
            prev_gaze = gaze.copy()

            cycle = 2.0 * math.pi * (t - start_time) / max(motion.head_motion_period_s, 0.1)
            head_yaw = motion.head_yaw_amplitude_deg * math.sin(cycle + head_phase)
            head_pitch = motion.head_pitch_amplitude_deg * math.sin(cycle * 0.7 + head_phase * 1.3)
            head_roll = motion.head_roll_amplitude_deg * math.sin(cycle * 0.5 + head_phase * 0.7)
            lighting = _clip(
                p.lighting.intensity
                * (1.0 + p.lighting.variation * math.sin(light_phase + t * 0.7)),
                0.15,
                2.0,
            )
            pupil = float(
                _clip(
                    3.0 + 1.1 * (1.0 - _clip(lighting, 0.0, 1.5) / 1.5)
                    + rng.normal(0.0, 0.08),
                    1.8,
                    7.5,
                )
            )
            max_x = max(abs(motion.saccade_range_x_deg[0]), abs(motion.saccade_range_x_deg[1]), 1.0)
            max_y = max(abs(motion.saccade_range_y_deg[0]), abs(motion.saccade_range_y_deg[1]), 1.0)

            samples.append(
                TwinSample(
                    t=round(t, 6),
                    gaze_x=float(_clip(gaze[0] / max_x, -1.0, 1.0)),
                    gaze_y=float(_clip(gaze[1] / max_y, -1.0, 1.0)),
                    gaze_x_deg=float(gaze[0]),
                    gaze_y_deg=float(gaze[1]),
                    state=state,
                    blink=float(blink),
                    pupil_mm=pupil,
                    fixation_id=fixation_id,
                    saccade_id=saccade_id,
                    blink_id=blink_id,
                    gaze_speed_deg_s=speed,
                    head_yaw_deg=float(head_yaw),
                    head_pitch_deg=float(head_pitch),
                    head_roll_deg=float(head_roll),
                    lighting=float(lighting),
                    occlusion=float(occlusion),
                    is_occluded=occlusion > 0.05,
                    is_blink=blink > 0.5,
                )
            )
        return TwinSequence(
            sequence_id=sequence_id,
            params=replace(p, fps=fps, seed=self.seed),
            samples=samples,
            split=split,
            started_at=time.time(),
        )


# --------------------------------------------------------------------------- #
# Bridge to gaze_core
# --------------------------------------------------------------------------- #
def sample_to_face(sample: TwinSample, params: EyeTwinParams) -> FaceObservation:
    """Convert a twin sample to the backend-agnostic ``gaze_core`` contract.

    The iris is displaced inside a synthetic eyelid aperture proportionally to
    the labelled gaze, with the same conventions as the procedural simulator
    in ``gaze_core``.  This makes it possible to run the *real* analytics on
    simulated signals (closed-loop validation).
    """
    openness = _clip(1.0 - sample.blink, 0.0, 1.0) * max(params.appearance.lid_open, 0.05)
    eye_w = 0.085
    half_w = eye_w / 2.0
    half_h = 0.0135 * (0.12 + 0.88 * openness)
    eyes: dict[str, EyeGeometry] = {}
    for side, cx in (("right", 0.445), ("left", 0.555)):
        cy = 0.5
        iris_x = cx + sample.gaze_x * half_w * params.iris_excursion_ratio
        iris_y = cy - sample.gaze_y * half_h * 0.82
        eyes[side] = EyeGeometry(
            inner=(cx + half_w * (1 if side == "right" else -1), cy),
            outer=(cx - half_w * (1 if side == "right" else -1), cy),
            upper=(cx, cy - half_h),
            lower=(cx, cy + half_h),
            iris=(iris_x, iris_y),
            iris_radius=0.0105,
            side=side,
        )
    return FaceObservation(
        timestamp=sample.t,
        left_eye=eyes["left"],
        right_eye=eyes["right"],
        confidence=1.0,
        yaw_deg=sample.head_yaw_deg,
        pitch_deg=sample.head_pitch_deg,
        roll_deg=sample.head_roll_deg,
        face_center=(0.5, 0.5),
        face_size=180.0,
        blink_score=sample.blink,
        source=f"eye_twin:{params.profile}",
    )


class EyeTwinObservationSource:
    """Iterable of ``FaceObservation`` generated for a duration and seed."""

    def __init__(
        self,
        profile: str = "alert",
        seed: int = 0,
        duration_s: float = 20.0,
        fps: float | None = None,
        params: EyeTwinParams | None = None,
    ) -> None:
        self.params = copy.deepcopy(params) if params is not None else preset(profile)
        self.generator = EyeTwinGenerator(self.params, seed=seed)
        self.sequence = self.generator.sequence(
            duration_s=duration_s, fps=fps, sequence_id=f"{self.params.profile}-seed{seed}"
        )

    def __iter__(self) -> Iterator[FaceObservation]:
        for sample in self.sequence.samples:
            yield sample_to_face(sample, self.params)


# --------------------------------------------------------------------------- #
# Schematic renderer
# --------------------------------------------------------------------------- #
def _smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - edge0) / max(edge1 - edge0, 1e-9), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _falloff(distance: np.ndarray, radius: float, softness: float) -> np.ndarray:
    """1 inside ``radius``, smoothly 0 outside, with anti-aliased edge."""
    t = np.clip((radius - distance) / max(softness, 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _rgb(color: Sequence[int]) -> np.ndarray:
    return np.asarray(color, dtype=np.float64)[:3]


class ProceduralEyeRenderer:
    """Numpy-only schematic eye renderer.

    Renders a single eye from a :class:`TwinSample`:

    * elliptic eyelid aperture controlling sclera/iris visibility,
    * iris with limbal ring, radial texture, pupil and specular highlight,
    * eyelash/lid shading, skin background and lighting colour,
    * camera yaw/pitch/roll warp and distance scaling,
    * blink occlusion (eyelids close), external occluders, sensor noise.

    Limitations: no cornea refraction, tear film, volumetric tissue, real
    textures or lens distortion.  Images are *schematic*, suitable for demos
    and pipeline tests, not for claims about human-eye realism.
    """

    name = "procedural-schematic"

    def __init__(self, size: tuple[int, int] = (96, 160)) -> None:
        height, width = size
        if height < 32 or width < 48:
            raise ValueError("Renderer size must be at least 32x48")
        self.size = (int(height), int(width))

    def render(self, sample: TwinSample, params: EyeTwinParams) -> np.ndarray:
        """Return an ``H x W x 3`` uint8 RGB image."""
        h, w = self.size
        appearance = params.appearance
        eye_open = _clip((1.0 - sample.blink) * appearance.lid_open, 0.0, 1.05)
        # Coordinates in units of half-width so circles stay circular.
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
        px = (xx - (w - 1) / 2.0) / ((w - 1) / 2.0)
        py = ((h - 1) / 2.0 - yy) / ((w - 1) / 2.0)

        # Camera pose warp (small-angle approximation, inverse mapped).
        yaw = math.radians(params.camera.yaw_deg + sample.head_yaw_deg)
        pitch = math.radians(params.camera.pitch_deg + sample.head_pitch_deg)
        roll = math.radians(params.camera.roll_deg + sample.head_roll_deg)
        distance = max(params.camera.distance, 0.4)
        x = px / distance
        y = py / distance
        x = x - 0.30 * math.tan(yaw)
        y = y - 0.30 * math.tan(pitch)
        cos_r, sin_r = math.cos(roll), math.sin(roll)
        xr = x * cos_r - y * sin_r
        yr = x * sin_r + y * cos_r

        light = _clip(sample.lighting, 0.1, 1.8)
        light_dir = math.radians(params.lighting.azimuth_deg)
        skin = _rgb(appearance.skin_color)
        vignette = np.clip(1.0 - 0.30 * (xr**2 + yr**2), 0.35, 1.0)
        skin_shade = (0.55 + 0.45 * light) * vignette
        rgb = skin[None, None, :] * skin_shade[:, :, None]
        rgb *= np.array([1.0, 0.985, 0.965])

        # Eyelid aperture (smooth ellipse) and per-pixel "openness".
        aw = 0.80
        ah = max(0.012, 0.315 * eye_open)
        cy = 0.015
        d = np.sqrt((xr / aw) ** 2 + ((yr - cy) / ah) ** 2)
        aperture = _falloff(d, 1.0, 0.012)  # 1 inside, 0 outside

        # Sclera with soft shading and a warm inner canthus hint.
        sclera = _rgb(appearance.sclera_color)
        shade = 0.78 + 0.22 * np.clip(1.0 - 0.45 * xr + 0.2 * yr, 0.0, 1.5)
        sclera_col = sclera[None, None, :] * shade[:, :, None] * (0.6 + 0.4 * light)
        rgb = rgb * (1.0 - aperture[:, :, None]) + sclera_col * aperture[:, :, None]

        # Iris: limited excursion keeps it inside the aperture.
        iris_x = sample.gaze_x * 0.55 * params.iris_excursion_ratio / 0.55
        iris_y = cy + sample.gaze_y * 0.20
        iris_r = appearance.iris_radius_ratio
        dist_iris = np.hypot(xr - iris_x, yr - iris_y)
        iris_mask = _falloff(dist_iris, iris_r, 0.02) * aperture
        angle = np.arctan2(yr - iris_y, xr - iris_x)
        radial = np.clip(dist_iris / iris_r, 0.0, 1.0)
        texture = (
            0.82
            + 0.10 * np.sin(radial * 38.0 + texture_phase(params, sample))
            + 0.06 * np.sin(angle * 19.0)
        )
        limbal = 1.0 - 0.35 * _smoothstep(0.72, 1.0, radial)
        iris_base = _rgb(appearance.iris_color)
        iris_col = (
            iris_base[None, None, :]
            * (texture * limbal)[:, :, None]
            * (0.55 + 0.45 * light)
        )
        # Warm/cool tint from light azimuth.
        iris_col = iris_col * np.array(
            [1.0 + 0.10 * math.cos(light_dir), 1.0, 1.0 - 0.08 * math.cos(light_dir)]
        )
        rgb = rgb * (1.0 - iris_mask[:, :, None]) + iris_col * iris_mask[:, :, None]

        # Pupil.
        pupil_r = max(0.07, iris_r * appearance.pupil_ratio * (1.0 - 0.04 * sample.pupil_mm))
        pupil_mask = _falloff(dist_iris, pupil_r, 0.012) * aperture
        rgb = rgb * (1.0 - 0.92 * pupil_mask[:, :, None])

        # Specular highlight offset toward the light.
        hi_x = iris_x - 0.09 * math.cos(light_dir) - 0.02
        hi_y = iris_y + 0.10
        hi_d = np.hypot(xr - hi_x, yr - hi_y)
        highlight = _falloff(hi_d, 0.055, 0.03) * aperture * (0.35 + 0.55 * light)
        rgb = rgb * (1.0 - highlight[:, :, None]) + 255.0 * highlight[:, :, None]

        # Lash / eyelid shading near the upper and lower boundaries.
        safe = np.clip(1.0 - (xr / aw) ** 2, 0.0, 1.0)
        boundary = cy + ah * np.sqrt(safe)
        band = np.exp(-(((yr - boundary) / (0.035 + 0.02 * eye_open)) ** 2))
        band *= aperture * (0.45 + 0.55 * appearance.lash_strength)
        rgb *= (1.0 - 0.72 * band[:, :, None])
        lower_boundary = cy - ah * np.sqrt(safe)
        lower_band = np.exp(-(((yr - lower_boundary) / 0.025) ** 2)) * aperture * 0.25
        rgb *= (1.0 - 0.35 * lower_band[:, :, None])

        # External occlusion: a soft shadow/hand-like overlay.
        if sample.occlusion > 0.01:
            occ_mask = _smoothstep(0.0, 0.6, sample.occlusion)
            occ_color = skin[None, None, :] * (0.28 + 0.12 * light)
            rgb = rgb * (1.0 - 0.85 * occ_mask) + occ_color * (0.85 * occ_mask)

        # Sensor noise, deterministic per sample time.
        if params.noise_sigma > 0:
            rng = np.random.default_rng(
                (params.seed * 1_000_003 + int(round(sample.t * 1000)) * 97) % (2**32)
            )
            rgb += rng.normal(0.0, params.noise_sigma * 255.0, size=rgb.shape)
        return np.clip(rgb, 0, 255).astype(np.uint8)


def texture_phase(params: EyeTwinParams, sample: TwinSample) -> float:
    """Stable phase offset so iris texture does not flicker between frames."""
    return float((params.seed % 97) + sample.fixation_id * 0.7)


def write_png(path: str | Path, rgb: np.ndarray) -> Path:
    """Write an RGB ``uint8`` array to PNG using only the standard library."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.ascontiguousarray(rgb, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("write_png expects an HxWx3 RGB array")
    h, w = image.shape[:2]
    raw = b"".join(b"\x00" + image[row].tobytes() for row in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    payload = b"\x89PNG\r\n\x1a\n"
    payload += chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
    payload += chunk(b"IDAT", zlib.compress(raw, 6))
    payload += chunk(b"IEND", b"")
    path.write_bytes(payload)
    return path


# --------------------------------------------------------------------------- #
# Dataset build
# --------------------------------------------------------------------------- #
@dataclass
class DatasetConfig:
    """Configuration for :class:`EyeDatasetBuilder`."""

    output_dir: Path = Path("artifacts/eye_dataset")
    sequences: int = 6
    duration_s: float = 6.0
    fps: float = 15.0
    seed: int = 42
    render_images: bool = True
    image_size: tuple[int, int] = (96, 160)
    splits: tuple[float, float, float] = (0.7, 0.15, 0.15)
    profiles: tuple[str, ...] = PRESETS
    domain_randomization: bool = True
    frames_per_second_saved: float | None = None  # None = every frame

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        if self.sequences < 3:
            raise ValueError(
                "sequences must be >= 3 so train/val/test splits are all populated"
            )
        if abs(sum(self.splits) - 1.0) > 1e-6:
            raise ValueError("splits must sum to 1.0")
        if any(s < 0 for s in self.splits):
            raise ValueError("splits must be non-negative")
        if not self.profiles:
            raise ValueError("profiles must not be empty")
        for name in self.profiles:
            if name not in PRESETS:
                raise ValueError(f"Unknown preset {name!r} in profiles")
        if self.frames_per_second_saved is not None and self.frames_per_second_saved <= 0:
            raise ValueError("frames_per_second_saved must be positive or None")


@dataclass
class DatasetManifest:
    """Summary of a generated dataset, written to ``manifest.json``."""

    root: str
    created_at: float
    seed: int
    sequences: int
    frames: int
    images: int
    split_counts: dict[str, int]
    profile_counts: dict[str, int]
    renderer: str
    image_size: list[int]
    labels_file: str
    labels_sha256: str
    metadata_dir: str
    simulated: bool = True
    photorealistic: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _randomize_params(
    base: EyeTwinParams, rng: np.random.Generator
) -> EyeTwinParams:
    """Domain randomization of appearance, pose, lighting and noise.

    Ranges are deliberately modest and documented in code; they are not
    calibrated to a population.
    """
    params = copy.deepcopy(base)
    skin = np.array(params.appearance.skin_color, dtype=np.float64)
    iris = np.array(params.appearance.iris_color, dtype=np.float64)
    skin *= rng.uniform(0.78, 1.18)
    iris *= rng.uniform(0.7, 1.25)
    params.appearance = replace(
        params.appearance,
        skin_color=tuple(int(v) for v in np.clip(skin, 20, 245)),
        iris_color=tuple(int(v) for v in np.clip(iris, 25, 220)),
        iris_radius_ratio=float(_clip(params.appearance.iris_radius_ratio * rng.uniform(0.9, 1.1), 0.24, 0.36)),
        pupil_ratio=float(_clip(params.appearance.pupil_ratio * rng.uniform(0.85, 1.15), 0.25, 0.55)),
        lid_open=float(_clip(params.appearance.lid_open * rng.uniform(0.9, 1.05), 0.6, 1.08)),
        lash_strength=float(_clip(rng.uniform(0.4, 1.0), 0.0, 1.0)),
    )
    params.lighting = replace(
        params.lighting,
        intensity=float(rng.uniform(0.55, 1.3)),
        ambient=float(rng.uniform(0.15, 0.5)),
        azimuth_deg=float(rng.uniform(-70, 70)),
        glare=float(rng.uniform(0.0, 0.3)),
        variation=float(rng.uniform(0.0, 0.2)),
    )
    params.camera = replace(
        params.camera,
        yaw_deg=float(rng.uniform(-15, 15)),
        pitch_deg=float(rng.uniform(-10, 10)),
        roll_deg=float(rng.uniform(-8, 8)),
        distance=float(rng.uniform(0.9, 1.25)),
        jitter_deg=float(rng.uniform(0.0, 1.5)),
    )
    m = params.motion
    params.motion = replace(
        m,
        fixation_duration_mean_s=float(_clip(m.fixation_duration_mean_s * rng.uniform(0.75, 1.3), 0.08, 1.5)),
        fixation_noise_deg=float(_clip(m.fixation_noise_deg * rng.uniform(0.7, 1.5), 0.05, 1.2)),
        saccade_amplitude_mean_deg=float(_clip(m.saccade_amplitude_mean_deg * rng.uniform(0.7, 1.4), 1.0, 30.0)),
        saccade_peak_velocity_deg_s=float(_clip(m.saccade_peak_velocity_deg_s * rng.uniform(0.8, 1.25), 80.0, 700.0)),
        blink_rate_per_min=float(_clip(m.blink_rate_per_min * rng.uniform(0.6, 1.5), 3.0, 45.0)),
        blink_duration_mean_s=float(_clip(m.blink_duration_mean_s * rng.uniform(0.8, 1.3), 0.08, 0.5)),
        head_yaw_amplitude_deg=float(rng.uniform(0.0, 5.0)),
        head_pitch_amplitude_deg=float(rng.uniform(0.0, 3.0)),
    )
    params.noise_sigma = float(_clip(params.noise_sigma * rng.uniform(0.5, 2.0), 0.0, 0.08))
    params.occlusion_probability = float(
        _clip(base.occlusion_probability + rng.uniform(0.0, 0.04), 0.0, 0.15)
    )
    return params


class EyeDatasetBuilder:
    """Builds a small image + labels dataset from twin sequences.

    Output layout::

        <root>/
          images/<split>/<sequence_id>/frame_000000.png
          labels.csv
          metadata/sequences.json
          metadata/<sequence_id>.json
          manifest.json

    No claim is made that these images improve real-world models; they are
    synthetic, schematic and intended for pipeline tests, demos and
    augmentation experiments.
    """

    def __init__(self, config: DatasetConfig) -> None:
        self.config = config
        self.renderer = ProceduralEyeRenderer(config.image_size)

    def build(self) -> DatasetManifest:
        cfg = self.config
        root = cfg.output_dir
        root.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(cfg.seed)
        n = cfg.sequences
        # Seeded split assignment independent of generation order.
        split_labels = np.array(["train"] * n, dtype=object)
        n_val = int(round(n * cfg.splits[1]))
        n_test = int(round(n * cfg.splits[2]))
        if n >= 3:
            # Ensure non-zero requested splits each get at least one sequence
            # and leave at least one for training.
            if cfg.splits[1] > 0:
                n_val = max(1, n_val)
            if cfg.splits[2] > 0:
                n_test = max(1, n_test)
            n_test = min(n_test, max(0, n - 1 - n_val))
            n_val = min(n_val, max(0, n - 1 - n_test))
        order = rng.permutation(n)
        if n >= 3:
            split_labels[order[:n_val]] = "val"
            split_labels[order[n_val:n_val + n_test]] = "test"
        label_rows: list[dict[str, Any]] = []
        metadata: list[dict[str, Any]] = []
        profile_counts: dict[str, int] = {}
        split_counts: dict[str, int] = {"train": 0, "val": 0, "test": 0}
        images_written = 0
        frames_written = 0
        frame_stride = 1
        if cfg.frames_per_second_saved is not None:
            frame_stride = max(1, int(round(cfg.fps / cfg.frames_per_second_saved)))

        for index in range(n):
            profile = cfg.profiles[index % len(cfg.profiles)]
            seed = int(rng.integers(0, 2**31 - 1))
            params = preset(profile)
            params.seed = seed
            if cfg.domain_randomization:
                params = _randomize_params(params, rng)
                params.seed = seed
            split = str(split_labels[index])
            sequence_id = f"{profile}_{index:03d}"
            generator = EyeTwinGenerator(params, seed=seed)
            sequence = generator.sequence(
                duration_s=cfg.duration_s,
                fps=cfg.fps,
                sequence_id=sequence_id,
                split=split,
            )
            profile_counts[profile] = profile_counts.get(profile, 0) + 1
            split_counts[split] = split_counts.get(split, 0) + 1
            rel_dir = Path("images") / split / sequence_id
            (root / rel_dir).mkdir(parents=True, exist_ok=True)

            for frame_index, sample in enumerate(sequence.samples):
                if frame_index % frame_stride != 0:
                    continue
                row = {
                    "sequence_id": sequence_id,
                    "split": split,
                    "profile": profile,
                    "seed": seed,
                    "frame": frame_index,
                    "filename": "",
                    **sample.to_row(),
                }
                if cfg.render_images:
                    # Per-frame camera jitter (domain randomization).
                    render_params = params
                    if cfg.domain_randomization and params.camera.jitter_deg > 0:
                        jitter = rng.normal(0.0, params.camera.jitter_deg, size=3)
                        render_params = replace(
                            params,
                            camera=replace(
                                params.camera,
                                yaw_deg=params.camera.yaw_deg + float(jitter[0]),
                                pitch_deg=params.camera.pitch_deg + float(jitter[1]),
                                roll_deg=params.camera.roll_deg + float(jitter[2]),
                            ),
                        )
                    image = self.renderer.render(sample, render_params)
                    filename = f"frame_{frame_index:06d}.png"
                    write_png(root / rel_dir / filename, image)
                    row["filename"] = (rel_dir / filename).as_posix()
                    images_written += 1
                label_rows.append(row)
                frames_written += 1

            meta = sequence.metadata(image_dir=rel_dir.as_posix())
            metadata.append(meta)
            (root / "metadata").mkdir(exist_ok=True)
            (root / "metadata" / f"{sequence_id}.json").write_text(
                json.dumps(meta, indent=2), encoding="utf-8"
            )

        labels_path = root / "labels.csv"
        fieldnames = list(label_rows[0].keys()) if label_rows else [
            "sequence_id", "split", "profile", "seed", "frame", "filename", "t"
        ]
        with labels_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(label_rows)
        (root / "metadata" / "sequences.json").write_text(
            json.dumps(
                {
                    "schema": "gazeos.eye_twin.dataset_meta.v1",
                    "simulated": True,
                    "photorealistic": False,
                    "sequences": metadata,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        digest = hashlib.sha256(labels_path.read_bytes()).hexdigest()
        manifest = DatasetManifest(
            root=str(root),
            created_at=time.time(),
            seed=cfg.seed,
            sequences=n,
            frames=frames_written,
            images=images_written,
            split_counts=split_counts,
            profile_counts=profile_counts,
            renderer=self.renderer.name,
            image_size=[cfg.image_size[0], cfg.image_size[1]],
            labels_file=labels_path.name,
            labels_sha256=digest,
            metadata_dir="metadata",
            notes=[
                "Synthetic schematic eye images; not photorealistic.",
                "Same seed reproduces sequences, splits and labels.",
                "Validate any model trained on this data against real recordings.",
            ],
        )
        (root / "manifest.json").write_text(
            json.dumps(manifest.to_dict(), indent=2), encoding="utf-8"
        )
        LOG.info(
            "Dataset written to %s (%d sequences, %d frames, %d images)",
            root, n, frames_written, images_written,
        )
        return manifest


# --------------------------------------------------------------------------- #
# Optional Blender integration (interface only)
# --------------------------------------------------------------------------- #
@dataclass
class BlenderIntegration:
    """Export a Blender job specification for a user-supplied eye rig.

    This class **does not render photorealistic eyes by itself**.  It writes a
    JSON job describing per-frame gaze/blink/pose states plus a starter
    ``render_eye_sequence.py`` script.  To produce images you need:

    1. Blender 3.6+ (4.x recommended) installed,
    2. a rigged eye/face model (``template.blend``) whose objects/armature
       the script can drive,
    3. adapting the starter script's ``apply_state`` function to your rig,
    4. running::

           blender -b template.blend -P render_eye_sequence.py -- job.json
    """

    blender_binary: str | None = None
    template_blend: str | Path | None = None

    @property
    def available(self) -> bool:
        """True only if a Blender executable is discoverable."""
        binary = self.blender_binary or shutil.which("blender")
        return bool(binary and Path(binary).exists())

    def export_job(self, sequence: TwinSequence, out_dir: str | Path) -> Path:
        """Write ``job.json`` and a starter Blender script; return the job path."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        job = {
            "schema": "gazeos.blender_job.v1",
            "sequence_id": sequence.sequence_id,
            "simulated": True,
            "photorealistic_requires_engine": True,
            "template_blend": str(self.template_blend) if self.template_blend else None,
            "fps": sequence.params.fps,
            "frames": [
                {
                    "frame": i + 1,
                    "t": s.t,
                    "gaze_x": s.gaze_x,
                    "gaze_y": s.gaze_y,
                    "gaze_x_deg": s.gaze_x_deg,
                    "gaze_y_deg": s.gaze_y_deg,
                    "blink": s.blink,
                    "head_yaw_deg": s.head_yaw_deg,
                    "head_pitch_deg": s.head_pitch_deg,
                    "head_roll_deg": s.head_roll_deg,
                    "lighting": s.lighting,
                    "occlusion": s.occlusion,
                }
                for i, s in enumerate(sequence.samples)
            ],
        }
        job_path = out / "job.json"
        job_path.write_text(json.dumps(job, indent=2), encoding="utf-8")
        (out / "render_eye_sequence.py").write_text(_BLENDER_SCRIPT, encoding="utf-8")
        (out / "README_blender.txt").write_text(
            "Blender integration (optional, manual)\n"
            "=====================================\n"
            "1. Install Blender 3.6+.\n"
            "2. Provide a rigged eye/face model and set its path below.\n"
            "3. Adapt apply_state() to your rig's bones/shape keys.\n"
            "4. Run:\n"
            "   blender -b template.blend -P render_eye_sequence.py -- job.json\n"
            "5. Output PNG frames are written to ./render/ next to job.json.\n\n"
            "The procedural renderer is schematic; only a 3D engine with a\n"
            "suitable asset can produce physically plausible eye images.\n",
            encoding="utf-8",
        )
        LOG.info("Blender job exported to %s (Blender available=%s)", job_path, self.available)
        return job_path


_BLENDER_SCRIPT = '''"""Starter Blender script generated by EyeTwin.

Adapt apply_state() to your own rig. This script intentionally does not
ship a face/eye model; photorealism requires a suitable asset and materials.

Usage:
    blender -b template.blend -P render_eye_sequence.py -- job.json
"""
import json
import math
import os
import sys

import bpy

JOB_PATH = sys.argv[sys.argv.index("--") + 1]
with open(JOB_PATH, "r", encoding="utf-8") as fh:
    job = json.load(fh)

scene = bpy.context.scene
scene.render.fps = int(round(job.get("fps", 30)))
out_dir = os.path.join(os.path.dirname(os.path.abspath(JOB_PATH)), "render")
os.makedirs(out_dir, exist_ok=True)

# Example handles -- replace with the names in your template.blend.
eye_l = bpy.data.objects.get("Eye.L")
eye_r = bpy.data.objects.get("Eye.R")
head = bpy.data.objects.get("Head")


def apply_state(frame):
    """Map one twin state to the rig.

    Suggested mapping:
      * rotate eye objects by gaze_x_deg/gaze_y_deg,
      * drive eyelid shape keys/bones with blink amount,
      * rotate the head object by head yaw/pitch/roll.
    """
    gx = math.radians(frame["gaze_x_deg"])
    gy = math.radians(frame["gaze_y_deg"])
    for eye in (eye_l, eye_r):
        if eye is not None:
            eye.rotation_mode = "XYZ"
            eye.rotation_euler = (gy, 0.0, gx)
    if head is not None:
        head.rotation_mode = "XYZ"
        head.rotation_euler = (
            math.radians(frame["head_pitch_deg"]),
            math.radians(frame["head_roll_deg"]),
            math.radians(frame["head_yaw_deg"]),
        )
    # TODO: eyelid blink and occlusion via shape keys/materials.


for item in job["frames"]:
    scene.frame_set(item["frame"])
    apply_state(item)
    scene.render.filepath = os.path.join(out_dir, f"frame_{item['frame']:06d}.png")
    bpy.ops.render.render(write_still=True)

print("Rendered", len(job["frames"]), "frames to", out_dir)
'''


# --------------------------------------------------------------------------- #
# Preview rendering
# --------------------------------------------------------------------------- #
def render_preview(
    out_dir: str | Path,
    profile: str = "alert",
    seed: int = 1,
    frames: int = 5,
    size: tuple[int, int] = (96, 160),
    duration_s: float = 4.0,
) -> Path:
    """Render a contact sheet and returns its path.

    Useful to inspect what the schematic renderer actually produces before
    building a dataset.  The sheet is ordered by time and the label JSON next
    to it describes each tile.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    params = preset(profile)
    params.seed = seed
    sequence = EyeTwinGenerator(params, seed=seed).sequence(
        duration_s=duration_s, sequence_id=f"{profile}_preview", split="preview"
    )
    renderer = ProceduralEyeRenderer(size)
    n = len(sequence.samples)
    indices = np.linspace(0, n - 1, num=max(2, frames)).astype(int)
    tiles = []
    meta = []
    for idx in indices:
        sample = sequence.samples[int(idx)]
        image = renderer.render(sample, params)
        write_png(out / f"frame_{int(idx):05d}.png", image)
        tiles.append(image)
        meta.append(
            {
                "frame": int(idx),
                "t": sample.t,
                "gaze_x_deg": round(sample.gaze_x_deg, 2),
                "gaze_y_deg": round(sample.gaze_y_deg, 2),
                "blink": round(sample.blink, 3),
                "state": sample.state,
            }
        )
    sheet = np.concatenate(tiles, axis=1)
    sheet_path = out / "contact_sheet.png"
    write_png(sheet_path, sheet)
    (out / "preview_labels.json").write_text(
        json.dumps(
            {
                "note": "Schematic rendering, not photorealistic.",
                "profile": profile,
                "seed": seed,
                "size": list(size),
                "frames": meta,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return sheet_path


# --------------------------------------------------------------------------- #
# Closed-loop validation through gaze_core
# --------------------------------------------------------------------------- #
def _correlation(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _affine_rmse(ground_truth: np.ndarray, estimate: np.ndarray) -> float | None:
    """RMSE after a least-squares affine fit (calibration-adjusted)."""
    if len(ground_truth) < 3:
        return None
    A = np.vstack([ground_truth, np.ones_like(ground_truth)]).T
    try:
        coefficients, *_ = np.linalg.lstsq(A, estimate, rcond=None)
    except np.linalg.LinAlgError:
        return None
    return float(np.sqrt(np.mean((A @ coefficients - estimate) ** 2)))


def _episodes(flags: Sequence[bool], timestamps: Sequence[float]) -> list[tuple[float, float]]:
    """Return (start, end) pairs for contiguous True runs."""
    episodes: list[tuple[float, float]] = []
    start: float | None = None
    for flag, t in zip(flags, timestamps, strict=False):
        if flag and start is None:
            start = t
        elif not flag and start is not None:
            # End at the first frame on the other side of the threshold, so a
            # two-frame blink is not reported as one frame long.
            episodes.append((start, t))
            start = None
    if start is not None:
        episodes.append((start, timestamps[-1]))
    return episodes


def validate_against_core(
    profile: str = "alert",
    seconds: float = 20.0,
    fps: float = 30.0,
    seed: int = 7,
    config: GazeConfig | None = None,
) -> dict[str, Any]:
    """Replay twin signals through ``gaze_core`` and compare with ground truth.

    This is a **closed-loop consistency check**, not an accuracy benchmark on
    humans.  It verifies that estimation, blink/fixation detection and event
    wiring behave sensibly on known signals and that the twin's labels and the
    analytics agree.  Real-world accuracy depends on camera, users, head pose
    and calibration and is not established here.
    """
    cfg = config or make_config("research")
    params = preset(profile)
    params.seed = seed
    sequence = EyeTwinGenerator(params, seed=seed).sequence(
        duration_s=seconds, fps=fps, sequence_id=f"{profile}_validation"
    )
    pipeline = GazePipeline(cfg)
    detected_blinks: list[float] = []
    detected_fixations: list[float] = []
    pipeline.bus.subscribe(
        lambda e: detected_blinks.append(float(e.data.get("duration_ms", 0.0)))
        if e.type == EventType.BLINK else None
    )
    pipeline.bus.subscribe(
        lambda e: detected_fixations.append(float(e.data.get("duration_ms", 0.0)))
        if e.type == EventType.FIXATION else None
    )

    gt_x: list[float] = []
    gt_y: list[float] = []
    est_x: list[float] = []
    est_y: list[float] = []
    est_conf: list[float] = []
    closed_flags: list[bool] = []
    timestamps: list[float] = []
    for sample in sequence.samples:
        report = pipeline.process(sample_to_face(sample, params))
        timestamps.append(sample.t)
        # Ground-truth "closed" uses the same openness threshold as the core.
        closed_flags.append((1.0 - sample.blink) < cfg.blink_open_threshold)
        if report.sample is not None:
            gt_x.append(sample.gaze_x)
            gt_y.append(sample.gaze_y)
            est_x.append(report.sample.gaze_x)
            est_y.append(report.sample.gaze_y)
            est_conf.append(report.sample.confidence)

    gt_blink_episodes = [
        (a, b) for a, b in _episodes(closed_flags, timestamps)
        if (b - a) * 1000.0 >= cfg.blink_min_ms
    ]
    gt_fixation_episodes = [
        (a, b) for a, b in _episodes(
            [s.state == "fixation" for s in sequence.samples], timestamps
        )
        if (b - a) * 1000.0 >= cfg.fixation_min_ms
    ]
    gx = np.asarray(gt_x)
    gy = np.asarray(gt_y)
    ex = np.asarray(est_x)
    ey = np.asarray(est_y)
    mean_conf = float(np.mean(est_conf)) if est_conf else 0.0
    return {
        "profile": profile,
        "seed": seed,
        "seconds": seconds,
        "fps": fps,
        "frames": len(sequence.samples),
        "gaze": {
            "correlation_x": _correlation(gx, ex),
            "correlation_y": _correlation(gy, ey),
            "rmse_normalized_raw": {
                "x": float(round(float(np.sqrt(np.mean((gx - ex) ** 2))), 5)),
                "y": float(round(float(np.sqrt(np.mean((gy - ey) ** 2))), 5)),
            },
            "rmse_normalized_after_affine_fit": {
                "x": round(_affine_rmse(gx, ex) or 0.0, 5),
                "y": round(_affine_rmse(gy, ey) or 0.0, 5),
            },
            "mean_confidence": round(mean_conf, 4),
        },
        "blinks": {
            "ground_truth_count": len(gt_blink_episodes),
            "detected_count": len(detected_blinks),
            "ground_truth_mean_ms": round(
                statistics.fmean([(b - a) * 1000.0 for a, b in gt_blink_episodes]), 1
            ) if gt_blink_episodes else None,
            "detected_mean_ms": round(statistics.fmean(detected_blinks), 1) if detected_blinks else None,
        },
        "fixations": {
            "ground_truth_count": len(gt_fixation_episodes),
            "detected_count": len(detected_fixations),
            "ground_truth_mean_ms": round(
                statistics.fmean([(b - a) * 1000.0 for a, b in gt_fixation_episodes]), 1
            ) if gt_fixation_episodes else None,
            "detected_mean_ms": round(statistics.fmean(detected_fixations), 1) if detected_fixations else None,
        },
        "caveats": [
            "Closed-loop synthetic check only; not an accuracy claim for humans.",
            "Gaze RMSE is in normalized eye-space units, not degrees or pixels.",
            "Blink ground truth uses the configured openness threshold.",
            "Fixation labels come from the generator's state machine, not a human annotator.",
        ],
    }


# --------------------------------------------------------------------------- #
# Self tests
# --------------------------------------------------------------------------- #
def _run_selftest() -> int:
    """Dependency-free checks for generation, labels, rendering and datasets."""
    import tempfile

    checks: list[tuple[str, bool, str]] = []

    def check(name: str, condition: bool, detail: str = "") -> None:
        checks.append((name, bool(condition), detail))

    # Reproducibility
    seq_a = EyeTwinGenerator(preset("alert"), seed=11).sequence(3.0, fps=30, sequence_id="a")
    seq_b = EyeTwinGenerator(preset("alert"), seed=11).sequence(3.0, fps=30, sequence_id="b")
    seq_c = EyeTwinGenerator(preset("alert"), seed=12).sequence(3.0, fps=30, sequence_id="c")
    same = [s.gaze_x for s in seq_a.samples] == [s.gaze_x for s in seq_b.samples]
    check("seed reproducibility", same, f"n={len(seq_a.samples)}")
    different = [s.gaze_x for s in seq_a.samples] != [s.gaze_x for s in seq_c.samples]
    check("different seeds differ", different)

    # Temporal plausibility (longer sequence so blink count is meaningful)
    blink_seq = EyeTwinGenerator(preset("alert"), seed=5).sequence(20.0, fps=30, sequence_id="blinks")
    stats = blink_seq.episodes()
    blurbs = f"blinks={stats['blinks']} fix={stats['fixations']} sacc={stats['saccades']}"
    check("blinks generated", stats["blinks"] >= 1, blurbs)
    check("fixations generated", stats["fixations"] >= 1, blurbs)
    speeds = [s.gaze_speed_deg_s for s in seq_a.samples if s.state == "saccade"]
    check(
        "saccade speeds plausible",
        bool(speeds) and 20.0 < max(speeds) < 3000.0,
        f"max={max(speeds):.0f} deg/s" if speeds else "none",
    )
    check(
        "blink amount in range",
        all(0.0 <= s.blink <= 1.0 for s in seq_a.samples),
    )

    # Serialization
    params = preset("fatigued")
    params.seed = 5
    roundtrip = EyeTwinParams.from_dict(json.loads(json.dumps(params.to_dict())))
    check(
        "params JSON round trip",
        roundtrip.motion.blink_rate_per_min == params.motion.blink_rate_per_min
        and roundtrip.appearance.skin_color == params.appearance.skin_color,
    )

    # Core bridge and sign conventions
    sample = TwinSample(
        t=0.0, gaze_x=0.5, gaze_y=0.4, gaze_x_deg=7.5, gaze_y_deg=3.6,
        state="fixation", blink=0.0, pupil_mm=3.5, fixation_id=1,
        saccade_id=0, blink_id=0, gaze_speed_deg_s=0.0, head_yaw_deg=0.0,
        head_pitch_deg=0.0, head_roll_deg=0.0, lighting=1.0, occlusion=0.0,
        is_occluded=False, is_blink=False,
    )
    face = sample_to_face(sample, preset("alert"))
    check(
        "twin->core gaze signs",
        face.left_eye.horizontal_ratio() > 0 and face.left_eye.vertical_ratio() > 0,
        f"h={face.left_eye.horizontal_ratio():.2f} v={face.left_eye.vertical_ratio():.2f}",
    )
    check("twin blink label", face.mean_openness > 0.9, f"openness={face.mean_openness:.2f}")

    # Renderer
    renderer = ProceduralEyeRenderer((48, 80))
    open_image = renderer.render(seq_a.samples[0], seq_a.params)
    closed_sample = replace(seq_a.samples[0], blink=1.0)
    closed_image = renderer.render(closed_sample, seq_a.params)
    check(
        "renderer shape/dtype",
        open_image.shape == (48, 80, 3) and open_image.dtype == np.uint8,
    )
    check(
        "blink changes rendering",
        float(np.mean(np.abs(open_image.astype(float) - closed_image.astype(float)))) > 1.0,
    )

    # PNG + dataset
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        png_path = write_png(tmp_path / "x.png", open_image)
        check(
            "png writer",
            png_path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n",
            f"{png_path.stat().st_size} bytes",
        )
        config = DatasetConfig(
            output_dir=tmp_path / "ds",
            sequences=6,
            duration_s=2.0,
            fps=10.0,
            seed=99,
            image_size=(48, 80),
        )
        manifest = EyeDatasetBuilder(config).build()
        check(
            "dataset splits",
            sum(manifest.split_counts.values()) == 6
            and manifest.split_counts["train"] >= 3
            and manifest.split_counts["val"] >= 1
            and manifest.split_counts["test"] >= 1,
            str(manifest.split_counts),
        )
        check(
            "dataset files",
            (tmp_path / "ds" / "labels.csv").exists()
            and (tmp_path / "ds" / "manifest.json").exists()
            and (tmp_path / "ds" / "metadata" / "sequences.json").exists()
            and manifest.images > 0,
            f"images={manifest.images}",
        )
        rows = list(csv.DictReader((tmp_path / "ds" / "labels.csv").open(encoding="utf-8")))
        check("labels rows == frames", len(rows) == manifest.frames, f"rows={len(rows)}")
        check(
            "metadata flags simulated",
            json.loads((tmp_path / "ds" / "manifest.json").read_text())["photorealistic"] is False,
        )
        manifest_b = EyeDatasetBuilder(config).build()
        check(
            "dataset reproducible labels",
            manifest.labels_sha256 == manifest_b.labels_sha256,
            manifest.labels_sha256[:12],
        )
        preview = render_preview(tmp_path / "preview", frames=4)
        check("preview contact sheet", preview.exists() and preview.stat().st_size > 100)

    # Closed-loop validation runs and is serialisable.
    report = validate_against_core(profile="alert", seconds=8.0, fps=30.0, seed=3)
    json.dumps(report)
    check(
        "closed-loop validation",
        report["gaze"]["correlation_x"] is not None
        and report["blinks"]["ground_truth_count"] >= 0,
        f"r_x={report['gaze']['correlation_x']:.2f}"
        if report["gaze"]["correlation_x"] is not None else "r_x=n/a",
    )

    passed = sum(1 for _, ok, _ in checks if ok)
    for name, ok, detail in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" ({detail})" if detail else ""))
    print(f"\n{passed}/{len(checks)} checks passed")
    return 0 if passed == len(checks) else 1


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="EyeTwin: seeded synthetic eye movement + schematic datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("presets", help="List eye-twin presets.")

    generate = sub.add_parser("generate", help="Build an image + labels dataset.")
    generate.add_argument("--out", default="artifacts/eye_dataset")
    generate.add_argument("--sequences", type=int, default=6)
    generate.add_argument("--duration", type=float, default=6.0)
    generate.add_argument("--fps", type=float, default=15.0)
    generate.add_argument("--seed", type=int, default=42)
    generate.add_argument("--profiles", nargs="+", default=list(PRESETS), choices=PRESETS)
    generate.add_argument("--splits", default="0.7,0.15,0.15")
    generate.add_argument("--no-images", action="store_true")
    generate.add_argument("--no-domain-randomization", action="store_true")
    generate.add_argument("--image-height", type=int, default=96)
    generate.add_argument("--image-width", type=int, default=160)
    generate.add_argument(
        "--frames-per-second", type=float, default=None,
        help="Subsample saved frames (default: every frame).",
    )

    preview = sub.add_parser("preview", help="Render a contact sheet.")
    preview.add_argument("--out", default="artifacts/eye_preview")
    preview.add_argument("--profile", default="alert", choices=PRESETS)
    preview.add_argument("--seed", type=int, default=1)
    preview.add_argument("--frames", type=int, default=5)

    validate = sub.add_parser("validate", help="Run the twin through gaze_core.")
    validate.add_argument("--profile", default="alert", choices=PRESETS)
    validate.add_argument("--seconds", type=float, default=20.0)
    validate.add_argument("--fps", type=float, default=30.0)
    validate.add_argument("--seed", type=int, default=7)
    validate.add_argument(
        "--config-profile", default="research",
        choices=("generic", "driver_monitoring", "learning_analytics", "screen_fatigue", "research"),
    )

    blender = sub.add_parser("blender-job", help="Export a Blender job specification.")
    blender.add_argument("--out", default="artifacts/blender_job")
    blender.add_argument("--profile", default="driving", choices=PRESETS)
    blender.add_argument("--seed", type=int, default=3)
    blender.add_argument("--seconds", type=float, default=4.0)
    blender.add_argument("--blender", default=None, help="Optional Blender executable path.")
    blender.add_argument("--template", default=None, help="Optional template .blend path.")

    sub.add_parser("selftest", help="Run dependency-free self tests.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if args.command == "presets":
        for name in PRESETS:
            p = preset(name)
            print(
                f"{name:<10} blink={p.motion.blink_rate_per_min:>4.0f}/min  "
                f"blink_ms={p.motion.blink_duration_mean_s * 1000:>4.0f}  "
                f"fix_ms={p.motion.fixation_duration_mean_s * 1000:>4.0f}  "
                f"lid_open={p.appearance.lid_open:.2f}"
            )
        return 0
    if args.command == "generate":
        try:
            splits = tuple(float(v) for v in args.splits.split(","))
            if len(splits) != 3:
                raise ValueError
        except ValueError:
            print("--splits must be three comma-separated numbers, e.g. 0.7,0.15,0.15", file=sys.stderr)
            return 2
        config = DatasetConfig(
            output_dir=Path(args.out),
            sequences=args.sequences,
            duration_s=args.duration,
            fps=args.fps,
            seed=args.seed,
            render_images=not args.no_images,
            image_size=(args.image_height, args.image_width),
            splits=splits,  # type: ignore[arg-type]
            profiles=tuple(args.profiles),
            domain_randomization=not args.no_domain_randomization,
            frames_per_second_saved=args.frames_per_second,
        )
        try:
            manifest = EyeDatasetBuilder(config).build()
        except (OSError, ValueError) as exc:
            print(f"Dataset build failed: {exc}", file=sys.stderr)
            return 2
        print(json.dumps(manifest.to_dict(), indent=2))
        print(f"\nDataset ready at {manifest.root}")
        print("Reminder: synthetic schematic images; validate on real data.")
        return 0
    if args.command == "preview":
        path = render_preview(args.out, args.profile, args.seed, args.frames)
        print(f"Contact sheet: {path}")
        return 0
    if args.command == "validate":
        report = validate_against_core(
            args.profile, args.seconds, args.fps, args.seed,
            make_config(args.config_profile),
        )
        print(json.dumps(report, indent=2))
        print("\nThis is a closed-loop synthetic check, not a human accuracy claim.")
        return 0
    if args.command == "blender-job":
        params = preset(args.profile)
        params.seed = args.seed
        sequence = EyeTwinGenerator(params, seed=args.seed).sequence(
            duration_s=args.seconds, sequence_id=f"{args.profile}_blender"
        )
        integration = BlenderIntegration(args.blender, args.template)
        path = integration.export_job(sequence, args.out)
        print(f"Blender job: {path}")
        print(f"Blender executable detected: {integration.available}")
        print("Adapt render_eye_sequence.py to your rig, then run Blender manually.")
        return 0
    if args.command == "selftest":
        return _run_selftest()
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
