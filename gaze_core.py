"""GazeOS core: a generic, reusable eye-movement intelligence library.

``gaze_core`` turns face landmarks into structured, timestamped eye-movement
signals (gaze, blinks, fixations, saccades), aggregates them into windowed
temporal features, compares them against a per-session baseline, and emits
anomaly events.  It is deliberately application-agnostic: application profiles
(driver monitoring, learning analytics, screen fatigue, research) only change
thresholds and rules.

Design principles
-----------------
* **Local by default.** Frames are processed in memory.  Nothing is uploaded.
  Frames are never written to disk unless the caller explicitly provides a
  frame dump directory *and* opts in to identifiable storage.
* **Pluggable.** Landmark backends, frame sources, calibration models and
  event subscribers are interfaces.  MediaPipe is the default real backend;
  OpenCV is only needed for webcam/video IO and previews.
* **Computable on CPU.** All analytics are rule/statistics based and clearly
  documented as heuristics, not medically validated measurements.
* **Observable.** Every frame yields a :class:`GazeFrame`; additionally an
  :class:`EventBus` delivers edge-triggered events (blinks, fixations,
  saccades, anomalies, face loss).

Coordinate conventions
----------------------
* Image/landmark coordinates are MediaPipe-style normalized ``(x, y)`` in
  ``[0, 1]`` with ``y`` pointing **down**.
* ``GazeSample.gaze_x`` / ``gaze_y`` are normalized eye-space estimates in
  ``[-1, 1]``.  ``gaze_x`` increases toward the **right side of the camera
  image**; ``gaze_y`` increases **upward**.  These are *not* screen
  coordinates until a calibration model is applied.
* ``ScreenPoint`` values are normalized screen coordinates ``[0, 1]``.

Dependencies
------------
Required: ``numpy``.  Optional: ``opencv-python`` (camera/video sources and
previews) and ``mediapipe`` (real face landmarks).  The ``demo`` and
``selftest`` commands run with numpy alone.

Safety
------
This is experimental software.  It does not diagnose drowsiness, disease or
disability, and it is not a safety-certified driver-monitoring or medical
device.  See the repository README for limitations.

Run ``python gaze_core.py --help`` for the command line interface.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import statistics
import sys
import threading
import time
from collections import Counter, deque
from dataclasses import asdict, dataclass, field, fields, replace
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np

LOG = logging.getLogger("gazeos.core")

__all__ = [
    "Anomaly",
    "AnomalyDetector",
    "BaselineTracker",
    "BlinkDetector",
    "BlinkUpdate",
    "CalibrationModel",
    "CameraSource",
    "CameraUnavailableError",
    "DependencyMissingError",
    "EventBus",
    "EventType",
    "EyeGeometry",
    "EyeLandmarkDetector",
    "FaceObservation",
    "FixationDetector",
    "FramePacket",
    "FrameSource",
    "GazeCalibrator",
    "GazeConfig",
    "GazeDynamics",
    "GazeEvent",
    "GazeEstimator",
    "GazeFrame",
    "GazePipeline",
    "GazeSample",
    "GazeSession",
    "GazeSessionLogger",
    "MediaPipeEyeLandmarkDetector",
    "ModelNotFoundError",
    "ProceduralFaceSimulator",
    "ScreenPoint",
    "SessionSummary",
    "TemporalFeatureExtractor",
    "TemporalFeatures",
    "VideoFileSource",
    "make_config",
    "profile_names",
]

PROFILE_NAMES: tuple[str, ...] = (
    "generic",
    "driver_monitoring",
    "learning_analytics",
    "screen_fatigue",
    "research",
)

FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
DEFAULT_MODEL_PATH = Path("models") / "face_landmarker.task"


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #
class GazeOSError(RuntimeError):
    """Base class for recoverable GazeOS errors."""


class DependencyMissingError(GazeOSError):
    """An optional dependency is required for the requested operation."""


class ModelNotFoundError(GazeOSError):
    """A required model file is missing or unreadable."""


class CameraUnavailableError(GazeOSError):
    """A camera could not be opened (permissions, index or driver issue)."""


class CalibrationError(GazeOSError):
    """Calibration data is missing, malformed or degenerate."""


# --------------------------------------------------------------------------- #
# Small math helpers (image-space, y points down)
# --------------------------------------------------------------------------- #
Vec2 = tuple[float, float]


def _clamp(value: float, low: float, high: float) -> float:
    return low if value < low else high if value > high else value


def _dist(a: Vec2, b: Vec2) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _sub(a: Vec2, b: Vec2) -> Vec2:
    return (a[0] - b[0], a[1] - b[1])


def _mid(a: Vec2, b: Vec2) -> Vec2:
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def _dot(a: Vec2, b: Vec2) -> float:
    return a[0] * b[0] + a[1] * b[1]


def _unit(a: Vec2) -> Vec2:
    length = math.hypot(a[0], a[1])
    if length < 1e-9:
        return (1.0, 0.0)
    return (a[0] / length, a[1] / length)


def _round(value: float | None, digits: int = 4) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _jsonable(value: Any) -> Any:
    """Convert numpy scalars/arrays and dataclasses into JSON-safe values."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    return value


# --------------------------------------------------------------------------- #
# Configuration and application profiles
# --------------------------------------------------------------------------- #
@dataclass
class GazeConfig:
    """All tunable parameters for the core pipeline.

    Use :func:`make_config` to build one from a named application profile, or
    construct directly and pass it to :class:`GazePipeline`.

    Thresholds are *heuristics*.  They were chosen to be defensible for a
    prototype, not to match clinical or industry standards.  Eye openness in
    particular varies strongly between cameras and users -- tune
    ``blink_open_threshold`` against your own recordings.
    """

    profile: str = "generic"

    # Detection
    max_faces: int = 1
    min_face_confidence: float = 0.30
    detect_every_n_frames: int = 1
    face_lost_grace_s: float = 1.0

    # Gaze mapping (eye-in-head geometry -> normalized gaze)
    gaze_gain_x: float = 1.35
    gaze_gain_y: float = 1.15
    head_yaw_compensation: float = 0.010  # gaze units per degree (heuristic)
    head_pitch_compensation: float = 0.008
    gaze_smoothing: float = 0.65  # EMA alpha; higher = less smoothing
    gaze_saccade_bypass: float = 0.22  # jump size that skips smoothing

    # Blink detection
    blink_open_threshold: float = 0.24
    blink_close_hysteresis: float = 0.04
    blink_min_ms: float = 40.0
    blink_max_ms: float = 800.0
    prolonged_closure_ms: float = 1200.0

    # Fixations / saccades
    saccade_threshold_deg_s: float = 90.0
    fixation_dispersion_deg: float = 2.5
    fixation_min_ms: float = 100.0
    degrees_per_norm_unit: float = 30.0
    gaze_max_dt_s: float = 0.25

    # Windows / baseline
    window_s: float = 30.0
    baseline_warmup_s: float = 5.0
    baseline_min_samples: int = 60
    baseline_max_samples: int = 1800
    anomaly_z: float = 3.5

    # Anomaly thresholds
    perclos_warn: float = 0.30
    perclos_critical: float = 0.45
    blink_rate_low: float = 6.0
    blink_rate_high: float = 35.0
    blink_duration_warn_ms: float = 500.0
    gaze_stability_min: float = 0.25
    fixation_long_ms: float = 2500.0
    head_yaw_max_deg: float = 40.0
    head_pitch_max_deg: float = 30.0
    on_target_min_ratio: float = 0.60
    min_blinks_for_rate_rules: int = 5

    # Attention zone in raw gaze space: (x_min, x_max, y_min, y_max).
    # ``None`` disables the on-target proxy.
    attention_zone: tuple[float, float, float, float] | None = None

    # Camera defaults (used by CameraSource helpers / CLI)
    camera_index: int = 0
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: float = 30.0
    mirror_preview: bool = True

    # Privacy: raw frames are only stored on explicit, separate opt-in.
    store_identifiable: bool = False

    # Outcome/debug limits
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        if self.profile not in PROFILE_NAMES:
            raise ValueError(
                f"Unknown profile {self.profile!r}; choose from {PROFILE_NAMES}"
            )
        if not 0.0 < self.blink_open_threshold < 1.0:
            raise ValueError("blink_open_threshold must be in (0, 1)")
        if self.window_s <= 0 or self.baseline_warmup_s < 0:
            raise ValueError("window_s must be > 0 and baseline_warmup_s >= 0")
        if self.gaze_gain_x <= 0 or self.gaze_gain_y <= 0:
            raise ValueError("gaze gains must be positive")
        if not 0.0 < self.gaze_smoothing <= 1.0:
            raise ValueError("gaze_smoothing must be in (0, 1]")
        if self.detect_every_n_frames < 1:
            raise ValueError("detect_every_n_frames must be >= 1")
        if self.attention_zone is not None:
            x0, x1, y0, y1 = self.attention_zone
            if x0 >= x1 or y0 >= y1:
                raise ValueError("attention_zone must be (x_min, x_max, y_min, y_max)")

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


_PROFILE_OVERRIDES: dict[str, dict[str, Any]] = {
    # Balanced defaults for general webcam use.
    "generic": {},
    # Research: keep more history, fewer automatic conclusions.
    "research": {
        "window_s": 60.0,
        "baseline_min_samples": 90,
        "anomaly_z": 4.0,
        "min_face_confidence": 0.20,
        "min_blinks_for_rate_rules": 8,
    },
    # Driver monitoring: short window, drowsiness-focused, road-ahead zone.
    # The "off-target" proxy is camera-space only and not road truth.
    "driver_monitoring": {
        "window_s": 15.0,
        "baseline_warmup_s": 4.0,
        "perclos_warn": 0.25,
        "perclos_critical": 0.40,
        "blink_max_ms": 1000.0,
        "prolonged_closure_ms": 1000.0,
        "blink_rate_high": 30.0,
        "blink_duration_warn_ms": 450.0,
        "head_yaw_max_deg": 35.0,
        "head_pitch_max_deg": 25.0,
        "attention_zone": (-0.55, 0.55, -0.7, 0.7),
        "on_target_min_ratio": 0.60,
    },
    # Learning analytics: fixation/blink descriptives, no fatigue alarmism.
    "learning_analytics": {
        "window_s": 45.0,
        "fixation_min_ms": 120.0,
        "perclos_warn": 0.40,
        "perclos_critical": 0.60,
        "blink_rate_high": 40.0,
        "fixation_long_ms": 3000.0,
        "attention_zone": (-0.85, 0.85, -0.85, 0.85),
        "on_target_min_ratio": 0.70,
        "min_blinks_for_rate_rules": 3,
    },
    # Screen fatigue: blink suppression and prolonged fixation are the signals.
    "screen_fatigue": {
        "window_s": 30.0,
        "blink_rate_low": 8.0,
        "blink_rate_high": 30.0,
        "blink_duration_warn_ms": 600.0,
        "gaze_stability_min": 0.20,
        "fixation_long_ms": 2200.0,
        "min_blinks_for_rate_rules": 4,
    },
}


def profile_names() -> tuple[str, ...]:
    """Return the names of the built-in application profiles."""
    return PROFILE_NAMES


def make_config(profile: str = "research", **overrides: Any) -> GazeConfig:
    """Build a :class:`GazeConfig` from a named profile plus overrides.

    Parameters
    ----------
    profile:
        One of :func:`profile_names`.
    overrides:
        Any ``GazeConfig`` field.  Unknown fields raise ``ValueError`` so that
        typos do not silently change behaviour.
    """
    if profile not in _PROFILE_OVERRIDES:
        raise ValueError(f"Unknown profile {profile!r}; choose from {PROFILE_NAMES}")
    valid = {f.name for f in fields(GazeConfig)}
    unknown = set(overrides) - valid
    if unknown:
        raise ValueError(f"Unknown config field(s): {sorted(unknown)}")
    base = {
        "profile": profile,
        **_PROFILE_OVERRIDES[profile],
        **overrides,
    }
    return GazeConfig(**base)


def load_config(path: str | Path, profile: str = "research") -> GazeConfig:
    """Load a JSON config file (flat keys) merged onto a profile."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Config JSON must be an object of field -> value")
    data.setdefault("profile", profile)
    return make_config(str(data.pop("profile")), **data)


# --------------------------------------------------------------------------- #
# Data contracts
# --------------------------------------------------------------------------- #
class EventType(str, Enum):
    """Edge-triggered event types published by the pipeline."""

    SESSION_STARTED = "session_started"
    SESSION_ENDED = "session_ended"
    FACE_FOUND = "face_found"
    FACE_LOST = "face_lost"
    BLINK = "blink"
    PROLONGED_CLOSURE = "prolonged_closure"
    FIXATION = "fixation"
    SACCADE = "saccade"
    DROWSINESS = "drowsiness"
    ANOMALY = "anomaly"
    FRAME_ERROR = "frame_error"
    DEGRADED = "degraded"


@dataclass(frozen=True)
class EyeGeometry:
    """One eye described by five landmarks in normalized image space.

    ``inner``/``outer`` are the eye corners, ``upper``/``lower`` the eyelid
    points and ``iris`` the iris centre (mean of the iris landmarks).
    """

    inner: Vec2
    outer: Vec2
    upper: Vec2
    lower: Vec2
    iris: Vec2
    iris_radius: float = 0.0
    side: str = "left"

    @property
    def width(self) -> float:
        return _dist(self.inner, self.outer)

    @property
    def lid_gap(self) -> float:
        return _dist(self.upper, self.lower)

    @property
    def center(self) -> Vec2:
        return _mid(self.inner, self.outer)

    def _axes(self) -> tuple[Vec2, Vec2]:
        """Return (horizontal, vertical-up) unit axes in image space."""
        u = _unit(_sub(self.outer, self.inner))
        if u[0] < 0:  # normalize so +u always points image-right
            u = (-u[0], -u[1])
        v = (u[1], -u[0])  # image-space upward normal
        return u, v

    def horizontal_ratio(self) -> float:
        """Iris displacement along the eye axis, in half eye-widths."""
        u, _ = self._axes()
        half = max(self.width / 2.0, 1e-6)
        return _dot(_sub(self.iris, self.center), u) / half

    def vertical_ratio(self) -> float:
        """Iris displacement toward the upper lid, in half lid-gap units."""
        _, v = self._axes()
        lid_mid = _mid(self.upper, self.lower)
        half = max(self.lid_gap / 2.0, 1e-6)
        return _dot(_sub(self.iris, lid_mid), v) / half

    def openness(self) -> float:
        """Lid gap relative to a typical open-eye ratio, clamped to [0, 1]."""
        if self.width < 1e-6:
            return 0.0
        return _clamp(self.lid_gap / self.width / 0.32, 0.0, 1.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "inner": [_round(self.inner[0]), _round(self.inner[1])],
            "outer": [_round(self.outer[0]), _round(self.outer[1])],
            "upper": [_round(self.upper[0]), _round(self.upper[1])],
            "lower": [_round(self.lower[0]), _round(self.lower[1])],
            "iris": [_round(self.iris[0]), _round(self.iris[1])],
            "iris_radius": _round(self.iris_radius),
            "side": self.side,
        }


@dataclass(frozen=True)
class FaceObservation:
    """A timestamped, backend-agnostic observation of one face.

    Any detector (MediaPipe, the digital twin, or a custom backend) produces
    this contract, so the rest of the pipeline never depends on a specific
    landmark library.
    """

    timestamp: float
    left_eye: EyeGeometry
    right_eye: EyeGeometry
    confidence: float = 1.0
    yaw_deg: float = 0.0
    pitch_deg: float = 0.0
    roll_deg: float = 0.0
    face_center: Vec2 = (0.5, 0.5)
    face_size: float = 0.3
    blink_score: float | None = None  # 0=open .. 1=closed (blendshape)
    source: str = "unknown"

    @property
    def mean_openness(self) -> float:
        """Eyelid openness in ``[0, 1]`` (0 closed, 1 open).

        When a detector provides a blink blendshape (MediaPipe does), it is
        the primary signal because per-eye lid geometry is noisy at small
        angles and varies between faces.  Geometric lid distance is used as a
        fallback for backends without blendshapes.
        """
        if self.blink_score is not None:
            return _clamp(1.0 - self.blink_score, 0.0, 1.0)
        return _clamp(
            (self.left_eye.openness() + self.right_eye.openness()) / 2.0,
            0.0,
            1.0,
        )


@dataclass(frozen=True)
class GazeSample:
    """Estimated gaze for one frame, in normalized eye space."""

    timestamp: float
    gaze_x: float
    gaze_y: float
    confidence: float
    yaw_deg: float = 0.0
    pitch_deg: float = 0.0
    roll_deg: float = 0.0
    raw_gaze_x: float | None = None
    raw_gaze_y: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "t": _round(self.timestamp, 5),
            "gx": _round(self.gaze_x),
            "gy": _round(self.gaze_y),
            "conf": _round(self.confidence),
            "yaw": _round(self.yaw_deg, 2),
            "pitch": _round(self.pitch_deg, 2),
            "roll": _round(self.roll_deg, 2),
        }


@dataclass(frozen=True)
class ScreenPoint:
    """Calibrated screen position, normalized to ``[0, 1]``."""

    x: float
    y: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "x", _clamp(float(self.x), 0.0, 1.0))
        object.__setattr__(self, "y", _clamp(float(self.y), 0.0, 1.0))

    def to_pixels(self, width: int, height: int) -> tuple[int, int]:
        return int(round(self.x * (width - 1))), int(round(self.y * (height - 1)))

    def to_dict(self) -> dict[str, float]:
        return {"x": _round(self.x), "y": _round(self.y)}


@dataclass(frozen=True)
class TemporalFeatures:
    """Windowed eye-movement features computed by the extractor.

    All durations are milliseconds, all rates are per minute.  Values are
    ``None`` when there is not enough valid data yet.
    """

    window_s: float = 0.0
    valid_ratio: float = 0.0
    mean_openness: float = 0.0
    perclos: float = 0.0
    blink_count: int = 0
    blink_rate_per_min: float = 0.0
    mean_blink_ms: float | None = None
    last_blink_ms: float | None = None
    fixation_count: int = 0
    mean_fixation_ms: float | None = None
    current_fixation_ms: float = 0.0
    fixation_ratio: float = 0.0
    saccade_count: int = 0
    mean_saccade_speed_deg_s: float | None = None
    max_saccade_speed_deg_s: float | None = None
    gaze_std: float = 0.0
    gaze_stability: float = 0.0
    on_target_ratio: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_s": _round(self.window_s, 2),
            "valid_ratio": _round(self.valid_ratio, 3),
            "openness": _round(self.mean_openness, 3),
            "perclos": _round(self.perclos, 3),
            "blink_count": self.blink_count,
            "blink_rate": _round(self.blink_rate_per_min, 2),
            "blink_mean_ms": _round(self.mean_blink_ms, 1),
            "blink_last_ms": _round(self.last_blink_ms, 1),
            "fix_count": self.fixation_count,
            "fix_mean_ms": _round(self.mean_fixation_ms, 1),
            "fix_current_ms": _round(self.current_fixation_ms, 1),
            "fix_ratio": _round(self.fixation_ratio, 3),
            "saccade_count": self.saccade_count,
            "saccade_mean_dps": _round(self.mean_saccade_speed_deg_s, 1),
            "saccade_max_dps": _round(self.max_saccade_speed_deg_s, 1),
            "gaze_std": _round(self.gaze_std, 4),
            "stability": _round(self.gaze_stability, 3),
            "on_target_ratio": _round(self.on_target_ratio, 3),
        }


@dataclass(frozen=True)
class BaselineStat:
    """Robust baseline estimate for one metric within this session."""

    metric: str
    median: float
    mad: float
    z: float
    samples: int
    ready: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "median": _round(self.median, 3),
            "mad": _round(self.mad, 3),
            "z": _round(self.z, 2),
            "samples": self.samples,
            "ready": self.ready,
        }


@dataclass(frozen=True)
class Anomaly:
    """A statistical or rule-based deviation from the user's own baseline."""

    code: str
    severity: float
    message: str
    metric: str
    value: float | None = None
    baseline_z: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": _round(self.severity, 2),
            "message": self.message,
            "metric": self.metric,
            "value": _round(self.value),
            "z": _round(self.baseline_z, 2),
        }


@dataclass(frozen=True)
class GazeEvent:
    """Structured event delivered to callbacks and the event bus."""

    type: EventType
    timestamp: float
    frame_index: int
    message: str
    data: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event": self.type.value,
            "t": _round(self.timestamp, 5),
            "frame": self.frame_index,
            "message": self.message,
            "data": _jsonable(dict(self.data)),
        }


@dataclass(frozen=True)
class GazeFrame:
    """Complete per-frame output of :class:`GazePipeline`."""

    frame_index: int
    timestamp: float
    sample: GazeSample | None
    screen_point: ScreenPoint | None
    features: TemporalFeatures
    face_visible: bool
    anomalies: tuple[Anomaly, ...] = ()
    events: tuple[GazeEvent, ...] = ()
    processing_ms: float = 0.0
    source: str = "unknown"
    baseline: Mapping[str, BaselineStat] = field(default_factory=dict)

    @property
    def anomaly_codes(self) -> tuple[str, ...]:
        return tuple(a.code for a in self.anomalies)

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame": self.frame_index,
            "t": _round(self.timestamp, 5),
            "face": self.face_visible,
            "sample": self.sample.to_dict() if self.sample else None,
            "screen": self.screen_point.to_dict() if self.screen_point else None,
            "features": self.features.to_dict(),
            "anomalies": [a.to_dict() for a in self.anomalies],
            "events": [e.to_dict() for e in self.events],
            "processing_ms": _round(self.processing_ms, 3),
            "source": self.source,
            "baseline": {k: v.to_dict() for k, v in self.baseline.items()},
        }


@dataclass
class SessionSummary:
    """Aggregate description of one processing session."""

    profile: str
    source: str
    started_at: float
    duration_s: float
    frames: int
    frames_with_face: int
    mean_confidence: float
    mean_fps: float
    blink_count: int
    fixation_count: int
    saccade_count: int
    mean_blink_ms: float | None
    mean_fixation_ms: float | None
    mean_saccade_speed_deg_s: float | None
    perclos: float
    anomaly_counts: dict[str, int]
    event_counts: dict[str, int]
    detector: str
    notes: list[str] = field(default_factory=list)
    privacy: str = "frames not stored; aggregate features only"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return _jsonable(data)

    def format_text(self) -> str:
        lines = [
            f"GazeOS session summary ({self.profile})",
            f"  source       : {self.source}",
            f"  detector     : {self.detector}",
            f"  duration     : {self.duration_s:.1f} s "
            f"({self.frames} frames, {self.mean_fps:.1f} fps, "
            f"{self.frames_with_face} with face)",
            f"  confidence   : {self.mean_confidence:.2f}",
            f"  blinks       : {self.blink_count}"
            + (
                f" (mean {self.mean_blink_ms:.0f} ms)"
                if self.mean_blink_ms is not None
                else ""
            ),
            f"  fixations    : {self.fixation_count}"
            + (
                f" (mean {self.mean_fixation_ms:.0f} ms)"
                if self.mean_fixation_ms is not None
                else ""
            ),
            f"  saccades     : {self.saccade_count}"
            + (
                f" (mean {self.mean_saccade_speed_deg_s:.0f} deg/s)"
                if self.mean_saccade_speed_deg_s is not None
                else ""
            ),
            f"  PERCLOS-like : {self.perclos:.2f}",
            f"  anomalies    : {dict(self.anomaly_counts) or 'none'}",
            f"  privacy      : {self.privacy}",
        ]
        if self.notes:
            lines.append("  notes        : " + "; ".join(self.notes))
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Event bus
# --------------------------------------------------------------------------- #
class EventBus:
    """Thread-safe, synchronous publish/subscribe for :class:`GazeEvent`.

    Subscribers may filter by event type.  Exceptions raised by a subscriber
    are logged and do not stop other subscribers.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._subs: list[
            tuple[Callable[[GazeEvent], None], frozenset[EventType] | None]
        ] = []

    def subscribe(
        self,
        handler: Callable[[GazeEvent], None],
        event_types: Sequence[EventType] | None = None,
    ) -> Callable[[], None]:
        """Register ``handler`` and return an unsubscribe callable."""
        entry = (
            handler,
            frozenset(event_types) if event_types is not None else None,
        )
        with self._lock:
            self._subs.append(entry)

        def unsubscribe() -> None:
            with self._lock:
                try:
                    self._subs.remove(entry)
                except ValueError:
                    pass

        return unsubscribe

    def publish(self, event: GazeEvent) -> None:
        with self._lock:
            subs = list(self._subs)
        for handler, allowed in subs:
            if allowed is not None and event.type not in allowed:
                continue
            try:
                handler(event)
            except Exception:  # pragma: no cover - defensive
                LOG.exception("Event handler failed for %s", event.type.value)


# --------------------------------------------------------------------------- #
# Gaze estimation
# --------------------------------------------------------------------------- #
class GazeEstimator:
    """Geometry-based gaze estimator with optional head-pose compensation.

    This is intentionally simple and transparent: iris position inside the
    eyelid aperture, averaged over both eyes, scaled by configurable gains,
    plus a small linear head-pose term.  A calibrated mapping (see
    :class:`GazeCalibrator`) is required for screen coordinates; raw gaze is
    not a pixel-accurate point of regard.
    """

    def __init__(self, config: GazeConfig | None = None) -> None:
        self.config = config or make_config("research")

    def _eye_weight(self, eye: EyeGeometry) -> float:
        return 1.0 if eye.width > 1e-5 else 0.0

    def estimate(self, face: FaceObservation) -> GazeSample:
        """Estimate gaze and a heuristic confidence for one observation."""
        left, right = face.left_eye, face.right_eye
        wl, wr = self._eye_weight(left), self._eye_weight(right)
        if wl + wr <= 0:
            return GazeSample(face.timestamp, 0.0, 0.0, 0.0,
                              yaw_deg=face.yaw_deg, pitch_deg=face.pitch_deg,
                              roll_deg=face.roll_deg)

        h = (wl * left.horizontal_ratio() + wr * right.horizontal_ratio()) / (wl + wr)
        v = (wl * left.vertical_ratio() + wr * right.vertical_ratio()) / (wl + wr)

        raw_x = _clamp(
            h * self.config.gaze_gain_x
            + face.yaw_deg * self.config.head_yaw_compensation,
            -1.0,
            1.0,
        )
        raw_y = _clamp(
            v * self.config.gaze_gain_y
            + face.pitch_deg * self.config.head_pitch_compensation,
            -1.0,
            1.0,
        )

        confidence = self._confidence(face, wl, wr)
        return GazeSample(
            timestamp=face.timestamp,
            gaze_x=raw_x,
            gaze_y=raw_y,
            confidence=confidence,
            yaw_deg=face.yaw_deg,
            pitch_deg=face.pitch_deg,
            roll_deg=face.roll_deg,
            raw_gaze_x=raw_x,
            raw_gaze_y=raw_y,
        )

    def _confidence(self, face: FaceObservation, wl: float, wr: float) -> float:
        # Backend confidence is a prior; add geometric and visibility checks.
        quality = _clamp(face.confidence, 0.0, 1.0)
        if wl + wr < 1.5:
            quality *= 0.7  # one eye only
        openness = face.mean_openness
        quality *= 0.25 + 0.75 * _clamp(openness / 0.6, 0.0, 1.0)
        pose = max(abs(face.yaw_deg) / 60.0, abs(face.pitch_deg) / 45.0, 0.0)
        quality *= 1.0 - 0.5 * _clamp(pose, 0.0, 1.0)
        for eye in (face.left_eye, face.right_eye):
            if eye.width <= 1e-5:
                continue
            # Iris radius is typically ~1/3 of the visible eye width.
            if eye.iris_radius > 0:
                ratio = eye.iris_radius / eye.width
                if ratio < 0.10 or ratio > 0.75:
                    quality *= 0.75
            if eye.lid_gap > 0.6 * eye.width:
                quality *= 0.85
        return _clamp(quality, 0.0, 1.0)


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #
def _calibration_features(sample: GazeSample) -> list[float]:
    x = sample.raw_gaze_x if sample.raw_gaze_x is not None else sample.gaze_x
    y = sample.raw_gaze_y if sample.raw_gaze_y is not None else sample.gaze_y
    yaw = sample.yaw_deg / 45.0
    pitch = sample.pitch_deg / 45.0
    return [1.0, x, y, x * y, x * x, y * y, yaw, pitch]


@dataclass
class CalibrationModel:
    """Ridge-regression map from raw gaze features to normalized screen XY.

    Fitting 9 points produces a smooth low-order mapping.  This is a
    prototype calibration: it cannot remove strong lens distortion, rolling
    shutter effects or head-pose coupling.  Treat accuracy as unverified until
    you measure it on your own setup.
    """

    weights: list[list[float]]  # 8 x 2
    rmse: tuple[float, float]
    n_points: int
    created_at: float = field(default_factory=time.time)
    feature_names: tuple[str, ...] = (
        "1", "gx", "gy", "gx*gy", "gx^2", "gy^2", "yaw/45", "pitch/45",
    )

    def predict(self, sample: GazeSample) -> ScreenPoint:
        feats = np.asarray(_calibration_features(sample), dtype=np.float64)
        weights = np.asarray(self.weights, dtype=np.float64)
        out = feats @ weights
        return ScreenPoint(float(out[0]), float(out[1]))

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "gazeos.ridge_calibration.v1",
            "weights": [[round(v, 8) for v in row] for row in self.weights],
            "rmse": [round(self.rmse[0], 5), round(self.rmse[1], 5)],
            "n_points": self.n_points,
            "created_at": self.created_at,
            "feature_names": list(self.feature_names),
            "note": "Prototype calibration; accuracy not certified.",
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CalibrationModel":
        try:
            weights = [[float(v) for v in row] for row in data["weights"]]
            rmse = data.get("rmse", [0.0, 0.0])
            return cls(
                weights=weights,
                rmse=(float(rmse[0]), float(rmse[1])),
                n_points=int(data.get("n_points", 0)),
                created_at=float(data.get("created_at", time.time())),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise CalibrationError(f"Invalid calibration model: {exc}") from exc

    @classmethod
    def load(cls, path: str | Path) -> "CalibrationModel":
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except OSError as exc:
            raise CalibrationError(f"Cannot read calibration {path}: {exc}") from exc
        return cls.from_dict(data)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


class GazeCalibrator:
    """Collects raw-gaze/target pairs and fits a :class:`CalibrationModel`."""

    def __init__(self, ridge_lambda: float = 1e-3) -> None:
        self.ridge_lambda = ridge_lambda
        self._samples: list[tuple[list[float], float, float]] = []

    def add(self, sample: GazeSample, target_x: float, target_y: float) -> None:
        self._samples.append(
            (
                _calibration_features(sample),
                _clamp(target_x, 0.0, 1.0),
                _clamp(target_y, 0.0, 1.0),
            )
        )

    @property
    def count(self) -> int:
        return len(self._samples)

    def fit(self) -> CalibrationModel:
        """Fit the ridge mapping.  Raises :class:`CalibrationError` on failure."""
        if len(self._samples) < 6:
            raise CalibrationError("At least 6 calibration points are required")
        X = np.asarray([s[0] for s in self._samples], dtype=np.float64)
        Y = np.asarray([[s[1], s[2]] for s in self._samples], dtype=np.float64)
        # Features already include a constant column; regularize all but it.
        reg = np.eye(X.shape[1], dtype=np.float64) * self.ridge_lambda
        reg[0, 0] = 0.0
        try:
            W = np.linalg.solve(X.T @ X + reg, X.T @ Y)
        except np.linalg.LinAlgError as exc:
            raise CalibrationError("Calibration matrix is singular") from exc
        residual = X @ W - Y
        rmse = tuple(float(v) for v in np.sqrt(np.mean(residual**2, axis=0)))
        return CalibrationModel(
            weights=W.tolist(),
            rmse=(rmse[0], rmse[1]),
            n_points=len(self._samples),
        )

    @classmethod
    def from_csv(cls, path: str | Path) -> "GazeCalibrator":
        """Load pairs from CSV with ``gaze_x,gaze_y,target_x,target_y`` headers.

        Optional ``yaw_deg,pitch_deg`` columns are applied if present.
        """
        cal = cls()
        with Path(path).open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            required = {"gaze_x", "gaze_y", "target_x", "target_y"}
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise CalibrationError(
                    f"{path} needs CSV columns {sorted(required)}"
                )
            for row in reader:
                try:
                    sample = GazeSample(
                        timestamp=0.0,
                        gaze_x=float(row["gaze_x"]),
                        gaze_y=float(row["gaze_y"]),
                        confidence=1.0,
                        raw_gaze_x=float(row["gaze_x"]),
                        raw_gaze_y=float(row["gaze_y"]),
                        yaw_deg=float(row.get("yaw_deg") or 0.0),
                        pitch_deg=float(row.get("pitch_deg") or 0.0),
                    )
                    cal.add(sample, float(row["target_x"]), float(row["target_y"]))
                except (TypeError, ValueError) as exc:
                    raise CalibrationError(f"Bad calibration row {row}: {exc}") from exc
        return cal


# --------------------------------------------------------------------------- #
# Blink detection
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BlinkUpdate:
    """Result of feeding one openness sample into :class:`BlinkDetector`."""

    closed: bool
    just_closed: bool
    blink_ended: bool
    duration_ms: float | None
    current_closure_ms: float
    prolonged: bool


class BlinkDetector:
    """Hysteresis-based blink detector with durations and prolonged closures.

    Blinks are defined by eyelid openness crossing ``blink_open_threshold``.
    A blink ends when openness rises above ``threshold + hysteresis``; its
    duration is the time between those crossings.  Closures longer than
    ``blink_max_ms`` are classified as prolonged but still reported with a
    duration.  This is not an EOG-grade blink detector.
    """

    def __init__(self, config: GazeConfig | None = None) -> None:
        self.config = config or make_config("research")
        self.reset()

    def reset(self) -> None:
        self._closed = False
        self._closed_since: float | None = None
        self._last_duration_ms: float | None = None
        self._prolonged_fired = False
        self.blink_count = 0
        self.prolonged_count = 0

    def update(self, timestamp: float, openness: float) -> BlinkUpdate:
        cfg = self.config
        just_closed = False
        blink_ended = False
        duration_ms: float | None = None
        prolonged = False

        if not self._closed and openness < cfg.blink_open_threshold:
            self._closed = True
            self._closed_since = timestamp
            self._prolonged_fired = False
            just_closed = True
        elif self._closed and openness > cfg.blink_open_threshold + cfg.blink_close_hysteresis:
            start = self._closed_since if self._closed_since is not None else timestamp
            raw_ms = max(0.0, (timestamp - start) * 1000.0)
            if raw_ms >= cfg.blink_min_ms:
                duration_ms = raw_ms
                self._last_duration_ms = raw_ms
                self.blink_count += 1
                blink_ended = True
            self._closed = False
            self._closed_since = None
            self._prolonged_fired = False

        current = self.current_closure_ms(timestamp)
        if current >= cfg.prolonged_closure_ms and not self._prolonged_fired:
            self._prolonged_fired = True
            self.prolonged_count += 1
            prolonged = True

        return BlinkUpdate(
            closed=self._closed,
            just_closed=just_closed,
            blink_ended=blink_ended,
            duration_ms=duration_ms,
            current_closure_ms=current,
            prolonged=prolonged,
        )

    def current_closure_ms(self, timestamp: float) -> float:
        if not self._closed or self._closed_since is None:
            return 0.0
        return max(0.0, (timestamp - self._closed_since) * 1000.0)

    @property
    def last_duration_ms(self) -> float | None:
        return self._last_duration_ms


# --------------------------------------------------------------------------- #
# Fixation / saccade dynamics
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GazeDynamics:
    """Per-sample fixation/saccade state and any completed segments."""

    fixation_active: bool
    fixation_ms: float
    saccade_active: bool
    saccade_speed_deg_s: float
    ended_fixation_ms: float | None = None
    ended_saccade_peak_deg_s: float | None = None


class FixationDetector:
    """Dispersion-threshold (I-DT style) fixation detector.

    Consecutive samples whose dispersion stays below
    ``fixation_dispersion_deg`` for at least ``fixation_min_ms`` form a
    fixation.  Larger inter-sample angular speed than
    ``saccade_threshold_deg_s`` marks a saccade.  Angular values are
    approximate: normalized gaze is scaled by ``degrees_per_norm_unit`` and
    not by a measured eye radius.
    """

    def __init__(self, config: GazeConfig | None = None) -> None:
        self.config = config or make_config("research")
        self.reset()

    def reset(self) -> None:
        self._prev: tuple[float, float, float] | None = None
        self._run: deque[tuple[float, float, float]] = deque(maxlen=120)
        self._run_start: float | None = None
        self._saccade_active = False
        self._saccade_peak = 0.0
        self._last_speed = 0.0

    def _close_run(self) -> float | None:
        if self._run_start is None:
            return None
        duration = (self._run[-1][0] - self._run_start) * 1000.0 if self._run else 0.0
        self._run_start = None
        self._run.clear()
        if duration >= self.config.fixation_min_ms:
            return duration
        return None

    def update(
        self,
        timestamp: float,
        sample: GazeSample | None,
        valid: bool = True,
    ) -> GazeDynamics:
        cfg = self.config
        if not valid or sample is None:
            ended = self._close_run()
            self._prev = None
            self._saccade_active = False
            self._last_speed = 0.0
            return GazeDynamics(
                fixation_active=False,
                fixation_ms=0.0,
                saccade_active=False,
                saccade_speed_deg_s=0.0,
                ended_fixation_ms=ended,
            )

        speed = 0.0
        if self._prev is not None:
            dt = timestamp - self._prev[0]
            if 0.0 < dt <= cfg.gaze_max_dt_s:
                dx = sample.gaze_x - self._prev[1]
                dy = sample.gaze_y - self._prev[2]
                speed = (
                    math.hypot(dx, dy)
                    * cfg.degrees_per_norm_unit
                    / dt
                )
            else:
                # Gap: the previous fixation is no longer contiguous.
                self._close_run()
                self._saccade_active = False
        self._prev = (timestamp, sample.gaze_x, sample.gaze_y)
        # Light EMA on speed for display stability; detection uses raw speed.
        self._last_speed = 0.7 * speed + 0.3 * self._last_speed

        ended_fixation: float | None = None
        ended_saccade: float | None = None

        if speed > cfg.saccade_threshold_deg_s:
            ended_fixation = self._close_run()
            if not self._saccade_active:
                self._saccade_active = True
                self._saccade_peak = 0.0
            self._saccade_peak = max(self._saccade_peak, speed)
            return GazeDynamics(
                fixation_active=False,
                fixation_ms=0.0,
                saccade_active=True,
                saccade_speed_deg_s=speed,
                ended_fixation_ms=ended_fixation,
            )

        if self._saccade_active:
            ended_saccade = self._saccade_peak
            self._saccade_active = False
            self._saccade_peak = 0.0

        self._run.append((timestamp, sample.gaze_x, sample.gaze_y))
        cutoff = timestamp - 0.20
        while len(self._run) > 1 and self._run[0][0] < cutoff:
            self._run.popleft()
        xs = [s[1] for s in self._run]
        ys = [s[2] for s in self._run]
        dispersion = (
            math.hypot(max(xs) - min(xs), max(ys) - min(ys))
            * cfg.degrees_per_norm_unit
        )
        if dispersion > cfg.fixation_dispersion_deg:
            ended_fixation = self._close_run() or ended_fixation
            self._run_start = timestamp
            self._run.append((timestamp, sample.gaze_x, sample.gaze_y))
            active = False
            fixation_ms = 0.0
        else:
            if self._run_start is None:
                self._run_start = self._run[0][0]
            fixation_ms = (timestamp - self._run_start) * 1000.0
            active = fixation_ms >= cfg.fixation_min_ms

        return GazeDynamics(
            fixation_active=active,
            fixation_ms=fixation_ms,
            saccade_active=False,
            saccade_speed_deg_s=self._last_speed,
            ended_fixation_ms=ended_fixation,
            ended_saccade_peak_deg_s=ended_saccade,
        )

    def finish(self) -> float | None:
        """Close the in-progress run at end of stream; return its duration."""
        return self._close_run()


# --------------------------------------------------------------------------- #
# Temporal features
# --------------------------------------------------------------------------- #
class TemporalFeatureExtractor:
    """Aggregates per-sample signals into windowed eye-movement features.

    The extractor keeps a bounded history (one window) of gaze samples and
    completed blinks/fixations/saccades.  ``PERCLOS`` here is a simple
    fraction of observed time with eyes closed -- not the standard 80%
    eyelid criterion and not a validated drowsiness metric.
    """

    def __init__(self, config: GazeConfig | None = None) -> None:
        self.config = config or make_config("research")
        self.reset()

    def reset(self) -> None:
        self._start: float | None = None
        self._last_t: float | None = None
        self._gaze: deque[tuple[float, float, float, bool]] = deque(maxlen=4000)
        self._validity: deque[tuple[float, bool]] = deque(maxlen=4000)
        self._closed_flags: deque[tuple[float, bool]] = deque(maxlen=4000)
        self._openness: deque[tuple[float, float]] = deque(maxlen=4000)
        self._blinks: deque[tuple[float, float]] = deque()
        self._fixations: deque[tuple[float, float]] = deque()
        self._saccades: deque[tuple[float, float]] = deque()
        self._last_blink_ms: float | None = None
        self._current_fix_ms = 0.0
        self._current_fix_active = False

    def _prune(self, now: float) -> None:
        cutoff = now - self.config.window_s
        while self._blinks and self._blinks[0][0] < cutoff:
            self._blinks.popleft()
        while self._fixations and self._fixations[0][0] < cutoff:
            self._fixations.popleft()
        while self._saccades and self._saccades[0][0] < cutoff:
            self._saccades.popleft()
        while self._gaze and self._gaze[0][0] < cutoff:
            self._gaze.popleft()
        while self._validity and self._validity[0][0] < cutoff:
            self._validity.popleft()
        while self._closed_flags and self._closed_flags[0][0] < cutoff:
            self._closed_flags.popleft()
        while self._openness and self._openness[0][0] < cutoff:
            self._openness.popleft()

    def update(
        self,
        timestamp: float,
        sample: GazeSample | None,
        openness: float | None,
        closed: bool,
        dynamics: GazeDynamics,
    ) -> TemporalFeatures:
        """Ingest one frame and return the current windowed features.

        ``openness`` is ``None`` when no face is visible; such frames do not
        contribute to openness statistics.
        """
        if self._start is None:
            self._start = timestamp
        if openness is not None:
            self._openness.append((timestamp, openness))
            self._closed_flags.append((timestamp, closed))
        self._last_t = timestamp

        if dynamics.ended_fixation_ms is not None:
            self._fixations.append((timestamp, dynamics.ended_fixation_ms))
        if dynamics.ended_saccade_peak_deg_s is not None:
            self._saccades.append((timestamp, dynamics.ended_saccade_peak_deg_s))
        self._current_fix_ms = dynamics.fixation_ms
        self._current_fix_active = dynamics.fixation_active

        valid = sample is not None and sample.confidence >= self.config.min_face_confidence
        self._validity.append((timestamp, valid and not closed))
        if valid and not closed:
            on_target = False
            if self.config.attention_zone is not None:
                x0, x1, y0, y1 = self.config.attention_zone
                on_target = (
                    x0 <= sample.gaze_x <= x1 and y0 <= sample.gaze_y <= y1
                )
            self._gaze.append((timestamp, sample.gaze_x, sample.gaze_y, on_target))

        now = timestamp
        elapsed = max(now - self._start, 1e-6)
        window_s = min(elapsed, self.config.window_s)
        self._prune(now)

        blinks = list(self._blinks)
        fixations = list(self._fixations)
        saccades = list(self._saccades)
        closed_ratio = (
            sum(1 for _, flag in self._closed_flags if flag) / len(self._closed_flags)
            if self._closed_flags else 0.0
        )

        blink_durations = [d for _, d in blinks]
        fixation_durations = [d for _, d in fixations]
        if self._current_fix_active:
            fix_time = sum(fixation_durations) + self._current_fix_ms
        else:
            fix_time = sum(fixation_durations)
        saccade_speeds = [s for _, s in saccades]

        eye_std_x = eye_std_y = 0.0
        stability = 0.0
        on_target_ratio: float | None = None
        if len(self._gaze) >= 5:
            xs = np.fromiter((g[1] for g in self._gaze), dtype=np.float64)
            ys = np.fromiter((g[2] for g in self._gaze), dtype=np.float64)
            eye_std_x = float(np.std(xs))
            eye_std_y = float(np.std(ys))
            std_mag = math.hypot(eye_std_x, eye_std_y)
            stability = _clamp(1.0 - std_mag / 0.5, 0.0, 1.0)
            if self.config.attention_zone is not None:
                on_target_ratio = float(np.mean([g[3] for g in self._gaze]))

        valid_frames = sum(1 for _, is_valid in self._validity if is_valid)
        valid_ratio = valid_frames / len(self._validity) if self._validity else 0.0

        rate_denominator = max(min(elapsed, self.config.window_s), 2.0)
        return TemporalFeatures(
            window_s=window_s,
            valid_ratio=valid_ratio,
            mean_openness=(
                statistics.fmean(value for _, value in self._openness)
                if self._openness else 0.0
            ),
            perclos=closed_ratio,
            blink_count=len(blinks),
            blink_rate_per_min=len(blinks) * 60.0 / rate_denominator,
            mean_blink_ms=statistics.fmean(blink_durations) if blink_durations else None,
            last_blink_ms=self._last_blink_ms,
            fixation_count=len(fixations),
            mean_fixation_ms=statistics.fmean(fixation_durations) if fixation_durations else None,
            current_fixation_ms=self._current_fix_ms if self._current_fix_active else 0.0,
            fixation_ratio=_clamp(fix_time / max(window_s, 1e-6), 0.0, 1.0),
            saccade_count=len(saccades),
            mean_saccade_speed_deg_s=statistics.fmean(saccade_speeds) if saccade_speeds else None,
            max_saccade_speed_deg_s=max(saccade_speeds) if saccade_speeds else None,
            gaze_std=math.hypot(eye_std_x, eye_std_y),
            gaze_stability=stability,
            on_target_ratio=on_target_ratio,
        )

    def record_blink(self, timestamp: float, duration_ms: float) -> None:
        """Record a completed blink (called by the pipeline)."""
        self._blinks.append((timestamp, duration_ms))
        self._last_blink_ms = duration_ms


# --------------------------------------------------------------------------- #
# Baseline tracking
# --------------------------------------------------------------------------- #
class BaselineTracker:
    """Robust per-session baseline (median/MAD) for selected feature metrics.

    The baseline adapts to the individual and the camera instead of applying
    population thresholds.  It needs valid samples and a warm-up period before
    z-scores are reported (``ready``).  A session baseline is not a clinical
    norm and should not be used to infer health.
    """

    METRICS: tuple[str, ...] = (
        "blink_rate_per_min",
        "mean_blink_ms",
        "mean_fixation_ms",
        "mean_saccade_speed_deg_s",
        "perclos",
        "gaze_stability",
    )

    def __init__(self, config: GazeConfig | None = None) -> None:
        self.config = config or make_config("research")
        self._values: dict[str, deque[float]] = {
            m: deque(maxlen=self.config.baseline_max_samples) for m in self.METRICS
        }
        self._started: float | None = None

    def reset(self) -> None:
        for values in self._values.values():
            values.clear()
        self._started = None

    def update(self, timestamp: float, features: TemporalFeatures) -> None:
        if self._started is None:
            self._started = timestamp
        if timestamp - self._started < self.config.baseline_warmup_s:
            return
        if features.valid_ratio < 0.05:
            return
        for metric in self.METRICS:
            value = getattr(features, metric, None)
            if value is None:
                continue
            self._values[metric].append(float(value))

    def snapshot(self, features: TemporalFeatures) -> dict[str, BaselineStat]:
        stats: dict[str, BaselineStat] = {}
        for metric in self.METRICS:
            values = self._values[metric]
            current = getattr(features, metric, None)
            if current is None or len(values) < self.config.baseline_min_samples:
                continue
            median = statistics.fmean(values) if len(values) < 10 else statistics.median(values)
            deviations = [abs(v - median) for v in values]
            mad = statistics.median(deviations)
            scale = 1.4826 * mad
            z = 0.0
            if scale > 1e-9:
                z = (float(current) - median) / scale
            elif abs(float(current) - median) > 1e-9:
                z = math.copysign(4.0, float(current) - median)
            stats[metric] = BaselineStat(
                metric=metric,
                median=median,
                mad=mad,
                z=z,
                samples=len(values),
                ready=True,
            )
        return stats


# --------------------------------------------------------------------------- #
# Anomaly detection
# --------------------------------------------------------------------------- #
class AnomalyDetector:
    """Rule + robust-baseline anomaly flags.

    Anomalies describe *deviation from this session's own baseline* or from
    configurable thresholds.  They are research/demo signals and are **not**
    medical, diagnostic or safety-certified conclusions.
    """

    def __init__(self, config: GazeConfig | None = None) -> None:
        self.config = config or make_config("research")

    def detect(
        self,
        timestamp: float,
        features: TemporalFeatures,
        baseline: Mapping[str, BaselineStat],
        sample: GazeSample | None,
        closure_ms: float,
        face_present: bool,
    ) -> list[Anomaly]:
        cfg = self.config
        found: dict[str, Anomaly] = {}

        def add(anomaly: Anomaly) -> None:
            existing = found.get(anomaly.code)
            if existing is None or anomaly.severity > existing.severity:
                found[anomaly.code] = anomaly

        if not face_present:
            add(Anomaly("face_lost", 0.4, "Face is not visible.", "face", 0.0))
            return list(found.values())

        if features.window_s >= 2.0:
            if features.perclos >= cfg.perclos_critical:
                add(Anomaly(
                    "perclos_critical", 0.9,
                    "Sustained eye closure above the configured critical level.",
                    "perclos", features.perclos,
                ))
            elif features.perclos >= cfg.perclos_warn:
                add(Anomaly(
                    "perclos_elevated", 0.55,
                    "Eye-closed ratio above the configured warning level.",
                    "perclos", features.perclos,
                ))

            if (
                features.blink_count >= cfg.min_blinks_for_rate_rules
                and features.blink_rate_per_min > cfg.blink_rate_high
            ):
                add(Anomaly(
                    "blink_rate_elevated", 0.35,
                    "Blink rate is above this profile's typical range.",
                    "blink_rate_per_min", features.blink_rate_per_min,
                ))
            if (
                features.blink_count >= cfg.min_blinks_for_rate_rules
                and features.blink_rate_per_min < cfg.blink_rate_low
            ):
                add(Anomaly(
                    "blink_rate_low", 0.35,
                    "Blink rate is below this profile's typical range.",
                    "blink_rate_per_min", features.blink_rate_per_min,
                ))

            if (
                features.mean_blink_ms is not None
                and features.mean_blink_ms > cfg.blink_duration_warn_ms
            ):
                add(Anomaly(
                    "blink_duration_elevated", 0.5,
                    "Average blink duration is above the configured warning level.",
                    "mean_blink_ms", features.mean_blink_ms,
                ))

            if (
                features.fixation_count >= 2
                and features.gaze_stability < cfg.gaze_stability_min
            ):
                add(Anomaly(
                    "gaze_instability", 0.3,
                    "Gaze is less stable than this profile's reference level.",
                    "gaze_stability", features.gaze_stability,
                ))

            if (
                features.mean_fixation_ms is not None
                and features.mean_fixation_ms > cfg.fixation_long_ms
            ):
                add(Anomaly(
                    "fixation_prolonged", 0.25,
                    "Fixations are unusually long for this profile.",
                    "mean_fixation_ms", features.mean_fixation_ms,
                ))

            if (
                features.on_target_ratio is not None
                and features.window_s >= max(cfg.baseline_warmup_s, 3.0)
                and features.on_target_ratio < cfg.on_target_min_ratio
            ):
                add(Anomaly(
                    "off_target_proxy", 0.3,
                    "Gaze is often outside the configured attention zone "
                    "(camera-space proxy, not a road or screen truth).",
                    "on_target_ratio", features.on_target_ratio,
                ))

        if closure_ms >= cfg.prolonged_closure_ms:
            add(Anomaly(
                "prolonged_closure", 0.85,
                f"Eyes have been closed for {closure_ms / 1000.0:.1f} s "
                "(relative to this configuration).",
                "closure_ms", closure_ms,
            ))

        if sample is not None:
            if abs(sample.yaw_deg) > cfg.head_yaw_max_deg:
                add(Anomaly(
                    "head_yaw_extreme", 0.3,
                    "Head yaw is outside the configured range.",
                    "head_yaw_deg", sample.yaw_deg,
                ))
            if abs(sample.pitch_deg) > cfg.head_pitch_max_deg:
                add(Anomaly(
                    "head_pitch_extreme", 0.3,
                    "Head pitch is outside the configured range.",
                    "head_pitch_deg", sample.pitch_deg,
                ))
            if sample.confidence < cfg.min_face_confidence:
                add(Anomaly(
                    "low_confidence", 0.2,
                    "Gaze confidence is low; treat this frame as unreliable.",
                    "confidence", sample.confidence,
                ))

        z_metrics = {
            "perclos",
            "blink_rate_per_min",
            "mean_blink_ms",
            "gaze_stability",
            "mean_saccade_speed_deg_s",
        }
        for metric, stat in baseline.items():
            if metric not in z_metrics or not stat.ready:
                continue
            if abs(stat.z) >= cfg.anomaly_z:
                add(Anomaly(
                    f"baseline_shift:{metric}",
                    _clamp(abs(stat.z) / (cfg.anomaly_z * 2.0), 0.1, 0.9),
                    f"{metric} deviates from this session's baseline.",
                    metric,
                    stat.median + stat.z * stat.mad * 1.4826 if stat.mad else stat.median,
                    baseline_z=stat.z,
                ))

        return sorted(found.values(), key=lambda a: a.severity, reverse=True)


# --------------------------------------------------------------------------- #
# Frame sources
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FramePacket:
    """One captured frame with a monotonic-ish timestamp in seconds."""

    frame_index: int
    timestamp: float
    image: np.ndarray | None = None


class FrameSource:
    """Interface for camera, video-file and custom frame providers.

    Implementations must set timestamps in seconds, ideally on a monotonic
    clock.  ``read`` returns ``None`` when the source is exhausted.
    """

    name: str = "frame-source"

    def open(self) -> None:  # pragma: no cover - interface default
        pass

    def read(self) -> FramePacket | None:  # pragma: no cover - interface
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover - interface default
        pass

    def __enter__(self) -> "FrameSource":
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __iter__(self) -> Iterator[FramePacket]:
        while True:
            packet = self.read()
            if packet is None:
                return
            yield packet


def import_cv2() -> Any:
    """Import OpenCV or raise a helpful :class:`DependencyMissingError`."""
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise DependencyMissingError(
            "OpenCV is required for camera/video input and previews. "
            "Install it with: pip install opencv-python"
        ) from exc
    return cv2


class CameraSource(FrameSource):
    """Webcam input via OpenCV.

    The frame is **not** mirrored: gaze coordinates stay in camera space.
    Mirroring is applied to preview windows only.  If the camera cannot be
    opened, :class:`CameraUnavailableError` explains the common causes,
    including operating-system permission prompts.
    """

    def __init__(
        self,
        camera_index: int = 0,
        width: int = 640,
        height: int = 480,
        fps: float = 30.0,
    ) -> None:
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        self.name = f"camera:{camera_index}"
        self._capture: Any = None
        self._index = -1

    def open(self) -> None:
        if self._capture is not None:
            return
        cv2 = import_cv2()
        capture = cv2.VideoCapture(self.camera_index)
        if not capture.isOpened():
            capture.release()
            raise CameraUnavailableError(
                f"Could not open camera index {self.camera_index}. "
                "Check that (1) the OS grants camera permission to your "
                "terminal/IDE (macOS: System Settings > Privacy & Security > "
                "Camera; Linux: video group + /dev/video* permissions; "
                "Windows: Settings > Privacy > Camera), (2) no other app is "
                "using the camera, and (3) the index is correct. "
                "For a hardware-free run use `python gaze_core.py demo`."
            )
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        capture.set(cv2.CAP_PROP_FPS, self.fps)
        self._capture = capture

    def read(self) -> FramePacket | None:
        if self._capture is None:
            self.open()
        ok, image = self._capture.read()
        now = time.monotonic()
        if not ok or image is None:
            LOG.warning("Camera read failed on %s; ending stream", self.name)
            return None
        self._index += 1
        return FramePacket(frame_index=self._index, timestamp=now, image=image)

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None


class VideoFileSource(FrameSource):
    """Video file input via OpenCV with real-time pacing by default."""

    def __init__(self, path: str | Path, pace: bool = True, loop: bool = False) -> None:
        self.path = Path(path)
        self.pace = pace
        self.loop = loop
        self.name = f"video:{self.path.name}"
        self._capture: Any = None
        self._index = -1
        self._start = 0.0
        self._fps = 30.0

    def open(self) -> None:
        if self._capture is not None:
            return
        if not self.path.exists():
            raise FileNotFoundError(f"Video file not found: {self.path}")
        cv2 = import_cv2()
        capture = cv2.VideoCapture(str(self.path))
        if not capture.isOpened():
            raise GazeOSError(f"OpenCV could not open video: {self.path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        self._fps = fps if 1.0 <= fps <= 240.0 else 30.0
        self._capture = capture
        self._start = time.monotonic()
        self._index = -1

    def read(self) -> FramePacket | None:
        if self._capture is None:
            self.open()
        ok, image = self._capture.read()
        if not ok or image is None:
            if self.loop:
                cv2 = import_cv2()
                self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self._start = time.monotonic()
                self._index = -1
                ok, image = self._capture.read()
            if not ok or image is None:
                return None
        self._index += 1
        timestamp = time.monotonic()
        if self.pace:
            target = self._start + (self._index + 1) / self._fps
            delay = target - timestamp
            if delay > 0:
                time.sleep(min(delay, 0.1))
        return FramePacket(self._index, timestamp, image)

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None


# --------------------------------------------------------------------------- #
# Eye landmark detection (MediaPipe Face Landmarker)
# --------------------------------------------------------------------------- #
# MediaPipe Face Mesh indices.  Anatomical "left" appears on the right side of
# the unmirrored camera image; all math below uses image space, so this only
# matters for docstrings and the ``side`` field.
_EYE_INDEX = {
    "right": {"inner": 133, "outer": 33, "upper": 159, "lower": 145,
              "iris": (469, 470, 471, 472)},
    "left": {"inner": 362, "outer": 263, "upper": 386, "lower": 374,
             "iris": (474, 475, 476, 477)},
}
_BLENDSHAPE_NAMES = ("eyeBlinkLeft", "eyeBlinkRight")


class EyeLandmarkDetector:
    """Backend interface producing :class:`FaceObservation` objects."""

    name: str = "eye-landmark-backend"

    def load(self) -> None:
        """Load model weights.  Idempotent; may raise on failure."""

    def detect(self, image: np.ndarray, timestamp: float) -> list[FaceObservation]:
        raise NotImplementedError

    def close(self) -> None:
        pass


class MediaPipeEyeLandmarkDetector(EyeLandmarkDetector):
    """MediaPipe Face Landmarker backend with iris landmarks.

    The model file (``face_landmarker.task``) is **not** bundled.  Download it
    once::

        mkdir -p models
        curl -L -o models/face_landmarker.task \\
          https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task

    or point ``--model`` / ``GAZEOS_FACE_MODEL`` at an existing file.

    Notes
    -----
    * Runs on CPU by default (MediaPipe delegates are optional).
    * Head pose is read from the facial transformation matrix and is
      approximate.
    * The Tasks API does not expose a per-face detection score here, so the
      backend confidence is 1.0 and downstream geometric quality checks are
      responsible for filtering unreliable frames.
    """

    name = "mediapipe.face_landmarker"

    def __init__(
        self,
        model_path: str | Path | None = None,
        num_faces: int = 1,
        min_detection_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        delegate: str = "CPU",
    ) -> None:
        env_path = os.environ.get("GAZEOS_FACE_MODEL")
        self.model_path = Path(model_path or env_path or DEFAULT_MODEL_PATH)
        self.num_faces = max(1, int(num_faces))
        self.min_detection_confidence = min_detection_confidence
        self.min_presence_confidence = min_presence_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.delegate = delegate
        self._landmarker: Any = None
        self._last_ts_ms = -1

    def load(self) -> None:
        if self._landmarker is not None:
            return
        if not self.model_path.exists():
            raise ModelNotFoundError(
                f"MediaPipe face model not found at {self.model_path}. "
                "Download it with:\n"
                "  mkdir -p models && curl -L -o models/face_landmarker.task \\\n"
                f"    {FACE_LANDMARKER_URL}\n"
                "or pass --model / set GAZEOS_FACE_MODEL. "
                "For a no-model demo use: python gaze_core.py demo"
            )
        try:
            import mediapipe as mp  # type: ignore[import-not-found]
            from mediapipe.tasks import python as mp_python  # type: ignore
            from mediapipe.tasks.python import vision  # type: ignore
        except ImportError as exc:  # pragma: no cover - environment specific
            raise DependencyMissingError(
                "The MediaPipe backend requires the `mediapipe` package. "
                "Install with: pip install mediapipe"
            ) from exc
        self._mp = mp
        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=str(self.model_path),
                delegate=(
                    mp_python.BaseOptions.Delegate.GPU
                    if self.delegate.upper() == "GPU"
                    else mp_python.BaseOptions.Delegate.CPU
                ),
            ),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=self.num_faces,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            min_face_detection_confidence=self.min_detection_confidence,
            min_face_presence_confidence=self.min_presence_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )
        try:
            self._landmarker = vision.FaceLandmarker.create_from_options(options)
        except Exception as exc:  # pragma: no cover - model/runtime specific
            raise ModelNotFoundError(
                f"Failed to initialise MediaPipe FaceLandmarker: {exc}"
            ) from exc
        LOG.info("Loaded MediaPipe face landmarker from %s", self.model_path)

    def detect(self, image: np.ndarray, timestamp: float) -> list[FaceObservation]:
        if self._landmarker is None:
            self.load()
        if image is None or image.size == 0:
            return []
        rgb = np.ascontiguousarray(image[:, :, ::-1])  # BGR -> RGB
        mp_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB, data=rgb
        )
        ts_ms = int(timestamp * 1000.0)
        if ts_ms <= self._last_ts_ms:
            ts_ms = self._last_ts_ms + 1
        self._last_ts_ms = ts_ms
        try:
            result = self._landmarker.detect_for_video(mp_image, ts_ms)
        except Exception:
            LOG.exception("MediaPipe detection failed")
            return []
        faces: list[FaceObservation] = []
        landmarks_list = getattr(result, "face_landmarks", None) or []
        blendshapes_list = getattr(result, "face_blendshapes", None) or []
        matrices = getattr(result, "facial_transformation_matrixes", None) or []
        h, w = image.shape[:2]
        for idx, landmarks in enumerate(landmarks_list):
            if len(landmarks) < 478:
                continue
            try:
                left = self._eye_geometry(landmarks, "left")
                right = self._eye_geometry(landmarks, "right")
            except (IndexError, ValueError):
                continue
            blink_score: float | None = None
            if idx < len(blendshapes_list) and blendshapes_list[idx]:
                scores: list[float] = []
                for category in blendshapes_list[idx]:
                    if getattr(category, "category_name", "") in _BLENDSHAPE_NAMES:
                        scores.append(float(category.score))
                if scores:
                    blink_score = sum(scores) / len(scores)
            yaw = pitch = roll = 0.0
            if idx < len(matrices):
                yaw, pitch, roll = self._head_pose(matrices[idx])
            xs = [lm.x for lm in landmarks]
            ys = [lm.y for lm in landmarks]
            x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
            faces.append(
                FaceObservation(
                    timestamp=timestamp,
                    left_eye=left,
                    right_eye=right,
                    confidence=1.0,
                    yaw_deg=yaw,
                    pitch_deg=pitch,
                    roll_deg=roll,
                    face_center=((x0 + x1) / 2.0, (y0 + y1) / 2.0),
                    face_size=float(max(x1 - x0, y1 - y0) * max(w, h)),
                    blink_score=blink_score,
                    source=f"mediapipe:{w}x{h}",
                )
            )
        faces.sort(key=lambda f: f.face_size, reverse=True)
        return faces[: self.num_faces]

    @staticmethod
    def _eye_geometry(landmarks: Sequence[Any], side: str) -> EyeGeometry:
        idx = _EYE_INDEX[side]
        pts = [(float(landmarks[i].x), float(landmarks[i].y)) for i in idx["iris"]]
        iris = (
            sum(p[0] for p in pts) / len(pts),
            sum(p[1] for p in pts) / len(pts),
        )
        radius = sum(_dist(p, iris) for p in pts) / len(pts)
        return EyeGeometry(
            inner=(float(landmarks[idx["inner"]].x), float(landmarks[idx["inner"]].y)),
            outer=(float(landmarks[idx["outer"]].x), float(landmarks[idx["outer"]].y)),
            upper=(float(landmarks[idx["upper"]].x), float(landmarks[idx["upper"]].y)),
            lower=(float(landmarks[idx["lower"]].x), float(landmarks[idx["lower"]].y)),
            iris=iris,
            iris_radius=radius,
            side=side,
        )

    @staticmethod
    def _head_pose(matrix: Any) -> tuple[float, float, float]:
        """Extract approximate yaw/pitch/roll (degrees) from a 4x4 matrix."""
        try:
            arr = np.asarray(matrix, dtype=np.float64)
            if arr.shape == (4, 4):
                rot = arr[:3, :3]
            elif arr.ndim == 2 and arr.shape[0] >= 3:
                rot = arr[:3, :3]
            else:
                return 0.0, 0.0, 0.0
            yaw = math.degrees(math.asin(_clamp(-rot[2, 0], -1.0, 1.0)))
            pitch = math.degrees(math.atan2(rot[2, 1], rot[2, 2]))
            roll = math.degrees(math.atan2(rot[1, 0], rot[0, 0]))
            return float(yaw), float(pitch), float(roll)
        except Exception:  # pragma: no cover - defensive
            return 0.0, 0.0, 0.0

    def close(self) -> None:
        if self._landmarker is not None:
            try:
                self._landmarker.close()
            except Exception:  # pragma: no cover - defensive
                LOG.debug("FaceLandmarker close failed", exc_info=True)
            self._landmarker = None


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
class GazePipeline:
    """Orchestrates estimation, blink/fixation detection, features and events.

    The pipeline consumes backend-agnostic :class:`FaceObservation` objects, so
    it can be used with MediaPipe, the synthetic twin, or a custom detector
    without touching the analytics.

    Parameters
    ----------
    config:
        Application profile and thresholds.
    calibration:
        Optional mapping from raw gaze to normalized screen coordinates.
    bus:
        Optional event bus; a new one is created if omitted.
    """

    def __init__(
        self,
        config: GazeConfig | None = None,
        calibration: CalibrationModel | None = None,
        bus: EventBus | None = None,
    ) -> None:
        self.config = config or make_config("research")
        self.calibration = calibration
        self.bus = bus or EventBus()
        self.estimator = GazeEstimator(self.config)
        self.blink = BlinkDetector(self.config)
        self.fixations = FixationDetector(self.config)
        self.features = TemporalFeatureExtractor(self.config)
        self.baseline = BaselineTracker(self.config)
        self.detector = AnomalyDetector(self.config)
        self.reset()

    def reset(self) -> None:
        self.blink.reset()
        self.fixations.reset()
        self.features.reset()
        self.baseline.reset()
        self.frame_index = -1
        self._ema: tuple[float, float] | None = None
        self._last_timestamp: float | None = None
        self._last_face_time: float | None = None
        self._face_visible = False
        self._face_lost_emitted = False
        self._active_anomalies: set[str] = set()
        self._prolonged_pending = False
        self._drowsy_emitted = False
        self._frames = 0
        self._frames_with_face = 0
        self._confidence_sum = 0.0
        self._perclos_samples: deque[float] = deque(maxlen=300)
        self._all_blinks: list[float] = []
        self._all_fixations: list[float] = []
        self._all_saccades: list[float] = []
        self._event_counts: Counter[str] = Counter()
        self._anomaly_counts: Counter[str] = Counter()
        self.notes: list[str] = []
        self.detector_name = "none"
        self.last_frame: GazeFrame | None = None

    # -- helpers ---------------------------------------------------------- #
    def _emit(
        self,
        event_type: EventType,
        timestamp: float,
        frame_index: int,
        message: str,
        data: dict[str, Any] | None = None,
    ) -> GazeEvent:
        event = GazeEvent(event_type, timestamp, frame_index, message, data or {})
        self._event_counts[event_type.value] += 1
        self.bus.publish(event)
        return event

    def _smooth(self, sample: GazeSample, timestamp: float) -> GazeSample:
        if self._ema is None or (
            self._last_timestamp is not None
            and timestamp - self._last_timestamp > 0.3
        ):
            self._ema = (sample.gaze_x, sample.gaze_y)
            return sample
        prev_x, prev_y = self._ema
        jump = math.hypot(sample.gaze_x - prev_x, sample.gaze_y - prev_y)
        alpha = 1.0 if jump > self.config.gaze_saccade_bypass else self.config.gaze_smoothing
        x = alpha * sample.gaze_x + (1.0 - alpha) * prev_x
        y = alpha * sample.gaze_y + (1.0 - alpha) * prev_y
        self._ema = (x, y)
        return replace(sample, gaze_x=x, gaze_y=y)

    # -- main entry ------------------------------------------------------- #
    def process(
        self,
        face: FaceObservation | None,
        timestamp: float | None = None,
        frame_index: int | None = None,
    ) -> GazeFrame:
        """Process one frame observation and return structured output."""
        started = time.perf_counter()
        if frame_index is None:
            self.frame_index += 1
            frame_index = self.frame_index
        else:
            self.frame_index = frame_index
        if timestamp is None:
            timestamp = face.timestamp if face is not None else time.monotonic()

        events: list[GazeEvent] = []
        sample: GazeSample | None = None
        screen_point: ScreenPoint | None = None
        dynamics = GazeDynamics(False, 0.0, False, 0.0)
        blink_update = BlinkUpdate(False, False, False, None, 0.0, False)

        if face is None:
            dynamics = self.fixations.update(timestamp, None, valid=False)
            if (
                not self._face_lost_emitted
                and (
                    self._last_face_time is None
                    or timestamp - self._last_face_time >= self.config.face_lost_grace_s
                )
            ):
                events.append(self._emit(
                    EventType.FACE_LOST, timestamp, frame_index,
                    "No face detected; eye analytics paused.",
                ))
                self._face_lost_emitted = True
                self._face_visible = False
                # Do not count the face-absence gap as a blink/closure.
                self.blink.reset()
                self._prolonged_pending = False
            features = self.features.update(
                timestamp, None, openness=None, closed=False, dynamics=dynamics
            )
            anomalies = self.detector.detect(
                timestamp, features, {}, None, 0.0, face_present=False
            )
            # Only surface face-loss as an anomaly once per loss episode.
            anomalies = [a for a in anomalies if a.code not in self._active_anomalies]
            self._active_anomalies = {"face_lost"}
            for anomaly in anomalies:
                self._anomaly_counts[anomaly.code] += 1
            self._frames += 1
            report = GazeFrame(
                frame_index=frame_index,
                timestamp=timestamp,
                sample=None,
                screen_point=None,
                features=features,
                face_visible=False,
                anomalies=tuple(anomalies),
                events=tuple(events),
                processing_ms=(time.perf_counter() - started) * 1000.0,
                source="none",
                baseline={},
            )
            self._last_timestamp = timestamp
            self.last_frame = report
            return report

        # --- face present ------------------------------------------------ #
        if not self._face_visible or self._face_lost_emitted:
            events.append(self._emit(
                EventType.FACE_FOUND, timestamp, frame_index, "Face detected.",
                {"face_size": round(face.face_size, 1)},
            ))
        self._face_visible = True
        self._face_lost_emitted = False
        self._last_face_time = timestamp
        self._frames += 1
        self._frames_with_face += 1

        sample = self.estimator.estimate(face)
        self._confidence_sum += sample.confidence
        openness = face.mean_openness
        blink_update = self.blink.update(timestamp, openness)

        if blink_update.blink_ended and blink_update.duration_ms is not None:
            duration = blink_update.duration_ms
            self._all_blinks.append(duration)
            self.features.record_blink(timestamp, duration)
            events.append(self._emit(
                EventType.BLINK, timestamp, frame_index,
                f"Blink ({duration:.0f} ms).",
                {"duration_ms": round(duration, 1)},
            ))
            if self._prolonged_pending:
                events.append(self._emit(
                    EventType.PROLONGED_CLOSURE, timestamp, frame_index,
                    "Prolonged eye closure ended.",
                    {"duration_ms": round(duration, 1)},
                ))
                self._prolonged_pending = False
                self._drowsy_emitted = False

        if blink_update.prolonged:
            self._prolonged_pending = True
            events.append(self._emit(
                EventType.DROWSINESS, timestamp, frame_index,
                "Eyes closed longer than the configured prolonged-closure threshold.",
                {"closure_ms": round(blink_update.current_closure_ms, 1)},
            ))

        gaze_valid = (
            not blink_update.closed
            and sample.confidence >= self.config.min_face_confidence
            and face.mean_openness > self.config.blink_open_threshold * 0.8
        )
        if gaze_valid:
            sample = self._smooth(sample, timestamp)
        dynamics = self.fixations.update(timestamp, sample, valid=gaze_valid)
        if dynamics.ended_fixation_ms is not None:
            self._all_fixations.append(dynamics.ended_fixation_ms)
            events.append(self._emit(
                EventType.FIXATION, timestamp, frame_index,
                f"Fixation ended ({dynamics.ended_fixation_ms:.0f} ms).",
                {"duration_ms": round(dynamics.ended_fixation_ms, 1)},
            ))
        if dynamics.ended_saccade_peak_deg_s is not None:
            peak = dynamics.ended_saccade_peak_deg_s
            self._all_saccades.append(peak)
            events.append(self._emit(
                EventType.SACCADE, timestamp, frame_index,
                f"Saccade (peak ~{peak:.0f} deg/s).",
                {"peak_speed_deg_s": round(peak, 1)},
            ))

        if (
            self.calibration is not None
            and gaze_valid
            and sample.confidence >= self.config.min_face_confidence
        ):
            screen_point = self.calibration.predict(sample)

        features = self.features.update(
            timestamp,
            sample if gaze_valid else None,
            openness=openness,
            closed=blink_update.closed,
            dynamics=dynamics,
        )
        self.baseline.update(timestamp, features)
        baseline_stats = self.baseline.snapshot(features)
        anomalies = self.detector.detect(
            timestamp,
            features,
            baseline_stats,
            sample,
            blink_update.current_closure_ms,
            face_present=True,
        )
        current_codes = {a.code for a in anomalies}
        for anomaly in anomalies:
            if anomaly.code not in self._active_anomalies:
                self._anomaly_counts[anomaly.code] += 1
                events.append(self._emit(
                    EventType.ANOMALY, timestamp, frame_index,
                    anomaly.message,
                    anomaly.to_dict(),
                ))
                if anomaly.code in {"perclos_critical", "prolonged_closure"} and not self._drowsy_emitted:
                    self._drowsy_emitted = True
                    events.append(self._emit(
                        EventType.DROWSINESS, timestamp, frame_index,
                        "Drowsiness-related indicator crossed its threshold.",
                        {"code": anomaly.code},
                    ))
        self._active_anomalies = current_codes
        self._perclos_samples.append(features.perclos)
        self._last_timestamp = timestamp

        report = GazeFrame(
            frame_index=frame_index,
            timestamp=timestamp,
            sample=sample,
            screen_point=screen_point,
            features=features,
            face_visible=True,
            anomalies=tuple(anomalies),
            events=tuple(events),
            processing_ms=(time.perf_counter() - started) * 1000.0,
            source=face.source,
            baseline=baseline_stats,
        )
        self.last_frame = report
        return report

    # -- lifecycle -------------------------------------------------------- #
    def finish(self, timestamp: float | None = None) -> None:
        """Flush an in-progress fixation at end of stream.

        Without this, a short session that ends during a fixation would report
        zero fixations.  Safe to call once; subsequent calls are no-ops.
        """
        ended = self.fixations.finish()
        if ended is None:
            return
        t = timestamp if timestamp is not None else (self._last_timestamp or 0.0)
        self._all_fixations.append(ended)
        self._emit(
            EventType.FIXATION, t, self.frame_index,
            f"Fixation ended at stream end ({ended:.0f} ms).",
            {"duration_ms": round(ended, 1), "flushed": True},
        )

    # -- summary ---------------------------------------------------------- #
    def summary(
        self,
        duration_s: float,
        source: str = "unknown",
        started_at: float | None = None,
    ) -> SessionSummary:
        """Build an aggregate, non-identifiable session summary."""
        frames_with_face = max(self._frames_with_face, 1)
        mean_conf = self._confidence_sum / frames_with_face if self._frames_with_face else 0.0
        perclos = (
            statistics.fmean(self._perclos_samples) if self._perclos_samples else 0.0
        )
        return SessionSummary(
            profile=self.config.profile,
            source=source,
            started_at=started_at if started_at is not None else time.time(),
            duration_s=duration_s,
            frames=self._frames,
            frames_with_face=self._frames_with_face,
            mean_confidence=round(mean_conf, 4),
            mean_fps=round(self._frames / duration_s, 2) if duration_s > 0 else 0.0,
            blink_count=len(self._all_blinks),
            fixation_count=len(self._all_fixations),
            saccade_count=len(self._all_saccades),
            mean_blink_ms=(
                round(statistics.fmean(self._all_blinks), 1)
                if self._all_blinks else None
            ),
            mean_fixation_ms=(
                round(statistics.fmean(self._all_fixations), 1)
                if self._all_fixations else None
            ),
            mean_saccade_speed_deg_s=(
                round(statistics.fmean(self._all_saccades), 1)
                if self._all_saccades else None
            ),
            perclos=round(perclos, 4),
            anomaly_counts=dict(self._anomaly_counts),
            event_counts=dict(self._event_counts),
            detector=self.detector_name,
            notes=list(self.notes),
        )


# --------------------------------------------------------------------------- #
# Structured logging (opt-in, aggregates only)
# --------------------------------------------------------------------------- #
class GazeSessionLogger:
    """Writes per-frame JSONL plus a summary JSON when explicitly configured.

    Raw frames are never written here.  Gaze features and timestamps are still
    behavioural data: obtain consent and apply retention rules before enabling
    this in studies.
    """

    def __init__(self, path: str | Path, metadata: Mapping[str, Any] | None = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("w", encoding="utf-8")
        header = {
            "record": "header",
            "schema": "gazeos.frame.v1",
            "created_at": time.time(),
            "privacy": "aggregate gaze features only; no images or landmarks",
            "metadata": _jsonable(dict(metadata or {})),
        }
        self._handle.write(json.dumps(header) + "\n")
        self._closed = False

    def log(self, frame: GazeFrame) -> None:
        if self._closed:
            return
        self._handle.write(json.dumps(frame.to_dict(), separators=(",", ":")) + "\n")

    def close(self, summary: SessionSummary | None = None) -> None:
        if self._closed:
            return
        self._handle.flush()
        self._handle.close()
        self._closed = True
        if summary is not None:
            out = self.path.with_suffix(self.path.suffix + ".summary.json")
            out.write_text(json.dumps(summary.to_dict(), indent=2), encoding="utf-8")

    def __enter__(self) -> "GazeSessionLogger":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


# --------------------------------------------------------------------------- #
# Session orchestration
# --------------------------------------------------------------------------- #
class GazeSession:
    """Connects a frame source, a landmark detector and a pipeline.

    This is a thin, synchronous orchestrator suitable for one camera stream.
    It performs per-frame error handling, optional preview rendering, callback
    dispatch and opt-in JSONL logging.
    """

    def __init__(
        self,
        source: FrameSource,
        config: GazeConfig | None = None,
        detector: EyeLandmarkDetector | None = None,
        calibration: CalibrationModel | None = None,
        bus: EventBus | None = None,
        logger: GazeSessionLogger | None = None,
    ) -> None:
        self.source = source
        self.config = config or make_config("research")
        self.detector = detector or MediaPipeEyeLandmarkDetector(
            num_faces=self.config.max_faces
        )
        self.bus = bus or EventBus()
        self.pipeline = GazePipeline(self.config, calibration, self.bus)
        self.pipeline.detector_name = getattr(self.detector, "name", "custom")
        self.logger = logger

    def open(self) -> None:
        self.source.open()
        try:
            self.detector.load()
        except Exception:
            self.source.close()
            raise

    def run(
        self,
        max_frames: int | None = None,
        max_seconds: float | None = None,
        display: bool = False,
        on_frame: Callable[[GazeFrame], None] | None = None,
        on_packet: Callable[[FramePacket, GazeFrame], None] | None = None,
    ) -> SessionSummary:
        """Run until the source ends or a limit is reached; return a summary.

        ``on_packet`` receives the raw frame packet together with its report.
        Use it for explicit, consent-gated storage; the core never stores
        frames implicitly.
        """
        started_wall = time.time()
        started = time.monotonic()
        errors = 0
        processed = 0
        cv2: Any = None
        if display:
            try:
                cv2 = import_cv2()
            except DependencyMissingError as exc:
                LOG.warning("%s Continuing without preview.", exc)
                display = False
        self.open()
        self.bus.publish(GazeEvent(
            EventType.SESSION_STARTED, 0.0, -1,
            f"Session started with profile {self.config.profile}.",
            {"profile": self.config.profile, "detector": self.pipeline.detector_name},
        ))
        try:
            for packet in self.source:
                if max_frames is not None and processed >= max_frames:
                    break
                elapsed = time.monotonic() - started
                if max_seconds is not None and elapsed >= max_seconds:
                    break
                if (
                    self.config.detect_every_n_frames > 1
                    and packet.frame_index % self.config.detect_every_n_frames != 0
                ):
                    continue
                face: FaceObservation | None = None
                try:
                    faces = self.detector.detect(packet.image, packet.timestamp)
                    if faces:
                        face = max(faces, key=lambda f: f.face_size)
                except Exception as exc:
                    errors += 1
                    self.pipeline.notes.append(f"detector error: {type(exc).__name__}")
                    LOG.exception("Landmark detection failed")
                    self.bus.publish(GazeEvent(
                        EventType.FRAME_ERROR, packet.timestamp, packet.frame_index,
                        f"Landmark detection failed: {exc}",
                    ))
                    if errors >= 30:
                        self.pipeline.notes.append("stopped after 30 detector errors")
                        break
                    continue
                report = self.pipeline.process(
                    face, timestamp=packet.timestamp, frame_index=packet.frame_index
                )
                processed += 1
                if self.logger is not None:
                    self.logger.log(report)
                if on_frame is not None:
                    try:
                        on_frame(report)
                    except Exception:  # pragma: no cover - user callback
                        LOG.exception("on_frame callback failed")
                if on_packet is not None:
                    try:
                        on_packet(packet, report)
                    except Exception:  # pragma: no cover - user callback
                        LOG.exception("on_packet callback failed")
                if display and cv2 is not None and packet.image is not None:
                    canvas = render_overlay(packet.image, report, face, self.config)
                    if self.config.mirror_preview:
                        canvas = cv2.flip(canvas, 1)
                    cv2.imshow("GazeOS | gaze_core", canvas)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        break
        finally:
            duration = time.monotonic() - started
            try:
                self.source.close()
            finally:
                self.detector.close()
                if display and cv2 is not None:
                    try:
                        cv2.destroyAllWindows()
                    except Exception:  # pragma: no cover - GUI backend
                        pass
                self.pipeline.finish(duration)
                summary = self.pipeline.summary(duration, self.source.name, started_wall)
                self.bus.publish(GazeEvent(
                    EventType.SESSION_ENDED, duration, self.pipeline.frame_index,
                    "Session ended.", summary.to_dict(),
                ))
                if self.logger is not None:
                    self.logger.close(summary)
        return summary


# --------------------------------------------------------------------------- #
# Preview rendering
# --------------------------------------------------------------------------- #
def render_overlay(
    image: np.ndarray,
    report: GazeFrame,
    face: FaceObservation | None = None,
    config: GazeConfig | None = None,
) -> np.ndarray:
    """Draw a lightweight debug overlay (requires OpenCV).

    Intended for local development only.  The gaze dot is a schematic of the
    normalized estimate, not a measured point of regard.
    """
    cv2 = import_cv2()
    canvas = image.copy()
    h, w = canvas.shape[:2]
    if face is not None:
        for eye in (face.left_eye, face.right_eye):
            color = (80, 220, 80)
            for point in (eye.inner, eye.outer, eye.upper, eye.lower, eye.iris):
                cv2.circle(canvas, (int(point[0] * w), int(point[1] * h)), 2, color, -1)
            cv2.circle(
                canvas,
                (int(eye.iris[0] * w), int(eye.iris[1] * h)),
                max(2, int(eye.iris_radius * w)),
                (0, 200, 255),
                1,
            )
            cv2.line(
                canvas,
                (int(eye.inner[0] * w), int(eye.inner[1] * h)),
                (int(eye.outer[0] * w), int(eye.outer[1] * h)),
                (180, 180, 180),
                1,
            )
    f = report.features
    lines = [
        f"profile {config.profile if config else report.source}  face={'yes' if report.face_visible else 'no'}",
        f"blinks {f.blink_count}  rate {f.blink_rate_per_min:.1f}/min",
        f"fix {f.fixation_count}  sacc {f.saccade_count}",
        f"PERCLOS-like {f.perclos:.2f}  stability {f.gaze_stability:.2f}",
        f"anomalies {', '.join(report.anomaly_codes) or 'none'}",
    ]
    if report.sample is not None:
        lines.insert(0, f"gaze ({report.sample.gaze_x:+.2f}, {report.sample.gaze_y:+.2f}) conf {report.sample.confidence:.2f}")
        dot_x = int(w * 0.5 + report.sample.gaze_x * w * 0.18)
        dot_y = int(h * 0.18 - report.sample.gaze_y * h * 0.12)
        cv2.circle(canvas, (dot_x, dot_y), 8, (40, 220, 255), -1)
        cv2.circle(canvas, (dot_x, dot_y), 12, (20, 120, 160), 1)
    for i, line in enumerate(lines):
        y = 22 + i * 20
        cv2.putText(canvas, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                    (240, 240, 240), 1, cv2.LINE_AA)
    return canvas


# --------------------------------------------------------------------------- #
# Procedural smoke-test simulator
# --------------------------------------------------------------------------- #
class ProceduralFaceSimulator:
    """Minimal synthetic eye-landmark generator (no images, no model files).

    This exists so the pipeline, CLI and tests run on any machine.  It models
    fixation/saccade gaze shifts and blinks with simple procedural geometry --
    it is *not* a realistic eye renderer.  For controllable, seeded, rendered
    sequences and datasets, use ``eye_twin.py``.
    """

    def __init__(self, fps: float = 30.0, seed: int = 0) -> None:
        self.fps = float(fps)
        self.rng = np.random.default_rng(seed)

    def make_face(
        self,
        t: float,
        gaze_x: float,
        gaze_y: float,
        openness: float,
        yaw_deg: float = 0.0,
    ) -> FaceObservation:
        """Build a FaceObservation whose iris position matches ``gaze``."""
        eye_w = 0.085
        half_h = 0.0135 * (0.12 + 0.88 * _clamp(openness, 0.0, 1.0))
        eyes: dict[str, EyeGeometry] = {}
        for side, cx in (("right", 0.445), ("left", 0.555)):
            cy = 0.5
            half_w = eye_w / 2.0
            iris_x = cx + gaze_x * half_w * 0.55
            iris_y = cy - gaze_y * half_h * 0.85
            eyes[side] = EyeGeometry(
                inner=(cx + half_w * (1 if side == "right" else -1), cy),
                outer=(cx - half_w * (1 if side == "right" else -1), cy),
                upper=(cx, cy - half_h),
                lower=(cx, cy + half_h),
                iris=(iris_x, iris_y),
                iris_radius=0.011,
                side=side,
            )
        blink = 1.0 - _clamp(openness, 0.0, 1.0)
        return FaceObservation(
            timestamp=t,
            left_eye=eyes["left"],
            right_eye=eyes["right"],
            confidence=1.0,
            yaw_deg=yaw_deg,
            blink_score=blink if blink > 0.15 else None,
            face_center=(0.5, 0.5),
            face_size=180.0,
            source="procedural-simulator",
        )

    def observations(
        self, duration_s: float = 20.0, start_time: float = 0.0
    ) -> Iterator[FaceObservation]:
        """Yield temporally coherent synthetic observations."""
        fps = self.fps
        n = max(1, int(duration_s * fps))
        dt = 1.0 / fps
        gaze = np.zeros((n, 2), dtype=np.float64)
        openness = np.ones(n, dtype=np.float64)
        yaw = np.zeros(n, dtype=np.float64)
        target = self.rng.uniform(-0.6, 0.6, size=2)
        next_switch = 0
        for i in range(n):
            if i >= next_switch:
                target = self.rng.uniform(-0.75, 0.75, size=2)
                next_switch = i + max(1, int(fps * self.rng.uniform(0.25, 0.9)))
            gaze[i] = target + self.rng.normal(0.0, 0.015, size=2)
            yaw[i] = 3.0 * math.sin(2.0 * math.pi * i / (fps * 8.0))
        blink_rate_per_min = 15.0
        p_blink = blink_rate_per_min / 60.0 * dt
        i = 0
        while i < n:
            if self.rng.random() < p_blink:
                blink_ms = self.rng.uniform(90.0, 260.0)
                blink_frames = max(2, int(blink_ms / 1000.0 * fps))
                for j in range(blink_frames):
                    if i + j >= n:
                        break
                    phase = (j + 1) / (blink_frames + 1)
                    openness[i + j] = 1.0 - math.sin(math.pi * phase)
                i += blink_frames
            i += 1
        for i in range(n):
            yield self.make_face(
                start_time + i * dt,
                float(gaze[i, 0]),
                float(gaze[i, 1]),
                float(openness[i]),
                float(yaw[i]),
            )


# --------------------------------------------------------------------------- #
# Calibration helpers
# --------------------------------------------------------------------------- #
def run_gui_calibration(
    source: FrameSource,
    detector: EyeLandmarkDetector,
    output: str | Path,
    config: GazeConfig | None = None,
) -> CalibrationModel:
    """Interactive 9-point calibration using an OpenCV window.

    Look at each target and press SPACE.  The most recent 12 reliable frames
    are averaged.  Requires a display and a working camera.
    """
    cv2 = import_cv2()
    cfg = config or make_config("research")
    estimator = GazeEstimator(cfg)
    preview_pipeline = GazePipeline(cfg)
    calibrator = GazeCalibrator()
    targets = [(x, y) for y in (0.15, 0.50, 0.85) for x in (0.15, 0.50, 0.85)]
    window = "GazeOS calibration - look at target, press SPACE"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 960, 600)
    source.open()
    detector.load()
    try:
        for target_index, (tx, ty) in enumerate(targets, start=1):
            latest: FramePacket | None = None
            while True:
                packet = source.read()
                if packet is None:
                    raise CalibrationError("Frame source ended during calibration")
                latest = packet
                faces = detector.detect(packet.image, packet.timestamp)
                canvas = packet.image.copy()
                h, w = canvas.shape[:2]
                cx, cy = int(tx * (w - 1)), int(ty * (h - 1))
                cv2.circle(canvas, (cx, cy), 18, (0, 220, 255), -1)
                cv2.circle(canvas, (cx, cy), 22, (80, 80, 80), 2)
                face = max(faces, key=lambda f: f.face_size) if faces else None
                report = preview_pipeline.process(face, packet.timestamp)
                canvas = render_overlay(canvas, report, face, cfg)
                cv2.putText(
                    canvas,
                    f"target {target_index}/{len(targets)} - SPACE to capture, Q to abort",
                    (16, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA,
                )
                cv2.putText(
                    canvas,
                    f"target {target_index}/{len(targets)} - SPACE to capture, Q to abort",
                    (16, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA,
                )
                cv2.imshow(window, canvas)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    raise CalibrationError("Calibration aborted by user")
                if key == ord(" ") and latest is not None:
                    break
            samples: list[GazeSample] = []
            attempts = 0
            while len(samples) < 12 and attempts < 240:
                attempts += 1
                packet = source.read()
                if packet is None:
                    break
                faces = detector.detect(packet.image, packet.timestamp)
                face = max(faces, key=lambda f: f.face_size) if faces else None
                if face is None:
                    continue
                sample = estimator.estimate(face)
                if sample.confidence >= 0.35 and face.mean_openness > cfg.blink_open_threshold:
                    samples.append(sample)
            if len(samples) < 5:
                LOG.warning("Few reliable samples for target %d; skipped", target_index)
                continue
            mean_sample = GazeSample(
                timestamp=samples[-1].timestamp,
                gaze_x=statistics.fmean(s.gaze_x for s in samples),
                gaze_y=statistics.fmean(s.gaze_y for s in samples),
                confidence=statistics.fmean(s.confidence for s in samples),
                yaw_deg=statistics.fmean(s.yaw_deg for s in samples),
                pitch_deg=statistics.fmean(s.pitch_deg for s in samples),
                roll_deg=statistics.fmean(s.roll_deg for s in samples),
                raw_gaze_x=statistics.fmean(
                    s.raw_gaze_x if s.raw_gaze_x is not None else s.gaze_x for s in samples
                ),
                raw_gaze_y=statistics.fmean(
                    s.raw_gaze_y if s.raw_gaze_y is not None else s.gaze_y for s in samples
                ),
            )
            calibrator.add(mean_sample, tx, ty)
            LOG.info("Captured target %d/%d", target_index, len(targets))
    finally:
        source.close()
        detector.close()
        try:
            cv2.destroyAllWindows()
        except Exception:  # pragma: no cover - GUI backend
            pass
    model = calibrator.fit()
    model.save(output)
    LOG.info(
        "Calibration saved to %s (%d points, RMSE %.3f/%.3f)",
        output, model.n_points, model.rmse[0], model.rmse[1],
    )
    return model


# --------------------------------------------------------------------------- #
# Self test / demo
# --------------------------------------------------------------------------- #
def _demo(
    profile: str,
    seconds: float,
    seed: int,
    jsonl: str | None,
    verbose: bool,
) -> int:
    config = make_config(profile)
    rng_seed = seed
    sim = ProceduralFaceSimulator(seed=rng_seed)
    pipeline = GazePipeline(config)
    logger = GazeSessionLogger(jsonl, {"profile": profile, "generator": "procedural"}) if jsonl else None
    if verbose:
        pipeline.bus.subscribe(
            lambda e: print(f"  [{e.timestamp:7.2f}s] {e.type.value:<18} {e.message}"),
        )
    started = time.time()
    try:
        last_features = TemporalFeatures()
        for face in sim.observations(seconds):
            report = pipeline.process(face)
            last_features = report.features
            if logger:
                logger.log(report)
    finally:
        pipeline.finish(seconds)
        summary = pipeline.summary(seconds, f"simulator:seed={seed}", started)
        if logger:
            logger.close(summary)
    print(summary.format_text())
    print(
        f"  window data  : {last_features.blink_count} blinks, "
        f"{last_features.fixation_count} fixations in the last window"
    )
    print("  note         : synthetic signals; this does not measure accuracy.")
    return 0


def _run_selftest() -> int:
    """Run dependency-free checks of the analytics and orchestration."""
    checks: list[tuple[str, bool, str]] = []

    def check(name: str, condition: bool, detail: str = "") -> None:
        checks.append((name, bool(condition), detail))

    # Geometry round trip
    sim = ProceduralFaceSimulator(seed=1)
    face = sim.make_face(0.0, 0.45, 0.30, 1.0)
    sample = GazeEstimator().estimate(face)
    check("gaze sign/right", sample.gaze_x > 0.0, f"gx={sample.gaze_x:.3f}")
    check("gaze sign/up", sample.gaze_y > 0.0, f"gy={sample.gaze_y:.3f}")
    check("gaze confidence", sample.confidence > 0.5, f"conf={sample.confidence:.2f}")

    # Blink detection
    blinks = BlinkDetector(make_config("research"))
    t = 0.0
    for _ in range(10):
        blinks.update(t, 1.0)
        t += 1 / 30
    for _ in range(9):
        blinks.update(t, 0.02)
        t += 1 / 30
    ended = blinks.update(t, 1.0)
    check(
        "blink detected",
        ended.blink_ended and ended.duration_ms is not None and 250 <= ended.duration_ms <= 350,
        f"duration={ended.duration_ms}",
    )

    # Fixation/saccade
    fix = FixationDetector(make_config("research"))
    t = 0.0
    ended_fixation = None
    for i in range(60):
        s = GazeSample(t, 0.1 + 0.002 * math.sin(i), 0.0, 0.9)
        dyn = fix.update(t, s)
        if dyn.ended_fixation_ms is not None:
            ended_fixation = dyn.ended_fixation_ms
        t += 1 / 60
    for j in range(5):
        s = GazeSample(t, 0.1 + j * 0.2, 0.2, 0.9)
        dyn = fix.update(t, s)
        if dyn.ended_fixation_ms is not None:
            ended_fixation = dyn.ended_fixation_ms
        t += 1 / 60
    check("fixation detected", ended_fixation is not None, f"duration={ended_fixation}")

    # Event bus
    bus = EventBus()
    received: list[GazeEvent] = []
    unsubscribe = bus.subscribe(received.append, [EventType.BLINK])
    bus.publish(GazeEvent(EventType.BLINK, 0, 0, "x"))
    bus.publish(GazeEvent(EventType.SACCADE, 0, 0, "y"))
    unsubscribe()
    bus.publish(GazeEvent(EventType.BLINK, 0, 0, "z"))
    check("event bus filtering", len(received) == 1)

    # Full pipeline over 20 seconds of synthetic data
    pipeline = GazePipeline(make_config("driver_monitoring"))
    events: list[GazeEvent] = []
    pipeline.bus.subscribe(events.append)
    for f in ProceduralFaceSimulator(fps=30, seed=2).observations(20.0):
        pipeline.process(f)
    summary = pipeline.summary(20.0, "selftest")
    check("pipeline frames", summary.frames == 600, f"frames={summary.frames}")
    check("pipeline blinks", summary.blink_count > 0, f"blinks={summary.blink_count}")
    check(
        "pipeline fixations", summary.fixation_count > 0,
        f"fixations={summary.fixation_count}",
    )
    check("pipeline events", bool(events), f"events={len(events)}")
    serialized = json.dumps(summary.to_dict())
    check("summary JSON", len(serialized) > 100)

    # Calibration fit recovers a known linear map.
    cal = GazeCalibrator()
    for i in range(12):
        gx = -0.8 + 1.6 * (i % 4) / 3
        gy = -0.8 + 1.6 * (i // 4) / 2
        s = GazeSample(0.0, gx, gy, 1.0, raw_gaze_x=gx, raw_gaze_y=gy)
        cal.add(s, 0.5 + 0.4 * gx, 0.5 - 0.4 * gy)
    model = cal.fit()
    p = model.predict(GazeSample(0.0, 0.3, -0.2, 1.0, raw_gaze_x=0.3, raw_gaze_y=-0.2))
    check(
        "calibration mapping",
        abs(p.x - (0.5 + 0.4 * 0.3)) < 0.02 and abs(p.y - (0.5 + 0.4 * 0.2)) < 0.02,
        f"point=({p.x:.3f},{p.y:.3f}) rmse={model.rmse}",
    )

    # Config validation
    try:
        make_config("research", window_s=-1)
        check("config validation", False)
    except ValueError:
        check("config validation", True)

    # JSONL logger
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "frames.jsonl"
        logger = GazeSessionLogger(path, {"test": True})
        report = GazePipeline().process(None, timestamp=0.0)
        logger.log(report)
        logger.close()
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        check("jsonl logger", len(lines) == 2, f"lines={len(lines)}")

    passed = sum(1 for _, ok, _ in checks if ok)
    for name, ok, detail in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" ({detail})" if detail else ""))
    print(f"\n{passed}/{len(checks)} checks passed")
    return 0 if passed == len(checks) else 1


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--profile", default="research", choices=PROFILE_NAMES)
    parser.add_argument("--log-level", default="INFO")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GazeOS core eye-movement intelligence (local, rule-based).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("profiles", help="List built-in application profiles.")

    demo = sub.add_parser("demo", help="Run the pipeline on a procedural simulator.")
    _add_common(demo)
    demo.add_argument("--seconds", type=float, default=20.0)
    demo.add_argument("--seed", type=int, default=0)
    demo.add_argument("--jsonl", default=None, help="Write per-frame JSONL here.")
    demo.add_argument("--quiet", action="store_true", help="Hide event log.")

    run = sub.add_parser("run", help="Run on a webcam or video file.")
    _add_common(run)
    run.add_argument("--camera", type=int, default=0)
    run.add_argument("--video", default=None)
    run.add_argument("--model", default=None, help="Face Landmarker .task file.")
    run.add_argument("--max-seconds", type=float, default=None)
    run.add_argument("--max-frames", type=int, default=None)
    run.add_argument("--display", action="store_true", help="Show OpenCV preview.")
    run.add_argument("--calibration", default=None, help="Calibration JSON.")
    run.add_argument("--jsonl", default=None, help="Write per-frame JSONL here.")
    run.add_argument("--save-frames", default=None,
                     help="Directory for raw frames (requires explicit consent).")
    run.add_argument("--store-identifiable-frames", action="store_true")
    run.add_argument("--detect-every", type=int, default=1)

    cal = sub.add_parser("calibrate", help="Fit a calibration model.")
    cal.add_argument("--points", default=None, help="CSV of raw->target pairs (headless).")
    cal.add_argument("--camera", type=int, default=0)
    cal.add_argument("--model", default=None)
    cal.add_argument("--output", default="calibration.json")

    sub.add_parser("selftest", help="Run dependency-free self tests.")
    return parser


def _run_live(args: argparse.Namespace) -> int:
    config = make_config(
        args.profile,
        detect_every_n_frames=max(1, args.detect_every),
    )
    if args.save_frames:
        if not args.store_identifiable_frames:
            print(
                "Refusing to save raw frames without --store-identifiable-frames. "
                "Raw video is identifiable personal data; obtain consent first.",
                file=sys.stderr,
            )
            return 2
        if not config.store_identifiable:
            config.store_identifiable = True
        LOG.warning(
            "Raw frame storage enabled: frames may identify people. "
            "Store securely and delete per your consent agreement."
        )
    try:
        calibration = CalibrationModel.load(args.calibration) if args.calibration else None
    except CalibrationError as exc:
        LOG.error("%s", exc)
        return 2
    if args.video:
        source: FrameSource = VideoFileSource(args.video)
    else:
        source = CameraSource(
            args.camera, config.camera_width, config.camera_height, config.camera_fps
        )
    detector = MediaPipeEyeLandmarkDetector(
        model_path=args.model, num_faces=config.max_faces
    )
    logger = GazeSessionLogger(
        args.jsonl,
        {"profile": config.profile, "source": source.name, "model": str(detector.model_path)},
    ) if args.jsonl else None
    session = GazeSession(source, config, detector, calibration, logger=logger)
    frame_writer: Callable[[FramePacket, GazeFrame], None] | None = None
    if args.save_frames:
        save_dir = Path(args.save_frames)
        save_dir.mkdir(parents=True, exist_ok=True)
        state = {"n": 0}

        def write_packet(packet: FramePacket, report: GazeFrame) -> None:
            if packet.image is None:
                return
            state["n"] += 1
            if state["n"] % 15 != 0:  # ~2 fps at 30 fps capture
                return
            cv2 = import_cv2()
            cv2.imwrite(str(save_dir / f"frame_{packet.frame_index:07d}.jpg"), packet.image)

        frame_writer = write_packet
    try:
        summary = session.run(
            max_frames=args.max_frames,
            max_seconds=args.max_seconds,
            display=args.display,
            on_packet=frame_writer,
        )
    except KeyboardInterrupt:
        LOG.info("Interrupted by user.")
        return 130
    except (GazeOSError, FileNotFoundError) as exc:
        LOG.error("%s", exc)
        return 2
    print(summary.format_text())
    return 0


def _run_calibrate(args: argparse.Namespace) -> int:
    if args.points:
        try:
            model = GazeCalibrator.from_csv(args.points).fit()
        except CalibrationError as exc:
            LOG.error("%s", exc)
            return 2
        model.save(args.output)
        print(f"Saved calibration to {args.output} (RMSE {model.rmse[0]:.4f}/{model.rmse[1]:.4f})")
        return 0
    config = make_config("research")
    source = CameraSource(args.camera, config.camera_width, config.camera_height, config.camera_fps)
    detector = MediaPipeEyeLandmarkDetector(model_path=args.model)
    try:
        run_gui_calibration(source, detector, args.output, config)
    except (GazeOSError, CalibrationError) as exc:
        LOG.error("%s", exc)
        return 2
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(getattr(args, "log_level", "INFO")), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.command == "profiles":
        for name in PROFILE_NAMES:
            cfg = make_config(name)
            print(
                f"{name:<20} window={cfg.window_s:>4.0f}s  "
                f"perclos(warn/crit)={cfg.perclos_warn:.2f}/{cfg.perclos_critical:.2f}  "
                f"blink={cfg.blink_rate_low:.0f}-{cfg.blink_rate_high:.0f}/min"
            )
        return 0
    if args.command == "demo":
        return _demo(args.profile, args.seconds, args.seed, args.jsonl, not args.quiet)
    if args.command == "run":
        return _run_live(args)
    if args.command == "calibrate":
        return _run_calibrate(args)
    if args.command == "selftest":
        return _run_selftest()
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
