"""GazeAssist: an experimental, real-time scene-understanding prototype.

**This is not a certified mobility aid.**  It is a hackathon / research
prototype that turns a camera feed into short spoken and haptic cues about
detected obstacles.  It must never replace a white cane, guide dog, human
assistance or orientation-and-mobility training.  Detection, distance and path
estimates can be wrong, late or missing.  See the repository README and the
banner printed at startup.

What it does
------------
* Captures frames from a webcam, video file, or an entirely simulated scene.
* Runs a pluggable object detector (Ultralytics YOLO, OpenCV Haar fallback,
  motion heuristic, or a scripted mock for demos).
* Optionally estimates monocular depth with an ONNX model, or geometric
  distance from bounding-box size and known object heights.
* Fuses detections, proximity, closing speed and time-to-collision into
  ranked hazards (left / ahead / right), plus a coarse walkable-path score.
* Emits concise speech via the operating system's TTS and optional haptic
  cues through a console or a serial device.

What it deliberately does **not** do
------------------------------------
* It does not route video frames through an LLM, and it does not use cloud
  services.  All processing is local.
* It does not claim certified safety, medical benefit or measured accuracy.
  Latency and FPS are *measured*, never assumed.
* It does not require a blind user to control navigation with gaze.  Optional
  eye analytics are disabled by default and intended only for sighted
  operators/researchers studying fatigue; they never drive navigation.

Run ``python gaze_assist.py --help``.  Start with the hardware-free demo::

    python gaze_assist.py demo --seconds 30

Live camera use requires explicit risk acknowledgement::

    python gaze_assist.py run --camera 0 --detector auto --voice \\
        --acknowledge-risks
"""

from __future__ import annotations

import abc
import argparse
import importlib.util
import json
import logging
import math
import os
import shutil
import statistics
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from gaze_core import (
    CameraSource,
    FramePacket,
    FrameSource,
    GazeFrame,
    GazeOSError,
    GazePipeline,
    MediaPipeEyeLandmarkDetector,
    VideoFileSource,
    import_cv2,
    make_config,
)

LOG = logging.getLogger("gazeos.assist")

__all__ = [
    "AlertManager",
    "AssistConfig",
    "AssistPipeline",
    "AssistStats",
    "BBox",
    "ConsoleOutput",
    "DepthMap",
    "Detection",
    "GeometricDepthEstimator",
    "Hazard",
    "MockDetector",
    "MockSceneSource",
    "MotionObstacleDetector",
    "NullOutput",
    "OnnxDepthEstimator",
    "RiskConfig",
    "RiskEngine",
    "SceneReport",
    "SerialHapticOutput",
    "SpeechOutput",
    "UltralyticsYOLODetector",
    "create_detector",
]

SAFETY_BANNER = """
+---------------------------------------------------------------------------+
|  GazeAssist - EXPERIMENTAL PROTOTYPE - NOT A CERTIFIED MOBILITY AID       |
|                                                                           |
|  Detection and distance estimates can be wrong, late or absent.           |
|  Never use this as a substitute for a white cane, guide dog, human        |
|  assistance, or orientation & mobility training. Do not rely on it near   |
|  traffic, stairs, platforms or other hazardous environments.              |
+---------------------------------------------------------------------------+
"""

BBoxTuple = tuple[float, float, float, float]


def _clamp(value: float, low: float, high: float) -> float:
    return low if value < low else high if value > high else value


def _label_text(label: str) -> str:
    return label.replace("_", " ")


# --------------------------------------------------------------------------- #
# Data contracts
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BBox:
    """Axis-aligned box in normalized image coordinates (y points down)."""

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        x1 = _clamp(float(self.x1), 0.0, 1.0)
        y1 = _clamp(float(self.y1), 0.0, 1.0)
        x2 = _clamp(float(self.x2), 0.0, 1.0)
        y2 = _clamp(float(self.y2), 0.0, 1.0)
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Invalid bounding box: {(self.x1, self.y1, self.x2, self.y2)}")
        object.__setattr__(self, "x1", x1)
        object.__setattr__(self, "y1", y1)
        object.__setattr__(self, "x2", x2)
        object.__setattr__(self, "y2", y2)

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2.0

    @property
    def area(self) -> float:
        return self.width * self.height

    def iou(self, other: "BBox") -> float:
        dx1 = max(self.x1, other.x1)
        dy1 = max(self.y1, other.y1)
        dx2 = min(self.x2, other.x2)
        dy2 = min(self.y2, other.y2)
        if dx2 <= dx1 or dy2 <= dy1:
            return 0.0
        inter = (dx2 - dx1) * (dy2 - dy1)
        return inter / max(self.area + other.area - inter, 1e-9)

    def as_tuple(self) -> BBoxTuple:
        return (self.x1, self.y1, self.x2, self.y2)


def bbox_from_pixels(
    x1: float, y1: float, x2: float, y2: float, width: int, height: int
) -> BBox:
    """Convert pixel coordinates to a normalized, clamped :class:`BBox`."""
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    return BBox(x1 / width, y1 / height, x2 / width, y2 / height)


@dataclass(frozen=True)
class Detection:
    """A detector-agnostic scene observation.

    ``distance_m`` is optional: some backends estimate it, others leave it to
    the depth/geometric layer.  ``confidence`` is only meaningful as a
    probability for learned detectors; fallbacks document theirs as heuristic.
    """

    label: str
    confidence: float
    box: BBox
    distance_m: float | None = None
    track_id: int | None = None
    source: str = "detector"

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if self.distance_m is not None and self.distance_m < 0:
            raise ValueError("distance_m must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "confidence": round(self.confidence, 3),
            "box": [round(v, 4) for v in self.box.as_tuple()],
            "distance_m": round(self.distance_m, 2) if self.distance_m is not None else None,
            "source": self.source,
        }


@dataclass
class DepthMap:
    """Monocular depth estimate.

    ``inverse_depth`` is normalized so larger values are **closer**, which is
    convenient for free-space checks.  It is relative, not metric, unless a
    calibrated model supplies ``metric_distance``.
    """

    inverse_depth: np.ndarray
    source: str
    metric_distance: np.ndarray | None = None

    def closeness_in(self, box: BBox) -> float | None:
        """Mean relative closeness inside a box (0..1), or ``None``."""
        h, w = self.inverse_depth.shape[:2]
        x1 = int(_clamp(box.x1, 0, 1) * (w - 1))
        x2 = max(x1 + 1, int(_clamp(box.x2, 0, 1) * (w - 1)))
        y1 = int(_clamp(box.y1, 0, 1) * (h - 1))
        y2 = max(y1 + 1, int(_clamp(box.y2, 0, 1) * (h - 1)))
        crop = self.inverse_depth[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        return float(np.clip(np.mean(crop), 0.0, 1.0))

    def distance_in(self, box: BBox) -> float | None:
        """Median metric distance inside a box, if this map is metric."""
        if self.metric_distance is None:
            return None
        h, w = self.metric_distance.shape[:2]
        x1 = int(_clamp(box.x1, 0, 1) * (w - 1))
        x2 = max(x1 + 1, int(_clamp(box.x2, 0, 1) * (w - 1)))
        y1 = int(_clamp(box.y1, 0, 1) * (h - 1))
        y2 = max(y1 + 1, int(_clamp(box.y2, 0, 1) * (h - 1)))
        crop = self.metric_distance[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        value = float(np.median(crop))
        return value if math.isfinite(value) and value > 0 else None


@dataclass(frozen=True)
class Hazard:
    """A ranked collision or navigation risk."""

    label: str
    direction: str  # "left" | "ahead" | "right"
    distance_m: float | None
    ttc_s: float | None
    risk: float
    priority: int  # 1=critical, 2=warning, 3=caution
    box: BBox
    source: str = "risk-engine"

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "direction": self.direction,
            "distance_m": round(self.distance_m, 2) if self.distance_m is not None else None,
            "ttc_s": round(self.ttc_s, 2) if self.ttc_s is not None else None,
            "risk": round(self.risk, 3),
            "priority": self.priority,
        }


@dataclass(frozen=True)
class SceneReport:
    """Structured frame-level scene understanding output."""

    timestamp: float
    hazards: tuple[Hazard, ...]
    detections: tuple[Detection, ...]
    walkable_score: float
    path_hint: str
    summary: str
    inference_ms: float
    detector: str
    depth_backend: str
    degraded: bool = False
    notes: tuple[str, ...] = ()

    @property
    def top_hazard(self) -> Hazard | None:
        return self.hazards[0] if self.hazards else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "t": round(self.timestamp, 4),
            "hazards": [h.to_dict() for h in self.hazards],
            "detections": [d.to_dict() for d in self.detections],
            "walkable_score": round(self.walkable_score, 3),
            "path_hint": self.path_hint,
            "summary": self.summary,
            "inference_ms": round(self.inference_ms, 2),
            "detector": self.detector,
            "depth_backend": self.depth_backend,
            "degraded": self.degraded,
            "notes": list(self.notes),
        }


# --------------------------------------------------------------------------- #
# Object detection backends
# --------------------------------------------------------------------------- #
class ObjectDetector(abc.ABC):
    """Interface for pluggable object detectors."""

    name: str = "detector"
    ready: bool = False

    def load(self) -> None:
        """Load weights/resources.  Must raise a clear error on failure."""

    @abc.abstractmethod
    def detect(self, frame: np.ndarray, timestamp: float) -> list[Detection]:
        """Return detections in normalized image coordinates."""

    def close(self) -> None:
        pass

    @property
    def notes(self) -> list[str]:
        return []


HAZARD_CLASSES: dict[str, str] = {
    "person": "person",
    "bicycle": "bicycle",
    "motorcycle": "motorcycle",
    "car": "car",
    "bus": "bus",
    "truck": "truck",
    "train": "train",
    "chair": "chair",
    "bench": "bench",
    "dog": "dog",
    "potted plant": "obstacle",
    "stop sign": "obstacle",
    "traffic light": "obstacle",
    # These labels are not in COCO. They are accepted so a custom YOLO model
    # trained for accessibility scenes can feed the risk engine unchanged.
    "door": "door",
    "doorway": "door",
    "stairs": "stairs",
    "staircase": "stairs",
    "steps": "stairs",
    "curb": "obstacle",
    "pole": "obstacle",
    "traffic cone": "obstacle",
    "wall": "obstacle",
}


class UltralyticsYOLODetector(ObjectDetector):
    """Ultralytics YOLO detector (optional dependency, CPU by default).

    Install with ``pip install ultralytics``.  Passing ``yolo11n.pt`` lets
    Ultralytics download the COCO weights on first use (network required);
    a local weights file can also be supplied.  COCO has no door/stairs
    classes -- those labels are supported by the risk engine and demo, but
    require a custom model or another backend.
    """

    def __init__(
        self,
        weights: str | Path | None = None,
        device: str = "cpu",
        conf: float = 0.35,
        imgsz: int = 416,
        max_detections: int = 20,
    ) -> None:
        self.weights = str(
            weights or os.environ.get("GAZEASSIST_YOLO_MODEL") or "yolo11n.pt"
        )
        self.device = device
        self.conf = conf
        self.imgsz = imgsz
        self.max_detections = max_detections
        self.name = f"ultralytics:{Path(self.weights).name}"
        self._model: Any = None
        self._names: dict[int, str] = {}
        self._notes: list[str] = []

    @property
    def notes(self) -> list[str]:
        return list(self._notes)

    def load(self) -> None:
        if self._model is not None:
            return
        try:
            from ultralytics import YOLO  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - environment specific
            raise GazeOSError(
                "Ultralytics is not installed. Install with: pip install ultralytics. "
                "Without it, use --detector haar or --detector motion."
            ) from exc
        path = Path(self.weights)
        looks_like_path = ("/" in self.weights) or ("\\" in self.weights)
        if looks_like_path and not path.exists():
            raise GazeOSError(
                f"YOLO weights not found: {path}. Provide a valid file with --weights."
            )
        if not looks_like_path and not path.exists():
            self._notes.append(
                f"{self.weights} not found locally; Ultralytics may download it (network)."
            )
            LOG.warning("YOLO weights %s not local; attempting download", self.weights)
        try:
            self._model = YOLO(str(path))
        except Exception as exc:  # pragma: no cover - model/runtime specific
            raise GazeOSError(f"Could not load YOLO model {path}: {exc}") from exc
        if str(self.device).lower().startswith("cuda"):
            try:
                import torch  # type: ignore[import-not-found]

                if not torch.cuda.is_available():
                    self._notes.append("CUDA requested but unavailable; using CPU.")
                    LOG.warning("CUDA requested but unavailable; falling back to CPU")
                    self.device = "cpu"
            except ImportError:
                self.device = "cpu"
        names = getattr(self._model, "names", {}) or {}
        self._names = names if isinstance(names, dict) else dict(enumerate(names))
        self.ready = True
        LOG.info("Loaded %s on %s", self.name, self.device)

    def detect(self, frame: np.ndarray, timestamp: float) -> list[Detection]:
        if self._model is None:
            self.load()
        try:
            results = self._model.predict(
                source=frame,
                conf=self.conf,
                imgsz=self.imgsz,
                device=self.device,
                max_det=self.max_detections,
                verbose=False,
            )
        except Exception as exc:
            LOG.exception("YOLO inference failed")
            raise GazeOSError(f"YOLO inference failed: {exc}") from exc
        if not results:
            return []
        boxes = getattr(results[0], "boxes", None)
        if boxes is None:
            return []
        H, W = frame.shape[:2]
        detections: list[Detection] = []
        for box in boxes:
            try:
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
            except Exception:  # pragma: no cover - backend shape change
                continue
            raw_name = self._names.get(cls_id, str(cls_id))
            label = HAZARD_CLASSES.get(str(raw_name).lower())
            if label is None:
                continue
            try:
                bbox = bbox_from_pixels(x1, y1, x2, y2, W, H)
            except ValueError:
                continue
            detections.append(
                Detection(label, conf, bbox, distance_m=None, source="yolo")
            )
        return detections

    def close(self) -> None:
        self._model = None
        self.ready = False


class HaarPedestrianDetector(ObjectDetector):
    """Zero-download OpenCV Haar fallback for people and faces.

    Only frontal faces and full-body pedestrians are detected; there are no
    vehicles, doors, stairs or generic obstacles.  Confidence is a fixed
    heuristic, **not** a calibrated probability.  Useful as a last-resort
    fallback, not as a scene-understanding system.
    """

    name = "opencv-haar"

    def __init__(self, min_faces: int = 40) -> None:
        self.min_faces = min_faces
        self._body: Any = None
        self._face: Any = None

    def load(self) -> None:
        if self.ready:
            return
        cv2 = import_cv2()
        base = Path(cv2.data.haarcascades)
        body_path = base / "haarcascade_fullbody.xml"
        face_path = base / "haarcascade_frontalface_default.xml"
        if not body_path.exists() or not face_path.exists():
            raise GazeOSError(
                f"OpenCV Haar cascades not found under {base}. "
                "Reinstall opencv-python or use --detector motion."
            )
        self._body = cv2.CascadeClassifier(str(body_path))
        self._face = cv2.CascadeClassifier(str(face_path))
        if self._body.empty() or self._face.empty():
            raise GazeOSError("OpenCV could not load Haar cascade files")
        self.ready = True
        LOG.info("Loaded OpenCV Haar fallback detector")

    def detect(self, frame: np.ndarray, timestamp: float) -> list[Detection]:
        if not self.ready:
            self.load()
        cv2 = import_cv2()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        H, W = gray.shape[:2]
        raw: list[tuple[int, int, int, int]] = []
        for cascade, min_size in (
            (self._body, (48, 96)),
            (self._face, (self.min_faces, self.min_faces)),
        ):
            found = cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=4, minSize=min_size
            )
            raw.extend([tuple(int(v) for v in box) for box in found])
        boxes: list[BBox] = []
        for x, y, w, h in sorted(raw, key=lambda b: b[2] * b[3], reverse=True):
            try:
                bbox = bbox_from_pixels(x, y, x + w, y + h, W, H)
            except ValueError:
                continue
            if any(bbox.iou(existing) > 0.4 for existing in boxes):
                continue
            boxes.append(bbox)
        # Fixed heuristic confidence; documented as such.
        return [Detection("person", 0.55, b, source="haar(heuristic)") for b in boxes[:10]]

    def close(self) -> None:
        self._body = self._face = None
        self.ready = False


class MotionObstacleDetector(ObjectDetector):
    """Frame-difference "obstacle" heuristic for a static camera.

    Detects moving regions (or large brightness changes) in the lower part of
    the frame.  It cannot name objects and produces false positives when the
    camera moves.  Confidence is a size-based heuristic.  This is a fallback
    for environments with OpenCV but no learned detector weights.
    """

    name = "opencv-motion"

    def __init__(self, min_area_ratio: float = 0.012) -> None:
        self.min_area_ratio = min_area_ratio
        self._previous: np.ndarray | None = None

    def load(self) -> None:
        import_cv2()
        self.ready = True

    def detect(self, frame: np.ndarray, timestamp: float) -> list[Detection]:
        if not self.ready:
            self.load()
        cv2 = import_cv2()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (9, 9), 0)
        if self._previous is None:
            self._previous = gray
            return []
        diff = cv2.absdiff(self._previous, gray)
        self._previous = gray
        _, mask = cv2.threshold(diff, 22, 255, cv2.THRESH_BINARY)
        mask = cv2.dilate(mask, np.ones((9, 9), np.uint8), iterations=2)
        H, W = mask.shape[:2]
        mask[: int(H * 0.25), :] = 0  # ignore the sky/background band
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections: list[Detection] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area_ratio * W * H:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if y < H * 0.3:
                continue
            try:
                bbox = bbox_from_pixels(x, y, x + w, y + h, W, H)
            except ValueError:
                continue
            conf = float(np.clip(0.3 + bbox.area * 2.0, 0.3, 0.85))
            detections.append(
                Detection("obstacle", conf, bbox, source="motion(heuristic)")
            )
        return sorted(detections, key=lambda d: d.box.area, reverse=True)[:8]

    def close(self) -> None:
        self._previous = None
        self.ready = False


MOCK_SCENARIO_SECONDS = 30.0


def scripted_detections(t: float) -> list[Detection]:
    """Deterministic walking scenario used by the hardware-free demo."""
    phase = t % MOCK_SCENARIO_SECONDS
    out: list[Detection] = [
        Detection(
            "person",
            0.72,
            BBox(0.06, 0.44, 0.17, 0.94),
            distance_m=6.5,
            source="mock",
        )
    ]
    if 3.0 <= phase < 9.0:
        p = (phase - 3.0) / 6.0
        width = 0.16 + 0.10 * p
        out.append(
            Detection(
                "chair",
                0.78,
                BBox(0.5 - width / 2, 0.58, 0.5 + width / 2, 0.95),
                distance_m=5.0 - 4.2 * p,
                source="mock",
            )
        )
    if 9.0 <= phase < 14.0:
        p = (phase - 9.0) / 5.0
        width = 0.12 + 0.16 * p
        out.append(
            Detection(
                "car",
                0.86,
                BBox(0.62 - width / 2, 0.48, 0.62 + width / 2, 0.94),
                distance_m=10.0 - 8.0 * p,
                source="mock",
            )
        )
    if 14.0 <= phase < 19.0:
        p = (phase - 14.0) / 5.0
        cx = 0.24 + 0.48 * p
        out.append(
            Detection(
                "person",
                0.82,
                BBox(cx - 0.07, 0.40, cx + 0.07, 0.95),
                distance_m=1.7,
                source="mock",
            )
        )
    if 19.0 <= phase < 24.0:
        p = (phase - 19.0) / 5.0
        out.append(
            Detection(
                "stairs",
                0.80,
                BBox(0.30, 0.68 + 0.04 * p, 0.70, 0.99),
                distance_m=1.6 - 0.5 * p,
                source="mock",
            )
        )
    if 24.0 <= phase < 28.0:
        out.append(
            Detection(
                "door",
                0.74,
                BBox(0.08, 0.24, 0.34, 0.92),
                distance_m=4.5,
                source="mock",
            )
        )
    return out


class NullDetector(ObjectDetector):
    """No-op detector used after all fallbacks fail; keeps the loop alive."""

    name = "none"
    ready = True

    def load(self) -> None:
        self.ready = True

    def detect(self, frame: np.ndarray, timestamp: float) -> list[Detection]:
        return []


class MockDetector(ObjectDetector):
    """Scripted detector for demos/tests; ignores the pixels entirely.

    ``start_offset`` shifts the scenario clock so demos/tests can begin
    immediately before an interesting event.
    """

    name = "mock-scenario"

    def __init__(self, start_offset: float = 0.0) -> None:
        self.start_offset = start_offset

    def load(self) -> None:
        self.ready = True

    def detect(self, frame: np.ndarray, timestamp: float) -> list[Detection]:
        self.ready = True
        return scripted_detections(timestamp + self.start_offset)

    def close(self) -> None:
        self.ready = False


def create_detector(
    spec: str = "auto",
    weights: str | Path | None = None,
    conf: float = 0.35,
    imgsz: int = 416,
    device: str = "cpu",
) -> ObjectDetector:
    """Build a detector by name with a documented fallback chain.

    ``auto`` prefers Ultralytics when installed, then Haar, then motion.  It
    never silently pretends to understand scenes when only a heuristic is
    available: the chosen backend name is reported in every report.
    """
    spec = spec.lower()
    if spec in {"yolo", "ultralytics"}:
        return UltralyticsYOLODetector(weights, device, conf, imgsz)
    if spec in {"haar", "opencv-haar"}:
        return HaarPedestrianDetector()
    if spec in {"motion", "opencv-motion"}:
        return MotionObstacleDetector()
    if spec == "mock":
        return MockDetector()
    if spec != "auto":
        raise ValueError(f"Unknown detector {spec!r}")
    if importlib.util.find_spec("ultralytics") is not None:
        LOG.info("auto detector: Ultralytics YOLO")
        return UltralyticsYOLODetector(weights, device, conf, imgsz)
    if importlib.util.find_spec("cv2") is not None:
        LOG.warning("auto detector: Ultralytics unavailable; using Haar fallback")
        return HaarPedestrianDetector()
    raise GazeOSError(
        "No detector backend available. Install opencv-python (Haar/motion "
        "fallback) or ultralytics (YOLO)."
    )


# --------------------------------------------------------------------------- #
# Depth backends
# --------------------------------------------------------------------------- #
class DepthEstimator(abc.ABC):
    """Interface for pluggable depth/distance estimation."""

    name: str = "depth"
    ready: bool = False

    def load(self) -> None:
        pass

    def estimate(self, frame: np.ndarray) -> DepthMap | None:
        return None

    def distance_for(
        self, detection: Detection, frame_shape: tuple[int, int], depth: DepthMap | None
    ) -> float | None:
        return detection.distance_m

    def close(self) -> None:
        pass

    @property
    def notes(self) -> list[str]:
        return []


ASSUMED_HEIGHTS_M: dict[str, float] = {
    "person": 1.70,
    "driver": 1.70,
    "bicycle": 1.05,
    "motorcycle": 1.20,
    "car": 1.50,
    "truck": 3.00,
    "bus": 3.20,
    "train": 3.50,
    "chair": 0.90,
    "bench": 0.90,
    "door": 2.00,
    "dog": 0.55,
}


class GeometricDepthEstimator(DepthEstimator):
    """Bounding-box-height distance estimate (monocular, flat-ground assumption).

    ``distance = assumed_height * focal_px / box_height_px`` with
    ``focal_px = (H/2) / tan(vertical_fov/2)``.  It is only as good as the
    assumed object height, and it fails for partially visible objects or
    unusual camera pitch.  Objects without a height model (e.g. stairs) get
    ``None`` -- the risk engine then uses a conservative size-based cue
    without claiming a distance.
    """

    name = "geometric-height"

    def __init__(self, vertical_fov_deg: float = 55.0) -> None:
        self.vertical_fov_deg = vertical_fov_deg
        self.ready = True

    def distance_for(
        self, detection: Detection, frame_shape: tuple[int, int], depth: DepthMap | None
    ) -> float | None:
        if detection.distance_m is not None:
            return detection.distance_m
        assumed = ASSUMED_HEIGHTS_M.get(detection.label)
        if assumed is None:
            return None
        frame_h = frame_shape[0]
        box_h_px = detection.box.height * frame_h
        if box_h_px < 6:
            return None
        focal = (frame_h / 2.0) / math.tan(math.radians(self.vertical_fov_deg) / 2.0)
        distance = assumed * focal / box_h_px
        if not math.isfinite(distance):
            return None
        return float(_clamp(distance, 0.3, 40.0))

    @property
    def notes(self) -> list[str]:
        return ["Distances from assumed object heights; not measured depth."]


class OnnxDepthEstimator(DepthEstimator):
    """Relative monocular depth from an ONNX model via OpenCV DNN.

    Weights are **not** bundled (no license/quality claim).  Export a MiDaS- or
    Depth-Anything-style single-image ONNX model and pass ``--depth-model``.
    Output is treated as relative inverse depth; it is used for free-space
    checks, not metric distance.
    """

    name = "onnx-relative-depth"

    def __init__(
        self,
        model_path: str | Path,
        input_size: int = 256,
        scale: float = 1.0 / 255.0,
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] = (0.229, 0.224, 0.225),
        invert: bool = False,
    ) -> None:
        self.model_path = Path(model_path)
        self.input_size = int(input_size)
        self.scale = scale
        self.mean = mean
        self.std = std
        self.invert = invert
        self._net: Any = None

    def load(self) -> None:
        if self.ready:
            return
        if not self.model_path.exists():
            raise GazeOSError(
                f"Depth model not found: {self.model_path}. Provide an ONNX "
                "depth model with --depth-model, or use --depth geometric."
            )
        cv2 = import_cv2()
        try:
            self._net = cv2.dnn.readNetFromONNX(str(self.model_path))
        except Exception as exc:
            raise GazeOSError(f"Could not load ONNX depth model: {exc}") from exc
        self.ready = True
        LOG.info("Loaded ONNX depth model %s", self.model_path)

    def estimate(self, frame: np.ndarray) -> DepthMap | None:
        if self._net is None:
            self.load()
        cv2 = import_cv2()
        resized = cv2.resize(frame, (self.input_size, self.input_size))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) * self.scale
        rgb = (rgb - np.asarray(self.mean, dtype=np.float32)) / np.asarray(
            self.std, dtype=np.float32
        )
        blob = np.transpose(rgb, (2, 0, 1))[None, ...]
        self._net.setInput(blob)
        try:
            output = self._net.forward()
        except Exception as exc:
            LOG.exception("Depth inference failed")
            raise GazeOSError(f"Depth inference failed: {exc}") from exc
        depth = np.squeeze(output).astype(np.float32)
        if depth.ndim != 2:
            raise GazeOSError(f"Unexpected depth output shape: {output.shape}")
        depth = cv2.resize(depth, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
        lo, hi = float(np.percentile(depth, 2)), float(np.percentile(depth, 98))
        if hi - lo < 1e-6:
            normalized = np.zeros_like(depth)
        else:
            normalized = np.clip((depth - lo) / (hi - lo), 0.0, 1.0)
        if self.invert:
            normalized = 1.0 - normalized
        # Common MiDaS-style outputs are inverse depth (larger = closer).
        return DepthMap(inverse_depth=normalized, source=self.name)

    def distance_for(
        self, detection: Detection, frame_shape: tuple[int, int], depth: DepthMap | None
    ) -> float | None:
        if detection.distance_m is not None:
            return detection.distance_m
        if depth is not None:
            return depth.distance_in(detection.box)
        return None

    def close(self) -> None:
        self._net = None
        self.ready = False


class NoDepthEstimator(DepthEstimator):
    """Depth disabled; detections must supply distances or run relative-only."""

    name = "none"

    def __init__(self) -> None:
        self.ready = True


def create_depth_estimator(
    spec: str = "geometric",
    model_path: str | Path | None = None,
    invert: bool = False,
) -> DepthEstimator:
    spec = spec.lower()
    if spec == "none":
        return NoDepthEstimator()
    if spec == "geometric":
        return GeometricDepthEstimator()
    if spec in {"onnx", "midas"}:
        if model_path is None:
            raise GazeOSError(
                "--depth onnx requires --depth-model /path/to/depth.onnx"
            )
        return OnnxDepthEstimator(Path(model_path), invert=invert)
    raise ValueError(f"Unknown depth backend {spec!r}")


# --------------------------------------------------------------------------- #
# Risk engine
# --------------------------------------------------------------------------- #
def _default_hazard_weights() -> dict[str, float]:
    return {
        "person": 0.95,
        "bicycle": 1.0,
        "motorcycle": 1.0,
        "car": 1.0,
        "truck": 1.0,
        "bus": 1.0,
        "train": 1.0,
        "stairs": 1.0,
        "door": 0.45,
        "chair": 0.65,
        "bench": 0.65,
        "obstacle": 0.75,
        "dog": 0.6,
    }


@dataclass
class RiskConfig:
    """Thresholds for proximity, TTC and path analysis.

    Values are prototype defaults, not validated stopping-safety margins.
    """

    caution_m: float = 3.0
    warning_m: float = 1.8
    critical_m: float = 0.9
    ttc_warning_s: float = 3.0
    ttc_critical_s: float = 1.6
    lateral_left_max: float = 0.38
    lateral_right_min: float = 0.62
    assumption_window_s: float = 2.0
    max_hazards: int = 5
    hazard_weights: dict[str, float] = field(default_factory=_default_hazard_weights)

    def __post_init__(self) -> None:
        if not (0 < self.critical_m < self.warning_m < self.caution_m):
            raise ValueError("Require 0 < critical_m < warning_m < caution_m")
        if not (0 < self.ttc_critical_s < self.ttc_warning_s):
            raise ValueError("Require 0 < ttc_critical_s < ttc_warning_s")
        if not (0 < self.lateral_left_max < self.lateral_right_min < 1):
            raise ValueError("Lateral zone boundaries must be ordered within (0, 1)")


class RiskEngine:
    """Combines distance, closing speed, TTC, direction and object class.

    It maintains short per-(label, direction) distance histories to estimate
    closing speed.  With only relative depth (no metres), it falls back to a
    conservative bounding-box-size cue and reports no distance.  Time-to-
    collision assumes constant closing speed; it is not a guarantee.
    """

    def __init__(self, config: RiskConfig | None = None) -> None:
        self.config = config or RiskConfig()
        self._history: dict[tuple[str, int | str], deque[tuple[float, float]]] = {}
        self._tracks: dict[int, tuple[BBox, str, float]] = {}
        self._next_track_id = 1

    def _track_for(self, det: Detection, timestamp: float) -> int | str:
        """Associate detections with short-lived identities by IoU.

        Detectors without tracking (Haar/motion, YOLO predict mode) otherwise
        mix histories of different objects in the same zone, which can invent
        a false time-to-collision.  IoU association is deliberately simple and
        is not a production tracker.
        """
        if det.track_id is not None:
            return f"ext:{det.track_id}"
        best_id: int | None = None
        best_iou = 0.18
        for track_id, (box, label, last_seen) in list(self._tracks.items()):
            if label != det.label or timestamp - last_seen > 1.5:
                continue
            iou = det.box.iou(box)
            if iou > best_iou:
                best_iou = iou
                best_id = track_id
        if best_id is None:
            best_id = self._next_track_id
            self._next_track_id += 1
        self._tracks[best_id] = (det.box, det.label, timestamp)
        # Prune stale tracks.
        for track_id, (_, _, last_seen) in list(self._tracks.items()):
            if timestamp - last_seen > 2.5:
                del self._tracks[track_id]
        return best_id

    def _direction(self, box: BBox) -> str:
        if box.center_x < self.config.lateral_left_max:
            return "left"
        if box.center_x > self.config.lateral_right_min:
            return "right"
        return "ahead"

    def _closing_speed(self, key: tuple[str, str], t: float, distance: float) -> float:
        history = self._history.setdefault(key, deque(maxlen=12))
        history.append((t, distance))
        while history and t - history[0][0] > self.config.assumption_window_s:
            history.popleft()
        if len(history) < 3 or history[-1][0] - history[0][0] < 0.25:
            return 0.0
        times = np.array([p[0] for p in history])
        distances = np.array([p[1] for p in history])
        try:
            slope = float(np.polyfit(times, distances, 1)[0])
        except np.linalg.LinAlgError:
            return 0.0
        return max(0.0, -slope)

    def evaluate(
        self,
        detections: Sequence[Detection],
        depth: DepthMap | None,
        frame_shape: tuple[int, int],
        timestamp: float,
    ) -> list[Hazard]:
        """Rank detections into hazards (highest priority/risk first).

        Distances must already be attached to detections; the pipeline fuses
        depth and geometric backends before calling this method.
        """
        cfg = self.config
        candidates: dict[tuple[str, str], Hazard] = {}
        for det in detections:
            direction = self._direction(det.box)
            distance = det.distance_m
            closing = 0.0
            ttc: float | None = None
            if distance is not None:
                track = self._track_for(det, timestamp)
                closing = self._closing_speed((det.label, track), timestamp, distance)
                if closing > 0.15 and distance > 0.0:
                    ttc = distance / closing
            weight = cfg.hazard_weights.get(det.label, 0.7)
            direction_factor = 1.0 if direction == "ahead" else 0.85
            if distance is not None:
                prox = _clamp(
                    (cfg.caution_m - distance) / (cfg.caution_m - cfg.critical_m),
                    0.0,
                    1.0,
                )
                ttc_score = 0.0
                if ttc is not None:
                    ttc_score = _clamp(
                        (cfg.ttc_warning_s - ttc)
                        / (cfg.ttc_warning_s - cfg.ttc_critical_s),
                        0.0,
                        1.0,
                    )
                size_score = _clamp((det.box.height - 0.12) / 0.6, 0.0, 1.0)
                risk = _clamp(
                    weight
                    * direction_factor
                    * (0.65 * prox + 0.25 * ttc_score + 0.10 * size_score),
                    0.0,
                    1.0,
                )
                if (
                    distance <= cfg.critical_m
                    or (ttc is not None and ttc <= cfg.ttc_critical_s)
                    or (det.label == "stairs" and distance <= 1.6)
                ):
                    priority = 1
                elif distance <= cfg.warning_m or (
                    ttc is not None and ttc <= cfg.ttc_warning_s
                ):
                    priority = 2
                elif risk >= 0.22:
                    priority = 3
                else:
                    continue
            else:
                # Relative-only mode: a big, low, central box is treated
                # conservatively; no distance is invented.
                size_score = _clamp((det.box.height - 0.18) / 0.6, 0.0, 1.0)
                if det.box.y2 < 0.72:
                    size_score *= 0.5
                risk = _clamp(
                    weight * direction_factor * (0.22 + 0.55 * size_score),
                    0.0,
                    1.0,
                )
                if risk < 0.30:
                    continue
                priority = 2 if (det.box.y2 > 0.86 and direction == "ahead") else 3
                ttc = None
            hazard = Hazard(
                label=det.label,
                direction=direction,
                distance_m=distance,
                ttc_s=ttc,
                risk=risk,
                priority=priority,
                box=det.box,
                source=det.source,
            )
            key = (det.label, direction)
            existing = candidates.get(key)
            if (
                existing is None
                or hazard.priority < existing.priority
                or (hazard.priority == existing.priority and hazard.risk > existing.risk)
            ):
                candidates[key] = hazard
        hazards = sorted(
            candidates.values(), key=lambda h: (h.priority, -h.risk)
        )
        return hazards[: cfg.max_hazards]

    def path_analysis(
        self,
        detections: Sequence[Detection],
        hazards: Sequence[Hazard],
        depth: DepthMap | None,
    ) -> tuple[float, str, str]:
        """Coarse walkable-path score (0..1), descriptive hint and summary.

        This is not a navigable map.  Hints describe the camera view and must
        not be treated as safe route instructions.
        """
        left_penalty = 0.0
        center_penalty = 0.0
        right_penalty = 0.0
        for hazard in hazards:
            severity = {1: 0.65, 2: 0.35, 3: 0.15}.get(hazard.priority, 0.1)
            if hazard.direction == "left":
                left_penalty += severity
            elif hazard.direction == "right":
                right_penalty += severity
            else:
                center_penalty += severity
        if depth is not None:
            center = BBox(0.38, 0.58, 0.62, 0.95)
            closeness = depth.closeness_in(center)
            if closeness is not None:
                center_penalty += 0.35 * closeness
        score = _clamp(1.0 - center_penalty - 0.25 * min(left_penalty, right_penalty), 0.0, 1.0)
        if center_penalty <= 0.05:
            hint = "no immediate obstacle detected ahead"
            summary = "No immediate obstacle detected in the center view."
        elif center_penalty >= 0.6:
            if left_penalty + 0.1 < right_penalty:
                hint = "center blocked; the left side of the view appears clearer"
            elif right_penalty + 0.1 < left_penalty:
                hint = "center blocked; the right side of the view appears clearer"
            else:
                hint = "center blocked; no clearly clearer side detected"
            summary = "Obstacle close in the center view."
        else:
            hint = "obstacle ahead; approach with caution"
            summary = "Possible obstacle ahead."
        return score, hint, summary


# --------------------------------------------------------------------------- #
# Output channels
# --------------------------------------------------------------------------- #
class AssistOutput(abc.ABC):
    """Interface for speech/haptic output.  Never block the vision loop."""

    def open(self) -> None:
        pass

    @abc.abstractmethod
    def announce(self, text: str, priority: int = 3) -> None:
        """Emit a short message; ``priority`` 1 (critical) .. 3 (info)."""

    def haptic(self, direction: str, priority: int) -> None:
        pass

    def close(self) -> None:
        pass


class NullOutput(AssistOutput):
    """Discards all output (tests, benchmark harnesses)."""

    def announce(self, text: str, priority: int = 3) -> None:
        pass


class ConsoleOutput(AssistOutput):
    """Developer-facing text output with optional haptic log lines."""

    def __init__(self, echo_haptics: bool = True) -> None:
        self.echo_haptics = echo_haptics
        self._lock = threading.Lock()

    def announce(self, text: str, priority: int = 3) -> None:
        tag = {1: "STOP", 2: "WARN", 3: "INFO"}.get(priority, "INFO")
        with self._lock:
            print(f"[{tag}] {text}", flush=True)

    def haptic(self, direction: str, priority: int) -> None:
        if not self.echo_haptics:
            return
        level = {1: "strong", 2: "medium", 3: "light"}.get(priority, "light")
        with self._lock:
            print(f"[haptic] {direction} ({level})", flush=True)


class SpeechOutput(AssistOutput):
    """Non-blocking OS text-to-speech with priority dropping and rate limits.

    Uses ``say`` (macOS), ``espeak-ng``/``espeak``/``spd-say`` (Linux) or
    Windows SAPI through PowerShell.  No cloud TTS and no LLM: utterances are
    short templates built by :class:`AlertManager`.  Speech duration is not
    known in advance, so announcements are queued and the newest critical
    message can displace lower-priority ones.
    """

    def __init__(self, echo: bool = True, max_queue: int = 4) -> None:
        self.echo = echo
        self.max_queue = max_queue
        self._command: list[str] | None = None
        self._items: list[tuple[int, int, str]] = []
        self._condition = threading.Condition()
        self._seq = 0
        self._running = False
        self._thread: threading.Thread | None = None
        self.last_error: str | None = None

    @staticmethod
    def detect_command() -> list[str] | None:
        if sys.platform == "darwin" and shutil.which("say"):
            return ["say", "-r", "220"]
        for name in ("espeak-ng", "espeak"):
            if shutil.which(name):
                return [name, "-s", "165"]
        if shutil.which("spd-say"):
            return ["spd-say", "-r", "0"]
        if sys.platform.startswith("win") and shutil.which("powershell"):
            return ["powershell", "-NoProfile", "-Command"]
        return None

    @classmethod
    def available(cls) -> bool:
        return cls.detect_command() is not None

    def open(self) -> None:
        if self._running:
            return
        command = self.detect_command()
        if command is None:
            raise GazeOSError(
                "No system text-to-speech found. Use console output, or install "
                "espeak-ng (Linux) / use macOS 'say' / Windows SAPI."
            )
        self._command = command
        self._running = True
        self._thread = threading.Thread(target=self._worker, name="gazeassist-tts", daemon=True)
        self._thread.start()

    def _speak(self, text: str) -> None:
        assert self._command is not None
        try:
            if self._command[-1] == "-Command":  # Windows PowerShell / SAPI
                script = (
                    "Add-Type -AssemblyName System.Speech;"
                    "$s=New-Object System.Speech.Synthesis.SpeechSynthesizer;"
                    "$s.Speak([Console]::In.ReadToEnd())"
                )
                subprocess.run(
                    self._command + [script],
                    input=text,
                    text=True,
                    timeout=20,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            else:
                subprocess.run(
                    self._command + [text],
                    timeout=20,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
        except Exception as exc:  # pragma: no cover - OS/TTS specific
            self.last_error = str(exc)
            LOG.debug("TTS failed: %s", exc)

    def _worker(self) -> None:
        while True:
            with self._condition:
                while self._running and not self._items:
                    self._condition.wait(timeout=0.2)
                if not self._running and not self._items:
                    return
                self._items.sort(key=lambda item: (item[0], item[1]))
                _, _, text = self._items.pop(0)
            self._speak(text)

    def announce(self, text: str, priority: int = 3) -> None:
        text = " ".join(text.replace("\n", " ").split())[:160]
        if not text:
            return
        if self.echo:
            print(f"[voice] {text}", flush=True)
        with self._condition:
            self._seq += 1
            entry = (int(priority), self._seq, text)
            if len(self._items) >= self.max_queue:
                worst = max(self._items, key=lambda item: item[0])
                if entry[0] < worst[0]:
                    self._items.remove(worst)
                else:
                    return
            self._items.append(entry)
            self._condition.notify()

    def haptic(self, direction: str, priority: int) -> None:
        if self.echo:
            level = {1: "strong", 2: "medium", 3: "light"}.get(priority, "light")
            print(f"[haptic] {direction} ({level})", flush=True)

    def close(self) -> None:
        with self._condition:
            self._running = False
            self._items.clear()
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None


class SerialHapticOutput(AssistOutput):
    """Optional serial haptic protocol: ``direction,priority\\n``.

    Requires ``pyserial`` and a microcontroller-driven actuator (belt, wrist
    band).  The protocol is intentionally trivial; this class has not been
    validated on specific hardware and does not provide safety guarantees.
    """

    def __init__(self, port: str, baud: int = 115200) -> None:
        self.port = port
        self.baud = baud
        self._serial: Any = None

    def open(self) -> None:
        try:
            import serial  # type: ignore[import-not-found]
        except ImportError as exc:
            raise GazeOSError(
                "pyserial is required for --haptic-port. Install: pip install pyserial"
            ) from exc
        try:
            self._serial = serial.Serial(self.port, self.baud, timeout=0.1)
        except Exception as exc:
            raise GazeOSError(f"Cannot open haptic serial port {self.port}: {exc}") from exc
        LOG.info("Haptic serial port %s open", self.port)

    def announce(self, text: str, priority: int = 3) -> None:
        pass

    def haptic(self, direction: str, priority: int) -> None:
        if self._serial is None:
            return
        try:
            self._serial.write(f"{direction},{priority}\n".encode("ascii"))
        except Exception:
            LOG.debug("Haptic write failed", exc_info=True)

    def close(self) -> None:
        if self._serial is not None:
            try:
                self._serial.close()
            except Exception:
                pass
            self._serial = None


class MultiOutput(AssistOutput):
    """Fan out announcements/haptics to several channels."""

    def __init__(self, outputs: Sequence[AssistOutput]) -> None:
        self.outputs = list(outputs)

    def open(self) -> None:
        for output in self.outputs:
            output.open()

    def announce(self, text: str, priority: int = 3) -> None:
        for output in self.outputs:
            try:
                output.announce(text, priority)
            except Exception:
                LOG.exception("Output channel failed")

    def haptic(self, direction: str, priority: int) -> None:
        for output in self.outputs:
            try:
                output.haptic(direction, priority)
            except Exception:
                LOG.exception("Haptic channel failed")

    def close(self) -> None:
        for output in reversed(self.outputs):
            try:
                output.close()
            except Exception:
                LOG.exception("Output close failed")


# --------------------------------------------------------------------------- #
# Alerting
# --------------------------------------------------------------------------- #
@dataclass
class AlertPolicy:
    """Throttling and wording policy for spoken/haptic alerts."""

    general_interval_s: float = 2.0
    critical_interval_s: float = 2.5
    repeat_same_key_s: float = 5.0
    clear_interval_s: float = 12.0
    haptic_interval_s: float = 1.0
    clear_enabled: bool = True


def format_hazard(hazard: Hazard) -> str:
    """Short, template-based utterance.  No LLM is involved."""
    label = _label_text(hazard.label)
    direction = {
        "left": "on your left",
        "ahead": "ahead",
        "right": "on your right",
    }.get(hazard.direction, "ahead")
    if hazard.priority == 1:
        if hazard.label == "stairs":
            return "Stairs very close ahead. Stop."
        if hazard.distance_m is not None and hazard.ttc_s is not None:
            return (
                f"{label.capitalize()} approaching {direction}, "
                f"{hazard.distance_m:.1f} meters. Stop."
            )
        if hazard.distance_m is not None:
            return f"{label.capitalize()}, {hazard.distance_m:.1f} meters {direction}. Stop."
        return f"{label.capitalize()} very close {direction}. Stop."
    if hazard.ttc_s is not None and hazard.ttc_s <= 3.5:
        return f"{label.capitalize()} approaching {direction}, about {hazard.ttc_s:.0f} seconds."
    if hazard.distance_m is not None:
        return f"{label.capitalize()}, {hazard.distance_m:.1f} meters {direction}."
    return f"Possible {label} {direction}."


class AlertManager:
    """Turns ranked hazards into throttled speech and haptic cues.

    Deduplicates by (label, direction), respects cooldowns, lets critical
    hazards bypass the general throttle, and avoids flooding the TTS queue.
    It is deliberately conservative: with no hazard it announces only that
    the *view* appears clear, never that it is safe to walk.
    """

    def __init__(
        self,
        output: AssistOutput,
        policy: AlertPolicy | None = None,
    ) -> None:
        self.output = output
        self.policy = policy or AlertPolicy()
        self._last_general = -1e9
        self._last_critical = -1e9
        self._last_haptic = -1e9
        self._last_clear = -1e9
        self._last_key_time: dict[tuple[str, str], float] = {}

    def update(self, report: SceneReport) -> list[str]:
        now = report.timestamp
        emitted: list[str] = []
        top = report.top_hazard
        if top is not None:
            key = (top.label, top.direction)
            same_key_ok = now - self._last_key_time.get(key, -1e9) >= self.policy.repeat_same_key_s
            if top.priority == 1:
                if now - self._last_critical >= self.policy.critical_interval_s:
                    text = format_hazard(top)
                    self.output.announce(text, 1)
                    emitted.append(text)
                    self._last_critical = now
                    self._last_general = now
                    self._last_key_time[key] = now
                if now - self._last_haptic >= self.policy.haptic_interval_s:
                    self.output.haptic(top.direction, 1)
                    self._last_haptic = now
            elif same_key_ok and now - self._last_general >= self.policy.general_interval_s:
                text = format_hazard(top)
                self.output.announce(text, top.priority)
                emitted.append(text)
                self._last_general = now
                self._last_key_time[key] = now
                if now - self._last_haptic >= self.policy.haptic_interval_s:
                    self.output.haptic(top.direction, top.priority)
                    self._last_haptic = now
            self._last_clear = now
        elif self.policy.clear_enabled:
            if (
                report.walkable_score >= 0.65
                and now - self._last_clear >= self.policy.clear_interval_s
                and now - self._last_general >= self.policy.general_interval_s
            ):
                text = "No immediate obstacle detected ahead."
                self.output.announce(text, 3)
                emitted.append(text)
                self._last_clear = now
                self._last_general = now
        return emitted


# --------------------------------------------------------------------------- #
# Demo scene + capture
# --------------------------------------------------------------------------- #
class MockSceneSource(FrameSource):
    """Paced synthetic camera feed so the demo needs no webcam or GPU."""

    name = "mock-scene"

    def __init__(self, fps: float = 30.0, width: int = 640, height: int = 480) -> None:
        self.fps = fps
        self.width = width
        self.height = height
        self._index = -1
        self._start = 0.0

    def open(self) -> None:
        self._start = time.monotonic()
        self._index = -1

    def _frame(self, index: int) -> np.ndarray:
        w, h = self.width, self.height
        yy_full, xx_full = np.mgrid[0:h, 0:w].astype(np.float32)
        yy = yy_full / max(h - 1, 1)
        xx = xx_full / max(w - 1, 1)
        sky_top = np.array([120, 150, 190], dtype=np.float32)
        sky_bottom = np.array([205, 220, 230], dtype=np.float32)
        ground_top = np.array([120, 125, 115], dtype=np.float32)
        ground_bottom = np.array([72, 78, 70], dtype=np.float32)
        horizon = 0.45
        sky_t = np.clip(yy / horizon, 0, 1)[:, :, None]
        sky = sky_top[None, None, :] * (1 - sky_t) + sky_bottom[None, None, :] * sky_t
        ground_t = np.clip((yy - horizon) / (1 - horizon), 0, 1)[:, :, None]
        ground = ground_top[None, None, :] * (1 - ground_t) + ground_bottom[None, None, :] * ground_t
        frame = np.where((yy < horizon)[:, :, None], sky, ground).astype(np.float32)
        # Center path trapezoid.
        path_width = 0.10 + 0.28 * np.clip((yy - horizon) / (1 - horizon), 0, 1)
        path_mask = (np.abs(xx - 0.5) < path_width) & (yy > horizon)
        frame[path_mask] = frame[path_mask] * 0.55 + np.array([150, 145, 135]) * 0.45
        # Sun and a subtle walking bob to keep the scene non-static.
        phase = index / max(self.fps, 1.0)
        cx = 0.78 + 0.02 * math.sin(phase)
        cy = 0.16
        sun = (xx - cx) ** 2 + ((yy - cy) * (h / w)) ** 2 < 0.0025
        frame[sun] = np.array([250, 235, 190])
        return np.clip(frame, 0, 255).astype(np.uint8)

    def read(self) -> FramePacket | None:
        self._index += 1
        target = self._start + (self._index + 1) / max(self.fps, 1.0)
        delay = target - time.monotonic()
        if delay > 0:
            time.sleep(min(delay, 0.05))
        return FramePacket(
            frame_index=self._index,
            timestamp=time.monotonic(),
            image=self._frame(self._index),
        )


class CaptureWorker(threading.Thread):
    """Continuously reads frames and keeps only the newest (frame skipping)."""

    def __init__(self, source: FrameSource) -> None:
        super().__init__(name="gazeassist-capture", daemon=True)
        self.source = source
        self._lock = threading.Lock()
        self._latest: FramePacket | None = None
        self._stop_event = threading.Event()
        self.ended = False
        self.error: str | None = None
        self.captured = 0

    def run(self) -> None:
        try:
            while not self._stop_event.is_set():
                packet = self.source.read()
                if packet is None:
                    self.ended = True
                    return
                with self._lock:
                    self._latest = packet  # drop any older unprocessed frame
                    self.captured += 1
        except Exception as exc:  # pragma: no cover - hardware specific
            self.error = str(exc)
            self.ended = True
            LOG.exception("Capture worker failed")

    def latest(self) -> FramePacket | None:
        with self._lock:
            packet = self._latest
            self._latest = None
            return packet

    def stop(self) -> None:
        self._stop_event.set()


# --------------------------------------------------------------------------- #
# Optional eye analytics (research/operator only)
# --------------------------------------------------------------------------- #
class EyeAnalyticsMonitor:
    """Optional gaze_core analytics on the same camera feed.

    This is **not** an input method for blind users and never influences
    hazard detection or alerts about obstacles.  It is intended for sighted
    researchers/operators studying blink and fatigue proxies, with consent.
    """

    def __init__(
        self,
        profile: str = "screen_fatigue",
        model_path: str | Path | None = None,
        every_n_frames: int = 6,
    ) -> None:
        self.config = make_config(profile)
        self.every_n_frames = max(1, every_n_frames)
        self.detector = MediaPipeEyeLandmarkDetector(model_path=model_path)
        self.pipeline = GazePipeline(self.config)
        self.pipeline.detector_name = self.detector.name
        self._counter = 0
        self._announced: set[str] = set()
        self.enabled = False
        self.last_report: GazeFrame | None = None

    def load(self) -> None:
        self.detector.load()
        self.enabled = True

    def process(self, image: np.ndarray, timestamp: float) -> str | None:
        """Return a fatigue-related message at most once per code, if any."""
        if not self.enabled:
            return None
        self._counter += 1
        if self._counter % self.every_n_frames != 0:
            return None
        faces = self.detector.detect(image, timestamp)
        face = max(faces, key=lambda f: f.face_size) if faces else None
        report = self.pipeline.process(face, timestamp)
        self.last_report = report
        fatigue_codes = {
            "perclos_elevated",
            "perclos_critical",
            "prolonged_closure",
            "blink_duration_elevated",
            "blink_rate_elevated",
            "blink_rate_low",
        }
        for anomaly in report.anomalies:
            if anomaly.code in self._announced:
                continue
            relevant = (
                anomaly.code in fatigue_codes
                or anomaly.code.startswith("baseline_shift:perclos")
                or anomaly.code.startswith("baseline_shift:blink")
            )
            if relevant and anomaly.severity >= 0.55:
                self._announced.add(anomaly.code)
                return f"Eye fatigue indicator: {anomaly.message}"
        return None

    def close(self) -> None:
        self.detector.close()
        self.enabled = False


# --------------------------------------------------------------------------- #
# Application pipeline
# --------------------------------------------------------------------------- #
@dataclass
class AssistConfig:
    """Runtime configuration for the assistive prototype."""

    detector: str = "auto"
    weights: str | None = None
    device: str = "cpu"
    conf_threshold: float = 0.35
    imgsz: int = 416
    depth: str = "geometric"
    depth_model: str | None = None
    depth_invert: bool = False
    detector_hz: float = 6.0
    depth_hz: float = 2.0
    loop_hz: float = 30.0
    max_detections: int = 20
    camera_index: int = 0
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: float = 30.0
    display: bool = False
    eye_analytics: bool = False
    eye_profile: str = "screen_fatigue"
    eye_model: str | None = None
    eye_every_n_frames: int = 6
    alert_policy: AlertPolicy = field(default_factory=AlertPolicy)
    risk: RiskConfig = field(default_factory=RiskConfig)

    def __post_init__(self) -> None:
        if self.detector_hz <= 0 or self.depth_hz < 0 or self.loop_hz <= 0:
            raise ValueError("Inference/loop rates must be positive (depth may be 0)")
        if not 0.0 < self.conf_threshold <= 1.0:
            raise ValueError("conf_threshold must be in (0, 1]")
        if self.camera_width < 160 or self.camera_height < 120:
            raise ValueError("Camera resolution is too small")


@dataclass
class AssistStats:
    """Measured session statistics.  Values are observed, not guaranteed."""

    duration_s: float
    frames_captured: int
    frames_processed: int
    detector_runs: int
    depth_runs: int
    mean_detector_ms: float
    p95_detector_ms: float
    mean_depth_ms: float
    capture_fps: float
    processed_fps: float
    announcements: int
    critical_alerts: int
    detector: str
    depth: str
    degraded: bool
    errors: int
    notes: list[str] = field(default_factory=list)

    def format_text(self) -> str:
        lines = [
            "GazeAssist session (EXPERIMENTAL - not a mobility aid)",
            f"  duration        : {self.duration_s:.1f} s",
            f"  frames          : {self.frames_captured} captured, "
            f"{self.frames_processed} processed",
            f"  measured rate   : capture {self.capture_fps:.1f} fps, "
            f"processed {self.processed_fps:.1f} fps",
            f"  detector        : {self.detector} "
            f"({self.detector_runs} runs, mean {self.mean_detector_ms:.1f} ms, "
            f"p95 {self.p95_detector_ms:.1f} ms)",
            f"  depth           : {self.depth} "
            f"({self.depth_runs} runs, mean {self.mean_depth_ms:.1f} ms)",
            f"  alerts          : {self.announcements} "
            f"({self.critical_alerts} critical)",
            f"  degraded        : {self.degraded}; errors: {self.errors}",
        ]
        if self.notes:
            lines.append("  notes           : " + "; ".join(self.notes))
        lines.append("  latency figures are measured on this machine for this run.")
        return "\n".join(lines)


class AssistLogger:
    """Opt-in JSONL logging of scene reports (no images)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("w", encoding="utf-8")
        self._handle.write(
            json.dumps(
                {
                    "record": "header",
                    "schema": "gazeos.assist.scene.v1",
                    "created_at": time.time(),
                    "privacy": "scene detections/hazards only; no images",
                }
            )
            + "\n"
        )

    def log(self, report: SceneReport) -> None:
        self._handle.write(json.dumps(report.to_dict(), separators=(",", ":")) + "\n")

    def close(self, stats: AssistStats | None = None) -> None:
        self._handle.flush()
        self._handle.close()
        if stats is not None:
            out = self.path.with_suffix(self.path.suffix + ".summary.json")
            out.write_text(json.dumps(asdict(stats), indent=2), encoding="utf-8")


def render_assist_overlay(frame: np.ndarray, report: SceneReport) -> np.ndarray:
    """Draw boxes, hazards and a persistent experimental banner (OpenCV)."""
    cv2 = import_cv2()
    canvas = frame.copy()
    h, w = canvas.shape[:2]
    colors = {1: (40, 40, 230), 2: (30, 160, 245), 3: (60, 200, 200)}
    for det in report.detections:
        box = det.box
        x1, y1, x2, y2 = int(box.x1 * w), int(box.y1 * h), int(box.x2 * w), int(box.y2 * h)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (200, 200, 200), 1)
    for hazard in report.hazards:
        box = hazard.box
        x1, y1, x2, y2 = int(box.x1 * w), int(box.y1 * h), int(box.x2 * w), int(box.y2 * h)
        color = colors.get(hazard.priority, (200, 200, 200))
        thickness = 3 if hazard.priority == 1 else 2
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
        text = f"{hazard.label} {hazard.direction}"
        if hazard.distance_m is not None:
            text += f" {hazard.distance_m:.1f}m"
        if hazard.ttc_s is not None:
            text += f" ttc {hazard.ttc_s:.0f}s"
        cv2.putText(canvas, text, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, text, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, color, 1, cv2.LINE_AA)
    # Central path guide.
    cv2.line(canvas, (int(w * 0.38), h), (int(w * 0.46), int(h * 0.55)), (180, 220, 180), 1)
    cv2.line(canvas, (int(w * 0.62), h), (int(w * 0.54), int(h * 0.55)), (180, 220, 180), 1)
    banner = "EXPERIMENTAL - NOT A MOBILITY AID"
    cv2.rectangle(canvas, (0, 0), (w, 30), (30, 30, 160), -1)
    cv2.putText(canvas, banner, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    stats = (
        f"{report.detector} | {report.depth_backend} | "
        f"path {report.walkable_score:.2f} | infer {report.inference_ms:.0f} ms"
    )
    cv2.putText(canvas, stats, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(canvas, stats, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (240, 240, 240), 1, cv2.LINE_AA)
    return canvas


class AssistPipeline:
    """Asynchronous-ish capture + rate-limited inference + local alerting.

    Capture runs in a thread and always keeps only the newest frame.  The main
    loop runs the detector at ``detector_hz`` and depth at ``depth_hz``;
    inference is never queued behind old frames.  Errors trigger documented
    fallbacks (learned -> Haar -> motion, depth disabled) rather than crashing.
    """

    def __init__(
        self,
        source: FrameSource,
        config: AssistConfig,
        output: AssistOutput,
        detector: ObjectDetector | None = None,
        depth: DepthEstimator | None = None,
        eye_monitor: EyeAnalyticsMonitor | None = None,
        logger: AssistLogger | None = None,
    ) -> None:
        self.source = source
        self.config = config
        self.output = output
        self.detector = detector or create_detector(
            config.detector, config.weights, config.conf_threshold,
            config.imgsz, config.device,
        )
        self.depth = depth or create_depth_estimator(
            config.depth, config.depth_model, config.depth_invert
        )
        self.eye_monitor = eye_monitor
        self.logger = logger
        self.risk = RiskEngine(config.risk)
        self.alerts = AlertManager(output, config.alert_policy)
        self._detector_errors = 0
        self._depth_errors = 0
        self._eye_errors = 0
        self._degraded = False
        self._notes: list[str] = []
        self._stats_announcements = 0
        self._stats_critical = 0
        self._detector_error_total = 0

    def open(self) -> None:
        self.source.open()
        try:
            self.detector.load()
        except Exception as exc:
            LOG.warning("Detector load failed (%s); trying fallbacks", exc)
            self._notes.append(f"detector unavailable: {exc}")
            self._swap_fallback()
        try:
            self.depth.load()
        except Exception as exc:
            LOG.warning("Depth load failed (%s); using geometric fallback", exc)
            self._notes.append(f"depth unavailable: {exc}")
            self.depth = GeometricDepthEstimator()
            self._degraded = True
        if self.eye_monitor is not None:
            try:
                self.eye_monitor.load()
            except Exception as exc:
                self._notes.append(f"eye analytics disabled: {exc}")
                LOG.warning("Eye analytics disabled: %s", exc)
        self.output.open()

    def _swap_fallback(self) -> None:
        old = getattr(self.detector, "name", "detector")
        try:
            self.detector.close()
        except Exception:
            pass
        candidates = ["haar", "motion"] if "motion" not in old else ["haar"]
        for name in candidates:
            try:
                fallback = create_detector(name, conf=self.config.conf_threshold)
                fallback.load()
                self.detector = fallback
                self._degraded = True
                self._notes.append(f"detector fallback: {old} -> {fallback.name}")
                LOG.warning("Fell back from %s to %s", old, fallback.name)
                return
            except Exception as exc:
                LOG.warning("Fallback %s unavailable: %s", name, exc)
        self._notes.append("no detector available after errors; running without detection")
        self.detector = NullDetector()
        self._degraded = True

    def _detect(self, image: np.ndarray, timestamp: float) -> list[Detection]:
        try:
            detections = self.detector.detect(image, timestamp)
            self._detector_errors = 0
        except Exception as exc:
            self._detector_errors += 1
            self._detector_error_total += 1
            LOG.error("Detector error %d: %s", self._detector_errors, exc)
            if self._detector_errors >= 3:
                self._swap_fallback()
                self._detector_errors = 0
            return []
        filtered = [
            d for d in detections
            if d.confidence >= self.config.conf_threshold
        ]
        filtered.sort(key=lambda d: d.box.area, reverse=True)
        return filtered[: self.config.max_detections]

    def _estimate_depth(self, image: np.ndarray) -> DepthMap | None:
        try:
            result = self.depth.estimate(image)
            self._depth_errors = 0
            return result
        except Exception as exc:
            self._depth_errors += 1
            LOG.error("Depth error %d: %s", self._depth_errors, exc)
            if self._depth_errors >= 3:
                self._notes.append("depth disabled after repeated errors")
                self.depth = NoDepthEstimator()
                self._degraded = True
            return None

    def _annotate_distances(
        self,
        detections: Sequence[Detection],
        depth_map: DepthMap | None,
        frame_shape: tuple[int, int],
    ) -> list[Detection]:
        annotated: list[Detection] = []
        for det in detections:
            distance = det.distance_m
            if distance is None and depth_map is not None:
                distance = depth_map.distance_in(det.box)
            if distance is None:
                distance = self.depth.distance_for(det, frame_shape, depth_map)
            annotated.append(replace(det, distance_m=distance))
        return annotated

    def run(
        self,
        max_seconds: float | None = None,
        on_report: Callable[[SceneReport], None] | None = None,
    ) -> AssistStats:
        """Run until stopped, returning measured statistics."""
        started = time.monotonic()
        detector_times: list[float] = []
        depth_times: list[float] = []
        frames_processed = 0
        detector_runs = 0
        depth_runs = 0
        errors = 0
        last_detector = -1e9
        last_depth = -1e9
        last_depth_map: DepthMap | None = None
        display_enabled = self.config.display
        cv2: Any = None
        if display_enabled:
            try:
                cv2 = import_cv2()
                cv2.namedWindow("GazeAssist (experimental)", cv2.WINDOW_NORMAL)
            except Exception as exc:
                LOG.warning("Display unavailable (%s); continuing headless", exc)
                display_enabled = False
        self.open()
        worker = CaptureWorker(self.source)
        worker.start()
        self._notes.extend(self.detector.notes)
        self._notes.extend(self.depth.notes)
        try:
            while True:
                now = time.monotonic()
                elapsed = now - started
                if max_seconds is not None and elapsed >= max_seconds:
                    break
                packet = worker.latest()
                if packet is None:
                    if worker.ended:
                        if worker.error:
                            errors += 1
                            self._notes.append(f"capture error: {worker.error}")
                        break
                    time.sleep(min(0.005, 1.0 / self.config.loop_hz))
                    continue
                if packet.image is None:
                    continue
                want_detector = elapsed - last_detector >= 1.0 / self.config.detector_hz
                if not want_detector:
                    continue
                if worker.error:
                    errors += 1
                    self._notes.append(f"capture error: {worker.error}")
                    break
                t0 = time.perf_counter()
                detections = self._detect(packet.image, elapsed)
                detector_times.append((time.perf_counter() - t0) * 1000.0)
                detector_runs += 1
                last_detector = elapsed

                want_depth = (
                    self.config.depth_hz > 0
                    and elapsed - last_depth >= 1.0 / self.config.depth_hz
                )
                if want_depth:
                    t1 = time.perf_counter()
                    last_depth_map = self._estimate_depth(packet.image)
                    depth_times.append((time.perf_counter() - t1) * 1000.0)
                    depth_runs += 1
                    last_depth = elapsed

                annotated = self._annotate_distances(
                    detections, last_depth_map, packet.image.shape[:2]
                )
                hazards = self.risk.evaluate(
                    annotated, last_depth_map, packet.image.shape[:2], elapsed
                )
                walkable, path_hint, summary = self.risk.path_analysis(
                    annotated, hazards, last_depth_map
                )
                inference_ms = (time.perf_counter() - t0) * 1000.0
                report = SceneReport(
                    timestamp=elapsed,
                    hazards=tuple(hazards),
                    detections=tuple(annotated),
                    walkable_score=walkable,
                    path_hint=path_hint,
                    summary=summary,
                    inference_ms=inference_ms,
                    detector=getattr(self.detector, "name", "detector"),
                    depth_backend=getattr(self.depth, "name", "none"),
                    degraded=self._degraded,
                    notes=tuple(self._notes[-3:]),
                )
                emitted = self.alerts.update(report)
                self._stats_announcements += len(emitted)
                if emitted and hazards and hazards[0].priority == 1:
                    self._stats_critical += 1
                if self.eye_monitor is not None and self.eye_monitor.enabled:
                    try:
                        message = self.eye_monitor.process(packet.image, elapsed)
                        if message:
                            self.output.announce(message, 3)
                    except Exception as exc:
                        self._eye_errors += 1
                        if self._eye_errors >= 3:
                            self.eye_monitor.close()
                            self._notes.append("eye analytics disabled after errors")
                            LOG.warning("Eye analytics disabled: %s", exc)
                frames_processed += 1
                if self.logger is not None:
                    self.logger.log(report)
                if on_report is not None:
                    try:
                        on_report(report)
                    except Exception:
                        LOG.exception("on_report callback failed")
                if display_enabled and cv2 is not None:
                    canvas = render_assist_overlay(packet.image, report)
                    cv2.imshow("GazeAssist (experimental)", canvas)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        break
                # Risk evaluation is cheap; sleep to avoid a busy loop.
                time.sleep(min(0.01, 1.0 / max(self.config.loop_hz, 1.0)))
        finally:
            worker.stop()
            worker.join(timeout=1.0)
            try:
                self.detector.close()
            except Exception:
                pass
            try:
                self.depth.close()
            except Exception:
                pass
            if self.eye_monitor is not None:
                try:
                    self.eye_monitor.close()
                except Exception:
                    pass
            try:
                self.source.close()
            except Exception:
                pass
            try:
                self.output.close()
            except Exception:
                pass
            if display_enabled and cv2 is not None:
                try:
                    cv2.destroyAllWindows()
                except Exception:
                    pass
        duration = max(time.monotonic() - started, 1e-6)
        stats = AssistStats(
            duration_s=round(duration, 2),
            frames_captured=worker.captured,
            frames_processed=frames_processed,
            detector_runs=detector_runs,
            depth_runs=depth_runs,
            mean_detector_ms=round(statistics.fmean(detector_times), 2) if detector_times else 0.0,
            p95_detector_ms=round(
                float(np.percentile(detector_times, 95)), 2
            ) if detector_times else 0.0,
            mean_depth_ms=round(statistics.fmean(depth_times), 2) if depth_times else 0.0,
            capture_fps=round(worker.captured / duration, 2),
            processed_fps=round(frames_processed / duration, 2),
            announcements=self._stats_announcements,
            critical_alerts=self._stats_critical,
            detector=getattr(self.detector, "name", "detector"),
            depth=getattr(self.depth, "name", "none"),
            degraded=self._degraded,
            errors=errors + self._detector_error_total + self._depth_errors + self._eye_errors,
            notes=list(self._notes),
        )
        if self.logger is not None:
            self.logger.close(stats)
        return stats


# --------------------------------------------------------------------------- #
# Environment doctor
# --------------------------------------------------------------------------- #
def doctor_report(camera_index: int = 0, check_camera: bool = False) -> list[tuple[str, str, str]]:
    """Return (status, item, detail) rows; status is PASS/WARN/FAIL."""
    rows: list[tuple[str, str, str]] = []
    py_ok = sys.version_info >= (3, 10)
    rows.append(
        ("PASS" if py_ok else "FAIL", "python", sys.version.split()[0])
    )
    rows.append(
        ("PASS" if importlib.util.find_spec("numpy") else "FAIL", "numpy", "required")
    )
    cv2_ok = importlib.util.find_spec("cv2") is not None
    rows.append(
        ("PASS" if cv2_ok else "WARN", "opencv", "camera/video/display + Haar fallback" if cv2_ok else "pip install opencv-python")
    )
    mp_ok = importlib.util.find_spec("mediapipe") is not None
    rows.append(
        ("PASS" if mp_ok else "WARN", "mediapipe", "optional eye analytics" if mp_ok else "pip install mediapipe")
    )
    yolo_ok = importlib.util.find_spec("ultralytics") is not None
    rows.append(
        ("PASS" if yolo_ok else "WARN", "ultralytics", "optional YOLO detector" if yolo_ok else "pip install ultralytics")
    )
    tts = SpeechOutput.detect_command()
    rows.append(
        ("PASS" if tts else "WARN", "system TTS", " ".join(tts) if tts else "install espeak-ng or use console output")
    )
    face_model = Path(os.environ.get("GAZEOS_FACE_MODEL", "models/face_landmarker.task"))
    rows.append(
        ("PASS" if face_model.exists() else "WARN", "face model", str(face_model) if face_model.exists() else "needed only for --eye-analytics")
    )
    weights = Path(os.environ.get("GAZEASSIST_YOLO_MODEL", "yolo11n.pt"))
    rows.append(
        ("PASS" if weights.exists() else "WARN", "YOLO weights", str(weights) if weights.exists() else "ultralytics may download yolo11n.pt")
    )
    cuda = "unavailable"
    if importlib.util.find_spec("torch") is not None:
        try:
            import torch  # type: ignore[import-not-found]

            cuda = "available" if torch.cuda.is_available() else "unavailable (CPU fallback ok)"
        except Exception:
            pass
    rows.append(("PASS", "compute", f"CUDA {cuda}; CPU analytics always available"))
    if check_camera:
        if not cv2_ok:
            rows.append(("FAIL", f"camera {camera_index}", "OpenCV not installed"))
        else:
            cv2 = import_cv2()
            capture = cv2.VideoCapture(camera_index)
            ok = capture.isOpened()
            ok_read = False
            if ok:
                ok_read, _ = capture.read()
            capture.release()
            rows.append(
                (
                    "PASS" if ok and ok_read else "FAIL",
                    f"camera {camera_index}",
                    "opened and returned a frame" if ok and ok_read else
                    "could not open; check OS camera permission, other apps, index",
                )
            )
    else:
        rows.append(("INFO", "camera", "not probed (use --camera-check; may trigger permission prompt)"))
    return rows


def _print_doctor(rows: Sequence[tuple[str, str, str]]) -> int:
    print(SAFETY_BANNER.strip())
    print()
    for status, item, detail in rows:
        print(f"[{status:<4}] {item:<12} {detail}")
    failed = any(status == "FAIL" for status, _, _ in rows)
    print("\nPASS core requirements." if not failed else "\nFix FAIL items before a live run.")
    return 0 if not failed else 1


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
EYE_PROFILES = ("alert", "fatigued", "reading", "driving", "scanning")


def _make_output(voice: bool, haptic_port: str | None) -> AssistOutput:
    outputs: list[AssistOutput] = [ConsoleOutput()]
    if voice:
        if SpeechOutput.available():
            outputs.append(SpeechOutput(echo=False))
        else:
            LOG.warning("No system TTS available; continuing with console output only")
    if haptic_port:
        outputs.append(SerialHapticOutput(haptic_port))
    return MultiOutput(outputs) if len(outputs) > 1 else outputs[0]


def _build_config(args: argparse.Namespace, voice: bool) -> AssistConfig:
    return AssistConfig(
        detector=args.detector,
        weights=args.weights,
        device=args.device,
        conf_threshold=args.conf,
        imgsz=args.imgsz,
        depth=args.depth,
        depth_model=args.depth_model,
        depth_invert=getattr(args, "depth_invert", False),
        detector_hz=args.detector_hz,
        depth_hz=args.depth_hz,
        display=args.display,
        eye_analytics=getattr(args, "eye_analytics", False),
        eye_profile=getattr(args, "eye_profile", "screen_fatigue"),
        eye_model=getattr(args, "eye_model", None),
        alert_policy=AlertPolicy(clear_enabled=not getattr(args, "no_clear", False)),
    )


def _print_scene_callback(quiet: bool) -> Callable[[SceneReport], None]:
    state = {"last": 0.0}

    def callback(report: SceneReport) -> None:
        if quiet or report.timestamp - state["last"] < 2.0:
            return
        state["last"] = report.timestamp
        top = report.top_hazard
        top_text = (
            f"{top.label} {top.direction} "
            + (f"{top.distance_m:.1f}m " if top.distance_m is not None else "")
            + f"(p{top.priority})"
            if top else "none"
        )
        print(
            f"[scene {report.timestamp:5.1f}s] top={top_text} | "
            f"path={report.path_hint} | walkable={report.walkable_score:.2f} | "
            f"detections={len(report.detections)} | {report.inference_ms:.0f} ms",
            flush=True,
        )

    return callback


def _run_demo(args: argparse.Namespace) -> int:
    print(SAFETY_BANNER)
    print(
        "DEMO MODE: simulated scene, no camera, no model weights. "
        f"Running {args.seconds:.0f}s of scripted events.\n"
        "Scenario: chair ahead -> car approaching -> person crossing -> stairs -> door."
    )
    source = MockSceneSource(fps=float(args.fps), width=640, height=480)
    detector = MockDetector(start_offset=0.0)
    depth = GeometricDepthEstimator()
    config = AssistConfig(
        detector="mock",
        detector_hz=args.detector_hz,
        depth_hz=0.0,
        display=args.display,
        alert_policy=AlertPolicy(clear_enabled=not args.no_clear),
    )
    output = _make_output(args.voice, None)
    pipeline = AssistPipeline(source, config, output, detector=detector, depth=depth)
    try:
        stats = pipeline.run(
            max_seconds=args.seconds,
            on_report=_print_scene_callback(args.quiet),
        )
    except KeyboardInterrupt:
        print("\nDemo interrupted.")
        return 130
    print()
    print(stats.format_text())
    print("\nDemo complete. This was a simulation, not a real-world evaluation.")
    return 0


def _run_live(args: argparse.Namespace) -> int:
    print(SAFETY_BANNER)
    if not args.acknowledge_risks:
        print(
            "Live mode requires --acknowledge-risks. This prototype is not a "
            "certified mobility aid and must not replace a white cane, guide dog "
            "or human assistance. Use `python gaze_assist.py demo` for a "
            "hardware-free demonstration.",
            file=sys.stderr,
        )
        return 2
    try:
        config = _build_config(args, args.voice)
    except ValueError as exc:
        print(f"Invalid configuration: {exc}", file=sys.stderr)
        return 2
    if args.video:
        source: FrameSource = VideoFileSource(args.video)
    else:
        source = CameraSource(
            args.camera, args.width, args.height, args.fps
        )
    eye_monitor: EyeAnalyticsMonitor | None = None
    if args.eye_analytics:
        try:
            eye_monitor = EyeAnalyticsMonitor(
                args.eye_profile, args.eye_model, args.eye_every
            )
        except Exception as exc:
            LOG.warning("Eye analytics unavailable: %s", exc)
    logger = AssistLogger(args.jsonl) if args.jsonl else None
    output = _make_output(args.voice, args.haptic_port)
    pipeline = AssistPipeline(
        source, config, output, eye_monitor=eye_monitor, logger=logger
    )
    try:
        stats = pipeline.run(
            max_seconds=args.seconds,
            on_report=_print_scene_callback(args.quiet),
        )
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        return 130
    except (GazeOSError, FileNotFoundError, RuntimeError) as exc:
        print(f"Run failed: {exc}", file=sys.stderr)
        return 2
    print()
    print(stats.format_text())
    print("\nReminder: experimental prototype; do not rely on it for mobility.")
    return 0


def _run_eye_demo(args: argparse.Namespace) -> int:
    """Demonstrate the *optional* eye-analytics path using the digital twin."""
    from eye_twin import EyeTwinObservationSource, preset

    from gaze_core import GazePipeline, make_config

    print(SAFETY_BANNER)
    print(
        "EYE ANALYTICS DEMO (research/operator use only).\n"
        "This is synthetic eye data from eye_twin.py, not a navigation aid. "
        "A blind user is never required to steer with gaze."
    )
    params = preset(args.profile)
    params.seed = args.seed
    source = EyeTwinObservationSource(
        profile=args.profile, seed=args.seed, duration_s=args.seconds, fps=args.fps,
        params=params,
    )
    pipeline = GazePipeline(make_config("screen_fatigue"))
    pipeline.detector_name = "eye_twin-simulated"
    seen: set[str] = set()
    for face in source:
        report = pipeline.process(face)
        for anomaly in report.anomalies:
            if anomaly.code not in seen and anomaly.code != "face_lost":
                seen.add(anomaly.code)
                print(f"[analytics {report.timestamp:6.2f}s] {anomaly.code}: {anomaly.message}")
    pipeline.finish(args.seconds)
    summary = pipeline.summary(args.seconds, f"eye_twin:{args.profile}", time.time() - args.seconds)
    print()
    print(summary.format_text())
    print("\nSynthetic data; this does not validate real-world accuracy.")
    return 0


# --------------------------------------------------------------------------- #
# Self test
# --------------------------------------------------------------------------- #
def _run_selftest() -> int:
    import tempfile

    checks: list[tuple[str, bool, str]] = []

    def check(name: str, condition: bool, detail: str = "") -> None:
        checks.append((name, bool(condition), detail))

    # Geometry and distance
    box = BBox(0.4, 0.5, 0.55, 0.95)
    check("bbox area/iou", box.area > 0 and box.iou(box) == 1.0)
    det = Detection("person", 0.9, box)
    geo = GeometricDepthEstimator()
    d = geo.distance_for(det, (480, 640), None)
    # person height 1.7 m, box 0.45*480=216 px, focal ~460 px -> ~3.6 m
    check("geometric distance plausible", d is not None and 2.0 < d < 6.0, f"{d:.2f} m" if d else "None")

    # Risk ranking / critical
    engine = RiskEngine()
    car = Detection("car", 0.9, BBox(0.4, 0.5, 0.6, 0.9), distance_m=0.5)
    hazards = engine.evaluate([car], None, (480, 640), 1.0)
    check("critical hazard", bool(hazards) and hazards[0].priority == 1, str([h.to_dict() for h in hazards]))
    far = Detection("person", 0.9, BBox(0.44, 0.4, 0.5, 0.7), distance_m=8.0)
    check("far object below alert", engine.evaluate([far], None, (480, 640), 1.1) == [])
    stairs = Detection("stairs", 0.9, BBox(0.3, 0.7, 0.7, 1.0), distance_m=1.2)
    sh = engine.evaluate([stairs], None, (480, 640), 1.2)
    check(
        "stairs warning",
        bool(sh) and sh[0].priority == 1 and "stairs" in format_hazard(sh[0]).lower(),
        format_hazard(sh[0]) if sh else "none",
    )

    # TTC with approaching car
    ttc_engine = RiskEngine()
    ttc = None
    for i in range(7):
        t = i * 0.2
        approaching = Detection("car", 0.9, BBox(0.4, 0.4, 0.6, 0.95), distance_m=2.6 - 0.8 * t)
        hs = ttc_engine.evaluate([approaching], None, (480, 640), t)
        if hs:
            ttc = hs[0].ttc_s or ttc
    check("time-to-collision estimated", ttc is not None and ttc < 4.0, f"ttc={ttc}")

    # Path analysis
    score, hint, _ = engine.path_analysis([stairs], sh, None)
    check("path blocked score", score < 0.7 and "blocked" in hint, f"{score:.2f} {hint}")

    # Alert throttling and wording
    class Recorder(NullOutput):
        def __init__(self) -> None:
            self.messages: list[tuple[str, int]] = []

        def announce(self, text: str, priority: int = 3) -> None:
            self.messages.append((text, priority))

    recorder = Recorder()
    manager = AlertManager(recorder)
    report = SceneReport(
        1.0, (hazards[0],), (car,), 0.1, "blocked", "blocked", 5.0, "test", "none"
    )
    manager.update(report)
    manager.update(report)
    check(
        "alert throttle + wording",
        len(recorder.messages) == 1 and "Stop" in recorder.messages[0][0],
        str(recorder.messages),
    )

    # Scripted scenario labels cover vehicle/stairs/people.
    labels = set()
    for t in range(30):
        labels.update(d.label for d in scripted_detections(t))
    check(
        "scripted scenario coverage",
        {"person", "car", "stairs", "door"} <= labels,
        ",".join(sorted(labels)),
    )

    # Full pipeline with mock scene (no camera/GPU).
    config = AssistConfig(
        detector="mock", detector_hz=10.0, depth_hz=1.0, display=False,
        conf_threshold=0.3,
    )
    source = MockSceneSource(fps=30)
    detector = MockDetector(start_offset=7.0)
    pipeline = AssistPipeline(
        source, config, NullOutput(), detector=detector,
        depth=GeometricDepthEstimator(),
    )
    stats = pipeline.run(max_seconds=1.2)
    check(
        "mock pipeline runs",
        stats.frames_processed >= 3 and stats.detector_runs >= 3,
        f"processed={stats.frames_processed}",
    )
    check(
        "measured fps sane",
        0 < stats.processed_fps <= 20 and stats.capture_fps > 0,
        f"cap={stats.capture_fps:.1f} proc={stats.processed_fps:.1f}",
    )
    stats_json = json.dumps(asdict(stats))
    check("stats JSON", len(stats_json) > 100)

    # Report serialization
    sample_report = SceneReport(
        1.0, (hazards[0],), (car,), 0.2, "blocked", "obstacle", 4.0, "test", "none"
    )
    check("scene report JSON", json.loads(json.dumps(sample_report.to_dict()))["hazards"])

    # Logger writes header + report + summary.
    with tempfile.TemporaryDirectory() as tmp:
        logger = AssistLogger(Path(tmp) / "scene.jsonl")
        logger.log(sample_report)
        logger.close(stats)
        lines = (Path(tmp) / "scene.jsonl").read_text(encoding="utf-8").strip().splitlines()
        check("assist JSONL logger", len(lines) == 2, f"lines={len(lines)}")

    # Twin integration path (no MediaPipe needed).
    from eye_twin import EyeTwinObservationSource

    twin = EyeTwinObservationSource(profile="alert", seed=2, duration_s=2.0, fps=30)
    count = sum(1 for _ in twin)
    check("eye_twin integration", count == 60, f"faces={count}")

    passed = sum(1 for _, ok, _ in checks if ok)
    for name, ok, detail in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" ({detail})" if detail else ""))
    print(f"\n{passed}/{len(checks)} checks passed")
    return 0 if passed == len(checks) else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GazeAssist: experimental scene understanding for assistive demos. "
                    "NOT a certified mobility aid.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", help="Check dependencies, models and camera.")
    doctor.add_argument("--camera-check", action="store_true")
    doctor.add_argument("--camera", type=int, default=0)

    demo = sub.add_parser("demo", help="Hardware-free scripted demonstration.")
    demo.add_argument("--seconds", type=float, default=30.0)
    demo.add_argument("--fps", type=float, default=30.0)
    demo.add_argument("--detector-hz", type=float, default=6.0)
    demo.add_argument("--voice", action="store_true", help="Enable system TTS.")
    demo.add_argument("--display", action="store_true")
    demo.add_argument("--no-clear", action="store_true")
    demo.add_argument("--quiet", action="store_true")

    run = sub.add_parser("run", help="Live camera or video run (experimental).")
    run.add_argument("--camera", type=int, default=0)
    run.add_argument("--video", default=None)
    run.add_argument("--detector", default="auto", choices=("auto", "yolo", "haar", "motion"))
    run.add_argument("--weights", default=None, help="YOLO weights path or filename.")
    run.add_argument("--device", default="cpu")
    run.add_argument("--conf", type=float, default=0.35)
    run.add_argument("--imgsz", type=int, default=416)
    run.add_argument("--depth", default="geometric", choices=("geometric", "onnx", "none"))
    run.add_argument("--depth-model", default=None)
    run.add_argument("--depth-invert", action="store_true",
                     help="Invert ONNX depth output (use if farther=larger in your model).")
    run.add_argument("--detector-hz", type=float, default=6.0)
    run.add_argument("--depth-hz", type=float, default=2.0)
    run.add_argument("--width", type=int, default=640)
    run.add_argument("--height", type=int, default=480)
    run.add_argument("--fps", type=float, default=30.0)
    run.add_argument("--seconds", type=float, default=None)
    run.add_argument("--voice", action="store_true")
    run.add_argument("--haptic-port", default=None, help="Serial port for haptics.")
    run.add_argument("--display", action="store_true")
    run.add_argument("--eye-analytics", action="store_true",
                     help="Optional research-only eye analytics; never used for navigation.")
    run.add_argument("--eye-profile", default="screen_fatigue",
                     choices=("generic", "driver_monitoring", "learning_analytics", "screen_fatigue", "research"))
    run.add_argument("--eye-model", default=None)
    run.add_argument("--eye-every", type=int, default=6)
    run.add_argument("--jsonl", default=None)
    run.add_argument("--no-clear", action="store_true")
    run.add_argument("--quiet", action="store_true")
    run.add_argument("--acknowledge-risks", action="store_true")

    eye_demo = sub.add_parser("eye-demo", help="Optional eye analytics on synthetic twin data.")
    eye_demo.add_argument("--profile", default="fatigued", choices=EYE_PROFILES)
    eye_demo.add_argument("--seconds", type=float, default=20.0)
    eye_demo.add_argument("--fps", type=float, default=30.0)
    eye_demo.add_argument("--seed", type=int, default=3)

    sub.add_parser("selftest", help="Dependency-free self tests.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    if args.command == "doctor":
        return _print_doctor(doctor_report(args.camera, args.camera_check))
    if args.command == "selftest":
        return _run_selftest()
    if args.command == "eye-demo":
        try:
            return _run_eye_demo(args)
        except (GazeOSError, ValueError) as exc:
            print(f"Eye demo failed: {exc}", file=sys.stderr)
            return 2
    if args.command == "demo":
        try:
            return _run_demo(args)
        except (GazeOSError, ValueError) as exc:
            print(f"Demo failed: {exc}", file=sys.stderr)
            return 2
    if args.command == "run":
        return _run_live(args)
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
