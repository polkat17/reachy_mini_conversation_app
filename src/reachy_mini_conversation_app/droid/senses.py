"""Extra senses: a sound classifier (music, cats) and a cat detector, both small ONNX models run on the CPU."""

import csv
import time
import logging
import threading
from typing import Literal
from dataclasses import dataclass
from collections.abc import Callable

import numpy as np
import onnxruntime as ort
from numpy.typing import NDArray
from huggingface_hub import hf_hub_download


logger = logging.getLogger(__name__)

# YAMNet (Apache-2.0, AudioSet's 521 classes), converted to ONNX.
_YAMNET_REPO = "zeropointnine/yamnet-onnx"
_YAMNET_REVISION = "ac2ca3bd45d12ec1f19f1144205ea529b4e9dedf"
_YAMNET_WINDOW = 15600  # 0.975 s at 16 kHz, YAMNet's patch length
# SSD-MobileNet v1 (Apache-2.0, COCO classes) from the ONNX model zoo.
_SSD_REPO = "onnxmodelzoo/ssd_mobilenet_v1_12"
_SSD_REVISION = "019281f3fcb151a90e491f3b2f0273f9f31bd6be"
_COCO_CAT = 17

MUSIC_LABELS = frozenset({"Music"})
CAT_LABELS = frozenset({"Cat", "Meow", "Purr", "Caterwaul", "Hiss"})

SoundEventKind = Literal["music_started", "music_stopped", "meow"]


def _session(path: str) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    return ort.InferenceSession(path, options, providers=["CPUExecutionProvider"])


class SoundClassifier:
    """Score one 16 kHz audio window for music and cat sounds."""

    def __init__(self) -> None:
        """Download (once) and load YAMNet with its class names."""
        self._session = _session(hf_hub_download(_YAMNET_REPO, "yamnet.onnx", revision=_YAMNET_REVISION))
        class_map = hf_hub_download(_YAMNET_REPO, "yamnet_class_map.csv", revision=_YAMNET_REVISION)
        with open(class_map, encoding="utf-8") as file:
            names = [row["display_name"] for row in csv.DictReader(file)]
        self._music = [i for i, name in enumerate(names) if name in MUSIC_LABELS]
        self._cat = [i for i, name in enumerate(names) if name in CAT_LABELS]

    def scores(self, window: NDArray[np.float32]) -> tuple[float, float]:
        """Return (music, cat) scores in [0, 1] for one window."""
        (frame_scores, *_rest) = self._session.run(None, {"waveform": window.astype(np.float32)})
        mean = np.asarray(frame_scores).mean(axis=0)
        return float(mean[self._music].max()), float(mean[self._cat].max())


@dataclass(frozen=True)
class SoundEvent:
    """Music starting or stopping, or a meow."""

    kind: SoundEventKind
    score: float


class SoundWatcher:
    """Turn per-window scores into events, with hysteresis for music and a cooldown for meows."""

    MUSIC_ON, MUSIC_OFF = 0.3, 0.12
    MUSIC_ON_WINDOWS, MUSIC_OFF_WINDOWS = 3, 5
    MEOW_THRESHOLD, MEOW_COOLDOWN_S = 0.3, 120.0

    def __init__(self, score: Callable[[NDArray[np.float32]], tuple[float, float]]) -> None:
        """Wrap a scoring function (normally ``SoundClassifier.scores``)."""
        self._score = score
        self._buffer = np.zeros(0, dtype=np.float32)
        self.music_playing = False
        self._music_streak = 0
        self._last_meow = float("-inf")

    def feed(self, samples: NDArray[np.float32], now: float | None = None) -> list[SoundEvent]:
        """Consume 16 kHz mono audio and return any events from completed windows."""
        self._buffer = np.concatenate([self._buffer, samples])
        events: list[SoundEvent] = []
        while self._buffer.size >= _YAMNET_WINDOW:
            window, self._buffer = self._buffer[:_YAMNET_WINDOW], self._buffer[_YAMNET_WINDOW:]
            events += self._judge(*self._score(window), time.monotonic() if now is None else now)
        return events

    def _judge(self, music: float, cat: float, now: float) -> list[SoundEvent]:
        events: list[SoundEvent] = []
        if self.music_playing:
            self._music_streak = self._music_streak + 1 if music < self.MUSIC_OFF else 0
            if self._music_streak >= self.MUSIC_OFF_WINDOWS:
                self.music_playing, self._music_streak = False, 0
                events.append(SoundEvent("music_stopped", music))
        else:
            self._music_streak = self._music_streak + 1 if music >= self.MUSIC_ON else 0
            if self._music_streak >= self.MUSIC_ON_WINDOWS:
                self.music_playing, self._music_streak = True, 0
                events.append(SoundEvent("music_started", music))
        if cat >= self.MEOW_THRESHOLD and now - self._last_meow >= self.MEOW_COOLDOWN_S:
            self._last_meow = now
            events.append(SoundEvent("meow", cat))
        return events


@dataclass(frozen=True)
class CatSighting:
    """A detected cat: its centre in pixels and the detector's confidence."""

    u: int
    v: int
    score: float


class CatDetector:
    """Find cats in BGR camera frames."""

    MIN_SCORE = 0.5

    def __init__(self) -> None:
        """Download (once) and load SSD-MobileNet v1."""
        self._session = _session(hf_hub_download(_SSD_REPO, "ssd_mobilenet_v1_12.onnx", revision=_SSD_REVISION))
        self._lock = threading.Lock()

    def detect(self, frame_bgr: NDArray[np.uint8]) -> list[CatSighting]:
        """Return cats found in the frame, most confident first."""
        height, width = frame_bgr.shape[:2]
        with self._lock:
            outputs = self._session.run(None, {"inputs": np.ascontiguousarray(frame_bgr[..., ::-1])[np.newaxis]})
        named = dict(zip([output.name for output in self._session.get_outputs()], outputs))
        count = int(np.asarray(named["num_detections"])[0])
        boxes = np.asarray(named["detection_boxes"])[0][:count]
        classes = np.asarray(named["detection_classes"])[0][:count]
        scores = np.asarray(named["detection_scores"])[0][:count]
        cats = [
            CatSighting(int((box[1] + box[3]) / 2 * width), int((box[0] + box[2]) / 2 * height), float(score))
            for box, label, score in zip(boxes, classes, scores)
            if int(label) == _COCO_CAT and score >= self.MIN_SCORE
        ]
        return sorted(cats, key=lambda cat: cat.score, reverse=True)
