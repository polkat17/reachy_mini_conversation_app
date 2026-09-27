"""Presence: who is in front of the camera, and when people arrive or leave."""

import time
import logging
import threading
from typing import Literal
from dataclasses import dataclass
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from reachy_mini.vision.face_detector import Face, FaceDetector
from reachy_mini_conversation_app.droid.faces import FaceEmbedder, FaceRegistry
from reachy_mini_conversation_app.droid.senses import CatDetector, CatSighting


logger = logging.getLogger(__name__)

STRANGER = "stranger"
CAT = "cat"
_CAT_CHECK_EVERY = 3  # frames; the cat detector is the heavier model
_MAX_FACES_PER_FRAME = 3
_MIN_FACE_WIDTH_PX = 40


@dataclass(frozen=True)
class Sighting:
    """One face seen in one frame: an enrolled name, or ``STRANGER``."""

    name: str
    is_owner: bool = False


@dataclass(frozen=True)
class PresenceEvent:
    """Someone arrived after being away (``away_s`` is inf for a first sighting) or left after ``stayed_s``."""

    kind: Literal["arrived", "left"]
    name: str
    is_owner: bool
    away_s: float = 0.0
    stayed_s: float = 0.0


@dataclass
class _Track:
    is_owner: bool
    first_seen: float
    last_seen: float
    present: bool = True


class PresenceTracker:
    """Turn per-frame sightings into arrivals and departures; pure logic, no camera."""

    ABSENT_AFTER_S = 90.0

    def __init__(self) -> None:
        """Start with nobody seen."""
        self._tracks: dict[str, _Track] = {}

    def update(self, sightings: list[Sighting], now: float) -> list[PresenceEvent]:
        """Record this frame's sightings and return the resulting events."""
        events: list[PresenceEvent] = []
        for sighting in sightings:
            track = self._tracks.get(sighting.name)
            if track is None:
                self._tracks[sighting.name] = _Track(sighting.is_owner, now, now)
                events.append(PresenceEvent("arrived", sighting.name, sighting.is_owner, away_s=float("inf")))
            elif not track.present:
                events.append(PresenceEvent("arrived", sighting.name, track.is_owner, away_s=now - track.last_seen))
                track.present, track.first_seen, track.last_seen = True, now, now
            else:
                track.last_seen = now
        for name, track in self._tracks.items():
            if track.present and now - track.last_seen >= self.ABSENT_AFTER_S:
                track.present = False
                events.append(PresenceEvent("left", name, track.is_owner, stayed_s=track.last_seen - track.first_seen))
        return events

    def present(self) -> list[str]:
        """Return who is currently considered present."""
        return [name for name, track in self._tracks.items() if track.present]

    def owner_present_for(self, now: float) -> float:
        """Return how long the owner has been continuously present, or 0."""
        for track in self._tracks.values():
            if track.is_owner and track.present:
                return now - track.first_seen
        return 0.0


class PresenceSensor:
    """Background camera loop: detect faces, recognise them, and report presence events."""

    def __init__(
        self,
        get_frame: Callable[[], NDArray[np.uint8] | None],
        registry: FaceRegistry,
        on_event: Callable[[PresenceEvent], None],
        interval_s: Callable[[], float],
        watch_cats: bool = True,
    ) -> None:
        """Configure the frame source, known faces, event callback, polling interval and cat watching."""
        self._get_frame = get_frame
        self.registry = registry
        self._on_event = on_event
        self._interval_s = interval_s
        self.tracker = PresenceTracker()
        self._camera_lock = threading.Lock()
        self._stop = threading.Event()
        self._detector: FaceDetector | None = None
        self._embedder: FaceEmbedder | None = None
        self._watch_cats = watch_cats
        self._cat_detector: CatDetector | None = None
        self._frames_seen = 0
        self.last_cat: CatSighting | None = None

    def start(self) -> None:
        """Run the camera loop on a daemon thread."""
        threading.Thread(target=self._run, daemon=True, name="droid-presence").start()

    def stop(self) -> None:
        """Stop the camera loop."""
        self._stop.set()

    def _models(self) -> tuple[FaceDetector, FaceEmbedder]:
        if self._detector is None or self._embedder is None:
            self._detector = FaceDetector()
            self._embedder = FaceEmbedder()
        return self._detector, self._embedder

    def _faces(self, frame: NDArray[np.uint8]) -> list[Face]:
        detector, _ = self._models()
        faces = [face for face in detector.detect(frame) if face.bbox[2] >= _MIN_FACE_WIDTH_PX]
        return sorted(faces, key=lambda face: face.bbox[2] * face.bbox[3], reverse=True)[:_MAX_FACES_PER_FRAME]

    def _run(self) -> None:
        while not self._stop.wait(self._interval_s()):
            try:
                with self._camera_lock:
                    sightings = self.observe()
                events = self.tracker.update(sightings, time.monotonic())
            except Exception as e:
                logger.warning("Presence check failed: %s", e)
                continue
            for event in events:
                logger.info("Presence: %s %s", event.name, event.kind)
                self._on_event(event)

    def observe(self) -> list[Sighting]:
        """Look once and return who is in view."""
        frame = self._get_frame()
        if frame is None:
            return []
        _, embedder = self._models()
        sightings: list[Sighting] = []
        for face in self._faces(frame):
            person, _score = self.registry.identify(embedder.embed(frame, face))
            sightings.append(Sighting(person.name, person.is_owner) if person else Sighting(STRANGER))
        self._frames_seen += 1
        if self._watch_cats and self._frames_seen % _CAT_CHECK_EVERY == 0:
            if self._cat_detector is None:
                self._cat_detector = CatDetector()
            cats = self._cat_detector.detect(frame)
            if cats:
                self.last_cat = cats[0]
                sightings.append(Sighting(CAT))
        elif self._watch_cats and CAT in self.tracker.present():
            # Between cat checks, keep a present cat's track alive.
            sightings.append(Sighting(CAT))
        return sightings

    def capture_embeddings(
        self, samples: int = 8, timeout_s: float = 20.0, *, exclude_known: bool = False
    ) -> list[NDArray[np.float32]]:
        """Collect embeddings of the largest face over several frames (the person should turn slightly)."""
        collected: list[NDArray[np.float32]] = []
        deadline = time.monotonic() + timeout_s
        with self._camera_lock:
            _, embedder = self._models()
            while len(collected) < samples and time.monotonic() < deadline:
                frame = self._get_frame()
                faces = self._faces(frame) if frame is not None else []
                if exclude_known and frame is not None:
                    faces = [face for face in faces if self.registry.identify(embedder.embed(frame, face))[0] is None]
                if frame is not None and faces:
                    collected.append(embedder.embed(frame, faces[0]))
                time.sleep(0.4)
        return collected
