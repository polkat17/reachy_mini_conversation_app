import threading
from types import SimpleNamespace

import numpy as np

from reachy_mini_conversation_app.droid.senses import SoundEvent, CatDetector, SoundWatcher


WINDOW = np.zeros(15600, dtype=np.float32)


def _watcher(scores: list[tuple[float, float]]) -> SoundWatcher:
    queue = list(scores)
    return SoundWatcher(lambda _window: queue.pop(0))


def test_music_needs_several_windows_to_start_and_stop() -> None:
    """Music starts after three musical windows and stops after five quiet ones."""
    watcher = _watcher([(0.6, 0)] * 3 + [(0.0, 0)] * 5)

    events = [event.kind for i in range(8) for event in watcher.feed(WINDOW, now=float(i))]

    assert events == ["music_started", "music_stopped"]


def test_single_musical_blip_is_ignored() -> None:
    """A lone beep that sounds musical does not start a dance."""
    watcher = _watcher([(0.6, 0), (0.0, 0), (0.6, 0), (0.0, 0)])

    assert [e for i in range(4) for e in watcher.feed(WINDOW, now=float(i))] == []
    assert watcher.music_playing is False


def test_meows_have_a_cooldown() -> None:
    """Repeated meows produce one event per cooldown period."""
    watcher = _watcher([(0, 0.8), (0, 0.8), (0, 0.8)])

    first = watcher.feed(WINDOW, now=0.0)
    second = watcher.feed(WINDOW, now=10.0)
    third = watcher.feed(WINDOW, now=10.0 + SoundWatcher.MEOW_COOLDOWN_S)

    assert first == [SoundEvent("meow", 0.8)] and second == [] and third == [SoundEvent("meow", 0.8)]


def test_audio_is_buffered_into_windows() -> None:
    """Short chunks accumulate until a full window is available."""
    calls: list[int] = []
    watcher = SoundWatcher(lambda window: (calls.append(window.size), (0.0, 0.0))[1])

    for _ in range(19):
        watcher.feed(np.zeros(1600, dtype=np.float32), now=0.0)

    assert calls == [15600]


def test_cat_detector_keeps_confident_cats_only() -> None:
    """Only COCO class 17 above the threshold counts, located at the box centre in pixels."""
    detector = CatDetector.__new__(CatDetector)
    outputs = [
        np.array([[[0.1, 0.2, 0.5, 0.6], [0.0, 0.0, 1.0, 1.0], [0.2, 0.2, 0.3, 0.3]]], dtype=np.float32),
        np.array([[17, 1, 17]], dtype=np.float32),
        np.array([[0.9, 0.95, 0.3]], dtype=np.float32),
        np.array([3], dtype=np.float32),
    ]
    names = ["detection_boxes", "detection_classes", "detection_scores", "num_detections"]
    detector._session = SimpleNamespace(  # type: ignore[assignment]
        run=lambda *_args: outputs, get_outputs=lambda: [SimpleNamespace(name=n) for n in names]
    )
    detector._lock = threading.Lock()

    cats = detector.detect(np.zeros((480, 640, 3), dtype=np.uint8))

    assert [(cat.u, cat.v, round(cat.score, 2)) for cat in cats] == [(256, 144, 0.9)]
