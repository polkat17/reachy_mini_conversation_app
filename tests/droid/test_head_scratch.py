from k3_droid.droid.head_scratch import HeadScratchDetector


STILL = ((0.0, 0.0, 0.0), (0.0, 0.0, 9.81))
SCRATCH = ((0.15, 0.05, 0.02), (0.1, 0.0, 9.81))


def _feed(detector: HeadScratchDetector, samples: list) -> list[bool]:
    return [detector.update(gyro, accel) for gyro, accel in samples]


def test_a_sustained_scratch_fires_once() -> None:
    """About a second of jiggling fires, and continued scratching does not fire again."""
    detector = HeadScratchDetector()
    fired = _feed(detector, [STILL] * 5 + [SCRATCH] * 30)
    assert fired.count(True) == 1


def test_a_single_bump_does_not_fire() -> None:
    """A knock lasting a few samples stays below the sustained-activity bar."""
    detector = HeadScratchDetector()
    bump = ((0.5, 0.5, 0.0), (3.0, 0.0, 12.0))
    assert not any(_feed(detector, [STILL] * 5 + [bump] * 3 + [STILL] * 20))


def test_vibration_alone_counts_and_rearms_after_stillness() -> None:
    """Accelerometer vibration counts as activity; a still window re-arms the detector."""
    detector = HeadScratchDetector()
    shaking = [((0.0, 0.0, 0.0), (0.0, 0.0, 9.81 + (0.8 if i % 2 else -0.8))) for i in range(30)]
    first = _feed(detector, [STILL] * 5 + shaking)
    again = _feed(detector, [STILL] * HeadScratchDetector.WINDOW_SAMPLES + shaking)
    assert first.count(True) == 1
    assert again.count(True) == 1
