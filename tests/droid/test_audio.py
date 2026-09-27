import numpy as np

from reachy_mini_conversation_app.droid.audio import LISTENER_SAMPLE_RATE, LoudnessTrigger, to_mono_16k


def test_to_mono_16k_downmixes_and_resamples() -> None:
    """Stereo 48 kHz int16 becomes mono 16 kHz float32."""
    frame = np.full((4800, 2), 16384, dtype=np.int16)

    samples = to_mono_16k(48000, frame)

    assert samples.dtype == np.float32
    assert samples.shape == (1600,)
    assert abs(float(np.median(samples)) - 0.5) < 0.01


def test_loudness_trigger_needs_sustained_sound() -> None:
    """Short noises do not fire; sustained speech-level sound does."""
    trigger = LoudnessTrigger(threshold_rms=0.05, min_duration_s=0.5)
    loud = np.full(LISTENER_SAMPLE_RATE // 10, 0.2, dtype=np.float32)
    quiet = np.zeros(LISTENER_SAMPLE_RATE // 10, dtype=np.float32)

    assert not any(trigger.feed(loud) for _ in range(3))
    assert not trigger.feed(quiet)
    fired = [trigger.feed(loud) for _ in range(5)]

    assert fired == [False, False, False, False, True]


def test_loudness_trigger_reset() -> None:
    """reset() drops accumulated loudness."""
    trigger = LoudnessTrigger(threshold_rms=0.05, min_duration_s=0.2)
    loud = np.full(LISTENER_SAMPLE_RATE // 10, 0.2, dtype=np.float32)
    trigger.feed(loud)
    trigger.reset()

    assert not trigger.feed(loud)
