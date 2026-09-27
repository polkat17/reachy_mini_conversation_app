"""Microphone helpers shared by the droid's listeners (wake trigger, wake phrase, sound classifier)."""

import numpy as np
from numpy.typing import NDArray
from scipy.signal import resample_poly

from reachy_mini_conversation_app.streaming import AudioArray, audio_to_float32


LISTENER_SAMPLE_RATE = 16000


def to_mono_16k(sample_rate: int, frame: AudioArray) -> NDArray[np.float32]:
    """Return a mono float32 copy of ``frame`` resampled to 16 kHz."""
    samples = audio_to_float32(frame)
    if samples.ndim == 2:
        channel_axis = 1 if samples.shape[1] <= samples.shape[0] else 0
        samples = samples.mean(axis=channel_axis)
    if sample_rate != LISTENER_SAMPLE_RATE and samples.size and sample_rate > 0:
        divisor = np.gcd(sample_rate, LISTENER_SAMPLE_RATE)
        samples = resample_poly(samples, LISTENER_SAMPLE_RATE // divisor, sample_rate // divisor)
    return np.asarray(samples, dtype=np.float32)


class LoudnessTrigger:
    """Fire once sound stays above ``threshold_rms`` for ``min_duration_s`` (the wake trigger before a wake phrase exists)."""

    def __init__(self, threshold_rms: float = 0.04, min_duration_s: float = 0.6) -> None:
        """Configure the trigger level and how long the sound must last."""
        self.threshold_rms = threshold_rms
        self.min_duration_s = min_duration_s
        self._loud_s = 0.0

    def reset(self) -> None:
        """Forget any partially accumulated loud sound."""
        self._loud_s = 0.0

    def feed(self, samples: NDArray[np.float32]) -> bool:
        """Consume 16 kHz mono samples and return True when the trigger fires."""
        if samples.size == 0:
            return False
        rms = float(np.sqrt(np.mean(np.square(samples))))
        if rms >= self.threshold_rms:
            self._loud_s += samples.size / LISTENER_SAMPLE_RATE
        else:
            self._loud_s = 0.0
        if self._loud_s >= self.min_duration_s:
            self._loud_s = 0.0
            return True
        return False
