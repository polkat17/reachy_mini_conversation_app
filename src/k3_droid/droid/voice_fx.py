"""The droid voice filter: band-pass, subtle ring modulation and a short metallic resonance, streamed chunk by chunk.

Every stage is causal and keeps its state between chunks, so it adds no buffering latency.
"""

import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, lfilter, sosfilt, lfilter_zi, sosfilt_zi


class DroidVoiceFilter:
    """Stateful per-stream filter for mono float32 audio."""

    def __init__(
        self,
        sample_rate: int,
        *,
        band_hz: tuple[float, float] = (280.0, 3800.0),
        ring_hz: float = 45.0,
        ring_mix: float = 0.18,
        resonance_ms: float = 6.0,
        resonance_feedback: float = 0.32,
        resonance_mix: float = 0.25,
    ) -> None:
        """Design the filters for ``sample_rate``; defaults keep the voice clear with a light metallic edge."""
        self.sample_rate = sample_rate
        nyquist = sample_rate / 2.0
        low, high = band_hz[0] / nyquist, min(band_hz[1] / nyquist, 0.99)
        self._sos = butter(2, [low, high], btype="bandpass", output="sos")
        self._sos_state = sosfilt_zi(self._sos) * 0.0
        self._ring_step = 2 * np.pi * ring_hz / sample_rate
        self._ring_phase = 0.0
        self._ring_mix = ring_mix
        delay = max(1, int(sample_rate * resonance_ms / 1000.0))
        # Feedback comb y[n] = x[n] + g * y[n - delay]: the "metal tube" resonance.
        self._comb_a = np.zeros(delay + 1)
        self._comb_a[0], self._comb_a[delay] = 1.0, -resonance_feedback
        self._comb_state = lfilter_zi([1.0], self._comb_a) * 0.0
        self._resonance_mix = resonance_mix

    def process(self, chunk: NDArray[np.float32]) -> NDArray[np.float32]:
        """Filter one chunk and return a new array of the same length."""
        if chunk.size == 0:
            return chunk
        signal = chunk.astype(np.float64)
        banded, self._sos_state = sosfilt(self._sos, signal, zi=self._sos_state)

        phases = self._ring_phase + self._ring_step * np.arange(signal.size)
        self._ring_phase = float((phases[-1] + self._ring_step) % (2 * np.pi))
        ringed = (1.0 - self._ring_mix) * banded + self._ring_mix * banded * np.sin(phases)

        resonant, self._comb_state = lfilter([1.0], self._comb_a, ringed, zi=self._comb_state)
        mixed = (1.0 - self._resonance_mix) * ringed + self._resonance_mix * resonant
        # Soft clip keeps the resonance from ever distorting harshly.
        clipped: NDArray[np.float32] = np.tanh(1.2 * mixed).astype(np.float32)
        return clipped
