"""Synthesised droid beeps: short chirp sequences rendered to WAV files for the robot speaker."""

from pathlib import Path

import numpy as np
from scipy.io import wavfile
from numpy.typing import NDArray


BEEP_SAMPLE_RATE = 22050
_BEEP_VERSION = 1

# Each chirp is (start Hz, end Hz, duration s, silence after s).
BEEP_PATTERNS: dict[str, tuple[tuple[float, float, float, float], ...]] = {
    "affirmative": ((900, 1300, 0.08, 0.04), (1300, 1800, 0.10, 0.0)),
    "negative": ((1400, 900, 0.12, 0.05), (700, 450, 0.18, 0.0)),
    "happy": ((1000, 1600, 0.06, 0.03), (1600, 1200, 0.06, 0.03), (1200, 2000, 0.09, 0.0)),
    "sad": ((900, 700, 0.2, 0.06), (650, 400, 0.35, 0.0)),
    "curious": ((800, 800, 0.07, 0.05), (900, 1500, 0.16, 0.0)),
    "alarm": ((2000, 2000, 0.07, 0.04), (2000, 2000, 0.07, 0.04), (2000, 2000, 0.07, 0.0)),
    "thinking": ((1100, 1000, 0.05, 0.08), (1000, 1100, 0.05, 0.08), (1100, 1000, 0.05, 0.0)),
    "boot": ((400, 900, 0.15, 0.05), (900, 900, 0.06, 0.04), (1200, 1800, 0.12, 0.0)),
    "shutdown": ((1600, 900, 0.14, 0.04), (800, 300, 0.4, 0.0)),
    "trill": ((1800, 2600, 0.04, 0.01), (2600, 1800, 0.04, 0.01), (1800, 2800, 0.05, 0.0)),
}


def render_beep(pattern: str, sample_rate: int = BEEP_SAMPLE_RATE) -> NDArray[np.int16]:
    """Render one named pattern to mono 16-bit PCM."""
    chirps = BEEP_PATTERNS[pattern]
    pieces: list[NDArray[np.float64]] = []
    for start_hz, end_hz, duration_s, gap_s in chirps:
        n = int(duration_s * sample_rate)
        t = np.arange(n) / sample_rate
        # Linear frequency sweep; the phase is the integral of the instantaneous frequency.
        phase = 2 * np.pi * (start_hz * t + (end_hz - start_hz) * t**2 / (2 * duration_s))
        tone = np.sin(phase) + 0.3 * np.sin(2 * phase) + 0.1 * np.sign(np.sin(phase))
        envelope = np.minimum(1.0, np.minimum(t, duration_s - t) / 0.008)
        pieces.append(tone * envelope)
        pieces.append(np.zeros(int(gap_s * sample_rate)))
    signal = np.concatenate(pieces)
    signal = 0.5 * signal / max(float(np.max(np.abs(signal))), 1e-9)
    return (signal * 32767).astype(np.int16)


def beep_file(pattern: str, cache_dir: Path) -> Path:
    """Return a WAV file for ``pattern``, rendering it into ``cache_dir`` on first use."""
    if pattern not in BEEP_PATTERNS:
        raise KeyError(f"unknown beep pattern {pattern!r}")
    path = cache_dir / f"beep-{pattern}-v{_BEEP_VERSION}.wav"
    if not path.is_file():
        cache_dir.mkdir(parents=True, exist_ok=True)
        wavfile.write(path, BEEP_SAMPLE_RATE, render_beep(pattern))
    return path
