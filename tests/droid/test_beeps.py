from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from reachy_mini_conversation_app.droid.beeps import BEEP_PATTERNS, BEEP_SAMPLE_RATE, beep_file, render_beep


@pytest.mark.parametrize("pattern", sorted(BEEP_PATTERNS))
def test_every_pattern_renders_audible_pcm(pattern: str) -> None:
    """Patterns render to non-silent, non-clipping 16-bit audio."""
    pcm = render_beep(pattern)

    assert pcm.dtype == np.int16
    assert 0.05 * BEEP_SAMPLE_RATE < pcm.size < 2 * BEEP_SAMPLE_RATE
    assert 0 < int(np.abs(pcm).max()) < 32767


def test_beep_file_is_rendered_once(tmp_path: Path) -> None:
    """The WAV cache is reused on later calls."""
    first = beep_file("happy", tmp_path)
    mtime = first.stat().st_mtime_ns

    second = beep_file("happy", tmp_path)

    rate, data = wavfile.read(second)
    assert second == first and second.stat().st_mtime_ns == mtime
    assert rate == BEEP_SAMPLE_RATE and data.size > 0


def test_unknown_pattern_is_rejected(tmp_path: Path) -> None:
    """Unknown names raise instead of writing a file."""
    with pytest.raises(KeyError):
        beep_file("kazoo", tmp_path)
