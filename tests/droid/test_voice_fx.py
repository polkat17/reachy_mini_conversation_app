import numpy as np

from reachy_mini_conversation_app.droid.voice_fx import DroidVoiceFilter


def _speechlike(seconds: float = 1.0, rate: int = 16000) -> np.ndarray:
    t = np.arange(int(seconds * rate)) / rate
    return (0.4 * np.sin(2 * np.pi * 220 * t) + 0.2 * np.sin(2 * np.pi * 1200 * t)).astype(np.float32)


def test_streaming_matches_one_shot() -> None:
    """Chunked processing carries filter state across chunks, so it equals processing the whole signal."""
    signal = _speechlike()

    whole = DroidVoiceFilter(16000).process(signal)
    streamed_filter = DroidVoiceFilter(16000)
    streamed = np.concatenate([streamed_filter.process(signal[i : i + 800]) for i in range(0, signal.size, 800)])

    np.testing.assert_allclose(streamed, whole, atol=1e-5)


def test_output_is_bounded_and_same_length() -> None:
    """The filter never lengthens chunks (no added latency) and never clips past full scale."""
    loud = np.clip(_speechlike() * 4, -1, 1)

    out = DroidVoiceFilter(16000).process(loud)

    assert out.shape == loud.shape and out.dtype == np.float32
    assert float(np.max(np.abs(out))) < 1.0


def test_band_pass_removes_rumble() -> None:
    """Low-frequency rumble is strongly attenuated while the voice band passes."""
    t = np.arange(16000) / 16000
    rumble = (0.5 * np.sin(2 * np.pi * 40 * t)).astype(np.float32)
    voice = (0.5 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)

    rumble_out = DroidVoiceFilter(16000, ring_mix=0.0, resonance_mix=0.0).process(rumble)[4000:]
    voice_out = DroidVoiceFilter(16000, ring_mix=0.0, resonance_mix=0.0).process(voice)[4000:]

    assert np.sqrt(np.mean(rumble_out**2)) < 0.1 * np.sqrt(np.mean(voice_out**2))


def test_empty_chunk() -> None:
    """Empty chunks pass through."""
    assert DroidVoiceFilter(16000).process(np.zeros(0, dtype=np.float32)).size == 0
