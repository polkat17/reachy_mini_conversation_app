from k3_droid.droid.antenna import AntennaFlickDetector


def test_flick_fires_once_until_released() -> None:
    """A push past the press threshold fires once; it re-arms only after the antenna settles back."""
    detector = AntennaFlickDetector()
    rest = (-0.17, 0.17)

    readings = [rest, (-0.17, 0.30), (-0.17, 0.50), (-0.17, 0.55), (-0.17, 0.20), (-0.17, 0.50)]

    assert [detector.update(reading) for reading in readings] == [False, False, True, False, False, True]


def test_either_antenna_counts_and_reset_rebases() -> None:
    """Both antennas are buttons, and reset() takes a new resting reference."""
    detector = AntennaFlickDetector()
    detector.update((0.0, 0.0))

    assert detector.update((-0.4, 0.0)) is True

    detector.reset()
    detector.update((-0.4, 0.0))
    assert detector.update((-0.45, 0.0)) is False
