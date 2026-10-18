"""Antennas as a button: a flick on either antenna wakes a dormant droid.

The antennas are compliant (low gain), so pushing one moves it away from its commanded position. The press is
measured against a reference taken once the robot is still, with hysteresis so one flick fires only once.
"""


class AntennaFlickDetector:
    """Detect a push on either antenna relative to a resting reference."""

    PRESS_DELTA_RAD = 0.25
    RELEASE_DELTA_RAD = 0.10

    def __init__(self) -> None:
        """Start without a reference; the first reading becomes it."""
        self._reference: tuple[float, float] | None = None
        self._armed = False

    def reset(self) -> None:
        """Forget the reference, e.g. after the robot moved on purpose."""
        self._reference = None
        self._armed = False

    def update(self, antennas: tuple[float, float]) -> bool:
        """Feed the current antenna angles; returns True once per flick."""
        if self._reference is None:
            self._reference = antennas
            self._armed = True
            return False
        delta = max(abs(now - rest) for now, rest in zip(antennas, self._reference))
        if not self._armed:
            self._armed = delta <= self.RELEASE_DELTA_RAD
            return False
        if delta >= self.PRESS_DELTA_RAD:
            self._armed = False
            return True
        return False
