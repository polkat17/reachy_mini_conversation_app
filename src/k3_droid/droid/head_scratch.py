"""Head scratch as a wake gesture, felt through the IMU in the robot's head.

A scratch is a second or so of small, continuous jiggling: the gyroscope picks up the tiny rotations and the
accelerometer the vibration on top of gravity. A single bump, or a knock on the table, is one short spike, so
the detector only fires when enough samples in a sliding window are active. Thresholds are first guesses to
tune on the robot; the peak values are logged at debug level for that.
"""

import math
import logging
from collections import deque


logger = logging.getLogger(__name__)

Vector = tuple[float, float, float]


class HeadScratchDetector:
    """Detect sustained jiggling of the head from IMU samples taken at a steady rate."""

    GYRO_ACTIVE_RAD_S = 0.08
    ACCEL_ACTIVE_M_S2 = 0.4
    WINDOW_SAMPLES = 15  # 1.5 s at the 10 Hz dormant poll
    ACTIVE_SAMPLES_TO_FIRE = 8  # about 0.8 s of scratching inside the window
    ACCEL_BASELINE_ALPHA = 0.05  # slow average that tracks gravity

    def __init__(self) -> None:
        """Start with an empty window and no gravity estimate."""
        self._window: deque[bool] = deque(maxlen=self.WINDOW_SAMPLES)
        self._gravity: Vector | None = None
        self._armed = True

    def reset(self) -> None:
        """Forget the window and gravity estimate, e.g. after the robot moved on purpose."""
        self._window.clear()
        self._gravity = None
        self._armed = True

    def update(self, gyroscope: Vector, accelerometer: Vector) -> bool:
        """Feed one IMU sample; returns True once per scratch."""
        if self._gravity is None:
            self._gravity = accelerometer
            return False
        rotation = math.sqrt(sum(g * g for g in gyroscope))
        vibration = math.sqrt(sum((a - g) ** 2 for a, g in zip(accelerometer, self._gravity)))
        active = rotation >= self.GYRO_ACTIVE_RAD_S or vibration >= self.ACCEL_ACTIVE_M_S2
        if active:
            logger.debug("Head jiggle: gyro %.3f rad/s, vibration %.3f m/s2", rotation, vibration)
        else:
            # Track gravity from still samples only, so a knock does not drag the baseline along.
            alpha = self.ACCEL_BASELINE_ALPHA
            self._gravity = (
                self._gravity[0] + alpha * (accelerometer[0] - self._gravity[0]),
                self._gravity[1] + alpha * (accelerometer[1] - self._gravity[1]),
                self._gravity[2] + alpha * (accelerometer[2] - self._gravity[2]),
            )
        self._window.append(active)

        if not self._armed:
            # Re-arm once the head has been still for a whole window.
            self._armed = len(self._window) == self.WINDOW_SAMPLES and not any(self._window)
            return False
        if sum(self._window) >= self.ACTIVE_SAMPLES_TO_FIRE:
            self._armed = False
            return True
        return False
