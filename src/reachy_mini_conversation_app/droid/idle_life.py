"""Idle life: small self-directed behaviours while nobody is talking to the droid."""

import random
from typing import Literal
from datetime import time, datetime
from dataclasses import dataclass

from reachy_mini_conversation_app.droid.identity import InteractionProtocols


IdleActionKind = Literal["beep", "emotion", "look", "remark"]

_REMARK_GAP_S: dict[str, float | None] = {"never": None, "rarely": 3600.0, "sometimes": 1200.0, "often": 600.0}


@dataclass(frozen=True)
class IdleAction:
    """One idle behaviour: a beep pattern, an emotion intent, a look direction, or a spoken remark."""

    kind: IdleActionKind
    value: str = ""


def parse_quiet_hours(spec: str) -> tuple[time, time] | None:
    """Parse ``"23:00-07:00"`` into start and end times; None when empty or malformed."""
    try:
        start_text, end_text = (part.strip() for part in spec.split("-", 1))
        return time.fromisoformat(start_text), time.fromisoformat(end_text)
    except ValueError:
        return None


def in_quiet_hours(spec: str, now: datetime) -> bool:
    """Return whether ``now`` falls inside the owner's quiet hours (ranges may wrap midnight)."""
    window = parse_quiet_hours(spec)
    if window is None:
        return False
    start, end = window
    current = now.time()
    if start <= end:
        return start <= current < end
    return current >= start or current < end


class IdleLife:
    """Pick an idle behaviour now and then, never while the conversation is active."""

    MIN_IDLE_S = 60.0
    ACTION_INTERVAL_S = (90.0, 240.0)

    def __init__(self, rng: random.Random | None = None) -> None:
        """Start with no pending action; ``rng`` makes choices reproducible in tests."""
        self._rng = rng or random.Random()
        self._next_action_at = 0.0
        self._last_remark_at = float("-inf")

    def next_action(
        self,
        idle_s: float,
        now_s: float,
        protocols: InteractionProtocols,
        wall_clock: datetime,
    ) -> IdleAction | None:
        """Return the behaviour to perform now, if any; ``now_s`` is a monotonic timestamp."""
        if idle_s < self.MIN_IDLE_S or now_s < self._next_action_at:
            return None
        self._next_action_at = now_s + self._rng.uniform(*self.ACTION_INTERVAL_S)

        if in_quiet_hours(protocols.quiet_hours, wall_clock):
            return IdleAction("look", self._rng.choice(("left", "right", "down")))

        remark_gap = _REMARK_GAP_S.get(protocols.unprompted_remarks, _REMARK_GAP_S["sometimes"])
        if remark_gap is not None and now_s - self._last_remark_at >= remark_gap and self._rng.random() < 0.35:
            self._last_remark_at = now_s
            return IdleAction("remark")

        return self._rng.choice(
            (
                IdleAction("beep", "thinking"),
                IdleAction("beep", "curious"),
                IdleAction("emotion", "curious"),
                IdleAction("emotion", "bored"),
                IdleAction("look", self._rng.choice(("left", "right", "up"))),
                IdleAction("look", self._rng.choice(("left", "right"))),
            )
        )


def idle_remark_prompt(idle_minutes: float) -> str:
    """Return the sensor event that asks the model for one unprompted remark."""
    return (
        f"[SYSTEM EVENT] No conversation for {idle_minutes:.0f} minutes. Make one short, unprompted, in-character "
        "remark: a self-maintenance report, an observation, or a thought about your owner's interests. "
        "Do not ask how you can help."
    )
