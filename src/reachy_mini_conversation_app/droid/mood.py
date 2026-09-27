"""Moods over days: a slow-changing state derived from how much the droid has seen its owner."""

from dataclasses import dataclass


@dataclass(frozen=True)
class MoodInputs:
    """What the mood depends on."""

    hours_since_owner: float
    conversations_today: int
    conversations_this_week: int
    recent_owner_moods: list[str]


@dataclass(frozen=True)
class Mood:
    """A mood label, the reason behind it, and a fitting idle emotion."""

    label: str
    reason: str
    emotion: str


_NEGATIVE_OWNER_MOODS = {"sad", "stressed", "tired", "anxious", "angry", "frustrated", "upset", "worried"}


def current_mood(inputs: MoodInputs, owner: str) -> Mood:
    """Return the droid's mood for today."""
    if inputs.hours_since_owner >= 48:
        return Mood("lonely", f"you have not seen {owner} for {inputs.hours_since_owner / 24:.0f} days", "lonely")
    if inputs.recent_owner_moods and sum(m.lower() in _NEGATIVE_OWNER_MOODS for m in inputs.recent_owner_moods) >= 2:
        return Mood("protective", f"{owner} has seemed low lately; be extra gentle and supportive", "worried")
    if inputs.conversations_today >= 5:
        return Mood("energetic", f"you have talked with {owner} a lot today", "happy")
    if inputs.hours_since_owner >= 12:
        return Mood("bored", f"{owner} has been away for {inputs.hours_since_owner:.0f} hours", "bored")
    if inputs.conversations_this_week >= 10:
        return Mood("content", f"a good week with {owner}", "happy")
    return Mood("curious", "a normal day; you are keen to hear what happens next", "curious")
