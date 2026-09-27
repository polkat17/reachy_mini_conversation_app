import math

from reachy_mini_conversation_app.droid.identity import Identity, InteractionProtocols
from reachy_mini_conversation_app.droid.presence import STRANGER, Sighting, PresenceEvent, PresenceTracker
from reachy_mini_conversation_app.droid.reactions import presence_prompt, break_nudge_prompt


OWNER = Sighting("Pasha", is_owner=True)


def test_first_sighting_is_an_arrival() -> None:
    """A never-seen person arrives with an infinite absence."""
    events = PresenceTracker().update([OWNER], now=0.0)

    assert events == [PresenceEvent("arrived", "Pasha", True, away_s=math.inf)]


def test_leaving_and_returning() -> None:
    """Absence past the threshold is a departure; the return reports how long they were away."""
    tracker = PresenceTracker()
    tracker.update([OWNER], now=0.0)
    tracker.update([OWNER], now=300.0)

    left = tracker.update([], now=300.0 + PresenceTracker.ABSENT_AFTER_S)
    back = tracker.update([OWNER], now=4000.0)

    assert left == [PresenceEvent("left", "Pasha", True, stayed_s=300.0)]
    assert back == [PresenceEvent("arrived", "Pasha", True, away_s=3700.0)]
    assert tracker.present() == ["Pasha"]


def test_brief_gaps_do_not_count_as_leaving() -> None:
    """Looking away for a moment keeps continuous presence."""
    tracker = PresenceTracker()
    tracker.update([OWNER], now=0.0)
    tracker.update([], now=30.0)
    tracker.update([OWNER], now=60.0)

    assert tracker.owner_present_for(now=60.0) == 60.0


def test_owner_welcome_back_only_after_long_absence() -> None:
    """Short absences stay quiet; long ones get a welcome back with the duration."""
    identity = Identity()

    assert presence_prompt(PresenceEvent("arrived", "Pasha", True, away_s=120.0), identity) is None
    prompt = presence_prompt(PresenceEvent("arrived", "Pasha", True, away_s=3 * 3600.0), identity)

    assert prompt is not None and "3 hours" in prompt


def test_stranger_prompt_protects_owner_memories() -> None:
    """Strangers are greeted, but the droid is reminded not to share the owner's memories."""
    prompt = presence_prompt(PresenceEvent("arrived", STRANGER, False, away_s=math.inf), Identity())

    assert prompt is not None and "never share" in prompt


def test_greet_on_sight_off_silences_owner_greetings() -> None:
    """Owners who opted out of greetings are not greeted or waved off."""
    identity = Identity(protocols=InteractionProtocols(greet_on_sight=False))

    assert presence_prompt(PresenceEvent("arrived", "Pasha", True, away_s=math.inf), identity) is None
    assert presence_prompt(PresenceEvent("left", "Pasha", True, stayed_s=3600.0), identity) is None


def test_break_nudge_mentions_duration() -> None:
    """The break reminder says how long the owner has been at it."""
    assert "2 hours" in break_nudge_prompt(Identity(), 7200.0)
