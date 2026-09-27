import random
from datetime import datetime

import pytest

from reachy_mini_conversation_app.droid.identity import InteractionProtocols
from reachy_mini_conversation_app.droid.idle_life import IdleLife, in_quiet_hours, parse_quiet_hours


@pytest.mark.parametrize(
    ("spec", "hour", "expected"),
    [
        ("23:00-07:00", 23, True),
        ("23:00-07:00", 3, True),
        ("23:00-07:00", 12, False),
        ("13:00-15:00", 14, True),
        ("13:00-15:00", 16, False),
        ("", 3, False),
        ("garbage", 3, False),
    ],
)
def test_quiet_hours(spec: str, hour: int, expected: bool) -> None:
    """Quiet hours handle ranges that wrap midnight and ignore malformed specs."""
    assert in_quiet_hours(spec, datetime(2026, 1, 1, hour, 30)) is expected


def test_parse_quiet_hours_rejects_missing_dash() -> None:
    """A spec without a range is not a window."""
    assert parse_quiet_hours("23:00") is None


def test_no_action_before_idle_threshold() -> None:
    """Nothing happens while the conversation is recent."""
    life = IdleLife(random.Random(0))

    assert life.next_action(10.0, 1000.0, InteractionProtocols(), datetime(2026, 1, 1, 12)) is None


def test_actions_are_spaced_out() -> None:
    """After one action, the next waits at least the minimum interval."""
    life = IdleLife(random.Random(0))
    protocols = InteractionProtocols()
    noon = datetime(2026, 1, 1, 12)

    first = life.next_action(120.0, 1000.0, protocols, noon)
    soon = life.next_action(125.0, 1005.0, protocols, noon)
    later = life.next_action(400.0, 1000.0 + IdleLife.ACTION_INTERVAL_S[1], protocols, noon)

    assert first is not None and soon is None and later is not None


def test_quiet_hours_only_allow_silent_looks() -> None:
    """During quiet hours the droid never beeps or speaks."""
    life = IdleLife(random.Random(1))
    protocols = InteractionProtocols(quiet_hours="22:00-08:00")
    kinds = set()
    for step in range(30):
        action = life.next_action(600.0, step * 1000.0, protocols, datetime(2026, 1, 1, 2))
        assert action is not None
        kinds.add(action.kind)

    assert kinds == {"look"}


def test_never_remarks_when_disabled() -> None:
    """unprompted_remarks=never suppresses spoken remarks."""
    life = IdleLife(random.Random(2))
    protocols = InteractionProtocols(unprompted_remarks="never")
    kinds = {
        action.kind
        for step in range(50)
        if (action := life.next_action(600.0, step * 1000.0, protocols, datetime(2026, 1, 1, 12))) is not None
    }

    assert "remark" not in kinds
