import json
from datetime import date, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from k3_droid.droid import runtime as runtime_mod
from k3_droid.droid import day_review as day_review_mod
from k3_droid.droid.day_review import briefing_line, fetch_day_review
from k3_droid.tools.core_tools import ToolDependencies
from k3_droid.tools.day_review import DayReview


REVIEW = {
    "date": "2026-10-19",
    "spoken_summary": "Three meetings today; the budget review matters most.",
    "priorities": [{"rank": 1, "title": "Budget review at 10:00"}, {"rank": 2, "title": "Reply to Anna"}],
    "meetings": [{"start": "10:00", "end": "11:00", "title": "Budget review", "importance": "high"}],
}


def _serve(monkeypatch: pytest.MonkeyPatch, tmp_path, payload: object) -> None:
    path = tmp_path / "day_review.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(day_review_mod, "hf_hub_download", lambda *args, **kwargs: str(path))


def test_fetch_day_review_returns_todays_review(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Today's review is returned as written by the routine."""
    _serve(monkeypatch, tmp_path, REVIEW)
    review, note = fetch_day_review("owner/briefing", "token", date(2026, 10, 19))
    assert review == REVIEW
    assert note == "today's review loaded from owner/briefing"


def test_fetch_day_review_ignores_a_stale_or_broken_file(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Yesterday's review, non-objects and download failures all read as no review."""
    _serve(monkeypatch, tmp_path, REVIEW)
    assert fetch_day_review("owner/briefing", "token", date(2026, 10, 20)) == (
        None,
        "no review for today yet (latest is for 2026-10-19)",
    )
    _serve(monkeypatch, tmp_path, ["not", "a", "review"])
    assert fetch_day_review("owner/briefing", "token", date(2026, 10, 19))[0] is None

    def fail(*args: object, **kwargs: object) -> str:
        raise OSError("offline")

    monkeypatch.setattr(day_review_mod, "hf_hub_download", fail)
    assert fetch_day_review("owner/briefing", "token", date(2026, 10, 19)) == (
        None,
        "could not read owner/briefing: OSError",
    )


def test_briefing_line_keeps_the_summary_and_top_priorities() -> None:
    """The briefing line is the spoken summary plus the first priorities."""
    line = briefing_line(REVIEW)
    assert line.startswith("Three meetings today")
    assert "Top priorities: Budget review at 10:00; Reply to Anna." in line


class _FrozenDatetime(datetime):
    @classmethod
    def now(cls, tz=None):  # type: ignore[override]
        return cls(2026, 10, 19, 8, 0)


@pytest.mark.asyncio
async def test_morning_briefing_includes_the_day_review(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The first morning sighting adds the review, and the runtime caches the dataset read."""
    monkeypatch.setenv("DROID_DAY_REVIEW_DATASET", "owner/briefing")
    monkeypatch.setattr(runtime_mod, "datetime", _FrozenDatetime)
    fetch = MagicMock(return_value=(REVIEW, "today's review loaded from owner/briefing"))
    monkeypatch.setattr(runtime_mod, "fetch_day_review", fetch)
    runtime = runtime_mod.DroidRuntime(MagicMock(), tmp_path)
    runtime.identity = MagicMock(activated=True, owner_name="Pasha")
    runtime.owner_in_view = MagicMock(return_value=True)  # type: ignore[method-assign]
    runtime.seconds_since_activity = MagicMock(return_value=60.0)  # type: ignore[method-assign]
    runtime.say_event = AsyncMock(return_value=True)  # type: ignore[method-assign]

    await runtime._maybe_morning_briefing()

    prompt = runtime.say_event.await_args.args[0]
    assert "budget review matters most" in prompt
    assert "day_review tool" in prompt
    assert runtime.day_review() == REVIEW
    assert runtime.day_review_note == "today's review loaded from owner/briefing"
    fetch.assert_called_once()


@pytest.mark.asyncio
async def test_day_review_tool_stays_private_with_guests() -> None:
    """Only the owner hears the review; guests get a refusal."""
    droid = MagicMock()
    droid.owner_access_allowed.return_value = (False, "Pasha is not in view")
    result = await DayReview()(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock(), droid=droid))
    assert result == {"error": "Pasha is not in view"}
    droid.day_review.assert_not_called()

    droid.owner_access_allowed.return_value = (True, "")
    droid.day_review.return_value = REVIEW
    result = await DayReview()(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock(), droid=droid))
    assert result == REVIEW


class _NineFifty(datetime):
    @classmethod
    def now(cls, tz=None):  # type: ignore[override]
        return cls(2026, 10, 19, 9, 50)


@pytest.mark.asyncio
async def test_meeting_heads_up_fires_once_ten_minutes_before(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """At 09:50 the 10:00 meeting gets one heads-up with what to prepare; the next tick stays quiet."""
    monkeypatch.setattr(runtime_mod, "datetime", _NineFifty)
    review = dict(
        REVIEW, meetings=[{"start": "10:00", "title": "Budget review", "with": "Anna", "prepare": "Q3 numbers"}]
    )
    runtime = runtime_mod.DroidRuntime(MagicMock(), tmp_path)
    runtime.identity = MagicMock(activated=True, owner_name="Pasha", protocols=MagicMock(quiet_hours=""))
    runtime.seconds_since_activity = MagicMock(return_value=60.0)  # type: ignore[method-assign]
    runtime.day_review = MagicMock(return_value=review)  # type: ignore[method-assign]
    runtime.say_event = AsyncMock(return_value=True)  # type: ignore[method-assign]

    await runtime._maybe_meeting_heads_up()
    await runtime._maybe_meeting_heads_up()

    runtime.say_event.assert_awaited_once()
    prompt = runtime.say_event.await_args.args[0]
    assert "Budget review starts at 10:00" in prompt and "Q3 numbers" in prompt and "Anna" in prompt
