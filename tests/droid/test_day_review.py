import json
from datetime import date
from unittest.mock import MagicMock

import pytest

from k3_droid.droid import day_review as day_review_mod
from k3_droid.droid.day_review import fetch_day_review
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
