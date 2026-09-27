import json

import pytest

from reachy_mini_conversation_app.droid import journal as journal_mod
from reachy_mini_conversation_app.droid.llm import parse_json_object
from reachy_mini_conversation_app.droid.mood import MoodInputs, current_mood
from reachy_mini_conversation_app.droid.journal import SessionJournal, summarize_session, write_diary_entry
from reachy_mini_conversation_app.droid.identity import Identity


LINES = [("user", "My sister Anna visits on Friday."), ("assistant", "Noted. I will polish my antennas.")]


def test_journal_skips_sensor_events_and_resets() -> None:
    """Only real conversation lines are kept, and take() starts a fresh transcript."""
    journal = SessionJournal()
    journal.add("user", "  hello   there ")
    journal.add("assistant", "[SYSTEM EVENT] should not be stored")

    started_at, lines = journal.take()

    assert lines == [("user", "hello there")]
    assert journal.lines == [] and journal.started_at >= started_at


def test_summary_parses_model_json(monkeypatch: pytest.MonkeyPatch) -> None:
    """The model's JSON becomes an episode summary plus durable facts."""
    reply = {
        "summary": "Pasha said his sister Anna visits on Friday.",
        "mood": "happy",
        "tags": ["family"],
        "facts": [{"subject": "Anna", "text": "Anna is Pasha's sister"}, {"text": ""}],
    }
    monkeypatch.setattr(journal_mod, "complete", lambda *_args, **_kwargs: f"Sure!\n{json.dumps(reply)}")

    summary = summarize_session(LINES, Identity(), [])

    assert summary.mood == "happy" and summary.tags == ["family"]
    assert summary.facts == [("Anna", "Anna is Pasha's sister")]


def test_summary_falls_back_without_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no model reachable, an excerpt is still saved and no facts are invented."""
    monkeypatch.setattr(journal_mod, "complete", lambda *_args, **_kwargs: None)

    summary = summarize_session(LINES, Identity(), [])

    assert "sister Anna" in summary.summary and summary.facts == []


def test_diary_falls_back_without_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """The diary still gets an entry when the model is unavailable."""
    monkeypatch.setattr(journal_mod, "complete", lambda *_args, **_kwargs: None)

    assert "did not talk" in write_diary_entry(Identity(), [], [])


@pytest.mark.parametrize(
    ("text", "expected"),
    [('```json\n{"a": 1}\n```', {"a": 1}), ("no json here", None), ("[1, 2]", None), (None, None)],
)
def test_parse_json_object(text: str | None, expected: object) -> None:
    """JSON objects are pulled out of chatty replies; anything else is None."""
    assert parse_json_object(text) == expected


@pytest.mark.parametrize(
    ("inputs", "label"),
    [
        (MoodInputs(72, 0, 0, []), "lonely"),
        (MoodInputs(1, 1, 3, ["sad", "tired", "happy"]), "protective"),
        (MoodInputs(0.1, 6, 12, []), "energetic"),
        (MoodInputs(14, 0, 4, []), "bored"),
        (MoodInputs(2, 1, 12, []), "content"),
        (MoodInputs(2, 1, 3, []), "curious"),
    ],
)
def test_moods(inputs: MoodInputs, label: str) -> None:
    """Mood follows how much the droid has seen its owner, and how they seemed."""
    assert current_mood(inputs, "Pasha").label == label
