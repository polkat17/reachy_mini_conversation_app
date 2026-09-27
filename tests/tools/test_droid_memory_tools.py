import re
import hashlib
from pathlib import Path
from datetime import datetime
from unittest.mock import MagicMock

import numpy as np
import pytest
from numpy.typing import NDArray

from reachy_mini_conversation_app.droid import context as context_mod
from reachy_mini_conversation_app.tools.recall import Recall
from reachy_mini_conversation_app.droid.runtime import DroidRuntime
from reachy_mini_conversation_app.tools.memorize import Memorize
from reachy_mini_conversation_app.tools.reminder import Reminder, due_time
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.tools.read_diary import ReadDiary
from reachy_mini_conversation_app.tools.forget_memory import ForgetMemory
from reachy_mini_conversation_app.tools.list_memories import ListMemories


class _WordEmbedder:
    def embed(self, texts: list[str]) -> NDArray[np.float32]:
        rows = np.zeros((len(texts), 256), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in re.findall(r"[a-z]{3,}", text.lower()):
                rows[row, int(hashlib.md5(word.encode()).hexdigest(), 16) % 256] += 1.0
        return rows / np.maximum(np.linalg.norm(rows, axis=1, keepdims=True), 1e-9)


@pytest.fixture
def deps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ToolDependencies:
    """Dependencies with an activated, consenting droid and a word-hash embedder."""
    monkeypatch.setattr(context_mod, "TextEmbedder", _WordEmbedder)
    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock(), instance_path=tmp_path)
    droid = DroidRuntime(deps, tmp_path)
    droid.identity.activated = True
    droid.identity.memory_consent = True
    deps.droid = droid
    return deps


@pytest.mark.asyncio
async def test_memorize_then_recall_and_list(deps: ToolDependencies) -> None:
    """Saved facts are found by meaning and listed by subject."""
    await Memorize()(deps, text="Pasha supports Arsenal football club", subject="Pasha")

    recalled = await Recall()(deps, query="which football club")
    listed = await ListMemories()(deps, subject="Pasha")

    assert "Arsenal" in recalled["memories"][0]
    assert listed["facts"] == ["Pasha: Pasha supports Arsenal football club"]


@pytest.mark.asyncio
async def test_memorize_respects_consent(deps: ToolDependencies) -> None:
    """Without consent nothing is stored."""
    assert deps.droid is not None
    deps.droid.identity.memory_consent = False

    result = await Memorize()(deps, text="Pasha likes tea")

    assert "consent" in result["error"] and deps.droid.memory.facts() == []


@pytest.mark.asyncio
async def test_memories_stay_private_from_strangers(deps: ToolDependencies) -> None:
    """With only a stranger in view, memory tools refuse to share."""
    droid = deps.droid
    assert droid is not None
    droid.presence = MagicMock()
    droid.presence.tracker.present.return_value = ["stranger"]

    for result in (
        await Recall()(deps, query="anything"),
        await ListMemories()(deps),
        await ForgetMemory()(deps, query="anything"),
        await ReadDiary()(deps),
    ):
        assert "stays private" in result["error"]


@pytest.mark.asyncio
async def test_forget_needs_confirmation(deps: ToolDependencies) -> None:
    """The first call only lists matches; deletion happens with confirmed ids."""
    await Memorize()(deps, text="Pasha is allergic to peanuts", subject="Pasha")
    await Memorize()(deps, text="Miso the cat hates the vacuum", subject="Miso")

    found = await ForgetMemory()(deps, query="peanuts allergy")
    assert deps.droid is not None and len(deps.droid.memory.facts()) == 2

    deleted = await ForgetMemory()(deps, confirm_ids=[found["matches"][0]["id"]])

    assert deleted == {"status": "deleted 1 memories"}
    assert [fact.subject for fact in deps.droid.memory.facts()] == ["Miso"]


def test_due_time_parsing() -> None:
    """Timers, clock times (rolling to tomorrow) and full dates resolve; nonsense does not."""
    now = datetime(2026, 9, 27, 20, 0)

    assert due_time(15, None, now) == datetime(2026, 9, 27, 20, 15)
    assert due_time(None, "21:30", now) == datetime(2026, 9, 27, 21, 30)
    assert due_time(None, "07:00", now) == datetime(2026, 9, 28, 7, 0)
    assert due_time(None, "2026-10-02 09:00", now) == datetime(2026, 10, 2, 9, 0)
    assert due_time(None, "soonish", now) is None


@pytest.mark.asyncio
async def test_reminder_set_list_cancel(deps: ToolDependencies) -> None:
    """Reminders round-trip through the tool."""
    created = await Reminder()(deps, action="set", text="Pizza out of the oven", in_minutes=12)
    listed = await Reminder()(deps, action="list")
    cancelled = await Reminder()(deps, action="cancel", id=created["id"])

    assert created["status"] == "set"
    assert listed["reminders"][0]["text"] == "Pizza out of the oven"
    assert cancelled == {"status": "cancelled"}
    assert "error" in await Reminder()(deps, action="set", text="no time given")


@pytest.mark.asyncio
async def test_read_diary(deps: ToolDependencies) -> None:
    """The latest entry is read when no day is given."""
    assert deps.droid is not None
    deps.droid.memory.write_diary("2026-09-26", "Pasha taught me chess.")

    assert (await ReadDiary()(deps))["entry"] == "Pasha taught me chess."
    assert "error" in await ReadDiary()(deps, day="1999-01-01")
