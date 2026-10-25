import re
import hashlib
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from numpy.typing import NDArray

from k3_droid.droid import context as context_mod
from k3_droid.tools.recall import Recall
from k3_droid.droid.runtime import DroidRuntime
from k3_droid.tools.memorize import Memorize
from k3_droid.tools.core_tools import ToolDependencies
from k3_droid.tools.forget_memory import ForgetMemory
from k3_droid.tools.list_memories import ListMemories


class _WordEmbedder:
    def embed(self, texts: list[str]) -> NDArray[np.float32]:
        rows = np.zeros((len(texts), 256), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in re.findall(r"[a-z]{3,}", text.lower()):
                rows[row, int(hashlib.md5(word.encode()).hexdigest(), 16) % 256] += 1.0
        return rows / np.maximum(np.linalg.norm(rows, axis=1, keepdims=True), 1e-9)


@pytest.fixture
def deps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ToolDependencies:
    """Dependencies with a consenting droid and a word-hash embedder."""
    monkeypatch.setattr(context_mod, "TextEmbedder", _WordEmbedder)
    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock(), instance_path=tmp_path)
    droid = DroidRuntime(deps, tmp_path)
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
async def test_forget_needs_confirmation(deps: ToolDependencies) -> None:
    """The first call only lists matches; deletion happens with confirmed ids."""
    await Memorize()(deps, text="Pasha is allergic to peanuts", subject="Pasha")
    await Memorize()(deps, text="Miso the cat hates the vacuum", subject="Miso")

    found = await ForgetMemory()(deps, query="peanuts allergy")
    assert deps.droid is not None and len(deps.droid.memory.facts()) == 2

    deleted = await ForgetMemory()(deps, confirm_ids=[found["matches"][0]["id"]])

    assert deleted == {"status": "deleted 1 memories"}
    assert [fact.subject for fact in deps.droid.memory.facts()] == ["Miso"]
