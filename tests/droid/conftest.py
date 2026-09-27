"""Shared fixtures for droid tests: a deterministic embedder so memory tests never download a model."""

import re
import hashlib

import numpy as np
import pytest
from numpy.typing import NDArray

from reachy_mini_conversation_app.droid import context as context_mod


class BagOfWordsEmbedder:
    """Hash each word into a fixed vector; texts sharing words get similar embeddings."""

    def embed(self, texts: list[str]) -> NDArray[np.float32]:
        """Return one normalised bag-of-words vector per text."""
        rows = np.zeros((len(texts), 256), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in re.findall(r"[a-z0-9]+", text.lower()):
                if len(word) > 2:
                    rows[row, int(hashlib.md5(word.encode()).hexdigest(), 16) % 256] += 1.0
        norms = np.maximum(np.linalg.norm(rows, axis=1, keepdims=True), 1e-9)
        return rows / norms


@pytest.fixture(autouse=True)
def fake_text_embedder(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every memory store opened by the droid use the fake embedder."""
    monkeypatch.setattr(context_mod, "TextEmbedder", BagOfWordsEmbedder)
