"""The droid's long-term memory: facts, episodes, diary and reminders in SQLite, with semantic search.

Text only: raw audio and video are never stored. Vectors are kept as float32 blobs and searched by brute force,
which is instant for the few thousand rows a companion accumulates, so no vector index is needed.
"""

import time
import logging
import sqlite3
import threading
from typing import Literal, Protocol
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


logger = logging.getLogger(__name__)

MEMORY_DB_FILENAME = "memory.db"
DUPLICATE_SIMILARITY = 0.85

RefType = Literal["fact", "episode"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    subject TEXT NOT NULL DEFAULT '',
    text TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS episodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at REAL NOT NULL,
    ended_at REAL NOT NULL,
    summary TEXT NOT NULL,
    mood TEXT NOT NULL DEFAULT '',
    tags TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS embeddings (
    ref_type TEXT NOT NULL,
    ref_id INTEGER NOT NULL,
    vector BLOB NOT NULL,
    PRIMARY KEY (ref_type, ref_id)
);
CREATE TABLE IF NOT EXISTS diary (
    day TEXT PRIMARY KEY,
    entry TEXT NOT NULL,
    written_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS reminders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    due_at REAL NOT NULL,
    text TEXT NOT NULL,
    created_at REAL NOT NULL,
    delivered_at REAL
);
CREATE TABLE IF NOT EXISTS state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


class Embedder(Protocol):
    """Anything that turns texts into unit-length vectors."""

    def embed(self, texts: list[str]) -> NDArray[np.float32]:
        """Return one normalised row per text."""
        ...


@dataclass(frozen=True)
class Fact:
    """A stable fact, optionally about a subject (a person, a pet, a team)."""

    id: int
    subject: str
    text: str
    updated_at: float


@dataclass(frozen=True)
class Episode:
    """A summarised conversation session."""

    id: int
    started_at: float
    ended_at: float
    summary: str
    mood: str
    tags: str


@dataclass(frozen=True)
class Reminder:
    """A reminder or timer due at ``due_at`` (Unix time)."""

    id: int
    due_at: float
    text: str


@dataclass(frozen=True)
class Recollection:
    """A search hit: a fact or an episode with its similarity to the query."""

    ref_type: RefType
    ref_id: int
    text: str
    score: float


class MemoryStore:
    """Thread-safe access to ``memory.db``."""

    def __init__(self, path: Path, embedder: Embedder) -> None:
        """Open (and create) the database at ``path``."""
        path.parent.mkdir(parents=True, exist_ok=True)
        self._embedder = embedder
        self._lock = threading.RLock()
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.executescript(_SCHEMA)
        self._db.commit()

    def close(self) -> None:
        """Close the database connection."""
        with self._lock:
            self._db.close()

    # ---- embeddings ----------------------------------------------------------

    def _store_vector(self, ref_type: RefType, ref_id: int, text: str) -> None:
        vector = self._embedder.embed([text])[0]
        self._db.execute(
            "INSERT OR REPLACE INTO embeddings (ref_type, ref_id, vector) VALUES (?, ?, ?)",
            (ref_type, ref_id, vector.astype(np.float32).tobytes()),
        )

    def _vectors(self, ref_type: RefType | None = None) -> tuple[list[tuple[str, int]], NDArray[np.float32]]:
        query = "SELECT ref_type, ref_id, vector FROM embeddings"
        rows = self._db.execute(query + (" WHERE ref_type = ?" if ref_type else ""), (ref_type,) if ref_type else ())
        refs: list[tuple[str, int]] = []
        vectors: list[NDArray[np.float32]] = []
        for row_type, row_id, blob in rows:
            refs.append((row_type, row_id))
            vectors.append(np.frombuffer(blob, dtype=np.float32))
        return refs, (np.vstack(vectors) if vectors else np.zeros((0, 1), dtype=np.float32))

    # ---- facts ----------------------------------------------------------------

    def remember(self, text: str, subject: str = "") -> tuple[Fact, bool]:
        """Store a fact, or update a near-duplicate; returns the fact and whether it was new."""
        text, subject = " ".join(text.split()), subject.strip()
        now = time.time()
        with self._lock:
            duplicate = self._closest_fact(text)
            if duplicate is not None:
                self._db.execute(
                    "UPDATE facts SET text = ?, subject = ?, updated_at = ? WHERE id = ?",
                    (text, subject or duplicate.subject, now, duplicate.id),
                )
                self._store_vector("fact", duplicate.id, text)
                self._db.commit()
                return Fact(duplicate.id, subject or duplicate.subject, text, now), False
            cursor = self._db.execute(
                "INSERT INTO facts (subject, text, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (subject, text, now, now),
            )
            fact_id = int(cursor.lastrowid or 0)
            self._store_vector("fact", fact_id, text)
            self._db.commit()
            return Fact(fact_id, subject, text, now), True

    def _closest_fact(self, text: str) -> Fact | None:
        refs, vectors = self._vectors("fact")
        if not refs:
            return None
        scores = vectors @ self._embedder.embed([text])[0]
        best = int(np.argmax(scores))
        if float(scores[best]) < DUPLICATE_SIMILARITY:
            return None
        return self.fact(refs[best][1])

    def fact(self, fact_id: int) -> Fact | None:
        """Return one fact by id."""
        with self._lock:
            row = self._db.execute(
                "SELECT id, subject, text, updated_at FROM facts WHERE id = ?", (fact_id,)
            ).fetchone()
        return Fact(*row) if row else None

    def facts(self, subject: str = "", limit: int = 200) -> list[Fact]:
        """Return facts, most recently updated first, optionally only about ``subject``."""
        with self._lock:
            if subject:
                rows = self._db.execute(
                    "SELECT id, subject, text, updated_at FROM facts WHERE lower(subject) = lower(?) "
                    "ORDER BY updated_at DESC LIMIT ?",
                    (subject.strip(), limit),
                )
            else:
                rows = self._db.execute(
                    "SELECT id, subject, text, updated_at FROM facts ORDER BY updated_at DESC LIMIT ?", (limit,)
                )
            return [Fact(*row) for row in rows]

    def delete_facts(self, fact_ids: list[int]) -> int:
        """Delete facts by id; returns how many were removed."""
        with self._lock:
            removed = 0
            for fact_id in fact_ids:
                removed += self._db.execute("DELETE FROM facts WHERE id = ?", (fact_id,)).rowcount
                self._db.execute("DELETE FROM embeddings WHERE ref_type = 'fact' AND ref_id = ?", (fact_id,))
            self._db.commit()
            return removed

    # ---- episodes -------------------------------------------------------------

    def add_episode(self, started_at: float, ended_at: float, summary: str, mood: str, tags: list[str]) -> Episode:
        """Store one summarised session."""
        with self._lock:
            cursor = self._db.execute(
                "INSERT INTO episodes (started_at, ended_at, summary, mood, tags) VALUES (?, ?, ?, ?, ?)",
                (started_at, ended_at, summary, mood, ", ".join(tags)),
            )
            episode_id = int(cursor.lastrowid or 0)
            self._store_vector("episode", episode_id, summary)
            self._db.commit()
        return Episode(episode_id, started_at, ended_at, summary, mood, ", ".join(tags))

    def episodes(self, limit: int = 3, since: float = 0.0) -> list[Episode]:
        """Return the most recent episodes, newest first."""
        with self._lock:
            rows = self._db.execute(
                "SELECT id, started_at, ended_at, summary, mood, tags FROM episodes WHERE ended_at >= ? "
                "ORDER BY ended_at DESC LIMIT ?",
                (since, limit),
            )
            return [Episode(*row) for row in rows]

    def delete_episodes(self, episode_ids: list[int]) -> int:
        """Delete episodes by id; returns how many were removed."""
        with self._lock:
            removed = 0
            for episode_id in episode_ids:
                removed += self._db.execute("DELETE FROM episodes WHERE id = ?", (episode_id,)).rowcount
                self._db.execute("DELETE FROM embeddings WHERE ref_type = 'episode' AND ref_id = ?", (episode_id,))
            self._db.commit()
            return removed

    # ---- search ---------------------------------------------------------------

    def recall(self, query: str, k: int = 5, min_score: float = 0.25) -> list[Recollection]:
        """Return the facts and episodes most similar to ``query``."""
        with self._lock:
            refs, vectors = self._vectors()
            if not refs:
                return []
            scores = vectors @ self._embedder.embed([query])[0]
            hits: list[Recollection] = []
            for index in np.argsort(-scores)[: max(k, 1)]:
                score = float(scores[index])
                if score < min_score:
                    break
                ref_type, ref_id = refs[index]
                if ref_type == "fact":
                    row = self._db.execute("SELECT subject, text FROM facts WHERE id = ?", (ref_id,)).fetchone()
                    text = f"{row[0]}: {row[1]}" if row and row[0] else (row[1] if row else "")
                else:
                    row = self._db.execute("SELECT ended_at, summary FROM episodes WHERE id = ?", (ref_id,)).fetchone()
                    text = f"[{datetime.fromtimestamp(row[0]):%Y-%m-%d}] {row[1]}" if row else ""
                if text:
                    hits.append(Recollection("fact" if ref_type == "fact" else "episode", ref_id, text, score))
            return hits

    # ---- diary ----------------------------------------------------------------

    def write_diary(self, day: str, entry: str) -> None:
        """Store (or replace) the diary entry for ``day`` (YYYY-MM-DD)."""
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO diary (day, entry, written_at) VALUES (?, ?, ?)", (day, entry, time.time())
            )
            self._db.commit()

    def diary(self, day: str) -> str | None:
        """Return the diary entry for ``day``."""
        with self._lock:
            row = self._db.execute("SELECT entry FROM diary WHERE day = ?", (day,)).fetchone()
        return str(row[0]) if row else None

    def diary_days(self, limit: int = 14) -> list[str]:
        """Return the most recent days with a diary entry."""
        with self._lock:
            return [row[0] for row in self._db.execute("SELECT day FROM diary ORDER BY day DESC LIMIT ?", (limit,))]

    # ---- reminders --------------------------------------------------------------

    def add_reminder(self, due_at: float, text: str) -> Reminder:
        """Schedule a reminder."""
        with self._lock:
            cursor = self._db.execute(
                "INSERT INTO reminders (due_at, text, created_at) VALUES (?, ?, ?)", (due_at, text, time.time())
            )
            self._db.commit()
            return Reminder(int(cursor.lastrowid or 0), due_at, text)

    def pending_reminders(self) -> list[Reminder]:
        """Return undelivered reminders, soonest first."""
        with self._lock:
            rows = self._db.execute(
                "SELECT id, due_at, text FROM reminders WHERE delivered_at IS NULL ORDER BY due_at"
            )
            return [Reminder(*row) for row in rows]

    def due_reminders(self, now: float) -> list[Reminder]:
        """Return undelivered reminders due at or before ``now``."""
        return [reminder for reminder in self.pending_reminders() if reminder.due_at <= now]

    def mark_delivered(self, reminder_id: int) -> None:
        """Mark a reminder as delivered."""
        with self._lock:
            self._db.execute("UPDATE reminders SET delivered_at = ? WHERE id = ?", (time.time(), reminder_id))
            self._db.commit()

    def cancel_reminder(self, reminder_id: int) -> bool:
        """Delete a pending reminder."""
        with self._lock:
            removed = self._db.execute(
                "DELETE FROM reminders WHERE id = ? AND delivered_at IS NULL", (reminder_id,)
            ).rowcount
            self._db.commit()
            return removed > 0

    # ---- small key/value state (mood, last briefing, ...) -------------------------

    def get_state(self, key: str, default: str = "") -> str:
        """Read a stored value."""
        with self._lock:
            row = self._db.execute("SELECT value FROM state WHERE key = ?", (key,)).fetchone()
        return str(row[0]) if row else default

    def set_state(self, key: str, value: str) -> None:
        """Write a stored value."""
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO state (key, value) VALUES (?, ?)", (key, value))
            self._db.commit()

    def wipe(self) -> None:
        """Delete every memory (factory reset)."""
        with self._lock:
            for table in ("facts", "episodes", "embeddings", "diary", "reminders", "state"):
                self._db.execute(f"DELETE FROM {table}")
            self._db.commit()
