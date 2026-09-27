import time
from pathlib import Path

from reachy_mini_conversation_app.droid import context as context_mod
from reachy_mini_conversation_app.droid.memory_store import MemoryStore


def _store(tmp_path: Path) -> MemoryStore:
    # The autouse fixture in conftest.py swaps TextEmbedder for a deterministic bag-of-words embedder.
    return MemoryStore(tmp_path / "memory.db", context_mod.TextEmbedder())


def test_remember_updates_near_duplicates(tmp_path: Path) -> None:
    """Restating a fact updates it instead of adding a duplicate."""
    store = _store(tmp_path)
    first, created = store.remember("Pasha supports Arsenal football club", "Pasha")
    second, created_again = store.remember("Pasha supports Arsenal football club!", "")

    assert created is True and created_again is False
    assert second.id == first.id and second.subject == "Pasha"
    assert len(store.facts()) == 1


def test_recall_ranks_by_meaning(tmp_path: Path) -> None:
    """The closest fact or episode comes first; unrelated rows are filtered out."""
    store = _store(tmp_path)
    store.remember("Miso the cat loves chasing antennas", "Miso")
    store.remember("Pasha works as a software engineer", "Pasha")
    store.add_episode(time.time() - 60, time.time(), "Pasha lost at chess and wants a rematch", "annoyed", ["chess"])

    cat = store.recall("what does the cat love chasing")
    chess = store.recall("chess rematch")

    assert cat[0].ref_type == "fact" and "Miso" in cat[0].text
    assert chess[0].ref_type == "episode" and "chess" in chess[0].text
    assert store.recall("quantum spaghetti") == []


def test_facts_filter_and_delete(tmp_path: Path) -> None:
    """Facts can be listed per subject and deleted with their vectors."""
    store = _store(tmp_path)
    miso, _ = store.remember("Miso sleeps on the keyboard", "Miso")
    store.remember("Pasha drinks green tea", "Pasha")

    assert [fact.text for fact in store.facts("miso")] == ["Miso sleeps on the keyboard"]
    assert store.delete_facts([miso.id]) == 1
    assert store.recall("keyboard sleeping") == []


def test_reminders_lifecycle(tmp_path: Path) -> None:
    """Reminders become due, are delivered once, and can be cancelled while pending."""
    store = _store(tmp_path)
    now = time.time()
    due = store.add_reminder(now - 1, "Take the pizza out")
    later = store.add_reminder(now + 3600, "Call mum")

    assert [r.id for r in store.due_reminders(now)] == [due.id]
    store.mark_delivered(due.id)
    assert store.due_reminders(now) == []
    assert store.cancel_reminder(later.id) is True
    assert store.pending_reminders() == []


def test_diary_state_persistence_and_wipe(tmp_path: Path) -> None:
    """Everything survives reopening the database; wipe removes it all."""
    store = _store(tmp_path)
    store.write_diary("2026-09-26", "Pasha taught me chess. I lost gracefully.")
    store.set_state("mood", "smug")
    store.remember("Pasha likes chess")
    store.close()

    reopened = _store(tmp_path)
    assert reopened.diary("2026-09-26") is not None and reopened.diary_days() == ["2026-09-26"]
    assert reopened.get_state("mood") == "smug" and len(reopened.facts()) == 1

    reopened.wipe()
    assert reopened.facts() == [] and reopened.diary_days() == [] and reopened.get_state("mood") == ""
