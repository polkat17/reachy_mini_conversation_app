import time
from pathlib import Path

from reachy_mini_conversation_app.droid.context import (
    MEMORY_CHAR_BUDGET,
    OWNER_LAST_SEEN_KEY,
    open_memory,
    format_memory_block,
)
from reachy_mini_conversation_app.droid.identity import Identity


def test_memory_block_respects_consent(tmp_path: Path) -> None:
    """Without consent the block says memory is off and lists nothing."""
    memory = open_memory(tmp_path)
    memory.remember("Pasha likes tea")

    block = format_memory_block(memory, Identity(memory_consent=False), time.time())

    assert "switched off" in block and "tea" not in block


def test_memory_block_has_mood_reminders_sessions_and_facts(tmp_path: Path) -> None:
    """The block carries mood, upcoming reminders, recent sessions and facts."""
    memory = open_memory(tmp_path)
    now = time.time()
    memory.set_state(OWNER_LAST_SEEN_KEY, str(now - 3 * 86400))
    memory.remember("Pasha supports Arsenal", "Pasha")
    memory.add_episode(now - 600, now - 60, "Planned a trip to Lisbon", "excited", ["travel"])
    memory.add_reminder(now + 3600, "Book the flights")

    block = format_memory_block(memory, Identity(memory_consent=True), now)

    assert "Current mood: lonely" in block
    assert "Book the flights" in block and "Lisbon" in block and "Pasha: Pasha supports Arsenal" in block


def test_memory_block_stays_within_budget(tmp_path: Path) -> None:
    """However many facts exist, the injected block stays under the character budget."""
    memory = open_memory(tmp_path)
    for index in range(80):
        memory.remember(f"Fact number {index} about item{index} " + "detail " * 20)

    block = format_memory_block(memory, Identity(memory_consent=True), time.time())

    assert len(block) <= MEMORY_CHAR_BUDGET
