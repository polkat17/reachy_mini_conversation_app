"""Assemble the droid's per-session context: identity first, then memories and mood."""

import time
from pathlib import Path
from datetime import datetime

from reachy_mini_conversation_app.droid.mood import Mood, MoodInputs, current_mood
from reachy_mini_conversation_app.droid.identity import (
    Identity,
    load_identity,
    droid_data_dir,
    format_identity_for_prompt,
)
from reachy_mini_conversation_app.droid.activation import ActivationProgress, activation_prompt
from reachy_mini_conversation_app.droid.embeddings import TextEmbedder
from reachy_mini_conversation_app.droid.memory_store import MEMORY_DB_FILENAME, MemoryStore


MEMORY_CHAR_BUDGET = 6000  # about 1,500 tokens; everything else is reachable through recall
OWNER_LAST_SEEN_KEY = "owner_last_seen"


def open_memory(instance_path: str | Path | None) -> MemoryStore:
    """Open the droid's memory database (the embedding model loads only when a search needs it)."""
    return MemoryStore(droid_data_dir(instance_path) / MEMORY_DB_FILENAME, TextEmbedder())


def mood_for(memory: MemoryStore, identity: Identity, now: float) -> Mood:
    """Compute today's mood from recent memory."""
    last_seen = float(memory.get_state(OWNER_LAST_SEEN_KEY, str(now)))
    midnight = datetime.fromtimestamp(now).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
    week = memory.episodes(limit=100, since=now - 7 * 86400)
    inputs = MoodInputs(
        hours_since_owner=(now - last_seen) / 3600.0,
        conversations_today=sum(episode.ended_at >= midnight for episode in week),
        conversations_this_week=len(week),
        recent_owner_moods=[episode.mood for episode in week[:5] if episode.mood],
    )
    return current_mood(inputs, identity.owner_name)


def format_memory_block(memory: MemoryStore, identity: Identity, now: float) -> str:
    """Return the MEMORY block: mood, reminders, recent sessions and key facts, within the character budget."""
    if not identity.memory_consent:
        return "## MEMORY\nMemory is switched off: your owner did not consent. Do not store anything."
    mood = mood_for(memory, identity, now)
    lines = ["## MEMORY", f"Current mood: {mood.label} ({mood.reason}). Let it colour your tone slightly."]
    reminders = memory.pending_reminders()[:5]
    if reminders:
        lines.append("Upcoming reminders:")
        lines += [f"- {datetime.fromtimestamp(r.due_at):%a %H:%M}: {r.text}" for r in reminders]
    episodes = memory.episodes(limit=3)
    if episodes:
        lines.append("Recent sessions (mention them naturally when relevant):")
        lines += [f"- {datetime.fromtimestamp(e.ended_at):%a %d %b}: {e.summary}" for e in episodes]
    facts = memory.facts(limit=60)
    if facts:
        lines.append("Things you know (use recall for more):")
        lines += [f"- {f.subject + ': ' if f.subject else ''}{f.text}" for f in facts]
    block = ""
    for line in lines:
        if len(block) + len(line) + 1 > MEMORY_CHAR_BUDGET:
            break
        block += line + "\n"
    return block.rstrip()


def build_droid_context(instance_path: str | Path | None) -> str:
    """Return the block injected ahead of the droid persona instructions."""
    identity = load_identity(instance_path)
    sections = [format_identity_for_prompt(identity)]
    if not identity.activated:
        sections.append(activation_prompt(identity, ActivationProgress(droid_data_dir(instance_path)).stage))
    else:
        memory = open_memory(instance_path)
        try:
            sections.append(format_memory_block(memory, identity, time.time()))
        finally:
            memory.close()
    return "\n\n".join(sections)


def build_droid_greeting(instance_path: str | Path | None) -> str:
    """Return the prompt that opens each droid session."""
    identity = load_identity(instance_path)
    if not identity.activated:
        return (
            "[SYSTEM EVENT] Power on. Call activation_step with action 'status' and continue your activation "
            "protocol from the current stage. Open with a short droid-like boot announcement."
        )
    return (
        f"[SYSTEM EVENT] You just powered up or woke from dormant mode. Greet {identity.owner_name} in one short, "
        "in-character line with a system status, and vary the wording each time. If a recent session is relevant, "
        "you may refer to it."
    )
