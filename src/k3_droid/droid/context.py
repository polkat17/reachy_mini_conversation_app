"""Assemble the droid's per-session context: identity first, then memories."""

import time
from pathlib import Path
from datetime import datetime

from k3_droid.droid.identity import Identity, load_identity, droid_data_dir, format_identity_for_prompt
from k3_droid.droid.embeddings import TextEmbedder
from k3_droid.droid.memory_store import MEMORY_DB_FILENAME, MemoryStore


MEMORY_CHAR_BUDGET = 6000  # about 1,500 tokens; everything else is reachable through recall


def open_memory(instance_path: str | Path | None) -> MemoryStore:
    """Open the droid's memory database (the embedding model loads only when a search needs it)."""
    return MemoryStore(droid_data_dir(instance_path) / MEMORY_DB_FILENAME, TextEmbedder())


def format_memory_block(memory: MemoryStore, identity: Identity, now: float) -> str:
    """Return the MEMORY block: recent sessions and key facts, within the character budget."""
    if not identity.memory_consent:
        return "## MEMORY\nMemory is switched off: your owner did not consent. Do not store anything."
    lines = ["## MEMORY"]
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
    memory = open_memory(instance_path)
    try:
        sections.append(format_memory_block(memory, identity, time.time()))
    finally:
        memory.close()
    return "\n\n".join(sections)


def build_droid_greeting(instance_path: str | Path | None) -> str:
    """Return the prompt that opens each droid session."""
    identity = load_identity(instance_path)
    return (
        f"[SYSTEM EVENT] You just powered up. Greet {identity.owner_name} in one short, "
        "in-character line with a system status, and vary the wording each time. If a recent session is relevant, "
        "you may refer to it."
    )
