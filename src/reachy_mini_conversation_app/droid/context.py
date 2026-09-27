"""Assemble the droid's per-session context: identity first, then memories and mood."""

from pathlib import Path

from reachy_mini_conversation_app.droid.identity import load_identity, format_identity_for_prompt


def build_droid_context(instance_path: str | Path | None) -> str:
    """Return the block injected ahead of the droid persona instructions."""
    return format_identity_for_prompt(load_identity(instance_path))
