"""Assemble the droid's per-session context: identity first, then memories and mood."""

from pathlib import Path

from reachy_mini_conversation_app.droid.identity import load_identity, droid_data_dir, format_identity_for_prompt
from reachy_mini_conversation_app.droid.activation import ActivationProgress, activation_prompt


def build_droid_context(instance_path: str | Path | None) -> str:
    """Return the block injected ahead of the droid persona instructions."""
    identity = load_identity(instance_path)
    sections = [format_identity_for_prompt(identity)]
    if not identity.activated:
        sections.append(activation_prompt(identity, ActivationProgress(droid_data_dir(instance_path)).stage))
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
        "in-character line with a system status, and vary the wording each time."
    )
