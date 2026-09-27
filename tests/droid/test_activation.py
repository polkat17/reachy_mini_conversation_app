from pathlib import Path

import pytest

from reachy_mini_conversation_app import prompts as prompts_mod
from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.droid.identity import Identity, save_identity
from reachy_mini_conversation_app.droid.activation import (
    STAGE_IDS,
    ActivationError,
    ActivationProgress,
    summarize,
    apply_answer,
)


def test_progress_persists_and_resumes(tmp_path: Path) -> None:
    """A power cut mid-activation resumes at the same stage."""
    progress = ActivationProgress(tmp_path)
    progress.advance()
    progress.advance()

    assert ActivationProgress(tmp_path).stage.id == STAGE_IDS[2]


def test_redo_and_unknown_stage(tmp_path: Path) -> None:
    """Stages can be revisited by id; unknown ids are rejected."""
    progress = ActivationProgress(tmp_path)

    assert progress.go_to("voice").id == "voice"
    with pytest.raises(ActivationError):
        progress.go_to("warp_drive")


@pytest.mark.parametrize(
    ("stage", "value"),
    [
        ("designation", {"name": ""}),
        ("wake_phrase", {"phrase": "kay"}),
        ("calibration", {"humour": "slapstick", "talkativeness": "brief", "language": "English"}),
        ("protocols", {"unprompted_remarks": "sometimes", "quiet_hours": "late"}),
        ("protocols", {"unprompted_remarks": "sometimes", "break_nudge_minutes": "lots"}),
        ("memory_consent", {"consent": "maybe"}),
    ],
)
def test_invalid_answers_are_rejected(stage: str, value: dict[str, object]) -> None:
    """Answers the droid cannot use come back as errors it can relay."""
    with pytest.raises(ActivationError):
        apply_answer(stage, value, Identity())


def test_answers_fill_the_identity() -> None:
    """Valid answers land in the identity and show up in the summary."""
    identity = Identity()
    apply_answer("designation", {"name": "Bolt"}, identity)
    apply_answer("wake_phrase", {"phrase": "Hey Bolt!"}, identity)
    apply_answer("calibration", {"humour": "Sarcastic", "talkativeness": "brief", "language": "English"}, identity)
    apply_answer("directives", {"interests": ["chess", " "], "avoid_topics": ["politics"]}, identity)
    apply_answer(
        "protocols",
        {"greet_on_sight": True, "unprompted_remarks": "rarely", "quiet_hours": "none", "break_nudge_minutes": 90},
        identity,
    )
    apply_answer("household", {"members": [{"name": "Miso", "kind": "cat"}], "guest_policy": "polite"}, identity)
    apply_answer("memory_consent", {"consent": True}, identity)

    assert identity.droid_name == "Bolt" and identity.wake_phrase == "hey bolt"
    assert identity.humour == "sarcastic" and identity.interests == ["chess"]
    assert identity.protocols.quiet_hours == "" and identity.protocols.break_nudge_minutes == 90
    assert identity.household[0].name == "Miso" and identity.memory_consent is True
    assert "Miso (cat)" in summarize(identity)


def test_unactivated_droid_gets_activation_instructions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Before activation, the session instructions and greeting drive the protocol."""
    monkeypatch.setattr(config, "REACHY_MINI_CUSTOM_PROFILE", "droid")
    monkeypatch.setattr(config, "INSTANCE_PATH", tmp_path)

    instructions = prompts_mod.get_session_instructions(instance_path=tmp_path)
    greeting = prompts_mod.get_session_greeting_prompt()

    assert "## ACTIVATION PROTOCOL" in instructions and "Current stage: boot." in instructions
    assert "activation_step" in greeting

    save_identity(tmp_path, Identity(activated=True))

    assert "ACTIVATION PROTOCOL" not in prompts_mod.get_session_instructions(instance_path=tmp_path)
    assert "activation_step" not in prompts_mod.get_session_greeting_prompt()
