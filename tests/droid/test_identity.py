from pathlib import Path

import pytest

from reachy_mini_conversation_app import prompts as prompts_mod
from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.droid.identity import (
    IDENTITY_FILENAME,
    Identity,
    HouseholdMember,
    load_identity,
    save_identity,
    droid_data_dir,
    format_identity_for_prompt,
)


def test_missing_identity_uses_factory_defaults(tmp_path: Path) -> None:
    """A fresh droid is K3-L0, owned by Pasha, and not yet activated."""
    identity = load_identity(tmp_path)

    assert identity.droid_name == "K3-L0"
    assert identity.owner_name == "Pasha"
    assert identity.activated is False


def test_identity_round_trips_through_core_yaml(tmp_path: Path) -> None:
    """Saved identity is read back unchanged."""
    identity = Identity(droid_name="Bolt", interests=["chess"], household=[HouseholdMember("Miso", "cat")])
    identity.protocols.quiet_hours = "23:00-07:00"

    save_identity(tmp_path, identity)

    assert (droid_data_dir(tmp_path) / IDENTITY_FILENAME).is_file()
    assert load_identity(tmp_path) == identity


def test_hand_edited_identity_keeps_defaults_for_bad_fields(tmp_path: Path) -> None:
    """Malformed hand edits fall back per field instead of discarding the file."""
    path = droid_data_dir(tmp_path) / IDENTITY_FILENAME
    path.parent.mkdir(parents=True)
    path.write_text("droid_name: Zed\ninterests: not-a-list\nhousehold:\n  - notes: nameless\n", encoding="utf-8")

    identity = load_identity(tmp_path)

    assert identity.droid_name == "Zed"
    assert identity.interests == []
    assert identity.household == []


def test_unreadable_identity_falls_back(tmp_path: Path) -> None:
    """Invalid YAML never breaks startup."""
    path = droid_data_dir(tmp_path) / IDENTITY_FILENAME
    path.parent.mkdir(parents=True)
    path.write_text("droid_name: [unclosed", encoding="utf-8")

    assert load_identity(tmp_path) == Identity()


def test_prompt_block_mentions_household_and_quirks() -> None:
    """The identity block carries names, household and self-chosen quirks."""
    identity = Identity(household=[HouseholdMember("Miso", "cat", "likes antennas")], quirks=["dislikes Mondays"])

    block = format_identity_for_prompt(identity)

    assert "Your designation: K3-L0." in block
    assert "Miso (cat, likes antennas)" in block
    assert "dislikes Mondays" in block


def test_droid_profile_instructions_start_with_identity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The droid profile gets its identity block instead of the generic memory block."""
    monkeypatch.setattr(config, "REACHY_MINI_CUSTOM_PROFILE", "droid")
    save_identity(tmp_path, Identity(droid_name="Bolt"))

    instructions = prompts_mod.get_session_instructions(instance_path=tmp_path)

    assert instructions.startswith("## DROID IDENTITY")
    assert "Your designation: Bolt." in instructions
    assert "companion droid from the future" in instructions
