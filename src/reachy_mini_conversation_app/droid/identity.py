"""The droid's identity file (``core.yaml``): who it is, who its owner is, and how it behaves."""

import os
import logging
import threading
from typing import Any
from pathlib import Path
from dataclasses import field, asdict, dataclass

import yaml

from reachy_mini_conversation_app.config import config


logger = logging.getLogger(__name__)

DROID_PROFILE_NAME = "droid"
IDENTITY_FILENAME = "core.yaml"
DEFAULT_DROID_NAME = "K3-L0"
DEFAULT_OWNER_NAME = "Pasha"

_IDENTITY_LOCK = threading.Lock()


@dataclass
class HouseholdMember:
    """A person or pet the droid should know by name."""

    name: str
    kind: str = "person"
    notes: str = ""


@dataclass
class InteractionProtocols:
    """Owner-chosen rules for proactive behaviour."""

    greet_on_sight: bool = True
    unprompted_remarks: str = "sometimes"
    quiet_hours: str = ""
    follow_face: bool = True
    break_nudge_minutes: int = 120


@dataclass
class Identity:
    """Everything the droid knows about itself and its owner, always injected into its instructions."""

    droid_name: str = DEFAULT_DROID_NAME
    owner_name: str = DEFAULT_OWNER_NAME
    wake_phrase: str = ""
    voice: str = ""
    humour: str = "dry"
    talkativeness: str = "brief"
    language: str = "English"
    interests: list[str] = field(default_factory=list)
    avoid_topics: list[str] = field(default_factory=list)
    protocols: InteractionProtocols = field(default_factory=InteractionProtocols)
    household: list[HouseholdMember] = field(default_factory=list)
    guest_policy: str = "friendly, but never share the owner's memories"
    memory_consent: bool = False
    quirks: list[str] = field(default_factory=list)
    activated: bool = False

    def to_yaml_dict(self) -> dict[str, Any]:
        """Return the plain mapping written to ``core.yaml``."""
        return asdict(self)


def is_droid_profile_active() -> bool:
    """Return whether the droid profile drives the current session."""
    return (config.REACHY_MINI_CUSTOM_PROFILE or "").strip() == DROID_PROFILE_NAME


def droid_data_dir(instance_path: str | Path | None) -> Path:
    """Return the directory holding the droid's identity, memory and face data."""
    if instance_path is not None:
        return Path(instance_path).expanduser() / "droid"
    data_home = os.getenv("XDG_DATA_HOME")
    data_root = Path(data_home).expanduser() if data_home else Path.home() / ".local" / "share"
    return data_root / "reachy_mini_conversation_app" / "droid"


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _identity_from_mapping(raw: dict[str, Any]) -> Identity:
    """Build an identity from a hand-edited mapping, keeping defaults for missing or malformed fields."""
    identity = Identity()
    for key in ("droid_name", "owner_name", "wake_phrase", "voice", "humour", "talkativeness", "language"):
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            setattr(identity, key, value.strip())
    identity.guest_policy = str(raw.get("guest_policy") or identity.guest_policy)
    identity.interests = _string_list(raw.get("interests"))
    identity.avoid_topics = _string_list(raw.get("avoid_topics"))
    identity.quirks = _string_list(raw.get("quirks"))
    identity.memory_consent = raw.get("memory_consent") is True
    identity.activated = raw.get("activated") is True

    protocols = raw.get("protocols")
    if isinstance(protocols, dict):
        defaults = InteractionProtocols()
        identity.protocols = InteractionProtocols(
            greet_on_sight=bool(protocols.get("greet_on_sight", defaults.greet_on_sight)),
            unprompted_remarks=str(protocols.get("unprompted_remarks", defaults.unprompted_remarks)),
            quiet_hours=str(protocols.get("quiet_hours") or ""),
            follow_face=bool(protocols.get("follow_face", defaults.follow_face)),
            break_nudge_minutes=int(protocols.get("break_nudge_minutes", defaults.break_nudge_minutes)),
        )

    household = raw.get("household")
    if isinstance(household, list):
        identity.household = [
            HouseholdMember(
                name=str(member["name"]).strip(),
                kind=str(member.get("kind") or "person"),
                notes=str(member.get("notes") or ""),
            )
            for member in household
            if isinstance(member, dict) and str(member.get("name") or "").strip()
        ]
    return identity


def load_identity(instance_path: str | Path | None) -> Identity:
    """Read ``core.yaml``, falling back to the factory identity when it is missing or unreadable."""
    path = droid_data_dir(instance_path) / IDENTITY_FILENAME
    with _IDENTITY_LOCK:
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return Identity()
        except (OSError, yaml.YAMLError) as e:
            logger.warning("Failed to read droid identity at %s: %s", path, e)
            return Identity()
    if not isinstance(raw, dict):
        logger.warning("Ignoring droid identity at %s: expected a mapping", path)
        return Identity()
    try:
        return _identity_from_mapping(raw)
    except (TypeError, ValueError) as e:
        logger.warning("Ignoring malformed droid identity at %s: %s", path, e)
        return Identity()


def save_identity(instance_path: str | Path | None, identity: Identity) -> None:
    """Atomically write ``core.yaml``."""
    path = droid_data_dir(instance_path) / IDENTITY_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with _IDENTITY_LOCK:
        tmp_path.write_text(
            yaml.safe_dump(identity.to_yaml_dict(), sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        tmp_path.replace(path)


def format_identity_for_prompt(identity: Identity) -> str:
    """Return the DROID IDENTITY block injected ahead of the persona."""
    lines = [
        "## DROID IDENTITY",
        f"Your designation: {identity.droid_name}.",
        f"Your owner: {identity.owner_name}.",
        f"Humour setting: {identity.humour}. Talkativeness: {identity.talkativeness}. Language: {identity.language}.",
    ]
    if identity.interests:
        lines.append(f"{identity.owner_name}'s interests: {', '.join(identity.interests)}.")
    if identity.avoid_topics:
        lines.append(f"Never bring up: {', '.join(identity.avoid_topics)}.")
    if identity.household:
        members = ", ".join(
            f"{member.name} ({member.kind}{', ' + member.notes if member.notes else ''})"
            for member in identity.household
        )
        lines.append(f"Household: {members}.")
    lines.append(f"Guests: {identity.guest_policy}.")
    if identity.quirks:
        lines.append(f"Your self-chosen quirks: {'; '.join(identity.quirks)}.")
    return "\n".join(lines)
