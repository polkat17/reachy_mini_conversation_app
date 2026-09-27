"""The first-boot activation protocol: an ordered, resumable interview that fills in ``core.yaml``."""

import json
import logging
from typing import Any
from pathlib import Path
from dataclasses import dataclass

from reachy_mini_conversation_app.droid.identity import Identity, HouseholdMember
from reachy_mini_conversation_app.droid.idle_life import parse_quiet_hours
from reachy_mini_conversation_app.droid.wake_phrase import validate_phrase, normalize_phrase


logger = logging.getLogger(__name__)

STATE_FILENAME = "activation.json"
HUMOUR_LEVELS = ("dry", "sarcastic", "minimal", "warm")
TALKATIVENESS_LEVELS = ("brief", "chatty")
REMARK_LEVELS = ("never", "rarely", "sometimes", "often")


@dataclass(frozen=True)
class Stage:
    """One activation stage: its id and what the droid should do in it."""

    id: str
    instructions: str


STAGES: tuple[Stage, ...] = (
    Stage(
        "boot",
        "Stage 0, boot self-test. Call activation_step with action 'save' and no value: it runs the self-test "
        "motion. Then announce: systems online, activation protocol engaged, no owner registered yet.",
    ),
    Stage(
        "owner",
        "Stage 1, owner registration. Your owner on file is {owner}. Ask them to face your optical sensor and "
        "turn their head slightly, then call enroll_face with role 'owner'. When it succeeds, call "
        "activation_step 'save'.",
    ),
    Stage(
        "designation",
        "Stage 2, designation. Your factory designation is {droid_name}. Ask whether to keep it or assign a new "
        "name. Spell back the chosen name, then save value {{'name': ...}}.",
    ),
    Stage(
        "wake_phrase",
        "Stage 3, wake phrase. Ask your owner to choose the phrase that wakes you from dormant mode: two to four "
        "ordinary words, for example 'hey' plus your name, with letters and digits spelled as spoken ('kay three'). "
        "Save value {{'phrase': ...}}. Then ask them to say it three times and call activation_step "
        "'check_wake_test'. If the test fails, suggest a clearer phrase and repeat.",
    ),
    Stage(
        "voice",
        "Stage 4, vocal module. Offer two or three of these voices: {voices}. For each, call activation_step "
        "'try_voice' with value {{'voice': ...}} and speak one short sample line. Then save value {{'voice': ...}}.",
    ),
    Stage(
        "calibration",
        "Stage 5, personality calibration. Ask for humour (dry, sarcastic, minimal or warm), talkativeness "
        "(brief or chatty) and language. Say one sample line in the chosen style, confirm, then save value "
        "{{'humour': ..., 'talkativeness': ..., 'language': ...}}.",
    ),
    Stage(
        "directives",
        "Stage 6, primary directives. Ask what {owner} is interested in, which topics to follow (teams, games, "
        "news), and anything you must never bring up. Read the lists back, then save value "
        "{{'interests': [...], 'avoid_topics': [...]}}.",
    ),
    Stage(
        "protocols",
        "Stage 7, interaction protocols. Ask: greet them when you see them? how often to speak up unprompted "
        "(never, rarely, sometimes, often)? quiet hours (like 23:00-07:00, or none)? follow their face? break "
        "reminders after how many minutes at the desk (0 for never)? Read back, then save value "
        "{{'greet_on_sight': bool, 'unprompted_remarks': ..., 'quiet_hours': ..., 'follow_face': bool, "
        "'break_nudge_minutes': int}}.",
    ),
    Stage(
        "household",
        "Stage 8, household registry. Ask about other people and pets in the home (names, what they are, a short "
        "note) and how to treat guests. Save value {{'members': [{{'name': ..., 'kind': 'person'|'cat'|'dog'|..., "
        "'notes': ...}}], 'guest_policy': ...}}.",
    ),
    Stage(
        "memory_consent",
        "Stage 9, memory consent. Explain: you keep short text notes about conversations and facts, never "
        "recordings or images, stored only on this robot, and {owner} can say 'forget' at any time. Ask for "
        "consent and save value {{'consent': true or false}}.",
    ),
    Stage(
        "upgrade_briefing",
        "Stage 10, upgrade protocol. Explain: when you cannot do something, you can file an upgrade request; a "
        "coding assistant prepares the change and {owner} approves every upgrade on their phone before you "
        "install it while dormant. Then save.",
    ),
    Stage(
        "commit",
        "Stage 11, commit. Read the profile summary back briefly, ask for final confirmation, then call "
        "activation_step 'commit'. Afterwards greet {owner} as a fully activated droid.",
    ),
)
STAGE_IDS = tuple(stage.id for stage in STAGES)


class ActivationError(ValueError):
    """An answer that cannot be saved, with a message the droid can relay."""


def _require_str(value: dict[str, Any], key: str, max_len: int = 80) -> str:
    text = str(value.get(key) or "").strip()
    if not text or len(text) > max_len:
        raise ActivationError(f"'{key}' must be a non-empty text of at most {max_len} characters.")
    return text


def _string_list(value: dict[str, Any], key: str) -> list[str]:
    items = value.get(key) or []
    if not isinstance(items, list):
        raise ActivationError(f"'{key}' must be a list.")
    return [str(item).strip() for item in items if str(item).strip()]


def _choice(value: dict[str, Any], key: str, choices: tuple[str, ...]) -> str:
    choice = str(value.get(key) or "").strip().lower()
    if choice not in choices:
        raise ActivationError(f"'{key}' must be one of: {', '.join(choices)}.")
    return choice


def apply_answer(stage_id: str, value: dict[str, Any], identity: Identity) -> None:
    """Validate a stage answer and write it into ``identity``; raises ``ActivationError`` when invalid."""
    if stage_id == "designation":
        identity.droid_name = _require_str(value, "name", 24)
    elif stage_id == "wake_phrase":
        phrase = _require_str(value, "phrase")
        problem = validate_phrase(phrase, identity.droid_name)
        if problem:
            raise ActivationError(problem)
        identity.wake_phrase = normalize_phrase(phrase)
    elif stage_id == "voice":
        identity.voice = _require_str(value, "voice", 40)
    elif stage_id == "calibration":
        identity.humour = _choice(value, "humour", HUMOUR_LEVELS)
        identity.talkativeness = _choice(value, "talkativeness", TALKATIVENESS_LEVELS)
        identity.language = _require_str(value, "language", 40)
    elif stage_id == "directives":
        identity.interests = _string_list(value, "interests")
        identity.avoid_topics = _string_list(value, "avoid_topics")
    elif stage_id == "protocols":
        quiet_hours = str(value.get("quiet_hours") or "").strip()
        if quiet_hours.lower() in ("none", "no", "off"):
            quiet_hours = ""
        if quiet_hours and parse_quiet_hours(quiet_hours) is None:
            raise ActivationError("'quiet_hours' must look like 23:00-07:00, or be empty.")
        protocols = identity.protocols
        protocols.greet_on_sight = bool(value.get("greet_on_sight", protocols.greet_on_sight))
        protocols.unprompted_remarks = _choice(value, "unprompted_remarks", REMARK_LEVELS)
        protocols.quiet_hours = quiet_hours
        protocols.follow_face = bool(value.get("follow_face", protocols.follow_face))
        try:
            protocols.break_nudge_minutes = max(0, int(value.get("break_nudge_minutes", 0)))
        except (TypeError, ValueError) as e:
            raise ActivationError("'break_nudge_minutes' must be a whole number.") from e
    elif stage_id == "household":
        members = value.get("members") or []
        if not isinstance(members, list):
            raise ActivationError("'members' must be a list.")
        identity.household = [
            HouseholdMember(
                name=str(member.get("name")).strip(),
                kind=str(member.get("kind") or "person").strip(),
                notes=str(member.get("notes") or "").strip(),
            )
            for member in members
            if isinstance(member, dict) and str(member.get("name") or "").strip()
        ]
        policy = str(value.get("guest_policy") or "").strip()
        if policy:
            identity.guest_policy = policy
    elif stage_id == "memory_consent":
        if not isinstance(value.get("consent"), bool):
            raise ActivationError("'consent' must be true or false.")
        identity.memory_consent = value["consent"]


def summarize(identity: Identity) -> str:
    """Return the profile read back before committing."""
    protocols = identity.protocols
    household = ", ".join(f"{member.name} ({member.kind})" for member in identity.household) or "none"
    return (
        f"Designation {identity.droid_name}; owner {identity.owner_name}; wake phrase '{identity.wake_phrase}'; "
        f"voice {identity.voice or 'default'}; humour {identity.humour}, {identity.talkativeness}, "
        f"{identity.language}; interests {', '.join(identity.interests) or 'none'}; avoid "
        f"{', '.join(identity.avoid_topics) or 'nothing'}; greet on sight {protocols.greet_on_sight}; remarks "
        f"{protocols.unprompted_remarks}; quiet hours {protocols.quiet_hours or 'none'}; follow face "
        f"{protocols.follow_face}; break reminders {protocols.break_nudge_minutes or 'off'} min; household "
        f"{household}; memory consent {identity.memory_consent}."
    )


class ActivationProgress:
    """The current stage, persisted in ``activation.json`` so activation resumes after a power cut."""

    def __init__(self, data_dir: Path) -> None:
        """Load progress from ``data_dir``."""
        self._path = data_dir / STATE_FILENAME
        self.stage_index = self._load()

    def _load(self) -> int:
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            index = int(raw.get("stage_index", 0))
        except FileNotFoundError:
            return 0
        except (OSError, ValueError, TypeError, AttributeError) as e:
            logger.warning("Restarting activation; could not read %s: %s", self._path, e)
            return 0
        return min(max(index, 0), len(STAGES) - 1)

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps({"stage_index": self.stage_index}), encoding="utf-8")

    @property
    def stage(self) -> Stage:
        """Return the current stage."""
        return STAGES[self.stage_index]

    def advance(self) -> Stage:
        """Move to the next stage and return it."""
        self.stage_index = min(self.stage_index + 1, len(STAGES) - 1)
        self._save()
        return self.stage

    def go_to(self, stage_id: str) -> Stage:
        """Jump to ``stage_id`` (to redo it)."""
        if stage_id not in STAGE_IDS:
            raise ActivationError(f"Unknown stage '{stage_id}'. Stages: {', '.join(STAGE_IDS)}.")
        self.stage_index = STAGE_IDS.index(stage_id)
        self._save()
        return self.stage

    def reset(self) -> None:
        """Start activation over from the boot self-test."""
        self.stage_index = 0
        self._save()


def activation_prompt(identity: Identity, stage: Stage) -> str:
    """Return the block that puts a not-yet-activated droid into activation mode."""
    return (
        "## ACTIVATION PROTOCOL\n"
        "You have just been switched on for the first time and must complete your activation protocol with "
        f"{identity.owner_name} before normal operation. Work through it one stage at a time with the "
        "activation_step tool: ask one short question at a time, confirm each answer, and never skip ahead. "
        "If you are unsure which stage you are in, call activation_step with action 'status'. "
        "Your owner can say 'redo' to repeat a stage.\n"
        f"Current stage: {stage.id}."
    )
