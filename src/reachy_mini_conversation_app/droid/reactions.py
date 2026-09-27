"""What the droid says when its sensors notice something: pure functions returning ``[SYSTEM EVENT]`` prompts."""

from reachy_mini_conversation_app.droid.identity import Identity
from reachy_mini_conversation_app.droid.presence import CAT, STRANGER, PresenceEvent


WELCOME_BACK_AFTER_S = 10 * 60.0
GOODBYE_AFTER_STAY_S = 5 * 60.0


def _duration(seconds: float) -> str:
    minutes = int(seconds // 60)
    if minutes < 60:
        return f"{minutes} minutes"
    hours, minutes = divmod(minutes, 60)
    return f"{hours} hours" if minutes < 10 else f"{hours} hours {minutes} minutes"


def cat_names(identity: Identity) -> str:
    """Return how to refer to the household cat(s)."""
    names = [member.name for member in identity.household if member.kind.lower() in ("cat", "kitten")]
    return " or ".join(names) if names else "the cat"


def cat_prompt(identity: Identity, *, heard: bool) -> str:
    """Return the sensor event for noticing the cat."""
    sense = "You hear meowing: probably" if heard else "You see"
    return (
        f"[SYSTEM EVENT] {sense} {cat_names(identity)}. Talk to the cat in one or two short, gentle, playful lines "
        "(droids are fond of cats). Keep your movements small and slow so you do not startle it."
    )


def presence_prompt(event: PresenceEvent, identity: Identity) -> str | None:
    """Return the sensor event to voice for a presence change, or None to stay quiet."""
    owner = identity.owner_name
    if event.name == CAT:
        return cat_prompt(identity, heard=False) if event.kind == "arrived" else None
    if event.kind == "arrived":
        if event.name == STRANGER:
            return (
                "[SYSTEM EVENT] An unfamiliar person is in view. Greet them politely, introduce yourself, and ask "
                f"their name. You may chat freely, but never share anything you remember about {owner}. If {owner} "
                "introduces them, call enroll_face with role 'guest' and their name."
            )
        if event.is_owner:
            if not identity.protocols.greet_on_sight:
                return None
            if event.away_s == float("inf"):
                return f"[SYSTEM EVENT] {owner} is in view. Greet them briefly."
            if event.away_s >= WELCOME_BACK_AFTER_S:
                return (
                    f"[SYSTEM EVENT] {owner} is back after {_duration(event.away_s)} away. Welcome them back in one "
                    "short line, noting how long they were gone."
                )
            return None
        return f"[SYSTEM EVENT] {event.name}, a known guest, is in view. Greet them by name, briefly."
    if event.is_owner and event.stayed_s >= GOODBYE_AFTER_STAY_S and identity.protocols.greet_on_sight:
        return f"[SYSTEM EVENT] {owner} just walked out of view. Say a very short goodbye."
    return None


def break_nudge_prompt(identity: Identity, present_s: float) -> str:
    """Return the sensor event suggesting a break after a long desk session."""
    return (
        f"[SYSTEM EVENT] {identity.owner_name} has been in front of you for {_duration(present_s)}. Suggest a short "
        "stretch or water break in one line, in character."
    )
