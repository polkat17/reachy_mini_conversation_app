"""Decide whether something said in the room was meant for the droid.

With automatic replies off, the realtime server keeps every finished utterance as context but answers only when
asked. The droid answers when it is named, during a short follow-up window after its own reply, or when at most one
person is in view; people talking to each other otherwise get no reply.
"""

import re


# How people address the droid out loud besides its designation; transcripts come in Russian or English.
DEFAULT_ADDRESS_NAMES = ("к3", "кей три", "кейтри", "кэй три", "дроид", "дроидик", "робот", "k3", "droid", "robot")


def normalize_speech(text: str) -> str:
    """Lower-case, fold ё into е, and turn punctuation and hyphens into single spaces."""
    return " ".join(re.sub(r"[^\w]+", " ", text.lower().replace("ё", "е")).split())


def address_names(droid_name: str, extra: tuple[str, ...] = DEFAULT_ADDRESS_NAMES) -> tuple[str, ...]:
    """Return the normalized names the droid answers to: its designation, its leading part (K3 of K3-L0) and extras."""
    designation = normalize_speech(droid_name)
    names = {designation, *(normalize_speech(name) for name in extra)}
    if designation.split():
        names.add(designation.split()[0])
    return tuple(sorted(name for name in names if name))


def is_addressed(transcript: str, names: tuple[str, ...]) -> bool:
    """Return whether ``transcript`` names the droid as a whole word or phrase."""
    spoken = f" {normalize_speech(transcript)} "
    return any(f" {name} " in spoken for name in names)


def should_reply(
    transcript: str,
    *,
    names: tuple[str, ...],
    people_in_view: int,
    seconds_since_engaged: float,
    follow_up_window_s: float,
) -> bool:
    """Return whether the droid should answer this utterance."""
    if is_addressed(transcript, names):
        return True
    if seconds_since_engaged <= follow_up_window_s:
        return True
    return people_in_view <= 1
