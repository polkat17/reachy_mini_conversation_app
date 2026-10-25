"""Session journal: the rolling transcript and its end-of-session summary."""

import time
from dataclasses import field, dataclass

from k3_droid.droid.llm import complete, parse_json_object
from k3_droid.droid.identity import Identity


MAX_TRANSCRIPT_CHARS = 12000
CHECKPOINT_TURNS = 60


@dataclass
class SessionJournal:
    """Text transcript of the current conversation."""

    started_at: float = field(default_factory=time.time)
    lines: list[tuple[str, str]] = field(default_factory=list)

    def add(self, role: str, text: str) -> None:
        """Append one final transcript line."""
        text = " ".join(text.split())
        if text and not text.startswith("[SYSTEM EVENT]"):
            self.lines.append((role, text))

    def take(self) -> tuple[float, list[tuple[str, str]]]:
        """Return the transcript so far and start a new one."""
        started_at, lines = self.started_at, self.lines
        self.started_at, self.lines = time.time(), []
        return started_at, lines


@dataclass(frozen=True)
class SessionSummary:
    """What a session leaves behind in long-term memory."""

    summary: str
    mood: str
    tags: list[str]
    facts: list[tuple[str, str]]


def format_transcript(lines: list[tuple[str, str]], identity: Identity) -> str:
    """Render the transcript with names, keeping the most recent part when it is long."""
    speakers = {"user": identity.owner_name, "assistant": identity.droid_name}
    text = "\n".join(f"{speakers.get(role, role)}: {line}" for role, line in lines)
    return text[-MAX_TRANSCRIPT_CHARS:]


_SUMMARY_SYSTEM = (
    "You maintain the long-term memory of a companion robot. Given a conversation transcript, reply with JSON only: "
    '{"summary": "2-4 sentences in third person: what was discussed, decided or planned", '
    '"mood": "one word for the owner\'s mood", "tags": ["up to 5 short topic tags"], '
    '"facts": [{"subject": "who or what the fact is about", "text": "one durable fact"}]}. '
    "Only include facts likely to stay true for weeks (names, relationships, preferences, plans, important dates). "
    "Skip small talk and anything already listed as known. Never include passwords, addresses or payment details."
)


def summarize_session(lines: list[tuple[str, str]], identity: Identity, known_facts: list[str]) -> SessionSummary:
    """Summarise a transcript with the background model, or fall back to a plain excerpt when it is unavailable."""
    transcript = format_transcript(lines, identity)
    known = "\n".join(f"- {fact}" for fact in known_facts[:60]) or "- none"
    reply = parse_json_object(
        complete(_SUMMARY_SYSTEM, f"Owner: {identity.owner_name}\nKnown facts:\n{known}\n\nTranscript:\n{transcript}")
    )
    if reply is None:
        owner_lines = [line for role, line in lines if role == "user"]
        excerpt = " / ".join(owner_lines[:3])[:400] or "a short exchange"
        return SessionSummary(f"{identity.owner_name} talked with {identity.droid_name}: {excerpt}", "", [], [])

    facts: list[tuple[str, str]] = []
    for item in reply.get("facts") or []:
        if isinstance(item, dict) and str(item.get("text") or "").strip():
            facts.append((str(item.get("subject") or "").strip(), str(item["text"]).strip()))
    tags = [str(tag).strip() for tag in reply.get("tags") or [] if str(tag).strip()][:5]
    return SessionSummary(
        summary=str(reply.get("summary") or "").strip() or "Conversation without a summary.",
        mood=str(reply.get("mood") or "").strip()[:30],
        tags=tags,
        facts=facts[:10],
    )
