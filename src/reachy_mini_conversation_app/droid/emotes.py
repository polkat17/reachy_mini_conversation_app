"""Guarantee body language: queue a fallback emote when a spoken reply came without one."""

import re
from collections.abc import Callable

from reachy_mini_conversation_app.conversation_handler import ConversationEvent


EMOTE_TOOL_NAMES = frozenset({"play_emotion", "beep", "dance"})

_KEYWORD_EMOTIONS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("sorry", "unfortunate", "sad", "regret", "condolences"), "sad"),
    (("careful", "warning", "danger", "worried", "concern"), "worried"),
    (("whoa", "what?!", "unexpected", "startled", "alarming"), "startled"),
    (("of course", "obviously", "naturally", "as expected"), "smug"),
    (("great", "excellent", "congratulations", "wonderful", "hello", "welcome"), "happy"),
    (("let me think", "calculating", "processing", "hmm", "perhaps"), "thinking"),
)


def pick_fallback_emotion(reply: str) -> str:
    """Choose an emotion intent that fits a spoken reply."""
    lowered = reply.lower()
    for keywords, emotion in _KEYWORD_EMOTIONS:
        if any(re.search(rf"\b{re.escape(keyword)}", lowered) for keyword in keywords):
            return emotion
    if reply.rstrip().endswith("?"):
        return "curious"
    return "yes"


class EmoteGuard:
    """Watch conversation events and emote after any user turn whose spoken replies had no emote."""

    def __init__(self, play_emotion: Callable[[str], None]) -> None:
        """Store the callback that queues an emotion intent."""
        self._play_emotion = play_emotion
        self._emoted_this_turn = False
        self._reply_text = ""

    def on_event(self, event: ConversationEvent) -> None:
        """Update turn state and fire the fallback when a response finishes without an emote."""
        if event.kind in ("user_transcript", "session_started"):
            self._emoted_this_turn = False
            self._reply_text = ""
        elif event.kind == "tool_call" and event.text in EMOTE_TOOL_NAMES:
            self._emoted_this_turn = True
        elif event.kind == "assistant_transcript":
            self._reply_text = event.text
        elif event.kind == "response_done" and self._reply_text:
            if not self._emoted_this_turn:
                self._play_emotion(pick_fallback_emotion(self._reply_text))
                self._emoted_this_turn = True
            self._reply_text = ""
