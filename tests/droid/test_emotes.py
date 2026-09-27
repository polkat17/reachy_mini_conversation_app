import pytest

from reachy_mini_conversation_app.droid.emotes import EmoteGuard, pick_fallback_emotion
from reachy_mini_conversation_app.conversation_handler import ConversationEvent


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("I am sorry to hear that.", "sad"),
        ("Of course. I predicted this.", "smug"),
        ("Excellent work, Pasha.", "happy"),
        ("Where did you put it?", "curious"),
        ("Done.", "yes"),
    ],
)
def test_pick_fallback_emotion(reply: str, expected: str) -> None:
    """Replies map to a fitting emotion intent."""
    assert pick_fallback_emotion(reply) == expected


def _run(events: list[ConversationEvent]) -> list[str]:
    played: list[str] = []
    guard = EmoteGuard(played.append)
    for event in events:
        guard.on_event(event)
    return played


def test_reply_without_emote_gets_fallback() -> None:
    """A spoken reply with no emote tool call triggers exactly one fallback."""
    played = _run(
        [
            ConversationEvent("user_transcript", "hi"),
            ConversationEvent("assistant_transcript", "Hello, Pasha."),
            ConversationEvent("response_done"),
            ConversationEvent("response_done"),
        ]
    )

    assert played == ["happy"]


def test_reply_with_emote_is_left_alone() -> None:
    """An emote from the model in the same user turn suppresses the fallback."""
    played = _run(
        [
            ConversationEvent("user_transcript", "hi"),
            ConversationEvent("tool_call", "play_emotion"),
            ConversationEvent("response_done"),
            ConversationEvent("assistant_transcript", "Hello."),
            ConversationEvent("response_done"),
        ]
    )

    assert played == []


def test_tool_only_response_does_not_emote() -> None:
    """Responses without speech never trigger the fallback."""
    played = _run(
        [
            ConversationEvent("user_transcript", "look left"),
            ConversationEvent("tool_call", "move_head"),
            ConversationEvent("response_done"),
        ]
    )

    assert played == []
