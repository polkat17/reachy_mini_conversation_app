import pytest

from k3_droid.droid.body_language import emotion_for_reply, requested_emotion


@pytest.mark.parametrize(
    ("reply", "emotion"),
    [
        ("Поздравляю, отличная работа!", "happy"),
        ("Жаль это слышать.", "sad"),
        ("Ого, вот это новость.", "surprised"),
        ("Привет, Паша. Все системы в норме.", "welcoming"),
        ("Спасибо, очень мило.", "grateful"),
        ("Хм, дайте подумать.", "thinking"),
        ("Сейчас пятнадцать градусов и облачно.", None),
    ],
)
def test_emotion_for_reply(reply: str, emotion: str | None) -> None:
    """Russian replies map to fitting emotions; neutral ones get none."""
    assert emotion_for_reply(reply) == emotion


def test_word_stems_match_only_at_word_starts() -> None:
    """A short word inside another one ("пока" in "покажу") does not count."""
    assert emotion_for_reply("Сейчас покажу.") is None
    assert emotion_for_reply("Пока, до завтра.") == "goodbye"


@pytest.mark.parametrize(
    ("utterance", "emotion"),
    [
        ("Покажи радость!", "happy"),
        ("изобрази грусть", "sad"),
        ("Покажи какую-нибудь эмоцию", "random"),
        ("Я сегодня грустный", None),
        ("Покажи погоду", None),
    ],
)
def test_requested_emotion(utterance: str, emotion: str | None) -> None:
    """Only an explicit request to show a feeling counts."""
    assert requested_emotion(utterance) == emotion
