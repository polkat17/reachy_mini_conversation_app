import pytest

from k3_droid.droid.wake_phrase import VOSK_MODELS, phrase_language, validate_phrase, normalize_phrase


@pytest.mark.parametrize(
    ("phrase", "expected"),
    [("Hey K3!", "hey k three"), ("  Wake  UP, Bolt ", "wake up bolt"), ("R2 go", "r two go")],
)
def test_normalize_phrase(phrase: str, expected: str) -> None:
    """Phrases are lower-cased, digits spelled out, punctuation dropped."""
    assert normalize_phrase(phrase, "en") == expected


def test_normalize_russian_phrase() -> None:
    """Russian phrases keep Cyrillic, fold ё into е and spell digits in Russian."""
    assert normalize_phrase("Привет, К3! Ёлка", "ru") == "привет к три елка"


def test_wake_phrase_model_follows_the_language() -> None:
    """Russian gets the Russian model; unsupported languages fall back to English."""
    assert phrase_language("ru") == "ru"
    assert phrase_language("ru-RU") == "ru"
    assert phrase_language("de") == "en"
    assert VOSK_MODELS[phrase_language("ru")] == "vosk-model-small-ru-0.22"


@pytest.mark.parametrize(
    ("phrase", "ok"),
    [("hey kay three", True), ("kay", False), ("one two three four five", False), ("hi yo", False)],
)
def test_validate_phrase(phrase: str, ok: bool) -> None:
    """Two to four words, not too short."""
    assert (validate_phrase(phrase, "K3-L0", "en") is None) is ok


def test_bare_name_is_rejected() -> None:
    """The droid's own name alone would wake it on every mention."""
    assert validate_phrase("bolt one", "Bolt 1", "en") is not None
    assert validate_phrase("болт один", "Болт 1", "ru") is not None
