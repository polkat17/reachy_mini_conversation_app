import pytest

from reachy_mini_conversation_app.droid.wake_phrase import validate_phrase, normalize_phrase


@pytest.mark.parametrize(
    ("phrase", "expected"),
    [("Hey K3!", "hey k three"), ("  Wake  UP, Bolt ", "wake up bolt"), ("R2 go", "r two go")],
)
def test_normalize_phrase(phrase: str, expected: str) -> None:
    """Phrases are lower-cased, digits spelled out, punctuation dropped."""
    assert normalize_phrase(phrase) == expected


@pytest.mark.parametrize(
    ("phrase", "ok"),
    [("hey kay three", True), ("kay", False), ("one two three four five", False), ("hi yo", False)],
)
def test_validate_phrase(phrase: str, ok: bool) -> None:
    """Two to four words, not too short."""
    assert (validate_phrase(phrase, "K3-L0") is None) is ok


def test_bare_name_is_rejected() -> None:
    """The droid's own name alone would wake it on every mention."""
    assert validate_phrase("bolt one", "Bolt 1") is not None
