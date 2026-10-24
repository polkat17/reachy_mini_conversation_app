from k3_droid.droid.addressing import is_addressed, should_reply, address_names


NAMES = address_names("K3-L0")


def test_names_include_the_designation_its_short_form_and_spoken_aliases() -> None:
    """K3-L0 answers to "k3 l0", "k3", and the Russian ways of saying it."""
    assert {"k3 l0", "k3", "кей три", "дроид"} <= set(NAMES)


def test_is_addressed_matches_whole_words_in_russian_and_english() -> None:
    """The name counts as a whole word, whatever the punctuation or ё spelling."""
    assert is_addressed("Кей-три, какая погода?", NAMES)
    assert is_addressed("Слушай, дроид: включи свет", NAMES)
    assert is_addressed("Hey K3, turn left", NAMES)
    assert not is_addressed("Ты купила хлеб?", NAMES)
    assert not is_addressed("роботизированный пылесос сломался", NAMES)


def test_side_conversations_get_no_reply_when_two_people_are_in_view() -> None:
    """Two people talking to each other: silence, unless named or just after the droid's own reply."""
    common = {"names": NAMES, "follow_up_window_s": 30.0}
    assert not should_reply("Ты купила хлеб?", people_in_view=2, seconds_since_engaged=120.0, **common)
    assert should_reply("Кей-три, ты купил хлеб?", people_in_view=2, seconds_since_engaged=120.0, **common)
    assert should_reply("А завтра?", people_in_view=2, seconds_since_engaged=10.0, **common)
    assert should_reply("Какая погода?", people_in_view=1, seconds_since_engaged=120.0, **common)
