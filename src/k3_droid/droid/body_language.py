"""Pick the droid's body language from its own words, so emotions play while it speaks.

The small realtime model cannot reliably speak and call a tool in the same turn: it either splits the two or reads
the call out loud. So the model only talks, and the droid's code chooses an emotion from the reply (or from an
explicit "покажи радость") and queues it while the speech plays.
"""

import re


# (word stems, emotion intent); the first match wins, so stronger feelings come first. Russian and English.
# A stem matches at a word start; a trailing space makes it a whole word ("пока " must not match "покажу").
_REPLY_EMOTIONS: tuple[tuple[tuple[str, ...], str], ...] = (
    (
        ("ура ", "поздравля", "отлично", "супер ", "здорово", "великолепн", "great", "congratulations", "excellent"),
        "happy",
    ),
    (("жаль", "сожалею", "грустн", "печальн", "увы ", "неприятн", "sorry", "unfortunate", "sad"), "sad"),
    (("осторожн", "опасн", "тревож", "беспоко", "careful", "danger", "worried"), "worried"),
    (("ого ", "ничего себе", "неожиданн", "удивительн", "невероятн", "wow", "whoa", "amazing"), "surprised"),
    (
        ("привет", "здравствуй", "добро пожаловать", "доброе утро", "добрый день", "добрый вечер", "hello", "welcome"),
        "welcoming",
    ),
    (("пока ", "до свидания", "до встречи", "спокойной ночи", "goodbye", "good night"), "goodbye"),
    (("спасибо", "благодар", "thank"), "grateful"),
    (("разумеется", "естественно", "очевидно", "конечно же", "of course", "obviously"), "smug"),
    (("хм ", "дайте подумать", "дай подумать", "посмотрим", "возможно", "hmm", "let me think", "perhaps"), "thinking"),
    (("не понимаю", "не уверен", "странно", "confused", "not sure"), "confused"),
)

# What the owner might ask the droid to show ("покажи радость", "изобрази грусть", "show me happy").
_REQUEST_VERBS = ("покажи", "изобрази", "сыграй", "сделай", "show", "act")
_REQUESTED_EMOTIONS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("радост", "счасть", "весел", "happy", "joy"), "happy"),
    (("грус", "печал", "sad"), "sad"),
    (("злост", "злой", "гнев", "сердит", "angry"), "angry"),
    (("страх", "испуг", "боязн", "scared", "afraid"), "scared"),
    (("удивлен", "surprise"), "surprised"),
    (("любов", "нежност", "love"), "loving"),
    (("скук", "bored"), "bored"),
    (("усталост", "устал", "tired"), "tired"),
    (("смущен", "стыд", "embarrass"), "embarrassed"),
    (("отвращен", "disgust"), "disgusted"),
    (("задумчив", "размышл", "thinking"), "thinking"),
    (("любопытств", "интерес", "curious"), "curious"),
    (("эмоци", "emotion"), "random"),
)


def normalize(text: str) -> str:
    """Lower-case, fold ё into е and collapse punctuation into single spaces."""
    return " ".join(re.sub(r"[^\w]+", " ", text.lower().replace("ё", "е")).split())


def _matches(spoken: str, stems: tuple[str, ...]) -> bool:
    return any(f" {stem}" in f" {spoken} " for stem in stems)


def emotion_for_reply(reply: str) -> str | None:
    """Return the emotion intent that fits the droid's reply, or None when it is neutral."""
    spoken = normalize(reply)
    for stems, emotion in _REPLY_EMOTIONS:
        if _matches(spoken, stems):
            return emotion
    return None


def requested_emotion(utterance: str) -> str | None:
    """Return the emotion the owner explicitly asked to see, or None."""
    spoken = normalize(utterance)
    if not _matches(spoken, _REQUEST_VERBS):
        return None
    for stems, emotion in _REQUESTED_EMOTIONS:
        if _matches(spoken, stems):
            return emotion
    return None
