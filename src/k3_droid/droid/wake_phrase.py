"""Offline wake phrase spotting with Vosk: the recogniser only listens for the owner-chosen phrase."""

import re
import json
import logging
import zipfile
import threading
from pathlib import Path

import vosk
import httpx
import numpy as np
from numpy.typing import NDArray

from k3_droid.config import config
from k3_droid.droid.audio import LISTENER_SAMPLE_RATE


logger = logging.getLogger(__name__)

# Small offline models (about 40-45 MB each), picked by the conversation language; English is the fallback.
VOSK_MODELS = {"en": "vosk-model-small-en-us-0.15", "ru": "vosk-model-small-ru-0.22"}
VOSK_MODEL_BASE_URL = "https://alphacephei.com/vosk/models"
_DOWNLOAD_TIMEOUT_S = 300.0
_DIGIT_WORDS = {
    "en": ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"),
    "ru": ("ноль", "один", "два", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять"),
}


def phrase_language(language: str | None = None) -> str:
    """Return the supported wake-phrase language for a language code, from the config when not given."""
    code = (language if language is not None else config.REALTIME_TRANSCRIPTION_LANGUAGE) or "en"
    code = code.split("-")[0].lower()
    return code if code in VOSK_MODELS else "en"


def normalize_phrase(phrase: str, language: str | None = None) -> str:
    """Lower-case, spell out digits and drop punctuation: ``"Hey K3!"`` becomes ``"hey k three"``."""
    digits = _DIGIT_WORDS[phrase_language(language)]
    spelled = re.sub(r"\d", lambda match: f" {digits[int(match.group())]} ", phrase.lower().replace("ё", "е"))
    return " ".join(re.sub(r"[^a-zа-я' ]+", " ", spelled).split())


def validate_phrase(phrase: str, droid_name: str, language: str | None = None) -> str | None:
    """Return why ``phrase`` would make a poor wake phrase, or None when it is fine."""
    words = normalize_phrase(phrase, language).split()
    if not 2 <= len(words) <= 4:
        return "Use two to four words."
    if normalize_phrase(phrase, language) == normalize_phrase(droid_name, language):
        return "Add a word to the name so ordinary mentions do not wake me, for example 'hey' plus the name."
    if len(" ".join(words)) < 7:
        return "Use a longer phrase; very short phrases trigger by accident."
    return None


def ensure_vosk_model(models_dir: Path, language: str | None = None) -> Path:
    """Download and unpack the small Vosk model for the language once; returns its directory."""
    model_name = VOSK_MODELS[phrase_language(language)]
    model_dir = models_dir / model_name
    if (model_dir / "conf").is_dir():
        return model_dir
    models_dir.mkdir(parents=True, exist_ok=True)
    archive = models_dir / f"{model_name}.zip.part"
    url = f"{VOSK_MODEL_BASE_URL}/{model_name}.zip"
    logger.info("Downloading the wake phrase model from %s", url)
    with httpx.stream("GET", url, timeout=_DOWNLOAD_TIMEOUT_S, follow_redirects=True) as response:
        response.raise_for_status()
        with archive.open("wb") as file:
            for chunk in response.iter_bytes():
                file.write(chunk)
    with zipfile.ZipFile(archive) as zipped:
        zipped.extractall(models_dir)
    archive.unlink()
    return model_dir


class WakePhraseSpotter:
    """Listen for one phrase in 16 kHz mono audio; the model loads lazily on a background thread."""

    def __init__(self, models_dir: Path) -> None:
        """Remember where the model lives; call ``set_phrase`` to start listening."""
        self._models_dir = models_dir
        self._lock = threading.Lock()
        self._model: vosk.Model | None = None
        self._recognizer: vosk.KaldiRecognizer | None = None
        self._phrase = ""
        self._loading = False

    @property
    def phrase(self) -> str:
        """Return the normalised phrase being listened for."""
        return self._phrase

    @property
    def ready(self) -> bool:
        """Return whether audio is being matched against the phrase."""
        return self._recognizer is not None

    def set_phrase(self, phrase: str) -> None:
        """Listen for ``phrase`` from now on; an empty phrase stops listening."""
        with self._lock:
            self._phrase = normalize_phrase(phrase)
            self._recognizer = None
        if self._phrase:
            self._load_in_background()

    def unknown_words(self, phrase: str) -> list[str]:
        """Return words of ``phrase`` missing from the model vocabulary (only once the model is loaded)."""
        model = self._model
        if model is None:
            return []
        return [word for word in normalize_phrase(phrase).split() if model.vosk_model_find_word(word) < 0]

    def _load_in_background(self) -> None:
        if self._loading:
            return
        self._loading = True
        threading.Thread(target=self._load, daemon=True, name="droid-wake-phrase-load").start()

    def _load(self) -> None:
        try:
            if self._model is None:
                vosk.SetLogLevel(-1)
                self._model = vosk.Model(str(ensure_vosk_model(self._models_dir)))
            with self._lock:
                if self._phrase:
                    grammar = json.dumps([self._phrase, "[unk]"])
                    self._recognizer = vosk.KaldiRecognizer(self._model, LISTENER_SAMPLE_RATE, grammar)
                    logger.info("Listening for wake phrase %r", self._phrase)
        except (OSError, httpx.HTTPError, zipfile.BadZipFile) as e:
            logger.error("Wake phrase spotting unavailable: %s", e)
        finally:
            self._loading = False

    def feed(self, samples: NDArray[np.float32]) -> bool:
        """Consume audio and return True when the phrase was just heard."""
        with self._lock:
            recognizer = self._recognizer
            if recognizer is None or samples.size == 0:
                return False
            pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
            if not recognizer.AcceptWaveform(pcm):
                return False
            heard = json.loads(recognizer.Result()).get("text", "")
        return bool(heard) and heard == self._phrase
