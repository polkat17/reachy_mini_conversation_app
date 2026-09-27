"""One-shot text completions for background jobs (session summaries, diary), via Hugging Face inference."""

import os
import json
import logging
from typing import Any

import httpx
from huggingface_hub import InferenceClient
from huggingface_hub.errors import HfHubHTTPError

from reachy_mini_conversation_app.config import config


logger = logging.getLogger(__name__)

DEFAULT_SUMMARY_MODEL = "Qwen/Qwen2.5-7B-Instruct"
_TIMEOUT_S = 60.0


def summary_model() -> str:
    """Return the model used for background text jobs."""
    return (os.getenv("DROID_SUMMARY_MODEL") or DEFAULT_SUMMARY_MODEL).strip()


def complete(system: str, user: str, max_tokens: int = 600) -> str | None:
    """Return the model's reply, or None when no model is reachable (callers fall back to plain text)."""
    try:
        client = InferenceClient(model=summary_model(), token=config.HF_TOKEN or None, timeout=_TIMEOUT_S)
        response = client.chat_completion(
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=max_tokens,
            temperature=0.3,
        )
    except (HfHubHTTPError, httpx.HTTPError, OSError, ValueError) as e:
        logger.warning("Background LLM call failed: %s", e)
        return None
    content = response.choices[0].message.content if response.choices else None
    return content.strip() if content else None


def parse_json_object(text: str | None) -> dict[str, Any] | None:
    """Extract the first JSON object from a model reply (models sometimes wrap it in prose or fences)."""
    if not text:
        return None
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        value = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None
