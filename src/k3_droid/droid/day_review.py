"""Today's day review, written each weekday morning by a Claude routine to a private Hugging Face dataset.

The routine reads the owner's calendar and email and saves ``day_review.json``:

    {
      "date": "YYYY-MM-DD",
      "spoken_summary": "a few sentences to say out loud",
      "priorities": [{"rank": 1, "title": "...", "why": "...", "prepare": "..."}],
      "meetings": [{"start": "09:30", "end": "10:00", "title": "...", "importance": "high", "prepare": "..."}],
      "emails_to_answer": [{"from": "...", "subject": "...", "why": "..."}],
      "warnings": ["..."]
    }
"""

import json
import logging
from typing import Any
from datetime import date

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import HfHubHTTPError


logger = logging.getLogger(__name__)

DAY_REVIEW_FILENAME = "day_review.json"


def fetch_day_review(dataset: str, token: str | None, today: date) -> tuple[dict[str, Any] | None, str]:
    """Return today's review from the dataset (None when missing, stale or unreadable) and a status note."""
    try:
        path = hf_hub_download(dataset, DAY_REVIEW_FILENAME, repo_type="dataset", token=token or None)
        with open(path, encoding="utf-8") as f:
            review = json.load(f)
    except (HfHubHTTPError, OSError, ValueError) as e:
        logger.warning("Could not read the day review from %s: %s", dataset, e)
        return None, f"could not read {dataset}: {type(e).__name__}"
    if not isinstance(review, dict):
        return None, "the review file is not a JSON object"
    if review.get("date") != today.isoformat():
        return None, f"no review for today yet (latest is for {review.get('date') or 'an unknown day'})"
    return review, f"today's review loaded from {dataset}"
