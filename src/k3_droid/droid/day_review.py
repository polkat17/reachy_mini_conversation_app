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


def fetch_day_review(dataset: str, token: str | None, today: date) -> dict[str, Any] | None:
    """Return today's review from the dataset, or None when it is missing, stale or unreadable."""
    try:
        path = hf_hub_download(dataset, DAY_REVIEW_FILENAME, repo_type="dataset", token=token or None)
        with open(path, encoding="utf-8") as f:
            review = json.load(f)
    except (HfHubHTTPError, OSError, ValueError) as e:
        logger.warning("Could not read the day review from %s: %s", dataset, e)
        return None
    if not isinstance(review, dict) or review.get("date") != today.isoformat():
        return None
    return review


def briefing_line(review: dict[str, Any]) -> str:
    """Condense a review into one prompt line for the spoken morning briefing."""
    summary = str(review.get("spoken_summary") or "").strip()
    priorities = [
        str(item.get("title") or "").strip() for item in review.get("priorities") or [] if isinstance(item, dict)
    ]
    top = "; ".join(title for title in priorities[:3] if title)
    return summary + (f" Top priorities: {top}." if top else "")
