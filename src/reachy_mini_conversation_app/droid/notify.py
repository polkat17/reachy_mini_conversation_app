"""Phone notifications through ntfy (https://ntfy.sh): the owner subscribes to a private topic in the ntfy app."""

import logging

import httpx

from reachy_mini_conversation_app.droid.settings import DroidSettings


logger = logging.getLogger(__name__)

_NTFY_TIMEOUT_S = 10.0


def notifications_enabled(settings: DroidSettings) -> bool:
    """Return whether a ntfy topic is configured."""
    return bool(settings.ntfy_topic)


async def send_notification(
    settings: DroidSettings,
    message: str,
    *,
    title: str = "",
    priority: str = "default",
    tags: str = "robot",
) -> bool:
    """Push ``message`` to the owner's phone; returns whether ntfy accepted it."""
    if not notifications_enabled(settings):
        logger.info("Notification skipped (DROID_NTFY_TOPIC unset): %s", message[:120])
        return False
    headers = {"Priority": priority, "Tags": tags}
    if title:
        # HTTP headers are ASCII-only; the message body carries any other characters.
        headers["Title"] = title.encode("ascii", "ignore").decode("ascii")
    try:
        async with httpx.AsyncClient(timeout=_NTFY_TIMEOUT_S) as client:
            response = await client.post(
                f"{settings.ntfy_server}/{settings.ntfy_topic}",
                content=message.encode("utf-8"),
                headers=headers,
            )
            response.raise_for_status()
    except httpx.HTTPError as e:
        logger.warning("Failed to send notification: %s", e)
        return False
    return True
