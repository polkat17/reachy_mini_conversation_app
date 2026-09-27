import logging
from typing import TYPE_CHECKING, Any
from datetime import datetime, timedelta

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


if TYPE_CHECKING:
    from reachy_mini_conversation_app.droid.runtime import DroidRuntime


logger = logging.getLogger(__name__)


def due_time(in_minutes: object, at: object, now: datetime) -> datetime | None:
    """Resolve 'in N minutes' or 'at HH:MM' / 'YYYY-MM-DD HH:MM' (a past HH:MM means tomorrow)."""
    if in_minutes not in (None, ""):
        try:
            return now + timedelta(minutes=float(str(in_minutes)))
        except ValueError:
            return None
    text = str(at or "").strip()
    for layout in ("%Y-%m-%d %H:%M", "%H:%M"):
        try:
            parsed = datetime.strptime(text, layout)
        except ValueError:
            continue
        if layout == "%H:%M":
            parsed = now.replace(hour=parsed.hour, minute=parsed.minute, second=0, microsecond=0)
            if parsed <= now:
                parsed += timedelta(days=1)
        return parsed
    return None


class Reminder(Tool):
    """Set, list or cancel reminders and timers."""

    name = "reminder"
    description = (
        "Manage reminders and timers. 'set' needs text plus either in_minutes (timers) or at ('18:30' or "
        "'2026-10-02 09:00'). When due you announce it (waking if dormant) and it goes to the owner's phone if "
        "they are not in view."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["set", "list", "cancel"]},
            "text": {"type": "string"},
            "in_minutes": {"type": "number"},
            "at": {"type": "string"},
            "id": {"type": "integer", "description": "Reminder id for 'cancel'."},
        },
        "required": ["action"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> dict[str, Any]:
        """Run the reminder action."""
        droid: "DroidRuntime | None" = deps.droid
        if droid is None:
            return {"error": "reminders are only available in the droid profile"}
        action = kwargs.get("action")
        logger.info("Tool call: reminder action=%s", action)
        if action == "list":
            return {
                "reminders": [
                    {"id": r.id, "due": f"{datetime.fromtimestamp(r.due_at):%a %d %b %H:%M}", "text": r.text}
                    for r in droid.memory.pending_reminders()
                ]
            }
        if action == "cancel":
            cancelled = droid.memory.cancel_reminder(int(kwargs.get("id") or 0))
            return {"status": "cancelled"} if cancelled else {"error": "no pending reminder with that id"}
        text = str(kwargs.get("text") or "").strip()
        due = due_time(kwargs.get("in_minutes"), kwargs.get("at"), datetime.now())
        if not text or due is None:
            return {"error": "'set' needs text and either in_minutes or at (HH:MM or YYYY-MM-DD HH:MM)"}
        reminder = droid.memory.add_reminder(due.timestamp(), text)
        return {"status": "set", "id": reminder.id, "due": f"{due:%a %d %b %H:%M}"}
