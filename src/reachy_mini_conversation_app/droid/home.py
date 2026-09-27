"""Smart home control through Home Assistant's REST API (a long-lived access token on the owner's network).

Only low-risk device types are exposed; locks, alarms and garage doors are deliberately left out because a voice
command (or a guest) must never unlock the home.
"""

import difflib
from typing import Any
from dataclasses import dataclass

import httpx

from reachy_mini_conversation_app.droid.settings import DroidSettings


ALLOWED_DOMAINS = ("light", "switch", "fan", "scene", "script", "media_player")
ACTIONS: dict[str, dict[str, str]] = {
    "turn_on": {"light": "turn_on", "switch": "turn_on", "fan": "turn_on", "scene": "turn_on", "script": "turn_on"},
    "turn_off": {"light": "turn_off", "switch": "turn_off", "fan": "turn_off", "media_player": "turn_off"},
    "toggle": {"light": "toggle", "switch": "toggle", "fan": "toggle", "media_player": "media_play_pause"},
}
_TIMEOUT_S = 10.0


@dataclass(frozen=True)
class Device:
    """A controllable Home Assistant entity."""

    entity_id: str
    name: str
    state: str

    @property
    def domain(self) -> str:
        """Return the entity's domain, e.g. ``light``."""
        return self.entity_id.split(".", 1)[0]


class HomeError(RuntimeError):
    """Home Assistant could not be reached or refused the request."""


class HomeAssistant:
    """A minimal Home Assistant client for the allowed device types."""

    def __init__(self, settings: DroidSettings) -> None:
        """Use the configured URL and token."""
        self._base = settings.home_assistant_url
        self._headers = {"Authorization": f"Bearer {settings.home_assistant_token}"}

    async def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> Any:
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT_S) as client:
                response = await client.request(method, f"{self._base}{path}", json=payload, headers=self._headers)
        except httpx.HTTPError as e:
            raise HomeError(f"could not reach Home Assistant: {e}") from e
        if response.status_code >= 400:
            raise HomeError(f"Home Assistant returned {response.status_code}")
        return response.json()

    async def devices(self) -> list[Device]:
        """Return controllable devices."""
        states = await self._request("GET", "/api/states")
        return [
            Device(
                item["entity_id"], item.get("attributes", {}).get("friendly_name") or item["entity_id"], item["state"]
            )
            for item in states
            if isinstance(item, dict) and str(item.get("entity_id", "")).split(".", 1)[0] in ALLOWED_DOMAINS
        ]

    async def call(self, device: Device, action: str) -> None:
        """Run ``action`` (turn_on, turn_off, toggle) on ``device``."""
        service = ACTIONS[action].get(device.domain)
        if service is None:
            raise HomeError(f"{device.name} cannot {action.replace('_', ' ')}")
        await self._request("POST", f"/api/services/{device.domain}/{service}", {"entity_id": device.entity_id})


def match_device(devices: list[Device], name: str) -> Device | None:
    """Find the device whose friendly name best matches what the owner said."""
    wanted = name.strip().lower()
    by_name = {device.name.lower(): device for device in devices}
    if wanted in by_name:
        return by_name[wanted]
    contained = [device for device in devices if wanted and wanted in device.name.lower()]
    if len(contained) == 1:
        return contained[0]
    close = difflib.get_close_matches(wanted, list(by_name), n=1, cutoff=0.6)
    return by_name[close[0]] if close else None
