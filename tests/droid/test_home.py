import json
import dataclasses
from typing import Any

import httpx
import pytest

from reachy_mini_conversation_app.droid import home as home_mod
from reachy_mini_conversation_app.droid.home import Device, HomeError, HomeAssistant, match_device
from reachy_mini_conversation_app.droid.settings import DroidSettings


STATES = [
    {"entity_id": "light.desk_lamp", "state": "off", "attributes": {"friendly_name": "Desk lamp"}},
    {"entity_id": "switch.kettle", "state": "off", "attributes": {"friendly_name": "Kettle"}},
    {"entity_id": "lock.front_door", "state": "locked", "attributes": {"friendly_name": "Front door"}},
]


def _home(monkeypatch: pytest.MonkeyPatch, calls: list[httpx.Request]) -> HomeAssistant:
    real_client = httpx.AsyncClient

    def handle(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json=STATES if request.method == "GET" else [])

    def client_factory(**kwargs: Any) -> httpx.AsyncClient:
        return real_client(transport=httpx.MockTransport(handle), **kwargs)

    monkeypatch.setattr(home_mod.httpx, "AsyncClient", client_factory)
    settings = dataclasses.replace(
        DroidSettings.from_env(), home_assistant_url="http://ha.local:8123", home_assistant_token="abc"
    )
    return HomeAssistant(settings)


@pytest.mark.asyncio
async def test_locks_are_never_exposed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only allowed domains are listed: the front door lock is invisible to the droid."""
    devices = await _home(monkeypatch, []).devices()

    assert [device.name for device in devices] == ["Desk lamp", "Kettle"]


@pytest.mark.asyncio
async def test_call_posts_service(monkeypatch: pytest.MonkeyPatch) -> None:
    """Turning on a light posts to the light.turn_on service for that entity."""
    calls: list[httpx.Request] = []
    home = _home(monkeypatch, calls)

    await home.call(Device("light.desk_lamp", "Desk lamp", "off"), "turn_on")

    assert str(calls[0].url) == "http://ha.local:8123/api/services/light/turn_on"
    assert json.loads(calls[0].content) == {"entity_id": "light.desk_lamp"}
    assert calls[0].headers["Authorization"] == "Bearer abc"


@pytest.mark.asyncio
async def test_unsupported_action_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """Scenes cannot be turned off."""
    with pytest.raises(HomeError):
        await _home(monkeypatch, []).call(Device("scene.movie", "Movie", "scening"), "turn_off")


@pytest.mark.parametrize(
    ("said", "expected"), [("desk lamp", "Desk lamp"), ("lamp", "Desk lamp"), ("kettel", "Kettle")]
)
def test_match_device(said: str, expected: str) -> None:
    """Exact, partial and slightly misheard names find the right device."""
    devices = [Device("light.desk_lamp", "Desk lamp", "off"), Device("switch.kettle", "Kettle", "off")]

    matched = match_device(devices, said)

    assert matched is not None and matched.name == expected


def test_match_device_none() -> None:
    """Unrelated names match nothing."""
    assert match_device([Device("switch.kettle", "Kettle", "off")], "television") is None
