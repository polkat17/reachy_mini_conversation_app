import json
import dataclasses
from typing import Any

import httpx
import pytest

from reachy_mini_conversation_app.droid import upgrades as upgrades_mod
from reachy_mini_conversation_app.droid.identity import Identity
from reachy_mini_conversation_app.droid.settings import DroidSettings


def _settings() -> DroidSettings:
    return dataclasses.replace(DroidSettings.from_env(), github_repo="owner/fork", github_token="t0ken")


def _patch_client(monkeypatch: pytest.MonkeyPatch, handler: Any) -> None:
    real_client = httpx.AsyncClient

    def client_factory(**kwargs: Any) -> httpx.AsyncClient:
        return real_client(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(upgrades_mod.httpx, "AsyncClient", client_factory)


def test_issue_body_carries_spec_reason_and_agent_notes() -> None:
    """The coding agent gets the request, the reason, and the house rules."""
    body = upgrades_mod.issue_body(Identity(), "Announce train departures.", "Pasha keeps missing trains.")

    assert "Announce train departures." in body and "missing trains" in body
    assert "AGENTS.md" in body and "Never merge" in body


@pytest.mark.asyncio
async def test_file_request_opens_labelled_issue(monkeypatch: pytest.MonkeyPatch) -> None:
    """Filing posts a labelled issue with the bearer token."""
    seen: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(201, json={"number": 7, "title": "[Droid upgrade] Trains", "html_url": "https://x/7"})

    _patch_client(monkeypatch, handle)

    request = await upgrades_mod.file_request(_settings(), Identity(), "Trains", "spec", "reason")

    payload = json.loads(seen[0].content)
    assert request.number == 7 and str(seen[0].url) == "https://api.github.com/repos/owner/fork/issues"
    assert payload["labels"] == ["droid-upgrade"] and seen[0].headers["Authorization"] == "Bearer t0ken"


@pytest.mark.asyncio
async def test_file_request_surfaces_github_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """A rejected token becomes an error the droid can relay."""
    _patch_client(monkeypatch, lambda _request: httpx.Response(401, text="Bad credentials"))

    with pytest.raises(upgrades_mod.UpgradeRequestError, match="401"):
        await upgrades_mod.file_request(_settings(), Identity(), "Trains", "spec", "")
