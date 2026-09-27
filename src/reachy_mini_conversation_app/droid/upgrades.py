"""Self-development, part 1: the droid files upgrade requests as GitHub issues for a coding agent to implement.

The robot's token only needs permission to create issues. It never pushes code: a scheduled Claude Code session
turns labelled issues into pull requests, the owner merges them, and the updater installs merged code.
"""

import logging
from typing import Any
from datetime import datetime
from dataclasses import dataclass

import httpx

from reachy_mini_conversation_app.droid.identity import Identity
from reachy_mini_conversation_app.droid.settings import DroidSettings


logger = logging.getLogger(__name__)

UPGRADE_LABEL = "droid-upgrade"
_GITHUB_API = "https://api.github.com"
_TIMEOUT_S = 15.0


@dataclass(frozen=True)
class UpgradeRequest:
    """An open upgrade request."""

    number: int
    title: str
    url: str


class UpgradeRequestError(RuntimeError):
    """GitHub rejected or could not be reached for an upgrade request."""


def upgrades_enabled(settings: DroidSettings) -> bool:
    """Return whether a repository and token are configured."""
    return bool(settings.github_repo and settings.github_token)


def _headers(settings: DroidSettings) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {settings.github_token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def issue_body(identity: Identity, spec: str, reason: str) -> str:
    """Write the issue body the coding agent works from."""
    return (
        f"**Requested by** {identity.droid_name} on behalf of {identity.owner_name}, "
        f"{datetime.now():%Y-%m-%d %H:%M}.\n\n"
        f"## What the droid should be able to do\n{spec.strip()}\n\n"
        f"## Why\n{reason.strip() or 'Requested in conversation.'}\n\n"
        "## Implementation notes for the coding agent\n"
        "- Follow `AGENTS.md` and `DROID_ROADMAP.md`. Prefer a new tool in `src/reachy_mini_conversation_app/tools/` "
        "enabled in `profiles/droid/profile.md`, or a change inside `src/reachy_mini_conversation_app/droid/`.\n"
        "- Add tests, run the full gate, and open a pull request that references this issue. Never merge it.\n"
        "- Anything needing new secrets or hardware goes in the pull request description as a setup step.\n"
    )


async def list_open_requests(settings: DroidSettings) -> list[UpgradeRequest]:
    """Return open upgrade requests, newest first."""
    async with httpx.AsyncClient(timeout=_TIMEOUT_S) as client:
        response = await client.get(
            f"{_GITHUB_API}/repos/{settings.github_repo}/issues",
            params={"labels": UPGRADE_LABEL, "state": "open", "per_page": 30},
            headers=_headers(settings),
        )
    if response.status_code >= 400:
        raise UpgradeRequestError(f"GitHub returned {response.status_code} listing upgrade requests")
    items: list[dict[str, Any]] = response.json()
    return [UpgradeRequest(int(item["number"]), str(item["title"]), str(item["html_url"])) for item in items]


async def file_request(
    settings: DroidSettings, identity: Identity, title: str, spec: str, reason: str
) -> UpgradeRequest:
    """Open a labelled GitHub issue for the upgrade."""
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT_S) as client:
            response = await client.post(
                f"{_GITHUB_API}/repos/{settings.github_repo}/issues",
                json={
                    "title": f"[Droid upgrade] {title.strip()}",
                    "body": issue_body(identity, spec, reason),
                    "labels": [UPGRADE_LABEL],
                },
                headers=_headers(settings),
            )
    except httpx.HTTPError as e:
        raise UpgradeRequestError(f"could not reach GitHub: {e}") from e
    if response.status_code >= 400:
        raise UpgradeRequestError(f"GitHub returned {response.status_code}: {response.text[:200]}")
    item = response.json()
    return UpgradeRequest(int(item["number"]), str(item["title"]), str(item["html_url"]))
