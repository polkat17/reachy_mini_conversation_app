"""Pytest configuration for path setup."""

import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).parents[1].resolve()
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


# Make tests reproducible by ignoring machine-specific profile/tool env config.
# Without this, importing config during test collection can pick up a developer's
# local .env and fail before tests run.
os.environ["REACHY_MINI_SKIP_DOTENV"] = "1"
os.environ.pop("REACHY_MINI_CUSTOM_PROFILE", None)
os.environ.pop("REACHY_MINI_EXTERNAL_PROFILES_DIRECTORY", None)
os.environ.pop("REACHY_MINI_EXTERNAL_TOOLS_DIRECTORY", None)


import pytest  # noqa: E402


_LOCKED_PROFILE_MODULES = (
    "k3_droid.config",
    "k3_droid.console",
    "k3_droid.personality_routes",
    "k3_droid.profile_tool_routes",
    "k3_droid.tool_space_routes",
)


@pytest.fixture(autouse=True)
def _unlock_profile(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test upstream profile switching unlocked; K3 Droid ships locked to the droid (marked tests opt out)."""
    if request.node.get_closest_marker("k3_locked"):
        return
    import importlib

    for name in _LOCKED_PROFILE_MODULES:
        monkeypatch.setattr(importlib.import_module(name), "LOCKED_PROFILE", None, raising=False)
    config_mod = importlib.import_module("k3_droid.config")
    if config_mod.Config.REACHY_MINI_CUSTOM_PROFILE == "droid":
        monkeypatch.setattr(config_mod.Config, "REACHY_MINI_CUSTOM_PROFILE", None)
    if config_mod.config.REACHY_MINI_CUSTOM_PROFILE == "droid":
        monkeypatch.setattr(config_mod.config, "REACHY_MINI_CUSTOM_PROFILE", None)


def pytest_configure(config: pytest.Config) -> None:
    """Register the marker for tests that check the shipped droid lock."""
    config.addinivalue_line("markers", "k3_locked: run with K3 Droid's shipped LOCKED_PROFILE")
