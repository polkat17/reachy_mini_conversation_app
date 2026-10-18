"""Game commentary (future phase): the interface a game source implements. Nothing implements it yet.

Two sources are planned: the droid's own camera watching the owner play, and a capture card or screen feed of the
console. Each turns frames into short text descriptions on its own schedule; the droid runtime would subscribe,
rate-limit, and voice occasional ``[SYSTEM EVENT]`` commentary, staying quiet while the owner is talking.
"""

from typing import Literal, Protocol
from dataclasses import field, dataclass
from collections.abc import AsyncIterator


GameFeedSource = Literal["room_camera", "console_capture"]


@dataclass(frozen=True)
class FrameDescription:
    """What one sampled frame shows, already reduced to text so no image is kept."""

    source: GameFeedSource
    captured_at: float
    summary: str
    events: list[str] = field(default_factory=list)
    excitement: float = 0.0  # 0 calm .. 1 dramatic; drives how eagerly the droid comments
    game: str = ""


class GameFeed(Protocol):
    """A source of periodic frame descriptions for game commentary."""

    source: GameFeedSource

    async def start(self) -> None:
        """Begin sampling (open the capture device, start the describer)."""
        ...

    async def stop(self) -> None:
        """Stop sampling and release the device."""
        ...

    def descriptions(self) -> AsyncIterator[FrameDescription]:
        """Yield descriptions as they are produced (typically every 2 to 10 seconds)."""
        ...
