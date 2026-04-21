"""Base Connector protocol for Vektori integrations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Protocol
from datetime import datetime

if TYPE_CHECKING:
    from vektori.client import Vektori


class ConnectorResult:
    """Result object to represent the return of a connector sync."""

    def __init__(self, count: int, error: str | None = None) -> None:
        self.count = count
        self.error = error

    @property
    def success(self) -> bool:
        return self.error is None


class Connector(Protocol):
    """
    Protocol that every platform integration must implement.
    Platforms are sources of memory natively synced to Vektori.
    """

    @property
    def platform(self) -> str:
        """The identifier for this connector, e.g., 'gmail', 'github', 'notion'."""
        ...

    async def ingest(
        self,
        user_id: str,
        vektori: Vektori,
        since: datetime | None = None,
    ) -> int:
        """
        Pull data from the platform and store it into Vektori.
        Uses incremental sync if `since` is provided.

        Args:
            user_id: The owner of this data.
            vektori: The active Vektori client instance.
            since: Only fetch records created/modified after this date.

        Returns:
            The number of documents/records safely ingested.
        """
        ...

    async def watch(self, user_id: str, callback: Callable[[Any], None]) -> None:
        """
        Optional webhook installation to ingest data in real-time.
        """
        ...
