"""
Memory domain ports (abstract interfaces).

What it is: Port definitions for the Memory bounded context.
What it knows: Domain entity shapes (Clip, DailySummary).
What it doesn't know: Any infrastructure detail — no SQLite, no Telegram, no B2.
"""
from abc import ABC, abstractmethod
from datetime import date
from pathlib import Path
from typing import List, Optional

from home_brain.intelligence.domain.entities.observation import DailySummary
from home_brain.surveillance.domain.entities.clip import Clip


class IClipRetentionPort(ABC):
    """Outbound port — query and manage Clips for retention and archival."""

    @abstractmethod
    def find_clips_by_ids(self, clip_ids: List[str]) -> List[Clip]:
        """Return Clip objects for the given IDs. Missing IDs are silently skipped."""

    @abstractmethod
    def find_archivable_clips(self, retention_days: int) -> List[Clip]:
        """Return analysed Clips recorded more than retention_days ago."""

    @abstractmethod
    def mark_archived(self, clip_id: str, archive_url: str) -> None:
        """Set clip status to 'archived' and store the archive URL."""


class ISummaryQueryPort(ABC):
    """Outbound port — query and update DailySummary for notification."""

    @abstractmethod
    def find_summary_by_date(self, target_date: date) -> Optional[DailySummary]:
        """Return the DailySummary for target_date, or None."""

    @abstractmethod
    def mark_summary_notified(self, summary_id: str) -> None:
        """Set notified=True for the given summary."""


class INotificationPort(ABC):
    """Outbound port — deliver the daily summary to the owner via Telegram."""

    @abstractmethod
    def send_daily_summary(
        self,
        summary: DailySummary,
        clip_paths: List[Path],
    ) -> None:
        """Send text summary + up to 3 highlight clip video files."""


class IArchivePort(ABC):
    """Outbound port — upload a Clip to cold storage and return its URL."""

    @abstractmethod
    def upload(self, clip: Clip) -> str:
        """Upload clip file to archive. Returns the archive URL string."""
