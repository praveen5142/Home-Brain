"""
Memory Domain — MemoryService.

What it is: Application service orchestrating daily summary delivery to Telegram
            and archival of old clips to Backblaze B2.
What it knows: Port interfaces (IClipRetentionPort, ISummaryQueryPort,
               INotificationPort, IArchivePort) and domain entities (DailySummary, Clip).
What it doesn't know: SQLite, Telegram, boto3, ffmpeg — any infrastructure detail.

Two operations:
  deliver_daily_summary(date) — send summary + highlights to Telegram, idempotent.
  archive_old_clips(retention_days) — upload old clips to B2, delete local files.
"""

from datetime import date
from typing import List

from home_brain.memory.domain.ports.ports import (
    IArchivePort,
    IClipRetentionPort,
    INotificationPort,
    ISummaryQueryPort,
)
from home_brain.shared.logger import get_logger
from home_brain.surveillance.domain.entities.clip import Clip

logger = get_logger("memory.services.memory_service")


class MemoryService:
    def __init__(
        self,
        clip_retention: IClipRetentionPort,
        summary_query: ISummaryQueryPort,
        notification: INotificationPort,
        archive: IArchivePort,
    ) -> None:
        self._clip_retention = clip_retention
        self._summary_query = summary_query
        self._notification = notification
        self._archive = archive

    def deliver_daily_summary(self, target_date: date) -> None:
        """
        Send the DailySummary for target_date to Telegram with highlight clips.

        Idempotent: if already notified, logs and returns without sending again.
        Notification failure is fatal — summary is NOT marked notified.
        """
        summary = self._summary_query.find_summary_by_date(target_date)
        if summary is None:
            raise ValueError(f"No summary found for date {target_date}")

        if summary.notified:
            logger.info(f"Summary for {target_date} already notified — skipping")
            return

        clips = self._clip_retention.find_clips_by_ids(summary.highlight_clip_ids)
        existing_paths = [c.file_path for c in clips if c.file_path.exists()]

        self._notification.send_daily_summary(summary, existing_paths)
        self._summary_query.mark_summary_notified(summary.id)
        logger.info(f"Daily summary for {target_date} delivered and marked notified")

    def archive_old_clips(self, retention_days: int) -> List[Clip]:
        """
        Upload all archivable clips to B2, delete local files, mark as archived.

        Best-effort: on any exception per clip, logs a warning and continues.
        Returns the list of successfully archived clips.
        """
        clips = self._clip_retention.find_archivable_clips(retention_days)
        if not clips:
            logger.info("No clips to archive")
            return []

        archived = []
        for clip in clips:
            try:
                archive_url = self._archive.upload(clip)
                self._clip_retention.mark_archived(clip.id, archive_url)
                clip.file_path.unlink(missing_ok=True)
                archived.append(clip)
                logger.info(f"Archived and deleted clip {clip.id}")
            except Exception as exc:
                logger.warning(f"Failed to archive clip {clip.id}: {exc} — skipping")

        logger.info(f"Archival complete: {len(archived)}/{len(clips)} clips archived")
        return archived
