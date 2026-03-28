"""
Tests for MemoryService — deliver_daily_summary + archive_old_clips.

Uses in-memory fakes for all ports. Negative tests first, then positive.
"""

from datetime import date, datetime
from pathlib import Path
from typing import List, Optional
from unittest.mock import MagicMock

import pytest

from home_brain.intelligence.domain.entities.observation import DailySummary
from home_brain.memory.domain.ports.ports import (
    IArchivePort,
    IClipRetentionPort,
    INotificationPort,
    ISummaryQueryPort,
)
from home_brain.memory.domain.services.memory_service import MemoryService
from home_brain.surveillance.domain.entities.clip import Clip, ClipStatus


# ---------------------------------------------------------------------------
# In-memory fakes
# ---------------------------------------------------------------------------


class FakeClipRetention(IClipRetentionPort):
    def __init__(self, clips: List[Clip] = None, archivable: List[Clip] = None):
        self._clips = {c.id: c for c in (clips or [])}
        self._archivable = archivable or []
        self.archived: List[tuple] = []  # (clip_id, archive_url)

    def find_clips_by_ids(self, clip_ids: List[str]) -> List[Clip]:
        return [self._clips[cid] for cid in clip_ids if cid in self._clips]

    def find_archivable_clips(self, retention_days: int) -> List[Clip]:
        return list(self._archivable)

    def mark_archived(self, clip_id: str, archive_url: str) -> None:
        self.archived.append((clip_id, archive_url))


class FakeSummaryQuery(ISummaryQueryPort):
    def __init__(self, summary: Optional[DailySummary] = None):
        self._summary = summary
        self.notified_ids: List[str] = []

    def find_summary_by_date(self, target_date: date) -> Optional[DailySummary]:
        if self._summary and self._summary.date == target_date:
            return self._summary
        return None

    def mark_summary_notified(self, summary_id: str) -> None:
        self.notified_ids.append(summary_id)


class FakeNotification(INotificationPort):
    def __init__(self, should_raise: bool = False):
        self._should_raise = should_raise
        self.calls: List[tuple] = []  # (summary, clip_paths)

    def send_daily_summary(self, summary, clip_paths):
        if self._should_raise:
            raise RuntimeError("Telegram down")
        self.calls.append((summary, list(clip_paths)))


class FakeArchive(IArchivePort):
    def __init__(self, failing_ids: List[str] = None):
        self._failing_ids = set(failing_ids or [])
        self.uploaded: List[str] = []

    def upload(self, clip: Clip) -> str:
        if clip.id in self._failing_ids:
            raise RuntimeError(f"Upload failed for {clip.id}")
        self.uploaded.append(clip.id)
        return f"https://b2.example.com/{clip.id}.mp4"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_summary(notified: bool = False, highlight_ids: List[str] = None) -> DailySummary:
    return DailySummary(
        id="sum-1",
        date=date(2026, 3, 28),
        summary_text="A calm day.",
        mood_counts={"calm": 2},
        highlight_clip_ids=highlight_ids or [],
        notified=notified,
    )


def _make_clip(clip_id: str, file_path: Path = None) -> Clip:
    return Clip(
        id=clip_id,
        recorded_at=datetime(2026, 3, 28, 10, 0, 0),
        duration_seconds=30.0,
        file_path=file_path or Path(f"/data/{clip_id}.mp4"),
        status=ClipStatus.ANALYSED,
    )


def _make_service(
    clips=None,
    archivable=None,
    summary=None,
    notification=None,
    archive=None,
) -> tuple:
    clip_retention = FakeClipRetention(clips=clips, archivable=archivable)
    summary_query = FakeSummaryQuery(summary=summary)
    notif = notification or FakeNotification()
    arch = archive or FakeArchive()
    service = MemoryService(clip_retention, summary_query, notif, arch)
    return service, clip_retention, summary_query, notif, arch


# ---------------------------------------------------------------------------
# deliver_daily_summary — NEGATIVE TESTS
# ---------------------------------------------------------------------------


class TestDeliverDailySummaryNegative:
    def test_no_summary_for_date_raises_value_error(self):
        service, *_ = _make_service(summary=None)
        with pytest.raises(ValueError, match="No summary"):
            service.deliver_daily_summary(date(2026, 3, 28))

    def test_already_notified_skips_notification(self):
        summary = _make_summary(notified=True)
        notif = FakeNotification()
        service, _, _, notif, _ = _make_service(summary=summary, notification=notif)
        service.deliver_daily_summary(date(2026, 3, 28))
        assert notif.calls == []

    def test_notification_failure_propagates_and_does_not_mark_notified(self):
        summary = _make_summary(notified=False)
        notif = FakeNotification(should_raise=True)
        service, _, summary_query, _, _ = _make_service(
            summary=summary, notification=notif
        )
        with pytest.raises(RuntimeError, match="Telegram down"):
            service.deliver_daily_summary(date(2026, 3, 28))
        assert summary_query.notified_ids == []

    def test_all_highlight_clips_missing_sends_text_only(self, tmp_path):
        summary = _make_summary(highlight_ids=["clip-1", "clip-2"])
        clip1 = _make_clip("clip-1", tmp_path / "ghost1.mp4")  # does NOT exist on disk
        clip2 = _make_clip("clip-2", tmp_path / "ghost2.mp4")  # does NOT exist on disk
        notif = FakeNotification()
        service, _, _, notif, _ = _make_service(
            clips=[clip1, clip2], summary=summary, notification=notif
        )
        service.deliver_daily_summary(date(2026, 3, 28))
        assert len(notif.calls) == 1
        _, paths = notif.calls[0]
        assert paths == []


# ---------------------------------------------------------------------------
# deliver_daily_summary — POSITIVE TESTS
# ---------------------------------------------------------------------------


class TestDeliverDailySummaryPositive:
    def test_happy_path_sends_notification_and_marks_notified(self, tmp_path):
        clip_file = tmp_path / "clip-1.mp4"
        clip_file.write_bytes(b"x")
        summary = _make_summary(highlight_ids=["clip-1"])
        clip1 = _make_clip("clip-1", clip_file)
        notif = FakeNotification()
        service, _, summary_query, notif, _ = _make_service(
            clips=[clip1], summary=summary, notification=notif
        )
        service.deliver_daily_summary(date(2026, 3, 28))
        assert len(notif.calls) == 1
        assert "sum-1" in summary_query.notified_ids

    def test_mixed_present_and_missing_clips_passes_only_existing(self, tmp_path):
        present = tmp_path / "clip-1.mp4"
        present.write_bytes(b"x")
        summary = _make_summary(highlight_ids=["clip-1", "clip-2"])
        clip1 = _make_clip("clip-1", present)
        clip2 = _make_clip("clip-2", tmp_path / "ghost.mp4")  # missing
        notif = FakeNotification()
        service, _, _, notif, _ = _make_service(
            clips=[clip1, clip2], summary=summary, notification=notif
        )
        service.deliver_daily_summary(date(2026, 3, 28))
        _, paths = notif.calls[0]
        assert paths == [present]

    def test_correct_call_order(self, tmp_path):
        """find_summary → find_clips → notify → mark_notified"""
        clip_file = tmp_path / "clip-1.mp4"
        clip_file.write_bytes(b"x")
        summary = _make_summary(highlight_ids=["clip-1"])
        clip1 = _make_clip("clip-1", clip_file)

        call_log = []

        class LoggingClipRetention(FakeClipRetention):
            def find_clips_by_ids(self, clip_ids):
                call_log.append("find_clips")
                return super().find_clips_by_ids(clip_ids)

        class LoggingNotification(FakeNotification):
            def send_daily_summary(self, summary, clip_paths):
                call_log.append("notify")
                super().send_daily_summary(summary, clip_paths)

        class LoggingSummaryQuery(FakeSummaryQuery):
            def find_summary_by_date(self, target_date):
                call_log.append("find_summary")
                return super().find_summary_by_date(target_date)

            def mark_summary_notified(self, summary_id):
                call_log.append("mark_notified")
                super().mark_summary_notified(summary_id)

        clip_retention = LoggingClipRetention(clips=[clip1])
        summary_query = LoggingSummaryQuery(summary=summary)
        service = MemoryService(clip_retention, summary_query, LoggingNotification(), FakeArchive())
        service.deliver_daily_summary(date(2026, 3, 28))

        assert call_log == ["find_summary", "find_clips", "notify", "mark_notified"]


# ---------------------------------------------------------------------------
# archive_old_clips — NEGATIVE TESTS
# ---------------------------------------------------------------------------


class TestArchiveOldClipsNegative:
    def test_upload_failure_skips_clip_continues_others(self, tmp_path):
        file1 = tmp_path / "clip-1.mp4"
        file1.write_bytes(b"x")
        file2 = tmp_path / "clip-2.mp4"
        file2.write_bytes(b"x")
        clip1 = _make_clip("clip-1", file1)
        clip2 = _make_clip("clip-2", file2)
        arch = FakeArchive(failing_ids=["clip-1"])
        service, clip_retention, *_ = _make_service(
            archivable=[clip1, clip2], archive=arch
        )
        result = service.archive_old_clips(retention_days=30)
        assert len(result) == 1
        assert result[0].id == "clip-2"
        assert ("clip-2", "https://b2.example.com/clip-2.mp4") in clip_retention.archived

    def test_no_archivable_clips_returns_empty(self):
        service, *_ = _make_service(archivable=[])
        result = service.archive_old_clips(retention_days=30)
        assert result == []


# ---------------------------------------------------------------------------
# archive_old_clips — POSITIVE TESTS
# ---------------------------------------------------------------------------


class TestArchiveOldClipsPositive:
    def test_happy_path_upload_mark_delete(self, tmp_path):
        clip_file = tmp_path / "clip-1.mp4"
        clip_file.write_bytes(b"x")
        clip1 = _make_clip("clip-1", clip_file)
        arch = FakeArchive()
        service, clip_retention, *_ = _make_service(
            archivable=[clip1], archive=arch
        )
        result = service.archive_old_clips(retention_days=30)
        assert len(result) == 1
        assert "clip-1" in arch.uploaded
        assert ("clip-1", "https://b2.example.com/clip-1.mp4") in clip_retention.archived
        assert not clip_file.exists()  # local file deleted

    def test_two_clips_both_archived(self, tmp_path):
        file1 = tmp_path / "clip-1.mp4"
        file1.write_bytes(b"x")
        file2 = tmp_path / "clip-2.mp4"
        file2.write_bytes(b"x")
        clip1 = _make_clip("clip-1", file1)
        clip2 = _make_clip("clip-2", file2)
        arch = FakeArchive()
        service, *_ = _make_service(archivable=[clip1, clip2], archive=arch)
        result = service.archive_old_clips(retention_days=30)
        assert len(result) == 2
        assert set(arch.uploaded) == {"clip-1", "clip-2"}

    def test_local_file_deleted_after_archive(self, tmp_path):
        clip_file = tmp_path / "clip-1.mp4"
        clip_file.write_bytes(b"x")
        clip1 = _make_clip("clip-1", clip_file)
        service, *_ = _make_service(archivable=[clip1])
        service.archive_old_clips(retention_days=30)
        assert not clip_file.exists()

    def test_file_not_found_error_from_archive_skips_clip(self, tmp_path):
        """If archive raises FileNotFoundError, the clip is skipped (already gone)."""
        clip_file = tmp_path / "clip-1.mp4"
        clip_file.write_bytes(b"x")
        clip1 = _make_clip("clip-1", clip_file)

        class FileNotFoundArchive(IArchivePort):
            def upload(self, clip):
                raise FileNotFoundError("file gone")

        service, clip_retention, *_ = _make_service(
            archivable=[clip1], archive=FileNotFoundArchive()
        )
        result = service.archive_old_clips(retention_days=30)
        assert result == []
        assert clip_retention.archived == []
