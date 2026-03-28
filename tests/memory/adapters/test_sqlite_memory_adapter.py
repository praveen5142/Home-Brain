"""
Tests for SQLiteMemoryAdapter — IClipRetentionPort + ISummaryQueryPort.

Uses a temporary SQLite file per test. Negative tests first, then positive.
"""

import json
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

from home_brain.intelligence.domain.entities.observation import DailySummary, Mood
from home_brain.memory.adapters.sqlite_memory_adapter import SQLiteMemoryAdapter
from home_brain.surveillance.domain.entities.clip import ClipStatus


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def db_path(tmp_path) -> Path:
    return tmp_path / "test.sqlite"


@pytest.fixture
def adapter(db_path) -> SQLiteMemoryAdapter:
    return SQLiteMemoryAdapter(db_path)


def _insert_clip(db_path: Path, clip_id: str, status: str, days_old: int) -> None:
    """Insert a clip directly into clips table."""
    recorded_at = datetime.utcnow() - timedelta(days=days_old)
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS clips (
            id TEXT PRIMARY KEY,
            recorded_at TEXT NOT NULL,
            duration_s REAL NOT NULL,
            file_path TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            size_bytes INTEGER,
            archive_url TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)
    conn.execute(
        "INSERT INTO clips (id, recorded_at, duration_s, file_path, status) VALUES (?,?,?,?,?)",
        (clip_id, recorded_at.isoformat(), 30.0, f"/data/{clip_id}.mp4", status),
    )
    conn.commit()
    conn.close()


def _insert_summary(db_path: Path, target_date: date, notified: bool = False) -> str:
    """Insert a daily_summary and return its id."""
    import uuid
    summary_id = str(uuid.uuid4())
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS daily_summaries (
            id TEXT PRIMARY KEY,
            date TEXT NOT NULL UNIQUE,
            summary_text TEXT NOT NULL,
            mood_counts TEXT NOT NULL,
            highlight_clip_ids TEXT NOT NULL,
            notified INTEGER NOT NULL DEFAULT 0
        )
    """)
    conn.execute(
        "INSERT INTO daily_summaries (id, date, summary_text, mood_counts, highlight_clip_ids, notified) "
        "VALUES (?,?,?,?,?,?)",
        (summary_id, target_date.isoformat(), "A calm day.", json.dumps({"calm": 2}),
         json.dumps(["clip-1"]), int(notified)),
    )
    conn.commit()
    conn.close()
    return summary_id


# ---------------------------------------------------------------------------
# NEGATIVE TESTS
# ---------------------------------------------------------------------------


class TestSQLiteMemoryAdapterNegative:
    def test_find_clips_by_ids_empty_list_returns_empty(self, adapter):
        result = adapter.find_clips_by_ids([])
        assert result == []

    def test_find_clips_by_ids_nonexistent_ids_returns_empty(self, adapter, db_path):
        _insert_clip(db_path, "clip-real", "analysed", days_old=5)
        result = adapter.find_clips_by_ids(["ghost-1", "ghost-2"])
        assert result == []

    def test_find_archivable_clips_none_old_enough_returns_empty(self, adapter, db_path):
        _insert_clip(db_path, "clip-recent", "analysed", days_old=1)
        result = adapter.find_archivable_clips(retention_days=30)
        assert result == []

    def test_find_archivable_clips_ignores_pending_status(self, adapter, db_path):
        _insert_clip(db_path, "clip-pending", "pending", days_old=60)
        result = adapter.find_archivable_clips(retention_days=30)
        assert result == []

    def test_find_archivable_clips_ignores_already_archived(self, adapter, db_path):
        _insert_clip(db_path, "clip-archived", "archived", days_old=60)
        result = adapter.find_archivable_clips(retention_days=30)
        assert result == []

    def test_find_archivable_clips_ignores_failed_status(self, adapter, db_path):
        _insert_clip(db_path, "clip-failed", "failed", days_old=60)
        result = adapter.find_archivable_clips(retention_days=30)
        assert result == []

    def test_mark_archived_nonexistent_clip_no_error(self, adapter, db_path):
        _insert_clip(db_path, "clip-real", "analysed", days_old=5)
        # Should not raise
        adapter.mark_archived("ghost-id", "https://b2.example.com/archive.mp4")

    def test_find_summary_by_date_missing_returns_none(self, adapter):
        result = adapter.find_summary_by_date(date(2026, 1, 1))
        assert result is None

    def test_mark_summary_notified_nonexistent_id_no_error(self, adapter):
        # Should not raise
        adapter.mark_summary_notified("ghost-summary-id")


# ---------------------------------------------------------------------------
# POSITIVE TESTS
# ---------------------------------------------------------------------------


class TestSQLiteMemoryAdapterPositive:
    def test_schema_created_on_init(self, db_path):
        SQLiteMemoryAdapter(db_path)
        conn = sqlite3.connect(str(db_path))
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
        conn.close()
        assert "clips" in tables
        assert "daily_summaries" in tables

    def test_find_clips_by_ids_returns_matching(self, adapter, db_path):
        _insert_clip(db_path, "clip-1", "analysed", days_old=5)
        _insert_clip(db_path, "clip-2", "analysed", days_old=5)
        _insert_clip(db_path, "clip-3", "analysed", days_old=5)
        result = adapter.find_clips_by_ids(["clip-1", "clip-3"])
        ids = {c.id for c in result}
        assert ids == {"clip-1", "clip-3"}
        assert "clip-2" not in ids

    def test_find_clips_by_ids_correct_fields(self, adapter, db_path):
        _insert_clip(db_path, "clip-x", "analysed", days_old=3)
        clips = adapter.find_clips_by_ids(["clip-x"])
        assert len(clips) == 1
        c = clips[0]
        assert c.id == "clip-x"
        assert c.status == ClipStatus.ANALYSED
        assert c.file_path == Path("/data/clip-x.mp4")

    def test_find_archivable_clips_returns_old_analysed(self, adapter, db_path):
        _insert_clip(db_path, "clip-old", "analysed", days_old=60)
        result = adapter.find_archivable_clips(retention_days=30)
        assert len(result) == 1
        assert result[0].id == "clip-old"

    def test_find_archivable_clips_boundary_31_days_returned(self, adapter, db_path):
        _insert_clip(db_path, "clip-31", "analysed", days_old=31)
        result = adapter.find_archivable_clips(retention_days=30)
        assert any(c.id == "clip-31" for c in result)

    def test_find_archivable_clips_boundary_29_days_not_returned(self, adapter, db_path):
        _insert_clip(db_path, "clip-29", "analysed", days_old=29)
        result = adapter.find_archivable_clips(retention_days=30)
        assert not any(c.id == "clip-29" for c in result)

    def test_mark_archived_updates_status_and_url(self, adapter, db_path):
        _insert_clip(db_path, "clip-1", "analysed", days_old=60)
        url = "https://b2.example.com/clip.mp4"
        adapter.mark_archived("clip-1", url)
        conn = sqlite3.connect(str(db_path))
        row = conn.execute(
            "SELECT status, archive_url FROM clips WHERE id=?", ("clip-1",)
        ).fetchone()
        conn.close()
        assert row[0] == "archived"
        assert row[1] == url

    def test_find_summary_by_date_returns_correct_summary(self, adapter, db_path):
        sid = _insert_summary(db_path, date(2026, 3, 28), notified=False)
        result = adapter.find_summary_by_date(date(2026, 3, 28))
        assert result is not None
        assert result.id == sid
        assert result.date == date(2026, 3, 28)
        assert result.summary_text == "A calm day."
        assert result.mood_counts == {"calm": 2}
        assert result.notified is False

    def test_mark_summary_notified_sets_flag(self, adapter, db_path):
        sid = _insert_summary(db_path, date(2026, 3, 28), notified=False)
        adapter.mark_summary_notified(sid)
        conn = sqlite3.connect(str(db_path))
        row = conn.execute(
            "SELECT notified FROM daily_summaries WHERE id=?", (sid,)
        ).fetchone()
        conn.close()
        assert row[0] == 1

    def test_multiple_clips_only_old_ones_returned(self, adapter, db_path):
        _insert_clip(db_path, "clip-old", "analysed", days_old=60)
        _insert_clip(db_path, "clip-new", "analysed", days_old=5)
        result = adapter.find_archivable_clips(retention_days=30)
        ids = {c.id for c in result}
        assert "clip-old" in ids
        assert "clip-new" not in ids
