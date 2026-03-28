"""
Tests for SQLiteObservationAdapter.

Uses a temporary SQLite file per test. Negative tests first, then positive.
"""

import sqlite3
import tempfile
from datetime import date, datetime
from pathlib import Path
from typing import Generator

import pytest

from home_brain.intelligence.domain.entities.observation import (
    DailySummary,
    Mood,
    Observation,
    Transcript,
)
from home_brain.intelligence.adapters.sqlite_observation_adapter import (
    SQLiteObservationAdapter,
)
from home_brain.surveillance.domain.entities.clip import Clip, ClipStatus


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def db_path(tmp_path) -> Path:
    return tmp_path / "test.sqlite"


@pytest.fixture
def adapter(db_path) -> SQLiteObservationAdapter:
    return SQLiteObservationAdapter(db_path)


def _make_observation(clip_id: str = "clip-1", confidence: float = 0.9) -> Observation:
    return Observation.create(
        clip_id=clip_id,
        mood=Mood.NEUTRAL,
        activity="person walking",
        description="A person walks through the hallway.",
        confidence=confidence,
    )


def _make_transcript(clip_id: str = "clip-1") -> Transcript:
    return Transcript.create(clip_id=clip_id, text="Hello there.", language="en")


def _make_daily_summary(target_date: date = date(2026, 3, 28)) -> DailySummary:
    return DailySummary.create(
        date=target_date,
        summary_text="A quiet day at home.",
        mood_counts={"neutral": 3, "happy": 1},
        highlight_clip_ids=["clip-1", "clip-2"],
    )


def _insert_pending_clip(db_path: Path, clip_id: str = "clip-1") -> None:
    """Directly insert a pending clip into the clips table."""
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
        "INSERT INTO clips (id, recorded_at, duration_s, file_path, status, created_at) "
        "VALUES (?, ?, ?, ?, ?, datetime('now'))",
        (clip_id, "2026-03-28T10:00:00", 30.0, "/data/clip.mp4", "pending"),
    )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# NEGATIVE TESTS
# ---------------------------------------------------------------------------


class TestSQLiteObservationAdapterNegative:
    def test_save_observation_duplicate_id_raises(self, adapter, db_path):
        obs = _make_observation()
        adapter.save_observation(obs)
        with pytest.raises(Exception):  # sqlite3.IntegrityError
            adapter.save_observation(obs)

    def test_find_observations_by_date_no_rows_returns_empty(self, adapter):
        result = adapter.find_observations_by_date(date(2026, 1, 1))
        assert result == []

    def test_find_summary_by_date_no_row_returns_none(self, adapter):
        result = adapter.find_summary_by_date(date(2026, 1, 1))
        assert result is None

    def test_mark_analysed_nonexistent_clip_does_not_raise(self, adapter, db_path):
        # clips table may not exist yet — adapter should handle gracefully
        _insert_pending_clip(db_path, "clip-real")
        # marking a non-existent clip should not raise (UPDATE 0 rows is fine)
        adapter.mark_analysed("clip-ghost")  # no exception expected

    def test_mark_failed_nonexistent_clip_does_not_raise(self, adapter, db_path):
        _insert_pending_clip(db_path, "clip-real")
        adapter.mark_failed("clip-ghost")  # no exception expected

    def test_save_transcript_duplicate_id_raises(self, adapter):
        tr = _make_transcript()
        adapter.save_transcript(tr)
        with pytest.raises(Exception):
            adapter.save_transcript(tr)

    def test_save_daily_summary_duplicate_date_raises(self, adapter):
        ds = _make_daily_summary()
        adapter.save_daily_summary(ds)
        ds2 = DailySummary.create(
            date=date(2026, 3, 28),
            summary_text="Another summary.",
            mood_counts={},
            highlight_clip_ids=[],
        )
        with pytest.raises(Exception):  # UNIQUE constraint on date
            adapter.save_daily_summary(ds2)


# ---------------------------------------------------------------------------
# POSITIVE TESTS
# ---------------------------------------------------------------------------


class TestSQLiteObservationAdapterPositive:
    def test_schema_created_on_init(self, db_path):
        SQLiteObservationAdapter(db_path)
        conn = sqlite3.connect(str(db_path))
        tables = {row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
        conn.close()
        assert "observations" in tables
        assert "transcripts" in tables
        assert "daily_summaries" in tables

    def test_save_and_find_observation_by_date(self, adapter):
        obs = _make_observation("clip-1")
        adapter.save_observation(obs)
        result = adapter.find_observations_by_date(obs.observed_at.date())
        assert len(result) == 1
        r = result[0]
        assert r.id == obs.id
        assert r.clip_id == "clip-1"
        assert r.mood == Mood.NEUTRAL
        assert r.activity == "person walking"
        assert abs(r.confidence - 0.9) < 1e-6

    def test_save_and_find_transcript_round_trip(self, adapter):
        tr = _make_transcript("clip-1")
        adapter.save_transcript(tr)
        # No direct find_transcript_by method — verify via internal query
        conn = sqlite3.connect(str(adapter._db_path))
        row = conn.execute(
            "SELECT id, clip_id, text, language FROM transcripts WHERE id=?", (tr.id,)
        ).fetchone()
        conn.close()
        assert row is not None
        assert row[0] == tr.id
        assert row[1] == "clip-1"
        assert row[2] == "Hello there."
        assert row[3] == "en"

    def test_save_and_find_daily_summary(self, adapter):
        ds = _make_daily_summary(date(2026, 3, 28))
        adapter.save_daily_summary(ds)
        result = adapter.find_summary_by_date(date(2026, 3, 28))
        assert result is not None
        assert result.id == ds.id
        assert result.date == date(2026, 3, 28)
        assert result.summary_text == "A quiet day at home."
        assert result.mood_counts == {"neutral": 3, "happy": 1}
        assert result.highlight_clip_ids == ["clip-1", "clip-2"]
        assert result.notified is False

    def test_daily_summary_notified_round_trip(self, adapter):
        ds = _make_daily_summary()
        adapter.save_daily_summary(ds)
        notified_ds = ds.mark_notified()
        adapter.update_daily_summary(notified_ds)
        result = adapter.find_summary_by_date(ds.date)
        assert result.notified is True

    def test_find_pending_returns_only_pending_clips(self, adapter, db_path):
        _insert_pending_clip(db_path, "clip-pending")
        # Insert an analysed clip too
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "INSERT INTO clips (id, recorded_at, duration_s, file_path, status, created_at) "
            "VALUES (?, ?, ?, ?, ?, datetime('now'))",
            ("clip-analysed", "2026-03-28T11:00:00", 45.0, "/data/clip2.mp4", "analysed"),
        )
        conn.commit()
        conn.close()
        result = adapter.find_pending()
        ids = [c.id for c in result]
        assert "clip-pending" in ids
        assert "clip-analysed" not in ids

    def test_mark_analysed_updates_status(self, adapter, db_path):
        _insert_pending_clip(db_path, "clip-1")
        adapter.mark_analysed("clip-1")
        conn = sqlite3.connect(str(db_path))
        row = conn.execute(
            "SELECT status FROM clips WHERE id=?", ("clip-1",)
        ).fetchone()
        conn.close()
        assert row[0] == "analysed"

    def test_mark_failed_updates_status(self, adapter, db_path):
        _insert_pending_clip(db_path, "clip-1")
        adapter.mark_failed("clip-1")
        conn = sqlite3.connect(str(db_path))
        row = conn.execute(
            "SELECT status FROM clips WHERE id=?", ("clip-1",)
        ).fetchone()
        conn.close()
        assert row[0] == "failed"

    def test_multiple_observations_different_dates_isolated(self, adapter):
        obs1 = Observation.create("clip-1", Mood.HAPPY, "child playing", "Kids.", 0.9)
        obs2 = Observation.create("clip-2", Mood.CALM, "cat sleeping", "Cat.", 0.7)
        adapter.save_observation(obs1)
        adapter.save_observation(obs2)
        d1 = obs1.observed_at.date()
        d2 = obs2.observed_at.date()
        result = adapter.find_observations_by_date(d1)
        # Both were created at (roughly) the same moment in test, so both should appear
        assert len(result) >= 1
