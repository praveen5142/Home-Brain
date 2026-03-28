"""
Intelligence Adapters — SQLite Observation Storage.

What it is: Persistence adapter for Observation, Transcript, and DailySummary entities.
            Also implements IClipQueryPort to bridge with the Surveillance domain via SQL
            (not via direct adapter import — only the shared SQLite file is the link).
What it knows: SQL schema, sqlite3 API, entity ↔ row mapping, JSON serialisation for
               mood_counts and highlight_clip_ids.
What it doesn't know: domain rules, Claude API, Whisper, ffmpeg, or any other adapter.

All three Intelligence tables live in the same home_brain.sqlite file as the clips table.
Schema is created on first __init__ call via CREATE TABLE IF NOT EXISTS.
"""

import json
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Generator, List, Optional

from home_brain.intelligence.domain.entities.observation import (
    DailySummary,
    Mood,
    Observation,
    Transcript,
)
from home_brain.intelligence.domain.ports.ports import (
    IClipQueryPort,
    IObservationStorePort,
)
from home_brain.shared.logger import get_logger
from home_brain.surveillance.domain.entities.clip import Clip, ClipStatus

logger = get_logger("intelligence.adapters.sqlite")


class SQLiteObservationAdapter(IObservationStorePort, IClipQueryPort):
    """
    Implements both IObservationStorePort and IClipQueryPort.

    IClipQueryPort is satisfied by reading/writing the clips table that was
    created by the Surveillance domain's SQLiteClipStorageAdapter.
    The adapter ensures the clips table exists (CREATE TABLE IF NOT EXISTS)
    so tests work with a fresh DB.
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    # ------------------------------------------------------------------
    # Connection management (same pattern as SQLiteClipStorageAdapter)
    # ------------------------------------------------------------------

    @contextmanager
    def _connection(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _ensure_schema(self) -> None:
        with self._connection() as conn:
            # Ensure the clips table exists (may have been created by Surveillance adapter)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS clips (
                    id           TEXT PRIMARY KEY,
                    recorded_at  TEXT NOT NULL,
                    duration_s   REAL NOT NULL,
                    file_path    TEXT NOT NULL,
                    status       TEXT NOT NULL DEFAULT 'pending',
                    size_bytes   INTEGER,
                    archive_url  TEXT,
                    created_at   TEXT NOT NULL DEFAULT (datetime('now'))
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS observations (
                    id          TEXT PRIMARY KEY,
                    clip_id     TEXT NOT NULL,
                    mood        TEXT NOT NULL,
                    activity    TEXT NOT NULL,
                    description TEXT NOT NULL,
                    confidence  REAL NOT NULL,
                    observed_at TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_observations_clip_id
                ON observations(clip_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_observations_observed_at
                ON observations(observed_at)
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS transcripts (
                    id          TEXT PRIMARY KEY,
                    clip_id     TEXT NOT NULL,
                    text        TEXT NOT NULL,
                    language    TEXT NOT NULL,
                    observed_at TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS daily_summaries (
                    id                  TEXT PRIMARY KEY,
                    date                TEXT NOT NULL UNIQUE,
                    summary_text        TEXT NOT NULL,
                    mood_counts         TEXT NOT NULL,
                    highlight_clip_ids  TEXT NOT NULL,
                    notified            INTEGER NOT NULL DEFAULT 0
                )
            """)
        logger.debug(f"Intelligence schema ensured at {self._db_path}")

    # ------------------------------------------------------------------
    # IObservationStorePort
    # ------------------------------------------------------------------

    def save_observation(self, obs: Observation) -> None:
        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO observations
                    (id, clip_id, mood, activity, description, confidence, observed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    obs.id,
                    obs.clip_id,
                    obs.mood.value,
                    obs.activity,
                    obs.description,
                    obs.confidence,
                    obs.observed_at.isoformat(),
                ),
            )
        logger.debug(f"Observation saved: {obs.id} for clip {obs.clip_id}")

    def save_transcript(self, transcript: Transcript) -> None:
        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO transcripts (id, clip_id, text, language, observed_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    transcript.id,
                    transcript.clip_id,
                    transcript.text,
                    transcript.language,
                    transcript.observed_at.isoformat(),
                ),
            )
        logger.debug(f"Transcript saved: {transcript.id} for clip {transcript.clip_id}")

    def save_daily_summary(self, summary: DailySummary) -> None:
        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO daily_summaries
                    (id, date, summary_text, mood_counts, highlight_clip_ids, notified)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    summary.id,
                    summary.date.isoformat(),
                    summary.summary_text,
                    json.dumps(summary.mood_counts),
                    json.dumps(summary.highlight_clip_ids),
                    int(summary.notified),
                ),
            )
        logger.debug(f"DailySummary saved for {summary.date}")

    def update_daily_summary(self, summary: DailySummary) -> None:
        with self._connection() as conn:
            conn.execute(
                """
                UPDATE daily_summaries
                SET summary_text = ?, mood_counts = ?, highlight_clip_ids = ?, notified = ?
                WHERE id = ?
                """,
                (
                    summary.summary_text,
                    json.dumps(summary.mood_counts),
                    json.dumps(summary.highlight_clip_ids),
                    int(summary.notified),
                    summary.id,
                ),
            )
        logger.debug(f"DailySummary updated: {summary.id}")

    def find_observations_by_date(self, target_date: date) -> List[Observation]:
        date_prefix = target_date.strftime("%Y-%m-%d")
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT * FROM observations WHERE observed_at LIKE ? ORDER BY observed_at",
                (f"{date_prefix}%",),
            ).fetchall()
        return [self._row_to_observation(row) for row in rows]

    def find_summary_by_date(self, target_date: date) -> Optional[DailySummary]:
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM daily_summaries WHERE date = ?",
                (target_date.isoformat(),),
            ).fetchone()
        if row is None:
            return None
        return self._row_to_daily_summary(row)

    # ------------------------------------------------------------------
    # IClipQueryPort — bridges to clips table from Surveillance domain
    # ------------------------------------------------------------------

    def find_pending(self) -> List[Clip]:
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT * FROM clips WHERE status = 'pending' ORDER BY recorded_at",
            ).fetchall()
        return [self._row_to_clip(row) for row in rows]

    def mark_analysed(self, clip_id: str) -> None:
        with self._connection() as conn:
            result = conn.execute(
                "UPDATE clips SET status = 'analysed' WHERE id = ?", (clip_id,)
            )
        if result.rowcount == 0:
            logger.warning(f"mark_analysed: no clip found with id={clip_id}")

    def mark_failed(self, clip_id: str) -> None:
        with self._connection() as conn:
            result = conn.execute(
                "UPDATE clips SET status = 'failed' WHERE id = ?", (clip_id,)
            )
        if result.rowcount == 0:
            logger.warning(f"mark_failed: no clip found with id={clip_id}")

    # ------------------------------------------------------------------
    # Row mappers
    # ------------------------------------------------------------------

    def _row_to_observation(self, row: sqlite3.Row) -> Observation:
        return Observation(
            id=row["id"],
            clip_id=row["clip_id"],
            mood=Mood(row["mood"]),
            activity=row["activity"],
            description=row["description"],
            confidence=row["confidence"],
            observed_at=datetime.fromisoformat(row["observed_at"]),
        )

    def _row_to_daily_summary(self, row: sqlite3.Row) -> DailySummary:
        return DailySummary(
            id=row["id"],
            date=date.fromisoformat(row["date"]),
            summary_text=row["summary_text"],
            mood_counts=json.loads(row["mood_counts"]),
            highlight_clip_ids=json.loads(row["highlight_clip_ids"]),
            notified=bool(row["notified"]),
        )

    def _row_to_clip(self, row: sqlite3.Row) -> Clip:
        return Clip(
            id=row["id"],
            recorded_at=datetime.fromisoformat(row["recorded_at"]),
            duration_seconds=row["duration_s"],
            file_path=Path(row["file_path"]),
            status=ClipStatus(row["status"]),
            size_bytes=row["size_bytes"],
            archive_url=row["archive_url"],
        )
