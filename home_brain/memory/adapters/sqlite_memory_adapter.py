"""
Memory Adapters — SQLite Memory Storage.

What it is: Persistence adapter implementing IClipRetentionPort + ISummaryQueryPort.
            Reads and writes the clips and daily_summaries tables in home_brain.sqlite.
What it knows: SQL schema, sqlite3 API, entity ↔ row mapping, JSON for mood_counts
               and highlight_clip_ids.
What it doesn't know: domain rules, Claude API, Telegram, B2, or any other adapter.

The clips and daily_summaries tables may already exist (created by Surveillance or
Intelligence adapters). Schema is ensured on __init__ via CREATE TABLE IF NOT EXISTS.
"""

import json
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Generator, List, Optional

from home_brain.intelligence.domain.entities.observation import DailySummary
from home_brain.memory.domain.ports.ports import IClipRetentionPort, ISummaryQueryPort
from home_brain.shared.logger import get_logger
from home_brain.surveillance.domain.entities.clip import Clip, ClipStatus

logger = get_logger("memory.adapters.sqlite")


class SQLiteMemoryAdapter(IClipRetentionPort, ISummaryQueryPort):
    """
    Implements IClipRetentionPort and ISummaryQueryPort against the shared
    home_brain.sqlite database.

    No new tables are created — reads and writes the existing clips and
    daily_summaries tables.
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    # ------------------------------------------------------------------
    # Connection management
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
                CREATE TABLE IF NOT EXISTS daily_summaries (
                    id                  TEXT PRIMARY KEY,
                    date                TEXT NOT NULL UNIQUE,
                    summary_text        TEXT NOT NULL,
                    mood_counts         TEXT NOT NULL,
                    highlight_clip_ids  TEXT NOT NULL,
                    notified            INTEGER NOT NULL DEFAULT 0
                )
            """)
        logger.debug(f"Memory schema ensured at {self._db_path}")

    # ------------------------------------------------------------------
    # IClipRetentionPort
    # ------------------------------------------------------------------

    def find_clips_by_ids(self, clip_ids: List[str]) -> List[Clip]:
        if not clip_ids:
            return []
        placeholders = ",".join("?" * len(clip_ids))
        with self._connection() as conn:
            rows = conn.execute(
                f"SELECT * FROM clips WHERE id IN ({placeholders})",
                clip_ids,
            ).fetchall()
        return [self._row_to_clip(row) for row in rows]

    def find_archivable_clips(self, retention_days: int) -> List[Clip]:
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT * FROM clips WHERE status = 'analysed' "
                "AND recorded_at < datetime('now', ? || ' days')",
                (f"-{retention_days}",),
            ).fetchall()
        return [self._row_to_clip(row) for row in rows]

    def mark_archived(self, clip_id: str, archive_url: str) -> None:
        with self._connection() as conn:
            result = conn.execute(
                "UPDATE clips SET status = 'archived', archive_url = ? WHERE id = ?",
                (archive_url, clip_id),
            )
        if result.rowcount == 0:
            logger.warning(f"mark_archived: no clip found with id={clip_id}")

    # ------------------------------------------------------------------
    # ISummaryQueryPort
    # ------------------------------------------------------------------

    def find_summary_by_date(self, target_date: date) -> Optional[DailySummary]:
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM daily_summaries WHERE date = ?",
                (target_date.isoformat(),),
            ).fetchone()
        if row is None:
            return None
        return self._row_to_daily_summary(row)

    def mark_summary_notified(self, summary_id: str) -> None:
        with self._connection() as conn:
            result = conn.execute(
                "UPDATE daily_summaries SET notified = 1 WHERE id = ?",
                (summary_id,),
            )
        if result.rowcount == 0:
            logger.warning(f"mark_summary_notified: no summary found with id={summary_id}")

    # ------------------------------------------------------------------
    # Row mappers
    # ------------------------------------------------------------------

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

    def _row_to_daily_summary(self, row: sqlite3.Row) -> DailySummary:
        return DailySummary(
            id=row["id"],
            date=date.fromisoformat(row["date"]),
            summary_text=row["summary_text"],
            mood_counts=json.loads(row["mood_counts"]),
            highlight_clip_ids=json.loads(row["highlight_clip_ids"]),
            notified=bool(row["notified"]),
        )
