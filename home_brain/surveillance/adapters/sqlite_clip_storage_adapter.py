"""
Surveillance Adapters — SQLite Clip Storage

What it is: persistence adapter for Clip entities using SQLite.
What it knows: SQL schema, sqlite3 API, Clip ↔ row mapping.
What it doesn't know: domain rules, ffmpeg, any other adapter.

SQLite is the right choice here:
  - Zero server setup, single file, easy backup
  - Fast enough for thousands of clips
  - Works fully offline (no cloud dependency)

Schema lives here. Migrations handled simply via CREATE TABLE IF NOT EXISTS.
"""
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Generator, List, Optional

from ..domain.entities.clip import Clip, ClipStatus
from ..domain.ports.ports import IClipStoragePort
from ...shared.logger import get_logger

logger = get_logger("surveillance.adapters.sqlite")


class SQLiteClipStorageAdapter(IClipStoragePort):

    def __init__(self, db_path: Path):
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    @contextmanager
    def _connection(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")   # better concurrent reads
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
                CREATE INDEX IF NOT EXISTS idx_clips_recorded_at
                ON clips(recorded_at)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_clips_status
                ON clips(status)
            """)
        logger.debug(f"Schema ensured at {self._db_path}")

    # ── IClipStoragePort Implementation ──────────────────────────────────────

    def save(self, clip: Clip) -> None:
        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO clips (id, recorded_at, duration_s, file_path, status, size_bytes, archive_url)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    clip.id,
                    clip.recorded_at.isoformat(),
                    clip.duration_seconds,
                    str(clip.file_path),
                    clip.status.value,
                    clip.size_bytes,
                    clip.archive_url,
                ),
            )
        logger.debug(f"Clip saved: {clip.id} ({clip.recorded_at.strftime('%H:%M:%S')})")

    def find_by_date(self, target_date: date) -> List[Clip]:
        date_prefix = target_date.strftime("%Y-%m-%d")
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT * FROM clips WHERE recorded_at LIKE ? ORDER BY recorded_at",
                (f"{date_prefix}%",),
            ).fetchall()
        return [self._row_to_clip(row) for row in rows]

    def find_pending(self) -> List[Clip]:
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT * FROM clips WHERE status = 'pending' ORDER BY recorded_at",
            ).fetchall()
        return [self._row_to_clip(row) for row in rows]

    def update(self, clip: Clip) -> None:
        with self._connection() as conn:
            conn.execute(
                """
                UPDATE clips
                SET status = ?, size_bytes = ?, archive_url = ?
                WHERE id = ?
                """,
                (clip.status.value, clip.size_bytes, clip.archive_url, clip.id),
            )
        logger.debug(f"Clip updated: {clip.id} → status={clip.status.value}")

    # ── Helpers ───────────────────────────────────────────────────────────────

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
