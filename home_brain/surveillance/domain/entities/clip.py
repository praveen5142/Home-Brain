"""
Surveillance Domain — Entities

What it is: pure domain objects using the Ubiquitous Language.
What it knows: Stream, MotionWindow, Clip, ClipStatus — and their business rules.
What it doesn't know: ffmpeg, sqlite, file systems, any infrastructure detail.
"""
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional
import uuid


class ClipStatus(Enum):
    PENDING = "pending"      # extracted, awaiting analysis
    ANALYSED = "analysed"    # Intelligence domain processed it
    ARCHIVED = "archived"    # moved to Backblaze B2
    FAILED = "failed"        # extraction or analysis failed


@dataclass(frozen=True)
class MotionWindow:
    """
    A contiguous time window where activity was detected in the stream.
    The domain doesn't care HOW motion was detected — that's the adapter's job.
    """
    start_seconds: float       # offset from stream start
    end_seconds: float
    confidence: float          # 0.0–1.0, how strong the scene change was

    @property
    def duration_seconds(self) -> float:
        return self.end_seconds - self.start_seconds

    def overlaps_or_adjacent(self, other: "MotionWindow", gap_s: float) -> bool:
        return self.end_seconds + gap_s >= other.start_seconds

    def merge_with(self, other: "MotionWindow") -> "MotionWindow":
        return MotionWindow(
            start_seconds=min(self.start_seconds, other.start_seconds),
            end_seconds=max(self.end_seconds, other.end_seconds),
            confidence=max(self.confidence, other.confidence),
        )


@dataclass
class Clip:
    """
    A meaningful segment of footage extracted from a MotionWindow.
    Carries its own identity through the full pipeline.
    """
    id: str
    recorded_at: datetime          # wall-clock time of the clip start
    duration_seconds: float
    file_path: Path
    status: ClipStatus = ClipStatus.PENDING
    size_bytes: Optional[int] = None
    archive_url: Optional[str] = None

    @classmethod
    def create(
        cls,
        recorded_at: datetime,
        duration_seconds: float,
        file_path: Path,
    ) -> "Clip":
        return cls(
            id=str(uuid.uuid4()),
            recorded_at=recorded_at,
            duration_seconds=duration_seconds,
            file_path=file_path,
        )

    def mark_archived(self, archive_url: str) -> None:
        self.status = ClipStatus.ARCHIVED
        self.archive_url = archive_url

    def mark_failed(self) -> None:
        self.status = ClipStatus.FAILED

    def mark_analysed(self) -> None:
        self.status = ClipStatus.ANALYSED

    @property
    def is_ready_for_analysis(self) -> bool:
        return self.status == ClipStatus.PENDING and self.file_path.exists()
