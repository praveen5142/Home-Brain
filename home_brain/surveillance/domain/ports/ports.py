"""
Surveillance Domain — Ports

What it is: abstract interfaces (ports) that the Surveillance domain defines.
What it knows: domain entities (Clip, MotionWindow), Python ABCs.
What it doesn't know: ffmpeg, SQLite, or any concrete implementation.

Ports are contracts. Adapters implement them. The domain only ever speaks to ports.

  IStreamIngestionPort  → inbound: triggers the pipeline (called by CLI / scheduler)
  IMotionDetectionPort  → outbound: domain asks for motion windows
  IStreamRecorderPort   → outbound: domain asks to record / extract segments
  IClipStoragePort      → outbound: domain asks to persist / query Clips
"""
from abc import ABC, abstractmethod
from datetime import date
from pathlib import Path
from typing import List

from ..entities.clip import Clip, MotionWindow


# ── Inbound Port ──────────────────────────────────────────────────────────────

class IStreamIngestionPort(ABC):
    """
    Inbound port — what the outside world (scheduler, CLI) calls
    to trigger the ingestion pipeline.
    """

    @abstractmethod
    def run_daily_extraction(self, target_date: date) -> List[Clip]:
        """
        Pull the full day's footage, detect motion, extract clips.
        Returns all Clip entities created for that day.
        """
        ...


# ── Outbound Ports ────────────────────────────────────────────────────────────

class IMotionDetectionPort(ABC):
    """
    Outbound port — domain asks: "where is the motion in this recording?"
    The adapter answers using ffmpeg, OpenCV, or anything else.
    Domain does not care.
    """

    @abstractmethod
    def detect_motion_windows(
        self,
        recording_path: Path,
        scene_threshold: float,
    ) -> List[MotionWindow]:
        """
        Returns a list of MotionWindow objects for a recording file.
        Ordered by start_seconds ascending.
        """
        ...


class IStreamRecorderPort(ABC):
    """
    Outbound port — domain asks: "record the stream" or "cut a segment."
    """

    @abstractmethod
    def record_stream(
        self,
        rtsp_url: str,
        output_path: Path,
        duration_seconds: int,
    ) -> Path:
        """
        Records RTSP stream to a local file.
        Returns the path of the saved recording.
        """
        ...

    @abstractmethod
    def extract_segment(
        self,
        source_path: Path,
        output_path: Path,
        start_seconds: float,
        duration_seconds: float,
    ) -> Path:
        """
        Cuts a segment from a recording file.
        Returns path of the new clip file.
        """
        ...


class IClipStoragePort(ABC):
    """
    Outbound port — domain asks: "save / query Clip entities."
    """

    @abstractmethod
    def save(self, clip: Clip) -> None:
        ...

    @abstractmethod
    def find_by_date(self, target_date: date) -> List[Clip]:
        ...

    @abstractmethod
    def find_pending(self) -> List[Clip]:
        """Returns all clips awaiting Intelligence domain analysis."""
        ...

    @abstractmethod
    def update(self, clip: Clip) -> None:
        ...
