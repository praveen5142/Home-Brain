"""
Intelligence domain ports (abstract interfaces).

What it is: Port definitions for the Intelligence bounded context.
What it knows: Domain entity shapes (Observation, Transcript, DailySummary, Clip).
What it doesn't know: Any infrastructure detail — no SQLite, no API, no ffmpeg.
"""

from abc import ABC, abstractmethod
from datetime import date
from pathlib import Path
from typing import List, Optional

from home_brain.intelligence.domain.entities.observation import (
    DailySummary,
    Observation,
    Transcript,
)
from home_brain.surveillance.domain.entities.clip import Clip


class IClipQueryPort(ABC):
    """Outbound port — query and update Clips from the Surveillance domain."""

    @abstractmethod
    def find_pending(self) -> List[Clip]:
        """Return all Clips with status=pending."""

    @abstractmethod
    def mark_analysed(self, clip_id: str) -> None:
        """Mark a Clip as analysed."""

    @abstractmethod
    def mark_failed(self, clip_id: str) -> None:
        """Mark a Clip as failed."""


class IVideoAnalysisPort(ABC):
    """Outbound port — send clip frames to an AI model and receive an Observation."""

    @abstractmethod
    def analyse(self, clip_path: Path, clip_id: str) -> Observation:
        """Analyse a clip file and return an Observation."""


class ITranscriptionPort(ABC):
    """Outbound port — transcribe audio from a clip file."""

    @abstractmethod
    def transcribe(self, clip_path: Path, clip_id: str) -> Transcript:
        """Transcribe audio from a clip file and return a Transcript."""


class IObservationStorePort(ABC):
    """Outbound port — persist and query Intelligence domain aggregates."""

    @abstractmethod
    def save_observation(self, obs: Observation) -> None:
        """Persist an Observation."""

    @abstractmethod
    def save_transcript(self, transcript: Transcript) -> None:
        """Persist a Transcript."""

    @abstractmethod
    def save_daily_summary(self, summary: DailySummary) -> None:
        """Persist a DailySummary."""

    @abstractmethod
    def find_observations_by_date(self, target_date: date) -> List[Observation]:
        """Return all Observations for clips recorded on target_date."""

    @abstractmethod
    def find_summary_by_date(self, target_date: date) -> Optional[DailySummary]:
        """Return the DailySummary for target_date, or None if not yet generated."""

    @abstractmethod
    def update_daily_summary(self, summary: DailySummary) -> None:
        """Update an existing DailySummary (e.g. after mark_notified)."""
