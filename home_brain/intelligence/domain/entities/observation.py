"""
Intelligence domain entities: Observation, Transcript, DailySummary, Mood.

What it is: Pure domain objects for the Intelligence bounded context.
What it knows: Business rules for mood classification, confidence bounds,
               highlight limits, and notification state transitions.
What it doesn't know: Anything about databases, APIs, ffmpeg, or HTTP.
"""

import uuid
from dataclasses import dataclass, field, replace
from datetime import date, datetime
from enum import Enum
from typing import Optional


class Mood(str, Enum):
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    NEUTRAL = "neutral"
    CALM = "calm"


@dataclass
class Observation:
    """Claude's analysis of a single Clip: mood, activity, description, confidence."""

    id: str
    clip_id: str
    mood: Mood
    activity: str
    description: str
    confidence: float
    observed_at: datetime

    @classmethod
    def create(
        cls,
        clip_id: str,
        mood: "Mood | str",
        activity: str,
        description: str,
        confidence: float,
    ) -> "Observation":
        if isinstance(mood, str):
            try:
                mood = Mood(mood)
            except ValueError:
                valid = [m.value for m in Mood]
                raise ValueError(
                    f"Invalid mood '{mood}'. Must be one of: {valid}"
                )
        if not isinstance(confidence, (int, float)) or not (0.0 <= confidence <= 1.0):
            raise ValueError(
                f"confidence must be between 0.0 and 1.0, got {confidence}"
            )
        if not activity or not activity.strip():
            raise ValueError("activity must not be empty")
        if not description or not description.strip():
            raise ValueError("description must not be empty")
        return cls(
            id=str(uuid.uuid4()),
            clip_id=clip_id,
            mood=mood,
            activity=activity,
            description=description,
            confidence=float(confidence),
            observed_at=datetime.utcnow(),
        )


@dataclass
class Transcript:
    """Whisper's audio-to-text output for a single Clip."""

    id: str
    clip_id: str
    text: str
    language: str
    observed_at: datetime

    @classmethod
    def create(cls, clip_id: str, text: str, language: str) -> "Transcript":
        if not language or not language.strip():
            raise ValueError("language must not be empty")
        return cls(
            id=str(uuid.uuid4()),
            clip_id=clip_id,
            text=text,
            language=language,
            observed_at=datetime.utcnow(),
        )


@dataclass
class DailySummary:
    """Claude Sonnet's narrative of a full day's Observations."""

    id: str
    date: date
    summary_text: str
    mood_counts: dict
    highlight_clip_ids: list
    notified: bool

    @classmethod
    def create(
        cls,
        date: date,
        summary_text: str,
        mood_counts: dict,
        highlight_clip_ids: list,
    ) -> "DailySummary":
        if not summary_text or not summary_text.strip():
            raise ValueError("summary_text must not be empty")
        if len(highlight_clip_ids) > 3:
            raise ValueError(
                f"highlight_clip_ids must contain at most 3 entries, "
                f"got {len(highlight_clip_ids)}"
            )
        return cls(
            id=str(uuid.uuid4()),
            date=date,
            summary_text=summary_text,
            mood_counts=dict(mood_counts),
            highlight_clip_ids=list(highlight_clip_ids),
            notified=False,
        )

    def mark_notified(self) -> "DailySummary":
        """Return a new DailySummary with notified=True."""
        return replace(self, notified=True)
