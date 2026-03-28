"""
Tests for ClipAnalysisService — the Intelligence domain orchestrator.

Uses in-memory fakes for all 4 ports. No external dependencies.
Negative tests (RED) come first, then positive tests.
"""

import pytest
from datetime import date, datetime
from pathlib import Path
from typing import List, Optional
from unittest.mock import MagicMock

from home_brain.intelligence.domain.entities.observation import (
    DailySummary,
    Mood,
    Observation,
    Transcript,
)
from home_brain.intelligence.domain.ports.ports import (
    IClipQueryPort,
    IObservationStorePort,
    ITranscriptionPort,
    IVideoAnalysisPort,
)
from home_brain.intelligence.domain.services.clip_analysis_service import (
    ClipAnalysisService,
)
from home_brain.surveillance.domain.entities.clip import Clip, ClipStatus


# ---------------------------------------------------------------------------
# In-memory fakes
# ---------------------------------------------------------------------------


def _make_clip(clip_id: str = "clip-1", file_path: str = "/data/clip1.mp4") -> Clip:
    c = Clip.create(
        recorded_at=datetime(2026, 3, 28, 10, 0, 0),
        duration_seconds=30.0,
        file_path=Path(file_path),
    )
    # Override id for deterministic tests
    object.__setattr__(c, "id", clip_id)
    return c


def _make_observation(clip_id: str = "clip-1", confidence: float = 0.9) -> Observation:
    return Observation.create(
        clip_id=clip_id,
        mood=Mood.NEUTRAL,
        activity="person walking",
        description="A person walks through the hallway.",
        confidence=confidence,
    )


def _make_transcript(clip_id: str = "clip-1") -> Transcript:
    return Transcript.create(clip_id=clip_id, text="Hello.", language="en")


class FakeClipQuery(IClipQueryPort):
    def __init__(self, clips: List[Clip]):
        self._clips = list(clips)
        self.analysed: List[str] = []
        self.failed: List[str] = []

    def find_pending(self) -> List[Clip]:
        return [c for c in self._clips if c.status == ClipStatus.PENDING]

    def mark_analysed(self, clip_id: str) -> None:
        self.analysed.append(clip_id)
        for i, c in enumerate(self._clips):
            if c.id == clip_id:
                object.__setattr__(c, "status", ClipStatus.ANALYSED)

    def mark_failed(self, clip_id: str) -> None:
        self.failed.append(clip_id)
        for i, c in enumerate(self._clips):
            if c.id == clip_id:
                object.__setattr__(c, "status", ClipStatus.FAILED)


class FakeVideoAnalysis(IVideoAnalysisPort):
    def __init__(self, observations: dict):
        # clip_id → Observation OR Exception instance
        self._obs = observations

    def analyse(self, clip_path: Path, clip_id: str) -> Observation:
        result = self._obs.get(clip_id)
        if isinstance(result, Exception):
            raise result
        return result


class FakeTranscription(ITranscriptionPort):
    def __init__(self, transcripts: dict):
        # clip_id → Transcript OR Exception instance
        self._tr = transcripts

    def transcribe(self, clip_path: Path, clip_id: str) -> Transcript:
        result = self._tr.get(clip_id)
        if isinstance(result, Exception):
            raise result
        if result is None:
            return _make_transcript(clip_id)
        return result


class FakeObservationStore(IObservationStorePort):
    def __init__(self):
        self.observations: List[Observation] = []
        self.transcripts: List[Transcript] = []
        self.summaries: List[DailySummary] = []

    def save_observation(self, obs: Observation) -> None:
        self.observations.append(obs)

    def save_transcript(self, transcript: Transcript) -> None:
        self.transcripts.append(transcript)

    def save_daily_summary(self, summary: DailySummary) -> None:
        self.summaries.append(summary)

    def find_observations_by_date(self, target_date: date) -> List[Observation]:
        return [
            o for o in self.observations
            if o.observed_at.date() == target_date
        ]

    def find_summary_by_date(self, target_date: date) -> Optional[DailySummary]:
        for s in self.summaries:
            if s.date == target_date:
                return s
        return None

    def update_daily_summary(self, summary: DailySummary) -> None:
        for i, s in enumerate(self.summaries):
            if s.id == summary.id:
                self.summaries[i] = summary


def _build_service(clips, obs_map=None, tr_map=None):
    obs_map = obs_map or {}
    tr_map = tr_map or {}
    clip_query = FakeClipQuery(clips)
    video = FakeVideoAnalysis(obs_map)
    transcription = FakeTranscription(tr_map)
    store = FakeObservationStore()
    svc = ClipAnalysisService(
        clip_query=clip_query,
        video_analysis=video,
        transcription=transcription,
        observation_store=store,
    )
    return svc, clip_query, store


# ---------------------------------------------------------------------------
# NEGATIVE TESTS
# ---------------------------------------------------------------------------


class TestClipAnalysisServiceNegative:
    def test_video_analysis_failure_marks_clip_failed_not_analysed(self):
        clip = _make_clip("clip-1")
        svc, clip_query, store = _build_service(
            [clip],
            obs_map={"clip-1": RuntimeError("Claude API down")},
        )
        result = svc.analyse_pending_clips()
        assert result == []
        assert "clip-1" in clip_query.failed
        assert "clip-1" not in clip_query.analysed

    def test_video_analysis_failure_does_not_crash_other_clips(self):
        clip1 = _make_clip("clip-1", "/data/clip1.mp4")
        clip2 = _make_clip("clip-2", "/data/clip2.mp4")
        obs2 = _make_observation("clip-2")
        svc, clip_query, store = _build_service(
            [clip1, clip2],
            obs_map={
                "clip-1": RuntimeError("Claude down"),
                "clip-2": obs2,
            },
        )
        result = svc.analyse_pending_clips()
        assert len(result) == 1
        assert result[0].clip_id == "clip-2"
        assert "clip-1" in clip_query.failed
        assert "clip-2" in clip_query.analysed

    def test_no_pending_clips_returns_empty_list(self):
        svc, _, _ = _build_service([])
        result = svc.analyse_pending_clips()
        assert result == []

    def test_generate_daily_summary_with_no_observations_raises(self):
        svc, _, _ = _build_service([])
        with pytest.raises(ValueError, match="[Nn]o observations"):
            svc.generate_daily_summary(date(2026, 3, 28))

    def test_transcription_failure_still_saves_observation_and_marks_analysed(self):
        clip = _make_clip("clip-1")
        obs = _make_observation("clip-1")
        svc, clip_query, store = _build_service(
            [clip],
            obs_map={"clip-1": obs},
            tr_map={"clip-1": RuntimeError("Whisper OOM")},
        )
        result = svc.analyse_pending_clips()
        # observation saved, clip marked analysed despite transcription failure
        assert len(result) == 1
        assert len(store.observations) == 1
        assert "clip-1" in clip_query.analysed
        assert len(store.transcripts) == 0


# ---------------------------------------------------------------------------
# POSITIVE TESTS
# ---------------------------------------------------------------------------


class TestClipAnalysisServicePositive:
    def test_two_pending_clips_returns_two_observations(self):
        clip1 = _make_clip("clip-1", "/data/clip1.mp4")
        clip2 = _make_clip("clip-2", "/data/clip2.mp4")
        obs1 = _make_observation("clip-1", confidence=0.8)
        obs2 = _make_observation("clip-2", confidence=0.9)
        svc, clip_query, store = _build_service(
            [clip1, clip2],
            obs_map={"clip-1": obs1, "clip-2": obs2},
        )
        result = svc.analyse_pending_clips()
        assert len(result) == 2
        assert "clip-1" in clip_query.analysed
        assert "clip-2" in clip_query.analysed

    def test_observations_and_transcripts_saved(self):
        clip = _make_clip("clip-1")
        obs = _make_observation("clip-1")
        tr = _make_transcript("clip-1")
        svc, _, store = _build_service(
            [clip],
            obs_map={"clip-1": obs},
            tr_map={"clip-1": tr},
        )
        svc.analyse_pending_clips()
        assert len(store.observations) == 1
        assert len(store.transcripts) == 1

    def test_idempotent_second_call_returns_empty(self):
        clip = _make_clip("clip-1")
        obs = _make_observation("clip-1")
        svc, clip_query, store = _build_service(
            [clip],
            obs_map={"clip-1": obs},
        )
        svc.analyse_pending_clips()
        result2 = svc.analyse_pending_clips()
        assert result2 == []

    def test_generate_daily_summary_aggregates_mood_counts(self):
        # Inject observations directly into the store by calling analyse first
        clip1 = _make_clip("clip-1", "/data/clip1.mp4")
        clip2 = _make_clip("clip-2", "/data/clip2.mp4")
        clip3 = _make_clip("clip-3", "/data/clip3.mp4")

        today = date(2026, 3, 28)

        obs1 = Observation.create("clip-1", Mood.HAPPY, "playing", "Kids playing.", 0.9)
        obs2 = Observation.create("clip-2", Mood.NEUTRAL, "walking", "Person walking.", 0.7)
        obs3 = Observation.create("clip-3", Mood.HAPPY, "dancing", "Person dancing.", 0.85)

        svc, _, store = _build_service(
            [clip1, clip2, clip3],
            obs_map={"clip-1": obs1, "clip-2": obs2, "clip-3": obs3},
        )
        svc.analyse_pending_clips()
        summary = svc.generate_daily_summary(today)

        assert summary.date == today
        assert summary.mood_counts.get("happy") == 2
        assert summary.mood_counts.get("neutral") == 1

    def test_generate_daily_summary_picks_top_3_by_confidence(self):
        clips = [_make_clip(f"clip-{i}", f"/data/clip{i}.mp4") for i in range(1, 5)]
        obs_map = {
            "clip-1": Observation.create("clip-1", Mood.NEUTRAL, "a", "desc.", 0.5),
            "clip-2": Observation.create("clip-2", Mood.NEUTRAL, "b", "desc.", 0.95),
            "clip-3": Observation.create("clip-3", Mood.NEUTRAL, "c", "desc.", 0.85),
            "clip-4": Observation.create("clip-4", Mood.NEUTRAL, "d", "desc.", 0.75),
        }
        svc, _, store = _build_service(clips, obs_map=obs_map)
        svc.analyse_pending_clips()
        summary = svc.generate_daily_summary(date(2026, 3, 28))
        assert len(summary.highlight_clip_ids) == 3
        # clip-2 (0.95), clip-3 (0.85), clip-4 (0.75) — clip-1 excluded
        assert "clip-2" in summary.highlight_clip_ids
        assert "clip-1" not in summary.highlight_clip_ids

    def test_generate_daily_summary_saved_to_store(self):
        clip = _make_clip("clip-1")
        obs = _make_observation("clip-1")
        svc, _, store = _build_service([clip], obs_map={"clip-1": obs})
        svc.analyse_pending_clips()
        summary = svc.generate_daily_summary(date(2026, 3, 28))
        assert store.find_summary_by_date(date(2026, 3, 28)) is not None
