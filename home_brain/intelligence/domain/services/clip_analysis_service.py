"""
Intelligence domain service: ClipAnalysisService.

What it is: Orchestrator for the Intelligence bounded context pipeline.
What it knows: Domain ports (IClipQueryPort, IVideoAnalysisPort,
               ITranscriptionPort, IObservationStorePort) and domain entities.
What it doesn't know: SQLite, Claude API, Whisper — any infrastructure detail.
"""

from collections import Counter
from datetime import date
from typing import List

from home_brain.intelligence.domain.entities.observation import (
    DailySummary,
    Observation,
)
from home_brain.intelligence.domain.ports.ports import (
    IClipQueryPort,
    IObservationStorePort,
    ITranscriptionPort,
    IVideoAnalysisPort,
)
from home_brain.shared.logger import get_logger

logger = get_logger("intelligence.ClipAnalysisService")


class ClipAnalysisService:
    """Processes pending Clips → Observations + Transcripts, and generates DailySummaries."""

    def __init__(
        self,
        clip_query: IClipQueryPort,
        video_analysis: IVideoAnalysisPort,
        transcription: ITranscriptionPort,
        observation_store: IObservationStorePort,
    ) -> None:
        self._clip_query = clip_query
        self._video_analysis = video_analysis
        self._transcription = transcription
        self._store = observation_store

    def analyse_pending_clips(self) -> List[Observation]:
        """
        Fetch all pending Clips, analyse each with Vision + Transcription,
        persist results, and mark each clip analysed (or failed on error).

        Transcription is best-effort: a failure there does NOT prevent the
        clip from being marked as analysed.

        Returns the list of Observations produced in this run.
        """
        pending = self._clip_query.find_pending()
        if not pending:
            logger.info("No pending clips to analyse.")
            return []

        logger.info(f"Analysing {len(pending)} pending clip(s).")
        produced: List[Observation] = []

        for clip in pending:
            obs = self._analyse_single_clip(clip)
            if obs is not None:
                produced.append(obs)

        logger.info(f"Completed. {len(produced)}/{len(pending)} clips analysed successfully.")
        return produced

    def generate_daily_summary(self, target_date: date) -> DailySummary:
        """
        Load all Observations for target_date and build a DailySummary.

        Raises ValueError if there are no Observations for that date.
        """
        observations = self._store.find_observations_by_date(target_date)
        if not observations:
            raise ValueError(
                f"No observations found for {target_date}. "
                "Run analyse_pending_clips first."
            )

        mood_counts = dict(Counter(obs.mood.value for obs in observations))

        # Top 3 clips by confidence (highest first)
        sorted_obs = sorted(observations, key=lambda o: o.confidence, reverse=True)
        highlight_clip_ids = [o.clip_id for o in sorted_obs[:3]]

        summary_text = self._build_summary_text(observations, mood_counts)

        summary = DailySummary.create(
            date=target_date,
            summary_text=summary_text,
            mood_counts=mood_counts,
            highlight_clip_ids=highlight_clip_ids,
        )
        self._store.save_daily_summary(summary)
        logger.info(f"Daily summary saved for {target_date}: {len(observations)} observations.")
        return summary

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _analyse_single_clip(self, clip) -> "Observation | None":
        clip_id = clip.id
        clip_path = clip.file_path

        # Step 1: Video analysis (mandatory — failure marks clip failed)
        try:
            obs = self._video_analysis.analyse(clip_path, clip_id)
        except Exception as exc:
            logger.error(f"Video analysis failed for clip {clip_id}: {exc}")
            self._clip_query.mark_failed(clip_id)
            return None

        # Step 2: Persist observation
        self._store.save_observation(obs)

        # Step 3: Transcription (best-effort — failure still marks clip analysed)
        try:
            transcript = self._transcription.transcribe(clip_path, clip_id)
            self._store.save_transcript(transcript)
        except Exception as exc:
            logger.warning(f"Transcription failed for clip {clip_id} (non-fatal): {exc}")

        # Step 4: Mark clip as analysed
        self._clip_query.mark_analysed(clip_id)
        return obs

    def _build_summary_text(self, observations: List[Observation], mood_counts: dict) -> str:
        """Build a simple summary text from the observations list."""
        total = len(observations)
        dominant_mood = max(mood_counts, key=mood_counts.get)
        activities = "; ".join(obs.activity for obs in observations[:5])
        return (
            f"Day summary: {total} clip(s) analysed. "
            f"Dominant mood: {dominant_mood}. "
            f"Activities: {activities}."
        )
