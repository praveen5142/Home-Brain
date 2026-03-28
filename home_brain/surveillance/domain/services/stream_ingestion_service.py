"""
Surveillance Domain — StreamIngestionService

What it is: the heart of the Surveillance bounded context; full pipeline orchestration.
What it knows: domain rules for merging, filtering, and splitting MotionWindows.
What it doesn't know: ffmpeg, SQLite, env vars, file systems — zero infrastructure detail.

Domain Rules encoded here:
- Merge nearby motion windows to avoid micro-clips
- Skip clips under minimum duration (noise)
- Split clips over maximum duration (keep files manageable)
- Record wall-clock time for each clip (critical for daily summaries)
"""
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List

from ..entities.clip import Clip, MotionWindow
from ..ports.ports import (
    IClipStoragePort,
    IMotionDetectionPort,
    IStreamIngestionPort,
    IStreamRecorderPort,
)
from ....shared.config import AppConfig
from ....shared.logger import get_logger

logger = get_logger("surveillance.StreamIngestionService")


class StreamIngestionService(IStreamIngestionPort):
    """
    Orchestrates: record → detect motion → extract clips → store.
    Implements the inbound port so the scheduler can call it directly.
    """

    def __init__(
        self,
        config: AppConfig,
        recorder: IStreamRecorderPort,
        motion_detector: IMotionDetectionPort,
        clip_storage: IClipStoragePort,
    ):
        self._config = config
        self._recorder = recorder
        self._motion_detector = motion_detector
        self._clip_storage = clip_storage

    # ── Inbound Port Implementation ───────────────────────────────────────────

    def run_daily_extraction(self, target_date: date) -> List[Clip]:
        """
        Full pipeline for one day.
        In production this runs at ~06:00 for the previous day.
        For testing, pass today's date to record a short window.
        """
        logger.info(f"Starting daily extraction for {target_date}")

        # Step 1: Record (or locate existing daily recording)
        recording_path = self._get_recording_path(target_date)
        if not recording_path.exists():
            logger.info("No existing recording found — pulling from RTSP stream")
            recording_path = self._record_full_day(target_date, recording_path)
        else:
            logger.info(f"Using existing recording: {recording_path}")

        # Step 2: Detect motion windows
        logger.info("Detecting motion windows...")
        raw_windows = self._motion_detector.detect_motion_windows(
            recording_path=recording_path,
            scene_threshold=self._config.motion.scene_threshold,
        )
        logger.info(f"Raw motion windows detected: {len(raw_windows)}")

        # Step 3: Apply domain rules to clean up windows
        clean_windows = self._apply_domain_rules(raw_windows)
        logger.info(f"Clean motion windows after domain rules: {len(clean_windows)}")

        # Step 4: Extract clips for each window
        clips = self._extract_clips(recording_path, clean_windows, target_date)
        logger.info(f"Clips extracted: {len(clips)}")

        # Step 5: Persist all clips
        for clip in clips:
            self._clip_storage.save(clip)

        logger.info(f"Daily extraction complete. {len(clips)} clips saved.")
        return clips

    # ── Domain Rules ──────────────────────────────────────────────────────────

    def _apply_domain_rules(self, windows: List[MotionWindow]) -> List[MotionWindow]:
        """
        Three domain rules, applied in order:
        1. Merge windows that are close together (avoid fragmented clips)
        2. Drop windows that are too short (noise/glitch)
        3. Split windows that are too long (keep files manageable)
        """
        merged = self._merge_adjacent_windows(windows)
        filtered = self._filter_short_windows(merged)
        split = self._split_long_windows(filtered)
        return split

    def _merge_adjacent_windows(
        self, windows: List[MotionWindow]
    ) -> List[MotionWindow]:
        """Merge windows separated by less than merge_gap_s seconds."""
        if not windows:
            return []

        gap = self._config.motion.merge_gap_s
        merged: List[MotionWindow] = [windows[0]]

        for current in windows[1:]:
            last = merged[-1]
            if last.overlaps_or_adjacent(current, gap):
                merged[-1] = last.merge_with(current)
            else:
                merged.append(current)

        return merged

    def _filter_short_windows(
        self, windows: List[MotionWindow]
    ) -> List[MotionWindow]:
        """Drop windows shorter than min_clip_duration_s."""
        min_dur = self._config.motion.min_clip_duration_s
        return [w for w in windows if w.duration_seconds >= min_dur]

    def _split_long_windows(
        self, windows: List[MotionWindow]
    ) -> List[MotionWindow]:
        """Split windows longer than max_clip_duration_s into equal chunks."""
        max_dur = self._config.motion.max_clip_duration_s
        result: List[MotionWindow] = []

        for window in windows:
            if window.duration_seconds <= max_dur:
                result.append(window)
                continue

            start = window.start_seconds
            while start < window.end_seconds:
                end = min(start + max_dur, window.end_seconds)
                result.append(MotionWindow(
                    start_seconds=start,
                    end_seconds=end,
                    confidence=window.confidence,
                ))
                start = end

        return result

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_recording_path(self, target_date: date) -> Path:
        return (
            self._config.storage.clips_dir
            / target_date.strftime("%Y-%m-%d")
            / "full_recording.mp4"
        )

    def _record_full_day(self, target_date: date, output_path: Path) -> Path:
        """
        For a true 24h recording, duration = 86400s.
        For development/testing, set RECORD_DURATION_S=60 in .env.
        Duration comes from AppConfig — never read env vars directly in the domain.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return self._recorder.record_stream(
            rtsp_url=self._config.camera.rtsp_url,
            output_path=output_path,
            duration_seconds=self._config.record_duration_s,
        )

    def _extract_clips(
        self,
        recording_path: Path,
        windows: List[MotionWindow],
        target_date: date,
    ) -> List[Clip]:
        """
        For each MotionWindow, extract a clip file and create a Clip entity.
        Wall-clock time is reconstructed: recording_start + window.start_seconds.
        """
        # Assume recording started at midnight of target_date.
        # In production, read actual start time from file metadata.
        recording_start = datetime(
            target_date.year, target_date.month, target_date.day, 0, 0, 0
        )

        clips: List[Clip] = []
        clips_dir = self._config.storage.clips_dir / target_date.strftime("%Y-%m-%d")
        clips_dir.mkdir(parents=True, exist_ok=True)

        for i, window in enumerate(windows):
            clip_filename = f"clip_{i:04d}_{int(window.start_seconds):06d}s.mp4"
            clip_path = clips_dir / clip_filename

            try:
                self._recorder.extract_segment(
                    source_path=recording_path,
                    output_path=clip_path,
                    start_seconds=window.start_seconds,
                    duration_seconds=window.duration_seconds,
                )

                clip = Clip.create(
                    recorded_at=recording_start + timedelta(seconds=window.start_seconds),
                    duration_seconds=window.duration_seconds,
                    file_path=clip_path,
                )

                if clip_path.exists():
                    clip.size_bytes = clip_path.stat().st_size

                clips.append(clip)
                logger.debug(
                    f"Clip {i+1}/{len(windows)}: "
                    f"{clip.recorded_at.strftime('%H:%M:%S')} "
                    f"({window.duration_seconds:.1f}s) → {clip_path.name}"
                )

            except Exception as e:
                logger.error(f"Failed to extract clip {i}: {e}")
                continue

        return clips
