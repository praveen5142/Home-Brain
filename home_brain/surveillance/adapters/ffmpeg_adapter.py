"""
Surveillance Adapters — ffmpeg

What it is: two adapters backed by the ffmpeg CLI tool.
What it knows: ffmpeg commands, subprocess handling, showinfo log parsing.
What it doesn't know: domain rules, config, SQLite — nothing outside this file's job.

Two adapters in one file (they're tightly coupled to ffmpeg):
  FfmpegStreamRecorderAdapter  → implements IStreamRecorderPort
  FfmpegMotionDetectionAdapter → implements IMotionDetectionPort

This is the ONLY file that knows about ffmpeg.
If you ever swap ffmpeg for something else, only this file changes.

Motion Detection Strategy:
  Uses ffmpeg's `select=gt(scene,threshold)` filter to find frames where the
  scene changes significantly. Fast, CPU-only, no ML models needed.
"""
import json
import re
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

from ..domain.entities.clip import MotionWindow
from ..domain.ports.ports import IMotionDetectionPort, IStreamRecorderPort
from ...shared.logger import get_logger

logger = get_logger("surveillance.adapters.ffmpeg")


class FfmpegStreamRecorderAdapter(IStreamRecorderPort):

    def record_stream(
        self,
        rtsp_url: str,
        output_path: Path,
        duration_seconds: int,
    ) -> Path:
        """
        Pull from RTSP and save to file.
        Uses sub-stream (already configured in URL) to save bandwidth.
        Re-encodes to H.264 with CRF 28 — good quality / small size balance.
        """
        logger.info(f"Recording {duration_seconds}s from RTSP → {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            "ffmpeg",
            "-y",                           # overwrite output
            "-rtsp_transport", "tcp",        # more reliable than UDP on home WiFi
            "-i", rtsp_url,
            "-t", str(duration_seconds),
            "-c:v", "libx264",
            "-crf", "28",                   # quality: 18=best, 28=good/small
            "-preset", "veryfast",
            "-c:a", "aac",
            "-b:a", "64k",
            "-movflags", "+faststart",      # web-friendly mp4
            str(output_path),
        ]

        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=duration_seconds + 60
        )

        if result.returncode != 0:
            logger.error(f"ffmpeg record failed:\n{result.stderr[-2000:]}")
            raise RuntimeError(f"Stream recording failed for {output_path}")

        logger.info(
            f"Recording saved: {output_path} "
            f"({output_path.stat().st_size / 1024 / 1024:.1f} MB)"
        )
        return output_path

    def extract_segment(
        self,
        source_path: Path,
        output_path: Path,
        start_seconds: float,
        duration_seconds: float,
    ) -> Path:
        """
        Cut a segment from an existing recording.
        Uses -ss before -i (fast seek) + -t for duration.
        Stream copy (no re-encode) for speed — clips are already compressed.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            "ffmpeg",
            "-y",
            "-ss", str(start_seconds),
            "-i", str(source_path),
            "-t", str(duration_seconds),
            "-c", "copy",               # stream copy: fast, no quality loss
            "-avoid_negative_ts", "make_zero",
            str(output_path),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        if result.returncode != 0:
            logger.error(f"ffmpeg extract failed:\n{result.stderr[-1000:]}")
            raise RuntimeError(f"Segment extraction failed: {output_path}")

        return output_path


class FfmpegMotionDetectionAdapter(IMotionDetectionPort):

    def detect_motion_windows(
        self,
        recording_path: Path,
        scene_threshold: float,
    ) -> List[MotionWindow]:
        """
        Uses ffmpeg scene detection filter to find timestamps where
        the scene changes significantly.

        Returns MotionWindow list ordered by start_seconds.

        How it works:
        1. ffmpeg reads the video, applies `select=gt(scene,N)` filter
        2. For each scene-change frame, ffmpeg logs the timestamp + score
        3. We parse those timestamps → build MotionWindows (start → start+30s buffer)
        4. Domain service will merge overlapping windows
        """
        logger.info(f"Detecting motion in: {recording_path} (threshold={scene_threshold})")

        duration = self._get_duration(recording_path)
        if duration is None:
            logger.warning("Could not determine video duration")
            duration = 86400.0  # assume 24h if unknown

        scene_timestamps = self._run_scene_detection(recording_path, scene_threshold)
        logger.info(f"Scene change timestamps found: {len(scene_timestamps)}")

        if not scene_timestamps:
            logger.info("No motion detected in recording")
            return []

        # Each scene change starts a window; extend it forward 30s.
        # The domain service will merge adjacent ones.
        buffer_seconds = 30.0
        windows: List[MotionWindow] = []

        for ts, score in scene_timestamps:
            end = min(ts + buffer_seconds, duration)
            windows.append(MotionWindow(
                start_seconds=max(0.0, ts - 2.0),  # 2s look-back
                end_seconds=end,
                confidence=min(score, 1.0),
            ))

        return sorted(windows, key=lambda w: w.start_seconds)

    def _get_duration(self, path: Path) -> Optional[float]:
        """Get video duration in seconds using ffprobe."""
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-show_entries", "format=duration",
            "-of", "json",
            str(path),
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            data = json.loads(result.stdout)
            return float(data["format"]["duration"])
        except Exception as e:
            logger.warning(f"ffprobe duration failed: {e}")
            return None

    def _run_scene_detection(
        self, path: Path, threshold: float
    ) -> List[Tuple[float, float]]:
        """
        Returns list of (timestamp_seconds, scene_score) for each scene change.
        """
        cmd = [
            "ffmpeg",
            "-i", str(path),
            "-vf", f"select=gt(scene\\,{threshold}),showinfo",
            "-vsync", "vfr",
            "-f", "null",
            "-",
        ]

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=3600  # up to 1h for long recordings
            )
            # ffmpeg writes showinfo to stderr
            return self._parse_showinfo(result.stderr)
        except subprocess.TimeoutExpired:
            logger.error("Scene detection timed out")
            return []
        except Exception as e:
            logger.error(f"Scene detection failed: {e}")
            return []

    def _parse_showinfo(self, stderr_output: str) -> List[Tuple[float, float]]:
        """
        Parse ffmpeg showinfo output to extract timestamps.
        Line format: [Parsed_showinfo_1 @ ...] n:X pts:X pts_time:Y ...
        """
        timestamps: List[Tuple[float, float]] = []
        pts_pattern = re.compile(r"pts_time:([\d.]+)")
        scene_pattern = re.compile(r"lavfi\.scene_score=([\d.]+)")

        for line in stderr_output.split("\n"):
            if "pts_time" not in line:
                continue
            ts_match = pts_pattern.search(line)
            if not ts_match:
                continue
            ts = float(ts_match.group(1))
            score_match = scene_pattern.search(line)
            score = float(score_match.group(1)) if score_match else 0.5
            timestamps.append((ts, score))

        return timestamps
