"""
Intelligence Adapters — Claude Vision Adapter.

What it is: Implements IVideoAnalysisPort using Claude Haiku (per-clip) for frame analysis.
What it knows: anthropic SDK, ffmpeg frame extraction, base64 encoding, JSON parsing.
What it doesn't know: SQLite, Whisper, Telegram, domain rules beyond what Observation.create
                      enforces.

Cost note: uses claude-haiku-4-5-20251001 (~₹2–3/clip). Token counts are logged on every call.
"""

import base64
import json
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import List

try:
    import anthropic
except ImportError:  # pragma: no cover
    anthropic = None  # type: ignore

from home_brain.intelligence.domain.entities.observation import Mood, Observation
from home_brain.intelligence.domain.ports.ports import IVideoAnalysisPort
from home_brain.shared.logger import get_logger

logger = get_logger("intelligence.ClaudeVisionAdapter")

_HAIKU_MODEL = "claude-haiku-4-5-20251001"

_ANALYSIS_PROMPT = """\
You are analysing frames from a home security camera clip.
Look at all the provided images and describe what is happening.

Return ONLY valid JSON with exactly these keys:
{
  "mood": "<one of: happy, sad, angry, neutral, calm>",
  "activity": "<one concise line describing what is happening>",
  "description": "<2-3 sentence narrative description of the scene>",
  "confidence": <float between 0.0 and 1.0>
}

Do not include any text outside the JSON object."""


class ClaudeVisionAdapter(IVideoAnalysisPort):
    """Analyses a Clip by extracting key frames and sending them to Claude Haiku."""

    def __init__(self, api_key: str, max_frames: int = 5) -> None:
        self._api_key = api_key
        self._max_frames = max_frames

    def analyse(self, clip_path: Path, clip_id: str) -> Observation:
        """
        Extract frames from clip_path, send to Claude Haiku, parse response → Observation.

        Raises:
            FileNotFoundError: if clip_path does not exist.
            ValueError: if Claude returns invalid/unexpected JSON.
        """
        if not clip_path.exists():
            raise FileNotFoundError(f"Clip file not found: {clip_path}")

        frames: List[Path] = []
        try:
            frames = self._extract_frames(clip_path)
            raw = self._call_claude(frames, clip_id)
            obs = self._parse_response(raw, clip_id)
        finally:
            for f in frames:
                try:
                    f.unlink(missing_ok=True)
                except OSError:
                    pass

        return obs

    def _extract_frames(self, clip_path: Path) -> List[Path]:
        """
        Extract up to max_frames evenly-spaced frames from the clip using ffmpeg.
        Frames are written to a temporary directory as JPEG files.
        Returns a list of Paths to the extracted frame files.
        """
        tmp_dir = Path(tempfile.mkdtemp(prefix="hb_frames_"))
        output_pattern = str(tmp_dir / "frame_%06d.jpg")

        cmd = [
            "ffmpeg", "-y",
            "-i", str(clip_path),
            "-vf", f"select='not(mod(n,{max(1, self._max_frames)}))',scale=512:-1",
            "-vsync", "vfr",
            "-frames:v", str(self._max_frames),
            output_pattern,
        ]
        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=60,
        )
        if result.returncode != 0:
            logger.warning(f"ffmpeg frame extraction returned non-zero for {clip_path}")

        frames = sorted(tmp_dir.glob("frame_*.jpg"))
        if not frames:
            logger.warning(f"No frames extracted from {clip_path}, using placeholder approach.")
        return frames[:self._max_frames]

    def _call_claude(self, frames: List[Path], clip_id: str) -> dict:
        """Send frames to Claude Haiku and return the parsed JSON dict."""
        client = anthropic.Anthropic(api_key=self._api_key)

        image_content = []
        for frame_path in frames:
            data = base64.standard_b64encode(frame_path.read_bytes()).decode("utf-8")
            image_content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": data,
                },
            })

        image_content.append({"type": "text", "text": _ANALYSIS_PROMPT})

        response = client.messages.create(
            model=_HAIKU_MODEL,
            max_tokens=512,
            messages=[{"role": "user", "content": image_content}],
        )

        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        logger.debug(
            f"Claude Haiku tokens for clip {clip_id}: "
            f"input={input_tokens}, output={output_tokens}"
        )

        raw_text = response.content[0].text.strip()
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Claude returned invalid JSON for clip {clip_id}: {exc}\nRaw: {raw_text[:200]}"
            )

    def _parse_response(self, data: dict, clip_id: str) -> Observation:
        """Validate the JSON dict from Claude and construct an Observation."""
        required_keys = {"mood", "activity", "description", "confidence"}
        missing = required_keys - data.keys()
        if missing:
            raise ValueError(
                f"Claude response missing required keys {missing} for clip {clip_id}. "
                f"Got keys: {list(data.keys())}"
            )

        # Observation.create validates mood enum, confidence range, empty strings
        return Observation.create(
            clip_id=clip_id,
            mood=data["mood"],
            activity=data["activity"],
            description=data["description"],
            confidence=data["confidence"],
        )
