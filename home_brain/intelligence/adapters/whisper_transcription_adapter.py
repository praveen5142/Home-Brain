"""
Intelligence Adapters — Whisper Transcription Adapter.

What it is: Implements ITranscriptionPort using faster-whisper (local CPU inference).
What it knows: faster_whisper API, ffmpeg audio extraction, temp file management.
What it doesn't know: SQLite, Claude API, domain rules, Telegram, Backblaze.

Model is lazy-loaded on first transcribe() call to avoid startup overhead.
Audio is extracted to a temporary WAV file (16kHz, mono) before transcription.
"""

import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Optional

try:
    from faster_whisper import WhisperModel
except ImportError:  # pragma: no cover
    WhisperModel = None  # type: ignore

from home_brain.intelligence.domain.entities.observation import Transcript
from home_brain.intelligence.domain.ports.ports import ITranscriptionPort
from home_brain.shared.logger import get_logger

logger = get_logger("intelligence.WhisperTranscriptionAdapter")


class WhisperTranscriptionAdapter(ITranscriptionPort):
    """Transcribes audio from a Clip file using faster-whisper running on CPU."""

    def __init__(self, model_size: str = "small") -> None:
        self._model_size = model_size
        self._model: Optional[object] = None  # lazy-loaded

    def transcribe(self, clip_path: Path, clip_id: str) -> Transcript:
        """
        Extract audio from clip_path, run Whisper, return a Transcript.

        Raises:
            FileNotFoundError: if clip_path does not exist.
            RuntimeError: if ffmpeg audio extraction fails.
        """
        if not clip_path.exists():
            raise FileNotFoundError(f"Clip file not found: {clip_path}")

        audio_path: Optional[Path] = None
        try:
            audio_path = self._extract_audio(clip_path)
            model = self._load_model()
            segments, info = model.transcribe(str(audio_path), beam_size=5)
            text = "".join(seg.text for seg in segments)
            language = info.language
        finally:
            if audio_path and audio_path.exists():
                try:
                    audio_path.unlink()
                except OSError:
                    pass

        logger.debug(
            f"Transcribed clip {clip_id}: language={language}, "
            f"chars={len(text)}"
        )
        return Transcript.create(clip_id=clip_id, text=text, language=language)

    def _extract_audio(self, clip_path: Path) -> Path:
        """
        Use ffmpeg to extract audio as 16kHz mono WAV to a temp file.
        Raises RuntimeError if ffmpeg exits with non-zero.
        """
        tmp_file = Path(tempfile.mktemp(suffix=".wav", prefix="hb_audio_"))
        cmd = [
            "ffmpeg", "-y",
            "-i", str(clip_path),
            "-ar", "16000",
            "-ac", "1",
            "-f", "wav",
            str(tmp_file),
        ]
        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=120,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Audio extraction failed for {clip_path} "
                f"(exit={result.returncode}): {result.stderr.decode(errors='replace')[:200]}"
            )
        return tmp_file

    def _load_model(self):
        """Lazy-load the WhisperModel on first call."""
        if self._model is None:
            logger.info(f"Loading Whisper model '{self._model_size}' (CPU)…")
            self._model = WhisperModel(self._model_size, device="cpu", compute_type="int8")
            logger.info("Whisper model loaded.")
        return self._model
