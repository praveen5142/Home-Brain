"""
Tests for WhisperTranscriptionAdapter.

Mocks faster_whisper.WhisperModel and ffmpeg subprocess. No real files or GPU needed.
Negative tests first, then positive.
"""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from home_brain.intelligence.domain.entities.observation import Transcript
from home_brain.intelligence.adapters.whisper_transcription_adapter import (
    WhisperTranscriptionAdapter,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_adapter(model_size: str = "small") -> WhisperTranscriptionAdapter:
    return WhisperTranscriptionAdapter(model_size=model_size)


def _fake_whisper_segments(text: str = "Hello there.", language: str = "en"):
    """Return a (segments_iterable, info) tuple matching faster-whisper's API."""
    segment = MagicMock()
    segment.text = text
    info = MagicMock()
    info.language = language
    return [segment], info


def _fake_run_success(cmd, **kwargs):
    """Simulate a successful ffmpeg audio extraction."""
    result = MagicMock()
    result.returncode = 0
    result.stderr = b""
    # Create the output wav file so the adapter can proceed
    for arg in cmd:
        if str(arg).endswith(".wav"):
            Path(arg).write_bytes(b"RIFF fake wav")
            break
    return result


def _fake_run_failure(cmd, **kwargs):
    result = MagicMock()
    result.returncode = 1
    result.stderr = b"error: No such file"
    return result


# ---------------------------------------------------------------------------
# NEGATIVE TESTS
# ---------------------------------------------------------------------------


class TestWhisperTranscriptionAdapterNegative:
    def test_clip_not_found_raises_file_not_found(self, tmp_path):
        adapter = _make_adapter()
        missing = tmp_path / "nonexistent.mp4"
        with pytest.raises(FileNotFoundError):
            adapter.transcribe(missing, "clip-1")

    def test_ffmpeg_audio_extraction_failure_raises_runtime_error(self, tmp_path):
        clip = tmp_path / "clip.mp4"
        clip.write_bytes(b"fake video")
        adapter = _make_adapter()
        with patch("subprocess.run", side_effect=_fake_run_failure):
            with pytest.raises(RuntimeError, match="[Aa]udio"):
                adapter.transcribe(clip, "clip-1")

    def test_empty_segments_returns_transcript_with_empty_text(self, tmp_path):
        clip = tmp_path / "clip.mp4"
        clip.write_bytes(b"fake video")
        adapter = _make_adapter()

        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([], MagicMock(language="en"))

        with patch("subprocess.run", side_effect=_fake_run_success):
            with patch.object(adapter, "_load_model", return_value=mock_model):
                transcript = adapter.transcribe(clip, "clip-1")

        assert transcript.text == ""


# ---------------------------------------------------------------------------
# POSITIVE TESTS
# ---------------------------------------------------------------------------


class TestWhisperTranscriptionAdapterPositive:
    def test_valid_clip_returns_transcript_with_correct_fields(self, tmp_path):
        clip = tmp_path / "clip.mp4"
        clip.write_bytes(b"fake video")
        adapter = _make_adapter()
        segments, info = _fake_whisper_segments("Hello there.", "en")
        mock_model = MagicMock()
        mock_model.transcribe.return_value = (segments, info)

        with patch("subprocess.run", side_effect=_fake_run_success):
            with patch.object(adapter, "_load_model", return_value=mock_model):
                tr = adapter.transcribe(clip, "clip-99")

        assert isinstance(tr, Transcript)
        assert tr.clip_id == "clip-99"
        assert tr.text == "Hello there."
        assert tr.language == "en"

    def test_audio_temp_file_cleaned_up_after_transcription(self, tmp_path):
        clip = tmp_path / "clip.mp4"
        clip.write_bytes(b"fake video")
        adapter = _make_adapter()
        segments, info = _fake_whisper_segments()
        mock_model = MagicMock()
        mock_model.transcribe.return_value = (segments, info)
        created_wav_paths = []

        def tracking_run(cmd, **kwargs):
            result = _fake_run_success(cmd, **kwargs)
            for arg in cmd:
                if str(arg).endswith(".wav"):
                    created_wav_paths.append(Path(arg))
            return result

        with patch("subprocess.run", side_effect=tracking_run):
            with patch.object(adapter, "_load_model", return_value=mock_model):
                adapter.transcribe(clip, "clip-1")

        for wav in created_wav_paths:
            assert not wav.exists(), f"Temp WAV not cleaned up: {wav}"

    def test_language_detected_from_whisper_output(self, tmp_path):
        clip = tmp_path / "clip.mp4"
        clip.write_bytes(b"fake video")
        adapter = _make_adapter()
        segments, info = _fake_whisper_segments("Namaste.", "hi")
        mock_model = MagicMock()
        mock_model.transcribe.return_value = (segments, info)

        with patch("subprocess.run", side_effect=_fake_run_success):
            with patch.object(adapter, "_load_model", return_value=mock_model):
                tr = adapter.transcribe(clip, "clip-1")

        assert tr.language == "hi"

    def test_model_not_loaded_on_init(self, tmp_path):
        """WhisperModel must be lazy — not instantiated on __init__."""
        with patch(
            "home_brain.intelligence.adapters.whisper_transcription_adapter.WhisperModel"
        ) as mock_cls:
            adapter = _make_adapter()
            mock_cls.assert_not_called()

    def test_multiple_segments_concatenated(self, tmp_path):
        clip = tmp_path / "clip.mp4"
        clip.write_bytes(b"fake video")
        adapter = _make_adapter()

        seg1 = MagicMock()
        seg1.text = "Hello"
        seg2 = MagicMock()
        seg2.text = " world."
        info = MagicMock()
        info.language = "en"
        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([seg1, seg2], info)

        with patch("subprocess.run", side_effect=_fake_run_success):
            with patch.object(adapter, "_load_model", return_value=mock_model):
                tr = adapter.transcribe(clip, "clip-1")

        assert tr.text == "Hello world."
