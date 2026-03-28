"""
Tests for ClaudeVisionAdapter.

Mocks the anthropic SDK and ffmpeg subprocess calls so no real API or video files are needed.
Negative tests first, then positive.
"""

import json
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from home_brain.intelligence.domain.entities.observation import Mood, Observation
from home_brain.intelligence.adapters.claude_vision_adapter import ClaudeVisionAdapter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _valid_response_payload(
    mood="neutral",
    activity="person walking",
    description="A person walks through the hallway.",
    confidence=0.9,
) -> str:
    return json.dumps({
        "mood": mood,
        "activity": activity,
        "description": description,
        "confidence": confidence,
    })


def _make_adapter(api_key: str = "test-key") -> ClaudeVisionAdapter:
    return ClaudeVisionAdapter(api_key=api_key)


def _mock_anthropic_response(text: str) -> MagicMock:
    msg = MagicMock()
    msg.content = [MagicMock(text=text)]
    msg.usage.input_tokens = 100
    msg.usage.output_tokens = 50
    return msg


# ---------------------------------------------------------------------------
# NEGATIVE TESTS
# ---------------------------------------------------------------------------


class TestClaudeVisionAdapterNegative:
    def test_clip_path_not_found_raises_before_api_call(self, tmp_path):
        adapter = _make_adapter()
        missing = tmp_path / "nonexistent.mp4"
        with patch("home_brain.intelligence.adapters.claude_vision_adapter.anthropic") as mock_ant:
            with pytest.raises(FileNotFoundError):
                adapter.analyse(missing, "clip-1")
            mock_ant.Anthropic.assert_not_called()

    def _setup_clip_and_frame(self, tmp_path):
        clip = tmp_path / "clip.mp4"
        clip.write_bytes(b"fake video")
        frame = tmp_path / "frame.jpg"
        frame.write_bytes(b"fake jpeg")
        return clip, frame

    def test_malformed_json_response_raises_value_error(self, tmp_path):
        clip, frame = self._setup_clip_and_frame(tmp_path)
        adapter = _make_adapter()
        with patch.object(adapter, "_extract_frames", return_value=[frame]):
            with patch("home_brain.intelligence.adapters.claude_vision_adapter.anthropic") as mock_ant:
                client = mock_ant.Anthropic.return_value
                client.messages.create.return_value = _mock_anthropic_response("not valid json {{{")
                with pytest.raises(ValueError, match="[Jj][Ss][Oo][Nn]"):
                    adapter.analyse(clip, "clip-1")

    def test_missing_mood_key_raises_value_error(self, tmp_path):
        clip, frame = self._setup_clip_and_frame(tmp_path)
        adapter = _make_adapter()
        payload = json.dumps({"activity": "walking", "description": "desc", "confidence": 0.8})
        with patch.object(adapter, "_extract_frames", return_value=[frame]):
            with patch("home_brain.intelligence.adapters.claude_vision_adapter.anthropic") as mock_ant:
                client = mock_ant.Anthropic.return_value
                client.messages.create.return_value = _mock_anthropic_response(payload)
                with pytest.raises(ValueError, match="mood"):
                    adapter.analyse(clip, "clip-1")

    def test_invalid_mood_value_raises_value_error(self, tmp_path):
        clip, frame = self._setup_clip_and_frame(tmp_path)
        adapter = _make_adapter()
        payload = _valid_response_payload(mood="excited")
        with patch.object(adapter, "_extract_frames", return_value=[frame]):
            with patch("home_brain.intelligence.adapters.claude_vision_adapter.anthropic") as mock_ant:
                client = mock_ant.Anthropic.return_value
                client.messages.create.return_value = _mock_anthropic_response(payload)
                with pytest.raises(ValueError):
                    adapter.analyse(clip, "clip-1")

    def test_confidence_above_one_in_response_raises_value_error(self, tmp_path):
        clip, frame = self._setup_clip_and_frame(tmp_path)
        adapter = _make_adapter()
        payload = _valid_response_payload(confidence=1.5)
        with patch.object(adapter, "_extract_frames", return_value=[frame]):
            with patch("home_brain.intelligence.adapters.claude_vision_adapter.anthropic") as mock_ant:
                client = mock_ant.Anthropic.return_value
                client.messages.create.return_value = _mock_anthropic_response(payload)
                with pytest.raises(ValueError):
                    adapter.analyse(clip, "clip-1")

    def test_confidence_below_zero_in_response_raises_value_error(self, tmp_path):
        clip, frame = self._setup_clip_and_frame(tmp_path)
        adapter = _make_adapter()
        payload = _valid_response_payload(confidence=-0.1)
        with patch.object(adapter, "_extract_frames", return_value=[frame]):
            with patch("home_brain.intelligence.adapters.claude_vision_adapter.anthropic") as mock_ant:
                client = mock_ant.Anthropic.return_value
                client.messages.create.return_value = _mock_anthropic_response(payload)
                with pytest.raises(ValueError):
                    adapter.analyse(clip, "clip-1")


# ---------------------------------------------------------------------------
# POSITIVE TESTS
# ---------------------------------------------------------------------------


class TestClaudeVisionAdapterPositive:
    def test_valid_response_returns_observation(self, tmp_path):
        clip = tmp_path / "clip.mp4"
        clip.write_bytes(b"fake video")
        adapter = _make_adapter()
        payload = _valid_response_payload(
            mood="happy", activity="child playing", description="Kids playing.", confidence=0.95
        )
        frame = tmp_path / "frame_0.jpg"
        frame.write_bytes(b"fake jpeg")
        with patch.object(adapter, "_extract_frames", return_value=[frame]):
            with patch("home_brain.intelligence.adapters.claude_vision_adapter.anthropic") as mock_ant:
                client = mock_ant.Anthropic.return_value
                client.messages.create.return_value = _mock_anthropic_response(payload)
                obs = adapter.analyse(clip, "clip-42")
        assert isinstance(obs, Observation)
        assert obs.clip_id == "clip-42"
        assert obs.mood == Mood.HAPPY
        assert obs.activity == "child playing"
        assert obs.confidence == pytest.approx(0.95)

    def test_frames_cleaned_up_after_analysis(self, tmp_path):
        clip = tmp_path / "clip.mp4"
        clip.write_bytes(b"fake video")
        adapter = _make_adapter()
        payload = _valid_response_payload()
        frame = tmp_path / "frame_0.jpg"
        frame.write_bytes(b"fake jpeg")
        with patch.object(adapter, "_extract_frames", return_value=[frame]):
            with patch("home_brain.intelligence.adapters.claude_vision_adapter.anthropic") as mock_ant:
                client = mock_ant.Anthropic.return_value
                client.messages.create.return_value = _mock_anthropic_response(payload)
                adapter.analyse(clip, "clip-1")
        assert not frame.exists()

    def test_token_usage_logged(self, tmp_path, caplog):
        import logging
        clip = tmp_path / "clip.mp4"
        clip.write_bytes(b"fake video")
        adapter = _make_adapter()
        payload = _valid_response_payload()
        frame = tmp_path / "frame_0.jpg"
        frame.write_bytes(b"fake jpeg")
        with patch.object(adapter, "_extract_frames", return_value=[frame]):
            with patch("home_brain.intelligence.adapters.claude_vision_adapter.anthropic") as mock_ant:
                client = mock_ant.Anthropic.return_value
                client.messages.create.return_value = _mock_anthropic_response(payload)
                with caplog.at_level(logging.DEBUG, logger="intelligence.ClaudeVisionAdapter"):
                    adapter.analyse(clip, "clip-1")
        assert any("token" in record.message.lower() for record in caplog.records)

    def test_uses_haiku_model(self, tmp_path):
        clip = tmp_path / "clip.mp4"
        clip.write_bytes(b"fake video")
        adapter = _make_adapter()
        payload = _valid_response_payload()
        frame = tmp_path / "frame_0.jpg"
        frame.write_bytes(b"fake jpeg")
        with patch.object(adapter, "_extract_frames", return_value=[frame]):
            with patch("home_brain.intelligence.adapters.claude_vision_adapter.anthropic") as mock_ant:
                client = mock_ant.Anthropic.return_value
                client.messages.create.return_value = _mock_anthropic_response(payload)
                adapter.analyse(clip, "clip-1")
        create_call_kwargs = client.messages.create.call_args
        assert "claude-haiku" in create_call_kwargs.kwargs.get("model", "")

    def test_extract_frames_calls_ffmpeg(self, tmp_path):
        clip = tmp_path / "clip.mp4"
        clip.write_bytes(b"fake video")
        adapter = _make_adapter()
        fake_frame = tmp_path / "frame_000001.jpg"
        fake_frame.write_bytes(b"fake jpeg")

        def fake_run(cmd, **kwargs):
            # Create a fake frame file when ffmpeg is "called"
            for arg in cmd:
                if arg.endswith(".jpg") or "%06d" in str(arg):
                    # Create the expected output
                    fake_frame.write_bytes(b"fake jpeg")
            result = MagicMock()
            result.returncode = 0
            return result

        with patch("subprocess.run", side_effect=fake_run):
            frames = adapter._extract_frames(clip)
        assert isinstance(frames, list)
