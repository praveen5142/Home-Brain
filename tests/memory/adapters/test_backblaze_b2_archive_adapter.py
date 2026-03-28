"""
Tests for BackblazeB2ArchiveAdapter — IArchivePort.

Uses unittest.mock to patch boto3. No real B2 API calls.
Negative tests first, then positive.
"""

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest
from botocore.exceptions import ClientError

from home_brain.memory.adapters.backblaze_b2_archive_adapter import (
    BackblazeB2ArchiveAdapter,
)
from home_brain.surveillance.domain.entities.clip import Clip, ClipStatus


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


def _make_adapter() -> BackblazeB2ArchiveAdapter:
    return BackblazeB2ArchiveAdapter(
        key_id="test-key-id",
        application_key="test-app-key",
        bucket_name="home-brain-archive",
        endpoint_url="https://s3.us-west-004.backblazeb2.com",
    )


def _make_clip(file_path: Path) -> Clip:
    return Clip(
        id="clip-abc123",
        recorded_at=datetime(2026, 3, 28, 10, 0, 0),
        duration_seconds=30.0,
        file_path=file_path,
        status=ClipStatus.ANALYSED,
    )


# ---------------------------------------------------------------------------
# NEGATIVE TESTS
# ---------------------------------------------------------------------------


class TestBackblazeB2ArchiveAdapterNegative:
    def test_empty_key_id_raises_value_error(self):
        with pytest.raises(ValueError, match="key_id"):
            BackblazeB2ArchiveAdapter(
                key_id="",
                application_key="app-key",
                bucket_name="bucket",
                endpoint_url="https://s3.example.com",
            )

    def test_empty_application_key_raises_value_error(self):
        with pytest.raises(ValueError, match="application_key"):
            BackblazeB2ArchiveAdapter(
                key_id="key-id",
                application_key="",
                bucket_name="bucket",
                endpoint_url="https://s3.example.com",
            )

    def test_empty_bucket_name_raises_value_error(self):
        with pytest.raises(ValueError, match="bucket_name"):
            BackblazeB2ArchiveAdapter(
                key_id="key-id",
                application_key="app-key",
                bucket_name="",
                endpoint_url="https://s3.example.com",
            )

    def test_missing_file_raises_file_not_found_error(self, tmp_path):
        missing = tmp_path / "ghost.mp4"
        clip = _make_clip(missing)
        adapter = _make_adapter()

        with patch("home_brain.memory.adapters.backblaze_b2_archive_adapter.boto3"):
            with pytest.raises(FileNotFoundError):
                adapter.upload(clip)

    def test_boto3_client_error_raises_runtime_error(self, tmp_path):
        clip_path = tmp_path / "clip.mp4"
        clip_path.write_bytes(b"fake video")
        clip = _make_clip(clip_path)
        adapter = _make_adapter()

        error_response = {"Error": {"Code": "NoSuchBucket", "Message": "Bucket not found"}}
        with patch(
            "home_brain.memory.adapters.backblaze_b2_archive_adapter.boto3"
        ) as mock_boto:
            mock_client = MagicMock()
            mock_boto.client.return_value = mock_client
            mock_client.upload_file.side_effect = ClientError(error_response, "upload_file")

            with pytest.raises(RuntimeError, match="clip-abc123"):
                adapter.upload(clip)


# ---------------------------------------------------------------------------
# POSITIVE TESTS
# ---------------------------------------------------------------------------


class TestBackblazeB2ArchiveAdapterPositive:
    def test_upload_returns_correct_archive_url(self, tmp_path):
        clip_path = tmp_path / "clip.mp4"
        clip_path.write_bytes(b"fake video")
        clip = _make_clip(clip_path)
        adapter = _make_adapter()

        with patch(
            "home_brain.memory.adapters.backblaze_b2_archive_adapter.boto3"
        ) as mock_boto:
            mock_client = MagicMock()
            mock_boto.client.return_value = mock_client

            url = adapter.upload(clip)

        assert url.startswith("https://s3.us-west-004.backblazeb2.com/home-brain-archive/")
        assert "clip-abc123" in url

    def test_upload_file_called_with_correct_s3_key(self, tmp_path):
        clip_path = tmp_path / "clip.mp4"
        clip_path.write_bytes(b"fake video")
        clip = _make_clip(clip_path)
        adapter = _make_adapter()

        with patch(
            "home_brain.memory.adapters.backblaze_b2_archive_adapter.boto3"
        ) as mock_boto:
            mock_client = MagicMock()
            mock_boto.client.return_value = mock_client

            adapter.upload(clip)

        upload_call = mock_client.upload_file.call_args
        # Positional: (local_path, bucket, key)
        key = upload_call[0][2] if upload_call[0] else upload_call[1]["Key"]
        assert "2026-03-28" in key
        assert "clip-abc123" in key
        assert key.startswith("clips/")

    def test_boto3_client_created_with_correct_credentials(self, tmp_path):
        clip_path = tmp_path / "clip.mp4"
        clip_path.write_bytes(b"fake video")
        clip = _make_clip(clip_path)
        adapter = _make_adapter()

        with patch(
            "home_brain.memory.adapters.backblaze_b2_archive_adapter.boto3"
        ) as mock_boto:
            mock_client = MagicMock()
            mock_boto.client.return_value = mock_client

            adapter.upload(clip)

        mock_boto.client.assert_called_once()
        kwargs = mock_boto.client.call_args[1]
        assert kwargs["aws_access_key_id"] == "test-key-id"
        assert kwargs["aws_secret_access_key"] == "test-app-key"
        assert kwargs["endpoint_url"] == "https://s3.us-west-004.backblazeb2.com"

    def test_client_created_lazily_not_on_init(self):
        """boto3.client should NOT be called during __init__."""
        with patch(
            "home_brain.memory.adapters.backblaze_b2_archive_adapter.boto3"
        ) as mock_boto:
            _make_adapter()
            mock_boto.client.assert_not_called()
