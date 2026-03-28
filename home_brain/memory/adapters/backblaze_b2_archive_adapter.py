"""
Memory Adapters — Backblaze B2 Archive.

What it is: Outbound archive adapter that uploads Clip files to Backblaze B2
            using the S3-compatible API via boto3.
What it knows: boto3 S3 client API, B2 endpoint, S3 key naming convention.
What it doesn't know: domain rules, SQLite, Telegram, ffmpeg, or any other adapter.

The boto3 client is created lazily on the first upload() call (not on __init__)
so that unit tests can patch boto3 without the constructor triggering any API calls.
"""

from pathlib import Path

import boto3
from botocore.exceptions import ClientError

from home_brain.memory.domain.ports.ports import IArchivePort
from home_brain.shared.logger import get_logger
from home_brain.surveillance.domain.entities.clip import Clip

logger = get_logger("memory.adapters.backblaze_b2")


class BackblazeB2ArchiveAdapter(IArchivePort):
    """
    Uploads clips to Backblaze B2 using the S3-compatible endpoint.

    S3 key format: clips/YYYY-MM-DD/{clip_id}/{filename}.mp4
    Archive URL:   {endpoint_url}/{bucket_name}/{key}
    """

    def __init__(
        self,
        key_id: str,
        application_key: str,
        bucket_name: str,
        endpoint_url: str,
    ) -> None:
        if not key_id or not key_id.strip():
            raise ValueError("key_id must not be empty")
        if not application_key or not application_key.strip():
            raise ValueError("application_key must not be empty")
        if not bucket_name or not bucket_name.strip():
            raise ValueError("bucket_name must not be empty")
        self._key_id = key_id
        self._application_key = application_key
        self._bucket_name = bucket_name
        self._endpoint_url = endpoint_url.rstrip("/")
        self._client = None  # lazy

    def _get_client(self):
        if self._client is None:
            self._client = boto3.client(
                "s3",
                endpoint_url=self._endpoint_url,
                aws_access_key_id=self._key_id,
                aws_secret_access_key=self._application_key,
            )
        return self._client

    def upload(self, clip: Clip) -> str:
        if not clip.file_path.exists():
            raise FileNotFoundError(
                f"Clip file not found: {clip.file_path} (clip_id={clip.id})"
            )

        date_str = clip.recorded_at.strftime("%Y-%m-%d")
        key = f"clips/{date_str}/{clip.id}/{clip.file_path.name}"

        client = self._get_client()
        try:
            client.upload_file(str(clip.file_path), self._bucket_name, key)
        except ClientError as exc:
            raise RuntimeError(
                f"Failed to upload clip {clip.id} to B2: {exc}"
            ) from exc

        archive_url = f"{self._endpoint_url}/{self._bucket_name}/{key}"
        logger.info(f"Archived clip {clip.id} → {archive_url}")
        return archive_url
