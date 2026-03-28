"""
Shared configuration for Home Brain.

What it is: single source of all environment-driven config for the whole app.
What it knows: env var names, defaults, dataclass structure.
What it doesn't know: adapters, domains, ffmpeg, sqlite — nothing infrastructure-specific.

Setup: copy .env.example → .env and fill in your values.
"""
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class CameraConfig:
    ip: str
    port: int
    username: str
    password: str
    channel: int
    subtype: int  # 0=main stream (HD), 1=sub stream (low res, recommended)

    @property
    def rtsp_url(self) -> str:
        """
        CP Plus E44Q RTSP URL format.
        subtype=1 is the sub-stream (~360p) — saves disk, enough for analysis.
        """
        return (
            f"rtsp://{self.username}:{self.password}@{self.ip}:{self.port}"
            f"/cam/realmonitor?channel={self.channel}&subtype={self.subtype}"
        )


@dataclass(frozen=True)
class StorageConfig:
    base_dir: Path
    clips_dir: Path
    db_path: Path
    retention_days: int

    @classmethod
    def default(cls) -> "StorageConfig":
        base = Path(os.getenv("DATA_DIR", "./data"))
        return cls(
            base_dir=base,
            clips_dir=base / "clips",
            db_path=base / "home_brain.sqlite",
            retention_days=int(os.getenv("RETENTION_DAYS", "30")),
        )


@dataclass(frozen=True)
class MotionConfig:
    """
    Controls how aggressively we detect motion.
    scene_threshold: 0.0–1.0. Lower = more sensitive.
    min_clip_duration_s: ignore micro-motion under this.
    max_clip_duration_s: split long activity into chunks.
    merge_gap_s: if two windows are this close, merge them.
    """
    scene_threshold: float
    min_clip_duration_s: int
    max_clip_duration_s: int
    merge_gap_s: int

    @classmethod
    def default(cls) -> "MotionConfig":
        return cls(
            scene_threshold=float(os.getenv("SCENE_THRESHOLD", "0.3")),
            min_clip_duration_s=int(os.getenv("MIN_CLIP_DURATION", "5")),
            max_clip_duration_s=int(os.getenv("MAX_CLIP_DURATION", "120")),
            merge_gap_s=int(os.getenv("MERGE_GAP_S", "10")),
        )


@dataclass(frozen=True)
class IntelligenceConfig:
    """Phase 2: settings for Claude Vision and Whisper transcription."""
    anthropic_api_key: str
    whisper_model: str
    frame_extraction_fps: int

    @classmethod
    def default(cls) -> "IntelligenceConfig":
        return cls(
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
            whisper_model=os.getenv("WHISPER_MODEL", "small"),
            frame_extraction_fps=int(os.getenv("FRAME_EXTRACTION_FPS", "1")),
        )


@dataclass(frozen=True)
class AppConfig:
    camera: CameraConfig
    storage: StorageConfig
    motion: MotionConfig
    intelligence: IntelligenceConfig
    record_duration_s: int  # 86400 for production 24h; 60 for dev/test

    @classmethod
    def from_env(cls) -> "AppConfig":
        return cls(
            camera=CameraConfig(
                ip=os.getenv("CAMERA_IP", "192.168.1.100"),
                port=int(os.getenv("CAMERA_PORT", "554")),
                username=os.getenv("CAMERA_USER", "admin"),
                password=os.getenv("CAMERA_PASS", "admin"),
                channel=int(os.getenv("CAMERA_CHANNEL", "1")),
                subtype=int(os.getenv("CAMERA_SUBTYPE", "1")),
            ),
            storage=StorageConfig.default(),
            motion=MotionConfig.default(),
            intelligence=IntelligenceConfig.default(),
            record_duration_s=int(os.getenv("RECORD_DURATION_S", "86400")),
        )
