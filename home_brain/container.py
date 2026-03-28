"""
Composition Root — container.py

What it is: the ONLY place that wires adapters to ports.
What it knows: all adapters, all ports, AppConfig.
What it doesn't know: CLI commands, domain logic — it only assembles the graph.

In Hexagonal Architecture the composition root lives outside all domains.
No domain imports an adapter. All dependencies flow inward through ports.
No DI framework — manual wiring keeps this transparent and debuggable.
"""
from .shared.config import AppConfig
from .surveillance.adapters.ffmpeg_adapter import (
    FfmpegMotionDetectionAdapter,
    FfmpegStreamRecorderAdapter,
)
from .surveillance.adapters.sqlite_clip_storage_adapter import SQLiteClipStorageAdapter
from .surveillance.domain.ports.ports import (
    IClipStoragePort,
    IMotionDetectionPort,
    IStreamIngestionPort,
    IStreamRecorderPort,
)
from .surveillance.domain.services.stream_ingestion_service import StreamIngestionService


class Container:
    """
    Manual DI container. Each property is lazily initialised and cached.
    Callers depend on port interfaces — never on concrete adapter types.
    """

    def __init__(self, config: AppConfig):
        self._config = config
        self._recorder: IStreamRecorderPort | None = None
        self._motion_detector: IMotionDetectionPort | None = None
        self._clip_storage: IClipStoragePort | None = None
        self._ingestion_service: IStreamIngestionPort | None = None

    @property
    def recorder(self) -> IStreamRecorderPort:
        if self._recorder is None:
            self._recorder = FfmpegStreamRecorderAdapter()
        return self._recorder

    @property
    def motion_detector(self) -> IMotionDetectionPort:
        if self._motion_detector is None:
            self._motion_detector = FfmpegMotionDetectionAdapter()
        return self._motion_detector

    @property
    def clip_storage(self) -> IClipStoragePort:
        if self._clip_storage is None:
            self._clip_storage = SQLiteClipStorageAdapter(
                db_path=self._config.storage.db_path
            )
        return self._clip_storage

    @property
    def ingestion_service(self) -> IStreamIngestionPort:
        if self._ingestion_service is None:
            self._ingestion_service = StreamIngestionService(
                config=self._config,
                recorder=self.recorder,
                motion_detector=self.motion_detector,
                clip_storage=self.clip_storage,
            )
        return self._ingestion_service


def build_container() -> Container:
    config = AppConfig.from_env()
    return Container(config=config)
