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
from .intelligence.adapters.sqlite_observation_adapter import SQLiteObservationAdapter
from .intelligence.adapters.claude_vision_adapter import ClaudeVisionAdapter
from .intelligence.adapters.whisper_transcription_adapter import WhisperTranscriptionAdapter
from .intelligence.domain.ports.ports import (
    IClipQueryPort,
    IObservationStorePort,
    IVideoAnalysisPort,
    ITranscriptionPort,
)
from .intelligence.domain.services.clip_analysis_service import ClipAnalysisService


class Container:
    """
    Manual DI container. Each property is lazily initialised and cached.
    Callers depend on port interfaces — never on concrete adapter types.
    """

    def __init__(self, config: AppConfig):
        self._config = config
        # Surveillance
        self._recorder: IStreamRecorderPort | None = None
        self._motion_detector: IMotionDetectionPort | None = None
        self._clip_storage: IClipStoragePort | None = None
        self._ingestion_service: IStreamIngestionPort | None = None
        # Intelligence
        self._observation_store: IObservationStorePort | None = None
        self._clip_query: IClipQueryPort | None = None
        self._video_analysis: IVideoAnalysisPort | None = None
        self._transcription: ITranscriptionPort | None = None
        self._clip_analysis_service: ClipAnalysisService | None = None

    # ------------------------------------------------------------------
    # Surveillance domain
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Intelligence domain
    # ------------------------------------------------------------------

    @property
    def observation_store(self) -> IObservationStorePort:
        """SQLiteObservationAdapter also implements IClipQueryPort."""
        if self._observation_store is None:
            self._observation_store = SQLiteObservationAdapter(
                db_path=self._config.storage.db_path
            )
        return self._observation_store

    @property
    def clip_query(self) -> IClipQueryPort:
        """Reuse the same SQLiteObservationAdapter instance (dual-role)."""
        if self._clip_query is None:
            # Cast: SQLiteObservationAdapter satisfies both ports
            self._clip_query = self.observation_store  # type: ignore[assignment]
        return self._clip_query

    @property
    def video_analysis(self) -> IVideoAnalysisPort:
        if self._video_analysis is None:
            self._video_analysis = ClaudeVisionAdapter(
                api_key=self._config.intelligence.anthropic_api_key,
                max_frames=self._config.intelligence.frame_extraction_fps * 5,
            )
        return self._video_analysis

    @property
    def transcription(self) -> ITranscriptionPort:
        if self._transcription is None:
            self._transcription = WhisperTranscriptionAdapter(
                model_size=self._config.intelligence.whisper_model
            )
        return self._transcription

    @property
    def clip_analysis_service(self) -> ClipAnalysisService:
        if self._clip_analysis_service is None:
            self._clip_analysis_service = ClipAnalysisService(
                clip_query=self.clip_query,
                video_analysis=self.video_analysis,
                transcription=self.transcription,
                observation_store=self.observation_store,
            )
        return self._clip_analysis_service


def build_container() -> Container:
    config = AppConfig.from_env()
    return Container(config=config)
