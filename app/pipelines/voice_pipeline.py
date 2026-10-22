"""Voice pipeline: validate -> transcribe -> delegate to TextPipeline."""

from __future__ import annotations

from app.adapters.speech.audio_validator import validate_audio
from app.adapters.speech.base import SpeechAdapter
from app.core.errors import PipelineError
from app.core.logging import get_logger
from app.pipelines.text_pipeline import TextPipeline

logger = get_logger(__name__)


class VoicePipeline:
    """Processes voice notes through speech transcription and the text pipeline.

    Flow: validate bytes -> transcribe (Whisper) -> TextPipeline (RAG -> LLM -> validate)
    """

    def __init__(
        self,
        speech_adapter: SpeechAdapter,
        text_pipeline: TextPipeline,
        max_audio_bytes: int = 25 * 1024 * 1024,
        language: str = "ur",
    ):
        self._speech = speech_adapter
        self._text_pipeline = text_pipeline
        self._max_audio_bytes = max_audio_bytes
        self._language = language

    async def process(self, audio_bytes: bytes) -> str:
        """Run the full voice pipeline."""
        fmt = validate_audio(audio_bytes, max_bytes=self._max_audio_bytes)
        logger.info(
            "VoicePipeline: validated audio format=%s, size=%d bytes",
            fmt,
            len(audio_bytes),
        )

        from app.flow import LANGUAGES, current_language

        whisper_lang = LANGUAGES.get(current_language.get(), {}).get("whisper", self._language)
        try:
            transcription = await self._speech.transcribe(
                audio_bytes, language=whisper_lang
            )
        except PipelineError:
            raise
        except Exception as e:
            raise PipelineError(
                "VoicePipeline", f"Speech transcription failed: {e}"
            ) from e

        logger.info(
            "VoicePipeline: transcription complete (%d chars)",
            len(transcription),
        )

        if not transcription or not transcription.strip():
            raise PipelineError("VoicePipeline", "Transcription is empty")

        return await self._text_pipeline.process(transcription)
