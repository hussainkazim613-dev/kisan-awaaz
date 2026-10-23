"""Whisper speech-to-text adapter via OpenAI audio transcriptions API."""

from __future__ import annotations

import os

import httpx

from app.adapters.speech.audio_validator import validate_audio
from app.adapters.speech.base import SpeechAdapter
from app.core.errors import AdapterError
from app.core.logging import get_logger

logger = get_logger(__name__)

OPENAI_AUDIO_URL = os.environ.get(
    "SPEECH_API_URL", "https://api.groq.com/openai/v1/audio/transcriptions"
)

FORMAT_EXTENSIONS = {
    "ogg": "ogg",
    "mp3": "mp3",
    "wav": "wav",
    "m4a": "m4a",
}


class WhisperAdapter(SpeechAdapter):
    """Calls OpenAI Whisper for speech-to-text. TTS is not implemented for MVP."""

    def __init__(
        self,
        api_key: str,
        model: str = "whisper-1",
        timeout: int = 60,
        max_audio_bytes: int = 25 * 1024 * 1024,
    ):
        self._api_key = api_key
        self._model = model
        self._timeout = timeout
        self._max_audio_bytes = max_audio_bytes

    async def transcribe(self, audio_bytes: bytes, language: str = "ur") -> str:
        fmt = validate_audio(audio_bytes, max_bytes=self._max_audio_bytes)
        logger.info(
            "WhisperAdapter: transcribing audio format=%s, size=%d bytes, language=%s",
            fmt,
            len(audio_bytes),
            language,
        )

        extension = FORMAT_EXTENSIONS[fmt]
        filename = f"audio.{extension}"

        files = {"file": (filename, audio_bytes)}
        data = {"model": self._model, "language": language}
        headers = {"Authorization": f"Bearer {self._api_key}"}

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    OPENAI_AUDIO_URL,
                    files=files,
                    data=data,
                    headers=headers,
                )
                response.raise_for_status()
        except httpx.TimeoutException:
            raise AdapterError("Whisper", "API request timed out")
        except httpx.HTTPStatusError as e:
            raise AdapterError(
                "Whisper",
                f"API returned {e.response.status_code}: {e.response.text[:200]}",
            )
        except httpx.RequestError as e:
            raise AdapterError("Whisper", f"Request failed: {e}")

        return self._parse_response(response.json())

    async def synthesize(self, text: str, language: str = "ur") -> bytes:
        raise NotImplementedError("Text-to-speech is not implemented for MVP")

    def _parse_response(self, data: dict) -> str:
        """Extract the transcribed text from the OpenAI response."""
        if "text" not in data:
            raise AdapterError("Whisper", f"Unexpected response structure: missing 'text' key")

        text = data["text"].strip() if isinstance(data["text"], str) else ""
        if not text:
            raise AdapterError("Whisper", "Empty transcription returned")

        return text
