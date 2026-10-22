"""Text-to-speech for spoken replies (free, no API key): Microsoft Edge voices via edge-tts.

Urdu voice is used for Urdu, Punjabi, Sindhi and Saraiki (all Arabic-script, understood
by Pakistani farmers); Pashto has its own voice. Returns MP3 bytes (accepted by WhatsApp).
"""
from __future__ import annotations

import re

from app.core.logging import get_logger

logger = get_logger(__name__)

VOICES = {
    "ur": "ur-PK-UzmaNeural",
    "pa": "ur-PK-UzmaNeural",
    "sd": "ur-PK-UzmaNeural",
    "skr": "ur-PK-UzmaNeural",
    "ps": "ps-AF-LatifaNeural",
}
MAX_CHARS = 900


def _clean(text: str) -> str:
    text = re.sub(r"[*_#`>~|]", " ", text)
    text = re.sub(r"[\U0001F300-\U0001FAFF\u2600-\u27BF\uFE0F]", " ", text)
    return re.sub(r"\s+", " ", text).strip()[:MAX_CHARS]


async def synthesize(text: str, lang: str = "ur") -> bytes:
    import edge_tts  # imported lazily so the app still starts if the package is missing

    voice = VOICES.get(lang, VOICES["ur"])
    communicate = edge_tts.Communicate(_clean(text), voice, rate="-5%")
    audio = bytearray()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio += chunk["data"]
    if not audio:
        raise RuntimeError("TTS returned no audio")
    return bytes(audio)
