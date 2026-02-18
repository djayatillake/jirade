"""Text-to-speech for the Zoom meeting bot.

Converts Claude's text responses to audio using ElevenLabs API,
then sends the audio to the meeting via Recall.ai's output_audio endpoint.
"""

import base64
import logging

import httpx

logger = logging.getLogger(__name__)


class TTSClient:
    """Async client for ElevenLabs text-to-speech API."""

    def __init__(self, api_key: str, voice_id: str = ""):
        self.api_key = api_key
        self.voice_id = voice_id or "21m00Tcm4TlvDq8ikWAM"  # Default: Rachel
        self._client = httpx.AsyncClient(
            base_url="https://api.elevenlabs.io/v1",
            headers={
                "xi-api-key": self.api_key,
                "Content-Type": "application/json",
            },
            timeout=30.0,
        )

    async def synthesize(self, text: str) -> str:
        """Convert text to speech and return base64-encoded MP3.

        Args:
            text: Text to convert to speech.

        Returns:
            Base64-encoded MP3 audio data.
        """
        response = await self._client.post(
            f"/text-to-speech/{self.voice_id}",
            json={
                "text": text,
                "model_id": "eleven_turbo_v2_5",
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.75,
                    "style": 0.0,
                    "use_speaker_boost": True,
                },
            },
            headers={"Accept": "audio/mpeg"},
        )
        response.raise_for_status()

        # Response body is raw MP3 bytes
        mp3_bytes = response.content
        mp3_b64 = base64.b64encode(mp3_bytes).decode("utf-8")

        logger.info(f"TTS synthesized {len(text)} chars -> {len(mp3_bytes)} bytes MP3")
        return mp3_b64

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()
