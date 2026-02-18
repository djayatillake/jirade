"""Text-to-speech for the Zoom meeting bot using native macOS TTS.

Converts text responses to MP3 audio using the macOS `say` command,
then the audio can be sent to the meeting via Recall.ai's output_audio endpoint.

Requires: macOS with `say` command and `ffmpeg` for AIFF→MP3 conversion.
Install ffmpeg: brew install ffmpeg
"""

import asyncio
import base64
import logging
import shutil
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


class TTSClient:
    """Text-to-speech using native macOS `say` command."""

    def __init__(self, voice: str = "Samantha", rate: int = 195):
        """Initialize the TTS client.

        Args:
            voice: macOS voice name (run `say -v '?'` to list available voices).
            rate: Speech rate in words per minute (default ~195).

        Raises:
            RuntimeError: If `say` or `ffmpeg` is not available.
        """
        self.voice = voice
        self.rate = rate

        if not shutil.which("say"):
            raise RuntimeError("macOS `say` command not found - native TTS requires macOS")
        if not shutil.which("ffmpeg"):
            raise RuntimeError(
                "ffmpeg not found - required for audio conversion. Install with: brew install ffmpeg"
            )

    async def synthesize(self, text: str) -> str:
        """Convert text to speech and return base64-encoded MP3.

        Args:
            text: Text to convert to speech.

        Returns:
            Base64-encoded MP3 audio data.
        """
        with tempfile.TemporaryDirectory() as tmp:
            aiff_path = Path(tmp) / "speech.aiff"
            mp3_path = Path(tmp) / "speech.mp3"

            # Generate AIFF with macOS say
            proc = await asyncio.create_subprocess_exec(
                "say", "-v", self.voice, "-r", str(self.rate),
                "-o", str(aiff_path), text,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await proc.communicate()
            if proc.returncode != 0:
                raise RuntimeError(f"say failed: {stderr.decode()}")

            # Convert AIFF to MP3 with ffmpeg
            proc = await asyncio.create_subprocess_exec(
                "ffmpeg", "-i", str(aiff_path),
                "-codec:a", "libmp3lame", "-b:a", "128k",
                "-y", str(mp3_path),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await proc.communicate()
            if proc.returncode != 0:
                raise RuntimeError(f"ffmpeg failed: {stderr.decode()}")

            mp3_bytes = mp3_path.read_bytes()
            mp3_b64 = base64.b64encode(mp3_bytes).decode("utf-8")

            logger.info(f"TTS synthesized {len(text)} chars -> {len(mp3_bytes)} bytes MP3")
            return mp3_b64

    async def close(self) -> None:
        """No-op for interface compatibility."""
        pass
