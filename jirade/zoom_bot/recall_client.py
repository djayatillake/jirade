"""Async HTTP client for Recall.ai API.

Handles bot lifecycle: creating bots to join meetings, sending chat messages,
and managing bot state.

API docs: https://docs.recall.ai/
"""

import asyncio
import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class RecallClient:
    """Async client for the Recall.ai REST API."""

    def __init__(self, api_key: str, base_url: str = "https://us-west-2.recall.ai/api/v1"):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "Authorization": f"Token {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=30.0,
        )

    async def create_bot(
        self,
        meeting_url: str,
        bot_name: str = "jirade",
        bot_image: str | None = None,
        webhook_url: str | None = None,
        enable_audio_output: bool = False,
    ) -> dict[str, Any]:
        """Create a bot that joins a Zoom meeting.

        Args:
            meeting_url: Zoom meeting URL (e.g., https://zoom.us/j/123456789).
            bot_name: Display name for the bot in the meeting.
            bot_image: Optional URL to an avatar image for the bot.
            webhook_url: URL where Recall.ai sends transcription/status webhooks.
            enable_audio_output: If True, configure the bot for TTS audio output.

        Returns:
            Bot object from Recall.ai API including bot ID.
        """
        greeting = f"Hi! I'm {bot_name}. Say 'hey claude' followed by your question"
        if enable_audio_output:
            greeting += " and I'll respond out loud."
        else:
            greeting += " and I'll respond in chat."

        payload: dict[str, Any] = {
            "meeting_url": meeting_url,
            "bot_name": bot_name,
            "chat": {
                "on_bot_join": {
                    "send_to": "everyone",
                    "message": greeting,
                },
            },
        }

        # Configure transcription + real-time webhook
        recording_config: dict[str, Any] = {
            "transcript": {
                "provider": {"meeting_captions": {}},
            },
        }
        if webhook_url:
            recording_config["realtime_endpoints"] = [
                {"type": "webhook", "url": webhook_url, "events": ["transcript.data"]},
            ]
        payload["recording_config"] = recording_config

        if bot_image:
            payload["bot_image"] = bot_image

        # TODO: Uncomment when ready to enable TTS
        # if enable_audio_output:
        #     payload["automatic_audio_output"] = {"in_call_recording_enabled": True}

        response = await self._client.post("/bot", json=payload)
        response.raise_for_status()
        bot = response.json()
        logger.info(f"Created bot {bot['id']} for meeting: {meeting_url}")
        return bot

    async def get_bot(self, bot_id: str) -> dict[str, Any]:
        """Get current status of a bot.

        Args:
            bot_id: Recall.ai bot ID.

        Returns:
            Bot status object.
        """
        response = await self._client.get(f"/bot/{bot_id}")
        response.raise_for_status()
        return response.json()

    async def list_bots(self, limit: int = 20) -> list[dict[str, Any]]:
        """List recent bots.

        Args:
            limit: Maximum number of bots to return.

        Returns:
            List of bot objects.
        """
        response = await self._client.get("/bot", params={"limit": limit})
        response.raise_for_status()
        data = response.json()
        return data.get("results", [])

    async def send_chat_message(self, bot_id: str, message: str) -> dict[str, Any]:
        """Send a chat message in the meeting via the bot.

        Uses asyncio.to_thread to avoid event loop blocking issues when called
        from asyncio.Task contexts (e.g., wake word timeout handlers).

        Args:
            bot_id: Recall.ai bot ID.
            message: Message text to send in Zoom chat.

        Returns:
            Response from Recall.ai.
        """
        url = f"{self.base_url}/bot/{bot_id}/send_chat_message"
        result = await asyncio.to_thread(self._sync_post, url, {"message": message})
        logger.info(f"Sent chat message via bot {bot_id}: {message[:80]}...")
        return result

    def _sync_post(self, url: str, json_data: dict[str, Any]) -> dict[str, Any]:
        """Synchronous HTTP POST (runs in thread pool via asyncio.to_thread)."""
        response = httpx.post(
            url,
            json=json_data,
            headers={
                "Authorization": f"Token {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=15.0,
        )
        response.raise_for_status()
        return response.json()

    async def output_audio(self, bot_id: str, mp3_b64: str) -> dict[str, Any]:
        """Play audio in the meeting via the bot (TTS output).

        The bot speaks the audio out loud to all meeting participants.
        Requires the bot to have been created with enable_audio_output=True.

        Args:
            bot_id: Recall.ai bot ID.
            mp3_b64: Base64-encoded MP3 audio data.

        Returns:
            Response from Recall.ai.
        """
        response = await self._client.post(
            f"/bot/{bot_id}/output_audio",
            json={"kind": "mp3", "b64_data": mp3_b64},
        )
        response.raise_for_status()
        logger.info(f"Sent audio output via bot {bot_id}")
        return response.json()

    async def leave_call(self, bot_id: str) -> None:
        """Make the bot leave the meeting.

        Args:
            bot_id: Recall.ai bot ID.
        """
        response = await self._client.post(f"/bot/{bot_id}/leave_call")
        response.raise_for_status()
        logger.info(f"Bot {bot_id} leaving call")

    async def get_bot_transcript(self, bot_id: str) -> list[dict[str, Any]]:
        """Get the full transcript for a bot's meeting.

        Args:
            bot_id: Recall.ai bot ID.

        Returns:
            List of transcript entries.
        """
        response = await self._client.get(f"/bot/{bot_id}/transcript")
        response.raise_for_status()
        return response.json()

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()
