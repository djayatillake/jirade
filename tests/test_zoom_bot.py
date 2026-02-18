"""Tests for the Zoom meeting bot module."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from jirade.zoom_bot.config import ZoomBotSettings
from jirade.zoom_bot.transcript_handler import TranscriptHandler


# =========== TranscriptHandler Tests ===========


class TestTranscriptHandler:
    """Tests for wake word detection and query extraction."""

    @pytest.fixture
    def on_query(self):
        """Create a mock async callback."""
        return AsyncMock()

    @pytest.fixture
    def handler(self, on_query):
        """Create a TranscriptHandler with short timeout for tests."""
        return TranscriptHandler(
            wake_words=["hey claude", "hey jirade", "at claude"],
            wake_word_timeout=0.2,  # Short timeout for fast tests
            on_query=on_query,
        )

    async def test_wake_word_triggers_query(self, handler, on_query):
        """Wake word followed by query text should trigger callback."""
        await handler.handle_transcript("Alice", "hey claude what is the status of PR 3779", is_final=True)

        # Wait for timeout to finalize
        await asyncio.sleep(0.4)

        on_query.assert_called_once()
        args = on_query.call_args
        assert args[0][0] == "Alice"  # speaker
        assert "status" in args[0][1].lower()  # query contains the question
        assert "PR 3779" in args[0][1] or "pr 3779" in args[0][1].lower()

    async def test_no_wake_word_no_trigger(self, handler, on_query):
        """Normal speech without wake word should not trigger callback."""
        await handler.handle_transcript("Bob", "I think we should deploy on Friday", is_final=True)
        await asyncio.sleep(0.4)
        on_query.assert_not_called()

    async def test_wake_word_only_no_query(self, handler, on_query):
        """Wake word with no following text should not trigger callback."""
        await handler.handle_transcript("Alice", "hey claude", is_final=True)
        await asyncio.sleep(0.4)
        on_query.assert_not_called()

    async def test_multiple_wake_words(self, handler, on_query):
        """All configured wake words should work."""
        for wake_word in ["hey claude", "hey jirade", "at claude"]:
            on_query.reset_mock()
            handler._pending = None  # Reset state

            await handler.handle_transcript("Alice", f"{wake_word} check ticket DATA-500", is_final=True)
            await asyncio.sleep(0.4)

            on_query.assert_called_once()
            assert "DATA-500" in on_query.call_args[0][1]

    async def test_multi_segment_query(self, handler, on_query):
        """Query split across multiple segments should be concatenated."""
        await handler.handle_transcript("Alice", "hey claude what is the status", is_final=True)
        await handler.handle_transcript("Alice", "of the implementation score PR", is_final=True)

        # Wait for timeout
        await asyncio.sleep(0.4)

        on_query.assert_called_once()
        query = on_query.call_args[0][1]
        assert "status" in query.lower()
        assert "implementation score" in query.lower()

    async def test_different_speaker_finalizes(self, handler, on_query):
        """A different speaker should finalize the pending query."""
        await handler.handle_transcript("Alice", "hey claude check PR 100", is_final=True)
        await handler.handle_transcript("Bob", "yeah I agree", is_final=True)

        # Should finalize immediately when different speaker talks
        await asyncio.sleep(0.1)

        on_query.assert_called_once()
        assert on_query.call_args[0][0] == "Alice"
        assert "PR 100" in on_query.call_args[0][1]

    async def test_case_insensitive_wake_word(self, handler, on_query):
        """Wake words should be case-insensitive."""
        await handler.handle_transcript("Alice", "Hey Claude what time is standup", is_final=True)
        await asyncio.sleep(0.4)

        on_query.assert_called_once()
        assert "standup" in on_query.call_args[0][1].lower()

    async def test_wake_word_with_comma(self, handler, on_query):
        """Wake word followed by comma/period should still work."""
        await handler.handle_transcript("Alice", "hey claude, can you check Jira", is_final=True)
        await asyncio.sleep(0.4)

        on_query.assert_called_once()
        assert "jira" in on_query.call_args[0][1].lower()

    async def test_get_recent_context(self, handler):
        """Recent context should return formatted transcript."""
        await handler.handle_transcript("Alice", "Let's discuss the roadmap", is_final=True)
        await handler.handle_transcript("Bob", "I think we should focus on Q2", is_final=True)

        context = handler.get_recent_context(n=5)
        assert "Alice: Let's discuss the roadmap" in context
        assert "Bob: I think we should focus on Q2" in context


# =========== ZoomBotSettings Tests ===========


class TestZoomBotSettings:
    """Tests for Zoom bot configuration."""

    def test_defaults(self):
        """Default settings should have sensible values."""
        with patch.dict("os.environ", {}, clear=False):
            settings = ZoomBotSettings(recall_api_key="", webhook_url="")
            assert settings.webhook_port == 8090
            assert "hey claude" in settings.wake_words
            assert settings.response_mode == "chat"
            assert settings.bot_name == "jirade"

    def test_has_recall_api(self):
        """has_recall_api should check for API key."""
        settings = ZoomBotSettings(recall_api_key="test-key")
        assert settings.has_recall_api is True

        settings = ZoomBotSettings(recall_api_key="")
        assert settings.has_recall_api is False

    def test_has_tts(self):
        """has_tts should require both API key and voice ID."""
        settings = ZoomBotSettings(elevenlabs_api_key="key", elevenlabs_voice_id="voice")
        assert settings.has_tts is True

        settings = ZoomBotSettings(elevenlabs_api_key="key", elevenlabs_voice_id="")
        assert settings.has_tts is False


# =========== RecallClient Tests ===========


class TestRecallClient:
    """Tests for the Recall.ai HTTP client."""

    @pytest.fixture
    def mock_httpx(self):
        with patch("jirade.zoom_bot.recall_client.httpx.AsyncClient") as mock:
            client = AsyncMock()
            mock.return_value = client
            yield client

    async def test_create_bot(self, mock_httpx):
        from jirade.zoom_bot.recall_client import RecallClient

        mock_httpx.post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"id": "bot-123", "meeting_url": "https://zoom.us/j/123"},
            raise_for_status=lambda: None,
        )

        client = RecallClient(api_key="test-key")
        client._client = mock_httpx

        bot = await client.create_bot("https://zoom.us/j/123", bot_name="jirade")
        assert bot["id"] == "bot-123"
        mock_httpx.post.assert_called_once()

    async def test_send_chat_message(self, mock_httpx):
        from jirade.zoom_bot.recall_client import RecallClient

        mock_httpx.post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"status": "ok"},
            raise_for_status=lambda: None,
        )

        client = RecallClient(api_key="test-key")
        client._client = mock_httpx

        result = await client.send_chat_message("bot-123", "Hello team!")
        assert result["status"] == "ok"


# =========== Webhook Server Tests ===========


class TestWebhookServer:
    """Tests for the FastAPI webhook endpoints."""

    @pytest.fixture
    def app(self):
        from jirade.zoom_bot.server import create_app
        return create_app()

    async def test_health_endpoint(self, app):
        from httpx import ASGITransport, AsyncClient

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/health")
            assert response.status_code == 200
            assert response.json()["status"] == "ok"

    async def test_recall_webhook_returns_ok(self, app):
        """Webhook endpoint should return 200 even without initialized server."""
        from httpx import ASGITransport, AsyncClient

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/webhook/recall",
                json={"event": "bot.transcription", "data": {"bot_id": "test", "transcript": {}}},
            )
            # Should return 503 since _server is None (no lifespan)
            assert response.status_code == 503
