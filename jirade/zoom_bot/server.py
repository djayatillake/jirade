"""FastAPI webhook server for the Zoom meeting bot.

Receives real-time transcription events from Recall.ai, feeds them to the
transcript handler for wake word detection, and sends responses back to
the meeting via Recall.ai chat API.
"""

import asyncio
import json
import logging
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .agent import ZoomBotAgent
from .config import ZoomBotSettings, get_zoom_settings
from .recall_client import RecallClient
from .transcript_handler import TranscriptHandler
from .tunnel import TunnelManager
from .tts import TTSClient

logger = logging.getLogger(__name__)


class ZoomBotServer:
    """Manages the webhook server and all active bot sessions."""

    def __init__(
        self,
        settings: ZoomBotSettings | None = None,
        tunnel_manager: TunnelManager | None = None,
    ):
        self.settings = settings or get_zoom_settings()
        self.tunnel_manager = tunnel_manager
        self.recall = RecallClient(
            api_key=self.settings.recall_api_key,
            base_url=self.settings.recall_api_url,
        )
        self.agent = ZoomBotAgent(
            model=self.settings.claude_model,
            max_response_tokens=self.settings.max_response_tokens,
        )
        self.tts: TTSClient | None = None
        if self.settings.has_tts:
            self.tts = TTSClient(
                voice=self.settings.tts_voice,
                rate=self.settings.tts_rate,
            )
        # Map bot_id -> TranscriptHandler
        self._handlers: dict[str, TranscriptHandler] = {}
        # Map bot_id -> processing lock (prevent concurrent queries per bot)
        self._locks: dict[str, asyncio.Lock] = {}
        # Pending queries queue for relay mode (external responder like Claude Code)
        self._pending_queries: list[dict[str, Any]] = []
        self._query_counter = 0
        # Notification file for relay queries - external tools can tail -f this
        self._notify_file = Path(tempfile.gettempdir()) / "jirade-zoom-queries.jsonl"

    def get_handler(self, bot_id: str) -> TranscriptHandler:
        """Get or create a transcript handler for a bot."""
        if bot_id not in self._handlers:
            handler = TranscriptHandler(
                wake_words=self.settings.wake_words,
                wake_word_timeout=self.settings.wake_word_timeout,
                on_query=lambda speaker, query, _bid=bot_id: self._handle_query(_bid, speaker, query),
            )
            self._handlers[bot_id] = handler
            self._locks[bot_id] = asyncio.Lock()
        return self._handlers[bot_id]

    async def _handle_query(self, bot_id: str, speaker: str, query: str) -> None:
        """Handle a detected query.

        In 'relay' mode: queues the query for an external responder (e.g., Claude Code).
        In 'agent' mode: calls the built-in Claude agent to auto-respond.
        """
        handler = self._handlers.get(bot_id)
        context = handler.get_recent_context() if handler else ""

        if self.settings.response_mode == "relay":
            # Queue for external responder
            self._query_counter += 1
            pending = {
                "id": self._query_counter,
                "bot_id": bot_id,
                "speaker": speaker,
                "query": query,
                "context": context,
                "timestamp": asyncio.get_event_loop().time(),
            }
            self._pending_queries.append(pending)
            # Append to notification file so external tools (tail -f) get alerted
            try:
                with self._notify_file.open("a") as f:
                    f.write(json.dumps(pending) + "\n")
            except OSError:
                pass
            logger.info(f"[RELAY] Query #{self._query_counter} from {speaker}: {query}")
            return

        # Agent mode - auto-respond
        lock = self._locks.get(bot_id) or asyncio.Lock()

        async with lock:
            try:
                response = await self.agent.answer_query(
                    speaker=speaker,
                    query=query,
                    transcript_context=context,
                )

                # Truncate for Zoom chat (max ~4096 chars in practice)
                if len(response) > 2000:
                    response = response[:1997] + "..."

                if self.tts and self.settings.response_mode == "tts":
                    try:
                        mp3_b64 = await self.tts.synthesize(response)
                        await self.recall.output_audio(bot_id, mp3_b64)
                        await self.recall.send_chat_message(bot_id, f"@{speaker}: {response}")
                    except Exception:
                        logger.exception("TTS failed, falling back to chat-only")
                        await self.recall.send_chat_message(bot_id, f"@{speaker}: {response}")
                else:
                    await self.recall.send_chat_message(bot_id, f"@{speaker}: {response}")

            except Exception:
                logger.exception(f"Failed to handle query from {speaker} in bot {bot_id}")
                try:
                    await self.recall.send_chat_message(
                        bot_id,
                        f"@{speaker}: Sorry, I ran into an error processing that. Please try again.",
                    )
                except Exception:
                    logger.exception("Failed to send error message to chat")

    async def join_meeting(self, meeting_url: str) -> dict[str, Any]:
        """Create a bot to join a meeting.

        Args:
            meeting_url: Zoom meeting URL.

        Returns:
            Bot info from Recall.ai.
        """
        # Resolve webhook URL: static config > tunnel > None
        webhook_url = self.settings.webhook_url or None
        if not webhook_url and self.tunnel_manager and self.tunnel_manager.is_connected:
            webhook_url = self.tunnel_manager.webhook_url

        bot = await self.recall.create_bot(
            meeting_url=meeting_url,
            bot_name=self.settings.bot_name,
            bot_image=self.settings.bot_image or None,
            webhook_url=webhook_url,
            enable_audio_output=self.tts is not None,
        )

        bot_id = bot["id"]
        self.get_handler(bot_id)  # pre-create handler
        logger.info(f"Bot {bot_id} joining meeting: {meeting_url}")
        return bot

    async def leave_meeting(self, bot_id: str) -> None:
        """Make a bot leave its meeting."""
        await self.recall.leave_call(bot_id)
        self._handlers.pop(bot_id, None)
        self._locks.pop(bot_id, None)

    async def get_bot_status(self, bot_id: str) -> dict[str, Any]:
        """Get status of a bot."""
        return await self.recall.get_bot(bot_id)

    async def close(self) -> None:
        """Clean up resources."""
        await self.recall.close()
        if self.tts:
            await self.tts.close()


# Global server instance (set during lifespan)
_server: ZoomBotServer | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan handler - initialize and cleanup the bot server."""
    global _server
    settings = get_zoom_settings()

    # Start SSH tunnel if no static webhook URL and auto_tunnel is enabled
    tunnel: TunnelManager | None = None
    if not settings.webhook_url and settings.auto_tunnel:
        async def _on_url_changed(old_url: str, new_url: str) -> None:
            logger.warning(
                f"Tunnel URL changed: {old_url} -> {new_url}. "
                "Existing bots are orphaned - use /api/leave + /api/join to reconnect."
            )

        tunnel = TunnelManager(
            local_port=settings.webhook_port,
            tunnel_host=settings.tunnel_host,
            on_url_changed=_on_url_changed,
        )
        try:
            url = await tunnel.start(timeout=settings.tunnel_timeout)
            logger.info(f"Auto-tunnel started: {url}")
        except (RuntimeError, TimeoutError) as e:
            logger.error(f"Failed to start tunnel: {e}. Continuing without tunnel.")
            tunnel = None
    elif not settings.webhook_url:
        logger.warning(
            "No webhook URL configured and auto_tunnel is disabled. "
            "Recall.ai won't be able to send events."
        )

    _server = ZoomBotServer(settings, tunnel_manager=tunnel)
    logger.info(f"Zoom bot server started (webhook port: {settings.webhook_port})")
    if _server.tts:
        logger.info(f"TTS enabled (voice: {settings.tts_voice}, rate: {settings.tts_rate})")
    else:
        logger.info("TTS disabled (install ffmpeg to enable: brew install ffmpeg)")

    # Discover and register active bots from Recall.ai
    try:
        bots = await _server.recall.list_bots(limit=10)
        for bot in bots:
            status_changes = bot.get("status_changes", [])
            last_status = status_changes[-1].get("code", "") if status_changes else ""
            if last_status in ("in_call_recording", "in_call_not_recording"):
                bot_id = bot["id"]
                _server.get_handler(bot_id)
                logger.info(f"Recovered active bot: {bot_id} ({bot.get('bot_name', 'unknown')})")
    except Exception:
        logger.warning("Failed to recover active bots on startup", exc_info=True)

    yield

    if tunnel:
        await tunnel.stop()
    if _server:
        await _server.close()
    _server = None


def create_app() -> FastAPI:
    """Create the FastAPI application."""
    app = FastAPI(
        title="jirade Zoom Bot",
        description="Webhook server for jirade Zoom meeting bot",
        lifespan=lifespan,
    )

    @app.get("/health")
    async def health():
        return {"status": "ok", "service": "jirade-zoom-bot"}

    @app.post("/webhook/recall")
    async def recall_webhook(request: Request):
        """Handle webhook events from Recall.ai.

        Recall.ai sends events for transcription updates, bot status changes, etc.
        """
        if _server is None:
            return JSONResponse(status_code=503, content={"error": "Server not initialized"})

        body = await request.json()
        event_type = body.get("event", "")

        if event_type in ("bot.transcription", "transcript.data"):
            await _handle_transcription_event(body)
        elif event_type == "bot.status_change":
            await _handle_status_change(body)
        else:
            logger.debug(f"Unhandled webhook event: {event_type}")

        return {"status": "ok"}

    @app.post("/api/join")
    async def join_meeting(request: Request):
        """API endpoint to make the bot join a meeting."""
        if _server is None:
            return JSONResponse(status_code=503, content={"error": "Server not initialized"})

        body = await request.json()
        meeting_url = body.get("meeting_url")
        if not meeting_url:
            return JSONResponse(status_code=400, content={"error": "meeting_url is required"})

        try:
            bot = await _server.join_meeting(meeting_url)
            return {"status": "joining", "bot_id": bot["id"], "meeting_url": meeting_url}
        except Exception as e:
            logger.exception("Failed to join meeting")
            return JSONResponse(status_code=500, content={"error": str(e)})

    @app.post("/api/leave/{bot_id}")
    async def leave_meeting(bot_id: str):
        """API endpoint to make a bot leave its meeting."""
        if _server is None:
            return JSONResponse(status_code=503, content={"error": "Server not initialized"})

        try:
            await _server.leave_meeting(bot_id)
            return {"status": "leaving", "bot_id": bot_id}
        except Exception as e:
            logger.exception("Failed to leave meeting")
            return JSONResponse(status_code=500, content={"error": str(e)})

    @app.get("/api/status/{bot_id}")
    async def bot_status(bot_id: str):
        """Get status of a bot."""
        if _server is None:
            return JSONResponse(status_code=503, content={"error": "Server not initialized"})

        try:
            status = await _server.get_bot_status(bot_id)
            return status
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": str(e)})

    @app.get("/api/bots")
    async def list_bots():
        """List recent bots."""
        if _server is None:
            return JSONResponse(status_code=503, content={"error": "Server not initialized"})

        try:
            bots = await _server.recall.list_bots()
            return {"bots": bots}
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": str(e)})

    @app.get("/api/tunnel")
    async def tunnel_status():
        """Get the current tunnel status and URL."""
        if _server is None:
            return JSONResponse(status_code=503, content={"error": "Server not initialized"})

        tm = _server.tunnel_manager
        if tm is None:
            return {
                "enabled": False,
                "reason": "static webhook URL configured" if _server.settings.webhook_url else "auto_tunnel disabled",
                "webhook_url": _server.settings.webhook_url or None,
            }

        return {
            "enabled": True,
            "connected": tm.is_connected,
            "url": tm.url,
            "webhook_url": tm.webhook_url,
        }

    @app.get("/api/pending")
    async def pending_queries():
        """Get pending queries waiting for a response (relay mode)."""
        if _server is None:
            return JSONResponse(status_code=503, content={"error": "Server not initialized"})

        return {"queries": _server._pending_queries}

    @app.post("/api/respond")
    async def respond_to_query(request: Request):
        """Send a response to a pending query (relay mode).

        Body: {"message": "response text", "bot_id": "optional-bot-id", "speaker": "optional-speaker"}
        If bot_id is not provided, uses the first active bot.
        """
        if _server is None:
            return JSONResponse(status_code=503, content={"error": "Server not initialized"})

        body = await request.json()
        message = body.get("message", "")
        if not message:
            return JSONResponse(status_code=400, content={"error": "message is required"})

        bot_id = body.get("bot_id", "")
        speaker = body.get("speaker", "")

        # Default to first active handler's bot_id
        if not bot_id and _server._handlers:
            bot_id = next(iter(_server._handlers))

        if not bot_id:
            return JSONResponse(status_code=400, content={"error": "No active bot to respond through"})

        # Format with speaker mention if provided
        if speaker:
            chat_message = f"@{speaker}: {message}"
        else:
            chat_message = message

        # Truncate for Zoom chat
        if len(chat_message) > 2000:
            chat_message = chat_message[:1997] + "..."

        try:
            # Speak the response aloud if TTS is available
            spoke = False
            if _server.tts:
                try:
                    mp3_b64 = await _server.tts.synthesize(message)
                    await _server.recall.output_audio(bot_id, mp3_b64)
                    spoke = True
                except Exception:
                    logger.exception("TTS failed, falling back to chat-only")

            await _server.recall.send_chat_message(bot_id, chat_message)
            # Clear the pending query if it matches
            _server._pending_queries = [
                q for q in _server._pending_queries
                if not (q.get("bot_id") == bot_id and q.get("speaker") == speaker)
            ]
            logger.info(f"[RELAY] Response sent to {speaker} via bot {bot_id}: {message[:80]}")
            return {"status": "sent", "bot_id": bot_id, "spoke": spoke}
        except Exception as e:
            logger.exception("Failed to send relay response")
            return JSONResponse(status_code=500, content={"error": str(e)})

    return app


async def _handle_transcription_event(body: dict[str, Any]) -> None:
    """Handle a transcription webhook event.

    Supports multiple Recall.ai payload formats:
    - Regular webhook: {"event": "bot.transcription", "data": {"bot_id": ..., "transcript": {...}}}
    - Realtime endpoint: {"event": "transcript.data", "data": {"data": {"words": [...], "participant": {...}}}}
    """
    if _server is None:
        return

    data = body.get("data", {})

    # Extract bot_id - try multiple locations
    bot_id = data.get("bot_id", "") or body.get("bot_id", "")

    if not bot_id:
        # Realtime endpoints don't include bot_id - use first tracked handler,
        # or fall back to all active handlers
        if _server._handlers:
            bot_id = next(iter(_server._handlers))
        else:
            # No handlers yet - pick from known bots via a default
            # We'll create a handler for a placeholder and it will work
            logger.warning("No bot_id in webhook and no active handlers - attempting to use default")
            bot_id = "_default"

    # Parse transcript data - handle the nested realtime endpoint format
    # Realtime format: data.data.words[] + data.data.participant
    inner_data = data.get("data", {})
    transcript = data.get("transcript", {})

    # Extract speaker name
    speaker = (
        transcript.get("speaker", "")
        or data.get("speaker", "")
        or inner_data.get("participant", {}).get("name", "Unknown")
    )

    # Extract text from words
    words = inner_data.get("words", []) or transcript.get("words", "") or data.get("words", "")

    if isinstance(words, list):
        text = " ".join(w.get("text", "") for w in words)
    elif isinstance(words, str):
        text = words
    else:
        text = str(words)

    if not text.strip():
        text = transcript.get("original_transcript", "") or data.get("text", "")

    is_final = inner_data.get("is_final", transcript.get("is_final", True))

    logger.info(f"Transcript [{bot_id[:8]}] {speaker}: {text} (final={is_final})")

    if text.strip():
        handler = _server.get_handler(bot_id)
        await handler.handle_transcript(speaker=speaker, text=text, is_final=is_final)


async def _handle_status_change(body: dict[str, Any]) -> None:
    """Handle a bot status change event."""
    data = body.get("data", {})
    bot_id = data.get("bot_id", body.get("bot_id", ""))
    status = data.get("status", {})
    code = status.get("code", "unknown")

    logger.info(f"Bot {bot_id} status changed to: {code}")

    if code in ("done", "fatal"):
        # Bot left or crashed - clean up handler
        if _server:
            _server._handlers.pop(bot_id, None)
            _server._locks.pop(bot_id, None)


def run_server(host: str = "0.0.0.0", port: int = 8090) -> None:
    """Run the webhook server."""
    import uvicorn

    app = create_app()
    uvicorn.run(app, host=host, port=port)
