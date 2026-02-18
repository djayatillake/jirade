"""Transcript handler with wake word detection.

Buffers real-time transcription from Recall.ai and detects when the bot
is addressed. Extracts the query text after the wake word.
"""

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class TranscriptEntry:
    """A single transcription segment."""

    speaker: str
    text: str
    timestamp: float = field(default_factory=time.time)
    is_final: bool = False


@dataclass
class PendingQuery:
    """A detected wake word + subsequent text, waiting for silence to finalize."""

    speaker: str
    wake_word: str
    text_after_wake: str
    started_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)


class TranscriptHandler:
    """Handles real-time transcription and wake word detection.

    Buffers incoming transcript segments, detects wake words, and
    collects the query text following a wake word until a pause is detected.
    """

    def __init__(
        self,
        wake_words: list[str] | None = None,
        wake_word_timeout: float = 10.0,
        on_query: asyncio.coroutines | None = None,
    ):
        """Initialize the transcript handler.

        Args:
            wake_words: List of trigger phrases (case-insensitive).
            wake_word_timeout: Seconds of silence after wake word before finalizing query.
            on_query: Async callback called with (speaker, query_text) when a query is ready.
        """
        self.wake_words = [w.lower().strip() for w in (wake_words or ["hey claude", "hey jirade"])]
        self.wake_word_timeout = wake_word_timeout
        self.on_query = on_query

        # Build regex pattern that matches any wake word, allowing punctuation between words
        # e.g., "hey claude" also matches "Hey, Claude" or "hey. claude"
        parts = []
        for w in self.wake_words:
            words = w.split()
            flexible = r"[,.\s]*\s+".join(re.escape(word) for word in words)
            parts.append(flexible)
        self._wake_pattern = re.compile(r"(?:" + "|".join(parts) + r")\s*[,.]?\s*", re.IGNORECASE)

        self._pending: PendingQuery | None = None
        self._history: list[TranscriptEntry] = []
        self._timeout_task: asyncio.Task | None = None

    async def handle_transcript(self, speaker: str, text: str, is_final: bool = False) -> None:
        """Process an incoming transcript segment.

        Args:
            speaker: Name of the person speaking.
            text: Transcribed text.
            is_final: Whether this is a final (vs. interim) transcript.
        """
        entry = TranscriptEntry(speaker=speaker, text=text, is_final=is_final)
        self._history.append(entry)

        # Keep history bounded
        if len(self._history) > 500:
            self._history = self._history[-250:]

        text_lower = text.lower().strip()

        # Check for wake word in this segment
        match = self._wake_pattern.search(text_lower)

        if match:
            # Extract text after the wake word
            after_wake = text[match.end():].strip()

            logger.info(f"Wake word detected from {speaker}: '{match.group()}' -> query: '{after_wake}'")

            # Cancel any existing pending query
            if self._timeout_task and not self._timeout_task.done():
                self._timeout_task.cancel()

            self._pending = PendingQuery(
                speaker=speaker,
                wake_word=match.group().strip(),
                text_after_wake=after_wake,
            )

            # If there's already substantial text after the wake word and it's a final segment,
            # start the timeout to finalize
            if after_wake and is_final:
                self._timeout_task = asyncio.create_task(self._wait_and_finalize())

        elif self._pending is not None:
            # We have a pending query - append this segment if from same speaker
            if speaker == self._pending.speaker:
                if self._pending.text_after_wake:
                    self._pending.text_after_wake += " " + text.strip()
                else:
                    self._pending.text_after_wake = text.strip()
                self._pending.last_updated = time.time()

                # Reset timeout
                if self._timeout_task and not self._timeout_task.done():
                    self._timeout_task.cancel()

                if is_final:
                    self._timeout_task = asyncio.create_task(self._wait_and_finalize())
            else:
                # Different speaker - finalize what we have if there's content
                if self._pending.text_after_wake.strip():
                    await self._finalize_query()

    async def _wait_and_finalize(self) -> None:
        """Wait for the timeout period then finalize the pending query."""
        try:
            await asyncio.sleep(self.wake_word_timeout)
            if self._pending and self._pending.text_after_wake.strip():
                await self._finalize_query()
        except asyncio.CancelledError:
            pass  # Timeout was reset by new transcript

    async def _finalize_query(self) -> None:
        """Finalize the pending query and invoke the callback."""
        if self._pending is None:
            return

        query = self._pending.text_after_wake.strip()
        speaker = self._pending.speaker

        # Clean up
        if self._timeout_task and not self._timeout_task.done():
            self._timeout_task.cancel()
        self._pending = None

        if not query:
            logger.info(f"Wake word from {speaker} but no query text, ignoring")
            return

        logger.info(f"Query from {speaker}: {query}")

        if self.on_query:
            try:
                await self.on_query(speaker, query)
            except Exception:
                logger.exception(f"Error in query callback for: {query}")

    def get_recent_context(self, n: int = 20) -> str:
        """Get recent transcript as context string.

        Args:
            n: Number of recent entries to include.

        Returns:
            Formatted transcript context.
        """
        recent = self._history[-n:]
        lines = []
        for entry in recent:
            lines.append(f"{entry.speaker}: {entry.text}")
        return "\n".join(lines)
