"""Session management for REPL agent."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Session:
    """Manages conversation history and context for REPL sessions."""

    messages: list[dict[str, Any]] = field(default_factory=list)
    max_messages: int = 100  # Keep last N message pairs to avoid context overflow

    def add_user_message(self, content: str) -> None:
        """Add a user message to the session."""
        self.messages.append({"role": "user", "content": content})
        self._trim_if_needed()

    def add_assistant_message(self, content: Any) -> None:
        """Add an assistant message to the session."""
        self.messages.append({"role": "assistant", "content": content})
        self._trim_if_needed()

    def add_tool_results(self, tool_results: list[dict[str, Any]]) -> None:
        """Add tool results as a user message."""
        self.messages.append({"role": "user", "content": tool_results})
        self._trim_if_needed()

    def get_messages(self) -> list[dict[str, Any]]:
        """Get all messages in the session."""
        return self.messages

    def clear(self) -> None:
        """Clear the session history."""
        self.messages = []

    def _trim_if_needed(self) -> None:
        """Trim old messages if we exceed max_messages."""
        if len(self.messages) > self.max_messages:
            # Keep the most recent messages, but ensure we don't break mid-conversation
            # Always keep pairs (user + assistant) to maintain conversation integrity
            excess = len(self.messages) - self.max_messages
            # Round up to nearest even number to keep pairs
            if excess % 2 != 0:
                excess += 1
            self.messages = self.messages[excess:]
