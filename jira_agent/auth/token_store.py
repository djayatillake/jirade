"""Secure token storage using system keyring."""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import keyring


class TokenStore:
    """Secure token storage using system keyring with file fallback."""

    SERVICE_NAME = "jira-agent"

    def __init__(self, fallback_dir: Path | None = None):
        """Initialize token store.

        Args:
            fallback_dir: Directory for fallback file storage if keyring unavailable.
        """
        self.fallback_dir = fallback_dir or Path.home() / ".jira-agent" / "tokens"
        self._use_keyring = self._check_keyring_available()

    def _check_keyring_available(self) -> bool:
        """Check if system keyring is available."""
        try:
            keyring.get_keyring()
            return True
        except Exception:
            return False

    def save(self, service: str, tokens: dict[str, Any]) -> None:
        """Save tokens securely.

        Args:
            service: Service name (jira, github, databricks).
            tokens: Token data to store.
        """
        tokens["saved_at"] = datetime.utcnow().isoformat()

        if self._use_keyring:
            try:
                keyring.set_password(self.SERVICE_NAME, service, json.dumps(tokens))
                return
            except Exception:
                pass

        # Fallback to file storage
        self.fallback_dir.mkdir(parents=True, exist_ok=True)
        token_file = self.fallback_dir / f"{service}_tokens.json"
        token_file.write_text(json.dumps(tokens, indent=2))
        os.chmod(token_file, 0o600)

    def get(self, service: str) -> dict[str, Any] | None:
        """Retrieve stored tokens.

        Args:
            service: Service name (jira, github, databricks).

        Returns:
            Token data or None if not found.
        """
        if self._use_keyring:
            try:
                data = keyring.get_password(self.SERVICE_NAME, service)
                if data:
                    return json.loads(data)
            except Exception:
                pass

        # Fallback to file
        token_file = self.fallback_dir / f"{service}_tokens.json"
        if token_file.exists():
            try:
                return json.loads(token_file.read_text())
            except Exception:
                pass

        return None

    def delete(self, service: str) -> None:
        """Delete stored tokens.

        Args:
            service: Service name (jira, github, databricks).
        """
        if self._use_keyring:
            try:
                keyring.delete_password(self.SERVICE_NAME, service)
            except Exception:
                pass

        token_file = self.fallback_dir / f"{service}_tokens.json"
        if token_file.exists():
            token_file.unlink()

    def is_expired(self, service: str, buffer_seconds: int = 300) -> bool:
        """Check if tokens are expired.

        Args:
            service: Service name.
            buffer_seconds: Buffer time before actual expiry.

        Returns:
            True if tokens are expired or will expire within buffer.
        """
        tokens = self.get(service)
        if not tokens:
            return True

        saved_at_str = tokens.get("saved_at")
        if not saved_at_str:
            return True

        try:
            saved_at = datetime.fromisoformat(saved_at_str)
        except ValueError:
            return True

        expires_in = tokens.get("expires_in", 3600)
        expiry_time = saved_at + timedelta(seconds=expires_in - buffer_seconds)

        return datetime.utcnow() > expiry_time

    def has_valid_token(self, service: str) -> bool:
        """Check if service has valid (non-expired) tokens.

        Args:
            service: Service name.

        Returns:
            True if valid tokens exist.
        """
        tokens = self.get(service)
        if not tokens:
            return False

        # For services without expiry (like GitHub PAT)
        if "expires_in" not in tokens:
            return True

        return not self.is_expired(service)
