"""Configuration for the Zoom meeting bot."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from ..config import _JIRADE_ROOT


class ZoomBotSettings(BaseSettings):
    """Settings for the Zoom meeting bot.

    Loaded from environment variables with JIRADE_ZOOM_ prefix,
    or from the .env file in the jirade root.
    """

    model_config = SettingsConfigDict(
        env_prefix="JIRADE_ZOOM_",
        env_file=_JIRADE_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Recall.ai configuration
    recall_api_key: str = Field(default="", description="Recall.ai API key")
    recall_api_url: str = Field(
        default="https://us-west-2.recall.ai/api/v1",
        description="Recall.ai API base URL",
    )

    # Webhook configuration
    webhook_url: str = Field(
        default="",
        description="Public URL where Recall.ai sends webhook events (e.g., https://your-domain.com/webhook/recall)",
    )
    webhook_host: str = Field(default="0.0.0.0", description="Webhook server bind host")
    webhook_port: int = Field(default=8090, description="Webhook server port")
    webhook_secret: str = Field(default="", description="Secret for validating Recall.ai webhook signatures")

    # Wake word configuration
    wake_words: list[str] = Field(
        default=[
            "jirade", "jared", "jarad", "ji raid", "g raid",
            "hey claude", "hey jirade", "hey jared",
            "at claude", "at jirade", "at jared",
        ],
        description="Phrases that trigger the bot to respond (includes common STT misheard variants of 'jirade')",
    )
    wake_word_timeout: float = Field(
        default=10.0,
        description="Seconds to wait after wake word for the full query before processing",
    )

    # Tunnel configuration
    auto_tunnel: bool = Field(
        default=True,
        description="Automatically start SSH tunnel to localhost.run when no webhook_url is set",
    )
    tunnel_host: str = Field(
        default="localhost.run",
        description="SSH tunnel host (e.g., localhost.run)",
    )
    tunnel_timeout: float = Field(
        default=30.0,
        description="Seconds to wait for SSH tunnel to establish",
    )

    # Bot behavior
    bot_name: str = Field(default="jirade", description="Display name for the bot in the Zoom meeting")
    bot_image: str = Field(default="", description="URL to bot avatar image")
    response_mode: str = Field(
        default="relay",
        description="How the bot responds: 'relay' (queue for external responder like Claude Code), 'chat' (auto-respond via built-in agent), or 'tts' (text-to-speech)",
    )

    # TTS configuration (optional, for response_mode='tts')
    elevenlabs_api_key: str = Field(default="", description="ElevenLabs API key for TTS responses")
    elevenlabs_voice_id: str = Field(default="", description="ElevenLabs voice ID")

    # Claude configuration (inherits from parent but can override)
    claude_model: str = Field(
        default="claude-sonnet-4-20250514",
        description="Claude model for zoom bot responses (faster model preferred)",
    )
    max_response_tokens: int = Field(
        default=1024,
        description="Max tokens for Claude responses (keep short for chat/TTS)",
    )

    @property
    def has_recall_api(self) -> bool:
        """Check if Recall.ai is configured."""
        return bool(self.recall_api_key)

    @property
    def has_tts(self) -> bool:
        """Check if TTS is configured."""
        return bool(self.elevenlabs_api_key and self.elevenlabs_voice_id)


def get_zoom_settings() -> ZoomBotSettings:
    """Get the Zoom bot settings."""
    return ZoomBotSettings()
