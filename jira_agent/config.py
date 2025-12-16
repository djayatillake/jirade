"""Global agent configuration."""

import os
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AgentSettings(BaseSettings):
    """Global settings for the Jira Agent.

    These settings are loaded from environment variables.
    """

    model_config = SettingsConfigDict(
        env_prefix="JIRA_AGENT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Claude configuration
    anthropic_api_key: str = Field(
        default="",
        description="Anthropic API key for Claude",
        alias="ANTHROPIC_API_KEY",
    )
    claude_model: str = Field(
        default="claude-opus-4-5-20251101",
        description="Claude model to use (defaults to Opus 4.5)",
    )

    # Jira OAuth configuration
    jira_oauth_client_id: str = Field(default="", description="Jira OAuth 2.0 client ID")
    jira_oauth_client_secret: str = Field(default="", description="Jira OAuth 2.0 client secret")

    # GitHub configuration
    github_token: str = Field(default="", description="GitHub personal access token")

    # Databricks configuration
    databricks_host: str = Field(default="", description="Databricks workspace host URL")
    databricks_token: str = Field(default="", description="Databricks personal access token")
    databricks_http_path: str = Field(default="", description="Databricks SQL warehouse HTTP path")

    # Webhook server configuration
    webhook_secret: str = Field(default="", description="Secret for validating webhook signatures")
    webhook_host: str = Field(default="0.0.0.0", description="Webhook server host")
    webhook_port: int = Field(default=8080, description="Webhook server port")

    # Agent user identification
    agent_jira_user_id: str = Field(default="", description="Agent's Jira user ID for assignment detection")
    agent_jira_account_id: str = Field(default="", description="Agent's Jira account ID")

    # Workspace configuration
    workspace_dir: Path = Field(
        default_factory=lambda: Path("/tmp/jira-agent"),
        description="Directory where target repos are cloned",
    )

    # Logging
    log_level: str = Field(default="INFO", description="Logging level")

    @property
    def has_jira_oauth(self) -> bool:
        """Check if Jira OAuth credentials are configured."""
        return bool(self.jira_oauth_client_id and self.jira_oauth_client_secret)

    @property
    def has_github_token(self) -> bool:
        """Check if GitHub token is configured."""
        return bool(self.github_token)

    @property
    def has_databricks(self) -> bool:
        """Check if Databricks is configured."""
        return bool(self.databricks_host and self.databricks_token)

    @property
    def has_anthropic_key(self) -> bool:
        """Check if Anthropic API key is configured."""
        return bool(self.anthropic_api_key)


def get_settings() -> AgentSettings:
    """Get the global agent settings.

    Settings are loaded from environment variables with JIRA_AGENT_ prefix,
    or from ANTHROPIC_API_KEY for the API key.
    """
    return AgentSettings()
