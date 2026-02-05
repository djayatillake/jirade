"""Global agent configuration."""

import subprocess
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def get_stored_anthropic_key() -> str:
    """Get Anthropic API key from secure storage if available.

    Returns:
        Anthropic API key or empty string if not stored.
    """
    try:
        from .auth.token_store import TokenStore

        store = TokenStore()
        tokens = store.get("anthropic")
        if tokens and tokens.get("api_key"):
            return tokens["api_key"]
    except Exception:
        pass
    return ""


def get_gh_cli_token() -> str:
    """Get GitHub token from gh CLI if available.

    Returns:
        GitHub token or empty string if not available.
    """
    try:
        result = subprocess.run(
            ["gh", "auth", "token"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return ""


# Path to the jirade repo root (where .env lives)
_JIRADE_ROOT = Path(__file__).parent.parent


class AgentSettings(BaseSettings):
    """Global settings for Jirade (Jira Data Engineer).

    These settings are loaded from environment variables.
    """

    model_config = SettingsConfigDict(
        env_prefix="JIRADE_",
        env_file=_JIRADE_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Claude configuration
    anthropic_api_key: str = Field(
        default_factory=get_stored_anthropic_key,
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

    # GitHub configuration (falls back to gh CLI token if not set)
    github_token: str = Field(
        default_factory=get_gh_cli_token,
        description="GitHub token (auto-detected from gh CLI if not set)",
    )

    # Databricks configuration
    databricks_host: str = Field(default="", description="Databricks workspace host URL")
    databricks_http_path: str = Field(default="", description="Databricks SQL warehouse HTTP path")
    databricks_auth_type: str = Field(default="oauth", description="Databricks auth type: 'oauth' (default) or 'token'")
    databricks_token: str = Field(default="", description="Databricks personal access token (only needed if auth_type='token')")
    databricks_catalog: str = Field(default="", description="Default Unity Catalog to use (optional)")

    # dbt CI configuration (local Databricks CI)
    databricks_ci_catalog: str = Field(default="", description="Catalog for CI tables (must have CREATE SCHEMA permission)")
    dbt_event_time_lookback_days: int = Field(default=3, description="Days of data to include in CI runs for microbatch models")
    dbt_ci_schema_prefix: str = Field(default="jirade_ci", description="Prefix for CI schema names (e.g., jirade_ci_123)")

    # Webhook server configuration
    webhook_secret: str = Field(default="", description="Secret for validating webhook signatures")
    webhook_host: str = Field(default="0.0.0.0", description="Webhook server host")
    webhook_port: int = Field(default=8080, description="Webhook server port")

    # Agent user identification
    agent_jira_user_id: str = Field(default="", description="Agent's Jira user ID for assignment detection")
    agent_jira_account_id: str = Field(default="", description="Agent's Jira account ID")

    # Workspace configuration
    workspace_dir: Path = Field(
        default_factory=lambda: Path("/tmp/jirade"),
        description="Directory where target repos are cloned",
    )

    # Logging
    log_level: str = Field(default="INFO", description="Logging level")

    # Learning system configuration
    learning_enabled: bool = Field(
        default=True,
        description="Enable automatic learning capture from resolved failures",
    )
    jirade_repo: str = Field(
        default="djayatillake/jirade",
        description="GitHub repo for jirade (where learnings are published)",
    )
    learning_confidence_threshold: str = Field(
        default="medium",
        description="Minimum confidence level to capture learnings (low, medium, high)",
    )

    # Environment setup configuration
    auto_install_deps: bool = Field(
        default=True,
        description="Automatically install missing dependencies before processing tickets",
    )

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
        """Check if Databricks is configured.

        For OAuth auth, only host and http_path are required.
        For token auth, host, http_path, and token are required.
        """
        if not self.databricks_host or not self.databricks_http_path:
            return False
        if self.databricks_auth_type == "token" and not self.databricks_token:
            return False
        return True

    @property
    def has_anthropic_key(self) -> bool:
        """Check if Anthropic API key is configured."""
        return bool(self.anthropic_api_key)



def get_settings() -> AgentSettings:
    """Get the global agent settings.

    Settings are loaded from environment variables with JIRADE_ prefix,
    or from ANTHROPIC_API_KEY for the API key.
    """
    return AgentSettings()
