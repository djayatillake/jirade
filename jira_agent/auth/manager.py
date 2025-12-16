"""Authentication manager coordinating all auth providers."""

from ..config import AgentSettings
from .github_auth import GitHubAuth
from .jira_auth import JiraOAuth
from .token_store import TokenStore


class AuthManager:
    """Manages authentication for all services."""

    def __init__(self, settings: AgentSettings):
        """Initialize auth manager.

        Args:
            settings: Agent settings with credentials.
        """
        self.settings = settings
        self.token_store = TokenStore()

        self._jira_auth: JiraOAuth | None = None
        self._github_auth: GitHubAuth | None = None

    @property
    def jira(self) -> JiraOAuth:
        """Get Jira auth provider."""
        if self._jira_auth is None:
            self._jira_auth = JiraOAuth(
                client_id=self.settings.jira_oauth_client_id,
                client_secret=self.settings.jira_oauth_client_secret,
                token_store=self.token_store,
            )
        return self._jira_auth

    @property
    def github(self) -> GitHubAuth:
        """Get GitHub auth provider."""
        if self._github_auth is None:
            self._github_auth = GitHubAuth(
                token_store=self.token_store,
            )
        return self._github_auth

    def login(self, service: str) -> None:
        """Login to a specific service.

        Args:
            service: Service name (jira, github, databricks).
        """
        if service == "jira":
            self._login_jira()
        elif service == "github":
            self._login_github()
        elif service == "databricks":
            self._login_databricks()
        else:
            print(f"Unknown service: {service}")

    def _login_jira(self) -> None:
        """Handle Jira login."""
        if not self.settings.has_jira_oauth:
            print("Jira OAuth credentials not configured.")
            print("Set JIRA_AGENT_JIRA_OAUTH_CLIENT_ID and JIRA_AGENT_JIRA_OAUTH_CLIENT_SECRET")
            return

        try:
            self.jira.login()
        except Exception as e:
            print(f"Jira login failed: {e}")

    def _login_github(self) -> None:
        """Handle GitHub login."""
        if self.settings.has_github_token:
            # Use configured token
            try:
                self.github.login_with_token(self.settings.github_token)
            except Exception as e:
                print(f"GitHub token validation failed: {e}")
        else:
            # Prompt for token
            print("GitHub Personal Access Token not configured.")
            print("Please create a token at: https://github.com/settings/tokens")
            print("Required scopes: repo, read:user, read:org")
            print()
            token = input("Enter your GitHub Personal Access Token: ").strip()
            if token:
                try:
                    self.github.login_with_token(token)
                except Exception as e:
                    print(f"GitHub login failed: {e}")
            else:
                print("No token provided, skipping GitHub authentication")

    def _login_databricks(self) -> None:
        """Handle Databricks login."""
        if not self.settings.has_databricks:
            print("Databricks credentials not configured.")
            print("Set JIRA_AGENT_DATABRICKS_HOST and JIRA_AGENT_DATABRICKS_TOKEN")
            return

        # Validate Databricks connection
        try:
            from databricks.sdk import WorkspaceClient

            client = WorkspaceClient(
                host=self.settings.databricks_host,
                token=self.settings.databricks_token,
            )
            # Simple validation - get current user
            me = client.current_user.me()
            print(f"Databricks authentication successful! Logged in as {me.user_name}")

            # Store validation
            self.token_store.save(
                "databricks",
                {
                    "host": self.settings.databricks_host,
                    "user": me.user_name,
                    "validated": True,
                },
            )
        except Exception as e:
            print(f"Databricks validation failed: {e}")

    def login_all(self) -> None:
        """Login to all configured services."""
        print("Authenticating with all services...\n")

        print("=== Jira ===")
        self._login_jira()
        print()

        print("=== GitHub ===")
        self._login_github()
        print()

        print("=== Databricks ===")
        self._login_databricks()
        print()

        print("Authentication complete!")

    def logout(self, service: str) -> None:
        """Logout from a specific service.

        Args:
            service: Service name.
        """
        if service == "jira":
            self.jira.logout()
        elif service == "github":
            self.github.logout()
        elif service == "databricks":
            self.token_store.delete("databricks")
            print("Logged out of Databricks")
        else:
            print(f"Unknown service: {service}")

    def logout_all(self) -> None:
        """Logout from all services."""
        self.jira.logout()
        self.github.logout()
        self.token_store.delete("databricks")
        print("Logged out of all services")

    def print_status(self) -> None:
        """Print authentication status for all services."""
        print("Authentication Status")
        print("=" * 40)

        # Jira
        jira_status = "✓ Authenticated" if self.jira.is_authenticated() else "✗ Not authenticated"
        print(f"Jira:       {jira_status}")

        # GitHub
        github_status = "✓ Authenticated" if self.github.is_authenticated() else "✗ Not authenticated"
        github_user = self.github.get_user()
        if github_user:
            github_status += f" ({github_user})"
        print(f"GitHub:     {github_status}")

        # Databricks
        db_tokens = self.token_store.get("databricks")
        if db_tokens and db_tokens.get("validated"):
            db_status = f"✓ Authenticated ({db_tokens.get('user', 'unknown')})"
        else:
            db_status = "✗ Not authenticated"
        print(f"Databricks: {db_status}")
