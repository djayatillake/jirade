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
        elif service == "dbt_cloud":
            print("dbt Cloud integration has been removed in v0.4.0.")
            print("CI now runs locally against Databricks. See: jirade auth login --service=databricks")
        else:
            print(f"Unknown service: {service}")

    def _login_jira(self) -> None:
        """Handle Jira login."""
        if not self.settings.has_jira_oauth:
            print("Jira OAuth credentials not configured.")
            print("Set JIRADE_JIRA_OAUTH_CLIENT_ID and JIRADE_JIRA_OAUTH_CLIENT_SECRET")
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
        """Handle Databricks login and CI configuration."""
        if not self.settings.has_databricks:
            print("Databricks connection not configured.")
            print("Set JIRADE_DATABRICKS_HOST and JIRADE_DATABRICKS_HTTP_PATH")
            print("Auth: set JIRADE_DATABRICKS_AUTH_TYPE=oauth (default) or token")
            return

        # Validate Databricks connection
        try:
            from ..clients.databricks_client import DatabricksMetadataClient

            with DatabricksMetadataClient(
                host=self.settings.databricks_host,
                http_path=self.settings.databricks_http_path,
                auth_type=self.settings.databricks_auth_type,
                token=self.settings.databricks_token if self.settings.databricks_auth_type == "token" else None,
                catalog=self.settings.databricks_catalog or None,
            ) as client:
                # Simple validation - run a metadata query
                client.execute_metadata_query("SHOW SCHEMAS")
                print(f"Databricks connection successful!")
                print(f"  Host: {self.settings.databricks_host}")
                print(f"  Auth: {self.settings.databricks_auth_type}")

            # Store validation
            self.token_store.save(
                "databricks",
                {
                    "host": self.settings.databricks_host,
                    "auth_type": self.settings.databricks_auth_type,
                    "validated": True,
                },
            )

            # Check CI catalog configuration
            if self.settings.databricks_ci_catalog:
                print(f"  CI catalog: {self.settings.databricks_ci_catalog}")
            else:
                print()
                print("WARNING: JIRADE_DATABRICKS_CI_CATALOG is not set.")
                print("This is required for dbt CI (jirade_run_dbt_ci).")
                print("Set it to your dev catalog where you have CREATE SCHEMA permission.")
                print("Example: JIRADE_DATABRICKS_CI_CATALOG=development_yourname_metadata")

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
        elif service == "dbt_cloud":
            print("dbt Cloud integration has been removed in v0.4.0.")
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

        # GitHub - check both token store and settings (gh CLI)
        if self.settings.has_github_token:
            github_status = "✓ Authenticated (via gh CLI or env)"
        elif self.github.is_authenticated():
            github_user = self.github.get_user()
            github_status = f"✓ Authenticated ({github_user})" if github_user else "✓ Authenticated"
        else:
            github_status = "✗ Not authenticated"
        print(f"GitHub:     {github_status}")

        # Databricks
        if self.settings.has_databricks:
            db_tokens = self.token_store.get("databricks")
            if db_tokens and db_tokens.get("validated"):
                db_status = f"✓ Connected ({self.settings.databricks_auth_type} auth)"
            else:
                db_status = "✓ Configured (not validated - run: jirade auth login --service=databricks)"
            # Check CI catalog
            if self.settings.databricks_ci_catalog:
                db_status += f"\n            CI catalog: {self.settings.databricks_ci_catalog}"
            else:
                db_status += "\n            ⚠ CI catalog not set (JIRADE_DATABRICKS_CI_CATALOG)"
        else:
            db_status = "✗ NOT CONFIGURED (required for dbt CI)"
        print(f"Databricks: {db_status}")
