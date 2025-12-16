"""Jira OAuth 2.0 authentication."""

import webbrowser
from urllib.parse import urlencode

import requests

from .oauth_server import LocalOAuthServer
from .token_store import TokenStore


class JiraOAuth:
    """Jira OAuth 2.0 (3LO) authentication flow."""

    AUTH_URL = "https://auth.atlassian.com/authorize"
    TOKEN_URL = "https://auth.atlassian.com/oauth/token"
    RESOURCES_URL = "https://api.atlassian.com/oauth/token/accessible-resources"

    SCOPES = [
        "read:jira-work",
        "write:jira-work",
        "read:jira-user",
        "offline_access",
    ]

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        token_store: TokenStore,
        callback_port: int = 8888,
    ):
        """Initialize Jira OAuth.

        Args:
            client_id: OAuth app client ID.
            client_secret: OAuth app client secret.
            token_store: Token storage instance.
            callback_port: Port for OAuth callback server.
        """
        self.client_id = client_id
        self.client_secret = client_secret
        self.token_store = token_store
        self.callback_port = callback_port

    def login(self) -> dict:
        """Perform OAuth login flow.

        Opens browser for user authentication and waits for callback.

        Returns:
            Token data including access_token, refresh_token, cloud_id.

        Raises:
            Exception: If authentication fails.
        """
        server = LocalOAuthServer(port=self.callback_port)
        server.start()

        # Build authorization URL
        params = {
            "audience": "api.atlassian.com",
            "client_id": self.client_id,
            "scope": " ".join(self.SCOPES),
            "redirect_uri": server.callback_url,
            "response_type": "code",
            "prompt": "consent",
        }

        auth_url = f"{self.AUTH_URL}?{urlencode(params)}"

        print("Opening browser for Jira authentication...")
        print(f"If browser doesn't open, visit: {auth_url}")
        webbrowser.open(auth_url)

        # Wait for callback
        code = server.wait_for_code(timeout=120)
        server.shutdown()

        if not code:
            raise Exception("Failed to receive authorization code")

        # Exchange code for tokens
        tokens = self._exchange_code(code, server.callback_url)

        # Get accessible resources (cloud ID)
        cloud_id = self._get_cloud_id(tokens["access_token"])
        tokens["cloud_id"] = cloud_id

        # Store tokens
        self.token_store.save("jira", tokens)

        print("Jira authentication successful!")
        return tokens

    def _exchange_code(self, code: str, redirect_uri: str) -> dict:
        """Exchange authorization code for tokens.

        Args:
            code: Authorization code from callback.
            redirect_uri: Redirect URI used in authorization.

        Returns:
            Token response with access_token, refresh_token, expires_in.
        """
        response = requests.post(
            self.TOKEN_URL,
            json={
                "grant_type": "authorization_code",
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "code": code,
                "redirect_uri": redirect_uri,
            },
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()
        return response.json()

    def _get_cloud_id(self, access_token: str) -> str:
        """Get the Atlassian cloud ID for API calls.

        Args:
            access_token: Valid access token.

        Returns:
            Cloud ID for the first accessible resource.
        """
        response = requests.get(
            self.RESOURCES_URL,
            headers={"Authorization": f"Bearer {access_token}"},
        )
        response.raise_for_status()
        resources = response.json()

        if not resources:
            raise Exception("No accessible Jira resources found")

        # Return first resource's cloud ID
        return resources[0]["id"]

    def refresh_token(self) -> dict:
        """Refresh the access token.

        Returns:
            New token data.

        Raises:
            Exception: If no refresh token available.
        """
        tokens = self.token_store.get("jira")

        if not tokens or "refresh_token" not in tokens:
            raise Exception("No refresh token available. Please login again.")

        response = requests.post(
            self.TOKEN_URL,
            json={
                "grant_type": "refresh_token",
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "refresh_token": tokens["refresh_token"],
            },
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()

        new_tokens = response.json()
        # Preserve cloud_id
        new_tokens["cloud_id"] = tokens.get("cloud_id")
        self.token_store.save("jira", new_tokens)

        return new_tokens

    def get_access_token(self) -> str:
        """Get a valid access token, refreshing if necessary.

        Returns:
            Valid access token.

        Raises:
            Exception: If not authenticated.
        """
        tokens = self.token_store.get("jira")

        if not tokens:
            raise Exception("Not authenticated with Jira. Please run: jira-agent auth login")

        # Check if token is expired (with buffer)
        if self.token_store.is_expired("jira", buffer_seconds=300):
            tokens = self.refresh_token()

        return tokens["access_token"]

    def get_cloud_id(self) -> str:
        """Get the Jira cloud ID.

        Returns:
            Cloud ID for API calls.

        Raises:
            Exception: If not authenticated.
        """
        tokens = self.token_store.get("jira")

        if not tokens or "cloud_id" not in tokens:
            raise Exception("Not authenticated with Jira. Please run: jira-agent auth login")

        return tokens["cloud_id"]

    def logout(self) -> None:
        """Remove stored Jira tokens."""
        self.token_store.delete("jira")
        print("Logged out of Jira")

    def is_authenticated(self) -> bool:
        """Check if authenticated with Jira.

        Returns:
            True if valid tokens exist.
        """
        return self.token_store.has_valid_token("jira")
