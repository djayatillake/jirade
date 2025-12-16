"""GitHub authentication."""

import webbrowser
from urllib.parse import urlencode

import requests

from .oauth_server import LocalOAuthServer
from .token_store import TokenStore


class GitHubAuth:
    """GitHub authentication (OAuth or PAT)."""

    AUTH_URL = "https://github.com/login/oauth/authorize"
    TOKEN_URL = "https://github.com/login/oauth/access_token"
    USER_URL = "https://api.github.com/user"

    SCOPES = ["repo", "read:user", "read:org"]

    def __init__(
        self,
        token_store: TokenStore,
        client_id: str | None = None,
        client_secret: str | None = None,
        callback_port: int = 8889,
    ):
        """Initialize GitHub auth.

        Args:
            token_store: Token storage instance.
            client_id: OAuth app client ID (for OAuth flow).
            client_secret: OAuth app client secret (for OAuth flow).
            callback_port: Port for OAuth callback server.
        """
        self.token_store = token_store
        self.client_id = client_id
        self.client_secret = client_secret
        self.callback_port = callback_port

    def login_with_token(self, token: str) -> dict:
        """Login using a personal access token.

        Args:
            token: GitHub personal access token.

        Returns:
            Token data including user info.
        """
        # Verify token works
        response = requests.get(
            self.USER_URL,
            headers={"Authorization": f"token {token}"},
        )
        response.raise_for_status()
        user = response.json()

        tokens = {
            "access_token": token,
            "token_type": "pat",
            "user": user["login"],
            "user_id": user["id"],
        }

        self.token_store.save("github", tokens)
        print(f"GitHub authentication successful! Logged in as {user['login']}")
        return tokens

    def login_oauth(self) -> dict:
        """Perform OAuth login flow.

        Opens browser for user authentication.

        Returns:
            Token data.

        Raises:
            Exception: If OAuth credentials not configured or auth fails.
        """
        if not self.client_id or not self.client_secret:
            raise Exception(
                "GitHub OAuth credentials not configured. "
                "Use a Personal Access Token instead: jira-agent auth login --service=github"
            )

        server = LocalOAuthServer(port=self.callback_port)
        server.start()

        params = {
            "client_id": self.client_id,
            "redirect_uri": server.callback_url,
            "scope": " ".join(self.SCOPES),
        }

        auth_url = f"{self.AUTH_URL}?{urlencode(params)}"

        print("Opening browser for GitHub authentication...")
        print(f"If browser doesn't open, visit: {auth_url}")
        webbrowser.open(auth_url)

        code = server.wait_for_code(timeout=120)
        server.shutdown()

        if not code:
            raise Exception("Failed to receive authorization code")

        # Exchange code for token
        response = requests.post(
            self.TOKEN_URL,
            data={
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "code": code,
            },
            headers={"Accept": "application/json"},
        )
        response.raise_for_status()
        token_data = response.json()

        if "error" in token_data:
            raise Exception(f"GitHub OAuth error: {token_data['error_description']}")

        # Get user info
        user_response = requests.get(
            self.USER_URL,
            headers={"Authorization": f"token {token_data['access_token']}"},
        )
        user_response.raise_for_status()
        user = user_response.json()

        tokens = {
            **token_data,
            "user": user["login"],
            "user_id": user["id"],
        }

        self.token_store.save("github", tokens)
        print(f"GitHub authentication successful! Logged in as {user['login']}")
        return tokens

    def get_access_token(self) -> str:
        """Get the GitHub access token.

        Returns:
            Valid access token.

        Raises:
            Exception: If not authenticated.
        """
        tokens = self.token_store.get("github")

        if not tokens or "access_token" not in tokens:
            raise Exception("Not authenticated with GitHub. Please run: jira-agent auth login")

        return tokens["access_token"]

    def logout(self) -> None:
        """Remove stored GitHub tokens."""
        self.token_store.delete("github")
        print("Logged out of GitHub")

    def is_authenticated(self) -> bool:
        """Check if authenticated with GitHub.

        Returns:
            True if valid tokens exist.
        """
        return self.token_store.has_valid_token("github")

    def get_user(self) -> str | None:
        """Get authenticated user's login.

        Returns:
            Username or None.
        """
        tokens = self.token_store.get("github")
        return tokens.get("user") if tokens else None
