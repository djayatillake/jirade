"""Local OAuth callback server."""

import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse


class OAuthCallbackHandler(BaseHTTPRequestHandler):
    """Handle OAuth callback requests."""

    auth_code: str | None = None
    state: str | None = None
    error: str | None = None

    def do_GET(self) -> None:
        """Handle GET request from OAuth callback."""
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)

        if "code" in query:
            OAuthCallbackHandler.auth_code = query["code"][0]
            OAuthCallbackHandler.state = query.get("state", [None])[0]
            self._send_success_response()
        elif "error" in query:
            OAuthCallbackHandler.error = query["error"][0]
            error_desc = query.get("error_description", ["Unknown error"])[0]
            self._send_error_response(error_desc)
        else:
            self._send_error_response("Missing authorization code")

    def _send_success_response(self) -> None:
        """Send success HTML response."""
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Authentication Successful</title>
            <style>
                body {
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                    margin: 0;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                }
                .container {
                    text-align: center;
                    background: white;
                    padding: 40px 60px;
                    border-radius: 12px;
                    box-shadow: 0 10px 40px rgba(0,0,0,0.2);
                }
                h1 { color: #22c55e; margin-bottom: 10px; }
                p { color: #666; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>&#10004; Authentication Successful</h1>
                <p>You can close this window and return to the terminal.</p>
            </div>
        </body>
        </html>
        """
        self.wfile.write(html.encode())

    def _send_error_response(self, error_message: str) -> None:
        """Send error HTML response."""
        self.send_response(400)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Authentication Failed</title>
            <style>
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                    margin: 0;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                }}
                .container {{
                    text-align: center;
                    background: white;
                    padding: 40px 60px;
                    border-radius: 12px;
                    box-shadow: 0 10px 40px rgba(0,0,0,0.2);
                }}
                h1 {{ color: #ef4444; margin-bottom: 10px; }}
                p {{ color: #666; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>&#10008; Authentication Failed</h1>
                <p>{error_message}</p>
                <p>Please try again.</p>
            </div>
        </body>
        </html>
        """
        self.wfile.write(html.encode())

    def log_message(self, format: str, *args) -> None:
        """Suppress HTTP server logging."""
        pass


class LocalOAuthServer:
    """Local server for OAuth callbacks."""

    def __init__(self, port: int = 8888):
        """Initialize OAuth server.

        Args:
            port: Port to listen on.
        """
        self.port = port
        self.server: HTTPServer | None = None
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the OAuth callback server."""
        # Reset handler state
        OAuthCallbackHandler.auth_code = None
        OAuthCallbackHandler.state = None
        OAuthCallbackHandler.error = None

        self.server = HTTPServer(("localhost", self.port), OAuthCallbackHandler)
        self.thread = threading.Thread(target=self.server.handle_request)
        self.thread.start()

    def wait_for_code(self, timeout: float = 120) -> str | None:
        """Wait for authorization code.

        Args:
            timeout: Maximum time to wait in seconds.

        Returns:
            Authorization code or None if timeout/error.
        """
        if self.thread:
            self.thread.join(timeout=timeout)

        if OAuthCallbackHandler.error:
            raise Exception(f"OAuth error: {OAuthCallbackHandler.error}")

        return OAuthCallbackHandler.auth_code

    @property
    def callback_url(self) -> str:
        """Get the callback URL for OAuth configuration."""
        return f"http://localhost:{self.port}/callback"

    def shutdown(self) -> None:
        """Shutdown the server."""
        if self.server:
            self.server.server_close()
