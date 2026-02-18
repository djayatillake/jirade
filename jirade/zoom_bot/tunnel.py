"""SSH tunnel manager for automatic public URL provisioning.

Starts an SSH tunnel to localhost.run (or similar service) so Recall.ai
can send webhook events to the local server without manual setup.
"""

import asyncio
import logging
import re
import shutil
from typing import Callable, Coroutine

logger = logging.getLogger(__name__)

# Matches the HTTPS URL that localhost.run prints on connection
_URL_PATTERN = re.compile(r"(https://[a-z0-9]+\.lhr\.life)")


class TunnelManager:
    """Manages an SSH tunnel to localhost.run for public webhook URLs.

    Usage:
        tunnel = TunnelManager(local_port=8090)
        await tunnel.start()       # blocks until URL is available
        print(tunnel.webhook_url)  # https://abc123.lhr.life/webhook/recall
        ...
        await tunnel.stop()
    """

    def __init__(
        self,
        local_port: int = 8090,
        tunnel_host: str = "localhost.run",
        on_url_changed: Callable[[str, str], Coroutine] | None = None,
    ):
        self.local_port = local_port
        self.tunnel_host = tunnel_host
        self.on_url_changed = on_url_changed

        self._url: str = ""
        self._process: asyncio.subprocess.Process | None = None
        self._monitor_task: asyncio.Task | None = None
        self._stopping = False
        self._url_ready = asyncio.Event()

    @property
    def url(self) -> str:
        """The current tunnel base URL (e.g. https://abc123.lhr.life)."""
        return self._url

    @property
    def webhook_url(self) -> str:
        """Full webhook URL for Recall.ai (base URL + /webhook/recall)."""
        return f"{self._url}/webhook/recall" if self._url else ""

    @property
    def is_connected(self) -> bool:
        """Whether the tunnel is up and has a valid URL."""
        return bool(self._url and self._process and self._process.returncode is None)

    async def start(self, timeout: float = 30.0) -> str:
        """Start the SSH tunnel and wait for the public URL.

        Args:
            timeout: Max seconds to wait for the tunnel URL.

        Returns:
            The public base URL.

        Raises:
            RuntimeError: If SSH is not installed or tunnel fails to start.
            TimeoutError: If the URL isn't available within timeout.
        """
        if not shutil.which("ssh"):
            raise RuntimeError("ssh is not installed - cannot create tunnel")

        self._stopping = False
        self._url_ready.clear()

        await self._start_process()

        try:
            await asyncio.wait_for(self._url_ready.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            await self.stop()
            raise TimeoutError(
                f"SSH tunnel to {self.tunnel_host} did not produce a URL within {timeout}s"
            )

        logger.info(f"Tunnel established: {self._url}")
        return self._url

    async def stop(self) -> None:
        """Stop the SSH tunnel and clean up."""
        self._stopping = True

        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None

        await self._kill_process()
        self._url = ""
        logger.info("Tunnel stopped")

    async def _start_process(self) -> None:
        """Start the SSH subprocess."""
        self._process = await asyncio.create_subprocess_exec(
            "ssh",
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", "ServerAliveInterval=15",
            "-o", "ServerAliveCountMax=3",
            "-o", "ExitOnForwardFailure=yes",
            "-R", f"80:localhost:{self.local_port}",
            self.tunnel_host,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        self._monitor_task = asyncio.create_task(self._monitor())

    async def _kill_process(self) -> None:
        """Terminate the SSH process if running."""
        if self._process and self._process.returncode is None:
            self._process.terminate()
            try:
                await asyncio.wait_for(self._process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self._process.kill()
                await self._process.wait()
        self._process = None

    async def _monitor(self) -> None:
        """Read SSH output, parse URL, and handle reconnection."""
        backoff = 2.0
        max_backoff = 60.0

        while not self._stopping:
            try:
                await self._read_output()
            except asyncio.CancelledError:
                return

            if self._stopping:
                return

            # Process exited unexpectedly - reconnect with backoff
            old_url = self._url
            self._url = ""
            logger.warning(
                f"Tunnel process exited (was: {old_url or 'no url'}). "
                f"Reconnecting in {backoff:.0f}s..."
            )

            try:
                await asyncio.sleep(backoff)
            except asyncio.CancelledError:
                return

            backoff = min(backoff * 2, max_backoff)

            try:
                self._url_ready.clear()
                await self._start_process()
                # Wait for new URL before resetting backoff
                try:
                    await asyncio.wait_for(self._url_ready.wait(), timeout=30.0)
                    backoff = 2.0  # Reset on success
                    logger.info(f"Tunnel reconnected: {self._url}")

                    if old_url and old_url != self._url and self.on_url_changed:
                        try:
                            await self.on_url_changed(old_url, self._url)
                        except Exception:
                            logger.exception("on_url_changed callback failed")
                except asyncio.TimeoutError:
                    logger.error("Tunnel reconnect timed out, will retry...")
                    await self._kill_process()
            except Exception:
                logger.exception("Failed to restart tunnel process")

    async def _read_output(self) -> None:
        """Read lines from the SSH process stdout until it exits."""
        proc = self._process
        if not proc or not proc.stdout:
            return

        while True:
            line_bytes = await proc.stdout.readline()
            if not line_bytes:
                break  # EOF - process exited

            line = line_bytes.decode("utf-8", errors="replace").strip()
            if not line:
                continue

            logger.debug(f"[tunnel] {line}")

            match = _URL_PATTERN.search(line)
            if match:
                self._url = match.group(1)
                self._url_ready.set()
