"""Logging configuration for the Jira Agent."""

import logging
import sys
from datetime import datetime
from pathlib import Path


def setup_logging(level: str = "INFO", log_dir: Path | None = None) -> None:
    """Configure logging for the agent.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_dir: Optional directory for log files.
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Create formatters
    console_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

    file_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(console_formatter)

    handlers: list[logging.Handler] = [console_handler]

    # File handler (if log_dir provided)
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"agent_{datetime.now():%Y%m%d_%H%M%S}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_formatter)
        handlers.append(file_handler)

    # Configure root logger
    logging.basicConfig(
        level=logging.DEBUG,
        handlers=handlers,
        force=True,
    )

    # Suppress noisy loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)


class TicketLogger:
    """Context-aware logger for ticket processing."""

    def __init__(self, ticket_key: str):
        """Initialize logger for a specific ticket.

        Args:
            ticket_key: The Jira ticket key (e.g., AENG-1234).
        """
        self.ticket_key = ticket_key
        self.logger = logging.getLogger(f"jira_agent.ticket.{ticket_key}")

    def info(self, message: str) -> None:
        """Log info message with ticket context."""
        self.logger.info(f"[{self.ticket_key}] {message}")

    def warning(self, message: str) -> None:
        """Log warning message with ticket context."""
        self.logger.warning(f"[{self.ticket_key}] {message}")

    def error(self, message: str, exc: Exception | None = None) -> None:
        """Log error message with ticket context."""
        self.logger.error(f"[{self.ticket_key}] {message}", exc_info=exc)

    def debug(self, message: str) -> None:
        """Log debug message with ticket context."""
        self.logger.debug(f"[{self.ticket_key}] {message}")
