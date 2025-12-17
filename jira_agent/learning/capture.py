"""Learning capture during agent execution.

This module tracks failure/fix cycles during ticket processing and captures
learnings ONLY when fixes are verified to work.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from anthropic import Anthropic

from .models import (
    FailureRecord,
    FixAttempt,
    Learning,
    LearningConfidence,
)
from .storage import LearningStorage

logger = logging.getLogger(__name__)


class LearningCapture:
    """Captures learnings during agent execution.

    Tracks failures and fix attempts, and only generates learnings when
    fixes are verified to work.

    Usage:
        capture = LearningCapture(ticket_key="PROJ-1234", repo_name="your-org/your-repo")

        # When a failure occurs
        capture.record_failure("pre-commit", error_output, context)

        # When agent attempts a fix
        capture.record_fix_attempt("pre-commit", "Fixed formatting", files_changed)

        # When the same command succeeds (verification)
        capture.verify_fix_success("pre-commit")

        # At end of processing, get verified learnings
        learnings = capture.get_verified_learnings(claude_client)
    """

    def __init__(
        self,
        ticket_key: str,
        repo_name: str,
        enabled: bool = True,
    ):
        """Initialize learning capture.

        Args:
            ticket_key: Jira ticket key being processed.
            repo_name: Repository name (owner/name).
            enabled: Whether learning capture is enabled.
        """
        self.ticket_key = ticket_key
        self.repo_name = repo_name
        self.enabled = enabled

        # Track failures by type
        self._failures: dict[str, FailureRecord] = {}

        # Track fix attempts by type
        self._fix_attempts: dict[str, FixAttempt] = {}

        # Track which fixes have been verified
        self._verified_types: set[str] = set()

        # Current iteration (updated externally)
        self.current_iteration = 0

    def record_failure(
        self,
        failure_type: str,
        error_message: str,
        command: str | None = None,
        files_involved: list[str] | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Record a failure that occurred during processing.

        Args:
            failure_type: Type of failure (e.g., 'pre-commit', 'dbt-compile').
            error_message: The error message or output.
            command: The command that failed.
            files_involved: Files involved in the failure.
            context: Additional context.
        """
        if not self.enabled:
            return

        logger.debug(f"Recording failure: {failure_type}")

        self._failures[failure_type] = FailureRecord(
            failure_type=failure_type,
            error_message=error_message,
            command=command,
            files_involved=files_involved or [],
            iteration=self.current_iteration,
            context=context or {},
        )

        # Clear any previous verification for this type
        self._verified_types.discard(failure_type)

    def record_fix_attempt(
        self,
        failure_type: str,
        solution_description: str,
        files_modified: list[str] | None = None,
        code_changes: str | None = None,
    ) -> None:
        """Record an attempted fix for a failure.

        Args:
            failure_type: Type of failure this fix targets.
            solution_description: Description of what was done to fix.
            files_modified: Files that were modified.
            code_changes: Diff or summary of code changes.
        """
        if not self.enabled:
            return

        # Only record if we have a corresponding failure
        if failure_type not in self._failures:
            logger.debug(f"No failure recorded for {failure_type}, skipping fix recording")
            return

        logger.debug(f"Recording fix attempt: {failure_type}")

        self._fix_attempts[failure_type] = FixAttempt(
            failure_type=failure_type,
            solution_description=solution_description,
            files_modified=files_modified or [],
            code_changes=code_changes,
            iteration=self.current_iteration,
            verified=False,
        )

    def verify_fix_success(self, failure_type: str) -> bool:
        """Mark a fix as verified (the same operation now succeeds).

        Args:
            failure_type: Type of failure that was fixed.

        Returns:
            True if a learning can be captured (failure + fix + verification exists).
        """
        if not self.enabled:
            return False

        # Must have both failure and fix attempt recorded
        if failure_type not in self._failures:
            logger.debug(f"No failure recorded for {failure_type}")
            return False

        if failure_type not in self._fix_attempts:
            logger.debug(f"No fix attempt recorded for {failure_type}")
            return False

        logger.info(f"Fix verified for {failure_type}")

        # Mark as verified
        self._verified_types.add(failure_type)
        self._fix_attempts[failure_type].verified = True

        return True

    def has_pending_failure(self, failure_type: str) -> bool:
        """Check if there's a pending failure of the given type.

        Args:
            failure_type: Type of failure to check.

        Returns:
            True if there's an unverified failure.
        """
        return (
            failure_type in self._failures
            and failure_type not in self._verified_types
        )

    def get_verified_fix_types(self) -> list[str]:
        """Get list of failure types that have verified fixes.

        Returns:
            List of failure types with verified fixes.
        """
        return list(self._verified_types)

    def get_verified_learnings(
        self,
        claude_client: Anthropic | None = None,
        conversation_messages: list[dict] | None = None,
    ) -> list[Learning]:
        """Generate learnings from verified fixes.

        Args:
            claude_client: Optional Anthropic client for generating better descriptions.
            conversation_messages: Optional conversation history for context.

        Returns:
            List of Learning objects for verified fixes.
        """
        if not self.enabled:
            return []

        learnings = []

        for failure_type in self._verified_types:
            failure = self._failures.get(failure_type)
            fix = self._fix_attempts.get(failure_type)

            if not failure or not fix:
                continue

            # Generate learning description using Claude if available
            if claude_client and conversation_messages:
                title, applicability = self._generate_learning_description(
                    claude_client, failure, fix, conversation_messages
                )
            else:
                title = f"Fixed {failure_type} error"
                applicability = f"Apply when encountering {failure_type} failures"

            learning = Learning.from_verified_fix(
                ticket=self.ticket_key,
                repo=self.repo_name,
                failure=failure,
                fix=fix,
                title=title,
                applicability=applicability,
                confidence=LearningConfidence.HIGH,  # Verified fixes are high confidence
            )

            learnings.append(learning)
            logger.info(f"Generated learning: {learning.title}")

        return learnings

    def save_verified_learnings(
        self,
        repo_path: Path,
        claude_client: Anthropic | None = None,
        conversation_messages: list[dict] | None = None,
    ) -> list[Path]:
        """Generate and save verified learnings to the target repo.

        Args:
            repo_path: Path to the target repository.
            claude_client: Optional Anthropic client for generating descriptions.
            conversation_messages: Optional conversation history.

        Returns:
            List of paths to saved learning files.
        """
        learnings = self.get_verified_learnings(claude_client, conversation_messages)

        if not learnings:
            logger.debug("No verified learnings to save")
            return []

        storage = LearningStorage()
        saved_paths = []

        for learning in learnings:
            path = storage.save_to_target_repo(learning, repo_path)
            saved_paths.append(path)

        return saved_paths

    def _generate_learning_description(
        self,
        claude_client: Anthropic,
        failure: FailureRecord,
        fix: FixAttempt,
        messages: list[dict],
    ) -> tuple[str, str]:
        """Use Claude to generate a learning title and applicability.

        Args:
            claude_client: Anthropic client.
            failure: The failure record.
            fix: The fix attempt.
            messages: Conversation history for context.

        Returns:
            Tuple of (title, applicability).
        """
        prompt = f"""Based on this failure and fix, generate a concise learning title and applicability statement.

Failure Type: {failure.failure_type}
Error: {failure.error_message[:500]}
Fix: {fix.solution_description}
Files: {', '.join(fix.files_modified[:5])}

Respond in this exact format:
TITLE: <1 line title, max 60 chars>
APPLICABILITY: <1-2 sentences about when to apply this>"""

        try:
            response = claude_client.messages.create(
                model="claude-3-haiku-20240307",  # Use fast model for this
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
            )

            text = response.content[0].text
            title = "Fixed error"
            applicability = "Apply when similar errors occur"

            for line in text.split("\n"):
                if line.startswith("TITLE:"):
                    title = line.replace("TITLE:", "").strip()
                elif line.startswith("APPLICABILITY:"):
                    applicability = line.replace("APPLICABILITY:", "").strip()

            return title, applicability

        except Exception as e:
            logger.warning(f"Failed to generate learning description: {e}")
            return f"Fixed {failure.failure_type} error", f"Apply when {failure.failure_type} fails"

    def clear(self) -> None:
        """Clear all recorded failures and fixes."""
        self._failures.clear()
        self._fix_attempts.clear()
        self._verified_types.clear()
        self.current_iteration = 0


def detect_failure_type(command: str, output: str) -> str | None:
    """Detect the type of failure from a command and its output.

    Args:
        command: The command that was run.
        output: The command output (stdout + stderr).

    Returns:
        Failure type string or None if not detectable.
    """
    command_lower = command.lower()
    output_lower = output.lower()

    # Pre-commit failures
    if "pre-commit" in command_lower:
        return "pre-commit"

    # dbt compile failures
    if "dbt" in command_lower and "compile" in command_lower:
        return "dbt-compile"

    # dbt run failures
    if "dbt" in command_lower and "run" in command_lower:
        return "dbt-run"

    # pytest failures
    if "pytest" in command_lower or "test" in command_lower:
        return "pytest"

    # mypy failures
    if "mypy" in command_lower:
        return "mypy"

    # ruff/black/isort failures
    if "ruff" in command_lower:
        return "ruff"
    if "black" in command_lower:
        return "black"
    if "isort" in command_lower:
        return "isort"

    # Detect from output patterns
    if "compilation error" in output_lower or "syntax error" in output_lower:
        return "syntax-error"

    if "import error" in output_lower or "module not found" in output_lower:
        return "import-error"

    return None


def is_failure_output(output: str, exit_code: int | None = None) -> bool:
    """Determine if command output indicates a failure.

    Args:
        output: Command output string.
        exit_code: Optional exit code.

    Returns:
        True if the output indicates a failure.
    """
    if exit_code is not None and exit_code != 0:
        return True

    failure_indicators = [
        "error:",
        "failed",
        "failure",
        "exception",
        "traceback",
        "❌",
        "error",
    ]

    output_lower = output.lower()
    return any(indicator in output_lower for indicator in failure_indicators)
