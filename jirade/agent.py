"""Core PR/CI automation agent using Claude.

Jira ticket processing was removed in v0.10.0 — Atlassian interaction now
happens through the Rovo MCP connector in Claude Code. What remains here is
the GitHub-facing machinery: PR status checks, CI auto-fix, and review-comment
handling.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from anthropic import Anthropic

from .auth import AuthManager
# dbt Cloud client removed in v0.4.0 - CI now runs locally via Databricks
# Keep stubs for backwards compatibility with agent code paths
DbtCloudClient = None
RunStatus = None


def format_run_errors_for_prompt(errors):
    return ""
from .clients.github_client import GitHubClient, format_pr_status
from .config import AgentSettings
from .environment import EnvironmentChecker, PackageInstaller, RepoRequirements
from .environment.requirements import RequirementsParser
from .learning import LearningCapture, detect_failure_type, is_failure_output
from .pr_tracker import PRTracker
from .repo_config.schema import RepoConfig
from .tools.git_tools import GitTools
from .utils.logger import TicketLogger
from .utils.progress import ProgressDisplay

logger = logging.getLogger(__name__)


class JiraAgent:
    """Agent for PR status checks, CI fixes, and review-comment handling."""

    def __init__(
        self,
        settings: AgentSettings,
        repo_config: RepoConfig,
        dry_run: bool = False,
    ):
        """Initialize the Jira agent.

        Args:
            settings: Global agent settings.
            repo_config: Configuration for target repository.
            dry_run: If True, preview actions without making changes.
        """
        self.settings = settings
        self.repo_config = repo_config
        self.dry_run = dry_run

        self.auth = AuthManager(settings)
        self.claude = Anthropic(api_key=settings.anthropic_api_key)

        self._github_client: GitHubClient | None = None
        self._git_tools: GitTools | None = None
    async def _get_github_client(self) -> GitHubClient:
        """Get authenticated GitHub client."""
        if self._github_client is None:
            token = self.auth.github.get_access_token()
            self._github_client = GitHubClient(
                token,
                self.repo_config.repo.owner,
                self.repo_config.repo.name,
            )
        return self._github_client

    def _get_git_tools(self) -> GitTools:
        """Get Git tools instance."""
        if self._git_tools is None:
            token = self.auth.github.get_access_token()
            self._git_tools = GitTools(self.settings.workspace_dir, token)
        return self._git_tools

    async def _get_dbt_cloud_client(self):
        """dbt Cloud removed in v0.4.0. Always returns None."""
        return None

    def _get_dbt_cloud_ci_job_id(self) -> str | None:
        """dbt Cloud removed in v0.4.0. Always returns None."""
        return None

    def _get_dbt_cloud_lookback_days(self) -> int:
        """Get event-time lookback days."""
        return self.settings.dbt_event_time_lookback_days

    async def check_environment(
        self,
        repo_path: Path,
        auto_install: bool = False,
        ticket_logger: TicketLogger | None = None,
    ) -> dict[str, Any]:
        """Check and optionally set up environment for the repository.

        Args:
            repo_path: Path to the cloned repository.
            auto_install: If True, automatically install missing dependencies.
            ticket_logger: Optional logger for the ticket.

        Returns:
            Dict with 'ready' bool and 'issues' list.
        """
        log = ticket_logger or logger
        issues = []

        # Check system tools
        log.info("Checking system environment...")
        checker = EnvironmentChecker()
        report = checker.check_for_repo(repo_path, self.repo_config)

        if report.missing_required:
            log.warning(f"Missing required tools: {', '.join(report.missing_required)}")

            if auto_install:
                installer = PackageInstaller(repo_path, auto_confirm=True)
                for tool in report.missing_required:
                    log.info(f"Installing {tool}...")
                    result = installer.install_system_tool(tool)
                    if result.success:
                        log.info(f"Installed {tool}")
                    else:
                        issues.append(f"Failed to install {tool}: {result.error}")
            else:
                for tool in report.missing_required:
                    tool_check = next((t for t in report.tools if t.name == tool), None)
                    hint = tool_check.install_hint if tool_check else f"Install {tool}"
                    issues.append(f"Missing {tool}: {hint}")

        # Check repository requirements
        log.info("Checking repository requirements...")
        parser = RequirementsParser(repo_path)
        reqs = parser.parse_all()

        missing_python, missing_node = parser.get_missing_packages()

        if missing_python or missing_node:
            total_missing = len(missing_python) + len(missing_node)
            log.info(f"Found {total_missing} missing packages")

            if auto_install:
                log.info("Installing repository dependencies...")
                installer = PackageInstaller(repo_path, auto_confirm=True)
                results = installer.install_repo_requirements()

                for result in results:
                    if result.success:
                        log.info(f"Installed: {result.package}")
                    else:
                        issues.append(f"Failed to install {result.package}: {result.error}")
            else:
                # Provide setup commands instead
                if reqs.setup_commands:
                    issues.append(f"Run setup commands: {', '.join(reqs.setup_commands)}")
                else:
                    if missing_python:
                        issues.append(f"Missing {len(missing_python)} Python packages")
                    if missing_node:
                        issues.append(f"Missing {len(missing_node)} Node.js packages")

        # Check for pre-commit hooks
        if (repo_path / ".pre-commit-config.yaml").exists():
            pre_commit_installed = (repo_path / ".git" / "hooks" / "pre-commit").exists()
            if not pre_commit_installed:
                if auto_install:
                    installer = PackageInstaller(repo_path, auto_confirm=True)
                    result = installer.setup_pre_commit()
                    if result.success:
                        log.info("Installed pre-commit hooks")
                    else:
                        issues.append(f"Failed to install pre-commit hooks: {result.error}")
                else:
                    issues.append("Pre-commit hooks not installed. Run: pre-commit install")

        ready = len([i for i in issues if "Failed" in i or "Missing" in i]) == 0

        if ready:
            log.info("Environment is ready")
        else:
            log.warning(f"Environment has {len(issues)} issues")

        return {"ready": ready, "issues": issues}

    def _get_agent_tools(self) -> list[dict]:
        """Get tool definitions for the agent."""
        return [
            {
                "name": "preview_file",
                "description": "Preview the first 50 lines of a file. Use this BEFORE read_file to check if a file is relevant.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "File path relative to repository root",
                        },
                    },
                    "required": ["path"],
                },
            },
            {
                "name": "read_file",
                "description": "Read the full contents of a file. Only use this AFTER preview_file confirms the file is relevant. Limited to 500 lines.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "File path relative to repository root",
                        },
                    },
                    "required": ["path"],
                },
            },
            {
                "name": "write_file",
                "description": "Write content to a file in the repository (creates or overwrites)",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "File path relative to repository root",
                        },
                        "content": {
                            "type": "string",
                            "description": "Content to write to the file",
                        },
                    },
                    "required": ["path", "content"],
                },
            },
            {
                "name": "edit_file",
                "description": "Edit a file by replacing specific text. Use this for targeted edits instead of rewriting the entire file.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "File path relative to repository root",
                        },
                        "old_string": {
                            "type": "string",
                            "description": "The exact text to find and replace (must match exactly, including whitespace)",
                        },
                        "new_string": {
                            "type": "string",
                            "description": "The text to replace it with",
                        },
                    },
                    "required": ["path", "old_string", "new_string"],
                },
            },
            {
                "name": "list_directory",
                "description": "List files and directories in a path",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Directory path relative to repository root",
                            "default": ".",
                        },
                    },
                },
            },
            {
                "name": "search_files",
                "description": "Search for files matching a pattern",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": "Glob pattern to match files (e.g., '**/*.sql')",
                        },
                    },
                    "required": ["pattern"],
                },
            },
            {
                "name": "search_content",
                "description": "Search file contents for a pattern",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": "Text or regex pattern to search for",
                        },
                        "file_pattern": {
                            "type": "string",
                            "description": "Glob pattern to filter files (e.g., '*.sql')",
                            "default": "*",
                        },
                    },
                    "required": ["pattern"],
                },
            },
            {
                "name": "create_branch",
                "description": "Create a new git branch from the default branch",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "branch_name": {
                            "type": "string",
                            "description": "Name for the new branch",
                        },
                    },
                    "required": ["branch_name"],
                },
            },
            {
                "name": "commit_changes",
                "description": "Stage and commit all changes",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "Commit message",
                        },
                    },
                    "required": ["message"],
                },
            },
            {
                "name": "push_branch",
                "description": "Push the current branch to origin",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                },
            },
            {
                "name": "create_pull_request",
                "description": "Create a pull request for the current branch",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": "PR title",
                        },
                        "body": {
                            "type": "string",
                            "description": "PR description",
                        },
                    },
                    "required": ["title", "body"],
                },
            },
            {
                "name": "run_command",
                "description": "Run a shell command in the repository",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "Command to run",
                        },
                    },
                    "required": ["command"],
                },
            },
            {
                "name": "run_formatter",
                "description": "Run a code formatter on files. Supports: sqlfmt (SQL), black (Python), isort (Python imports), yamlfmt (YAML). This will modify files in place.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "formatter": {
                            "type": "string",
                            "enum": ["sqlfmt", "black", "isort", "yamlfmt"],
                            "description": "Formatter to run",
                        },
                        "files": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of file paths to format (relative to repo root)",
                        },
                    },
                    "required": ["formatter", "files"],
                },
            },
        ]

    async def _execute_tool(
        self,
        tool_name: str,
        tool_input: dict,
        repo_path: Path,
        ticket_logger: TicketLogger,
        learning_capture: LearningCapture | None = None,
        progress: ProgressDisplay | None = None,
        ticket_key: str | None = None,
    ) -> str:
        """Execute a tool and return the result.

        Args:
            tool_name: Name of the tool.
            tool_input: Tool input parameters.
            repo_path: Repository path.
            ticket_logger: Logger for this ticket.
            learning_capture: Optional learning capture for tracking failures/fixes.
            progress: Optional progress display for user feedback.
            ticket_key: Jira ticket key for tracking PRs.

        Returns:
            Tool result as string.
        """
        ticket_logger.debug(f"Executing tool: {tool_name}")

        try:
            git = self._get_git_tools()
            git.set_repo_path(repo_path)

            if tool_name == "preview_file":
                file_path = repo_path / tool_input["path"]
                if not file_path.exists():
                    return f"Error: File not found: {tool_input['path']}"
                try:
                    lines = file_path.read_text().splitlines()
                    total_lines = len(lines)
                    preview = "\n".join(lines[:50])
                    if total_lines > 50:
                        preview += f"\n\n[... {total_lines - 50} more lines. Use read_file if you need the full content.]"
                    return preview
                except UnicodeDecodeError:
                    return f"Error: Cannot read binary file: {tool_input['path']}"

            elif tool_name == "read_file":
                file_path = repo_path / tool_input["path"]
                if not file_path.exists():
                    return f"Error: File not found: {tool_input['path']}"
                try:
                    lines = file_path.read_text().splitlines()
                    total_lines = len(lines)
                    max_lines = 500
                    if total_lines > max_lines:
                        content = "\n".join(lines[:max_lines])
                        content += f"\n\n[File truncated. Showing {max_lines} of {total_lines} lines. If you need a specific section, use search_content to find it.]"
                        return content
                    return "\n".join(lines)
                except UnicodeDecodeError:
                    return f"Error: Cannot read binary file: {tool_input['path']}"

            elif tool_name == "write_file":
                file_path = repo_path / tool_input["path"]
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text(tool_input["content"])

                # Track file modifications as potential fix attempts
                if learning_capture:
                    for failure_type in learning_capture.get_verified_fix_types():
                        pass  # Already verified, no action needed
                    # Record fix attempts for any pending failures
                    for failure_type in list(learning_capture._failures.keys()):
                        if not learning_capture._fix_attempts.get(failure_type):
                            learning_capture.record_fix_attempt(
                                failure_type=failure_type,
                                solution_description=f"Modified file: {tool_input['path']}",
                                files_modified=[tool_input["path"]],
                            )

                return f"Successfully wrote to {tool_input['path']}"

            elif tool_name == "edit_file":
                file_path = repo_path / tool_input["path"]
                if not file_path.exists():
                    return f"Error: File not found: {tool_input['path']}"

                content = file_path.read_text()
                old_string = tool_input["old_string"]
                new_string = tool_input["new_string"]

                # Check if old_string exists in file
                if old_string not in content:
                    # Provide helpful error message
                    return (
                        f"Error: Could not find the text to replace in {tool_input['path']}. "
                        f"The old_string must match exactly (including whitespace and newlines). "
                        f"First 100 chars of old_string: {repr(old_string[:100])}"
                    )

                # Check for multiple occurrences
                count = content.count(old_string)
                if count > 1:
                    return (
                        f"Error: Found {count} occurrences of old_string in {tool_input['path']}. "
                        f"Please provide a more specific/unique string to replace."
                    )

                # Perform the replacement
                new_content = content.replace(old_string, new_string, 1)
                file_path.write_text(new_content)

                # Track file modifications
                if learning_capture:
                    for failure_type in list(learning_capture._failures.keys()):
                        if not learning_capture._fix_attempts.get(failure_type):
                            learning_capture.record_fix_attempt(
                                failure_type=failure_type,
                                solution_description=f"Edited file: {tool_input['path']}",
                                files_modified=[tool_input["path"]],
                            )

                return f"Successfully edited {tool_input['path']}"

            elif tool_name == "list_directory":
                dir_path = repo_path / tool_input.get("path", ".")
                if not dir_path.exists():
                    return f"Error: Directory not found: {tool_input.get('path', '.')}"
                items = list(dir_path.iterdir())
                result = []
                for item in sorted(items):
                    prefix = "d" if item.is_dir() else "f"
                    result.append(f"{prefix} {item.name}")
                return "\n".join(result)

            elif tool_name == "search_files":
                pattern = tool_input["pattern"]
                matches = list(repo_path.glob(pattern))
                if not matches:
                    return f"No files found matching pattern: {pattern}"
                return "\n".join(str(m.relative_to(repo_path)) for m in matches[:100])

            elif tool_name == "search_content":
                import subprocess

                pattern = tool_input["pattern"]
                file_pattern = tool_input.get("file_pattern", "*")
                result = subprocess.run(
                    ["grep", "-r", "-l", pattern, "--include", file_pattern, "."],
                    cwd=repo_path,
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0:
                    files = result.stdout.strip().split("\n")
                    total = len(files)
                    max_results = 30
                    if total > max_results:
                        files = files[:max_results]
                        return f"Found {total} files matching '{pattern}'. Showing first {max_results}:\n" + "\n".join(files) + f"\n\n[Use a more specific pattern to narrow results]"
                    return f"Found {total} files matching '{pattern}':\n" + "\n".join(files)
                return f"No matches found for pattern: {pattern}"

            elif tool_name == "create_branch":
                branch_name = tool_input["branch_name"]
                git.create_branch_from(
                    branch_name,
                    self.repo_config.repo.default_branch,
                )
                return f"Created and checked out branch: {branch_name}"

            elif tool_name == "commit_changes":
                message = tool_input["message"]
                git.stage_files()
                try:
                    sha = git.commit(message)
                    return f"Committed changes: {sha[:8]}"
                except Exception as e:
                    error_str = str(e)
                    # Check if this is a pre-commit hook failure
                    if "hook" in error_str.lower() or "pre-commit" in error_str.lower():
                        # Try to identify and fix common pre-commit issues
                        logger.info("Pre-commit hook failed, attempting auto-fix...")

                        # Run pre-commit autofixes
                        try:
                            code, stdout, stderr = git.run_command(
                                ["pre-commit", "run", "--all-files"],
                                check=False
                            )
                            # Pre-commit may have modified files, re-stage
                            git.stage_files()

                            # Try commit again
                            sha = git.commit(message)
                            return f"Committed changes (after pre-commit auto-fix): {sha[:8]}"
                        except Exception as retry_error:
                            # If still failing, provide helpful error
                            raise Exception(
                                f"Pre-commit hook failed and auto-fix unsuccessful. "
                                f"Original error: {error_str[:200]}. "
                                f"Retry error: {str(retry_error)[:200]}"
                            )
                    else:
                        raise

            elif tool_name == "push_branch":
                branch = git.get_current_branch()
                git.push(branch)
                return f"Pushed branch {branch} to origin"

            elif tool_name == "create_pull_request":
                github = await self._get_github_client()
                branch = git.get_current_branch()

                # Update dbt Cloud CI job with fresh event-time dates before creating PR
                dbt_cloud = await self._get_dbt_cloud_client()
                ci_job_id = self._get_dbt_cloud_ci_job_id()
                if dbt_cloud and ci_job_id:
                    try:
                        await dbt_cloud.update_ci_job_event_time_dates(
                            job_id=int(ci_job_id),
                            lookback_days=self._get_dbt_cloud_lookback_days(),
                        )
                        logger.info("Updated dbt Cloud CI job event-time dates")
                    except Exception as e:
                        logger.warning(f"Failed to update dbt Cloud CI job dates: {e}")

                # Add [jirade] suffix so any jirade instance can identify its PRs
                title = f"{tool_input['title']} [jirade]"
                try:
                    pr = await github.create_pull_request(
                        title=title,
                        body=tool_input["body"],
                        head=branch,
                        base=self.repo_config.repo.pr_target_branch,
                    )
                except Exception as e:
                    error_str = str(e)
                    # Handle 422 errors - usually means PR exists or no diff
                    if "422" in error_str:
                        # Check if PR already exists for this branch
                        existing_prs = await github.list_pull_requests(
                            state="open", head=f"{self.repo_config.repo.owner}:{branch}"
                        )
                        if existing_prs:
                            pr = existing_prs[0]
                            return f"PR already exists: #{pr['number']}: {pr['html_url']}"

                        # Check if there are any changes to push
                        if not git.has_changes() and not git.has_unpushed_commits():
                            return "No changes to create PR - branch may already be merged or have no diff"

                        # Re-raise if we can't handle it
                        raise Exception(f"Failed to create PR (422 error): {error_str[:200]}")
                    raise
                # Track the PR for monitoring
                if ticket_key:
                    tracker = PRTracker()
                    tracker.add_pr(
                        pr_number=pr["number"],
                        pr_url=pr["html_url"],
                        repo=self.repo_config.full_repo_name,
                        ticket_key=ticket_key,
                        branch=branch,
                    )

                # Trigger dbt Cloud CI (uses job's configured state:modified+1 selection)
                if dbt_cloud and ci_job_id:
                    try:
                        git_sha = pr.get("head", {}).get("sha")
                        run = await dbt_cloud.trigger_ci_run(
                            job_id=int(ci_job_id),
                            pr_number=pr["number"],
                            git_sha=git_sha,
                            git_branch=branch,
                        )
                        logger.info(
                            f"Triggered dbt Cloud CI run {run.get('id')} for PR #{pr['number']}"
                        )
                    except Exception as e:
                        logger.warning(f"Failed to trigger dbt Cloud CI: {e}")

                return f"Created PR #{pr['number']}: {pr['html_url']}"

            elif tool_name == "run_command":
                import shlex

                command_str = tool_input["command"]
                command = shlex.split(command_str)
                code, stdout, stderr = git.run_command(command)
                result = f"Exit code: {code}\n"
                if stdout:
                    result += f"stdout:\n{stdout}\n"
                if stderr:
                    result += f"stderr:\n{stderr}\n"

                # Track failures and verifications for learning
                combined_output = f"{stdout}\n{stderr}".strip()
                failure_type = detect_failure_type(command_str, combined_output)

                if failure_type:
                    if code != 0 or is_failure_output(combined_output, code):
                        # Record failure and show in progress
                        if learning_capture:
                            learning_capture.record_failure(
                                failure_type=failure_type,
                                error_message=combined_output[:2000],
                                command=command_str,
                            )
                        if progress:
                            progress.error(failure_type, combined_output[:500])
                            progress.healing_start(failure_type)

                    elif learning_capture and learning_capture.has_pending_failure(failure_type):
                        # Same type of command now succeeds - verify the fix
                        learning_capture.verify_fix_success(failure_type)
                        if progress:
                            progress.healing_success(failure_type)

                return result

            elif tool_name == "run_formatter":
                formatter = tool_input["formatter"]
                files = tool_input["files"]

                # Map formatter to command
                formatter_commands = {
                    "sqlfmt": ["pipx", "run", "sqlfmt"],
                    "black": ["pipx", "run", "black"],
                    "isort": ["pipx", "run", "isort"],
                    "yamlfmt": ["yamlfmt"],  # Assumes yamlfmt is installed
                }

                if formatter not in formatter_commands:
                    return f"Error: Unknown formatter: {formatter}"

                base_cmd = formatter_commands[formatter]
                full_paths = [str(repo_path / f) for f in files]

                try:
                    import subprocess
                    result = subprocess.run(
                        base_cmd + full_paths,
                        cwd=repo_path,
                        capture_output=True,
                        text=True,
                        timeout=120,
                    )
                    output = f"Exit code: {result.returncode}\n"
                    if result.stdout:
                        output += f"stdout:\n{result.stdout}\n"
                    if result.stderr:
                        output += f"stderr:\n{result.stderr}\n"

                    if result.returncode == 0:
                        output += f"\nSuccessfully formatted {len(files)} file(s) with {formatter}"
                    return output
                except FileNotFoundError:
                    return f"Error: {formatter} not found. Try installing with: pipx install {formatter}"
                except subprocess.TimeoutExpired:
                    return f"Error: {formatter} timed out after 120 seconds"

            else:
                return f"Unknown tool: {tool_name}"

        except Exception as e:
            ticket_logger.error(f"Tool {tool_name} failed: {e}")
            return f"Error executing {tool_name}: {str(e)}"

    async def check_pr_status(self, pr_number: int) -> dict[str, Any]:
        """Check status of a pull request.

        Args:
            pr_number: PR number.

        Returns:
            PR status summary.
        """
        github = await self._get_github_client()

        pr = await github.get_pull_request(pr_number)
        sha = pr["head"]["sha"]

        # Fetch both check runs (GitHub Actions) and commit statuses (CircleCI, etc.)
        checks = await github.get_check_runs(sha)
        statuses = await github.get_combined_status(sha)

        return format_pr_status(pr, checks, statuses)

    async def fix_ci_failures(self, pr_number: int) -> dict[str, Any]:
        """Attempt to fix CI failures on a PR.

        Args:
            pr_number: PR number.

        Returns:
            Fix result.
        """
        github = await self._get_github_client()

        pr = await github.get_pull_request(pr_number)
        sha = pr["head"]["sha"]

        # Fetch both check runs (GitHub Actions) and commit statuses (CircleCI, etc.)
        checks = await github.get_check_runs(sha)
        statuses = await github.get_combined_status(sha)

        # Check for failures in both APIs
        failed_checks = [c["name"] for c in checks if c.get("conclusion") == "failure"]
        failed_statuses = [
            s["context"] for s in statuses.get("statuses", [])
            if s.get("state") in ("failure", "error")
        ]
        all_failures = failed_checks + failed_statuses

        # Check dbt Cloud API directly for failures (may not be reported to GitHub)
        dbt_cloud_errors: list[dict[str, Any]] = []
        dbt_cloud_client = await self._get_dbt_cloud_client()

        if dbt_cloud_client:
            try:
                # Get the CI job ID
                ci_job_id = self.settings.dbt_cloud_ci_job_id
                if not ci_job_id and self.settings.dbt_cloud_project_id:
                    # Try to find CI job automatically
                    ci_job = await dbt_cloud_client.find_ci_job(
                        int(self.settings.dbt_cloud_project_id)
                    )
                    if ci_job:
                        ci_job_id = str(ci_job["id"])

                if ci_job_id:
                    # Get recent runs for this PR
                    runs = await dbt_cloud_client.get_runs_for_pr(
                        int(ci_job_id), pr_number, limit=5
                    )

                    # Find the most recent failed run
                    for run in runs:
                        if run.get("status") == RunStatus.ERROR:
                            run_id = run["id"]
                            logger.info(f"Found failed dbt Cloud run: {run_id}")
                            dbt_cloud_errors = await dbt_cloud_client.get_run_errors(run_id)
                            if dbt_cloud_errors:
                                logger.info(f"Retrieved {len(dbt_cloud_errors)} dbt Cloud errors")
                                # Add to all_failures if not already present
                                if not any("dbt" in f.lower() for f in all_failures):
                                    all_failures.append("dbt Cloud CI")
                            break
            except Exception as e:
                logger.warning(f"Failed to fetch dbt Cloud errors: {e}")

        if not all_failures:
            return {"fixed": True, "message": "No failed checks"}

        logger.info(f"Found CI failures: {all_failures}")

        # Check if dbt Cloud errors are infrastructure/permissions issues (not fixable by code)
        if dbt_cloud_errors:
            unfixable_keywords = [
                "permission_denied", "access denied", "unauthorized",
                "does not have", "not authorized", "insufficient privileges",
                "schema does not exist", "database does not exist",
                "connection refused", "timeout", "network error",
                "credential", "authentication failed"
            ]
            for error in dbt_cloud_errors:
                error_msg = error.get("message", "").lower()
                if any(kw in error_msg for kw in unfixable_keywords):
                    logger.info(f"dbt Cloud error is infrastructure/permissions issue, cannot auto-fix: {error.get('message', '')[:100]}")
                    return {
                        "fixed": False,
                        "error": f"Infrastructure/permissions issue (not fixable by code): {error.get('message', '')[:200]}",
                        "unfixable": True,
                    }

        # Clone repo and checkout PR branch
        git = self._get_git_tools()
        repo_path = git.clone_repo(
            self.repo_config.repo.owner,
            self.repo_config.repo.name,
        )
        git.checkout_branch(pr["head"]["ref"])

        # Get changed files for context (use pr_target_branch as that's what we merge to)
        changed_files = git.get_changed_files_from_branch(
            self.repo_config.repo.pr_target_branch
        )

        # Try running formatters directly on changed files (more reliable than pre-commit)
        sql_files = [f for f in changed_files if f.endswith(".sql")]
        py_files = [f for f in changed_files if f.endswith(".py")]

        formatted_any = False

        # Run sqlfmt on SQL files
        if sql_files:
            logger.info(f"Running sqlfmt on {len(sql_files)} SQL files")
            try:
                import subprocess
                full_paths = [str(repo_path / f) for f in sql_files]
                result = subprocess.run(
                    ["pipx", "run", "sqlfmt"] + full_paths,
                    cwd=repo_path,
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                if result.returncode == 0 and git.has_changes():
                    formatted_any = True
                    logger.info("sqlfmt made changes")
            except (FileNotFoundError, subprocess.TimeoutExpired) as e:
                logger.warning(f"sqlfmt failed: {e}")

        # Run black on Python files
        if py_files:
            logger.info(f"Running black on {len(py_files)} Python files")
            try:
                import subprocess
                full_paths = [str(repo_path / f) for f in py_files]
                result = subprocess.run(
                    ["pipx", "run", "black"] + full_paths,
                    cwd=repo_path,
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                if result.returncode == 0 and git.has_changes():
                    formatted_any = True
                    logger.info("black made changes")
            except (FileNotFoundError, subprocess.TimeoutExpired) as e:
                logger.warning(f"black failed: {e}")

        # If formatters made changes, commit and push (skip local hooks - CI will verify)
        if formatted_any and git.has_changes():
            git.stage_files()
            git.commit("style: auto-format files", skip_hooks=True)
            git.push()
            return {
                "fixed": True,
                "strategy": "formatters",
                "commit_sha": git.repo.head.commit.hexsha,
            }

        # If no formatting changes needed, the CI failure might be something else
        if not git.has_changes():
            logger.info("Formatters ran but no changes needed - CI failure may be environment-specific")
            # Check if the failure name suggests it's just a formatting issue
            formatting_keywords = ["format", "sqlfmt", "black", "isort", "lint", "style"]
            is_formatting_failure = any(
                kw in f.lower() for f in all_failures for kw in formatting_keywords
            )
            if not is_formatting_failure:
                # Not a formatting issue, try Claude
                logger.info("CI failure doesn't appear to be formatting-related, using Claude")
            else:
                # Formatting issue but no changes needed locally - might be CI env specific
                return {
                    "fixed": False,
                    "error": "Formatters found no issues locally. CI failure may be environment-specific or already fixed.",
                }

        # If simple strategies didn't work, use Claude to analyze and fix
        logger.info("Using Claude to analyze CI failures")

        # Create a ticket logger for the CI fix operation
        ci_logger = TicketLogger(f"PR-{pr_number}")

        return await self._fix_ci_with_claude(
            pr, repo_path, git, all_failures, changed_files, ci_logger, dbt_cloud_errors
        )

    async def _fix_ci_with_claude(
        self,
        pr: dict,
        repo_path: Path,
        git: GitTools,
        failures: list[str],
        changed_files: list[str],
        ci_logger: TicketLogger,
        dbt_cloud_errors: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Use Claude to analyze CI failures and attempt fixes.

        Args:
            pr: PR data.
            repo_path: Path to repository.
            git: Git tools instance.
            failures: List of failed check names.
            changed_files: List of changed files in the PR.
            ci_logger: Logger for CI fix operations.
            dbt_cloud_errors: Optional list of dbt Cloud error details.

        Returns:
            Fix result.
        """
        # Build dbt Cloud error context if available
        dbt_error_context = ""
        if dbt_cloud_errors:
            dbt_error_context = format_run_errors_for_prompt(dbt_cloud_errors)

        # Build context about the failure
        system_prompt = f"""You are an autonomous CI failure fixer for a data engineering repository.

## Repository Information
- Owner: {self.repo_config.repo.owner}
- Name: {self.repo_config.repo.name}
- Path: {repo_path}

## PR Information
- PR #{pr['number']}: {pr['title']}
- Branch: {pr['head']['ref']}

## Failed CI Checks
{chr(10).join(f'- {f}' for f in failures)}

## Changed Files
{chr(10).join(f'- {f}' for f in changed_files[:20])}

{dbt_error_context}

## Your Task
Analyze the CI failures and fix them. Common issues include:
- SQL syntax errors (read the error message, find the file, fix the SQL)
- Missing column references (check the upstream model for correct column names)
- SQL formatting (use `run_formatter` with sqlfmt)
- Python formatting (use `run_formatter` with black/isort)
- YAML formatting (use `run_formatter` with yamlfmt)
- Linting errors

## For dbt Cloud CI Errors
If there are dbt Cloud errors listed above:
1. Read the error message carefully - it contains the exact issue
2. Find the model file mentioned in the error (unique_id shows the path)
3. Read the file to understand the current code
4. Fix the SQL error (syntax, column references, etc.)
5. If needed, also check upstream models referenced in the error

## IMPORTANT: Use the run_formatter tool for formatting
For formatting issues, use the `run_formatter` tool instead of trying to manually fix files:
- For .sql files: `run_formatter` with formatter="sqlfmt"
- For .py files: `run_formatter` with formatter="black" then "isort"
- For .yml/.yaml files: `run_formatter` with formatter="yamlfmt"

## Workflow
1. If there are dbt Cloud errors, fix those SQL issues first
2. If CI failed with "pre-commit" or "sqlfmt" in the name, it's likely formatting
3. Run the appropriate formatter on the changed files using `run_formatter`
4. After making changes, commit with an appropriate message

Do NOT read files to manually fix formatting - use the formatters directly.
"""

        # Build user prompt based on error type
        if dbt_cloud_errors:
            user_prompt = f"""The CI check(s) failed: {', '.join(failures)}

The changed files are:
{chr(10).join(f'- {f}' for f in changed_files[:20])}

**dbt Cloud CI has reported errors.** The detailed error messages are in the system prompt above.

Please:
1. Read the dbt Cloud error messages carefully
2. Find the model file(s) with errors
3. Fix the SQL issues (syntax errors, missing columns, etc.)
4. Run sqlfmt on any SQL files you change
5. Commit the fixes"""
        else:
            user_prompt = f"""The CI check(s) failed: {', '.join(failures)}

The changed files are:
{chr(10).join(f'- {f}' for f in changed_files[:20])}

Since "pre-commit" is in the failure name, this is likely a formatting issue.
Please run the appropriate formatter on the changed files:
- For .sql files, use run_formatter with formatter="sqlfmt"
- For .py files, use run_formatter with formatter="black"

After formatting, commit the changes."""

        # Run agent loop to fix CI
        tools = self._get_agent_tools()
        messages = [{"role": "user", "content": user_prompt}]

        max_iterations = 20
        for iteration in range(max_iterations):
            ci_logger.info(f"CI fix iteration {iteration + 1}")

            response = self.claude.messages.create(
                model=self.settings.claude_model,
                max_tokens=8192,
                system=system_prompt,
                tools=tools,
                messages=messages,
            )

            ci_logger.info(f"Response stop_reason: {response.stop_reason}")

            if response.stop_reason == "end_turn":
                # Check if we made any changes
                if git.has_changes():
                    git.stage_files()
                    git.commit("fix: resolve CI failures", skip_hooks=True)
                    git.push()
                    ci_logger.info("Successfully fixed CI issues")
                    return {
                        "fixed": True,
                        "strategy": "claude",
                        "commit_sha": git.repo.head.commit.hexsha,
                    }
                ci_logger.warning("Claude finished but no changes were made")
                return {"fixed": False, "error": "Claude could not identify fixes"}

            if response.stop_reason == "tool_use":
                messages.append({"role": "assistant", "content": response.content})
                tool_results = []

                for content_block in response.content:
                    if content_block.type == "tool_use":
                        ci_logger.debug(f"Tool: {content_block.name}")
                        result = await self._execute_tool(
                            content_block.name,
                            content_block.input,
                            repo_path,
                            ci_logger,
                        )
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": content_block.id,
                            "content": result,
                        })

                messages.append({"role": "user", "content": tool_results})

        return {"fixed": False, "error": "Max iterations reached"}

    async def address_review_comments(
        self,
        pr_number: int,
        comments: list[dict[str, Any]],
        is_issue_comment: bool = False,
    ) -> dict[str, Any]:
        """Address review comments on a PR by making code changes.

        Args:
            pr_number: PR number.
            comments: List of review or issue comments to address.
            is_issue_comment: Whether these are issue comments (general PR comments).

        Returns:
            Result with success status and details.
        """
        if not comments:
            return {"success": False, "error": "No comments provided"}

        github = await self._get_github_client()

        # Get PR details
        pr = await github.get_pull_request(pr_number)
        branch = pr["head"]["ref"]

        logger.info(f"Addressing {len(comments)} review comment(s) on PR #{pr_number}")

        # Clone repo and checkout PR branch
        git = self._get_git_tools()
        repo_path = git.clone_repo(
            self.repo_config.repo.owner,
            self.repo_config.repo.name,
        )
        git.checkout_branch(branch)

        # Get changed files for context
        changed_files = git.get_changed_files_from_branch(
            self.repo_config.repo.pr_target_branch
        )

        # Format comments for Claude
        comments_context = []
        for comment in comments:
            body = comment.get("body", "")
            user = comment.get("user", {}).get("login", "unknown")

            # Check if this is a review comment (has file/line context) or issue comment
            if comment.get("path"):
                # Review comment with file context
                file_path = comment.get("path", "unknown")
                line = comment.get("line") or comment.get("original_line", "?")
                diff_hunk = comment.get("diff_hunk", "")

                comments_context.append(f"""
### Review comment by @{user} on `{file_path}` (line {line})
{body}

Diff context:
```
{diff_hunk}
```
""")
            else:
                # General PR comment (issue comment)
                comments_context.append(f"""
### General comment by @{user}
{body}

(This is a general PR comment, not attached to a specific file. You may need to search the codebase to find relevant files.)
""")

        # Create logger for this operation
        comment_logger = TicketLogger(f"PR-{pr_number}-comments")

        # Build prompt for Claude
        system_prompt = f"""You are an autonomous code reviewer assistant for a data engineering repository.

## Repository Information
- Owner: {self.repo_config.repo.owner}
- Name: {self.repo_config.repo.name}
- Path: {repo_path}

## PR Information
- PR #{pr['number']}: {pr['title']}
- Branch: {branch}

## Your Task
Address the review comments by making the requested code changes. After making changes:
1. Verify the changes are correct
2. The changes will be committed and pushed automatically

## Changed Files in This PR
{chr(10).join(f'- {f}' for f in changed_files[:20])}

## Guidelines
- Read the relevant files before making changes
- Make precise, targeted changes that address the feedback
- Do not make unrelated changes
- For SQL files, ensure proper formatting
- For Python files, follow existing code style
"""

        user_prompt = f"""Please address the following review comments on PR #{pr_number}:

{chr(10).join(comments_context)}

Read the relevant files and make the requested changes."""

        # Run agent loop
        tools = self._get_agent_tools()
        messages = [{"role": "user", "content": user_prompt}]

        max_iterations = 25
        addressed_comments = []

        for iteration in range(max_iterations):
            comment_logger.info(f"Comment addressing iteration {iteration + 1}")

            response = self.claude.messages.create(
                model=self.settings.claude_model,
                max_tokens=8192,
                system=system_prompt,
                tools=tools,
                messages=messages,
            )

            comment_logger.info(f"Response stop_reason: {response.stop_reason}")

            if response.stop_reason == "end_turn":
                # Check if we made any changes
                if git.has_changes():
                    git.stage_files()

                    # Create commit message
                    comment_count = len(comments)
                    commit_msg = f"fix: address {comment_count} review comment(s)"

                    git.commit(commit_msg, skip_hooks=True)
                    git.push()

                    comment_logger.info("Successfully addressed review comments")

                    # Reply to each comment
                    for comment in comments:
                        try:
                            if comment.get("path"):
                                # Review comment - reply inline
                                await github.reply_to_review_comment(
                                    pr_number,
                                    comment["id"],
                                    "✅ Addressed in the latest commit. [jirade]",
                                )
                            else:
                                # Issue comment - add general PR comment
                                await github.add_pr_comment(
                                    pr_number,
                                    f"✅ Addressed the feedback from @{comment.get('user', {}).get('login', 'unknown')} in the latest commit. [jirade]",
                                )
                            addressed_comments.append(comment["id"])
                        except Exception as e:
                            comment_logger.warning(f"Failed to reply to comment: {e}")

                    return {
                        "success": True,
                        "addressed": len(addressed_comments),
                        "commit_sha": git.repo.head.commit.hexsha,
                    }

                comment_logger.warning("Claude finished but no changes were made")

                # Still reply to comments explaining no changes were needed
                for comment in comments:
                    try:
                        if comment.get("path"):
                            # Review comment - reply inline
                            await github.reply_to_review_comment(
                                pr_number,
                                comment["id"],
                                "🤔 I analyzed this comment but couldn't determine what changes to make. A human may need to review. [jirade]",
                            )
                        else:
                            # Issue comment - add general PR comment
                            await github.add_pr_comment(
                                pr_number,
                                f"🤔 I analyzed the feedback from @{comment.get('user', {}).get('login', 'unknown')} but couldn't determine what changes to make. A human may need to review. [jirade]",
                            )
                    except Exception as e:
                        comment_logger.warning(f"Failed to reply to comment: {e}")

                return {"success": False, "error": "No changes were made"}

            if response.stop_reason == "tool_use":
                messages.append({"role": "assistant", "content": response.content})
                tool_results = []

                for content_block in response.content:
                    if content_block.type == "tool_use":
                        comment_logger.debug(f"Tool: {content_block.name}")
                        result = await self._execute_tool(
                            content_block.name,
                            content_block.input,
                            repo_path,
                            comment_logger,
                        )
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": content_block.id,
                            "content": result,
                        })

                messages.append({"role": "user", "content": tool_results})

        return {"success": False, "error": "Max iterations reached"}

    async def close(self) -> None:
        """Clean up resources."""
        if self._github_client:
            await self._github_client.close()
        pass  # dbt Cloud client removed in v0.4.0
