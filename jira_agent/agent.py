"""Core Jira automation agent using Claude."""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from anthropic import Anthropic

from .auth import AuthManager
from .clients.github_client import GitHubClient, format_pr_status
from .clients.jira_client import JiraClient, extract_text_from_adf, format_issue_summary
from .config import AgentSettings
from .environment import EnvironmentChecker, PackageInstaller, RepoRequirements
from .environment.requirements import RequirementsParser
from .learning import LearningCapture, detect_failure_type, is_failure_output
from .repo_config.schema import RepoConfig
from .tools.git_tools import GitTools, format_branch_name
from .utils.logger import TicketLogger
from .utils.progress import ProgressDisplay

logger = logging.getLogger(__name__)


class JiraAgent:
    """Autonomous agent for processing Jira tickets."""

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

        self._jira_client: JiraClient | None = None
        self._github_client: GitHubClient | None = None
        self._git_tools: GitTools | None = None

    async def _get_jira_client(self) -> JiraClient:
        """Get authenticated Jira client."""
        if self._jira_client is None:
            access_token = self.auth.jira.get_access_token()
            cloud_id = self.auth.jira.get_cloud_id()
            self._jira_client = JiraClient(cloud_id, access_token)
        return self._jira_client

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

    async def process_tickets(
        self,
        status_filter: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Process tickets from the Jira board.

        Args:
            status_filter: Filter tickets by status.
            limit: Maximum tickets to process.

        Returns:
            List of results for each ticket.
        """
        jira = await self._get_jira_client()

        # Get tickets from board
        if self.repo_config.jira.board_id:
            issues = await jira.get_board_issues(
                self.repo_config.jira.board_id,
                status=status_filter,
                max_results=limit,
            )
        else:
            # Use JQL search
            jql = f'project = "{self.repo_config.jira.project_key}"'
            if status_filter:
                jql += f' AND status = "{status_filter}"'
            issues = await jira.search_issues(jql, max_results=limit)

        logger.info(f"Found {len(issues)} tickets to process")

        results = []
        for issue in issues:
            ticket_key = issue["key"]
            result = await self.process_single_ticket(ticket_key)
            results.append(result)

        return results

    async def process_single_ticket(self, ticket_key: str, verbose: bool = True) -> dict[str, Any]:
        """Process a single Jira ticket.

        Args:
            ticket_key: Jira ticket key.
            verbose: If True, show detailed progress output.

        Returns:
            Processing result.
        """
        ticket_logger = TicketLogger(ticket_key)
        ticket_logger.info("Starting to process ticket")

        try:
            jira = await self._get_jira_client()

            # Get full ticket details
            issue = await jira.get_issue(ticket_key)
            issue_summary = format_issue_summary(issue)

            # Initialize progress display
            progress = ProgressDisplay(
                ticket_key=ticket_key,
                ticket_summary=issue_summary.get("summary", ""),
                verbose=verbose,
            )
            progress.start()

            # Check skip conditions
            progress.step("Checking skip conditions...", "🔍")
            skip_reason = await self._should_skip(ticket_key, issue)
            if skip_reason:
                ticket_logger.info(f"Skipping: {skip_reason}")
                progress.substep(f"Skipping: {skip_reason}", "⏭️")
                progress.complete(success=True)
                return {"ticket": ticket_key, "status": "skipped", "reason": skip_reason}

            # Clone/update repository
            progress.step("Preparing repository...", "📦")
            git = self._get_git_tools()
            repo_path = git.clone_repo(
                self.repo_config.repo.owner,
                self.repo_config.repo.name,
            )
            progress.substep(f"Repository ready at {repo_path}")
            ticket_logger.info(f"Repository ready at {repo_path}")

            # Pre-flight environment check
            progress.step("Checking environment...", "🔧")
            auto_install = getattr(self.settings, "auto_install_deps", True)
            env_result = await self.check_environment(
                repo_path,
                auto_install=auto_install,
                ticket_logger=ticket_logger,
            )

            if not env_result["ready"]:
                issues_str = "; ".join(env_result["issues"])
                ticket_logger.error(f"Environment not ready: {issues_str}")
                progress.error("Environment", issues_str)
                progress.complete(success=False, error="Environment not ready")
                return {
                    "ticket": ticket_key,
                    "status": "failed",
                    "error": f"Environment not ready: {issues_str}",
                }
            progress.substep("Environment ready", "✓")

            # Use Claude to analyze and implement the change
            progress.step("Running AI agent...", "🤖")
            result = await self._run_agent_for_ticket(
                issue_summary, repo_path, ticket_logger, progress
            )

            if result.get("pr_url"):
                # Add comment to Jira with PR link
                if not self.dry_run:
                    await jira.add_comment(
                        ticket_key,
                        f"Created PR: {result['pr_url']}",
                    )
                ticket_logger.info(f"Created PR: {result['pr_url']}")
                progress.pr_created(result["pr_url"])

            progress.complete(
                success=result.get("success", False),
                pr_url=result.get("pr_url"),
                error=result.get("error"),
            )

            return {
                "ticket": ticket_key,
                "status": "completed" if result.get("success") else "failed",
                "pr_url": result.get("pr_url"),
                "error": result.get("error"),
            }

        except Exception as e:
            ticket_logger.error(f"Failed to process: {e}", exc=e)
            if "progress" in locals():
                progress.complete(success=False, error=str(e))
            return {"ticket": ticket_key, "status": "failed", "error": str(e)}

    async def _should_skip(self, ticket_key: str, issue: dict) -> str | None:
        """Check if ticket should be skipped.

        Args:
            ticket_key: Ticket key.
            issue: Issue data.

        Returns:
            Skip reason or None if should process.
        """
        jira = await self._get_jira_client()

        # Check labels
        labels = issue.get("fields", {}).get("labels", [])
        skip_labels = set(self.repo_config.skip.labels)
        if skip_labels.intersection(set(labels)):
            return f"Has skip label: {skip_labels.intersection(set(labels))}"

        # Check comments for skip phrase or existing PR
        comments = await jira.get_issue_comments(ticket_key)
        for comment in comments:
            body = extract_text_from_adf(comment.get("body"))

            if self.repo_config.skip.comment_phrase in body:
                return f"Found skip phrase: {self.repo_config.skip.comment_phrase}"

            if "github.com" in body and "/pull/" in body:
                return "PR already exists"

        return None

    async def _run_agent_for_ticket(
        self,
        issue: dict[str, Any],
        repo_path: Path,
        ticket_logger: TicketLogger,
        progress: ProgressDisplay | None = None,
    ) -> dict[str, Any]:
        """Run Claude agent to implement ticket changes.

        Args:
            issue: Formatted issue summary.
            repo_path: Path to repository.
            ticket_logger: Logger for this ticket.
            progress: Optional progress display for user feedback.

        Returns:
            Result with pr_url if successful.
        """
        system_prompt = self._build_system_prompt(repo_path)
        user_prompt = self._build_task_prompt(issue)

        ticket_logger.info("Sending task to Claude Opus 4.5")
        if progress:
            progress.substep(f"Using model: {self.settings.claude_model}")

        if self.dry_run:
            ticket_logger.info("[DRY RUN] Would process with Claude")
            if progress:
                progress.substep("[DRY RUN] Skipping actual processing")
            return {"success": True, "dry_run": True}

        # Initialize learning capture
        learning_capture = LearningCapture(
            ticket_key=issue["key"],
            repo_name=self.repo_config.full_repo_name,
            enabled=getattr(self.settings, "learning_enabled", True),
        )

        # Define tools for the agent
        tools = self._get_agent_tools()

        # Initial message
        messages = [{"role": "user", "content": user_prompt}]

        # Agentic loop
        max_iterations = 50
        iteration = 0

        # Track recent tool calls to detect loops
        recent_tool_calls: list[tuple[str, str]] = []  # (tool_name, input_hash)
        max_repeated_calls = 3  # Max times same tool+input can repeat

        while iteration < max_iterations:
            iteration += 1
            learning_capture.current_iteration = iteration
            ticket_logger.debug(f"Agent iteration {iteration}")
            if progress:
                progress.iteration(iteration)

            response = self.claude.messages.create(
                model=self.settings.claude_model,
                max_tokens=8192,
                system=system_prompt,
                tools=tools,
                messages=messages,
            )

            # Extract any thinking/text from response
            if progress:
                for content_block in response.content:
                    if hasattr(content_block, "text") and content_block.text:
                        # Show Claude's thinking/reasoning (truncated)
                        text = content_block.text.strip()
                        if text and len(text) > 20:  # Only show substantial text
                            progress.thinking(text[:500])

            # Check for completion
            if response.stop_reason == "end_turn":
                ticket_logger.info("Agent completed task")
                if progress:
                    progress.substep("Agent finished processing", "✓")

                # Extract final result from response
                result = self._extract_result(response, messages)

                # Save any verified learnings
                if result.get("success"):
                    saved_paths = learning_capture.save_verified_learnings(
                        repo_path,
                        claude_client=self.claude,
                        conversation_messages=messages,
                    )
                    if saved_paths:
                        ticket_logger.info(f"Saved {len(saved_paths)} learnings")
                        if progress:
                            progress.substep(f"Captured {len(saved_paths)} learnings", "📚")

                return result

            # Process tool calls
            if response.stop_reason == "tool_use":
                # Add assistant response to messages
                messages.append({"role": "assistant", "content": response.content})

                # Execute tools and add results
                tool_results = []
                for content_block in response.content:
                    if content_block.type == "tool_use":
                        # Check for repeated tool calls (loop detection)
                        import hashlib
                        input_str = json.dumps(content_block.input, sort_keys=True)
                        input_hash = hashlib.md5(input_str.encode()).hexdigest()[:8]
                        call_signature = (content_block.name, input_hash)

                        # Count recent occurrences of this exact call
                        recent_count = recent_tool_calls[-10:].count(call_signature)
                        recent_tool_calls.append(call_signature)

                        if recent_count >= max_repeated_calls:
                            ticket_logger.warning(
                                f"Loop detected: {content_block.name} called {recent_count + 1} times with same input"
                            )
                            if progress:
                                progress.error(
                                    "Loop detected",
                                    f"{content_block.name} repeated {recent_count + 1} times - agent may be stuck"
                                )
                            # Return error to Claude to break the loop
                            result = (
                                f"Error: You have called {content_block.name} with the same input "
                                f"{recent_count + 1} times. Please try a different approach or "
                                f"complete the task with what you have. If an edit is failing, "
                                f"try reading the file first to get the exact text to replace."
                            )
                        else:
                            # Show tool call in progress
                            if progress:
                                progress.tool_call(content_block.name, content_block.input)

                            result = await self._execute_tool(
                                content_block.name,
                                content_block.input,
                                repo_path,
                                ticket_logger,
                                learning_capture,
                                progress,
                            )

                        # Show tool result
                        is_error = result.startswith("Error") or "Exit code: 1" in result
                        if progress:
                            progress.tool_result(
                                content_block.name,
                                success=not is_error,
                                output=result if is_error else "",
                            )

                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": content_block.id,
                            "content": result,
                        })

                messages.append({"role": "user", "content": tool_results})

        ticket_logger.warning("Agent reached max iterations")
        if progress:
            progress.error("Max iterations", f"Agent did not complete after {max_iterations} iterations")
        return {"success": False, "error": "Max iterations reached"}

    def _build_system_prompt(self, repo_path: Path) -> str:
        """Build system prompt for the agent."""
        return f"""You are an autonomous software engineering agent specializing in data engineering.
Your task is to implement changes to the repository based on Jira ticket requirements.

## Repository Information
- Owner: {self.repo_config.repo.owner}
- Name: {self.repo_config.repo.name}
- Path: {repo_path}
- Default Branch: {self.repo_config.repo.default_branch}
- PR Target: {self.repo_config.repo.pr_target_branch}

## Conventions
- Branch naming: {self.repo_config.branching.pattern}
- Commit style: {self.repo_config.commits.style}
- PR title pattern: {self.repo_config.pull_request.title_pattern}

## dbt Information
{"dbt is enabled for this repo. Projects: " + str([p.path for p in self.repo_config.dbt.projects]) if self.repo_config.dbt.enabled else "dbt is not enabled for this repo."}

## Workflow
1. Analyze the ticket requirements
2. Search the codebase for relevant files
3. Create a feature branch from {self.repo_config.repo.default_branch}
4. Make the necessary code changes
5. Run any validation (dbt compile, pre-commit)
6. Commit changes with a conventional commit message
7. Push the branch and create a PR
8. Return the PR URL

## Important Rules
- Always create a new branch from {self.repo_config.repo.default_branch}
- Follow existing code patterns in the repository
- Write clear, descriptive commit messages
- Create PRs targeting {self.repo_config.repo.pr_target_branch}
- If you encounter errors, try to fix them before giving up
"""

    def _build_task_prompt(self, issue: dict[str, Any]) -> str:
        """Build task prompt from issue."""
        return f"""Please implement the following Jira ticket:

**Ticket Key:** {issue['key']}
**Summary:** {issue['summary']}
**Type:** {issue.get('type', 'Unknown')}
**Priority:** {issue.get('priority', 'Unknown')}

**Description:**
{issue.get('description', 'No description provided.')}

Please:
1. Analyze what changes are needed
2. Search for relevant files in the repository
3. Create a feature branch
4. Implement the required changes
5. Commit and push the changes
6. Create a pull request

Start by exploring the codebase to understand the changes needed."""

    def _get_agent_tools(self) -> list[dict]:
        """Get tool definitions for the agent."""
        return [
            {
                "name": "read_file",
                "description": "Read the contents of a file in the repository",
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
        ]

    async def _execute_tool(
        self,
        tool_name: str,
        tool_input: dict,
        repo_path: Path,
        ticket_logger: TicketLogger,
        learning_capture: LearningCapture | None = None,
        progress: ProgressDisplay | None = None,
    ) -> str:
        """Execute a tool and return the result.

        Args:
            tool_name: Name of the tool.
            tool_input: Tool input parameters.
            repo_path: Repository path.
            ticket_logger: Logger for this ticket.
            learning_capture: Optional learning capture for tracking failures/fixes.
            progress: Optional progress display for user feedback.

        Returns:
            Tool result as string.
        """
        ticket_logger.debug(f"Executing tool: {tool_name}")

        try:
            git = self._get_git_tools()
            git.set_repo_path(repo_path)

            if tool_name == "read_file":
                file_path = repo_path / tool_input["path"]
                if not file_path.exists():
                    return f"Error: File not found: {tool_input['path']}"
                return file_path.read_text()

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
                    return result.stdout
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
                sha = git.commit(message)
                return f"Committed changes: {sha[:8]}"

            elif tool_name == "push_branch":
                branch = git.get_current_branch()
                git.push(branch)
                return f"Pushed branch {branch} to origin"

            elif tool_name == "create_pull_request":
                github = await self._get_github_client()
                branch = git.get_current_branch()
                pr = await github.create_pull_request(
                    title=tool_input["title"],
                    body=tool_input["body"],
                    head=branch,
                    base=self.repo_config.repo.pr_target_branch,
                )
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

            else:
                return f"Unknown tool: {tool_name}"

        except Exception as e:
            ticket_logger.error(f"Tool {tool_name} failed: {e}")
            return f"Error executing {tool_name}: {str(e)}"

    def _extract_result(self, response, messages: list) -> dict[str, Any]:
        """Extract final result from agent conversation.

        Args:
            response: Final Claude response.
            messages: Conversation history.

        Returns:
            Result dictionary.
        """
        import re

        # Look for PR URL in create_pull_request tool results only
        pr_url = None
        for msg in reversed(messages):
            # Check tool_result messages (user role contains tool results)
            if msg.get("role") == "user" and isinstance(msg.get("content"), list):
                for item in msg["content"]:
                    if isinstance(item, dict) and item.get("type") == "tool_result":
                        content = item.get("content", "")
                        # Only match if it looks like a create_pull_request result
                        # Format: "Created PR #N: https://github.com/..."
                        if content.startswith("Created PR #"):
                            match = re.search(r"https://github\.com/[^\s]+/pull/\d+", content)
                            if match:
                                pr_url = match.group(0)
                                break

            # Also check assistant messages for the tool_use that triggered this
            if msg.get("role") == "assistant" and isinstance(msg.get("content"), list):
                for item in msg["content"]:
                    if hasattr(item, "name") and item.name == "create_pull_request":
                        # Found the create_pull_request call, the next user message has the result
                        pass

            if pr_url:
                break

        return {
            "success": pr_url is not None,
            "pr_url": pr_url,
        }

    async def check_pr_status(self, pr_number: int) -> dict[str, Any]:
        """Check status of a pull request.

        Args:
            pr_number: PR number.

        Returns:
            PR status summary.
        """
        github = await self._get_github_client()

        pr = await github.get_pull_request(pr_number)
        checks = await github.get_check_runs(pr["head"]["sha"])

        return format_pr_status(pr, checks)

    async def fix_ci_failures(self, pr_number: int) -> dict[str, Any]:
        """Attempt to fix CI failures on a PR.

        Args:
            pr_number: PR number.

        Returns:
            Fix result.
        """
        github = await self._get_github_client()

        pr = await github.get_pull_request(pr_number)
        checks = await github.get_check_runs(pr["head"]["sha"])

        failed_checks = [c for c in checks if c["conclusion"] == "failure"]
        if not failed_checks:
            return {"fixed": True, "message": "No failed checks"}

        # Clone repo and checkout PR branch
        git = self._get_git_tools()
        repo_path = git.clone_repo(
            self.repo_config.repo.owner,
            self.repo_config.repo.name,
        )
        git.checkout_branch(pr["head"]["ref"])

        # Try auto-fix strategies
        for strategy in self.repo_config.ci.auto_fix:
            if strategy == "pre-commit":
                success, output = git.run_pre_commit()
                if not success and git.has_changes():
                    git.stage_files()
                    git.commit("style: auto-fix pre-commit issues")
                    git.push()
                    return {
                        "fixed": True,
                        "strategy": "pre-commit",
                        "commit_sha": git.repo.head.commit.hexsha,
                    }

        return {"fixed": False, "error": "Could not auto-fix CI failures"}

    async def transition_ticket_to_done(self, ticket_key: str) -> dict[str, Any]:
        """Transition a Jira ticket to Done status.

        Args:
            ticket_key: Jira ticket key.

        Returns:
            Result with success status.
        """
        logger.info(f"Transitioning {ticket_key} to Done")

        try:
            jira = await self._get_jira_client()

            # Get available transitions
            transitions = await jira.get_issue_transitions(ticket_key)

            # Find "Done" transition (common names: Done, Closed, Complete)
            done_transition = None
            done_names = ["done", "closed", "complete", "resolved"]

            for t in transitions:
                if t.get("name", "").lower() in done_names:
                    done_transition = t
                    break

            if not done_transition:
                # Log available transitions for debugging
                available = [t.get("name") for t in transitions]
                logger.warning(f"No 'Done' transition found. Available: {available}")
                return {
                    "success": False,
                    "error": f"No 'Done' transition available. Available: {available}",
                }

            # Perform transition
            await jira.transition_issue(ticket_key, done_transition["id"])

            # Add comment
            await jira.add_comment(
                ticket_key,
                f"PR merged. Automatically transitioned to {done_transition['name']}.",
            )

            return {
                "success": True,
                "transition": done_transition["name"],
                "ticket": ticket_key,
            }

        except Exception as e:
            logger.error(f"Failed to transition {ticket_key}: {e}")
            return {"success": False, "error": str(e)}

    async def close(self) -> None:
        """Clean up resources."""
        if self._jira_client:
            await self._jira_client.close()
        if self._github_client:
            await self._github_client.close()
