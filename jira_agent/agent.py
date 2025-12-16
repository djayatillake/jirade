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
from .repo_config.schema import RepoConfig
from .tools.git_tools import GitTools, format_branch_name
from .utils.logger import TicketLogger

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

    async def process_single_ticket(self, ticket_key: str) -> dict[str, Any]:
        """Process a single Jira ticket.

        Args:
            ticket_key: Jira ticket key.

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

            # Check skip conditions
            skip_reason = await self._should_skip(ticket_key, issue)
            if skip_reason:
                ticket_logger.info(f"Skipping: {skip_reason}")
                return {"ticket": ticket_key, "status": "skipped", "reason": skip_reason}

            # Clone/update repository
            git = self._get_git_tools()
            repo_path = git.clone_repo(
                self.repo_config.repo.owner,
                self.repo_config.repo.name,
            )
            ticket_logger.info(f"Repository ready at {repo_path}")

            # Use Claude to analyze and implement the change
            result = await self._run_agent_for_ticket(issue_summary, repo_path, ticket_logger)

            if result.get("pr_url"):
                # Add comment to Jira with PR link
                if not self.dry_run:
                    await jira.add_comment(
                        ticket_key,
                        f"Created PR: {result['pr_url']}",
                    )
                ticket_logger.info(f"Created PR: {result['pr_url']}")

            return {
                "ticket": ticket_key,
                "status": "completed" if result.get("success") else "failed",
                "pr_url": result.get("pr_url"),
                "error": result.get("error"),
            }

        except Exception as e:
            ticket_logger.error(f"Failed to process: {e}", exc=e)
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
    ) -> dict[str, Any]:
        """Run Claude agent to implement ticket changes.

        Args:
            issue: Formatted issue summary.
            repo_path: Path to repository.
            ticket_logger: Logger for this ticket.

        Returns:
            Result with pr_url if successful.
        """
        system_prompt = self._build_system_prompt(repo_path)
        user_prompt = self._build_task_prompt(issue)

        ticket_logger.info("Sending task to Claude Opus 4.5")

        if self.dry_run:
            ticket_logger.info("[DRY RUN] Would process with Claude")
            return {"success": True, "dry_run": True}

        # Define tools for the agent
        tools = self._get_agent_tools()

        # Initial message
        messages = [{"role": "user", "content": user_prompt}]

        # Agentic loop
        max_iterations = 50
        iteration = 0

        while iteration < max_iterations:
            iteration += 1
            ticket_logger.debug(f"Agent iteration {iteration}")

            response = self.claude.messages.create(
                model=self.settings.claude_model,
                max_tokens=8192,
                system=system_prompt,
                tools=tools,
                messages=messages,
            )

            # Check for completion
            if response.stop_reason == "end_turn":
                ticket_logger.info("Agent completed task")
                # Extract final result from response
                return self._extract_result(response, messages)

            # Process tool calls
            if response.stop_reason == "tool_use":
                # Add assistant response to messages
                messages.append({"role": "assistant", "content": response.content})

                # Execute tools and add results
                tool_results = []
                for content_block in response.content:
                    if content_block.type == "tool_use":
                        result = await self._execute_tool(
                            content_block.name,
                            content_block.input,
                            repo_path,
                            ticket_logger,
                        )
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": content_block.id,
                            "content": result,
                        })

                messages.append({"role": "user", "content": tool_results})

        ticket_logger.warning("Agent reached max iterations")
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
                "description": "Write content to a file in the repository",
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
    ) -> str:
        """Execute a tool and return the result.

        Args:
            tool_name: Name of the tool.
            tool_input: Tool input parameters.
            repo_path: Repository path.
            ticket_logger: Logger for this ticket.

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
                return f"Successfully wrote to {tool_input['path']}"

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

                command = shlex.split(tool_input["command"])
                code, stdout, stderr = git.run_command(command)
                result = f"Exit code: {code}\n"
                if stdout:
                    result += f"stdout:\n{stdout}\n"
                if stderr:
                    result += f"stderr:\n{stderr}\n"
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
        # Look for PR URL in messages
        pr_url = None
        for msg in reversed(messages):
            if isinstance(msg.get("content"), list):
                for item in msg["content"]:
                    if isinstance(item, dict) and item.get("type") == "tool_result":
                        content = item.get("content", "")
                        if "github.com" in content and "/pull/" in content:
                            # Extract URL
                            import re

                            match = re.search(r"https://github\.com/[^\s]+/pull/\d+", content)
                            if match:
                                pr_url = match.group(0)
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

    async def close(self) -> None:
        """Clean up resources."""
        if self._jira_client:
            await self._jira_client.close()
        if self._github_client:
            await self._github_client.close()
