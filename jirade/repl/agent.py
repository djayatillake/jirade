"""REPL Agent for interactive jirade sessions."""

import asyncio
import subprocess
from pathlib import Path
from typing import Any

from anthropic import Anthropic
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from ..agent import JiraAgent
from ..auth import AuthManager
from ..clients.github_client import GitHubClient
from ..clients.jira_client import JiraClient, format_issue_summary
from ..config import AgentSettings
from ..pr_tracker import PRTracker
from ..repo_config.schema import RepoConfig
from ..tools.git_tools import GitTools
from .session import Session
from .tools import get_tools


class REPLAgent:
    """Interactive REPL agent for jirade."""

    def __init__(
        self,
        settings: AgentSettings,
        repo_config: RepoConfig,
        repo_path: Path,
        model: str = "claude-sonnet-4-20250514",
    ):
        """Initialize the REPL agent.

        Args:
            settings: Agent settings with API keys.
            repo_config: Repository configuration.
            repo_path: Path to the repository.
            model: Claude model to use.
        """
        self.settings = settings
        self.repo_config = repo_config
        self.repo_path = repo_path
        self.model = model

        self.claude = Anthropic(api_key=settings.anthropic_api_key)
        self.auth = AuthManager(settings)
        self.session = Session()
        self.console = Console()

        # Lazy-initialized clients
        self._jira_client: JiraClient | None = None
        self._github_client: GitHubClient | None = None
        self._git_tools: GitTools | None = None

    def _get_system_prompt(self) -> str:
        """Build the system prompt for the REPL agent."""
        return f"""You are jirade, an AI assistant for data engineering tasks. You help users manage Jira tickets, GitHub pull requests, and work with code in their repository.

## Repository Context
- Owner: {self.repo_config.repo.owner}
- Name: {self.repo_config.repo.name}
- Default Branch: {self.repo_config.repo.default_branch}
- PR Target Branch: {self.repo_config.repo.pr_target_branch}
- Jira Project: {self.repo_config.jira.project_key}
- Trigger Status: "{self.repo_config.agent.status}"
- Done Status: "{self.repo_config.agent.done_status}"

## Your Capabilities
You have access to tools for:
- **Jira**: List tickets, get ticket details, transition status, add comments, search with JQL
- **GitHub**: List PRs, get PR details, check CI status, create PRs
- **Files**: Read, write, edit files, list directories, search for files and content
- **Git**: Check status, create branches, commit changes, push branches
- **Processing**: Fully process tickets (autonomous code changes), fix CI failures
- **Commands**: Run shell commands (dbt, pre-commit, etc.)

## Guidelines
1. Be conversational and helpful. Explain what you're doing and why.
2. When using tools, briefly explain the action before executing.
3. Format output nicely - use tables for lists, code blocks for code.
4. For ticket processing, confirm with the user before making significant changes.
5. If something fails, explain what happened and suggest alternatives.
6. Use conventional commits style for commit messages.
7. PRs created by jirade should have [jirade] prefix in the title.

## dbt Information
{"dbt is enabled. Projects: " + str([p.path for p in self.repo_config.dbt.projects]) if self.repo_config.dbt.enabled else "dbt is not enabled for this repo."}
"""

    async def _get_jira_client(self) -> JiraClient:
        """Get or create the Jira client."""
        if self._jira_client is None:
            jira_auth = self.auth.jira.get_credentials()
            self._jira_client = JiraClient(
                base_url=jira_auth["base_url"],
                email=jira_auth["email"],
                api_token=jira_auth["api_token"],
            )
        return self._jira_client

    async def _get_github_client(self) -> GitHubClient:
        """Get or create the GitHub client."""
        if self._github_client is None:
            self._github_client = GitHubClient(
                self.settings.github_token,
                self.repo_config.repo.owner,
                self.repo_config.repo.name,
            )
        return self._github_client

    def _get_git_tools(self) -> GitTools:
        """Get or create git tools."""
        if self._git_tools is None:
            self._git_tools = GitTools(self.settings.github_token)
            self._git_tools.set_repo_path(self.repo_path)
        return self._git_tools

    def _display_tool_call(self, name: str, inputs: dict[str, Any]) -> None:
        """Display a tool call to the user."""
        # Format inputs for display
        params = ", ".join(
            f'{k}="{v}"' if isinstance(v, str) else f"{k}={v}"
            for k, v in inputs.items()
            if v is not None
        )
        tool_text = Text()
        tool_text.append("  ⚡ Using: ", style="yellow")
        tool_text.append(f"{name}", style="cyan bold")
        tool_text.append(f"({params})", style="dim")
        self.console.print(tool_text)

    def _display_progress(self, message: str) -> None:
        """Display a progress message."""
        self.console.print(f"  → {message}", style="dim")

    def _display_success(self, message: str) -> None:
        """Display a success message."""
        self.console.print(f"✓ {message}", style="green bold")

    def _display_error(self, message: str) -> None:
        """Display an error message."""
        self.console.print(f"✗ {message}", style="red bold")

    async def _execute_tool(self, name: str, inputs: dict[str, Any]) -> str:
        """Execute a tool and return the result."""
        self._display_tool_call(name, inputs)

        try:
            # Jira tools
            if name == "list_jira_tickets":
                return await self._tool_list_jira_tickets(inputs)
            elif name == "get_jira_ticket":
                return await self._tool_get_jira_ticket(inputs)
            elif name == "transition_ticket":
                return await self._tool_transition_ticket(inputs)
            elif name == "add_ticket_comment":
                return await self._tool_add_ticket_comment(inputs)
            elif name == "search_tickets":
                return await self._tool_search_tickets(inputs)

            # GitHub tools
            elif name == "list_pull_requests":
                return await self._tool_list_pull_requests(inputs)
            elif name == "get_pull_request":
                return await self._tool_get_pull_request(inputs)
            elif name == "check_ci_status":
                return await self._tool_check_ci_status(inputs)

            # File tools
            elif name == "read_file":
                return self._tool_read_file(inputs)
            elif name == "write_file":
                return self._tool_write_file(inputs)
            elif name == "edit_file":
                return self._tool_edit_file(inputs)
            elif name == "list_directory":
                return self._tool_list_directory(inputs)
            elif name == "search_files":
                return self._tool_search_files(inputs)
            elif name == "search_content":
                return self._tool_search_content(inputs)

            # Git tools
            elif name == "git_status":
                return self._tool_git_status(inputs)
            elif name == "create_branch":
                return self._tool_create_branch(inputs)
            elif name == "commit_changes":
                return self._tool_commit_changes(inputs)
            elif name == "push_branch":
                return self._tool_push_branch(inputs)
            elif name == "create_pull_request":
                return await self._tool_create_pull_request(inputs)

            # Process tools
            elif name == "process_ticket":
                return await self._tool_process_ticket(inputs)
            elif name == "fix_ci":
                return await self._tool_fix_ci(inputs)

            # Utility tools
            elif name == "run_command":
                return self._tool_run_command(inputs)

            else:
                return f"Error: Unknown tool '{name}'"

        except Exception as e:
            return f"Error executing {name}: {str(e)}"

    # =========== Jira Tool Implementations ===========

    async def _tool_list_jira_tickets(self, inputs: dict[str, Any]) -> str:
        """List Jira tickets."""
        jira = await self._get_jira_client()
        status = inputs.get("status")
        limit = inputs.get("limit", 20)

        try:
            if self.repo_config.jira.board_id:
                issues = await jira.get_board_issues(
                    self.repo_config.jira.board_id,
                    status=status,
                    limit=limit,
                )
            else:
                jql = f'project = "{self.repo_config.jira.project_key}"'
                if status:
                    jql += f' AND status = "{status}"'
                jql += " ORDER BY updated DESC"
                issues = await jira.search_issues(jql, limit=limit)

            if not issues:
                return "No tickets found matching the criteria."

            lines = ["| Key | Summary | Status | Assignee |", "|-----|---------|--------|----------|"]
            for issue in issues:
                key = issue.get("key", "")
                fields = issue.get("fields", {})
                summary = fields.get("summary", "")[:50]
                status_name = fields.get("status", {}).get("name", "")
                assignee = fields.get("assignee", {})
                assignee_name = assignee.get("displayName", "Unassigned") if assignee else "Unassigned"
                lines.append(f"| {key} | {summary} | {status_name} | {assignee_name} |")

            return "\n".join(lines)
        except Exception as e:
            return f"Error listing tickets: {e}"

    async def _tool_get_jira_ticket(self, inputs: dict[str, Any]) -> str:
        """Get full ticket details."""
        jira = await self._get_jira_client()
        key = inputs["key"]

        try:
            issue = await jira.get_issue(key)
            summary = format_issue_summary(issue)
            return (
                f"## {summary['key']}: {summary['summary']}\n\n"
                f"**Status:** {summary['status']}\n"
                f"**Assignee:** {summary['assignee'] or 'Unassigned'}\n"
                f"**Priority:** {summary.get('priority', 'None')}\n\n"
                f"### Description\n{summary['description']}\n\n"
                f"### Acceptance Criteria\n{summary.get('acceptance_criteria', 'None specified')}"
            )
        except Exception as e:
            return f"Error getting ticket {key}: {e}"

    async def _tool_transition_ticket(self, inputs: dict[str, Any]) -> str:
        """Transition a ticket to a new status."""
        jira = await self._get_jira_client()
        key = inputs["key"]
        target_status = inputs["status"]

        try:
            transitions = await jira.get_issue_transitions(key)
            transition = next(
                (t for t in transitions if t["name"].lower() == target_status.lower()),
                None,
            )
            if not transition:
                available = ", ".join(t["name"] for t in transitions)
                return f"Cannot transition to '{target_status}'. Available: {available}"

            await jira.transition_issue(key, transition["id"])
            return f"Transitioned {key} to '{target_status}'"
        except Exception as e:
            return f"Error transitioning {key}: {e}"

    async def _tool_add_ticket_comment(self, inputs: dict[str, Any]) -> str:
        """Add a comment to a ticket."""
        jira = await self._get_jira_client()
        key = inputs["key"]
        comment = inputs["comment"]

        try:
            await jira.add_comment(key, comment)
            return f"Added comment to {key}"
        except Exception as e:
            return f"Error adding comment to {key}: {e}"

    async def _tool_search_tickets(self, inputs: dict[str, Any]) -> str:
        """Search tickets with JQL."""
        jira = await self._get_jira_client()
        jql = inputs["jql"]
        limit = inputs.get("limit", 20)

        try:
            issues = await jira.search_issues(jql, limit=limit)
            if not issues:
                return "No tickets found matching the query."

            lines = ["| Key | Summary | Status |", "|-----|---------|--------|"]
            for issue in issues:
                key = issue.get("key", "")
                fields = issue.get("fields", {})
                summary = fields.get("summary", "")[:50]
                status_name = fields.get("status", {}).get("name", "")
                lines.append(f"| {key} | {summary} | {status_name} |")

            return "\n".join(lines)
        except Exception as e:
            return f"Error searching tickets: {e}"

    # =========== GitHub Tool Implementations ===========

    async def _tool_list_pull_requests(self, inputs: dict[str, Any]) -> str:
        """List pull requests."""
        github = await self._get_github_client()
        state = inputs.get("state", "open")
        jirade_only = inputs.get("jirade_only", False)

        try:
            prs = await github.list_pull_requests(state=state, per_page=50)

            if jirade_only:
                prs = [pr for pr in prs if pr.get("title", "").startswith("[jirade]")]

            if not prs:
                return "No pull requests found."

            lines = ["| # | Title | State | Author |", "|---|-------|-------|--------|"]
            for pr in prs:
                number = pr.get("number", "")
                title = pr.get("title", "")[:50]
                pr_state = pr.get("state", "")
                author = pr.get("user", {}).get("login", "")
                lines.append(f"| #{number} | {title} | {pr_state} | {author} |")

            return "\n".join(lines)
        except Exception as e:
            return f"Error listing PRs: {e}"

    async def _tool_get_pull_request(self, inputs: dict[str, Any]) -> str:
        """Get PR details."""
        github = await self._get_github_client()
        number = inputs["number"]

        try:
            pr = await github.get_pull_request(number)
            reviews = await github.get_pr_reviews(number)

            state = pr.get("state", "")
            if pr.get("merged_at"):
                state = "merged"

            review_summary = "No reviews yet"
            if reviews:
                approved = sum(1 for r in reviews if r.get("state") == "APPROVED")
                changes_requested = sum(1 for r in reviews if r.get("state") == "CHANGES_REQUESTED")
                review_summary = f"{approved} approved, {changes_requested} changes requested"

            return (
                f"## PR #{number}: {pr.get('title', '')}\n\n"
                f"**State:** {state}\n"
                f"**Author:** {pr.get('user', {}).get('login', '')}\n"
                f"**Branch:** {pr.get('head', {}).get('ref', '')} → {pr.get('base', {}).get('ref', '')}\n"
                f"**Reviews:** {review_summary}\n"
                f"**URL:** {pr.get('html_url', '')}\n\n"
                f"### Description\n{pr.get('body', 'No description')}"
            )
        except Exception as e:
            return f"Error getting PR #{number}: {e}"

    async def _tool_check_ci_status(self, inputs: dict[str, Any]) -> str:
        """Check CI status for a PR."""
        github = await self._get_github_client()
        pr_number = inputs["pr_number"]

        try:
            pr = await github.get_pull_request(pr_number)
            sha = pr.get("head", {}).get("sha", "")

            checks = await github.get_check_runs(sha)
            statuses = await github.get_combined_status(sha)

            lines = ["| Check | Status | Conclusion |", "|-------|--------|------------|"]

            for check in checks:
                name = check.get("name", "")
                status = check.get("status", "")
                conclusion = check.get("conclusion", "pending")
                lines.append(f"| {name} | {status} | {conclusion} |")

            for status in statuses.get("statuses", []):
                context = status.get("context", "")
                state = status.get("state", "")
                lines.append(f"| {context} | - | {state} |")

            if len(lines) == 2:
                return "No CI checks found for this PR."

            return "\n".join(lines)
        except Exception as e:
            return f"Error checking CI status: {e}"

    # =========== File Tool Implementations ===========

    def _tool_read_file(self, inputs: dict[str, Any]) -> str:
        """Read a file."""
        path = self.repo_path / inputs["path"]
        max_lines = inputs.get("max_lines", 500)

        if not path.exists():
            return f"Error: File not found: {inputs['path']}"

        try:
            content = path.read_text()
            lines = content.splitlines()
            if len(lines) > max_lines:
                content = "\n".join(lines[:max_lines])
                content += f"\n\n[... {len(lines) - max_lines} more lines truncated]"
            return f"```\n{content}\n```"
        except UnicodeDecodeError:
            return f"Error: Cannot read binary file: {inputs['path']}"

    def _tool_write_file(self, inputs: dict[str, Any]) -> str:
        """Write a file."""
        path = self.repo_path / inputs["path"]
        content = inputs["content"]

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
            return f"Wrote {len(content)} bytes to {inputs['path']}"
        except Exception as e:
            return f"Error writing file: {e}"

    def _tool_edit_file(self, inputs: dict[str, Any]) -> str:
        """Edit a file with search/replace."""
        path = self.repo_path / inputs["path"]
        old_string = inputs["old_string"]
        new_string = inputs["new_string"]

        if not path.exists():
            return f"Error: File not found: {inputs['path']}"

        try:
            content = path.read_text()
            if old_string not in content:
                return f"Error: Could not find the specified text in {inputs['path']}"

            count = content.count(old_string)
            if count > 1:
                return f"Error: Found {count} matches. Please provide more context to make the match unique."

            new_content = content.replace(old_string, new_string)
            path.write_text(new_content)
            return f"Edited {inputs['path']}"
        except Exception as e:
            return f"Error editing file: {e}"

    def _tool_list_directory(self, inputs: dict[str, Any]) -> str:
        """List directory contents."""
        path = self.repo_path / inputs.get("path", ".")

        if not path.exists():
            return f"Error: Directory not found: {inputs.get('path', '.')}"

        try:
            entries = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name))
            lines = []
            for entry in entries[:100]:  # Limit to 100 entries
                prefix = "📁 " if entry.is_dir() else "📄 "
                lines.append(f"{prefix}{entry.name}")

            if len(list(path.iterdir())) > 100:
                lines.append(f"... and {len(list(path.iterdir())) - 100} more")

            return "\n".join(lines)
        except Exception as e:
            return f"Error listing directory: {e}"

    def _tool_search_files(self, inputs: dict[str, Any]) -> str:
        """Search for files by glob pattern."""
        pattern = inputs["pattern"]

        try:
            matches = list(self.repo_path.glob(pattern))[:50]
            if not matches:
                return f"No files found matching '{pattern}'"

            lines = [str(m.relative_to(self.repo_path)) for m in matches]
            result = "\n".join(lines)
            if len(list(self.repo_path.glob(pattern))) > 50:
                result += f"\n... and more (showing first 50)"
            return result
        except Exception as e:
            return f"Error searching files: {e}"

    def _tool_search_content(self, inputs: dict[str, Any]) -> str:
        """Search for content in files."""
        pattern = inputs["pattern"]
        file_pattern = inputs.get("file_pattern", "")

        try:
            cmd = ["grep", "-r", "-n", "-I", pattern, str(self.repo_path)]
            if file_pattern:
                cmd = ["grep", "-r", "-n", "-I", f"--include={file_pattern}", pattern, str(self.repo_path)]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            output = result.stdout

            if not output:
                return f"No matches found for '{pattern}'"

            # Trim to reasonable size
            lines = output.splitlines()[:30]
            # Make paths relative
            lines = [line.replace(str(self.repo_path) + "/", "") for line in lines]
            result_text = "\n".join(lines)
            if len(output.splitlines()) > 30:
                result_text += f"\n... and {len(output.splitlines()) - 30} more matches"
            return result_text
        except subprocess.TimeoutExpired:
            return "Search timed out - try a more specific pattern"
        except Exception as e:
            return f"Error searching content: {e}"

    # =========== Git Tool Implementations ===========

    def _tool_git_status(self, inputs: dict[str, Any]) -> str:
        """Get git status."""
        git = self._get_git_tools()
        try:
            branch = git.get_current_branch()
            has_changes = git.has_changes()
            code, stdout, stderr = git.run_command(["git", "status", "--short"])
            return f"**Branch:** {branch}\n**Has changes:** {has_changes}\n\n```\n{stdout}\n```"
        except Exception as e:
            return f"Error getting git status: {e}"

    def _tool_create_branch(self, inputs: dict[str, Any]) -> str:
        """Create a new branch."""
        git = self._get_git_tools()
        name = inputs["name"]
        from_branch = inputs.get("from_branch", self.repo_config.repo.default_branch)

        try:
            git.create_branch_from(name, from_branch)
            return f"Created and checked out branch: {name}"
        except Exception as e:
            return f"Error creating branch: {e}"

    def _tool_commit_changes(self, inputs: dict[str, Any]) -> str:
        """Commit changes."""
        git = self._get_git_tools()
        message = inputs["message"]

        try:
            git.stage_files()
            sha = git.commit(message)
            return f"Committed: {sha[:8]}"
        except Exception as e:
            return f"Error committing: {e}"

    def _tool_push_branch(self, inputs: dict[str, Any]) -> str:
        """Push branch to remote."""
        git = self._get_git_tools()

        try:
            branch = git.get_current_branch()
            git.push(branch)
            return f"Pushed branch: {branch}"
        except Exception as e:
            return f"Error pushing: {e}"

    async def _tool_create_pull_request(self, inputs: dict[str, Any]) -> str:
        """Create a pull request."""
        github = await self._get_github_client()
        git = self._get_git_tools()
        title = f"[jirade] {inputs['title']}"
        body = inputs["body"]

        try:
            branch = git.get_current_branch()
            pr = await github.create_pull_request(
                title=title,
                body=body,
                head=branch,
                base=self.repo_config.repo.pr_target_branch,
            )

            # Track the PR
            tracker = PRTracker()
            # Extract ticket key from title if present
            import re
            match = re.search(rf"\b({self.repo_config.jira.project_key}-\d+)\b", title, re.IGNORECASE)
            ticket_key = match.group(1).upper() if match else "UNKNOWN"
            tracker.add_pr(
                pr_number=pr["number"],
                pr_url=pr["html_url"],
                repo=self.repo_config.full_repo_name,
                ticket_key=ticket_key,
                branch=branch,
            )

            return f"Created PR #{pr['number']}: {pr['html_url']}"
        except Exception as e:
            return f"Error creating PR: {e}"

    # =========== Process Tool Implementations ===========

    async def _tool_process_ticket(self, inputs: dict[str, Any]) -> str:
        """Process a ticket using the JiraAgent."""
        key = inputs["key"]

        self._display_progress(f"Starting autonomous processing of {key}...")

        try:
            agent = JiraAgent(self.settings, self.repo_config, dry_run=False)
            result = await agent.process_single_ticket(key)

            if result.get("status") == "completed":
                pr_url = result.get("pr_url", "")
                return f"Successfully processed {key}!\nPR: {pr_url}"
            elif result.get("status") == "skipped":
                return f"Skipped {key}: {result.get('reason', 'Unknown reason')}"
            else:
                return f"Failed to process {key}: {result.get('error', 'Unknown error')}"
        except Exception as e:
            return f"Error processing ticket: {e}"

    async def _tool_fix_ci(self, inputs: dict[str, Any]) -> str:
        """Fix CI failures on a PR."""
        pr_number = inputs["pr_number"]

        self._display_progress(f"Attempting to fix CI on PR #{pr_number}...")

        try:
            agent = JiraAgent(self.settings, self.repo_config, dry_run=False)
            result = await agent.fix_ci_failures(pr_number)

            if result.get("fixed"):
                return f"Fixed CI issues on PR #{pr_number}"
            else:
                return f"Could not fix CI: {result.get('error', 'Unknown error')}"
        except Exception as e:
            return f"Error fixing CI: {e}"

    # =========== Utility Tool Implementations ===========

    def _tool_run_command(self, inputs: dict[str, Any]) -> str:
        """Run a shell command."""
        command = inputs["command"]

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=120,
                cwd=self.repo_path,
            )

            output = ""
            if result.stdout:
                output += f"stdout:\n{result.stdout}\n"
            if result.stderr:
                output += f"stderr:\n{result.stderr}\n"
            output += f"Exit code: {result.returncode}"

            return output
        except subprocess.TimeoutExpired:
            return "Command timed out after 120 seconds"
        except Exception as e:
            return f"Error running command: {e}"

    # =========== Main Chat Loop ===========

    async def chat(self, user_input: str) -> None:
        """Process a single chat turn."""
        self.session.add_user_message(user_input)

        tools = get_tools()
        max_iterations = 20

        for _ in range(max_iterations):
            response = self.claude.messages.create(
                model=self.model,
                max_tokens=4096,
                system=self._get_system_prompt(),
                tools=tools,
                messages=self.session.get_messages(),
            )

            # Handle text output
            for block in response.content:
                if hasattr(block, "text"):
                    self.console.print()
                    self.console.print(Markdown(block.text))

            # If no tool use, we're done
            if response.stop_reason == "end_turn":
                self.session.add_assistant_message(response.content)
                break

            # Handle tool use
            if response.stop_reason == "tool_use":
                self.session.add_assistant_message(response.content)
                tool_results = []

                for block in response.content:
                    if block.type == "tool_use":
                        result = await self._execute_tool(block.name, block.input)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result,
                        })

                self.session.add_tool_results(tool_results)

    async def run(self) -> None:
        """Run the REPL loop."""
        # Display welcome banner
        self.console.print(
            Panel(
                f"[bold cyan]jirade agent[/bold cyan] - {self.repo_config.full_repo_name}\n"
                f"[dim]Type your request, or 'exit' to quit[/dim]",
                border_style="cyan",
            )
        )
        self.console.print()

        while True:
            try:
                user_input = self.console.input("[bold green]you>[/bold green] ")

                if user_input.lower() in ("exit", "quit", "q"):
                    self.console.print("[dim]Goodbye![/dim]")
                    break

                if not user_input.strip():
                    continue

                self.console.print()
                await self.chat(user_input)
                self.console.print()

            except KeyboardInterrupt:
                self.console.print("\n[dim]Use 'exit' to quit[/dim]")
            except EOFError:
                break

        # Cleanup
        await self._cleanup()

    async def _cleanup(self) -> None:
        """Clean up resources."""
        if self._jira_client:
            await self._jira_client.close()
        if self._github_client:
            await self._github_client.close()
