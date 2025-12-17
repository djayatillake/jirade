"""Rich progress display for agent execution."""

import sys
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class StepStatus(Enum):
    """Status of a workflow step."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    RETRY = "retry"
    HEALED = "healed"


@dataclass
class ProgressDisplay:
    """Rich progress display for agent execution.

    Provides user-friendly output showing what the agent is doing,
    including tool calls, errors, and self-healing steps.
    """

    ticket_key: str
    ticket_summary: str = ""
    verbose: bool = True
    show_thinking: bool = True
    _current_step: str = ""
    _iteration: int = 0
    _errors_encountered: list = field(default_factory=list)
    _files_modified: list = field(default_factory=list)
    _start_time: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        self._start_time = datetime.now()

    def start(self) -> None:
        """Display the start of ticket processing."""
        print()
        print(f"┌{'─' * 58}┐")
        print(f"│ {self.ticket_key:<56} │")
        if self.ticket_summary:
            # Truncate summary if too long
            summary = self.ticket_summary[:54] + "..." if len(self.ticket_summary) > 54 else self.ticket_summary
            print(f"│ {summary:<56} │")
        print(f"└{'─' * 58}┘")
        print()

    def step(self, name: str, icon: str = "▸") -> None:
        """Display a workflow step."""
        self._current_step = name
        print(f"{icon} {name}")
        sys.stdout.flush()

    def substep(self, message: str, icon: str = "  ") -> None:
        """Display a substep or detail."""
        print(f"  {icon} {message}")
        sys.stdout.flush()

    def tool_call(self, tool_name: str, args: dict[str, Any]) -> None:
        """Display a tool being called."""
        icon = self._get_tool_icon(tool_name)

        # Format args for display
        if tool_name == "read_file":
            detail = args.get("path", "")
        elif tool_name == "write_file":
            path = args.get("path", "")
            self._files_modified.append(path)
            detail = path
        elif tool_name == "run_command":
            cmd = args.get("command", "")
            # Truncate long commands
            detail = cmd[:50] + "..." if len(cmd) > 50 else cmd
        elif tool_name == "search_files":
            detail = f"pattern: {args.get('pattern', '')}"
        elif tool_name == "create_branch":
            detail = args.get("branch_name", "")
        elif tool_name == "commit_changes":
            msg = args.get("message", "")
            detail = msg[:40] + "..." if len(msg) > 40 else msg
        elif tool_name == "create_pr":
            detail = args.get("title", "")[:40]
        else:
            detail = ""

        if detail:
            print(f"  {icon} {tool_name}: {detail}")
        else:
            print(f"  {icon} {tool_name}")
        sys.stdout.flush()

    def tool_result(self, tool_name: str, success: bool, output: str = "") -> None:
        """Display the result of a tool call."""
        if success:
            if tool_name == "run_command" and output and self.verbose:
                # Show truncated command output
                lines = output.strip().split("\n")
                if len(lines) > 3:
                    for line in lines[:2]:
                        print(f"    │ {line[:70]}")
                    print(f"    │ ... ({len(lines) - 2} more lines)")
                else:
                    for line in lines:
                        print(f"    │ {line[:70]}")
        sys.stdout.flush()

    def thinking(self, thought: str) -> None:
        """Display Claude's thinking/reasoning."""
        if not self.show_thinking:
            return

        print()
        print("  💭 Agent thinking:")
        # Wrap and indent the thought
        lines = thought.split("\n")
        for line in lines[:5]:  # Show first 5 lines
            if line.strip():
                print(f"     {line[:65]}")
        if len(lines) > 5:
            print(f"     ... ({len(lines) - 5} more lines)")
        print()
        sys.stdout.flush()

    def error(self, error_type: str, message: str) -> None:
        """Display an error that occurred."""
        self._errors_encountered.append({"type": error_type, "message": message})
        print()
        print(f"  ❌ Error: {error_type}")
        # Show truncated error message
        lines = message.split("\n")
        for line in lines[:5]:
            if line.strip():
                print(f"     {line[:70]}")
        if len(lines) > 5:
            print(f"     ... ({len(lines) - 5} more lines)")
        sys.stdout.flush()

    def healing_start(self, error_type: str) -> None:
        """Display the start of a self-healing attempt."""
        print()
        print(f"  🔧 Attempting to fix: {error_type}")
        sys.stdout.flush()

    def healing_step(self, action: str) -> None:
        """Display a healing action being taken."""
        print(f"     → {action}")
        sys.stdout.flush()

    def healing_success(self, error_type: str) -> None:
        """Display successful self-healing."""
        print(f"  ✅ Fixed: {error_type}")
        print()
        sys.stdout.flush()

    def healing_failed(self, error_type: str, reason: str) -> None:
        """Display failed self-healing attempt."""
        print(f"  ⚠️  Could not fix: {error_type}")
        print(f"     Reason: {reason[:60]}")
        print()
        sys.stdout.flush()

    def iteration(self, n: int) -> None:
        """Track iteration count (for debugging)."""
        self._iteration = n
        if self.verbose and n > 1 and n % 5 == 0:
            print(f"  ⏳ Agent iteration {n}...")
            sys.stdout.flush()

    def pr_created(self, url: str) -> None:
        """Display PR creation."""
        print()
        print(f"  🚀 Pull Request created!")
        print(f"     {url}")
        sys.stdout.flush()

    def complete(self, success: bool, pr_url: str | None = None, error: str | None = None) -> None:
        """Display completion status."""
        elapsed = datetime.now() - self._start_time
        elapsed_str = f"{elapsed.seconds // 60}m {elapsed.seconds % 60}s"

        print()
        print("─" * 60)

        if success:
            print(f"✅ {self.ticket_key} completed successfully")
            if pr_url:
                print(f"   PR: {pr_url}")
            if self._files_modified:
                unique_files = list(set(self._files_modified))
                print(f"   Files modified: {len(unique_files)}")
            if self._errors_encountered:
                healed = len(self._errors_encountered)
                print(f"   Errors encountered & healed: {healed}")
        else:
            print(f"❌ {self.ticket_key} failed")
            if error:
                print(f"   Error: {error[:60]}")

        print(f"   Duration: {elapsed_str}")
        print()
        sys.stdout.flush()

    def _get_tool_icon(self, tool_name: str) -> str:
        """Get icon for a tool."""
        icons = {
            "read_file": "📖",
            "write_file": "✏️",
            "run_command": "🔧",
            "search_files": "🔍",
            "list_directory": "📁",
            "create_branch": "🌿",
            "commit_changes": "📤",
            "push_branch": "⬆️",
            "create_pr": "🚀",
            "get_pr_status": "📊",
        }
        return icons.get(tool_name, "▸")


def format_command_output(output: str, max_lines: int = 10, max_width: int = 80) -> str:
    """Format command output for display.

    Args:
        output: Raw command output.
        max_lines: Maximum lines to show.
        max_width: Maximum line width.

    Returns:
        Formatted output string.
    """
    lines = output.strip().split("\n")
    formatted = []

    for i, line in enumerate(lines[:max_lines]):
        if len(line) > max_width:
            line = line[:max_width - 3] + "..."
        formatted.append(line)

    if len(lines) > max_lines:
        formatted.append(f"... ({len(lines) - max_lines} more lines)")

    return "\n".join(formatted)
