"""Tool handlers for MCP server."""

from collections.abc import Awaitable, Callable
from typing import Any

from .dbt_diff import handle_dbt_diff_tool
from .github import handle_github_tool
from .jira import handle_jira_tool

__all__ = ["dispatch_tool"]

ProgressCallback = Callable[[float, float | None, str | None], Awaitable[None]]


async def dispatch_tool(
    name: str,
    arguments: dict[str, Any],
    progress_cb: ProgressCallback | None = None,
) -> Any:
    """Dispatch a tool call to the appropriate handler.

    Args:
        name: Tool name.
        arguments: Tool arguments.
        progress_cb: Optional callback for sending progress notifications.

    Returns:
        Tool result.

    Raises:
        ValueError: If tool name is unknown.
    """
    if (
        name.startswith("jirade_search_jira")
        or name.startswith("jirade_get_issue")
        or name.startswith("jirade_add_comment")
        or name.startswith("jirade_transition")
    ):
        return await handle_jira_tool(name, arguments)
    elif name.startswith("jirade_list_prs") or name.startswith("jirade_get_pr") or name.startswith("jirade_get_ci"):
        return await handle_github_tool(name, arguments)
    elif name.startswith("jirade_watch_pr"):
        return await handle_github_tool(name, arguments)
    elif name.startswith("jirade_run_dbt") or name.startswith("jirade_post_diff") or name.startswith("jirade_analyze") or name.startswith("jirade_cleanup_ci") or name.startswith("jirade_generate_schema"):
        return await handle_dbt_diff_tool(name, arguments, progress_cb=progress_cb)
    else:
        raise ValueError(f"Unknown tool: {name}")
