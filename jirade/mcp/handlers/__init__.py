"""Tool handlers for MCP server."""

from typing import Any

from .dbt_cloud import handle_dbt_tool
from .github import handle_github_tool
from .jira import handle_jira_tool

__all__ = ["dispatch_tool"]


async def dispatch_tool(name: str, arguments: dict[str, Any]) -> Any:
    """Dispatch a tool call to the appropriate handler.

    Args:
        name: Tool name.
        arguments: Tool arguments.

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
    elif name.startswith("jirade_dbt"):
        return await handle_dbt_tool(name, arguments)
    else:
        raise ValueError(f"Unknown tool: {name}")
