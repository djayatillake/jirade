"""Jira tool handlers for MCP server."""

import logging
from typing import Any

from ...auth.manager import AuthManager
from ...clients.jira_client import JiraClient, extract_text_from_adf, format_issue_summary
from ...config import get_settings

logger = logging.getLogger(__name__)


async def get_jira_client() -> tuple[JiraClient, AuthManager]:
    """Get an authenticated Jira client.

    Returns:
        Tuple of (JiraClient, AuthManager).

    Raises:
        RuntimeError: If not authenticated.
    """
    settings = get_settings()
    auth = AuthManager(settings)

    if not auth.jira.is_authenticated():
        raise RuntimeError("Not authenticated with Jira. Run 'jirade auth login jira' first.")

    access_token = auth.jira.get_access_token()
    cloud_id = auth.jira.get_cloud_id()

    if not access_token or not cloud_id:
        raise RuntimeError("Jira authentication incomplete. Run 'jirade auth login jira' to re-authenticate.")

    client = JiraClient(cloud_id=cloud_id, access_token=access_token)
    return client, auth


async def handle_jira_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Handle a Jira tool call.

    Args:
        name: Tool name.
        arguments: Tool arguments.

    Returns:
        Tool result.
    """
    client, auth = await get_jira_client()

    try:
        if name == "jirade_search_jira":
            return await search_jira(client, arguments)
        elif name == "jirade_get_issue":
            return await get_issue(client, arguments)
        elif name == "jirade_add_comment":
            return await add_comment(client, arguments)
        elif name == "jirade_transition_issue":
            return await transition_issue(client, arguments)
        else:
            raise ValueError(f"Unknown Jira tool: {name}")
    finally:
        await client.close()


async def search_jira(client: JiraClient, arguments: dict[str, Any]) -> dict[str, Any]:
    """Search Jira issues using JQL.

    Args:
        client: Jira client.
        arguments: Tool arguments with 'jql' and optional 'limit'.

    Returns:
        Search results.
    """
    jql = arguments["jql"]
    limit = arguments.get("limit", 20)

    issues = await client.search_issues(jql, max_results=limit)

    results = []
    for issue in issues:
        results.append(format_issue_summary(issue))

    return {
        "total": len(results),
        "issues": results,
    }


async def get_issue(client: JiraClient, arguments: dict[str, Any]) -> dict[str, Any]:
    """Get full details for a Jira issue.

    Args:
        client: Jira client.
        arguments: Tool arguments with 'key'.

    Returns:
        Issue details.
    """
    key = arguments["key"]

    issue = await client.get_issue(key)
    fields = issue.get("fields", {})

    # Get comments
    comments = await client.get_issue_comments(key)
    formatted_comments = []
    for comment in comments:
        formatted_comments.append(
            {
                "author": comment.get("author", {}).get("displayName", "Unknown"),
                "created": comment.get("created"),
                "body": extract_text_from_adf(comment.get("body")),
            }
        )

    # Get available transitions
    transitions = await client.get_issue_transitions(key)
    available_transitions = [t["name"] for t in transitions]

    return {
        "key": issue.get("key"),
        "summary": fields.get("summary"),
        "description": extract_text_from_adf(fields.get("description")),
        "status": fields.get("status", {}).get("name"),
        "type": fields.get("issuetype", {}).get("name"),
        "priority": fields.get("priority", {}).get("name"),
        "labels": fields.get("labels", []),
        "assignee": fields.get("assignee", {}).get("displayName") if fields.get("assignee") else None,
        "reporter": fields.get("reporter", {}).get("displayName") if fields.get("reporter") else None,
        "created": fields.get("created"),
        "updated": fields.get("updated"),
        "comments": formatted_comments,
        "available_transitions": available_transitions,
    }


async def add_comment(client: JiraClient, arguments: dict[str, Any]) -> dict[str, Any]:
    """Add a comment to a Jira issue.

    Args:
        client: Jira client.
        arguments: Tool arguments with 'key' and 'comment'.

    Returns:
        Created comment info.
    """
    key = arguments["key"]
    comment_text = arguments["comment"]

    result = await client.add_comment(key, comment_text)

    return {
        "success": True,
        "issue_key": key,
        "comment_id": result.get("id"),
        "created": result.get("created"),
    }


async def transition_issue(client: JiraClient, arguments: dict[str, Any]) -> dict[str, Any]:
    """Transition a Jira issue to a new status.

    Args:
        client: Jira client.
        arguments: Tool arguments with 'key' and 'status'.

    Returns:
        Transition result.
    """
    key = arguments["key"]
    target_status = arguments["status"]

    # Get available transitions
    transitions = await client.get_issue_transitions(key)

    # Find matching transition (case-insensitive)
    transition_id = None
    matched_name = None
    for t in transitions:
        if t["name"].lower() == target_status.lower():
            transition_id = t["id"]
            matched_name = t["name"]
            break

    if not transition_id:
        available = [t["name"] for t in transitions]
        return {
            "success": False,
            "error": f"Transition to '{target_status}' not available",
            "available_transitions": available,
        }

    await client.transition_issue(key, transition_id)

    # Tag with "jirade" label when closing a ticket
    if matched_name and matched_name.lower() == "done":
        try:
            await client.add_label(key, "jirade")
        except Exception:
            logger.warning(f"Failed to add 'jirade' label to {key}")

    return {
        "success": True,
        "issue_key": key,
        "new_status": matched_name,
    }
