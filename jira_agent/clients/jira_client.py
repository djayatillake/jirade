"""Jira REST API client."""

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class JiraClient:
    """Client for Jira REST API."""

    def __init__(self, cloud_id: str, access_token: str):
        """Initialize Jira client.

        Args:
            cloud_id: Atlassian cloud ID.
            access_token: OAuth access token.
        """
        self.cloud_id = cloud_id
        self.base_url = f"https://api.atlassian.com/ex/jira/{cloud_id}/rest/api/3"
        self.agile_url = f"https://api.atlassian.com/ex/jira/{cloud_id}/rest/agile/1.0"
        self.headers = {
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        self._client = httpx.AsyncClient(headers=self.headers, timeout=30.0)

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()

    async def __aenter__(self) -> "JiraClient":
        return self

    async def __aexit__(self, *args) -> None:
        await self.close()

    async def _request(
        self,
        method: str,
        url: str,
        **kwargs,
    ) -> dict[str, Any]:
        """Make an HTTP request.

        Args:
            method: HTTP method.
            url: Full URL.
            **kwargs: Additional request arguments.

        Returns:
            JSON response.

        Raises:
            httpx.HTTPStatusError: If request fails.
        """
        response = await self._client.request(method, url, **kwargs)
        response.raise_for_status()

        if response.status_code == 204:
            return {}

        return response.json()

    async def get_issue(self, issue_key: str) -> dict[str, Any]:
        """Get issue details.

        Args:
            issue_key: Issue key (e.g., AENG-1234).

        Returns:
            Issue data.
        """
        url = f"{self.base_url}/issue/{issue_key}"
        params = {
            "expand": "renderedFields,names",
            "fields": "summary,description,status,assignee,labels,comment,issuetype,priority,customfield_*",
        }
        return await self._request("GET", url, params=params)

    async def get_issue_comments(self, issue_key: str) -> list[dict[str, Any]]:
        """Get comments on an issue.

        Args:
            issue_key: Issue key.

        Returns:
            List of comments.
        """
        url = f"{self.base_url}/issue/{issue_key}/comment"
        data = await self._request("GET", url)
        return data.get("comments", [])

    async def add_comment(self, issue_key: str, body: str) -> dict[str, Any]:
        """Add a comment to an issue.

        Args:
            issue_key: Issue key.
            body: Comment body (plain text or Atlassian Document Format).

        Returns:
            Created comment data.
        """
        url = f"{self.base_url}/issue/{issue_key}/comment"

        # Convert plain text to ADF if needed
        if not body.startswith("{"):
            body_adf = {
                "version": 1,
                "type": "doc",
                "content": [
                    {
                        "type": "paragraph",
                        "content": [{"type": "text", "text": body}],
                    }
                ],
            }
        else:
            import json

            body_adf = json.loads(body)

        return await self._request("POST", url, json={"body": body_adf})

    async def get_board_issues(
        self,
        board_id: int,
        status: str | None = None,
        max_results: int = 50,
        start_at: int = 0,
    ) -> list[dict[str, Any]]:
        """Get issues from a board.

        Args:
            board_id: Jira board ID.
            status: Filter by status name.
            max_results: Maximum results to return.
            start_at: Offset for pagination.

        Returns:
            List of issues.
        """
        url = f"{self.agile_url}/board/{board_id}/issue"
        params = {
            "maxResults": max_results,
            "startAt": start_at,
            "fields": "summary,status,assignee,labels,issuetype,priority",
        }

        if status:
            # Use JQL for status filter
            params["jql"] = f'status = "{status}"'

        data = await self._request("GET", url, params=params)
        return data.get("issues", [])

    async def search_issues(
        self,
        jql: str,
        max_results: int = 50,
        start_at: int = 0,
        fields: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Search issues using JQL.

        Args:
            jql: JQL query string.
            max_results: Maximum results.
            start_at: Offset for pagination.
            fields: Fields to return.

        Returns:
            List of matching issues.
        """
        # Use the new /jql endpoint (API v3)
        url = f"{self.base_url}/search/jql"
        params = {
            "jql": jql,
            "maxResults": max_results,
            "startAt": start_at,
        }
        if fields:
            params["fields"] = ",".join(fields)

        data = await self._request("GET", url, params=params)
        return data.get("issues", [])

    async def get_issue_transitions(self, issue_key: str) -> list[dict[str, Any]]:
        """Get available transitions for an issue.

        Args:
            issue_key: Issue key.

        Returns:
            List of available transitions.
        """
        url = f"{self.base_url}/issue/{issue_key}/transitions"
        data = await self._request("GET", url)
        return data.get("transitions", [])

    async def transition_issue(self, issue_key: str, transition_id: str) -> None:
        """Transition an issue to a new status.

        Args:
            issue_key: Issue key.
            transition_id: ID of the transition to perform.
        """
        url = f"{self.base_url}/issue/{issue_key}/transitions"
        await self._request("POST", url, json={"transition": {"id": transition_id}})

    async def assign_issue(self, issue_key: str, account_id: str | None) -> None:
        """Assign an issue to a user.

        Args:
            issue_key: Issue key.
            account_id: User's account ID, or None to unassign.
        """
        url = f"{self.base_url}/issue/{issue_key}/assignee"
        await self._request("PUT", url, json={"accountId": account_id})


def extract_text_from_adf(adf: dict[str, Any] | None) -> str:
    """Extract plain text from Atlassian Document Format.

    Args:
        adf: ADF document structure.

    Returns:
        Plain text content.
    """
    if not adf:
        return ""

    def extract_content(node: dict[str, Any]) -> str:
        if node.get("type") == "text":
            return node.get("text", "")

        content = node.get("content", [])
        texts = [extract_content(child) for child in content]
        return " ".join(texts)

    return extract_content(adf).strip()


def format_issue_summary(issue: dict[str, Any]) -> dict[str, Any]:
    """Format issue data for agent consumption.

    Args:
        issue: Raw issue data from API.

    Returns:
        Formatted issue summary.
    """
    fields = issue.get("fields", {})

    return {
        "key": issue.get("key"),
        "summary": fields.get("summary"),
        "description": extract_text_from_adf(fields.get("description")),
        "status": fields.get("status", {}).get("name"),
        "type": fields.get("issuetype", {}).get("name"),
        "priority": fields.get("priority", {}).get("name"),
        "labels": fields.get("labels", []),
        "assignee": fields.get("assignee", {}).get("displayName") if fields.get("assignee") else None,
    }
