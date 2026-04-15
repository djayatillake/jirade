"""Jira REST API client."""

import logging
import re
from typing import Any

import httpx

logger = logging.getLogger(__name__)


def _parse_inline(text: str) -> list[dict]:
    """Parse inline formatting: *bold* and `code`."""
    nodes: list[dict] = []
    # Split on *bold* and `code` markers
    parts = re.split(r"(\*[^*]+\*|`[^`]+`)", text)
    for part in parts:
        if not part:
            continue
        if part.startswith("*") and part.endswith("*") and len(part) > 2:
            nodes.append(
                {
                    "type": "text",
                    "text": part[1:-1],
                    "marks": [{"type": "strong"}],
                }
            )
        elif part.startswith("`") and part.endswith("`") and len(part) > 2:
            nodes.append(
                {
                    "type": "text",
                    "text": part[1:-1],
                    "marks": [{"type": "code"}],
                }
            )
        else:
            nodes.append({"type": "text", "text": part})
    return nodes or [{"type": "text", "text": text}]


def _plain_text_to_adf(body: str) -> dict:
    """Convert plain text with lightweight formatting to ADF.

    Supports:
    - Lines starting with # ## ### for headings (h1-h3)
    - Lines starting with - for bullet lists (consecutive lines grouped)
    - *bold* and `code` inline formatting
    - Blank lines as paragraph separators
    - Everything else as paragraphs
    """
    lines = body.split("\n")
    content: list[dict] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Blank line — skip
        if not stripped:
            i += 1
            continue

        # Headings: # text, ## text, ### text
        heading_match = re.match(r"^(#{1,3})\s+(.+)$", stripped)
        if heading_match:
            level = len(heading_match.group(1))
            content.append(
                {
                    "type": "heading",
                    "attrs": {"level": level},
                    "content": _parse_inline(heading_match.group(2)),
                }
            )
            i += 1
            continue

        # Bullet list: consecutive lines starting with -
        if stripped.startswith("- "):
            items: list[dict] = []
            while i < len(lines) and lines[i].strip().startswith("- "):
                item_text = lines[i].strip()[2:]
                items.append(
                    {
                        "type": "listItem",
                        "content": [
                            {
                                "type": "paragraph",
                                "content": _parse_inline(item_text),
                            }
                        ],
                    }
                )
                i += 1
            content.append({"type": "bulletList", "content": items})
            continue

        # Jira wiki table: ||header||header|| and |cell|cell| lines
        if stripped.startswith("||") or (stripped.startswith("|") and not stripped.startswith("|-")):
            headers: list[str] = []
            table_rows: list[list[str]] = []
            while i < len(lines):
                tline = lines[i].strip()
                if tline.startswith("||"):
                    # Header row: ||h1||h2||h3||
                    headers = [c.strip() for c in tline.strip("|").split("||") if c.strip()]
                elif tline.startswith("|") and not tline.startswith("|-"):
                    # Data row: |c1|c2|c3|
                    cells = [c.strip() for c in tline.strip("|").split("|") if c.strip()]
                    table_rows.append(cells)
                else:
                    break
                i += 1
            if headers or table_rows:
                if not headers and table_rows:
                    headers = table_rows.pop(0)
                content.append(build_adf_table(headers, table_rows))
            continue

        # Horizontal rule: ---
        if stripped == "---":
            content.append({"type": "rule"})
            i += 1
            continue

        # Regular paragraph
        content.append({"type": "paragraph", "content": _parse_inline(stripped)})
        i += 1

    return {"version": 1, "type": "doc", "content": content}


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
            issue_key: Issue key (e.g., PROJ-123).

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

    async def delete_comment(self, issue_key: str, comment_id: str) -> None:
        """Delete a comment from an issue.

        Args:
            issue_key: Issue key.
            comment_id: Comment ID to delete.
        """
        url = f"{self.base_url}/issue/{issue_key}/comment/{comment_id}"
        await self._request("DELETE", url)

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
            body_adf = _plain_text_to_adf(body)
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

    async def create_issue(
        self,
        project_key: str,
        summary: str,
        description_adf: dict[str, Any],
        issue_type: str = "Task",
        sprint_id: int | None = None,
        labels: list[str] | None = None,
    ) -> dict[str, Any]:
        """Create a new Jira issue.

        Args:
            project_key: Project key (e.g., 'AENG').
            summary: Issue summary/title.
            description_adf: Description in Atlassian Document Format.
            issue_type: Issue type name (default: 'Task').
            sprint_id: Sprint ID to assign the issue to (customfield_10010).
            labels: Labels to apply.

        Returns:
            Created issue data with 'key' and 'id'.
        """
        url = f"{self.base_url}/issue"
        fields: dict[str, Any] = {
            "project": {"key": project_key},
            "summary": summary,
            "description": description_adf,
            "issuetype": {"name": issue_type},
        }
        if sprint_id is not None:
            fields["customfield_10010"] = sprint_id
        if labels:
            fields["labels"] = labels
        return await self._request("POST", url, json={"fields": fields})

    async def get_active_sprint_id(self, project_key: str) -> int | None:
        """Find the active sprint ID for a project by inspecting open-sprint issues.

        Args:
            project_key: Project key (e.g., 'AENG').

        Returns:
            Sprint ID, or None if no active sprint found.
        """
        issues = await self.search_issues(
            jql=f"project = {project_key} AND sprint in openSprints() ORDER BY created DESC",
            max_results=1,
            fields=["customfield_10010"],
        )
        if not issues:
            return None
        sprints = issues[0].get("fields", {}).get("customfield_10010") or []
        for sprint in sprints:
            if sprint.get("state") == "active":
                return sprint["id"]
        return None

    async def add_label(self, issue_key: str, label: str) -> None:
        """Add a label to an issue (idempotent - won't duplicate if already present).

        Args:
            issue_key: Issue key.
            label: Label to add.
        """
        url = f"{self.base_url}/issue/{issue_key}"
        await self._request("PUT", url, json={
            "update": {"labels": [{"add": label}]}
        })


def build_adf_table(headers: list[str], rows: list[list[str]]) -> dict:
    """Build an ADF table node from headers and row data.

    Args:
        headers: Column header strings.
        rows: List of rows, each a list of cell strings.

    Returns:
        ADF table node dict.
    """
    def _cell(text: str, is_header: bool = False) -> dict:
        cell_type = "tableHeader" if is_header else "tableCell"
        return {
            "type": cell_type,
            "content": [{"type": "paragraph", "content": _parse_inline(str(text))}],
        }

    header_row = {
        "type": "tableRow",
        "content": [_cell(h, is_header=True) for h in headers],
    }
    data_rows = [
        {"type": "tableRow", "content": [_cell(c) for c in row]}
        for row in rows
    ]

    return {
        "type": "table",
        "attrs": {"isNumberColumnEnabled": False, "layout": "default"},
        "content": [header_row, *data_rows],
    }


def build_adf_document(sections: list[dict]) -> dict:
    """Build a complete ADF document from a list of section dicts.

    Each section dict can have:
        - type: "heading" with "level" and "text"
        - type: "paragraph" with "text"
        - type: "table" with "headers" and "rows"
        - type: "rule" (horizontal rule)

    Args:
        sections: List of section definitions.

    Returns:
        Complete ADF document dict.
    """
    content = []
    for section in sections:
        if section["type"] == "heading":
            content.append({
                "type": "heading",
                "attrs": {"level": section.get("level", 2)},
                "content": _parse_inline(section["text"]),
            })
        elif section["type"] == "paragraph":
            content.append({
                "type": "paragraph",
                "content": _parse_inline(section["text"]),
            })
        elif section["type"] == "table":
            content.append(build_adf_table(section["headers"], section["rows"]))
        elif section["type"] == "rule":
            content.append({"type": "rule"})
    return {"version": 1, "type": "doc", "content": content}


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
