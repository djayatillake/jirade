"""Confluence tool handlers for MCP server."""

import logging
from typing import Any

from ...auth.manager import AuthManager
from ...clients.confluence_client import ConfluenceClient, markdown_to_storage
from ...config import get_settings

logger = logging.getLogger(__name__)


async def get_confluence_client() -> ConfluenceClient:
    """Get an authenticated Confluence client.

    Reuses the Atlassian OAuth tokens minted for Jira — same access token,
    same cloud_id, but requires the Confluence scopes to have been granted
    at authorization time (introduced in jirade v0.6.0).

    Raises:
        RuntimeError: If not authenticated or missing Confluence scopes.
    """
    settings = get_settings()
    auth = AuthManager(settings)

    if not auth.jira.is_authenticated():
        raise RuntimeError("Not authenticated with Atlassian. Run 'jirade auth login --service=jira' first.")

    if not auth.jira.has_confluence_scopes():
        raise RuntimeError(
            "Atlassian token is missing Confluence scopes. "
            "Re-run 'jirade auth login --service=jira' to re-authorize with Confluence access. "
            "Make sure your OAuth app in https://developer.atlassian.com/console/myapps "
            "has these scopes added: read:confluence-content.all, read:confluence-space.summary, "
            "write:confluence-content."
        )

    access_token = auth.jira.get_access_token()
    cloud_id = auth.jira.get_cloud_id()
    return ConfluenceClient(cloud_id=cloud_id, access_token=access_token)


def _page_url(page: dict[str, Any]) -> str:
    """Build a public web URL for a Confluence v2 page response.

    v2 returns `_links.webui` as a relative path under the wiki base, e.g.
    `/spaces/AENG/pages/12345/Jirade`. The links section may also include
    a `base` we can prepend; if missing, the URL is left relative.
    """
    links = page.get("_links", {}) or {}
    base = links.get("base") or ""
    webui = links.get("webui") or ""
    if not webui:
        return ""
    return f"{base}{webui}" if base else webui


async def handle_confluence_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Dispatch Confluence tool calls."""
    client = await get_confluence_client()
    try:
        if name == "jirade_publish_confluence_page":
            return await publish_page(client, arguments)
        if name == "jirade_get_confluence_page":
            return await get_page(client, arguments)
        if name == "jirade_search_confluence":
            return await search_confluence(client, arguments)
        raise ValueError(f"Unknown Confluence tool: {name}")
    finally:
        await client.close()


async def publish_page(client: ConfluenceClient, arguments: dict[str, Any]) -> dict[str, Any]:
    """Create or update a Confluence page from markdown.

    If a page with the same title exists in the space (and same parent if
    given), it's updated; otherwise a new page is created. Markdown is
    converted to Confluence storage format inline.
    """
    space_key = arguments["space_key"]
    title = arguments["title"]
    body_markdown = arguments["body_markdown"]
    parent_title = arguments.get("parent_title")
    parent_id_arg = arguments.get("parent_id")

    parent_id: str | None = parent_id_arg
    if parent_title and not parent_id:
        parent = await client.find_page_by_title(space_key, parent_title)
        if not parent:
            raise RuntimeError(
                f"Parent page '{parent_title}' not found in space '{space_key}'. "
                "Create it first or pass parent_id directly."
            )
        parent_id = parent["id"]

    body_storage = markdown_to_storage(body_markdown)
    existing = await client.find_page_by_title(space_key, title)
    if existing:
        version = existing.get("version", {}).get("number", 1)
        page = await client.update_page(existing["id"], title, body_storage, version)
        action = "updated"
    else:
        page = await client.create_page(space_key, title, body_storage, parent_id)
        action = "created"

    return {
        "action": action,
        "page_id": page["id"],
        "title": page.get("title"),
        "version": page.get("version", {}).get("number"),
        "url": _page_url(page),
    }


async def get_page(client: ConfluenceClient, arguments: dict[str, Any]) -> dict[str, Any]:
    """Fetch a Confluence page by ID or by space+title."""
    page_id = arguments.get("page_id")
    if not page_id:
        space_key = arguments.get("space_key")
        title = arguments.get("title")
        if not (space_key and title):
            raise ValueError("Either page_id or (space_key, title) must be provided.")
        match = await client.find_page_by_title(space_key, title)
        if not match:
            return {"found": False}
        page_id = match["id"]

    page = await client.get_page(page_id)
    return {
        "found": True,
        "page_id": page["id"],
        "title": page.get("title"),
        "version": page.get("version", {}).get("number"),
        "space_key": page.get("space", {}).get("key"),
        "body_storage": page.get("body", {}).get("storage", {}).get("value"),
        "url": _page_url(page),
    }


async def search_confluence(client: ConfluenceClient, arguments: dict[str, Any]) -> dict[str, Any]:
    """Search Confluence content using CQL."""
    cql = arguments["cql"]
    limit = arguments.get("limit", 25)
    results = await client.search_cql(cql, limit=limit)
    return {
        "total": len(results),
        "results": [
            {
                "id": r.get("id"),
                "title": r.get("title"),
                "type": r.get("type"),
                "space_key": r.get("space", {}).get("key") if r.get("space") else None,
                "url": _page_url(r),
            }
            for r in results
        ],
    }
