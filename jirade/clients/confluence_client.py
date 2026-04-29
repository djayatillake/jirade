"""Confluence Cloud REST API v2 client.

Uses the same OAuth access token as JiraClient — Atlassian Cloud OAuth
issues a single token that works for both Jira and Confluence APIs as
long as the corresponding scopes were granted at authorization time.

API base for v2: https://api.atlassian.com/ex/confluence/{cloud_id}/wiki/api/v2
The legacy v1 /wiki/rest/api/content endpoints were retired and now return
410 Gone, so v2 is the only path forward.

CQL search is the one exception — it still lives at v1
/wiki/rest/api/search and works via OAuth as long as `search:confluence`
scope was granted.
"""

import logging
import re
from html import escape
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class ConfluenceClient:
    """Client for Confluence Cloud REST API v2 (with v1 fallback for CQL search)."""

    def __init__(self, cloud_id: str, access_token: str):
        """Initialize Confluence client.

        Args:
            cloud_id: Atlassian cloud ID (same as for JiraClient).
            access_token: OAuth access token with Confluence scopes.
        """
        self.cloud_id = cloud_id
        self.access_token = access_token
        self.base_url = f"https://api.atlassian.com/ex/confluence/{cloud_id}/wiki/api/v2"
        self.legacy_url = f"https://api.atlassian.com/ex/confluence/{cloud_id}/wiki/rest/api"
        self.headers = {
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        self._client = httpx.AsyncClient(headers=self.headers, timeout=30.0)
        # Cache of space-key → space-id resolutions
        self._space_id_cache: dict[str, str] = {}

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()

    async def __aenter__(self) -> "ConfluenceClient":
        return self

    async def __aexit__(self, *args) -> None:
        await self.close()

    async def _request(self, method: str, url: str, **kwargs) -> dict[str, Any]:
        response = await self._client.request(method, url, **kwargs)
        response.raise_for_status()
        if response.status_code == 204:
            return {}
        return response.json()

    async def get_space_id(self, space_key: str) -> str:
        """Resolve a space key (e.g. 'AENG') to a numeric space ID."""
        if space_key in self._space_id_cache:
            return self._space_id_cache[space_key]
        params = {"keys": space_key}
        data = await self._request("GET", f"{self.base_url}/spaces", params=params)
        results = data.get("results", [])
        if not results:
            raise RuntimeError(f"Confluence space '{space_key}' not found")
        space_id = results[0]["id"]
        self._space_id_cache[space_key] = space_id
        return space_id

    async def find_page_by_title(self, space_key: str, title: str) -> dict[str, Any] | None:
        """Find a page in a space by its exact title.

        Returns:
            Page data with id/title/version, or None if not found.
        """
        space_id = await self.get_space_id(space_key)
        params = {
            "space-id": space_id,
            "title": title,
            "limit": 25,
        }
        data = await self._request("GET", f"{self.base_url}/pages", params=params)
        results = data.get("results", [])
        # The space-id+title filter on v2 can return loose matches — re-filter exact title
        for page in results:
            if page.get("title") == title:
                return page
        return None

    async def get_page(self, page_id: str) -> dict[str, Any]:
        """Get a page by ID, including body in storage format."""
        params = {"body-format": "storage"}
        return await self._request("GET", f"{self.base_url}/pages/{page_id}", params=params)

    async def create_page(
        self,
        space_key: str,
        title: str,
        body_storage: str,
        parent_id: str | None = None,
    ) -> dict[str, Any]:
        """Create a Confluence page (v2 API)."""
        space_id = await self.get_space_id(space_key)
        payload: dict[str, Any] = {
            "spaceId": space_id,
            "status": "current",
            "title": title,
            "body": {"representation": "storage", "value": body_storage},
        }
        if parent_id:
            payload["parentId"] = parent_id
        return await self._request("POST", f"{self.base_url}/pages", json=payload)

    async def update_page(
        self,
        page_id: str,
        title: str,
        body_storage: str,
        version: int,
    ) -> dict[str, Any]:
        """Update an existing page (v2 API).

        Args:
            page_id: Page to update.
            title: New title (can be unchanged).
            body_storage: New body in storage format.
            version: Current version number — Confluence requires version+1 on update.
        """
        payload = {
            "id": page_id,
            "status": "current",
            "title": title,
            "body": {"representation": "storage", "value": body_storage},
            "version": {"number": version + 1},
        }
        return await self._request("PUT", f"{self.base_url}/pages/{page_id}", json=payload)

    async def search_cql(self, cql: str, limit: int = 25) -> list[dict[str, Any]]:
        """Search content using CQL.

        CQL search has not been migrated to v2 yet, so this uses the v1
        endpoint at /wiki/rest/api/search. Requires the `search:confluence`
        OAuth scope.
        """
        params = {"cql": cql, "limit": limit}
        data = await self._request("GET", f"{self.legacy_url}/search", params=params)
        return data.get("results", [])

    async def upsert_page(
        self,
        space_key: str,
        title: str,
        body_storage: str,
        parent_id: str | None = None,
    ) -> dict[str, Any]:
        """Create the page if it doesn't exist, otherwise update it."""
        existing = await self.find_page_by_title(space_key, title)
        if existing:
            version = existing.get("version", {}).get("number", 1)
            return await self.update_page(existing["id"], title, body_storage, version)
        return await self.create_page(space_key, title, body_storage, parent_id)


def page_url(page: dict[str, Any], cloud_base: str | None = None) -> str:
    """Build a public web URL for a Confluence page from API response.

    v2 returns `_links.webui` as a relative path (e.g. /spaces/AENG/pages/1234/Jirade).
    v2 also returns `_links.base` sometimes; if missing, fall back to a constructed
    base from the cloud_id (caller passes their site URL).
    """
    links = page.get("_links", {}) or {}
    base = links.get("base") or cloud_base or ""
    webui = links.get("webui") or ""
    return f"{base}{webui}" if webui else ""


# ============================================================
# Markdown → Confluence storage format converter
# ============================================================


def _inline_md_to_storage(text: str) -> str:
    """Convert inline markdown (bold, italic, code, links) to storage XHTML."""
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        if text[i] == "`":
            end = text.find("`", i + 1)
            if end != -1:
                out.append(f"<code>{escape(text[i + 1 : end])}</code>")
                i = end + 1
                continue
        if text[i : i + 2] in ("**", "__"):
            marker = text[i : i + 2]
            end = text.find(marker, i + 2)
            if end != -1:
                inner = _inline_md_to_storage(text[i + 2 : end])
                out.append(f"<strong>{inner}</strong>")
                i = end + 2
                continue
        if text[i] in ("*", "_") and (i == 0 or text[i - 1] != text[i]):
            marker = text[i]
            end = text.find(marker, i + 1)
            if end != -1 and (end + 1 >= n or text[end + 1] != marker):
                inner = _inline_md_to_storage(text[i + 1 : end])
                out.append(f"<em>{inner}</em>")
                i = end + 1
                continue
        if text[i] == "[":
            close_bracket = text.find("]", i)
            if close_bracket != -1 and close_bracket + 1 < n and text[close_bracket + 1] == "(":
                close_paren = text.find(")", close_bracket + 2)
                if close_paren != -1:
                    label = text[i + 1 : close_bracket]
                    url = text[close_bracket + 2 : close_paren]
                    out.append(f'<a href="{escape(url, quote=True)}">{escape(label)}</a>')
                    i = close_paren + 1
                    continue
        out.append(escape(text[i]))
        i += 1
    return "".join(out)


def markdown_to_storage(md: str) -> str:
    """Convert markdown to Confluence storage format (XHTML).

    Supports headings (h1-h4), paragraphs, bullet lists, ordered lists,
    fenced code blocks, GFM-style tables, horizontal rules, and inline
    formatting (bold/italic/code/links).
    """
    lines = md.split("\n")
    out: list[str] = []
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        if re.match(r"^(-{3,}|_{3,}|\*{3,})$", stripped):
            out.append("<hr/>")
            i += 1
            continue

        if stripped.startswith("```"):
            lang = stripped[3:].strip()
            code_lines: list[str] = []
            i += 1
            while i < n and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            i += 1
            code_body = "\n".join(code_lines)
            macro = (
                '<ac:structured-macro ac:name="code">'
                + (f'<ac:parameter ac:name="language">{escape(lang)}</ac:parameter>' if lang else "")
                + f"<ac:plain-text-body><![CDATA[{code_body}]]></ac:plain-text-body>"
                + "</ac:structured-macro>"
            )
            out.append(macro)
            continue

        heading_match = re.match(r"^(#{1,4})\s+(.+)$", stripped)
        if heading_match:
            level = len(heading_match.group(1))
            content = _inline_md_to_storage(heading_match.group(2))
            out.append(f"<h{level}>{content}</h{level}>")
            i += 1
            continue

        if "|" in line and i + 1 < n and re.match(r"^\s*\|?[\s:|-]+\|?\s*$", lines[i + 1]):
            header_cells = [c.strip() for c in stripped.strip("|").split("|")]
            i += 2
            rows: list[list[str]] = []
            while i < n and "|" in lines[i] and lines[i].strip():
                row_cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                rows.append(row_cells)
                i += 1
            table_html = ["<table><tbody>"]
            table_html.append("<tr>" + "".join(f"<th>{_inline_md_to_storage(c)}</th>" for c in header_cells) + "</tr>")
            for row in rows:
                table_html.append("<tr>" + "".join(f"<td>{_inline_md_to_storage(c)}</td>" for c in row) + "</tr>")
            table_html.append("</tbody></table>")
            out.append("".join(table_html))
            continue

        if re.match(r"^[-*]\s+", stripped):
            items: list[str] = []
            while i < n and re.match(r"^\s*[-*]\s+", lines[i]):
                item_text = re.sub(r"^\s*[-*]\s+", "", lines[i])
                items.append(f"<li>{_inline_md_to_storage(item_text)}</li>")
                i += 1
            out.append("<ul>" + "".join(items) + "</ul>")
            continue

        if re.match(r"^\d+\.\s+", stripped):
            items = []
            while i < n and re.match(r"^\s*\d+\.\s+", lines[i]):
                item_text = re.sub(r"^\s*\d+\.\s+", "", lines[i])
                items.append(f"<li>{_inline_md_to_storage(item_text)}</li>")
                i += 1
            out.append("<ol>" + "".join(items) + "</ol>")
            continue

        para_lines: list[str] = [stripped]
        i += 1
        while i < n and lines[i].strip() and not lines[i].lstrip().startswith(
            ("#", "-", "*", "```", "|", ">", "<")
        ) and not re.match(r"^\d+\.\s+", lines[i].strip()):
            para_lines.append(lines[i].strip())
            i += 1
        para_text = " ".join(para_lines)
        out.append(f"<p>{_inline_md_to_storage(para_text)}</p>")

    return "".join(out)
