"""Smoke tests for the tag_advisor MCP handler — mocks GitHub + Claude."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from jirade.mcp.handlers.tag_advisor import handle_tag_advisor_tool

FIXTURES = Path(__file__).parent / "fixtures"
GOVERNED_TAGS_TEXT = (FIXTURES / "governed_tags.yaml").read_text()

ADDED_MODEL = "dbt-databricks/models/mart/growth/mart__growth__rpt_new_signal.sql"
MODIFIED_MODEL = "dbt-databricks/models/mart/growth/mart__growth__rpt_existing.sql"

# Added model carries a placeholder tag inline → needs a suggestion.
MOCK_ADDED_SQL = """{{
  auto_config(materialized='table', databricks_tags={'domain': 'tbd'})
}}
SELECT * FROM {{ ref('mart__growth__rpt_dashboard_organization') }}
"""
# Modified model already has a real governed domain → 'ok', no Claude call.
MOCK_MODIFIED_SQL = """{{
  auto_config(materialized='table', databricks_tags={'domain': 'growth', 'sub_domain': 'usage'})
}}
SELECT * FROM {{ ref('mart__growth__rpt_dashboard_application_user') }}
"""

MOCK_PR_FILES = [
    {"status": "added", "filename": ADDED_MODEL},
    {"status": "modified", "filename": MODIFIED_MODEL},
    {"status": "modified", "filename": "dbt-databricks/models/staging/foo.sql"},  # wrong catalog
    {"status": "added", "filename": "README.md"},  # out of scope
]


@pytest.fixture
def mock_github_client():
    client = MagicMock()
    client.get_pull_request = AsyncMock(return_value={"head": {"sha": "cafef00d"}})
    client.get_pr_files = AsyncMock(return_value=MOCK_PR_FILES)
    client.close = AsyncMock()
    client.upsert_pr_comment = AsyncMock(return_value={"id": 42})
    client.get_pr_comments = AsyncMock(return_value=[])
    client.repo_url = "https://api.github.com/repos/algolia/data"

    async def mock_get_file_content(path: str, ref: str | None = None) -> str | None:
        if path.endswith("governed_tags.yaml"):
            return GOVERNED_TAGS_TEXT
        if path.endswith("rpt_new_signal.sql"):
            return MOCK_ADDED_SQL
        if path.endswith("rpt_existing.sql"):
            return MOCK_MODIFIED_SQL
        return None

    client.get_file_content = AsyncMock(side_effect=mock_get_file_content)
    client._request = AsyncMock(return_value=[])  # sibling-YML listing
    return client


@pytest.fixture
def mock_claude_response():
    import json as _json

    text_block = SimpleNamespace(
        type="text",
        text=_json.dumps(
            {
                "domain": "growth",
                "sub_domain": "usage",
                "confidence": "high",
                "rationale": "Activation signal on growth accounts.",
            }
        ),
    )
    resp = SimpleNamespace(content=[text_block])
    anthropic_client = MagicMock()
    anthropic_client.messages.create.return_value = resp
    return anthropic_client


def _settings():
    return SimpleNamespace(anthropic_api_key="test-key", claude_model="claude-opus-4-5-20251101")


@pytest.mark.asyncio
async def test_handler_end_to_end_dry_run(mock_github_client, mock_claude_response):
    with patch(
        "jirade.mcp.handlers.tag_advisor.get_github_client",
        new=AsyncMock(return_value=(mock_github_client, MagicMock())),
    ), patch(
        "jirade.mcp.handlers.tag_advisor.Anthropic", return_value=mock_claude_response
    ), patch(
        "jirade.mcp.handlers.tag_advisor.get_settings", return_value=_settings()
    ):
        result = await handle_tag_advisor_tool(
            "jirade_advise_tags_for_pr", {"pr_number": 77, "post_comment": False}
        )

    assert result["head_sha"] == "cafef00d"
    assert result["in_scope_count"] == 2  # added + modified growth models (A + M)
    assert result["comment_posted"] is False

    by_name = {d["table_name"]: d for d in result["decisions"]}
    assert by_name["rpt_new_signal"]["status"] == "suggested"
    assert by_name["rpt_new_signal"]["suggested_domain"] == "growth"
    assert by_name["rpt_existing"]["status"] == "ok"  # already tagged → no Claude

    assert "rpt_new_signal" in result["comment_body"]
    assert "domain: growth" in result["comment_body"]
    mock_github_client.upsert_pr_comment.assert_not_called()


@pytest.mark.asyncio
async def test_handler_posts_when_requested(mock_github_client, mock_claude_response):
    with patch(
        "jirade.mcp.handlers.tag_advisor.get_github_client",
        new=AsyncMock(return_value=(mock_github_client, MagicMock())),
    ), patch(
        "jirade.mcp.handlers.tag_advisor.Anthropic", return_value=mock_claude_response
    ), patch(
        "jirade.mcp.handlers.tag_advisor.get_settings", return_value=_settings()
    ):
        result = await handle_tag_advisor_tool(
            "jirade_advise_tags_for_pr", {"pr_number": 77, "post_comment": True}
        )

    assert result["comment_posted"] is True
    mock_github_client.upsert_pr_comment.assert_awaited_once()
    _args, kwargs = mock_github_client.upsert_pr_comment.call_args
    assert kwargs.get("marker") == "<!-- jirade:tag-advisor:v1 -->"


@pytest.mark.asyncio
async def test_handler_skips_write_when_unchanged(mock_github_client, mock_claude_response):
    with patch(
        "jirade.mcp.handlers.tag_advisor.get_github_client",
        new=AsyncMock(return_value=(mock_github_client, MagicMock())),
    ), patch(
        "jirade.mcp.handlers.tag_advisor.Anthropic", return_value=mock_claude_response
    ), patch(
        "jirade.mcp.handlers.tag_advisor.get_settings", return_value=_settings()
    ):
        first = await handle_tag_advisor_tool(
            "jirade_advise_tags_for_pr", {"pr_number": 77, "post_comment": False}
        )
        mock_github_client.get_pr_comments = AsyncMock(
            return_value=[{"id": 1, "body": first["comment_body"]}]
        )
        second = await handle_tag_advisor_tool(
            "jirade_advise_tags_for_pr", {"pr_number": 77, "post_comment": True}
        )

    assert second["comment_skipped_no_change"] is True
    assert second["comment_posted"] is False
    mock_github_client.upsert_pr_comment.assert_not_called()


@pytest.mark.asyncio
async def test_handler_raises_on_missing_governed_tags():
    client = MagicMock()
    client.get_pull_request = AsyncMock(return_value={"head": {"sha": "abc"}})
    client.get_pr_files = AsyncMock(return_value=[{"status": "added", "filename": ADDED_MODEL}])
    client.get_file_content = AsyncMock(return_value=None)  # governed_tags missing
    client.close = AsyncMock()
    client.repo_url = "https://api.github.com/repos/algolia/data"

    with patch(
        "jirade.mcp.handlers.tag_advisor.get_github_client",
        new=AsyncMock(return_value=(client, MagicMock())),
    ):
        with pytest.raises(RuntimeError, match="governed_tags.yaml not found"):
            await handle_tag_advisor_tool("jirade_advise_tags_for_pr", {"pr_number": 77})


@pytest.mark.asyncio
async def test_handler_rejects_unknown_tool_name():
    with pytest.raises(ValueError, match="Unknown tag-advisor tool"):
        await handle_tag_advisor_tool("jirade_unknown", {"pr_number": 1})


@pytest.mark.asyncio
async def test_handler_requires_pr_number():
    with patch("jirade.mcp.handlers.tag_advisor.get_github_client", new=AsyncMock()):
        with pytest.raises(ValueError, match="pr_number is required"):
            await handle_tag_advisor_tool("jirade_advise_tags_for_pr", {})
