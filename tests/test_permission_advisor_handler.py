"""Smoke tests for the permission_advisor MCP handler — mocks GitHub + Claude."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from jirade.mcp.handlers.permission_advisor import handle_permission_advisor_tool

FIXTURES = Path(__file__).parent / "fixtures"
GOVERNANCE_YAML_TEXT = (FIXTURES / "governance_state.yaml").read_text()
CAP_MATRIX_TEXT = (FIXTURES / "capability_matrix.csv").read_text()
DUM_TEXT = (FIXTURES / "dum.yaml").read_text()

# A PR adding ONE new mart/sales SQL file + ONE modified file (should be dropped)
MOCK_PR_FILES = [
    {
        "status": "added",
        "filename": "dbt-databricks/models/mart/sales/mart__sales__fact_new_signal.sql",
    },
    {
        "status": "modified",
        "filename": "dbt-databricks/models/mart/sales/some_existing.sql",
    },
    {"status": "added", "filename": "README.md"},  # out of scope
]

MOCK_NEW_SQL = """{{
  auto_config(
    materialized='table',
    databricks_tags={'domain': 'sales', 'sub_domain': 'opportunities'}
  )
}}
-- New sales fact.
SELECT * FROM {{ ref('mart__sales__fact_opportunity') }}
"""


@pytest.fixture
def mock_github_client():
    """Return a fully-mocked GitHubClient honoring the async API."""
    client = MagicMock()
    client.get_pull_request = AsyncMock(
        return_value={"head": {"sha": "deadbeef", "ref": "feat/new-tables"}}
    )
    client.get_pr_files = AsyncMock(return_value=MOCK_PR_FILES)
    client.close = AsyncMock()
    client.upsert_pr_comment = AsyncMock(return_value={"id": 99999})
    client.get_pr_comments = AsyncMock(return_value=[])  # no prior advisor comment
    client.repo_url = "https://api.github.com/repos/algolia/data"
    client.get_file_sha = AsyncMock(return_value="dumsha123")
    client.create_or_update_file = AsyncMock(return_value={"commit": {"sha": "newsha"}})

    async def mock_get_file_content(path: str, ref: str | None = None) -> str | None:
        if path.endswith("governance_state.yaml"):
            return GOVERNANCE_YAML_TEXT
        if path.endswith("capability_matrix.csv"):
            return CAP_MATRIX_TEXT
        if path.endswith("dum.yaml"):
            return DUM_TEXT
        if path.endswith("fact_new_signal.sql"):
            return MOCK_NEW_SQL
        return None

    client.get_file_content = AsyncMock(side_effect=mock_get_file_content)
    # _request is hit by the sibling-YML directory listing; return empty list.
    client._request = AsyncMock(return_value=[])
    return client


@pytest.fixture
def mock_claude_response():
    """Anthropic client mock returning a deterministic classification."""
    import json as _json
    text_block = SimpleNamespace(
        type="text",
        text=_json.dumps(
            {
                "capability_ids": ["OM1"],
                "confidence": "high",
                "rationale": "Pipeline progression signal — fits OM1 functions.",
                "similar_to": "fact_opportunity",
            }
        ),
    )
    resp = SimpleNamespace(content=[text_block])
    anthropic_client = MagicMock()
    anthropic_client.messages.create.return_value = resp
    return anthropic_client


@pytest.mark.asyncio
async def test_handler_end_to_end_dry_run(mock_github_client, mock_claude_response):
    """Handler resolves PR → filter → parse → consult → Claude → comment without posting."""
    with patch(
        "jirade.mcp.handlers.permission_advisor.get_github_client",
        new=AsyncMock(return_value=(mock_github_client, MagicMock())),
    ), patch(
        "jirade.mcp.handlers.permission_advisor.Anthropic",
        return_value=mock_claude_response,
    ), patch(
        "jirade.mcp.handlers.permission_advisor.get_settings",
        return_value=SimpleNamespace(
            anthropic_api_key="test-key", claude_model="claude-opus-4-5-20251101"
        ),
    ):
        result = await handle_permission_advisor_tool(
            "jirade_advise_permissions_for_pr",
            {"pr_number": 1234, "post_comment": False},
        )

    # Structural assertions
    assert result["pr_number"] == 1234
    assert result["head_sha"] == "deadbeef"
    assert result["in_scope_count"] == 1
    assert result["comment_posted"] is False
    assert len(result["decisions"]) == 1

    decision = result["decisions"][0]
    assert decision["table_name"] == "fact_new_signal"
    assert decision["catalog"] == "mart"
    assert decision["schema"] == "sales"
    assert decision["status"] == "llm_proposed"
    assert decision["capability_ids"] == ["OM1"]

    # Comment body sanity
    assert "fact_new_signal" in result["comment_body"]
    assert "OM1" in result["comment_body"]
    assert "jirade:permission-advisor:v1" in result["comment_body"]

    # We did NOT post (dry run)
    mock_github_client.upsert_pr_comment.assert_not_called()


@pytest.mark.asyncio
async def test_handler_posts_when_requested(mock_github_client, mock_claude_response):
    """When post_comment=True, the handler upserts via the comment marker."""
    with patch(
        "jirade.mcp.handlers.permission_advisor.get_github_client",
        new=AsyncMock(return_value=(mock_github_client, MagicMock())),
    ), patch(
        "jirade.mcp.handlers.permission_advisor.Anthropic",
        return_value=mock_claude_response,
    ), patch(
        "jirade.mcp.handlers.permission_advisor.get_settings",
        return_value=SimpleNamespace(
            anthropic_api_key="test-key", claude_model="claude-opus-4-5-20251101"
        ),
    ):
        result = await handle_permission_advisor_tool(
            "jirade_advise_permissions_for_pr",
            {"pr_number": 1234, "post_comment": True},
        )

    assert result["comment_posted"] is True
    mock_github_client.upsert_pr_comment.assert_awaited_once()
    args, kwargs = mock_github_client.upsert_pr_comment.call_args
    # Marker should be passed in
    assert kwargs.get("marker") == "<!-- jirade:permission-advisor:v1 -->"


@pytest.mark.asyncio
async def test_handler_dum_dry_run_proposes_without_committing(
    mock_github_client, mock_claude_response
):
    """Default (apply_dum_edit unset): grants are proposed in the comment but
    dum.yaml is not committed."""
    with patch(
        "jirade.mcp.handlers.permission_advisor.get_github_client",
        new=AsyncMock(return_value=(mock_github_client, MagicMock())),
    ), patch(
        "jirade.mcp.handlers.permission_advisor.Anthropic", return_value=mock_claude_response
    ), patch(
        "jirade.mcp.handlers.permission_advisor.get_settings",
        return_value=SimpleNamespace(
            anthropic_api_key="test-key", claude_model="claude-opus-4-5-20251101"
        ),
    ):
        result = await handle_permission_advisor_tool(
            "jirade_advise_permissions_for_pr", {"pr_number": 1234, "post_comment": False}
        )

    assert result["dum_grants_committed"] is False
    # High-confidence OM1 → Sales Leadership resolves to a dum block; proposed.
    assert "Access grants" in result["comment_body"]
    assert "mart.sales.fact_new_signal" in result["comment_body"]
    assert "Dry-run" in result["comment_body"]
    mock_github_client.create_or_update_file.assert_not_called()


@pytest.mark.asyncio
async def test_handler_dum_apply_commits_to_branch(mock_github_client, mock_claude_response):
    """apply_dum_edit=True writes the high-confidence grant and commits dum.yaml
    to the PR head branch."""
    with patch(
        "jirade.mcp.handlers.permission_advisor.get_github_client",
        new=AsyncMock(return_value=(mock_github_client, MagicMock())),
    ), patch(
        "jirade.mcp.handlers.permission_advisor.Anthropic", return_value=mock_claude_response
    ), patch(
        "jirade.mcp.handlers.permission_advisor.get_settings",
        return_value=SimpleNamespace(
            anthropic_api_key="test-key", claude_model="claude-opus-4-5-20251101"
        ),
    ):
        result = await handle_permission_advisor_tool(
            "jirade_advise_permissions_for_pr",
            {"pr_number": 1234, "apply_dum_edit": True},
        )

    assert result["dum_grants_committed"] is True
    mock_github_client.create_or_update_file.assert_awaited_once()
    _args, kwargs = mock_github_client.create_or_update_file.call_args
    assert kwargs["branch"] == "feat/new-tables"
    assert kwargs["sha"] == "dumsha123"
    # The committed dum content carries the new grant, sorted into the block.
    committed = kwargs["content"] if "content" in kwargs else _args[1]
    assert "mart.sales.fact_new_signal: read" in committed
    assert "Committed to" in result["comment_body"]


@pytest.mark.asyncio
async def test_handler_skips_write_when_comment_unchanged(mock_github_client, mock_claude_response):
    """A re-run whose rendered body matches the existing comment's hash performs
    no PATCH — a true no-op (the advertised idempotency)."""
    # First render the body the handler will produce, then seed it as the
    # existing comment so the hash matches on the second run.
    with patch(
        "jirade.mcp.handlers.permission_advisor.get_github_client",
        new=AsyncMock(return_value=(mock_github_client, MagicMock())),
    ), patch(
        "jirade.mcp.handlers.permission_advisor.Anthropic",
        return_value=mock_claude_response,
    ), patch(
        "jirade.mcp.handlers.permission_advisor.get_settings",
        return_value=SimpleNamespace(
            anthropic_api_key="test-key", claude_model="claude-opus-4-5-20251101"
        ),
    ):
        first = await handle_permission_advisor_tool(
            "jirade_advise_permissions_for_pr",
            {"pr_number": 1234, "post_comment": False},
        )
        # Seed the existing comment with the exact body just rendered.
        mock_github_client.get_pr_comments = AsyncMock(
            return_value=[{"id": 1, "body": first["comment_body"]}]
        )
        second = await handle_permission_advisor_tool(
            "jirade_advise_permissions_for_pr",
            {"pr_number": 1234, "post_comment": True},
        )

    assert second["comment_skipped_no_change"] is True
    assert second["comment_posted"] is False
    mock_github_client.upsert_pr_comment.assert_not_called()


@pytest.mark.asyncio
async def test_handler_skips_when_no_new_tables(mock_github_client):
    """If PR has zero in-scope additions, no Claude call, friendly comment."""
    mock_github_client.get_pr_files = AsyncMock(
        return_value=[{"status": "modified", "filename": "dbt-databricks/models/mart/sales/x.sql"}]
    )
    with patch(
        "jirade.mcp.handlers.permission_advisor.get_github_client",
        new=AsyncMock(return_value=(mock_github_client, MagicMock())),
    ), patch(
        "jirade.mcp.handlers.permission_advisor.get_settings",
        return_value=SimpleNamespace(
            anthropic_api_key="test-key", claude_model="claude-opus-4-5-20251101"
        ),
    ):
        result = await handle_permission_advisor_tool(
            "jirade_advise_permissions_for_pr",
            {"pr_number": 1234, "post_comment": False},
        )

    assert result["in_scope_count"] == 0
    assert result["decisions"] == []
    assert "No new tables" in result["comment_body"]


@pytest.mark.asyncio
async def test_handler_raises_on_missing_governance_state():
    client = MagicMock()
    client.get_pull_request = AsyncMock(return_value={"head": {"sha": "abc", "ref": "feat/x"}})
    client.get_pr_files = AsyncMock(
        return_value=[{"status": "added", "filename": "dbt-databricks/models/mart/sales/x.sql"}]
    )
    client.get_file_content = AsyncMock(return_value=None)  # governance_state missing
    client.close = AsyncMock()
    client.repo_url = "https://api.github.com/repos/algolia/data"

    with patch(
        "jirade.mcp.handlers.permission_advisor.get_github_client",
        new=AsyncMock(return_value=(client, MagicMock())),
    ):
        with pytest.raises(RuntimeError, match="governance_state.yaml not found"):
            await handle_permission_advisor_tool(
                "jirade_advise_permissions_for_pr",
                {"pr_number": 1234},
            )


@pytest.mark.asyncio
async def test_handler_rejects_unknown_tool_name():
    with pytest.raises(ValueError, match="Unknown permission-advisor tool"):
        await handle_permission_advisor_tool("jirade_unknown", {"pr_number": 1})


@pytest.mark.asyncio
async def test_handler_requires_pr_number():
    with patch(
        "jirade.mcp.handlers.permission_advisor.get_github_client",
        new=AsyncMock(),
    ):
        with pytest.raises(ValueError, match="pr_number is required"):
            await handle_permission_advisor_tool(
                "jirade_advise_permissions_for_pr", {}
            )
