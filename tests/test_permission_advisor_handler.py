"""Smoke tests for the permission_advisor MCP handler — mocks GitHub + Claude."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from jirade.mcp.handlers.permission_advisor import handle_permission_advisor_tool

FIXTURES = Path(__file__).parent / "fixtures"
DUM_TEXT = (FIXTURES / "dum.yaml").read_text()

# A PR adding ONE new mart/sales SQL file + modifying an already-permissioned
# one (rpt_opportunity is granted in the dum fixture → already_granted skip).
MOCK_PR_FILES = [
    {
        "status": "added",
        "filename": "dbt-databricks/models/mart/sales/mart__sales__fact_new_signal.sql",
    },
    {
        "status": "modified",
        "filename": "dbt-databricks/models/mart/sales/mart__sales__rpt_opportunity.sql",
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
SELECT * FROM {{ ref('mart__sales__rpt_opportunity') }}
"""

MOCK_EXISTING_SQL = "-- already permissioned\nSELECT 1\n"


@pytest.fixture
def mock_github_client():
    """Return a fully-mocked GitHubClient honoring the async API."""
    client = MagicMock()
    client.get_pull_request = AsyncMock(
        return_value={
            "head": {"sha": "deadbeef", "ref": "feat/new-tables"},
            "base": {"ref": "develop"},
        }
    )
    client.get_pr_files = AsyncMock(return_value=MOCK_PR_FILES)
    client.close = AsyncMock()
    client.upsert_pr_comment = AsyncMock(return_value={"id": 99999})
    client.get_pr_comments = AsyncMock(return_value=[])  # no prior advisor comment
    client.repo_url = "https://api.github.com/repos/algolia/data"
    client.get_file_sha = AsyncMock(return_value="dumsha123")
    # Re-read at the branch tip before committing (content + sha together).
    client.get_file_content_and_sha = AsyncMock(return_value=(DUM_TEXT, "dumsha123"))
    client.create_or_update_file = AsyncMock(return_value={"commit": {"sha": "newsha"}})

    async def mock_get_file_content(path: str, ref: str | None = None) -> str | None:
        if path.endswith("dum.yaml"):
            return DUM_TEXT
        if path.endswith("fact_new_signal.sql"):
            return MOCK_NEW_SQL
        if path.endswith("rpt_opportunity.sql"):
            return MOCK_EXISTING_SQL
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
                "similar_to": "rpt_opportunity",
            }
        ),
    )
    resp = SimpleNamespace(content=[text_block])
    anthropic_client = MagicMock()
    anthropic_client.messages.create.return_value = resp
    return anthropic_client


def _patches(gh, claude=None):
    """Common patch stack for the handler's collaborators."""
    stack = [
        patch(
            "jirade.mcp.handlers.permission_advisor.get_github_client",
            new=AsyncMock(return_value=(gh, MagicMock())),
        ),
        patch(
            "jirade.mcp.handlers.permission_advisor.get_settings",
            return_value=SimpleNamespace(
                anthropic_api_key="test-key", claude_model="claude-opus-4-5-20251101"
            ),
        ),
    ]
    if claude is not None:
        stack.append(
            patch("jirade.mcp.handlers.permission_advisor.Anthropic", return_value=claude)
        )
    return stack


def _apply(stack):
    for p in stack:
        p.start()


def _unapply(stack):
    for p in reversed(stack):
        p.stop()


@pytest.mark.asyncio
async def test_handler_end_to_end(mock_github_client, mock_claude_response):
    """Handler resolves PR → filter → parse → consult dum → Claude → comment.

    Two in-scope files: a new table (needs_llm → OM1) and a modified table that
    is already permissioned in dum.yaml (already_granted, no Claude call for it).
    """
    stack = _patches(mock_github_client, mock_claude_response)
    _apply(stack)
    try:
        result = await handle_permission_advisor_tool(
            "jirade_advise_permissions_for_pr",
            {"pr_number": 1234, "post_comment": False, "apply_dum_edit": False},
        )
    finally:
        _unapply(stack)

    assert result["pr_number"] == 1234
    assert result["head_sha"] == "deadbeef"
    assert result["in_scope_count"] == 2  # added + modified, both mart/sales *.sql
    assert result["comment_posted"] is False  # apply_dum_edit=False → no commit, no forced post

    by_name = {d["table_name"]: d for d in result["decisions"]}
    assert by_name["fact_new_signal"]["status"] == "llm_proposed"
    assert by_name["fact_new_signal"]["capability_ids"] == ["OM1"]
    assert by_name["rpt_opportunity"]["status"] == "already_granted"

    assert "fact_new_signal" in result["comment_body"]
    assert "OM1" in result["comment_body"]
    assert "jirade:permission-advisor:v1" in result["comment_body"]
    mock_github_client.upsert_pr_comment.assert_not_called()


@pytest.mark.asyncio
async def test_handler_posts_when_requested(mock_github_client, mock_claude_response):
    """When post_comment=True, the handler upserts via the comment marker."""
    stack = _patches(mock_github_client, mock_claude_response)
    _apply(stack)
    try:
        result = await handle_permission_advisor_tool(
            "jirade_advise_permissions_for_pr",
            {"pr_number": 1234, "post_comment": True},
        )
    finally:
        _unapply(stack)

    assert result["comment_posted"] is True
    mock_github_client.upsert_pr_comment.assert_awaited_once()
    _args, kwargs = mock_github_client.upsert_pr_comment.call_args
    assert kwargs.get("marker") == "<!-- jirade:permission-advisor:v1 -->"


@pytest.mark.asyncio
async def test_handler_commits_high_confidence_grant_by_default(
    mock_github_client, mock_claude_response
):
    """Default (apply_dum_edit unset): high-confidence OM1 → Sales Leadership
    matches group-division-sales-leadership → written and committed to the head."""
    stack = _patches(mock_github_client, mock_claude_response)
    _apply(stack)
    try:
        result = await handle_permission_advisor_tool(
            "jirade_advise_permissions_for_pr",
            {"pr_number": 1234, "post_comment": False},
        )
    finally:
        _unapply(stack)

    assert result["dum_grants_committed"] is True
    mock_github_client.create_or_update_file.assert_awaited_once()
    _args, kwargs = mock_github_client.create_or_update_file.call_args
    assert kwargs["branch"] == "feat/new-tables"
    assert kwargs["sha"] == "dumsha123"
    committed = kwargs["content"] if "content" in kwargs else _args[1]
    assert "mart.sales.fact_new_signal: read" in committed
    assert "Committed to" in result["comment_body"]


@pytest.mark.asyncio
async def test_handler_rereads_dum_at_branch_tip_before_commit(
    mock_github_client, mock_claude_response
):
    """The write re-reads dum.yaml at the branch tip (head_ref) — not the pinned
    head_sha — and commits against that fresh sha, so it can't clobber a
    concurrent edit."""
    stack = _patches(mock_github_client, mock_claude_response)
    _apply(stack)
    try:
        await handle_permission_advisor_tool(
            "jirade_advise_permissions_for_pr", {"pr_number": 1234}
        )
    finally:
        _unapply(stack)

    mock_github_client.get_file_content_and_sha.assert_awaited()
    _a, kw = mock_github_client.get_file_content_and_sha.call_args
    assert kw.get("ref") == "feat/new-tables"  # the moving branch, not head_sha
    _a, ckw = mock_github_client.create_or_update_file.call_args
    assert ckw["sha"] == "dumsha123"  # sha from the same re-read snapshot


@pytest.mark.asyncio
async def test_handler_retries_commit_on_concurrent_edit(
    mock_github_client, mock_claude_response
):
    """If dum.yaml changes between our read and write (sha mismatch → 409), the
    handler re-reads and retries rather than failing or clobbering."""
    import httpx

    resp = MagicMock(status_code=409)
    mock_github_client.create_or_update_file = AsyncMock(
        side_effect=[httpx.HTTPStatusError("conflict", request=MagicMock(), response=resp),
                     {"commit": {"sha": "newsha"}}]
    )
    stack = _patches(mock_github_client, mock_claude_response)
    _apply(stack)
    try:
        result = await handle_permission_advisor_tool(
            "jirade_advise_permissions_for_pr", {"pr_number": 1234}
        )
    finally:
        _unapply(stack)

    assert result["dum_grants_committed"] is True
    assert mock_github_client.create_or_update_file.await_count == 2  # retried once
    assert mock_github_client.get_file_content_and_sha.await_count >= 2  # re-read each attempt


@pytest.mark.asyncio
async def test_handler_dry_run_when_apply_dum_edit_false(
    mock_github_client, mock_claude_response
):
    """apply_dum_edit=False: grants are proposed in the comment but dum.yaml is
    not committed."""
    stack = _patches(mock_github_client, mock_claude_response)
    _apply(stack)
    try:
        result = await handle_permission_advisor_tool(
            "jirade_advise_permissions_for_pr",
            {"pr_number": 1234, "post_comment": False, "apply_dum_edit": False},
        )
    finally:
        _unapply(stack)

    assert result["dum_grants_committed"] is False
    assert "Access grants" in result["comment_body"]
    assert "mart.sales.fact_new_signal" in result["comment_body"]
    assert "Dry-run" in result["comment_body"]
    mock_github_client.create_or_update_file.assert_not_called()


@pytest.mark.asyncio
async def test_handler_reports_not_yet_grantable(mock_github_client, mock_claude_response):
    """The advisor reports proposed divisions that have no dum.yaml block, both in
    the result (for the agent) and the comment (for humans)."""
    stack = _patches(mock_github_client, mock_claude_response)
    _apply(stack)
    try:
        result = await handle_permission_advisor_tool(
            "jirade_advise_permissions_for_pr",
            {"pr_number": 1234, "post_comment": False},
        )
    finally:
        _unapply(stack)

    missing = result["not_yet_grantable_divisions"]
    # OM1 proposes many divisions; the dum fixture only has blocks for Sales
    # Leadership / Finance / Data Analysis, so the rest are not-yet-grantable.
    assert "Revenue Operations" in missing
    assert "Sales Leadership" not in missing  # it HAS a block
    assert "Not yet grantable" in result["comment_body"]


@pytest.mark.asyncio
async def test_handler_skips_write_when_comment_unchanged(mock_github_client, mock_claude_response):
    """A re-run whose rendered body matches the existing comment's hash performs
    no comment PATCH — the advertised idempotency."""
    stack = _patches(mock_github_client, mock_claude_response)
    _apply(stack)
    try:
        first = await handle_permission_advisor_tool(
            "jirade_advise_permissions_for_pr",
            {"pr_number": 1234, "post_comment": False, "apply_dum_edit": False},
        )
        mock_github_client.get_pr_comments = AsyncMock(
            return_value=[{"id": 1, "body": first["comment_body"]}]
        )
        second = await handle_permission_advisor_tool(
            "jirade_advise_permissions_for_pr",
            {"pr_number": 1234, "post_comment": True, "apply_dum_edit": False},
        )
    finally:
        _unapply(stack)

    assert second["comment_skipped_no_change"] is True
    assert second["comment_posted"] is False
    mock_github_client.upsert_pr_comment.assert_not_called()


@pytest.mark.asyncio
async def test_handler_commit_forces_comment(mock_github_client, mock_claude_response):
    """3a: a committed grant is never silent — the explanatory comment is posted
    even when post_comment is False."""
    stack = _patches(mock_github_client, mock_claude_response)
    _apply(stack)
    try:
        result = await handle_permission_advisor_tool(
            "jirade_advise_permissions_for_pr",
            {"pr_number": 1234, "post_comment": False},  # apply defaults true → commits
        )
    finally:
        _unapply(stack)

    assert result["dum_grants_committed"] is True
    assert result["comment_posted"] is True  # forced by the commit
    mock_github_client.upsert_pr_comment.assert_awaited_once()


@pytest.mark.asyncio
async def test_handler_fork_pr_degrades_to_advisory(mock_github_client, mock_claude_response):
    """3b: a fork PR (head repo != base repo) can't be written to — the tool
    skips the commit and stays advisory instead of crashing."""
    mock_github_client.get_pull_request = AsyncMock(
        return_value={
            "head": {"sha": "deadbeef", "ref": "feat/x", "repo": {"full_name": "someone/data-fork"}},
            "base": {"ref": "develop", "repo": {"full_name": "algolia/data"}},
        }
    )
    stack = _patches(mock_github_client, mock_claude_response)
    _apply(stack)
    try:
        result = await handle_permission_advisor_tool(
            "jirade_advise_permissions_for_pr", {"pr_number": 1234, "post_comment": True}
        )
    finally:
        _unapply(stack)

    assert result["dum_grants_committed"] is False           # not written to the fork
    mock_github_client.create_or_update_file.assert_not_called()
    assert "Fork PR" in result["comment_body"]               # advisory note present
    assert result["comment_posted"] is True                  # still gives advice


@pytest.mark.asyncio
async def test_handler_core_block_absent_shows_clarifying_note(
    mock_github_client, mock_claude_response
):
    """#4: a domain=Core table when group-analytics-core-tables isn't in dum.yaml
    yet → clear note, Core not lumped into 'not yet grantable', nothing committed."""
    dum_no_core = (
        "group-division-sales-leadership:\n"
        '  groups:\n    - "Okta Push - Division - Sales Leadership"\n'
        "  tables:\n    - analytics.dimensional.dim_account: read\n"
    )
    core_sql = (
        "{{\n  auto_config(materialized='table', databricks_tags={'domain': 'Core'})\n}}\n"
        "SELECT 1\n"
    )
    mock_github_client.get_pr_files = AsyncMock(return_value=[{
        "status": "added",
        "filename": "dbt-databricks/models/analytics/dimensional/analytics__dimensional__dim_thing.sql",
    }])

    async def gfc(path, ref=None):
        if path.endswith("dum.yaml"):
            return dum_no_core
        if path.endswith("dim_thing.sql"):
            return core_sql
        return None

    mock_github_client.get_file_content = AsyncMock(side_effect=gfc)
    stack = _patches(mock_github_client, mock_claude_response)
    _apply(stack)
    try:
        result = await handle_permission_advisor_tool(
            "jirade_advise_permissions_for_pr", {"pr_number": 1234, "post_comment": False}
        )
    finally:
        _unapply(stack)

    assert any(d["status"] == "core_domain" for d in result["decisions"])
    assert "isn't in `dum.yaml` yet" in result["comment_body"]
    assert "Core" not in result["not_yet_grantable_divisions"]  # explained, not lumped in
    assert result["dum_grants_committed"] is False
    mock_github_client.create_or_update_file.assert_not_called()


@pytest.mark.asyncio
async def test_handler_empty_scope_does_not_require_dum(mock_github_client):
    """A PR with no in-scope tables returns the empty note without loading
    dum.yaml — it must not fail even when dum.yaml is unavailable."""
    mock_github_client.get_pr_files = AsyncMock(
        return_value=[{"status": "modified", "filename": "dbt-databricks/models/staging/x.sql"}]
    )
    mock_github_client.get_file_content = AsyncMock(return_value=None)  # nothing loadable
    stack = _patches(mock_github_client)  # no Claude patch needed
    _apply(stack)
    try:
        result = await handle_permission_advisor_tool(
            "jirade_advise_permissions_for_pr",
            {"pr_number": 1234, "post_comment": False},
        )
    finally:
        _unapply(stack)

    assert result["in_scope_count"] == 0
    assert result["decisions"] == []
    assert "No new/unpermissioned tables" in result["comment_body"]
    mock_github_client.get_file_content.assert_not_called()  # short-circuited


@pytest.mark.asyncio
async def test_handler_reads_dum_from_base_when_missing_at_head(mock_claude_response):
    """dum.yaml absent at a stale head SHA is read from the base branch instead
    of hard-failing."""
    client = MagicMock()
    client.get_pull_request = AsyncMock(
        return_value={"head": {"sha": "stalehead", "ref": "feat/stale"}, "base": {"ref": "develop"}}
    )
    client.get_pr_files = AsyncMock(return_value=MOCK_PR_FILES)
    client.close = AsyncMock()
    client.upsert_pr_comment = AsyncMock(return_value={"id": 1})
    client.get_pr_comments = AsyncMock(return_value=[])
    client.repo_url = "https://api.github.com/repos/algolia/data"
    client._request = AsyncMock(return_value=[])

    async def get_file_content(path: str, ref: str | None = None) -> str | None:
        # dum.yaml exists only on the base branch, not the stale head.
        if path.endswith("dum.yaml"):
            return DUM_TEXT if ref == "develop" else None
        if path.endswith("fact_new_signal.sql"):
            return MOCK_NEW_SQL
        if path.endswith("rpt_opportunity.sql"):
            return MOCK_EXISTING_SQL
        return None

    client.get_file_content = AsyncMock(side_effect=get_file_content)

    stack = _patches(client, mock_claude_response)
    _apply(stack)
    try:
        result = await handle_permission_advisor_tool(
            "jirade_advise_permissions_for_pr", {"pr_number": 1234, "post_comment": False}
        )
    finally:
        _unapply(stack)

    # Ran off base-branch dum instead of raising.
    assert result["in_scope_count"] == 2
    assert any(d["status"] == "already_granted" for d in result["decisions"])


@pytest.mark.asyncio
async def test_handler_degrades_when_dum_missing(mock_github_client, mock_claude_response):
    """#6: dum.yaml missing at head AND base → advisory classification (no grants,
    no write) instead of crashing."""
    mock_github_client.get_pr_files = AsyncMock(return_value=[{
        "status": "added",
        "filename": "dbt-databricks/models/mart/sales/mart__sales__fact_new_signal.sql",
    }])

    async def gfc(path, ref=None):
        if path.endswith("dum.yaml"):
            return None  # missing at head AND base
        if path.endswith("fact_new_signal.sql"):
            return MOCK_NEW_SQL
        return None

    mock_github_client.get_file_content = AsyncMock(side_effect=gfc)
    stack = _patches(mock_github_client, mock_claude_response)
    _apply(stack)
    try:
        result = await handle_permission_advisor_tool(
            "jirade_advise_permissions_for_pr", {"pr_number": 1234, "post_comment": True}
        )
    finally:
        _unapply(stack)

    # Degraded, not crashed: still classified and commented, nothing written.
    assert result["in_scope_count"] == 1
    assert any(d["status"] == "llm_proposed" for d in result["decisions"])
    assert "fact_new_signal" in result["comment_body"]
    assert result["dum_grants_committed"] is False
    mock_github_client.create_or_update_file.assert_not_called()


@pytest.mark.asyncio
async def test_handler_empty_head_dum_not_committed(mock_github_client, mock_claude_response):
    """#5: an empty dum.yaml on the head is not treated as committable — content
    loads from base for context, but nothing is written to (and thus can't
    clobber) the head's file."""
    async def gfc(path, ref=None):
        if path.endswith("dum.yaml"):
            return "" if ref == "deadbeef" else DUM_TEXT  # empty on head, full on base
        if path.endswith("fact_new_signal.sql"):
            return MOCK_NEW_SQL
        if path.endswith("rpt_opportunity.sql"):
            return MOCK_EXISTING_SQL
        return None

    mock_github_client.get_file_content = AsyncMock(side_effect=gfc)
    stack = _patches(mock_github_client, mock_claude_response)
    _apply(stack)
    try:
        result = await handle_permission_advisor_tool(
            "jirade_advise_permissions_for_pr", {"pr_number": 1234}  # apply defaults true
        )
    finally:
        _unapply(stack)

    assert result["dum_grants_committed"] is False        # empty head is not committable
    mock_github_client.create_or_update_file.assert_not_called()  # no clobber of the head file


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
