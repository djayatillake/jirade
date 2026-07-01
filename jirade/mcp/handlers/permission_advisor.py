"""Permission advisor MCP tool handler.

Wires the pure logic in `jirade/tools/permission_advisor.py` to:
  • the GitHub client (fetch PR files / file content / upsert comment)
  • the Anthropic client (Claude call for un-classified new tables)

The handler is thin glue — all decisioning lives in the core module so it
stays unit-testable without network or LLM mocks proliferating here.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

import yaml
from anthropic import Anthropic

from ...config import get_settings
from ...tools.permission_advisor import (
    COMMENT_MARKER,
    build_pr_comment,
    classify_with_claude,
    comment_unchanged,
    consult_governance,
    filter_in_scope_paths,
    load_capability_matrix,
    parse_table_evidence,
)
from .github import get_github_client

logger = logging.getLogger(__name__)

# GitHub PR-file status → git-style single-letter code used by the core filter.
_STATUS_CODE = {
    "added":    "A",
    "modified": "M",
    "removed":  "D",
    "renamed":  "R",
    "changed":  "M",
}

DEFAULT_GOVERNANCE_PATH = "dbt-databricks/seeds/governance_state.yaml"
DEFAULT_MATRIX_PATH     = "dbt-databricks/seeds/capability_matrix.csv"


async def handle_permission_advisor_tool(
    name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    """Dispatch entry — currently a single tool, `jirade_advise_permissions_for_pr`."""
    if name != "jirade_advise_permissions_for_pr":
        raise ValueError(f"Unknown permission-advisor tool: {name}")

    owner = arguments.get("owner") or "algolia"
    repo = arguments.get("repo") or "data"
    if "pr_number" not in arguments:
        raise ValueError("pr_number is required")
    pr_number = int(arguments["pr_number"])
    post_comment = bool(arguments.get("post_comment", False))
    governance_path = arguments.get("governance_state_path", DEFAULT_GOVERNANCE_PATH)
    matrix_path = arguments.get("capability_matrix_path", DEFAULT_MATRIX_PATH)

    client, _auth = await get_github_client(owner, repo)
    try:
        # 1. Resolve PR head SHA — we read everything at that exact ref so the
        #    advisor sees the PR's view of the world, not main's.
        pr_info = await client.get_pull_request(pr_number)
        head_sha = pr_info["head"]["sha"]

        # 2. Filter PR files to in-scope additions
        files = await client.get_pr_files(pr_number)
        diff_entries: list[tuple[str, str]] = [
            (_STATUS_CODE.get(f["status"], ""), f["filename"]) for f in files
        ]
        in_scope = filter_in_scope_paths(diff_entries)
        logger.info(
            f"permission_advisor: PR #{pr_number} — "
            f"{len(diff_entries)} files, {len(in_scope)} in scope"
        )

        # 3. Load governance_state.yaml from the PR head
        gov_text = await client.get_file_content(governance_path, ref=head_sha)
        if not gov_text:
            raise RuntimeError(
                f"governance_state.yaml not found at {governance_path}@{head_sha}; "
                "commit it under dbt-databricks/seeds/ or pass --governance_state_path."
            )
        governance_state = yaml.safe_load(gov_text)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_root = Path(tmp)

            # 4. Materialize each in-scope SQL file (and sibling YAML if present)
            #    so parse_table_evidence can read them like a normal checkout.
            await _materialize_files(client, in_scope, head_sha, tmp_root)

            # 5. Parse + consult governance for each new table
            decisions = []
            for path in in_scope:
                if not (tmp_root / path).exists():
                    continue
                ev = parse_table_evidence(tmp_root, path)
                d = consult_governance(ev, governance_state)
                decisions.append(d)

            # 6. Claude only for the subset that needs it
            needs_llm = [d for d in decisions if d.status == "needs_llm"]
            if needs_llm:
                matrix = await _load_matrix(client, matrix_path, head_sha, tmp_root)
                valid_divisions = _divisions_from_state(governance_state)
                settings = get_settings()
                anthropic_client = Anthropic(api_key=settings.anthropic_api_key)
                for d in needs_llm:
                    classify_with_claude(
                        d,
                        client=anthropic_client,
                        capability_matrix=matrix,
                        valid_divisions=valid_divisions,
                        governance_state=governance_state,
                        model=settings.claude_model,
                    )

            # 7. Render comment
            body = build_pr_comment(decisions)

            # 8. Optionally post — upsert by marker, but skip the write entirely
            #    when an existing advisor comment already carries the same
            #    content hash (true no-op re-run, no PATCH, no notification).
            posted = False
            skipped_no_change = False
            if post_comment:
                existing = await _existing_advisor_comment(client, pr_number)
                if existing is not None and comment_unchanged(existing, body):
                    skipped_no_change = True
                else:
                    await client.upsert_pr_comment(pr_number, body, marker=COMMENT_MARKER)
                    posted = True

            return {
                "pr_number": pr_number,
                "owner": owner,
                "repo": repo,
                "head_sha": head_sha,
                "in_scope_count": len(in_scope),
                "decisions": [_summarize_decision(d) for d in decisions],
                "comment_body": body,
                "comment_posted": posted,
                "comment_skipped_no_change": skipped_no_change,
            }
    finally:
        await client.close()


# ── helpers ──────────────────────────────────────────────────────────────────
async def _existing_advisor_comment(client, pr_number: int) -> str | None:
    """Return the body of the current advisor comment (by marker), or None."""
    comments = await client.get_pr_comments(pr_number)
    for comment in comments or []:
        body = comment.get("body", "")
        if COMMENT_MARKER in body:
            return body
    return None


async def _materialize_files(
    client, in_scope: list[str], head_sha: str, tmp_root: Path
) -> None:
    """Write each in-scope SQL + any sibling .yml schema files to tmp_root."""
    sibling_dirs: set[str] = set()
    for path in in_scope:
        content = await client.get_file_content(path, ref=head_sha)
        if content is None:
            logger.warning(f"permission_advisor: could not fetch {path}@{head_sha}")
            continue
        local = tmp_root / path
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_text(content)
        sibling_dirs.add(str(Path(path).parent))

    # For sibling-YML descriptions, fetch any .yml/.yaml in those dirs at HEAD.
    # We use the GitHub Contents API for the directory listing.
    repo_url = client.repo_url  # type: ignore[attr-defined]
    import httpx
    for dir_path in sibling_dirs:
        try:
            data = await client._request("GET", f"{repo_url}/contents/{dir_path}", params={"ref": head_sha})  # type: ignore[attr-defined]
        except httpx.HTTPStatusError:
            continue
        if not isinstance(data, list):
            continue
        for entry in data:
            name = entry.get("name", "")
            if not (name.endswith(".yml") or name.endswith(".yaml")):
                continue
            file_path = f"{dir_path}/{name}"
            content = await client.get_file_content(file_path, ref=head_sha)
            if content is None:
                continue
            local = tmp_root / file_path
            local.parent.mkdir(parents=True, exist_ok=True)
            local.write_text(content)


async def _load_matrix(
    client, matrix_path: str, head_sha: str, tmp_root: Path
) -> list[dict[str, str]]:
    matrix_text = await client.get_file_content(matrix_path, ref=head_sha)
    if not matrix_text:
        logger.warning(
            f"permission_advisor: capability matrix not found at {matrix_path}@{head_sha} — "
            "Claude will be unable to propose caps."
        )
        return []
    local = tmp_root / "capability_matrix.csv"
    local.write_text(matrix_text)
    return load_capability_matrix(local)


def _divisions_from_state(governance_state: dict[str, Any]) -> list[str]:
    """Union of every allowed_division across every capability — the universe
    of valid division labels."""
    out: set[str] = set()
    for cap in governance_state.get("capability_lookup", {}).values():
        for d in cap.get("allowed_divisions", []):
            out.add(d)
    return sorted(out)


def _summarize_decision(d) -> dict[str, Any]:
    return {
        "table_name": d.evidence.table_name,
        "catalog": d.evidence.catalog,
        "schema": d.evidence.schema,
        "path": d.evidence.path,
        "status": d.status,
        "is_core": d.is_core,
        "capability_ids": d.capability_ids,
        "allowed_division_count": len(d.allowed_divisions),
        "rationale": d.rationale,
    }
