"""Permission advisor MCP tool handler.

Wires the pure logic in `jirade/tools/permission_advisor.py` to:
  • the GitHub client (fetch PR files / file content / upsert comment)
  • the Anthropic client (Claude call for un-permissioned new tables)

The handler is thin glue — all decisioning lives in the core module so it
stays unit-testable without network or LLM mocks proliferating here.

Source of truth: `dum.yaml`'s per-division RBAC blocks (`group-division-*`)
tell us what is already permissioned; the capability catalog + the
capability→divisions map are bundled with jirade. There is no
`governance_state.yaml` — the engine runs live per-PR.

Grant application (RBAC migration in progress): high-confidence proposals whose
division has a matching group-division-* block are written into that block and
committed to the PR branch (apply_dum_edit, default on). Divisions with no
matching block, and low/medium-confidence proposals, stay advisory. dbt
domain=Core tables are routed to the shared core block (group-analytics-core-tables).
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

from anthropic import Anthropic

from ...config import get_settings
from ...tools.dum_editor import (
    DUM_PATH,
    apply_grants,
    build_core_tables,
    build_grant_index,
    detect_division_drift,
    dump_dum,
    load_dum,
    render_drift_note,
    render_dum_summary,
)
from ...tools.permission_advisor import (
    BUNDLED_CAPABILITY_MATRIX,
    COMMENT_MARKER,
    build_pr_comment,
    classify_with_claude,
    comment_unchanged,
    consult_dum,
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

# Added + modified: a modified model can still be new-to-dum / unpermissioned.
_IN_SCOPE_STATUSES = ("A", "M")


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
    # Default on: high-confidence grants matching a group-division-* block are
    # committed to the PR branch. Pass apply_dum_edit=false for a dry-run that
    # only reports the proposed grants in the comment.
    apply_dum_edit = bool(arguments.get("apply_dum_edit", True))
    dum_path = arguments.get("dum_path", DUM_PATH)

    client, _auth = await get_github_client(owner, repo)
    try:
        # 1. Resolve PR refs. Model evidence is read at the head SHA (the PR's
        #    view); dum.yaml is read with a base-branch fallback so stale
        #    branches still resolve the existing-grant picture.
        pr_info = await client.get_pull_request(pr_number)
        head_sha = pr_info["head"]["sha"]
        head_ref = pr_info["head"]["ref"]
        base_ref = pr_info["base"]["ref"]

        # 2. Filter PR files to in-scope mart/analytics SQL (added or modified).
        files = await client.get_pr_files(pr_number)
        diff_entries: list[tuple[str, str]] = [
            (_STATUS_CODE.get(f["status"], ""), f["filename"]) for f in files
        ]
        in_scope = filter_in_scope_paths(diff_entries, statuses=_IN_SCOPE_STATUSES)
        logger.info(
            f"permission_advisor: PR #{pr_number} — "
            f"{len(diff_entries)} files, {len(in_scope)} in scope"
        )

        decisions: list = []
        drift = None
        extra_section = ""
        dum_committed = False

        # Nothing in scope → skip every load (a PR with no in-scope tables must
        # not fail on a missing dum.yaml) and render the empty note.
        if in_scope:
            # 3. Load dum.yaml — the source of truth for existing grants. Prefer
            #    the head copy (only it can be safely committed to); fall back to
            #    base so a stale branch still resolves the read-only picture.
            dum_text = await client.get_file_content(dum_path, ref=head_sha)
            dum_on_head = dum_text is not None
            if not dum_text and base_ref and base_ref != head_sha:
                dum_text = await client.get_file_content(dum_path, ref=base_ref)
            if not dum_text:
                raise RuntimeError(
                    f"{dum_path} not found at head {head_sha} or base {base_ref}; "
                    "cannot determine existing permissions. Commit dum.yaml or pass dum_path."
                )
            dum = load_dum(dum_text)
            grant_index = build_grant_index(dum)
            core_tables = build_core_tables(dum)

            with tempfile.TemporaryDirectory() as tmp:
                tmp_root = Path(tmp)

                # 4. Materialize each in-scope SQL file (+ sibling YAML).
                await _materialize_files(client, in_scope, head_sha, tmp_root)

                # 5. Parse + consult dum for each table (granted? core? mv-inherit?)
                for path in in_scope:
                    if not (tmp_root / path).exists():
                        continue
                    ev = parse_table_evidence(tmp_root, path)
                    decisions.append(consult_dum(ev, grant_index, core_tables))

                # 6. Claude only for the un-permissioned subset.
                needs_llm = [d for d in decisions if d.status == "needs_llm"]
                if needs_llm:
                    matrix = load_capability_matrix(BUNDLED_CAPABILITY_MATRIX)
                    settings = get_settings()
                    anthropic_client = Anthropic(api_key=settings.anthropic_api_key)
                    for d in needs_llm:
                        classify_with_claude(
                            d,
                            client=anthropic_client,
                            capability_matrix=matrix,
                            grant_index=grant_index,
                            model=settings.claude_model,
                        )

                # 6.6 Health-check: divisions the classifier proposed that have no
                #     group-division-* block in dum.yaml can't be granted yet.
                proposed_divisions = sorted(
                    {div for d in decisions for div in d.allowed_divisions}
                )
                drift = detect_division_drift(proposed_divisions, dum)
                drift_note = render_drift_note(drift)
                if drift.has_drift:
                    logger.warning(
                        "permission_advisor: proposed divisions with no dum.yaml "
                        f"block: {', '.join(drift.missing_in_dum)}"
                    )

                # 6.7 Apply high-confidence grants to matching group-division-*
                #     blocks. Always computed for the comment; committed to the PR
                #     branch when apply_dum_edit and the dum copy is from the head.
                dum_result = apply_grants(dum, decisions)
                will_commit = apply_dum_edit and dum_on_head and dum_result.changed
                dum_summary = render_dum_summary(dum_result, dum_path, will_commit)
                if will_commit:
                    sha = await client.get_file_sha(dum_path, ref=head_ref)
                    await client.create_or_update_file(
                        dum_path,
                        dump_dum(dum),
                        message=(
                            "chore(governance): grant table access for new "
                            "mart/analytics tables [permission-advisor]"
                        ),
                        branch=head_ref,
                        sha=sha,
                    )
                    dum_committed = True

                extra_section = "\n\n".join(s for s in (drift_note, dum_summary) if s)

        # 7. Render comment (drift note + dum summary embedded so the hash
        #    covers them).
        body = build_pr_comment(decisions, extra_section=extra_section)

        # 8. Optionally post — upsert by marker, but skip the write entirely
        #    when an existing advisor comment already carries the same content
        #    hash (true no-op re-run, no PATCH, no notification).
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
            "dum_grants_committed": dum_committed,
            "not_yet_grantable_divisions": drift.missing_in_dum if drift is not None else [],
        }
    finally:
        await client.close()


# ── helpers ──────────────────────────────────────────────────────────────────
async def _get_file_with_fallback(
    client, path: str, head_sha: str, base_ref: str | None
) -> str | None:
    """Read a repo file at the PR head, falling back to the base branch.

    dum.yaml is not something a PR is expected to carry — reading it from the
    base branch when the head lacks it keeps stale branches (and PRs opened
    before a config change landed) working.
    """
    content = await client.get_file_content(path, ref=head_sha)
    if content is not None:
        return content
    if base_ref and base_ref != head_sha:
        return await client.get_file_content(path, ref=base_ref)
    return None


async def _existing_advisor_comment(
    client, pr_number: int, marker: str = COMMENT_MARKER
) -> str | None:
    """Return the body of the current advisor comment (by marker), or None.

    Shared by the tag advisor, which passes its own marker.
    """
    comments = await client.get_pr_comments(pr_number)
    for comment in comments or []:
        body = comment.get("body", "")
        if marker in body:
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


def _summarize_decision(d) -> dict[str, Any]:
    return {
        "table_name": d.evidence.table_name,
        "catalog": d.evidence.catalog,
        "schema": d.evidence.schema,
        "path": d.evidence.path,
        "status": d.status,
        "capability_ids": d.capability_ids,
        "allowed_divisions": d.allowed_divisions,
        "allowed_division_count": len(d.allowed_divisions),
        "rationale": d.rationale,
    }
