"""Tag advisor MCP tool handler.

Wires the pure logic in `jirade/tools/tag_advisor.py` to the GitHub + Anthropic
clients. Mirrors the permission-advisor handler and reuses its file-materialize
and comment-lookup glue so decisioning stays in the pure core.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

from anthropic import Anthropic

from ...config import get_settings
from ...tools.permission_advisor import filter_in_scope_paths, parse_table_evidence
from ...tools.tag_advisor import (
    GOVERNED_TAGS_PATH,
    TAG_COMMENT_MARKER,
    assess_tag_gap,
    build_tag_comment,
    classify_tags_with_claude,
    parse_governed_tags,
    tag_comment_unchanged,
)
from .github import get_github_client
from .permission_advisor import (
    _STATUS_CODE,
    _existing_advisor_comment,
    _get_file_with_fallback,
    _materialize_files,
)

logger = logging.getLogger(__name__)


async def handle_tag_advisor_tool(
    name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    """Dispatch entry — currently a single tool, `jirade_advise_tags_for_pr`."""
    if name != "jirade_advise_tags_for_pr":
        raise ValueError(f"Unknown tag-advisor tool: {name}")

    owner = arguments.get("owner") or "algolia"
    repo = arguments.get("repo") or "data"
    if "pr_number" not in arguments:
        raise ValueError("pr_number is required")
    pr_number = int(arguments["pr_number"])
    post_comment = bool(arguments.get("post_comment", False))
    governed_tags_path = arguments.get("governed_tags_path", GOVERNED_TAGS_PATH)

    client, _auth = await get_github_client(owner, repo)
    try:
        # 1. Resolve PR refs — model evidence at head, config with base fallback.
        pr_info = await client.get_pull_request(pr_number)
        head_sha = pr_info["head"]["sha"]
        base_ref = pr_info["base"]["ref"]

        # 2. Filter PR files to in-scope additions AND modifications — a changed
        #    model can still be untagged or placeholder-tagged.
        files = await client.get_pr_files(pr_number)
        diff_entries: list[tuple[str, str]] = [
            (_STATUS_CODE.get(f["status"], ""), f["filename"]) for f in files
        ]
        in_scope = filter_in_scope_paths(diff_entries, statuses=("A", "M"))
        logger.info(
            f"tag_advisor: PR #{pr_number} — "
            f"{len(diff_entries)} files, {len(in_scope)} in scope"
        )

        decisions = []

        # Nothing in scope → skip the allowlist load (a PR with no mart/analytics
        # models must not fail on a missing governed_tags.yaml) and render empty.
        if in_scope:
            # 3. Load the governed-tag allowlist (head, then base branch).
            gov_text = await _get_file_with_fallback(
                client, governed_tags_path, head_sha, base_ref
            )
            if not gov_text:
                raise RuntimeError(
                    f"governed_tags.yaml not found at {governed_tags_path} "
                    f"(head {head_sha} or base {base_ref}); pass governed_tags_path "
                    "if it lives elsewhere."
                )
            governed_tags = parse_governed_tags(gov_text)

            with tempfile.TemporaryDirectory() as tmp:
                tmp_root = Path(tmp)

                # 4. Materialize each in-scope SQL + sibling schema.yml.
                await _materialize_files(client, in_scope, head_sha, tmp_root)

                # 5. Parse + assess the tag gap for each model.
                for path in in_scope:
                    if not (tmp_root / path).exists():
                        continue
                    ev = parse_table_evidence(tmp_root, path)
                    decisions.append(assess_tag_gap(ev))

                # 6. Claude only for models missing/placeholder-tagged.
                needs = [d for d in decisions if d.status == "needs_suggestion"]
                if needs:
                    settings = get_settings()
                    anthropic_client = Anthropic(api_key=settings.anthropic_api_key)
                    for d in needs:
                        classify_tags_with_claude(
                            d,
                            client=anthropic_client,
                            governed_tags=governed_tags,
                            model=settings.claude_model,
                        )

        # 7. Render comment.
        body = build_tag_comment(decisions)

        # 8. Optionally post — skip the write on an unchanged re-run.
        posted = False
        skipped_no_change = False
        if post_comment:
            existing = await _existing_advisor_comment(
                client, pr_number, TAG_COMMENT_MARKER
            )
            if existing is not None and tag_comment_unchanged(existing, body):
                skipped_no_change = True
            else:
                await client.upsert_pr_comment(
                    pr_number, body, marker=TAG_COMMENT_MARKER
                )
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


def _summarize_decision(d) -> dict[str, Any]:
    return {
        "table_name": d.evidence.table_name,
        "catalog": d.evidence.catalog,
        "schema": d.evidence.schema,
        "path": d.evidence.path,
        "status": d.status,
        "current_domain": d.current_domain,
        "current_sub_domain": d.current_sub_domain,
        "suggested_domain": d.suggested_domain,
        "suggested_sub_domain": d.suggested_sub_domain,
        "confidence": d.confidence,
        "new_value_proposals": d.new_value_proposals,
        "rationale": d.rationale,
    }
