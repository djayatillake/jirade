"""Tag Advisor — propose `databricks_tags` (domain / sub_domain) for dbt models.

Sibling of the Permission Advisor: same evidence-parsing pipeline, but instead
of proposing access divisions it proposes the governed Databricks tags a new or
changed mart/analytics model should carry. It is advisory-only — it renders a
copy-pasteable `schema.yml` suggestion and, when no governed value fits, a gated
`governed_tags.yaml` addition. It never mutates anything.

Pipeline (only stage 3 needs the LLM):
  1. filter_in_scope_paths(statuses=("A","M"))  — added + changed models
     parse_table_evidence                       — reused from permission_advisor
  2. assess_tag_gap        — is the model untagged or placeholder-tagged?
  3. classify_tags_with_claude — propose governed values (or a new-value + sign-off)
  4. build_tag_comment     — render idempotent markdown

Source of truth for the allowlist is the terraform-applied
`infra/deployments/databricks_governed_tags/governed_tags.yaml`; `main.tf` feeds
that same file into `databricks_tag_policy` via `yamldecode`, so editing the
YAML is sufficient — no `.tf` change is needed to add a value.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import yaml

from .permission_advisor import (
    ORG_NAME,
    TableEvidence,
    append_content_hash,
    content_unchanged,
)

# ── Repo conventions ─────────────────────────────────────────────────────────
# Where the governed-tag allowlist lives in algolia/data. The .tf in the same
# directory derives databricks_tag_policy from this YAML, so it is the single
# editable source of truth.
GOVERNED_TAGS_PATH = "infra/deployments/databricks_governed_tags/governed_tags.yaml"

# Governed values that exist purely as placeholders — a model carrying one still
# needs a real classification, so we treat them the same as an absent tag.
PLACEHOLDER_TAG_VALUES = frozenset({"", "tbd", "unclassified"})

DEFAULT_CLAUDE_MODEL = "claude-opus-4-5-20251101"


# ── Data shapes ──────────────────────────────────────────────────────────────
@dataclass
class TagDecision:
    """Per-model outcome for tag suggestion."""

    evidence: TableEvidence
    status: str  # "ok" | "needs_suggestion" | "suggested" | "needs_new_value" | "llm_failed"
    current_domain: str = ""
    current_sub_domain: str = ""
    suggested_domain: str = ""
    suggested_sub_domain: str = ""
    confidence: str = ""
    rationale: str = ""
    # (tag_key, value) pairs that are NOT yet in the governed allowlist and
    # would need a governed_tags.yaml addition (governance sign-off).
    new_value_proposals: list[tuple[str, str]] = field(default_factory=list)


# ── Governed-tag allowlist ───────────────────────────────────────────────────
def parse_governed_tags(yaml_text: str) -> dict[str, list[str]]:
    """Parse governed_tags.yaml → {tag_key: [allowed value names]}.

    Shape (see the file docstring): a `tag_policies` list, each with a
    `tag_key` and a `values` list of `{name: ...}` entries. main.tf feeds this
    exact file into databricks_tag_policy via yamldecode.
    """
    data = yaml.safe_load(yaml_text) or {}
    out: dict[str, list[str]] = {}
    for policy in data.get("tag_policies", []) or []:
        if not isinstance(policy, dict):
            continue
        key = policy.get("tag_key")
        if not key:
            continue
        out[key] = [
            v["name"].strip()
            for v in (policy.get("values") or [])
            if isinstance(v, dict) and v.get("name")
        ]
    return out


def _selectable(values: list[str]) -> list[str]:
    """Governed values minus placeholders — what Claude is allowed to pick."""
    return [v for v in values if v.lower() not in PLACEHOLDER_TAG_VALUES]


# ── Stage 2: gap assessment (pure, no LLM) ────────────────────────────────────
def assess_tag_gap(evidence: TableEvidence) -> TagDecision:
    """Decide whether this model needs a domain-tag suggestion.

    A model is in scope for suggestion when its `domain` tag is absent or a
    placeholder (`tbd` / `unclassified`). A real governed domain → status 'ok'.
    """
    domain = (evidence.dbt_domain or "").strip()
    sub_domain = (evidence.dbt_sub_domain or "").strip()
    needs = domain.lower() in PLACEHOLDER_TAG_VALUES
    return TagDecision(
        evidence=evidence,
        status="needs_suggestion" if needs else "ok",
        current_domain=domain,
        current_sub_domain=sub_domain,
    )


# ── Stage 3: Claude, constrained to the governed allowlist ────────────────────
def classify_tags_with_claude(
    decision: TagDecision,
    *,
    client: Any,
    governed_tags: dict[str, list[str]],
    model: str | None = None,
) -> TagDecision:
    """Ask Claude to propose a governed domain (+ optional sub_domain).

    Mutates and returns the decision:
      • 'suggested'       — proposed values are all in the allowlist
      • 'needs_new_value' — Claude proposes a value not yet governed; recorded
                            in new_value_proposals for a gated governed_tags.yaml
                            addition (the schema.yml suggestion still uses it)
      • 'llm_failed'      — API error, no domain, or only a placeholder returned
    Never invents beyond what Claude returns; placeholder picks are rejected.
    """
    if decision.status != "needs_suggestion":
        return decision

    ev = decision.evidence
    domains = _selectable(governed_tags.get("domain", []))
    sub_domains = _selectable(governed_tags.get("sub_domain", []))

    prompt = f"""You are assigning Databricks governance tags to a dbt model at {ORG_NAME}.
Pick the single best `domain` (required) and, only if clearly applicable, a `sub_domain`.

MODEL:           {ev.table_name}
CATALOG.SCHEMA:  {ev.catalog}.{ev.schema}
CURRENT DOMAIN:  {decision.current_domain or '(none)'} {'(placeholder — needs a real value)' if decision.current_domain else ''}
DESCRIPTION:     {ev.description or '(none)'}
DRIVING TABLES:  {', '.join(ev.refs) or '(none)'}

GOVERNED domain VALUES (pick ONE of these):
{', '.join(domains)}

GOVERNED sub_domain VALUES (pick ONE, or null):
{', '.join(sub_domains)}

Rules:
- Prefer an existing governed value. Only propose a NEW value if none genuinely
  fit — then set the value to your proposed new name and confidence accordingly.
- Never pick `tbd` or `unclassified`.
- If genuinely unsure, set confidence='low' and explain why.

Output JSON only — no prose:
{{"domain": "...", "sub_domain": "... or null", "confidence": "high|medium|low",
  "rationale": "one sentence"}}
"""

    try:
        resp = client.messages.create(
            model=model or DEFAULT_CLAUDE_MODEL,
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        text = ""
        for block in getattr(resp, "content", []) or []:
            if getattr(block, "type", "") == "text":
                text = getattr(block, "text", "")
                break
        payload = _extract_json(text)
    except Exception as e:  # noqa: BLE001 — surface any Claude/JSON issue
        decision.status = "llm_failed"
        decision.rationale = f"Claude call failed: {e!s}"
        return decision

    domain = (payload.get("domain") or "").strip()
    sub_domain = (payload.get("sub_domain") or "").strip()
    if sub_domain.lower() in {"null", "none"}:
        sub_domain = ""

    if not domain or domain.lower() in PLACEHOLDER_TAG_VALUES:
        decision.status = "llm_failed"
        decision.rationale = "Claude did not return a usable domain value"
        return decision

    governed_domains = set(governed_tags.get("domain", []))
    governed_sub = set(governed_tags.get("sub_domain", []))

    new_values: list[tuple[str, str]] = []
    if domain not in governed_domains:
        new_values.append(("domain", domain))
    if sub_domain and sub_domain not in governed_sub:
        new_values.append(("sub_domain", sub_domain))

    decision.suggested_domain = domain
    decision.suggested_sub_domain = sub_domain
    decision.confidence = payload.get("confidence", "medium")
    decision.rationale = (payload.get("rationale") or "").strip()
    decision.new_value_proposals = new_values
    decision.status = "needs_new_value" if new_values else "suggested"
    return decision


def _extract_json(text: str) -> dict[str, Any]:
    """Pull the first JSON object out of Claude's response (tolerant of prose)."""
    if not text:
        return {}
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return {}
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {}


# ── governed_tags.yaml addition (never .tf — the .tf derives from the YAML) ───
def propose_governed_tag_addition(tag_key: str, new_value: str) -> str:
    """Render a suggested governed_tags.yaml edit appending a new allowed value.

    Editing the YAML is sufficient: main.tf's `for` loop turns it into a
    databricks_tag_policy value automatically — no HCL change required.
    """
    return (
        f"In `{GOVERNED_TAGS_PATH}`, under `- tag_key: {tag_key}` → `values:` add:\n"
        f"```yaml\n      - name: {new_value}\n```"
    )


# ── Stage 4: comment builder (pure markdown, idempotent) ─────────────────────
TAG_COMMENT_MARKER = "<!-- jirade:tag-advisor:v1 -->"
_TAG_MARKER_ID = "jirade:tag-advisor:v1"

_SOURCE_LABEL = {
    "suggested": "suggested",
    "needs_new_value": "⚠ new value",
    "llm_failed": "⚠ needs review",
}


def build_tag_comment(decisions: list[TagDecision]) -> str:
    """Render the consolidated tag-advisor PR comment (idempotent via hash)."""
    needs_action = [
        d for d in decisions if d.status in ("suggested", "needs_new_value", "llm_failed")
    ]
    ok = [d for d in decisions if d.status == "ok"]

    if not decisions:
        body = (
            f"{TAG_COMMENT_MARKER}\n"
            "### 🏷️ Tag Advisor\n\n"
            "No new/changed models under `mart` / `analytics` needing a `domain` tag. ✅\n"
        )
        return append_content_hash(body, _TAG_MARKER_ID)

    lines: list[str] = [
        TAG_COMMENT_MARKER,
        "### 🏷️ Tag Advisor",
        "",
        f"Reviewed **{len(decisions)} model(s)** for governed `databricks_tags`.",
        "",
    ]

    if needs_action:
        lines.append("| Model | Catalog.Schema | Current | Suggested | Conf | Source |")
        lines.append("|---|---|---|---|---|---|")
        for d in needs_action:
            ev = d.evidence
            lines.append(
                f"| `{ev.table_name}` | `{ev.catalog}.{ev.schema}` | "
                f"{_fmt_tag(d.current_domain, d.current_sub_domain)} | "
                f"{_fmt_tag(d.suggested_domain, d.suggested_sub_domain)} | "
                f"{d.confidence or '—'} | {_SOURCE_LABEL.get(d.status, d.status)} |"
            )
        lines.append("")
        lines.append("<details><summary>Rationale + suggested schema.yml</summary>\n")
        for d in needs_action:
            lines.append(f"**`{d.evidence.table_name}`** — {d.rationale or '_no rationale_'}")
            if d.suggested_domain:
                lines.append(_schema_yml_suggestion(d))
            lines.append("")
        lines.append("</details>")

        new_values = _dedup_new_values(needs_action)
        if new_values:
            lines.append("")
            lines.append("#### ⚠ New governed tag values — requires governance sign-off")
            lines.append("")
            lines.append(
                "These values are **not yet in the allowlist**. Adding them is a "
                "governance change; approve before merging:"
            )
            lines.append("")
            for tag_key, value in new_values:
                lines.append(f"- {propose_governed_tag_addition(tag_key, value)}")
            lines.append("")

    if ok:
        lines.append("")
        lines.append(
            f"<sub>{len(ok)} model(s) already carry a governed `domain` tag — "
            f"{', '.join(f'`{d.evidence.table_name}`' for d in ok)}</sub>"
        )

    lines.append("")
    lines.append(
        f"> 🏷️ Suggestions are advisory — apply them to the model's `schema.yml`. "
        f"New values also need a `{GOVERNED_TAGS_PATH.split('/')[-1]}` change."
    )
    body = "\n".join(lines) + "\n"
    return append_content_hash(body, _TAG_MARKER_ID)


def tag_comment_unchanged(prior_body: str, new_body: str) -> bool:
    """True when two tag-advisor comments carry the same content hash."""
    return content_unchanged(prior_body, new_body, _TAG_MARKER_ID)


# ── Internal helpers ─────────────────────────────────────────────────────────
def _fmt_tag(domain: str, sub_domain: str) -> str:
    if not domain:
        return "—"
    return f"`{domain}`" + (f" / `{sub_domain}`" if sub_domain else "")


def _schema_yml_suggestion(d: TagDecision) -> str:
    rows = [
        "    config:",
        "      databricks_tags:",
        f"        domain: {d.suggested_domain}",
    ]
    if d.suggested_sub_domain:
        rows.append(f"        sub_domain: {d.suggested_sub_domain}")
    return "```yaml\n" + "\n".join(rows) + "\n```"


def _dedup_new_values(decisions: list[TagDecision]) -> list[tuple[str, str]]:
    """Union of every new-value proposal across models, order-preserving."""
    seen: set[tuple[str, str]] = set()
    out: list[tuple[str, str]] = []
    for d in decisions:
        for pair in d.new_value_proposals:
            if pair not in seen:
                seen.add(pair)
                out.append(pair)
    return out
