"""Permission Advisor — propose allowed divisions for new dbt tables.

This module is the *business logic core* of the Permission Advisor. The MCP
handler (jirade/mcp/handlers/permission_advisor.py) calls into here; same
goes for any CLI surface.

Pipeline (5 stages — only stage 4 needs network/LLM):
  1. filter_in_scope_paths   — keep only added/modified mart/analytics *.sql
  2. parse_table_evidence    — extract metadata from SQL + sibling YML
  3. consult_dum             — read-only lookup vs dum.yaml's per-division grants
                               (skip already-permissioned; mv-inherit if possible)
  4. classify_with_claude    — only invoked for status == 'needs_llm'
  5. build_pr_comment        — render idempotent markdown

The capability catalog (`capability_matrix.csv`) and the capability→divisions
map (`capability_divisions.CAPABILITY_DIVISIONS`) are static inputs bundled with
jirade, so the engine runs live per-PR with no `governance_state.yaml` to sync.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import yaml

from .capability_divisions import CAPABILITY_DIVISIONS, divisions_for, valid_capability_ids

# The capability catalog ships with jirade (see pyproject include).
BUNDLED_CAPABILITY_MATRIX = Path(__file__).parent / "data" / "capability_matrix.csv"

# ── Repo conventions (Algolia-specific — kept in one place, not scattered) ────
# These encode how the algolia/data repo lays out dbt models and names things.
# A different repo would only need to override this block, not the logic below.
DBT_MODELS_PREFIX = "dbt-databricks/models"
IN_SCOPE_CATALOGS = ("mart", "analytics")
SQL_SUFFIX = ".sql"
MODEL_NAME_SEP = "__"  # dbt model names are catalog__schema__table
METRIC_VIEW_PREFIX = "mv_"

# ── Governance vocabulary (surfaced to the operator in prompt + comment) ──────
ORG_NAME = "Algolia"
DIVISION_SOURCE = "HR / Bamboo source of truth"

# dbt domain tag value marking a shared/universal table, and the division label
# it maps to (the shared core block in dum.yaml).
CORE_DOMAIN = "core"
CORE_DIVISION = "Core"

# Default Claude model — single source of truth for the core's LLM default. The
# MCP handler overrides it with the configured settings.claude_model.
DEFAULT_CLAUDE_MODEL = "claude-opus-4-5-20251101"

# Match {{ ref('mart__sales__fact_x') }} — captures the inner model name.
_REF_RE = re.compile(r"\{\{\s*ref\(\s*['\"]([^'\"]+)['\"]\s*\)\s*\}\}")

# Match databricks_tags={'domain':'X','sub_domain':'Y'} inside auto_config().
# Used only as a fallback — production models declare tags in schema.yml (see
# _model_meta_from_sibling_yml); metric-view SQL embeds them in auto_config().
_SQL_TAG_RE = re.compile(r"databricks_tags\s*=\s*\{([^}]*)\}", re.DOTALL)
_KV_RE = re.compile(r"['\"]([a-z_]+)['\"]\s*:\s*['\"]([^'\"]*)['\"]")


# ── Data shapes ──────────────────────────────────────────────────────────────
@dataclass
class TableEvidence:
    """Everything we know about a new table at parse time (no LLM yet)."""

    table_name: str
    catalog: str
    schema: str
    path: str
    description: str = ""
    dbt_domain: str = ""
    dbt_sub_domain: str = ""
    refs: list[str] = field(default_factory=list)


@dataclass
class AdvisorDecision:
    """Per-table outcome after consulting dum.yaml's per-division grants."""

    evidence: TableEvidence
    status: str           # "already_granted" | "inherits_from_ref" | "needs_llm"
                          #   | "llm_proposed" | "llm_failed" (set later)
    capability_ids: list[str] = field(default_factory=list)   # populated for non-LLM paths
    allowed_divisions: list[str] = field(default_factory=list)
    rationale: str = ""
    is_core: bool = False
    # "high" | "medium" | "low" — deterministic paths are "high"; LLM paths carry
    # the model's own confidence. Drives whether a grant is written to dum.yaml.
    confidence: str = ""


# ── Public API ───────────────────────────────────────────────────────────────
def filter_in_scope_paths(
    diff_entries: list[tuple[str, str]],
    statuses: tuple[str, ...] = ("A",),
) -> list[str]:
    """Filter (status, path) pairs to in-scope mart/analytics SQL files.

    Args:
        diff_entries: list of (git_status, path), e.g. [('A', 'dbt-.../foo.sql')].
        statuses: git statuses to keep. Defaults to added-only (`("A",)`) for
                  the permission advisor; the tag advisor passes `("A", "M")`
                  since a modified model can still be untagged/mis-tagged.
                  Deletions and renames are always dropped.

    Returns:
        Paths only (relative to repo root), sorted.
    """
    out: list[str] = []
    for status, path in diff_entries:
        if status not in statuses:
            continue
        if not path.endswith(SQL_SUFFIX):
            continue
        if not path.startswith(DBT_MODELS_PREFIX + "/"):
            continue
        # path looks like dbt-databricks/models/<catalog>/<schema>/.../file.sql
        rest = path[len(DBT_MODELS_PREFIX) + 1 :]
        catalog = rest.split("/", 1)[0]
        if catalog in IN_SCOPE_CATALOGS:
            out.append(path)
    return sorted(out)


def parse_table_evidence(repo_root: Path, path: str) -> TableEvidence:
    """Read the new SQL file (+ sibling YAML, if any) and extract everything
    deterministic about the table — no LLM, no governance lookups yet.

    Args:
        repo_root: absolute path to the cloned algolia/data repo.
        path: repo-relative path to the SQL file.

    Returns:
        TableEvidence dataclass.
    """
    full = repo_root / path

    rest = path[len(DBT_MODELS_PREFIX) + 1 :]
    parts = rest.split("/")
    catalog = parts[0]
    schema = parts[1] if len(parts) > 2 else ""

    table_name = full.stem.rsplit(MODEL_NAME_SEP, 1)[-1]

    try:
        text = full.read_text()
    except OSError:
        text = ""

    refs = [m.rsplit(MODEL_NAME_SEP, 1)[-1] for m in _REF_RE.findall(text)]

    # Tags + description come from the sibling schema.yml (where production
    # models declare them). Fall back to SQL-embedded databricks_tags for the
    # metric-view auto_config() style that carries them inline.
    meta = _model_meta_from_sibling_yml(full.parent, full.stem)
    description = _description_from_meta(meta)
    yml_config = (meta or {}).get("config") or {}
    yml_tags = yml_config.get("databricks_tags") or {}
    domain = (yml_tags.get("domain") or "").strip()
    sub_domain = (yml_tags.get("sub_domain") or "").strip()

    if not domain and not sub_domain:
        tag_m = _SQL_TAG_RE.search(text)
        if tag_m:
            pairs = dict(_KV_RE.findall(tag_m.group(1)))
            domain = pairs.get("domain", "")
            sub_domain = pairs.get("sub_domain", "")

    return TableEvidence(
        table_name=table_name,
        catalog=catalog,
        schema=schema,
        path=path,
        description=description,
        dbt_domain=domain,
        dbt_sub_domain=sub_domain,
        refs=refs,
    )


def table_id_of(evidence: TableEvidence) -> str:
    """The Unity Catalog securable a grant targets: catalog.schema.table."""
    return f"{evidence.catalog}.{evidence.schema}.{evidence.table_name}"


def consult_dum(
    evidence: TableEvidence,
    grant_index: dict[str, set[str]],
    core_tables: set[str] | None = None,
) -> AdvisorDecision:
    """Decide path based on what dum.yaml has already granted.

    Outcomes:
      • already_granted   → table already has a per-division RBAC grant; report it
      • core_domain       → dbt domain=Core → grant to the shared Core group
      • inherits_from_ref → mv_* whose driving ref(s) are already granted
      • needs_llm         → caller must run Claude to propose divisions

    Args:
        evidence: parsed TableEvidence (from parse_table_evidence).
        grant_index: table_id → set(divisions), from dum_editor.build_grant_index.
        core_tables: securables granted under the shared core block (from
            dum_editor.build_core_tables). mv inheritance ignores refs in this set
            so a metric view doesn't inherit a core dimension's grants.
    """
    core_tables = core_tables or set()
    table_id = table_id_of(evidence)

    # Case A: already permissioned — nothing to propose.
    if table_id in grant_index:
        divs = sorted(grant_index[table_id])
        return AdvisorDecision(
            evidence=evidence,
            status="already_granted",
            allowed_divisions=divs,
            rationale=f"Already granted in dum.yaml to {len(divs)} division(s).",
            confidence="high",
        )

    # Case B: dbt domain=Core → shared Core access group (deterministic).
    if (evidence.dbt_domain or "").strip().lower() == CORE_DOMAIN:
        return AdvisorDecision(
            evidence=evidence,
            status="core_domain",
            allowed_divisions=[CORE_DIVISION],
            rationale="Tagged `domain=Core` → shared Core access group.",
            confidence="high",
        )

    # Case C: mv_* inherits divisions from its granted driving table(s).
    #   refs are bare table names, matched against grant_index by suffix. Refs
    #   that are core tables (granted to the shared core block) are ignored so a
    #   generic dimension can't leak its grants into the metric view.
    if evidence.table_name.startswith(METRIC_VIEW_PREFIX) and evidence.refs:
        core_bare = {t.rsplit(".", 1)[-1] for t in core_tables}
        inherited: set[str] = set()
        contributors: list[str] = []
        for ref in evidence.refs:
            if ref in core_bare:
                continue  # core/universal ref — does not contribute divisions
            hit_divs: set[str] = set()
            for tid, divs in grant_index.items():
                if tid.rsplit(".", 1)[-1] == ref:
                    hit_divs |= divs
            if hit_divs:
                contributors.append(ref)
                inherited |= hit_divs
        if inherited:
            return AdvisorDecision(
                evidence=evidence,
                status="inherits_from_ref",
                allowed_divisions=sorted(inherited),
                rationale="Inherited from granted driving table(s): "
                + ", ".join(contributors),
                confidence="high",
            )

    # Case D: needs the LLM to propose divisions.
    return AdvisorDecision(evidence=evidence, status="needs_llm")


def _model_meta_from_sibling_yml(model_dir: Path, model_stem: str) -> dict[str, Any] | None:
    """Return the sibling-schema.yml model entry (name/description/config/…).

    dbt schema YAML files declare models with a `name:`, `description:`, and a
    `config:` block that carries `databricks_tags`. The model_stem is the SQL
    file's stem; we match against either the full stem or its last
    `__`-segment (bare table name). Both `.yml` and `.yaml` are checked.
    """
    bare = model_stem.rsplit(MODEL_NAME_SEP, 1)[-1]
    for yml in [*model_dir.glob("*.yml"), *model_dir.glob("*.yaml")]:
        try:
            data = yaml.safe_load(yml.read_text())
        except (OSError, yaml.YAMLError):
            continue
        if not isinstance(data, dict) or "models" not in data:
            continue
        for m in data.get("models") or []:
            if isinstance(m, dict) and m.get("name") in (model_stem, bare):
                return m
    return None


def _description_from_meta(meta: dict[str, Any] | None) -> str:
    """Pull a usable description out of a schema.yml model entry (skips
    unrendered Jinja doc blocks)."""
    if not meta:
        return ""
    desc = (meta.get("description") or "").strip()
    return desc if desc and not desc.startswith("{{") else ""


# ── Claude layer (only called for status == 'needs_llm') ─────────────────────
class _ClaudeClient(Protocol):
    """Duck-typed Anthropic-like client.

    The Anthropic SDK's `messages.create(...)` shape; we declare it as a
    Protocol so tests can pass a mock without depending on the real SDK.
    """

    def messages(self) -> Any: ...  # pragma: no cover  (structural)


def load_capability_matrix(path: Path) -> list[dict[str, str]]:
    """Read capability_matrix.csv into a list of dicts.

    Expected columns (any extras are ignored, missing ones default to ''):
      ID, Title, Domain, Group, Description, Functions, KPIs
    """
    out: list[dict[str, str]] = []
    with path.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            cid = (row.get("ID") or "").strip()
            if not cid:
                continue
            out.append(
                {
                    "id": cid,
                    "title": (row.get("Title") or "").strip(),
                    "domain": (row.get("Domain") or "").strip(),
                    "group": (row.get("Group") or "").strip(),
                    "description": (row.get("Description") or "").strip(),
                    "functions": (row.get("Functions") or "").strip(),
                    "kpis": (row.get("KPIs") or "").strip(),
                }
            )
    return out


def _format_cap_matrix(caps: list[dict[str, str]]) -> str:
    def _multi(s: str) -> str:
        return ", ".join(p.strip() for p in re.split(r"[\n;]+", s) if p.strip())

    return "\n".join(
        f"- {c['id']} | {c['title']} ({c['group'] or c['domain']})\n"
        f"    Description: {c['description']}\n"
        f"    Functions:   {_multi(c['functions'])}\n"
        f"    KPIs:        {_multi(c['kpis'])}"
        for c in caps
    )


def classify_with_claude(
    decision: AdvisorDecision,
    *,
    client: Any,
    capability_matrix: list[dict[str, str]],
    grant_index: dict[str, set[str]] | None = None,
    model: str | None = None,
    max_caps: int = 2,
) -> AdvisorDecision:
    """Ask Claude to propose 1-2 caps for a needs_llm decision.

    Mutates and returns the input decision (status → 'llm_proposed' on success,
    'llm_failed' on error). Divisions are resolved from the bundled
    capability→divisions map (CAPABILITY_DIVISIONS).
    """
    if decision.status != "needs_llm":
        return decision

    ev = decision.evidence
    grant_index = grant_index or {}

    # Surface what dum.yaml already grants for this table's refs so Claude can
    # lean on precedent ('similar_to') as a signal.
    ref_context = []
    for r in ev.refs:
        divs: set[str] = set()
        for tid, d in grant_index.items():
            if tid.rsplit(".", 1)[-1] == r:
                divs |= d
        if divs:
            ref_context.append(f"  {r}: granted to {', '.join(sorted(divs))}")
    refs_block = "\n".join(ref_context) or "  (none of the refs are granted in dum.yaml)"

    prompt = f"""You are classifying a new dbt table for access governance at {ORG_NAME}.
Pick the best 1-2 capabilities from the matrix below. Be conservative — tighter access wins.

TABLE:           {ev.table_name}
CATALOG.SCHEMA:  {ev.catalog}.{ev.schema}
DBT TAGS:        domain={ev.dbt_domain!r}, sub_domain={ev.dbt_sub_domain!r}
DBT DESCRIPTION: {ev.description or '(none)'}

DRIVING TABLES (and the divisions dum.yaml already grants them, if any):
{refs_block}

CAPABILITY MATRIX (pick from these IDs only):
{_format_cap_matrix(capability_matrix)}

Rules:
- Return at most {max_caps} caps. One cap is better than two; only return two if
  both clearly fit.
- Don't invent cap IDs. Only suggest — a human reviews before access is granted.
- If genuinely unsure, set confidence='low' and explain why.

Output JSON only — no prose:
{{"capability_ids": ["..."], "confidence": "high|medium|low",
  "rationale": "one sentence", "similar_to": "tablename or null"}}
"""

    try:
        resp = client.messages.create(
            model=model or DEFAULT_CLAUDE_MODEL,
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        # Extract first text block from the response.
        text = ""
        for block in getattr(resp, "content", []) or []:
            if getattr(block, "type", "") == "text":
                text = getattr(block, "text", "")
                break
        payload = _extract_json(text)
        cap_ids = [c.strip() for c in payload.get("capability_ids") or [] if c.strip()]
        # Filter to caps that exist in both the matrix and the divisions map.
        matrix_ids = {c["id"] for c in capability_matrix}
        valid = matrix_ids & valid_capability_ids()
        cap_ids = [c for c in cap_ids if c in valid][:max_caps]
        if not cap_ids:
            decision.status = "llm_failed"
            decision.rationale = "Claude returned no valid capability IDs"
            return decision
        decision.status = "llm_proposed"
        decision.capability_ids = cap_ids
        decision.allowed_divisions = divisions_for(cap_ids)
        confidence = payload.get("confidence", "medium")
        decision.confidence = confidence
        sim = payload.get("similar_to") or ""
        decision.rationale = (
            f"[{confidence}] {payload.get('rationale', '').strip()}"
            + (f" (similar to {sim})" if sim and sim != "null" else "")
        )
    except Exception as e:  # noqa: BLE001 — surface any Claude/JSON issue
        decision.status = "llm_failed"
        decision.rationale = f"Claude call failed: {e!s}"
    return decision


def _extract_json(text: str) -> dict[str, Any]:
    """Pull the first JSON object out of Claude's response (tolerant of prose)."""
    if not text:
        return {}
    # Find the first { ... last } span and try to parse it.
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return {}
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {}


# ── Comment builder (pure markdown, idempotent) ──────────────────────────────
COMMENT_MARKER = "<!-- jirade:permission-advisor:v1 -->"


def build_pr_comment(decisions: list[AdvisorDecision], extra_section: str = "") -> str:
    """Render the consolidated PR comment.

    The comment is idempotent: it starts with COMMENT_MARKER and ends with a
    hash line so callers can detect 'no-op re-runs' and avoid spamming.

    extra_section, if given, is inserted before the footer (and thus covered by
    the content hash) — used for the dum.yaml grant summary.
    """
    needs_action = [
        d for d in decisions
        if d.status in ("core_domain", "inherits_from_ref", "llm_proposed", "llm_failed")
    ]
    already = [d for d in decisions if d.status == "already_granted"]

    if not decisions:
        body = (
            f"{COMMENT_MARKER}\n"
            "### 🛡️ Permission Advisor\n\n"
            "No new/unpermissioned tables under `mart` / `analytics` in this PR. ✅\n"
        )
        return _append_hash(body)

    lines: list[str] = [
        COMMENT_MARKER,
        "### 🛡️ Permission Advisor",
        "",
        f"Reviewed **{len(decisions)} table(s)** under `mart` / `analytics`.",
        "",
    ]

    if needs_action:
        lines.append("| Table | Catalog.Schema | Granted? | Proposed divisions | Source |")
        lines.append("|---|---|---|---|---|")
        for d in needs_action:
            ev = d.evidence
            div_count = len(d.allowed_divisions)
            div_summary = (
                f"{div_count} division(s)" if div_count else "_no divisions resolved_"
            )
            src = _short_status(d)
            lines.append(
                f"| `{ev.table_name}` | `{ev.catalog}.{ev.schema}` | ❌ not granted | "
                f"{div_summary} | {src} |"
            )
        lines.append("")
        lines.append("<details><summary>Rationale + full division lists</summary>\n")
        for d in needs_action:
            ev = d.evidence
            caps = "/".join(d.capability_ids)
            cap_note = f" _(caps: {caps})_" if caps else ""
            lines.append(f"**`{ev.table_name}`**{cap_note} — {d.rationale or '_no rationale_'}")
            if d.allowed_divisions:
                lines.append(
                    "<br/>Divisions: " + ", ".join(f"`{x}`" for x in d.allowed_divisions)
                )
            lines.append("")
        lines.append("</details>")

    if already:
        lines.append("")
        lines.append(
            f"<sub>✅ Skipped {len(already)} table(s) already permissioned in `dum.yaml` — "
            f"{', '.join(f'`{d.evidence.table_name}`' for d in already)}</sub>"
        )

    if extra_section:
        lines.append("")
        lines.append(extra_section)

    lines.append("")
    lines.append(
        "> ❓ Disagree? Reply on this PR or adjust the `dum.yaml` grant."
    )
    body = "\n".join(lines) + "\n"
    return _append_hash(body)


def _short_status(d: AdvisorDecision) -> str:
    return {
        "core_domain": "core-tag",
        "inherits_from_ref": "ref-inherit",
        "llm_proposed": "advised",
        "llm_failed": "⚠ needs review",
    }.get(d.status, d.status)


# ── Shared idempotency helpers (marker-parameterized; reused by tag_advisor) ──
def append_content_hash(body: str, marker_id: str) -> str:
    """Append a stable content-hash marker so re-runs can detect no-op identity.

    marker_id is the stable identifier (e.g. 'jirade:permission-advisor:v1');
    the hash is computed over the body as rendered before this line is added.
    """
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]
    return body + f"\n<!-- {marker_id} hash={digest} -->\n"


def content_unchanged(prior_body: str, new_body: str, marker_id: str) -> bool:
    """Return True when both comments carry the same non-empty content hash."""
    pattern = re.escape(marker_id) + r" hash=([0-9a-f]+)"

    def _hash(s: str) -> str:
        m = re.search(pattern, s or "")
        return m.group(1) if m else ""

    return _hash(prior_body) == _hash(new_body) and _hash(new_body) != ""


def _append_hash(body: str) -> str:
    """Permission-advisor content hash (marker `jirade:permission-advisor:v1`)."""
    return append_content_hash(body, "jirade:permission-advisor:v1")


def comment_unchanged(prior_body: str, new_body: str) -> bool:
    """Return True when the trailing hash marker matches between two comments."""
    return content_unchanged(prior_body, new_body, "jirade:permission-advisor:v1")
