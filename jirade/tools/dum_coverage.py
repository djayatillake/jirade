"""DUM division-access coverage check for new dbt tables.

A new table under `mart`/`analytics` is invisible to division-scoped users
(ThoughtSpot / Hex / Sentinel MCP under RBAC) until it is granted in
`infra/deployments/databricks_user_management/dum.yaml` — each
`group-division-*` block enumerates `catalog.schema.table: read` grants that
terraform turns into Unity Catalog grants for that division's Okta group.

This module answers, deterministically and without any LLM or governance
files: **for each NEW table a PR creates, which divisions can see it, and
which divisions probably should?** "Probably should" is derived from the DUM
itself — a division that can already read the new table's direct parents
(by table, schema, or catalog grant) is a candidate for the child too.

Used by `jirade_run_dbt_ci`, which appends the rendered section to the CI
diff report. Pure functions throughout — the caller supplies the loaded
dum.yaml document and the manifest dict.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .dum_editor import resolve_division_groups
from .permission_advisor import IN_SCOPE_CATALOGS

# How many suggested divisions to name in the summary table before eliding.
MAX_SUGGESTIONS_SHOWN = 8


# ── Data shapes ──────────────────────────────────────────────────────────────
@dataclass
class DivisionGrants:
    """Everything one `group-division-*` block can read (or better)."""

    division: str
    block: str
    tables: set[str] = field(default_factory=set)
    schemas: set[str] = field(default_factory=set)
    catalogs: set[str] = field(default_factory=set)

    def covers(self, relation: str) -> bool:
        """Whether this division has any grant that exposes `relation`
        (an exact table grant, or a broader schema/catalog grant)."""
        rel = relation.lower()
        parts = rel.split(".")
        if len(parts) != 3:
            return rel in self.tables
        catalog, schema, _table = parts
        return (
            rel in self.tables
            or f"{catalog}.{schema}" in self.schemas
            or catalog in self.catalogs
        )


@dataclass
class NewTableCoverage:
    """DUM coverage verdict for one new table."""

    model: str                       # dbt model name (catalog__schema__table)
    relation: str                    # catalog.schema.table
    granted_divisions: list[str] = field(default_factory=list)
    # division → the parent relations that division can read (why suggested)
    suggested_divisions: dict[str, list[str]] = field(default_factory=dict)
    parent_relations: list[str] = field(default_factory=list)

    @property
    def has_division_access(self) -> bool:
        return bool(self.granted_divisions)

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "relation": self.relation,
            "granted_divisions": self.granted_divisions,
            "suggested_divisions": self.suggested_divisions,
            "parent_relations": self.parent_relations,
            "has_division_access": self.has_division_access,
        }


# ── Relation naming ──────────────────────────────────────────────────────────
def relation_for_model(model_name: str) -> str | None:
    """catalog__schema__table dbt model name → catalog.schema.table.

    Mirrors the CI convention in `_get_prod_table_name`: two-part names map
    schema onto the catalog; unparseable names return None.
    """
    parts = model_name.split("__")
    if len(parts) >= 3:
        return f"{parts[0]}.{parts[1]}.{'_'.join(parts[2:])}"
    if len(parts) == 2:
        return f"{parts[0]}.{parts[0]}.{parts[1]}"
    return None


def _grant_key(entry: Any) -> str:
    """The securable identifier of a `- ident: privilege` YAML list entry."""
    if isinstance(entry, dict) and entry:
        return str(next(iter(entry.keys())))
    return str(entry)


# ── DUM indexing ─────────────────────────────────────────────────────────────
def build_division_grants(dum: Any) -> list[DivisionGrants]:
    """Index every real division block's grants for coverage lookups.

    Only blocks that `resolve_division_groups` recognises count — i.e. blocks
    carrying an Okta "… Division - <name>" group label. Personal top-up blocks
    (`group-division-x-userN`) and broad shared groups are ignored: a grant
    there is not division-wide access.
    """
    out: list[DivisionGrants] = []
    for division, block_key in sorted(resolve_division_groups(dum).items()):
        block = dum.get(block_key) or {}
        out.append(
            DivisionGrants(
                division=division,
                block=block_key,
                tables={_grant_key(e).lower() for e in block.get("tables") or []},
                schemas={_grant_key(e).lower() for e in block.get("schemas") or []},
                catalogs={_grant_key(e).lower() for e in block.get("catalogs") or []},
            )
        )
    return out


# ── Manifest parent extraction ───────────────────────────────────────────────
def parent_relations_from_manifest(manifest: dict[str, Any], model_name: str) -> list[str]:
    """Direct parents of a model as catalog.schema.table relation ids.

    Model parents resolve via the naming convention (the manifest's own
    database/schema fields reflect the CI overrides, not production); source
    parents resolve via the manifest source node's database/schema/identifier.
    Seeds and other node types are skipped.
    """
    nodes: dict[str, Any] = manifest.get("nodes", {})
    sources: dict[str, Any] = manifest.get("sources", {})

    node_id = next(
        (
            nid
            for nid, node in nodes.items()
            if node.get("resource_type") == "model" and node.get("name") == model_name
        ),
        None,
    )
    if node_id is None:
        return []

    relations: list[str] = []
    for parent_id in nodes[node_id].get("depends_on", {}).get("nodes", []):
        if parent_id.startswith("model."):
            parent = nodes.get(parent_id, {})
            rel = relation_for_model(parent.get("name", ""))
            if rel:
                relations.append(rel)
        elif parent_id.startswith("source."):
            src = sources.get(parent_id, {})
            database = src.get("database", "")
            schema = src.get("schema", "")
            identifier = src.get("identifier") or src.get("name", "")
            if database and schema and identifier:
                relations.append(f"{database}.{schema}.{identifier}")
    return sorted(set(relations))


# ── The check ────────────────────────────────────────────────────────────────
def check_new_table_coverage(
    parents_by_model: dict[str, list[str]], dum: Any
) -> list[NewTableCoverage]:
    """Coverage verdict for each new in-scope model.

    Args:
        parents_by_model: dbt model name → its direct-parent relation ids
            (empty list is fine — suggestions are then empty, coverage still
            evaluated).
        dum: loaded dum.yaml document.

    Returns:
        One NewTableCoverage per model whose catalog is in scope
        (mart/analytics), sorted by relation.
    """
    divisions = build_division_grants(dum)
    out: list[NewTableCoverage] = []

    for model_name, parents in parents_by_model.items():
        relation = relation_for_model(model_name)
        if not relation or relation.split(".")[0] not in IN_SCOPE_CATALOGS:
            continue

        granted = [d.division for d in divisions if d.covers(relation)]
        suggested: dict[str, list[str]] = {}
        for d in divisions:
            if d.division in granted:
                continue
            readable_parents = [p for p in parents if d.covers(p)]
            if readable_parents:
                suggested[d.division] = readable_parents

        out.append(
            NewTableCoverage(
                model=model_name,
                relation=relation,
                granted_divisions=granted,
                suggested_divisions=suggested,
                parent_relations=sorted(set(parents)),
            )
        )

    return sorted(out, key=lambda c: c.relation)


# ── Rendering ────────────────────────────────────────────────────────────────
def render_dum_coverage_section(
    coverages: list[NewTableCoverage],
    dum_path: str = "infra/deployments/databricks_user_management/dum.yaml",
) -> str:
    """Markdown section for the CI diff report. Empty string when no new
    in-scope tables were checked."""
    if not coverages:
        return ""

    missing = [c for c in coverages if not c.has_division_access]

    lines = [
        "### 🔐 Division access check (`dum.yaml`)",
        "",
        f"{len(coverages)} new table(s) under `mart`/`analytics` checked against "
        f"`group-division-*` grants in `{dum_path}` (PR branch).",
        "",
        "| New table | Divisions with access | Suggested divisions (can read its parents) |",
        "|---|---|---|",
    ]
    def _cell_names(names: list[str]) -> str:
        # Division labels can contain "|" (e.g. "AI Search | Personalization"),
        # which would break the markdown table — escape it inside cells.
        shown = ", ".join(f"`{n.replace('|', '\\|')}`" for n in names[:MAX_SUGGESTIONS_SHOWN])
        if len(names) > MAX_SUGGESTIONS_SHOWN:
            shown += f" +{len(names) - MAX_SUGGESTIONS_SHOWN} more"
        return shown

    for c in coverages:
        if c.granted_divisions:
            access = f"✅ {len(c.granted_divisions)}: " + _cell_names(c.granted_divisions)
        else:
            access = "❌ none"
        suggestions = sorted(c.suggested_divisions)
        shown = _cell_names(suggestions) if suggestions else "—"
        lines.append(f"| `{c.relation}` | {access} | {shown} |")

    # Per-table rationale, only where there is something to explain.
    explained = [c for c in coverages if c.suggested_divisions]
    if explained:
        lines.append("")
        lines.append("<details><summary>Why these divisions are suggested</summary>")
        lines.append("")
        for c in explained:
            lines.append(f"**`{c.relation}`**")
            for division in sorted(c.suggested_divisions):
                parents = c.suggested_divisions[division]
                shown_parents = ", ".join(f"`{p}`" for p in parents[:3])
                if len(parents) > 3:
                    shown_parents += f" +{len(parents) - 3} more"
                lines.append(f"- `{division}` reads {shown_parents}")
            lines.append("")
        lines.append("</details>")

    lines.append("")
    if missing:
        rels = ", ".join(f"`{c.relation}`" for c in missing)
        lines.append(
            f"⚠️ **{len(missing)} new table(s) have no `group-division-*` grant** ({rels}) — "
            "division-scoped users (ThoughtSpot / Hex) will not see them. "
            f"Add `- <catalog.schema.table>: read` to the appropriate division blocks in `{dum_path}`, "
            "or confirm the table is intentionally internal-only."
        )
    else:
        lines.append("✅ All new tables have at least one division grant in `dum.yaml`.")

    return "\n".join(lines)
