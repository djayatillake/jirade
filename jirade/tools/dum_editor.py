"""dum.yaml editor — apply governance grants to databricks_user_management.

The Permission Advisor decides `table → allowed_divisions`. The *applied* form
of that access lives in
`infra/deployments/databricks_user_management/dum.yaml`, where each
`group-division-*` block lists `catalog.schema.table: read` grants that
terraform turns into Unity Catalog grants for that division's Okta group.

This module writes those grants into dum.yaml for **high-confidence** decisions
only (deterministic mv-inherit + high-confidence LLM). Everything else is left
for a human. Edits are made with ruamel round-trip so the file's comments and
YAML anchors survive and the diff is minimal (added lines only).
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Any

from ruamel.yaml import YAML

from .permission_advisor import AdvisorDecision

# Path to the applied-access source of truth in algolia/data.
DUM_PATH = "infra/deployments/databricks_user_management/dum.yaml"

# Only these top-level blocks are per-division grant targets. The broad shared
# anchor blocks (many divisions under one `groups:`) are deliberately NOT
# touched — adding a table there would grant it far too widely.
DIVISION_BLOCK_PREFIX = "group-division-"

# How a division name appears inside a block's `groups:` list.
_DIVISION_LABEL_PREFIXES = ("Okta Push - Division - ", "Push - Division - ")

READ_PRIVILEGE = "read"


@dataclass
class DumApplyResult:
    """What apply_grants did (and chose not to do)."""

    applied: list[tuple[str, str, str]] = field(default_factory=list)          # (division, block, table)
    already_present: list[tuple[str, str, str]] = field(default_factory=list)  # (division, block, table)
    unmatched_divisions: list[tuple[str, str]] = field(default_factory=list)   # (division, table)
    skipped_low_confidence: list[str] = field(default_factory=list)            # table ids not written

    @property
    def changed(self) -> bool:
        return bool(self.applied)


def _yaml() -> YAML:
    y = YAML()
    y.preserve_quotes = True
    y.width = 4096  # never line-wrap our long table identifiers
    return y


def load_dum(text: str) -> Any:
    """Round-trip-load dum.yaml (preserving comments/anchors)."""
    return _yaml().load(text)


def dump_dum(dum: Any) -> str:
    """Serialize a round-tripped dum document back to text."""
    buf = io.StringIO()
    _yaml().dump(dum, buf)
    return buf.getvalue()


def resolve_division_groups(dum: Any) -> dict[str, str]:
    """Map division display-name → `group-division-*` block key.

    Built by reading each per-division block's `groups:` label(s) and stripping
    the known "… Division - " prefix. Broad shared/anchor blocks are ignored.
    """
    out: dict[str, str] = {}
    for key, block in (dum or {}).items():
        if not (isinstance(key, str) and key.startswith(DIVISION_BLOCK_PREFIX)):
            continue
        if not isinstance(block, dict):
            continue
        for label in block.get("groups") or []:
            name = _division_name(str(label))
            if name and name not in out:
                out[name] = key
    return out


def _division_name(label: str) -> str:
    for prefix in _DIVISION_LABEL_PREFIXES:
        if label.startswith(prefix):
            return label[len(prefix):].strip()
    return ""


def is_high_confidence(decision: AdvisorDecision) -> bool:
    """Whether a decision is trustworthy enough to write to dum.yaml.

    Deterministic mv-inheritance and high-confidence LLM proposals qualify;
    already-classified (existing), llm_failed, and medium/low LLM do not.
    """
    if decision.status == "inherits_from_ref":
        return True
    if decision.status == "llm_proposed" and (decision.confidence or "").lower() == "high":
        return True
    return False


def table_identifier(decision: AdvisorDecision) -> str:
    """The Unity Catalog identifier a grant targets: catalog.schema.table."""
    ev = decision.evidence
    return f"{ev.catalog}.{ev.schema}.{ev.table_name}"


def apply_grants(dum: Any, decisions: list[AdvisorDecision]) -> DumApplyResult:
    """Mutate `dum` in place, adding read grants for high-confidence decisions.

    For each high-confidence decision, add `catalog.schema.table: read` under
    every resolvable allowed-division block. Low-confidence / failed decisions
    are recorded in skipped_low_confidence and left untouched. Divisions with no
    matching block are recorded in unmatched_divisions (never invented).
    """
    resolver = resolve_division_groups(dum)
    result = DumApplyResult()

    for d in decisions:
        table_id = table_identifier(d)
        if not is_high_confidence(d):
            # Only note ones that actually proposed access but weren't confident.
            if d.status in ("llm_proposed", "llm_failed"):
                result.skipped_low_confidence.append(table_id)
            continue
        for division in d.allowed_divisions:
            block_key = resolver.get(division)
            if block_key is None:
                result.unmatched_divisions.append((division, table_id))
                continue
            if _add_table_grant(dum[block_key], table_id):
                result.applied.append((division, block_key, table_id))
            else:
                result.already_present.append((division, block_key, table_id))

    return result


def _add_table_grant(block: Any, table_id: str) -> bool:
    """Insert `- table_id: read` into a block's `tables:` list, sorted.

    Returns True if a grant was added, False if it was already present. Creates
    the `tables:` key if the block doesn't have one.
    """
    tables = block.get("tables")
    if tables is None:
        tables = []
        block["tables"] = tables

    for entry in tables:
        if _entry_key(entry) == table_id:
            return False

    idx = len(tables)
    for i, entry in enumerate(tables):
        if _entry_key(entry) > table_id:
            idx = i
            break
    tables.insert(idx, {table_id: READ_PRIVILEGE})
    return True


def _entry_key(entry: Any) -> str:
    """The single securable identifier a `- ident: read` entry grants on."""
    if isinstance(entry, dict) and entry:
        return str(next(iter(entry.keys())))
    return str(entry)
