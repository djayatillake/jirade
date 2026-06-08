"""Tests for column-level attribution of EXCEPT row diffs.

When the whole-row EXCEPT finds that rows differ, ``compare_tables`` probes
each comparable column on its own to report WHICH columns' values changed —
so a reviewer can confirm a single-column change stayed contained while still
seeing the intended column flagged as changed.

Covers:
- The single-column EXCEPT count query (reused for each probe) is whitelisted.
- Attribution flags exactly the changed column(s) when row counts match.
- Attribution is skipped (with a note) when row counts differ.
- Attribution no-ops cleanly when there is no row diff or when disabled.
- The probe limit truncates and records how many columns were checked.
"""

from __future__ import annotations

from jirade.clients.databricks_client import DatabricksMetadataClient


def _make_client() -> DatabricksMetadataClient:
    # No real connection is needed — every query method is overridden per test.
    return DatabricksMetadataClient(host="x", http_path="x", auth_type="token", token="x")


def _schema(cols: list[str]) -> list[dict[str, str]]:
    return [{"col_name": c, "data_type": "string"} for c in cols]


def _wire(
    client: DatabricksMetadataClient,
    *,
    base_count: int,
    ci_count: int,
    cols: list[str],
    changed: set[str],
) -> None:
    """Stub the query layer: identical schema, given row counts, and an
    EXCEPT that reports whole-row diffs plus per-column diffs for ``changed``."""
    client.get_row_count = lambda table, where_clause=None: (  # type: ignore[method-assign]
        ci_count if "ci" in table else base_count
    )
    client.get_table_schema = lambda table: _schema(cols)  # type: ignore[method-assign]
    client.get_null_count = lambda table, col, where_clause=None: 0  # type: ignore[method-assign]

    def fake_except(table_a, table_b, columns, where_clause=None):
        if len(columns) > 1:
            # whole-row diff: report rows differ so attribution kicks in
            return 5
        return 1 if columns[0] in changed else 0

    client._execute_except_count = fake_except  # type: ignore[method-assign]


class TestSingleColumnExceptWhitelisted:
    """Each probe reuses the existing single-column EXCEPT count query."""

    def setup_method(self) -> None:
        self.client = _make_client()

    def test_single_column_except_count_allowed(self) -> None:
        assert self.client.is_query_allowed(
            "SELECT COUNT(*) FROM (SELECT col FROM cat.sch.a EXCEPT SELECT col FROM cat.sch.b)"
        )

    def test_single_column_except_count_with_where_allowed(self) -> None:
        assert self.client.is_query_allowed(
            "SELECT COUNT(*) FROM ("
            "SELECT col FROM cat.sch.a WHERE dt >= '2026-01-01' AND dt < '2026-01-04' "
            "EXCEPT "
            "SELECT col FROM cat.sch.b WHERE dt >= '2026-01-01' AND dt < '2026-01-04')"
        )


class TestAttribution:
    def test_flags_only_the_changed_column(self) -> None:
        client = _make_client()
        _wire(client, base_count=100, ci_count=100, cols=["a", "b", "c"], changed={"b"})

        result = client.compare_tables(
            "cat.sch.prod", "cat.sch.ci", max_except_rows=500_000
        )

        ed = result["except_diff"]
        assert ed["changed_columns"] == ["b"]
        assert ed["columns_probed"] == 3
        assert "changed_columns_note" not in ed

    def test_no_collateral_when_only_intended_column_changes(self) -> None:
        # The intended column shows up (coverage retained) and nothing else does.
        client = _make_client()
        _wire(client, base_count=42, ci_count=42, cols=["id", "amount", "label"], changed={"amount"})

        result = client.compare_tables(
            "cat.sch.prod", "cat.sch.ci", max_except_rows=500_000
        )
        assert result["except_diff"]["changed_columns"] == ["amount"]

    def test_skipped_when_row_counts_differ(self) -> None:
        client = _make_client()
        _wire(client, base_count=100, ci_count=110, cols=["a", "b"], changed={"a"})

        result = client.compare_tables(
            "cat.sch.prod", "cat.sch.ci", max_except_rows=500_000
        )
        ed = result["except_diff"]
        assert "changed_columns" not in ed
        assert "row counts differ" in ed["changed_columns_note"]

    def test_noop_when_no_row_diff(self) -> None:
        client = _make_client()
        # changed=set() AND override whole-row diff to 0 → no diff at all
        _wire(client, base_count=100, ci_count=100, cols=["a", "b"], changed=set())
        client._execute_except_count = lambda *a, **k: 0  # type: ignore[method-assign]

        result = client.compare_tables(
            "cat.sch.prod", "cat.sch.ci", max_except_rows=500_000
        )
        ed = result["except_diff"]
        assert ed["rows_only_in_ci"] == 0 and ed["rows_only_in_prod"] == 0
        assert "changed_columns" not in ed
        assert "changed_columns_note" not in ed

    def test_disabled_when_max_probes_zero(self) -> None:
        client = _make_client()
        _wire(client, base_count=100, ci_count=100, cols=["a", "b"], changed={"a"})

        result = client.compare_tables(
            "cat.sch.prod", "cat.sch.ci", max_except_rows=500_000, max_column_probes=0
        )
        ed = result["except_diff"]
        assert "changed_columns" not in ed
        assert "changed_columns_note" not in ed

    def test_probe_limit_truncates(self) -> None:
        client = _make_client()
        _wire(client, base_count=100, ci_count=100, cols=["a", "b", "c", "d"], changed={"d"})

        result = client.compare_tables(
            "cat.sch.prod", "cat.sch.ci", max_except_rows=500_000, max_column_probes=2
        )
        ed = result["except_diff"]
        assert ed["columns_probed"] == 2
        assert ed["columns_compared"] == 4
        # only the first 2 sorted columns (a, b) were probed — neither changed
        assert ed["changed_columns"] == []
