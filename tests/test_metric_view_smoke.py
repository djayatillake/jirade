"""Tests for the metric_view smoke-query plumbing.

Covers the small, important things:
- The smoke-query regexes are in the query whitelist and match the expected
  shapes: MEASURE()-only (optional WHERE 1=0), and dimension probes where
  WHERE 1=0 is mandatory so no data values can ever be returned.
- _extract_metric_view_fields handles realistic + malformed manifest bodies
  without throwing.
"""

from __future__ import annotations

from jirade.clients.databricks_client import DatabricksMetadataClient
from jirade.mcp.handlers.dbt_diff import _extract_metric_view_fields


class TestSmokeWhitelist:
    def setup_method(self) -> None:
        # is_query_allowed is a method on the class; we don't need a real
        # connection to test the regex layer.
        self.client = DatabricksMetadataClient(
            host="x", http_path="x", auth_type="token", token="x"
        )

    # -- shape 1: MEASURE()-only ------------------------------------------

    def test_basic_measure_query_allowed(self) -> None:
        assert self.client.is_query_allowed(
            "SELECT MEASURE(arr_net_new_business) FROM mart.sales.mv_opportunity"
        )

    def test_measure_query_with_alias_allowed(self) -> None:
        assert self.client.is_query_allowed(
            "SELECT MEASURE(arr_net_new_business) AS arr FROM mart.sales.mv_opportunity"
        )

    def test_measure_query_backtick_quoted_identifiers_allowed(self) -> None:
        assert self.client.is_query_allowed(
            "SELECT MEASURE(`arr`) FROM `mart`.`sales`.`mv_opportunity`"
        )

    def test_measure_query_zero_scan_allowed(self) -> None:
        assert self.client.is_query_allowed(
            "SELECT MEASURE(`arr`) FROM mart.sales.mv_opportunity WHERE 1=0"
        )

    def test_combined_measures_zero_scan_allowed(self) -> None:
        assert self.client.is_query_allowed(
            "SELECT MEASURE(`arr`), MEASURE(`opportunities`) "
            "FROM mart.sales.mv_opportunity WHERE 1=0"
        )

    # -- shape 2: dimensions require WHERE 1=0 ----------------------------

    def test_dims_and_measures_zero_scan_allowed(self) -> None:
        assert self.client.is_query_allowed(
            "SELECT `close_quarter`, `sales_region`, MEASURE(`arr`) "
            "FROM mart.sales.mv_opportunity WHERE 1=0 GROUP BY ALL"
        )

    def test_single_dim_zero_scan_allowed(self) -> None:
        assert self.client.is_query_allowed(
            "SELECT `sales_region` FROM mart.sales.mv_opportunity WHERE 1=0 GROUP BY ALL"
        )

    def test_unaggregated_select_without_filter_still_blocked(self) -> None:
        # The whole point of the whitelist is "no raw data". A bare dim select
        # with no WHERE 1=0 would return real values — must stay blocked.
        assert not self.client.is_query_allowed(
            "SELECT account_id FROM mart.sales.mv_opportunity"
        )

    def test_dims_without_filter_blocked_even_with_group_by(self) -> None:
        # GROUP BY ALL without WHERE 1=0 returns distinct dim values — blocked.
        assert not self.client.is_query_allowed(
            "SELECT `sales_region` FROM mart.sales.mv_opportunity GROUP BY ALL"
        )

    def test_measure_with_real_filter_blocked(self) -> None:
        # Only the constant-false predicate is allowed — an arbitrary WHERE
        # could smuggle expressions; keep the pattern strict and predictable.
        assert not self.client.is_query_allowed(
            "SELECT MEASURE(arr) FROM mart.sales.mv WHERE close_date >= '2026-01-01'"
        )

    def test_expression_select_item_blocked(self) -> None:
        assert not self.client.is_query_allowed(
            "SELECT upper(sales_region) FROM mart.sales.mv_opportunity WHERE 1=0 GROUP BY ALL"
        )


class TestExtractFields:
    def test_well_formed_metric_view(self) -> None:
        node = {
            "compiled_code": (
                "version: 0.1\n"
                "source: mart.sales.fact_opportunity\n"
                "dimensions:\n"
                "  - name: sales_region\n"
                "    expr: sales_region\n"
                "  - name: close_quarter\n"
                "    expr: DATE_TRUNC('quarter', close_date)\n"
                "measures:\n"
                "  - name: arr_net_new_business\n"
                "    expr: SUM(arr_net_new_business)\n"
                "  - name: opportunities\n"
                "    expr: COUNT(DISTINCT opportunity_id)\n"
            ),
        }
        assert _extract_metric_view_fields(node) == {
            "measures": ["arr_net_new_business", "opportunities"],
            "dimensions": ["sales_region", "close_quarter"],
        }

    def test_falls_back_to_raw_code_when_compiled_missing(self) -> None:
        node = {
            "raw_code": (
                "{{ auto_config(materialized='metric_view') }}\n\n"
                "version: 0.1\n"
                "measures:\n"
                "  - name: cost\n"
                "    expr: SUM(cost)\n"
            ),
        }
        # raw_code starts with a Jinja block that yaml.safe_load won't parse.
        # We accept that the fallback returns empty lists rather than crashing
        # — the smoke test then no-ops gracefully.
        assert _extract_metric_view_fields(node) == {"measures": [], "dimensions": []}

    def test_empty_body_returns_empty(self) -> None:
        assert _extract_metric_view_fields({"compiled_code": ""}) == {
            "measures": [],
            "dimensions": [],
        }
        assert _extract_metric_view_fields({}) == {"measures": [], "dimensions": []}

    def test_malformed_yaml_returns_empty(self) -> None:
        assert _extract_metric_view_fields({"compiled_code": "::: not yaml :::"}) == {
            "measures": [],
            "dimensions": [],
        }

    def test_no_measures_block_returns_empty_measures(self) -> None:
        node = {"compiled_code": "version: 0.1\nsource: foo\n"}
        assert _extract_metric_view_fields(node) == {"measures": [], "dimensions": []}

    def test_field_without_name_skipped(self) -> None:
        node = {
            "compiled_code": (
                "dimensions:\n"
                "  - name: region\n"
                "    expr: region\n"
                "  - expr: territory\n"  # missing name
                "measures:\n"
                "  - name: good\n"
                "    expr: SUM(x)\n"
                "  - expr: SUM(y)\n"  # missing name
            ),
        }
        assert _extract_metric_view_fields(node) == {
            "measures": ["good"],
            "dimensions": ["region"],
        }
