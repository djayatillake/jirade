"""Tests for the metric_view smoke-query plumbing.

Covers the small, important things:
- The MEASURE() regex is in the query whitelist and matches the
  expected smoke-query shape.
- _extract_metric_view_measures handles realistic + malformed manifest bodies
  without throwing.
"""

from __future__ import annotations

from jirade.clients.databricks_client import DatabricksMetadataClient
from jirade.mcp.handlers.dbt_diff import _extract_metric_view_measures


class TestMeasureWhitelist:
    """The smoke query is `SELECT MEASURE(<m>) FROM <fqn>` — must be allowed."""

    def setup_method(self) -> None:
        # is_query_allowed is a method on the class; we don't need a real
        # connection to test the regex layer.
        self.client = DatabricksMetadataClient(
            host="x", http_path="x", auth_type="token", token="x"
        )

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

    def test_unaggregated_select_still_blocked(self) -> None:
        # The whole point of the whitelist is "no raw data". MEASURE() returns
        # an aggregate, but bare SELECT col FROM mv would be raw data — keep
        # it blocked even though it'd technically work against a metric view.
        assert not self.client.is_query_allowed(
            "SELECT account_id FROM mart.sales.mv_opportunity"
        )

    def test_measure_with_filter_blocked(self) -> None:
        # We deliberately don't allow WHERE in the smoke query — keeps the
        # pattern strict and predictable. Add it back if smoke tests start
        # needing date scoping.
        assert not self.client.is_query_allowed(
            "SELECT MEASURE(arr) FROM mart.sales.mv WHERE close_date >= '2026-01-01'"
        )


class TestExtractMeasures:
    def test_well_formed_metric_view(self) -> None:
        node = {
            "compiled_code": (
                "version: 0.1\n"
                "source: mart.sales.fact_opportunity\n"
                "measures:\n"
                "  - name: arr_net_new_business\n"
                "    expr: SUM(arr_net_new_business)\n"
                "  - name: opportunities\n"
                "    expr: COUNT(DISTINCT opportunity_id)\n"
            ),
        }
        assert _extract_metric_view_measures(node) == [
            "arr_net_new_business",
            "opportunities",
        ]

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
        # We accept that the fallback returns [] rather than crashing — the
        # smoke test then no-ops gracefully.
        assert _extract_metric_view_measures(node) == []

    def test_empty_body_returns_empty(self) -> None:
        assert _extract_metric_view_measures({"compiled_code": ""}) == []
        assert _extract_metric_view_measures({}) == []

    def test_malformed_yaml_returns_empty(self) -> None:
        assert _extract_metric_view_measures({"compiled_code": "::: not yaml :::"}) == []

    def test_no_measures_block_returns_empty(self) -> None:
        node = {"compiled_code": "version: 0.1\nsource: foo\n"}
        assert _extract_metric_view_measures(node) == []

    def test_measure_without_name_skipped(self) -> None:
        node = {
            "compiled_code": (
                "measures:\n"
                "  - name: good\n"
                "    expr: SUM(x)\n"
                "  - expr: SUM(y)\n"  # missing name
            ),
        }
        assert _extract_metric_view_measures(node) == ["good"]
