"""Tests for the Permission Advisor core (parse, dum lookup, Claude, comment)."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from jirade.tools.capability_divisions import divisions_for
from jirade.tools.dum_editor import build_core_tables, build_grant_index, load_dum
from jirade.tools.permission_advisor import (
    AdvisorDecision,
    TableEvidence,
    build_pr_comment,
    classify_with_claude,
    comment_unchanged,
    consult_dum,
    filter_in_scope_paths,
    load_capability_matrix,
    parse_table_evidence,
    table_id_of,
)

FIXTURES = Path(__file__).parent / "fixtures"
REPO_ROOT = FIXTURES / "repo_root"
DUM_YAML = FIXTURES / "dum.yaml"
CAP_MATRIX_CSV = FIXTURES / "capability_matrix.csv"


@pytest.fixture
def grant_index():
    """table_id → set(divisions) built from the dum fixture's per-division blocks."""
    return build_grant_index(load_dum(DUM_YAML.read_text()))


@pytest.fixture
def core_tables():
    """Securables granted under the core block (group-analytics-core-tables) in the dum fixture."""
    return build_core_tables(load_dum(DUM_YAML.read_text()))


# ── filter_in_scope_paths ────────────────────────────────────────────────────
class TestFilterInScopePaths:
    def test_keeps_mart_and_analytics_added_sql(self):
        diff = [
            ("A", "dbt-databricks/models/mart/sales/foo.sql"),
            ("A", "dbt-databricks/models/analytics/dimensional/bar.sql"),
        ]
        assert filter_in_scope_paths(diff) == [
            "dbt-databricks/models/analytics/dimensional/bar.sql",
            "dbt-databricks/models/mart/sales/foo.sql",
        ]

    def test_drops_modifications_by_default(self):
        diff = [("M", "dbt-databricks/models/mart/sales/foo.sql")]
        assert filter_in_scope_paths(diff) == []

    def test_keeps_modifications_when_requested(self):
        # The handler passes ("A", "M") — a modified model can still be unpermissioned.
        diff = [("M", "dbt-databricks/models/mart/sales/foo.sql")]
        assert filter_in_scope_paths(diff, statuses=("A", "M")) == [
            "dbt-databricks/models/mart/sales/foo.sql"
        ]

    def test_drops_deletions_and_renames(self):
        diff = [
            ("D", "dbt-databricks/models/mart/sales/foo.sql"),
            ("R100", "dbt-databricks/models/mart/sales/bar.sql"),
        ]
        assert filter_in_scope_paths(diff, statuses=("A", "M")) == []

    def test_drops_non_dbt_paths(self):
        diff = [
            ("A", "scripts/foo.sql"),
            ("A", "dbt-databricks/models/staging/foo.sql"),  # wrong catalog
            ("A", "dbt-databricks/models/mart/sales/foo.py"),  # wrong suffix
        ]
        assert filter_in_scope_paths(diff) == []

    def test_handles_empty_input(self):
        assert filter_in_scope_paths([]) == []


# ── parse_table_evidence ─────────────────────────────────────────────────────
class TestParseTableEvidence:
    def test_extracts_path_metadata(self):
        path = "dbt-databricks/models/mart/sales/mart__sales__fact_new_sales_signal.sql"
        ev = parse_table_evidence(REPO_ROOT, path)
        assert ev.table_name == "fact_new_sales_signal"
        assert ev.catalog == "mart"
        assert ev.schema == "sales"
        assert ev.path == path

    def test_extracts_databricks_tags(self):
        ev = parse_table_evidence(
            REPO_ROOT,
            "dbt-databricks/models/mart/sales/mart__sales__fact_new_sales_signal.sql",
        )
        assert ev.dbt_domain == "sales"
        assert ev.dbt_sub_domain == "opportunities"

    def test_extracts_refs(self):
        ev = parse_table_evidence(
            REPO_ROOT,
            "dbt-databricks/models/mart/sales/mart__sales__fact_new_sales_signal.sql",
        )
        assert ev.refs == ["fact_opportunity", "dim_account"]

    def test_pulls_description_from_sibling_yml(self):
        ev = parse_table_evidence(
            REPO_ROOT,
            "dbt-databricks/models/mart/sales/mart__sales__fact_new_sales_signal.sql",
        )
        assert "AE-touched account progression signal" in ev.description

    def test_handles_mv_with_inheritance_target(self):
        path = (
            "dbt-databricks/models/mart/customer_success/metric_views/"
            "mart__customer_success__mv_new_usage_signal.sql"
        )
        ev = parse_table_evidence(REPO_ROOT, path)
        assert ev.table_name == "mv_new_usage_signal"
        assert ev.catalog == "mart"
        assert ev.schema == "customer_success"
        assert ev.refs == ["rpt_current_usage", "dim_account"]
        assert ev.dbt_domain == "customer_success_professional_services"


# ── consult_dum ──────────────────────────────────────────────────────────────
class TestConsultDum:
    def test_already_granted_path(self, grant_index):
        ev = TableEvidence(
            table_name="dim_account",
            catalog="analytics",
            schema="dimensional",
            path="dbt-databricks/models/analytics/dimensional/...sql",
        )
        d = consult_dum(ev, grant_index)
        assert d.status == "already_granted"
        assert d.allowed_divisions == ["Sales Leadership"]
        assert d.confidence == "high"

    def test_table_id_matches_dum_identifier(self):
        ev = TableEvidence(
            table_name="dim_account", catalog="analytics", schema="dimensional", path="x.sql"
        )
        assert table_id_of(ev) == "analytics.dimensional.dim_account"

    def test_core_domain_routes_to_core_group(self, grant_index):
        ev = TableEvidence(
            table_name="dim_new_ref",
            catalog="analytics",
            schema="dimensional",
            path="x.sql",
            dbt_domain="Core",
        )
        d = consult_dum(ev, grant_index)
        assert d.status == "core_domain"
        assert d.allowed_divisions == ["Core"]
        assert d.confidence == "high"

    def test_core_domain_is_case_insensitive(self, grant_index):
        for tag in ("core", "CORE", " Core "):
            ev = TableEvidence(
                table_name="dim_x", catalog="analytics", schema="dimensional",
                path="x.sql", dbt_domain=tag,
            )
            assert consult_dum(ev, grant_index).status == "core_domain"

    def test_mv_inheritance_skips_core_refs(self, grant_index, core_tables):
        # dim_calendar is a core table (the core block (group-analytics-core-tables)); rpt_opportunity is
        # granted to Sales Leadership. Inheritance must ignore the core ref.
        ev = TableEvidence(
            table_name="mv_mixed",
            catalog="mart",
            schema="sales",
            path="x.sql",
            refs=["dim_calendar", "rpt_opportunity"],
        )
        d = consult_dum(ev, grant_index, core_tables)
        assert d.status == "inherits_from_ref"
        assert d.allowed_divisions == ["Sales Leadership"]  # Core excluded
        assert "rpt_opportunity" in d.rationale
        assert "dim_calendar" not in d.rationale

    def test_mv_with_only_core_refs_needs_llm(self, grant_index, core_tables):
        ev = TableEvidence(
            table_name="mv_core_only",
            catalog="mart",
            schema="sales",
            path="x.sql",
            refs=["dim_calendar"],  # core-only → nothing to inherit
        )
        assert consult_dum(ev, grant_index, core_tables).status == "needs_llm"

    def test_mv_inherits_divisions_from_granted_ref(self, grant_index):
        ev = TableEvidence(
            table_name="mv_new_signal",
            catalog="mart",
            schema="sales",
            path="dbt-databricks/models/.../mv_new_signal.sql",
            refs=["rpt_opportunity", "dim_date"],  # rpt_opportunity → Sales Leadership
        )
        d = consult_dum(ev, grant_index)
        assert d.status == "inherits_from_ref"
        assert "Sales Leadership" in d.allowed_divisions  # from rpt_opportunity
        assert "Data Analysis" in d.allowed_divisions      # from dim_date
        assert "rpt_opportunity" in d.rationale

    def test_new_table_needs_llm(self, grant_index):
        ev = TableEvidence(
            table_name="fact_brand_new",
            catalog="mart",
            schema="sales",
            path="dbt-databricks/models/mart/sales/...sql",
            refs=["rpt_opportunity"],
        )
        d = consult_dum(ev, grant_index)
        # Not in any grant block, and not an mv_* → caller must invoke Claude.
        assert d.status == "needs_llm"
        assert d.capability_ids == []

    def test_mv_with_no_granted_refs_needs_llm(self, grant_index):
        ev = TableEvidence(
            table_name="mv_ungrounded",
            catalog="mart",
            schema="sales",
            path="dbt-databricks/models/mart/sales/...sql",
            refs=["some_ungranted_table"],
        )
        d = consult_dum(ev, grant_index)
        assert d.status == "needs_llm"


# ── load_capability_matrix ──────────────────────────────────────────────────
class TestLoadCapabilityMatrix:
    def test_parses_matrix_fixture(self):
        caps = load_capability_matrix(CAP_MATRIX_CSV)
        ids = [c["id"] for c in caps]
        assert ids == ["CDM", "MDM", "OM1", "SPM", "SUB", "MM", "LGM"]
        sub = next(c for c in caps if c["id"] == "SUB")
        assert sub["title"] == "Subscription & Usage Billing"
        assert "MRR" in sub["kpis"]


# ── classify_with_claude ─────────────────────────────────────────────────────
def _make_mock_claude(payload: dict) -> MagicMock:
    """Return a mock Anthropic client whose messages.create returns `payload`."""
    import json as _json

    text_block = SimpleNamespace(type="text", text=_json.dumps(payload))
    resp = SimpleNamespace(content=[text_block])
    client = MagicMock()
    client.messages.create.return_value = resp
    return client


class TestClassifyWithClaude:
    @pytest.fixture
    def matrix(self):
        return load_capability_matrix(CAP_MATRIX_CSV)

    def test_classifies_new_table_with_valid_caps(self, matrix, grant_index):
        decision = AdvisorDecision(
            evidence=TableEvidence(
                table_name="fact_new_sales_signal",
                catalog="mart",
                schema="sales",
                path="dbt-databricks/models/mart/sales/foo.sql",
                refs=["rpt_opportunity"],
                dbt_domain="sales",
            ),
            status="needs_llm",
        )
        client = _make_mock_claude(
            {
                "capability_ids": ["OM1"],
                "confidence": "high",
                "rationale": "Pipeline progression signal, matches OM1 KPIs.",
                "similar_to": "rpt_opportunity",
            }
        )
        out = classify_with_claude(
            decision, client=client, capability_matrix=matrix, grant_index=grant_index
        )
        assert out.status == "llm_proposed"
        assert out.capability_ids == ["OM1"]
        # Divisions resolve from the bundled capability→divisions map.
        assert out.allowed_divisions == divisions_for(["OM1"])
        assert "Sales Leadership" in out.allowed_divisions
        assert "[high]" in out.rationale
        assert "rpt_opportunity" in out.rationale

    def test_filters_invented_cap_ids(self, matrix):
        decision = AdvisorDecision(
            evidence=TableEvidence(
                table_name="fact_x", catalog="mart", schema="sales", path="x.sql"
            ),
            status="needs_llm",
        )
        client = _make_mock_claude(
            {
                "capability_ids": ["FAKE", "OM1"],
                "confidence": "medium",
                "rationale": "Mixed signals",
                "similar_to": None,
            }
        )
        out = classify_with_claude(decision, client=client, capability_matrix=matrix)
        assert out.status == "llm_proposed"
        assert out.capability_ids == ["OM1"]  # FAKE stripped, OM1 kept

    def test_marks_llm_failed_when_no_valid_caps(self, matrix):
        decision = AdvisorDecision(
            evidence=TableEvidence(
                table_name="fact_x", catalog="mart", schema="sales", path="x.sql"
            ),
            status="needs_llm",
        )
        client = _make_mock_claude(
            {"capability_ids": ["FAKE", "ALSO_FAKE"], "confidence": "low", "rationale": "n/a"}
        )
        out = classify_with_claude(decision, client=client, capability_matrix=matrix)
        assert out.status == "llm_failed"

    def test_skips_non_needs_llm_decisions(self, matrix):
        decision = AdvisorDecision(
            evidence=TableEvidence(
                table_name="dim_account",
                catalog="analytics",
                schema="dimensional",
                path="x.sql",
            ),
            status="already_granted",
            allowed_divisions=["Sales Leadership"],
        )
        client = MagicMock()  # should NOT be called
        out = classify_with_claude(decision, client=client, capability_matrix=matrix)
        assert out.status == "already_granted"
        client.messages.create.assert_not_called()

    def test_handles_api_exception_gracefully(self, matrix):
        decision = AdvisorDecision(
            evidence=TableEvidence(
                table_name="fact_x", catalog="mart", schema="sales", path="x.sql"
            ),
            status="needs_llm",
        )
        client = MagicMock()
        client.messages.create.side_effect = RuntimeError("API down")
        out = classify_with_claude(decision, client=client, capability_matrix=matrix)
        assert out.status == "llm_failed"
        assert "API down" in out.rationale


# ── build_pr_comment ─────────────────────────────────────────────────────────
class TestBuildPrComment:
    def test_empty_decisions(self):
        body = build_pr_comment([])
        assert "<!-- jirade:permission-advisor:v1 -->" in body
        assert "No new/unpermissioned tables" in body
        assert "hash=" in body

    def test_renders_table_with_advised_caps(self):
        d = AdvisorDecision(
            evidence=TableEvidence(
                table_name="fact_new_signal",
                catalog="mart",
                schema="sales",
                path="x.sql",
            ),
            status="llm_proposed",
            capability_ids=["OM1"],
            allowed_divisions=["Sales Leadership", "Marketing Operations"],
            rationale="[high] new pipeline signal",
        )
        body = build_pr_comment([d])
        assert "fact_new_signal" in body
        assert "mart.sales" in body
        assert "OM1" in body            # shown in the details caps note
        assert "advised" in body
        assert "not granted" in body    # grant-status column
        assert "Sales Leadership" in body
        assert "<details>" in body

    def test_skipped_granted_appears_in_footer(self):
        skipped = AdvisorDecision(
            evidence=TableEvidence(
                table_name="dim_account",
                catalog="analytics",
                schema="dimensional",
                path="x.sql",
            ),
            status="already_granted",
            allowed_divisions=["Sales Leadership"],
        )
        body = build_pr_comment([skipped])
        assert "dim_account" in body
        assert "Skipped 1 table" in body
        assert "already permissioned" in body

    def test_inherits_from_ref_shown_with_label(self):
        d = AdvisorDecision(
            evidence=TableEvidence(
                table_name="mv_thing",
                catalog="mart",
                schema="customer_success",
                path="x.sql",
            ),
            status="inherits_from_ref",
            allowed_divisions=["Sales Leadership", "Finance"],
            rationale="Inherited from granted driving table(s): rpt_opportunity",
        )
        body = build_pr_comment([d])
        assert "ref-inherit" in body
        assert "mv_thing" in body
        assert "Finance" in body

    def test_llm_failed_shown_with_warning(self):
        d = AdvisorDecision(
            evidence=TableEvidence(
                table_name="weird_table",
                catalog="mart",
                schema="other",
                path="x.sql",
            ),
            status="llm_failed",
            rationale="Claude call failed: timeout",
        )
        body = build_pr_comment([d])
        assert "weird_table" in body
        assert "needs review" in body


# ── comment_unchanged (idempotency check) ────────────────────────────────────
class TestCommentUnchanged:
    def test_same_hash_returns_true(self):
        body = build_pr_comment([])
        assert comment_unchanged(body, body) is True

    def test_different_content_returns_false(self):
        a = build_pr_comment([])
        b = build_pr_comment(
            [
                AdvisorDecision(
                    evidence=TableEvidence(
                        table_name="x", catalog="mart", schema="sales", path="x.sql"
                    ),
                    status="llm_proposed",
                    capability_ids=["OM1"],
                    allowed_divisions=["Sales Leadership"],
                )
            ]
        )
        assert comment_unchanged(a, b) is False

    def test_missing_hash_returns_false(self):
        assert comment_unchanged("random text", "other text") is False
