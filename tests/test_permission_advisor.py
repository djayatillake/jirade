"""Tests for the Permission Advisor core (parse, governance, Claude, comment)."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from jirade.tools.permission_advisor import (
    AdvisorDecision,
    TableEvidence,
    build_pr_comment,
    classify_with_claude,
    comment_unchanged,
    consult_governance,
    filter_in_scope_paths,
    load_capability_matrix,
    load_governance_state,
    parse_table_evidence,
)

FIXTURES = Path(__file__).parent / "fixtures"
REPO_ROOT = FIXTURES / "repo_root"
GOVERNANCE_YAML = FIXTURES / "governance_state.yaml"
CAP_MATRIX_CSV = FIXTURES / "capability_matrix.csv"


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

    def test_drops_modifications(self):
        diff = [("M", "dbt-databricks/models/mart/sales/foo.sql")]
        assert filter_in_scope_paths(diff) == []

    def test_drops_deletions_and_renames(self):
        diff = [
            ("D", "dbt-databricks/models/mart/sales/foo.sql"),
            ("R100", "dbt-databricks/models/mart/sales/bar.sql"),
        ]
        assert filter_in_scope_paths(diff) == []

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


# ── consult_governance ───────────────────────────────────────────────────────
class TestConsultGovernance:
    @pytest.fixture
    def state(self):
        return load_governance_state(GOVERNANCE_YAML)

    def test_already_classified_path(self, state):
        ev = TableEvidence(
            table_name="fact_opportunity",
            catalog="mart",
            schema="sales",
            path="dbt-databricks/models/mart/sales/mart__sales__fact_opportunity.sql",
        )
        d = consult_governance(ev, state)
        assert d.status == "already_classified"
        assert d.capability_ids == ["OM1", "SPM"]
        # Union of OM1 + SPM allowed divisions
        assert "Sales Leadership" in d.allowed_divisions
        assert "Finance" in d.allowed_divisions
        assert "Marketing Operations" in d.allowed_divisions

    def test_mv_inherits_from_classified_ref(self, state):
        ev = TableEvidence(
            table_name="mv_new_usage_signal",
            catalog="mart",
            schema="customer_success",
            path="dbt-databricks/models/.../mv_new_usage_signal.sql",
            refs=["rpt_current_usage", "dim_account"],  # dim_account is core → skipped
        )
        d = consult_governance(ev, state)
        assert d.status == "inherits_from_ref"
        assert d.capability_ids == ["SUB"]
        assert "Customer Solutions" in d.allowed_divisions
        assert "rpt_current_usage=SUB" in d.rationale

    def test_new_table_needs_llm(self, state):
        ev = TableEvidence(
            table_name="fact_new_sales_signal",
            catalog="mart",
            schema="sales",
            path="dbt-databricks/models/mart/sales/...sql",
            refs=["fact_opportunity", "dim_account"],
        )
        d = consult_governance(ev, state)
        # fact_new_sales_signal is not in TABLE_OVERRIDES, and it's not an mv_*,
        # so inheritance path doesn't fire — caller must invoke Claude.
        assert d.status == "needs_llm"
        assert d.capability_ids == []

    def test_core_table_flag_set(self, state):
        ev = TableEvidence(
            table_name="dim_account",
            catalog="analytics",
            schema="dimensional",
            path="dbt-databricks/models/analytics/dimensional/...sql",
        )
        d = consult_governance(ev, state)
        assert d.is_core is True
        # dim_account is also in TABLE_OVERRIDES, so it's already_classified
        assert d.status == "already_classified"

    def test_mv_with_only_core_refs_still_needs_llm(self, state):
        # If every ref is core (universally accessible), there's nothing to
        # inherit; the mv itself needs proper classification.
        ev = TableEvidence(
            table_name="mv_core_only",
            catalog="mart",
            schema="sales",
            path="dbt-databricks/models/mart/sales/...sql",
            refs=["dim_account", "dim_date"],
        )
        d = consult_governance(ev, state)
        assert d.status == "needs_llm"


# ── load_governance_state ────────────────────────────────────────────────────
class TestLoadGovernanceState:
    def test_loads_fixture(self):
        state = load_governance_state(GOVERNANCE_YAML)
        assert "table_overrides" in state
        assert "capability_lookup" in state
        assert state["table_overrides"]["fact_opportunity"] == "OM1/SPM"

    def test_rejects_missing_keys(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text("table_overrides: {}\n")  # missing capability_lookup
        with pytest.raises(ValueError, match="capability_lookup"):
            load_governance_state(bad)


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
    def state(self):
        return load_governance_state(GOVERNANCE_YAML)

    @pytest.fixture
    def matrix(self):
        return load_capability_matrix(CAP_MATRIX_CSV)

    @pytest.fixture
    def divisions(self):
        # Tight list — what Bamboo would return.
        return ["Sales Leadership", "Sales Operations", "Marketing Operations", "Finance"]

    def test_classifies_new_table_with_valid_caps(self, state, matrix, divisions):
        decision = AdvisorDecision(
            evidence=TableEvidence(
                table_name="fact_new_sales_signal",
                catalog="mart",
                schema="sales",
                path="dbt-databricks/models/mart/sales/foo.sql",
                refs=["fact_opportunity"],
                dbt_domain="sales",
            ),
            status="needs_llm",
        )
        client = _make_mock_claude(
            {
                "capability_ids": ["OM1"],
                "confidence": "high",
                "rationale": "Pipeline progression signal, matches OM1 KPIs.",
                "similar_to": "fact_opportunity",
            }
        )
        out = classify_with_claude(
            decision,
            client=client,
            capability_matrix=matrix,
            valid_divisions=divisions,
            governance_state=state,
        )
        assert out.status == "llm_proposed"
        assert out.capability_ids == ["OM1"]
        # OM1's allowed_divisions in our fixture: Sales Leadership, Sales Operations, Marketing Operations
        assert "Sales Leadership" in out.allowed_divisions
        assert "[high]" in out.rationale
        assert "fact_opportunity" in out.rationale

    def test_filters_invented_cap_ids(self, state, matrix, divisions):
        decision = AdvisorDecision(
            evidence=TableEvidence(
                table_name="fact_x", catalog="mart", schema="sales", path="x.sql"
            ),
            status="needs_llm",
        )
        # Claude hallucinates a non-existent cap
        client = _make_mock_claude(
            {
                "capability_ids": ["FAKE", "OM1"],
                "confidence": "medium",
                "rationale": "Mixed signals",
                "similar_to": None,
            }
        )
        out = classify_with_claude(
            decision,
            client=client,
            capability_matrix=matrix,
            valid_divisions=divisions,
            governance_state=state,
        )
        assert out.status == "llm_proposed"
        assert out.capability_ids == ["OM1"]  # FAKE stripped, OM1 kept

    def test_marks_llm_failed_when_no_valid_caps(self, state, matrix, divisions):
        decision = AdvisorDecision(
            evidence=TableEvidence(
                table_name="fact_x", catalog="mart", schema="sales", path="x.sql"
            ),
            status="needs_llm",
        )
        client = _make_mock_claude(
            {"capability_ids": ["FAKE", "ALSO_FAKE"], "confidence": "low", "rationale": "n/a"}
        )
        out = classify_with_claude(
            decision,
            client=client,
            capability_matrix=matrix,
            valid_divisions=divisions,
            governance_state=state,
        )
        assert out.status == "llm_failed"

    def test_skips_non_needs_llm_decisions(self, state, matrix, divisions):
        decision = AdvisorDecision(
            evidence=TableEvidence(
                table_name="fact_opportunity",
                catalog="mart",
                schema="sales",
                path="x.sql",
            ),
            status="already_classified",
            capability_ids=["OM1", "SPM"],
        )
        client = MagicMock()  # should NOT be called
        out = classify_with_claude(
            decision,
            client=client,
            capability_matrix=matrix,
            valid_divisions=divisions,
            governance_state=state,
        )
        assert out.status == "already_classified"
        client.messages.create.assert_not_called()

    def test_handles_api_exception_gracefully(self, state, matrix, divisions):
        decision = AdvisorDecision(
            evidence=TableEvidence(
                table_name="fact_x", catalog="mart", schema="sales", path="x.sql"
            ),
            status="needs_llm",
        )
        client = MagicMock()
        client.messages.create.side_effect = RuntimeError("API down")
        out = classify_with_claude(
            decision,
            client=client,
            capability_matrix=matrix,
            valid_divisions=divisions,
            governance_state=state,
        )
        assert out.status == "llm_failed"
        assert "API down" in out.rationale


# ── build_pr_comment ─────────────────────────────────────────────────────────
class TestBuildPrComment:
    def test_empty_decisions(self):
        body = build_pr_comment([])
        assert "<!-- jirade:permission-advisor:v1 -->" in body
        assert "No new tables" in body
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
        assert "OM1" in body
        assert "advised" in body
        assert "Sales Leadership" in body
        assert "<details>" in body

    def test_skipped_classified_appears_in_footer(self):
        skipped = AdvisorDecision(
            evidence=TableEvidence(
                table_name="dim_account",
                catalog="analytics",
                schema="dimensional",
                path="x.sql",
            ),
            status="already_classified",
            capability_ids=["CDM", "MDM"],
        )
        body = build_pr_comment([skipped])
        assert "dim_account" in body
        assert "Skipped 1 table" in body

    def test_inherits_from_ref_shown_with_label(self):
        d = AdvisorDecision(
            evidence=TableEvidence(
                table_name="mv_thing",
                catalog="mart",
                schema="customer_success",
                path="x.sql",
            ),
            status="inherits_from_ref",
            capability_ids=["SUB"],
            allowed_divisions=["Customer Solutions", "Finance"],
            rationale="Inherited from driving table(s): rpt_current_usage=SUB",
        )
        body = build_pr_comment([d])
        assert "ref-inherit" in body
        assert "SUB" in body

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
