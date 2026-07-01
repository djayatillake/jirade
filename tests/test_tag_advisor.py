"""Tests for the Tag Advisor core (governed allowlist, gap, Claude, comment)."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from jirade.tools.permission_advisor import (
    TableEvidence,
    filter_in_scope_paths,
    parse_table_evidence,
)
from jirade.tools.tag_advisor import (
    GOVERNED_TAGS_PATH,
    TagDecision,
    assess_tag_gap,
    build_tag_comment,
    classify_tags_with_claude,
    parse_governed_tags,
    propose_governed_tag_addition,
    tag_comment_unchanged,
)

FIXTURES = Path(__file__).parent / "fixtures"
REPO_ROOT = FIXTURES / "repo_root"
GOVERNED_TAGS_YAML = FIXTURES / "governed_tags.yaml"

GROWTH_DIR = "dbt-databricks/models/mart/growth"
PLACEHOLDER_MODEL = f"{GROWTH_DIR}/mart__growth__rpt_new_activation_signal.sql"
TAGGED_MODEL = f"{GROWTH_DIR}/mart__growth__rpt_existing_adoption.sql"


def _make_mock_claude(payload: dict) -> MagicMock:
    import json as _json

    text_block = SimpleNamespace(type="text", text=_json.dumps(payload))
    resp = SimpleNamespace(content=[text_block])
    client = MagicMock()
    client.messages.create.return_value = resp
    return client


# ── parse_governed_tags ──────────────────────────────────────────────────────
class TestParseGovernedTags:
    def test_parses_fixture(self):
        gov = parse_governed_tags(GOVERNED_TAGS_YAML.read_text())
        assert "domain" in gov and "sub_domain" in gov
        assert "sales" in gov["domain"]
        assert "growth" in gov["domain"]
        # placeholders are governed values (present in the file) — kept here,
        # excluded only at selection time.
        assert "tbd" in gov["domain"]
        assert "usage" in gov["sub_domain"]

    def test_tolerates_empty_or_malformed(self):
        assert parse_governed_tags("") == {}
        assert parse_governed_tags("tag_policies: []\n") == {}
        # A policy missing tag_key is skipped, not fatal.
        assert parse_governed_tags("tag_policies:\n  - values: []\n") == {}


# ── parse_table_evidence reads tags from schema.yml (shared review fix) ───────
class TestSchemaYmlTagReading:
    def test_reads_placeholder_domain_from_schema_yml(self):
        ev = parse_table_evidence(REPO_ROOT, PLACEHOLDER_MODEL)
        assert ev.dbt_domain == "tbd"
        assert ev.dbt_sub_domain == ""
        assert "activation signal" in ev.description

    def test_reads_real_domain_and_sub_domain_from_schema_yml(self):
        ev = parse_table_evidence(REPO_ROOT, TAGGED_MODEL)
        assert ev.dbt_domain == "growth"
        assert ev.dbt_sub_domain == "usage"


# ── filter_in_scope_paths with the new statuses param ─────────────────────────
class TestFilterStatuses:
    def test_includes_modified_when_requested(self):
        diff = [
            ("A", "dbt-databricks/models/mart/sales/added.sql"),
            ("M", "dbt-databricks/models/mart/sales/changed.sql"),
            ("D", "dbt-databricks/models/mart/sales/gone.sql"),
        ]
        out = filter_in_scope_paths(diff, statuses=("A", "M"))
        assert out == [
            "dbt-databricks/models/mart/sales/added.sql",
            "dbt-databricks/models/mart/sales/changed.sql",
        ]

    def test_default_is_added_only(self):
        diff = [("M", "dbt-databricks/models/mart/sales/changed.sql")]
        assert filter_in_scope_paths(diff) == []


# ── assess_tag_gap ────────────────────────────────────────────────────────────
class TestAssessTagGap:
    def _ev(self, domain="", sub_domain=""):
        return TableEvidence(
            table_name="rpt_thing",
            catalog="mart",
            schema="growth",
            path="x.sql",
            dbt_domain=domain,
            dbt_sub_domain=sub_domain,
        )

    def test_placeholder_needs_suggestion(self):
        assert assess_tag_gap(self._ev("tbd")).status == "needs_suggestion"
        assert assess_tag_gap(self._ev("unclassified")).status == "needs_suggestion"

    def test_missing_needs_suggestion(self):
        d = assess_tag_gap(self._ev(""))
        assert d.status == "needs_suggestion"
        assert d.current_domain == ""

    def test_real_domain_is_ok(self):
        d = assess_tag_gap(self._ev("growth", "usage"))
        assert d.status == "ok"
        assert d.current_domain == "growth"
        assert d.current_sub_domain == "usage"


# ── classify_tags_with_claude ─────────────────────────────────────────────────
class TestClassifyTagsWithClaude:
    @pytest.fixture
    def governed(self):
        return parse_governed_tags(GOVERNED_TAGS_YAML.read_text())

    def _decision(self):
        return TagDecision(
            evidence=TableEvidence(
                table_name="rpt_new_activation_signal",
                catalog="mart",
                schema="growth",
                path=PLACEHOLDER_MODEL,
                dbt_domain="tbd",
                description="activation signal",
                refs=["rpt_dashboard_organization"],
            ),
            status="needs_suggestion",
            current_domain="tbd",
        )

    def test_governed_value_is_suggested(self, governed):
        client = _make_mock_claude(
            {"domain": "growth", "sub_domain": "usage", "confidence": "high",
             "rationale": "Activation/adoption signal fits growth."}
        )
        out = classify_tags_with_claude(self._decision(), client=client, governed_tags=governed)
        assert out.status == "suggested"
        assert out.suggested_domain == "growth"
        assert out.suggested_sub_domain == "usage"
        assert out.new_value_proposals == []

    def test_ungoverned_value_needs_sign_off(self, governed):
        client = _make_mock_claude(
            {"domain": "product_rnd", "sub_domain": None, "confidence": "medium",
             "rationale": "Looks like a product signal."}
        )
        out = classify_tags_with_claude(self._decision(), client=client, governed_tags=governed)
        assert out.status == "needs_new_value"
        assert out.suggested_domain == "product_rnd"
        assert ("domain", "product_rnd") in out.new_value_proposals

    def test_ungoverned_sub_domain_flagged(self, governed):
        client = _make_mock_claude(
            {"domain": "growth", "sub_domain": "ask_ai", "confidence": "medium",
             "rationale": "growth + a new sub_domain."}
        )
        out = classify_tags_with_claude(self._decision(), client=client, governed_tags=governed)
        assert out.status == "needs_new_value"
        assert ("sub_domain", "ask_ai") in out.new_value_proposals
        assert ("domain", "growth") not in out.new_value_proposals  # domain is governed

    def test_null_sub_domain_string_treated_as_empty(self, governed):
        client = _make_mock_claude(
            {"domain": "sales", "sub_domain": "null", "confidence": "high", "rationale": "x"}
        )
        out = classify_tags_with_claude(self._decision(), client=client, governed_tags=governed)
        assert out.status == "suggested"
        assert out.suggested_sub_domain == ""

    def test_placeholder_pick_is_failure(self, governed):
        client = _make_mock_claude(
            {"domain": "tbd", "sub_domain": None, "confidence": "low", "rationale": "unsure"}
        )
        out = classify_tags_with_claude(self._decision(), client=client, governed_tags=governed)
        assert out.status == "llm_failed"

    def test_no_domain_is_failure(self, governed):
        client = _make_mock_claude({"domain": "", "confidence": "low", "rationale": "n/a"})
        out = classify_tags_with_claude(self._decision(), client=client, governed_tags=governed)
        assert out.status == "llm_failed"

    def test_api_exception_is_failure(self, governed):
        client = MagicMock()
        client.messages.create.side_effect = RuntimeError("API down")
        out = classify_tags_with_claude(self._decision(), client=client, governed_tags=governed)
        assert out.status == "llm_failed"
        assert "API down" in out.rationale

    def test_skips_non_needs_suggestion(self, governed):
        d = TagDecision(
            evidence=TableEvidence(
                table_name="x", catalog="mart", schema="growth", path="x.sql", dbt_domain="growth"
            ),
            status="ok",
            current_domain="growth",
        )
        client = MagicMock()
        out = classify_tags_with_claude(d, client=client, governed_tags=governed)
        assert out.status == "ok"
        client.messages.create.assert_not_called()


# ── propose_governed_tag_addition ─────────────────────────────────────────────
class TestProposeGovernedTagAddition:
    def test_renders_yaml_snippet(self):
        snippet = propose_governed_tag_addition("domain", "product_rnd")
        assert GOVERNED_TAGS_PATH in snippet
        assert "tag_key: domain" in snippet
        assert "- name: product_rnd" in snippet


# ── build_tag_comment ─────────────────────────────────────────────────────────
class TestBuildTagComment:
    def _suggested(self):
        return TagDecision(
            evidence=TableEvidence(
                table_name="rpt_new_activation_signal", catalog="mart", schema="growth", path="x.sql"
            ),
            status="suggested",
            current_domain="tbd",
            suggested_domain="growth",
            suggested_sub_domain="usage",
            confidence="high",
            rationale="Activation signal fits growth.",
        )

    def test_empty_decisions(self):
        body = build_tag_comment([])
        assert "jirade:tag-advisor:v1" in body
        assert "No new/changed models" in body
        assert "hash=" in body

    def test_renders_suggested_with_schema_yml_block(self):
        body = build_tag_comment([self._suggested()])
        assert "rpt_new_activation_signal" in body
        assert "mart.growth" in body
        assert "suggested" in body
        assert "databricks_tags:" in body
        assert "domain: growth" in body
        assert "sub_domain: usage" in body

    def test_new_value_shows_sign_off_section(self):
        d = TagDecision(
            evidence=TableEvidence(
                table_name="rpt_x", catalog="mart", schema="growth", path="x.sql"
            ),
            status="needs_new_value",
            current_domain="tbd",
            suggested_domain="product_rnd",
            confidence="medium",
            rationale="new domain",
            new_value_proposals=[("domain", "product_rnd")],
        )
        body = build_tag_comment([d])
        assert "requires governance sign-off" in body
        assert "- name: product_rnd" in body
        assert GOVERNED_TAGS_PATH in body

    def test_ok_models_summarized_in_footer(self):
        ok = TagDecision(
            evidence=TableEvidence(
                table_name="rpt_existing", catalog="mart", schema="growth", path="x.sql"
            ),
            status="ok",
            current_domain="growth",
        )
        body = build_tag_comment([ok])
        assert "already carry a governed" in body
        assert "rpt_existing" in body

    def test_llm_failed_shows_needs_review(self):
        d = TagDecision(
            evidence=TableEvidence(
                table_name="rpt_weird", catalog="mart", schema="growth", path="x.sql"
            ),
            status="llm_failed",
            rationale="Claude call failed: timeout",
        )
        body = build_tag_comment([d])
        assert "rpt_weird" in body
        assert "needs review" in body


# ── tag_comment_unchanged (idempotency) ───────────────────────────────────────
class TestTagCommentUnchanged:
    def test_same_hash_returns_true(self):
        body = build_tag_comment([])
        assert tag_comment_unchanged(body, body) is True

    def test_different_content_returns_false(self):
        a = build_tag_comment([])
        b = build_tag_comment(
            [
                TagDecision(
                    evidence=TableEvidence(
                        table_name="x", catalog="mart", schema="growth", path="x.sql"
                    ),
                    status="suggested",
                    suggested_domain="growth",
                )
            ]
        )
        assert tag_comment_unchanged(a, b) is False

    def test_missing_hash_returns_false(self):
        assert tag_comment_unchanged("random", "other") is False
