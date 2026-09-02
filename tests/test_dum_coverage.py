"""Tests for the DUM division-access coverage check (jirade/tools/dum_coverage.py)."""

from jirade.tools.dum_coverage import (
    NewTableCoverage,
    build_division_grants,
    check_new_table_coverage,
    parent_relations_from_manifest,
    relation_for_model,
    render_dum_coverage_section,
)
from jirade.tools.dum_editor import load_dum

DUM_TEXT = """\
# Broad shared group — must never count as division access
group-hex-advanced-users:
  groups:
    - "Application - Hex Legacy"
  catalogs:
    - mart: read
    - analytics: read

group-division-ai:
  groups:
    - "Okta Push - Division - AI"
  tables:
    - analytics.dimensional.fact_ab_test: read
    - mart.product.mv_ab_tests: read

group-division-commercial:
  groups:
    - "Okta Push - Division - Commercial"
  schemas:
    - mart.sales: read
  tables:
    - analytics.dimensional.fact_opportunity: read

group-division-finance:
  groups:
    - "Okta Push - Division - Finance"
  tables:
    - analytics.dimensional.fact_opportunity: read
    - source.revenue.entries: read

# Personal top-up block — ignored (no Okta division label)
group-division-finance-user1:
  users:
    - someone@algolia.com
  tables:
    - mart.finance.rpt_secret: read
"""


def _dum():
    return load_dum(DUM_TEXT)


# ── relation_for_model ───────────────────────────────────────────────────────
class TestRelationForModel:
    def test_three_part_name(self):
        assert relation_for_model("mart__sales__rpt_opportunity") == "mart.sales.rpt_opportunity"

    def test_multi_part_table_name(self):
        assert (
            relation_for_model("analytics__dimensional__fact__weird")
            == "analytics.dimensional.fact_weird"
        )

    def test_two_part_name_maps_schema_to_catalog(self):
        assert relation_for_model("mart__thing") == "mart.mart.thing"

    def test_unparseable_returns_none(self):
        assert relation_for_model("plainname") is None


# ── build_division_grants ────────────────────────────────────────────────────
class TestBuildDivisionGrants:
    def test_indexes_only_real_division_blocks(self):
        grants = build_division_grants(_dum())
        assert [g.division for g in grants] == ["AI", "Commercial", "Finance"]

    def test_covers_table_schema_and_catalog_grants(self):
        grants = {g.division: g for g in build_division_grants(_dum())}
        assert grants["AI"].covers("mart.product.mv_ab_tests")
        assert grants["Commercial"].covers("mart.sales.anything_new")  # schema grant
        assert not grants["Finance"].covers("mart.sales.anything_new")

    def test_personal_topup_block_not_division_access(self):
        grants = {g.division: g for g in build_division_grants(_dum())}
        assert not grants["Finance"].covers("mart.finance.rpt_secret")


# ── check_new_table_coverage ─────────────────────────────────────────────────
class TestCheckNewTableCoverage:
    def test_covered_by_schema_grant(self):
        out = check_new_table_coverage({"mart__sales__rpt_new_bookings": []}, _dum())
        assert len(out) == 1
        assert out[0].granted_divisions == ["Commercial"]
        assert out[0].has_division_access

    def test_uncovered_table_suggests_parent_readers(self):
        out = check_new_table_coverage(
            {
                "mart__finance__rpt_arr_bridge": [
                    "analytics.dimensional.fact_opportunity",
                    "source.revenue.entries",
                ]
            },
            _dum(),
        )
        c = out[0]
        assert not c.has_division_access
        # Commercial + Finance both read fact_opportunity; Finance also reads entries.
        assert sorted(c.suggested_divisions) == ["Commercial", "Finance"]
        assert c.suggested_divisions["Finance"] == [
            "analytics.dimensional.fact_opportunity",
            "source.revenue.entries",
        ]

    def test_granted_division_not_also_suggested(self):
        out = check_new_table_coverage(
            {"mart__sales__rpt_x": ["analytics.dimensional.fact_opportunity"]}, _dum()
        )
        c = out[0]
        assert c.granted_divisions == ["Commercial"]
        assert sorted(c.suggested_divisions) == ["Finance"]

    def test_out_of_scope_catalogs_skipped(self):
        out = check_new_table_coverage(
            {"staging__salesforce__stg_new": [], "metadata__quality__rpt_x": []}, _dum()
        )
        assert out == []

    def test_broad_shared_group_does_not_count(self):
        # group-hex-advanced-users has mart: read, but it is not a division.
        out = check_new_table_coverage({"mart__growth__rpt_orphan": []}, _dum())
        assert out[0].granted_divisions == []


# ── parent_relations_from_manifest ───────────────────────────────────────────
class TestParentRelationsFromManifest:
    MANIFEST = {
        "nodes": {
            "model.algolia.mart__sales__rpt_new": {
                "resource_type": "model",
                "name": "mart__sales__rpt_new",
                "depends_on": {
                    "nodes": [
                        "model.algolia.analytics__dimensional__fact_opportunity",
                        "source.algolia.revenue.entries",
                        "seed.algolia.some_seed",
                    ]
                },
            },
            "model.algolia.analytics__dimensional__fact_opportunity": {
                "resource_type": "model",
                "name": "analytics__dimensional__fact_opportunity",
            },
        },
        "sources": {
            "source.algolia.revenue.entries": {
                "database": "source",
                "schema": "revenue",
                "identifier": "entries",
            },
        },
    }

    def test_resolves_model_and_source_parents(self):
        rels = parent_relations_from_manifest(self.MANIFEST, "mart__sales__rpt_new")
        assert rels == [
            "analytics.dimensional.fact_opportunity",
            "source.revenue.entries",
        ]

    def test_unknown_model_returns_empty(self):
        assert parent_relations_from_manifest(self.MANIFEST, "nope") == []


# ── render_dum_coverage_section ──────────────────────────────────────────────
class TestRenderSection:
    def test_empty_input_renders_nothing(self):
        assert render_dum_coverage_section([]) == ""

    def test_missing_table_produces_warning(self):
        out = check_new_table_coverage(
            {"mart__finance__rpt_arr_bridge": ["analytics.dimensional.fact_opportunity"]},
            _dum(),
        )
        section = render_dum_coverage_section(out)
        assert "Division access check" in section
        assert "`mart.finance.rpt_arr_bridge`" in section
        assert "❌ none" in section
        assert "no `group-division-*` grant" in section
        assert "`Commercial`" in section and "`Finance`" in section

    def test_all_covered_renders_green(self):
        out = check_new_table_coverage({"mart__sales__rpt_new": []}, _dum())
        section = render_dum_coverage_section(out)
        assert "✅ All new tables have at least one division grant" in section
        assert "❌" not in section

    def test_suggestion_overflow_elided(self):
        c = NewTableCoverage(
            model="mart__x__y",
            relation="mart.x.y",
            suggested_divisions={f"Div{i:02d}": ["mart.a.b"] for i in range(12)},
        )
        section = render_dum_coverage_section([c])
        assert "+4 more" in section
