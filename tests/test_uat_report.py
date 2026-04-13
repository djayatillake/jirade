"""Tests for UAT report feature."""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from jirade.clients.databricks_client import DatabricksMetadataClient
from jirade.clients.jira_client import build_adf_table, build_adf_document
from jirade.mcp.handlers.uat_report import (
    _extract_ticket_key,
    _format_markdown_report,
    _format_adf_report,
    _format_value,
    generate_uat_report,
    UAT_REPORT_MARKER,
)


# =============================================================================
# Unit tests: ticket key extraction
# =============================================================================


class TestExtractTicketKey:
    def test_from_pr_title(self):
        assert _extract_ticket_key(
            "feat(dbt): add current_sign_in_at (AENG-1937)", "", "AENG"
        ) == "AENG-1937"

    def test_from_branch_name(self):
        assert _extract_ticket_key(
            "some title", "feat/aeng-1937-last-sign-in-at", "AENG"
        ) == "AENG-1937"

    def test_title_takes_precedence(self):
        key = _extract_ticket_key(
            "fix(dbt): thing (AENG-100)", "feat/aeng-200-other", "AENG"
        )
        assert key == "AENG-100"

    def test_no_match(self):
        assert _extract_ticket_key("random title", "random-branch", "AENG") is None

    def test_different_project_key(self):
        assert _extract_ticket_key(
            "feat: something (DATA-456)", "", "DATA"
        ) == "DATA-456"


# =============================================================================
# Unit tests: value formatting
# =============================================================================


class TestFormatValue:
    def test_none(self):
        assert _format_value(None) == "NULL"

    def test_int(self):
        assert _format_value(206406) == "206,406"

    def test_float_whole(self):
        assert _format_value(100.0) == "100"

    def test_float_decimal(self):
        assert _format_value(72.5) == "72.5"

    def test_string(self):
        assert _format_value("hello") == "hello"

    def test_zero(self):
        assert _format_value(0) == "0"


# =============================================================================
# Unit tests: ADF table builder
# =============================================================================


class TestBuildAdfTable:
    def test_basic_table(self):
        table = build_adf_table(["Metric", "Value"], [["Total", "100"]])

        assert table["type"] == "table"
        assert len(table["content"]) == 2  # 1 header row + 1 data row

        header_row = table["content"][0]
        assert header_row["content"][0]["type"] == "tableHeader"
        assert header_row["content"][0]["content"][0]["content"][0]["text"] == "Metric"

        data_row = table["content"][1]
        assert data_row["content"][0]["type"] == "tableCell"
        assert data_row["content"][0]["content"][0]["content"][0]["text"] == "Total"

    def test_empty_rows(self):
        table = build_adf_table(["A"], [])
        assert len(table["content"]) == 1  # header only

    def test_multiple_rows(self):
        table = build_adf_table(["A", "B"], [["1", "2"], ["3", "4"], ["5", "6"]])
        assert len(table["content"]) == 4  # 1 header + 3 data


class TestBuildAdfDocument:
    def test_heading_and_table(self):
        doc = build_adf_document([
            {"type": "heading", "level": 2, "text": "Report"},
            {"type": "table", "headers": ["A"], "rows": [["1"]]},
        ])
        assert doc["version"] == 1
        assert doc["type"] == "doc"
        assert doc["content"][0]["type"] == "heading"
        assert doc["content"][1]["type"] == "table"

    def test_paragraph_and_rule(self):
        doc = build_adf_document([
            {"type": "paragraph", "text": "Hello"},
            {"type": "rule"},
        ])
        assert doc["content"][0]["type"] == "paragraph"
        assert doc["content"][1]["type"] == "rule"


# =============================================================================
# Unit tests: markdown report formatting
# =============================================================================


class TestFormatMarkdownReport:
    def test_basic_report(self):
        results = [
            {
                "label": "Field Population",
                "columns": ["Metric", "Count", "%"],
                "rows": [
                    ["Total rows", 206406, 100.0],
                    ["Both populated", 192878, 93.4],
                ],
            }
        ]

        md = _format_markdown_report(
            "Comparing current_sign_in_at vs last_sign_in_at",
            results,
            pr_number=3964,
            jira_ticket_key="AENG-1937",
        )

        assert UAT_REPORT_MARKER in md
        assert "UAT Data Impact Report" in md
        assert "PR #3964" in md
        assert "AENG-1937" in md
        assert "Field Population" in md
        assert "206,406" in md
        assert "192,878" in md

    def test_error_result(self):
        results = [{"label": "Bad Query", "columns": [], "rows": [], "error": "SQL error"}]
        md = _format_markdown_report("test", results, pr_number=1, jira_ticket_key=None)
        assert "**Error:** SQL error" in md

    def test_empty_result(self):
        results = [{"label": "Empty", "columns": [], "rows": []}]
        md = _format_markdown_report("test", results, pr_number=1, jira_ticket_key=None)
        assert "_No results_" in md

    def test_no_jira_key(self):
        results = [{"label": "Q1", "columns": ["a"], "rows": [["1"]]}]
        md = _format_markdown_report("desc", results, pr_number=1, jira_ticket_key=None)
        assert "AENG" not in md
        assert "PR #1" in md


# =============================================================================
# Unit tests: ADF report formatting
# =============================================================================


class TestFormatAdfReport:
    def test_produces_valid_adf(self):
        results = [
            {
                "label": "Comparison",
                "columns": ["Metric", "Value"],
                "rows": [["Total", "100"]],
            }
        ]

        adf = _format_adf_report("test desc", results, pr_number=1, jira_ticket_key="TEST-1")

        assert adf["version"] == 1
        assert adf["type"] == "doc"
        types = [n["type"] for n in adf["content"]]
        assert "heading" in types
        assert "table" in types
        assert "rule" in types

    def test_error_result_in_adf(self):
        results = [{"label": "Bad", "columns": [], "rows": [], "error": "fail"}]
        adf = _format_adf_report("test", results, pr_number=1, jira_ticket_key=None)
        paragraphs = [n for n in adf["content"] if n["type"] == "paragraph"]
        texts = [n["content"][0]["text"] for n in paragraphs if n.get("content")]
        assert any("Error" in t for t in texts)


# =============================================================================
# Unit tests: analytical query security validation
# =============================================================================


class TestAnalyticalQuerySecurity:
    """Test DatabricksMetadataClient.execute_analytical_query security checks."""

    def _make_client(self):
        """Create a client instance without connecting."""
        client = DatabricksMetadataClient.__new__(DatabricksMetadataClient)
        client._connection = None
        return client

    def test_rejects_insert(self):
        client = self._make_client()
        with pytest.raises(ValueError, match="must start with SELECT or WITH"):
            client.execute_analytical_query(
                "INSERT INTO foo.bar.baz VALUES (1)",
                "foo.jirade_ci_123_",
            )

    def test_rejects_drop(self):
        client = self._make_client()
        with pytest.raises(ValueError, match="must start with SELECT or WITH"):
            client.execute_analytical_query(
                "DROP TABLE foo.jirade_ci_123_schema.table",
                "foo.jirade_ci_123_",
            )

    def test_rejects_select_with_delete(self):
        client = self._make_client()
        with pytest.raises(ValueError, match="DML/DDL"):
            client.execute_analytical_query(
                "SELECT 1; DELETE FROM foo.jirade_ci_123_schema.table",
                "foo.jirade_ci_123_",
            )

    def test_rejects_non_ci_table(self):
        client = self._make_client()
        with pytest.raises(ValueError, match="not allowed"):
            client.execute_analytical_query(
                "SELECT count(*) FROM production.schema.table",
                "dev.jirade_ci_123_",
            )

    def test_rejects_non_ci_table_even_with_other_prod_allowed(self):
        """A prod table not in the allowlist should still be rejected."""
        client = self._make_client()
        with pytest.raises(ValueError, match="not allowed"):
            client.execute_analytical_query(
                "SELECT count(*) FROM production.other.table",
                "dev.jirade_ci_123_",
                allowed_prod_tables=["production.schema.allowed_table"],
            )

    def test_rejects_no_table_reference(self):
        client = self._make_client()
        with pytest.raises(ValueError, match="must reference at least one"):
            client.execute_analytical_query(
                "SELECT 1 + 1",
                "dev.jirade_ci_123_",
            )

    def test_accepts_valid_ci_query(self):
        """Valid query should pass validation but fail on execution (no connection)."""
        client = self._make_client()
        with pytest.raises(AttributeError):
            # Passes validation, fails on _get_connection() since no real connection
            client.execute_analytical_query(
                "SELECT count(*) FROM dev.jirade_ci_123_staging.users",
                "dev.jirade_ci_123_",
            )

    def test_accepts_allowed_prod_table(self):
        """Query referencing an allowed production table should pass validation."""
        client = self._make_client()
        with pytest.raises(AttributeError):
            # Passes validation, fails on _get_connection()
            client.execute_analytical_query(
                "SELECT count(*) FROM reverse_etl.salesforce.contacts",
                "dev.jirade_ci_123_",
                allowed_prod_tables=["reverse_etl.salesforce.contacts"],
            )

    def test_accepts_mixed_ci_and_prod(self):
        """Query referencing both CI and allowed prod tables should pass."""
        client = self._make_client()
        with pytest.raises(AttributeError):
            client.execute_analytical_query(
                "SELECT a.x, b.y FROM dev.jirade_ci_123_staging.users a JOIN reverse_etl.salesforce.contacts b ON a.id = b.id",
                "dev.jirade_ci_123_",
                allowed_prod_tables=["reverse_etl.salesforce.contacts"],
            )

    def test_prod_table_check_is_case_insensitive(self):
        """Production table allowlist should be case-insensitive."""
        client = self._make_client()
        with pytest.raises(AttributeError):
            client.execute_analytical_query(
                "SELECT count(*) FROM Reverse_ETL.Salesforce.Contacts",
                "dev.jirade_ci_123_",
                allowed_prod_tables=["reverse_etl.salesforce.contacts"],
            )

    def test_accepts_cte_query(self):
        """WITH clause queries should be accepted."""
        client = self._make_client()
        with pytest.raises(AttributeError):
            client.execute_analytical_query(
                "WITH cte AS (SELECT * FROM dev.jirade_ci_123_s.t) SELECT count(*) FROM cte",
                "dev.jirade_ci_123_",
            )


# =============================================================================
# Integration test: full report generation flow (mocked)
# =============================================================================


class TestGenerateUatReport:
    @pytest.fixture
    def mock_settings(self):
        with patch("jirade.mcp.handlers.uat_report.get_settings") as mock:
            settings = MagicMock()
            settings.github_token = "test-token"
            settings.databricks_host = "test.databricks.com"
            settings.databricks_http_path = "/sql/test"
            settings.databricks_auth_type = "token"
            settings.databricks_token = "test-db-token"
            settings.databricks_catalog = "analytics"
            settings.databricks_ci_catalog = "dev_metadata"
            mock.return_value = settings
            yield settings

    @pytest.fixture
    def mock_gh_client(self):
        with patch("jirade.mcp.handlers.uat_report.GitHubClient") as mock_cls:
            client = AsyncMock()
            client.get_pull_request.return_value = {
                "title": "feat(dbt): add current_sign_in_at (AENG-1937)",
                "head": {"ref": "feat/aeng-1937-last-sign-in-at"},
            }
            client.upsert_pr_comment.return_value = {"id": 1}
            client.close = AsyncMock()
            mock_cls.return_value = client
            yield client

    @pytest.fixture
    def mock_db_client(self):
        with patch("jirade.mcp.handlers.uat_report.DatabricksMetadataClient") as mock_cls:
            client = MagicMock()
            client.__enter__ = MagicMock(return_value=client)
            client.__exit__ = MagicMock(return_value=False)

            # Return realistic query results
            client.execute_analytical_query.side_effect = [
                # Query 1: field population
                [
                    {
                        "total_rows": 206406,
                        "current_non_null": 192878,
                        "last_non_null": 192878,
                        "both_null": 13528,
                    }
                ],
                # Query 2: value comparison
                [
                    {
                        "both_equal": 53024,
                        "both_differ": 139854,
                        "current_after_last": 139854,
                    }
                ],
            ]
            mock_cls.return_value = client
            yield client

    @pytest.fixture
    def mock_jira(self):
        with patch("jirade.mcp.handlers.uat_report.AuthManager") as mock_auth_cls:
            auth = MagicMock()
            auth.jira.is_authenticated.return_value = True
            auth.jira.get_access_token.return_value = "test-jira-token"
            auth.jira.get_cloud_id.return_value = "test-cloud-id"
            mock_auth_cls.return_value = auth

            with patch("jirade.mcp.handlers.uat_report.JiraClient") as mock_jira_cls:
                jira_client = AsyncMock()
                jira_client.add_comment.return_value = {"id": "999"}
                jira_client.close = AsyncMock()
                mock_jira_cls.return_value = jira_client
                yield jira_client

    @pytest.mark.asyncio
    async def test_full_flow(self, mock_settings, mock_gh_client, mock_db_client, mock_jira):
        """Test the complete UAT report generation and posting flow."""
        result = await generate_uat_report(
            owner="algolia",
            repo="data",
            pr_number=3964,
            description="Comparing current_sign_in_at vs last_sign_in_at to show impact of the field correction",
            queries=[
                {
                    "label": "Field Population",
                    "sql": "SELECT count(*) as total_rows FROM dev_metadata.jirade_ci_3964_reverse_etl_salesforce.contacts",
                },
                {
                    "label": "Value Comparison",
                    "sql": "SELECT sum(case when current_sign_in_at = last_sign_in_at then 1 else 0 end) as both_equal FROM dev_metadata.jirade_ci_3964_reverse_etl_salesforce.contacts",
                },
            ],
            jira_ticket_key=None,  # should auto-detect AENG-1937
            github_token="test-token",
        )

        # Verify overall success
        assert result["success"] is True
        assert result["query_count"] == 2
        assert result["queries_succeeded"] == 2
        assert result["queries_failed"] == 0

        # Verify ticket key auto-detection
        assert result["jira_ticket_key"] == "AENG-1937"

        # Verify posted to both destinations
        assert result["posted_to_pr"] is True
        assert result["posted_to_jira"] is True

        # Verify GitHub PR comment was posted with correct marker
        mock_gh_client.upsert_pr_comment.assert_called_once()
        call_args = mock_gh_client.upsert_pr_comment.call_args
        assert call_args.kwargs["pr_number"] == 3964
        assert UAT_REPORT_MARKER in call_args.kwargs["body"]
        assert "206,406" in call_args.kwargs["body"]

        # Verify Jira comment was posted as ADF
        mock_jira.add_comment.assert_called_once()
        jira_call_args = mock_jira.add_comment.call_args
        assert jira_call_args.args[0] == "AENG-1937"
        adf_body = json.loads(jira_call_args.args[1])
        assert adf_body["version"] == 1
        assert adf_body["type"] == "doc"

        # Verify Databricks queries were executed with CI schema prefix
        assert mock_db_client.execute_analytical_query.call_count == 2
        for call in mock_db_client.execute_analytical_query.call_args_list:
            assert call.kwargs["ci_schema_prefix"] == "dev_metadata.jirade_ci_3964_"

    @pytest.mark.asyncio
    async def test_no_queries_returns_error(self, mock_settings, mock_gh_client):
        result = await generate_uat_report(
            owner="algolia",
            repo="data",
            pr_number=1,
            description="test",
            queries=[],
            jira_ticket_key="TEST-1",
            github_token="test-token",
        )
        assert result["success"] is False
        assert "No queries" in result["error"]

    @pytest.mark.asyncio
    async def test_query_failure_captured(self, mock_settings, mock_gh_client, mock_jira):
        """A failing query should be captured in results, not crash the report."""
        with patch("jirade.mcp.handlers.uat_report.DatabricksMetadataClient") as mock_cls:
            client = MagicMock()
            client.__enter__ = MagicMock(return_value=client)
            client.__exit__ = MagicMock(return_value=False)
            client.execute_analytical_query.side_effect = ValueError("bad query")
            mock_cls.return_value = client

            result = await generate_uat_report(
                owner="algolia",
                repo="data",
                pr_number=1,
                description="test",
                queries=[{"label": "Bad Query", "sql": "SELECT bad"}],
                jira_ticket_key="TEST-1",
                github_token="test-token",
            )

            assert result["success"] is True
            assert result["queries_failed"] == 1
            assert result["query_results"][0]["error"] == "bad query"

    @pytest.mark.asyncio
    async def test_explicit_ticket_key_skips_detection(self, mock_settings, mock_gh_client, mock_db_client, mock_jira):
        """When jira_ticket_key is provided, don't call get_pull_request."""
        result = await generate_uat_report(
            owner="algolia",
            repo="data",
            pr_number=1,
            description="test",
            queries=[{"label": "Q", "sql": "SELECT 1 FROM dev_metadata.jirade_ci_1_s.t"}],
            jira_ticket_key="AENG-999",
            github_token="test-token",
        )

        assert result["jira_ticket_key"] == "AENG-999"
        mock_gh_client.get_pull_request.assert_not_called()
