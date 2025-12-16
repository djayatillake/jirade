"""Tests for dbt tools."""

import json
import pytest
from pathlib import Path

from jira_agent.tools.dbt_tools import DbtTools, parse_dbt_error


class TestDbtTools:
    """Tests for DbtTools class."""

    @pytest.fixture
    def dbt_project(self, tmp_path: Path, sample_manifest: dict) -> tuple[Path, dict]:
        """Create a mock dbt project structure."""
        project_path = tmp_path / "dbt-project"
        project_path.mkdir()

        # Create target directory with manifest
        target_path = project_path / "target"
        target_path.mkdir()
        manifest_path = target_path / "manifest.json"
        manifest_path.write_text(json.dumps(sample_manifest))

        # Create model files
        models_path = project_path / "models"
        staging_path = models_path / "staging"
        mart_path = models_path / "mart"
        staging_path.mkdir(parents=True)
        mart_path.mkdir(parents=True)

        (staging_path / "staging_orders.sql").write_text(
            "SELECT * FROM {{ source('raw', 'orders') }}"
        )
        (mart_path / "mart_orders.sql").write_text(
            "SELECT * FROM {{ ref('staging_orders') }}"
        )

        dbt_projects = [
            {"path": "dbt-project", "manifest_path": "target/manifest.json"}
        ]

        return tmp_path, dbt_projects

    def test_load_manifest(self, dbt_project: tuple, sample_manifest: dict):
        """Test loading dbt manifest."""
        repo_path, dbt_projects = dbt_project
        tools = DbtTools(repo_path, dbt_projects)

        manifest = tools.load_manifest("dbt-project")

        assert "nodes" in manifest
        assert "model.test.staging_orders" in manifest["nodes"]

    def test_find_models(self, dbt_project: tuple):
        """Test finding models by pattern."""
        repo_path, dbt_projects = dbt_project
        tools = DbtTools(repo_path, dbt_projects)

        # Find by partial name
        results = tools.find_models("orders")
        assert len(results) == 2

        # Find by exact pattern
        results = tools.find_models("staging_orders")
        assert len(results) == 1
        assert results[0]["name"] == "staging_orders"

    def test_find_models_with_wildcard(self, dbt_project: tuple):
        """Test finding models with wildcard pattern."""
        repo_path, dbt_projects = dbt_project
        tools = DbtTools(repo_path, dbt_projects)

        results = tools.find_models("staging_*")
        assert len(results) == 1
        assert results[0]["name"] == "staging_orders"

    def test_get_model_dependencies(self, dbt_project: tuple):
        """Test getting model dependencies."""
        repo_path, dbt_projects = dbt_project
        tools = DbtTools(repo_path, dbt_projects)

        deps = tools.get_model_dependencies("staging_orders", "dbt-project")

        assert "upstream" in deps
        assert "downstream" in deps
        assert "source.test.raw_orders" in deps["upstream"]
        assert "model.test.mart_orders" in deps["downstream"]

    def test_get_model_columns(self, dbt_project: tuple):
        """Test getting model column information."""
        repo_path, dbt_projects = dbt_project
        tools = DbtTools(repo_path, dbt_projects)

        columns = tools.get_model_columns("staging_orders", "dbt-project")

        assert len(columns) == 2
        column_names = [c["name"] for c in columns]
        assert "order_id" in column_names
        assert "customer_id" in column_names

    def test_get_model_sql(self, dbt_project: tuple):
        """Test getting model SQL content."""
        repo_path, dbt_projects = dbt_project
        tools = DbtTools(repo_path, dbt_projects)

        sql = tools.get_model_sql("staging_orders", "dbt-project")

        assert sql is not None
        assert "source('raw', 'orders')" in sql

    def test_get_sources(self, dbt_project: tuple):
        """Test getting source definitions."""
        repo_path, dbt_projects = dbt_project
        tools = DbtTools(repo_path, dbt_projects)

        sources = tools.get_sources("dbt-project")

        assert len(sources) == 1
        assert sources[0]["name"] == "raw_orders"

    def test_find_model_file(self, dbt_project: tuple):
        """Test finding model file path."""
        repo_path, dbt_projects = dbt_project
        tools = DbtTools(repo_path, dbt_projects)

        file_path = tools.find_model_file("staging_orders", "dbt-project")

        assert file_path is not None
        assert file_path.name == "staging_orders.sql"
        assert file_path.exists()

    def test_model_not_found(self, dbt_project: tuple):
        """Test error when model not found."""
        repo_path, dbt_projects = dbt_project
        tools = DbtTools(repo_path, dbt_projects)

        with pytest.raises(ValueError, match="Model not found"):
            tools.get_model_dependencies("nonexistent_model", "dbt-project")


class TestParseDbtError:
    """Tests for dbt error parsing."""

    def test_parse_compilation_error(self):
        """Test parsing compilation error."""
        error_output = """
Compilation Error in model staging_orders (models/staging/staging_orders.sql)
  'ref' is undefined. This can happen when you have a typo.
  line 5
"""
        result = parse_dbt_error(error_output)

        assert result["type"] == "compilation"
        assert result["model"] == "staging_orders"
        assert result["line"] == 5

    def test_parse_unknown_error(self):
        """Test parsing unknown error format."""
        error_output = "Some unknown error occurred"
        result = parse_dbt_error(error_output)

        assert result["type"] == "unknown"
        assert result["message"] == error_output
