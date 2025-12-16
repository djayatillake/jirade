"""Pytest configuration and fixtures."""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from jira_agent.config import AgentSettings
from jira_agent.repo_config.schema import (
    RepoConfig,
    RepoIdentification,
    JiraConfig,
    BranchingConfig,
    PullRequestConfig,
    CommitConfig,
    SkipConfig,
    DbtConfig,
    DatabricksConfig,
    CIConfig,
)


@pytest.fixture
def mock_settings() -> AgentSettings:
    """Create mock agent settings."""
    return AgentSettings(
        anthropic_api_key="test-api-key",
        claude_model="claude-opus-4-5-20251101",
        jira_oauth_client_id="test-client-id",
        jira_oauth_client_secret="test-client-secret",
        github_token="test-github-token",
        workspace_dir=Path("/tmp/test-workspace"),
    )


@pytest.fixture
def sample_repo_config() -> RepoConfig:
    """Create sample repository configuration."""
    return RepoConfig(
        repo=RepoIdentification(
            owner="test-org",
            name="test-repo",
            default_branch="main",
            pr_target_branch="develop",
        ),
        jira=JiraConfig(
            base_url="https://test.atlassian.net",
            project_key="TEST",
            board_id=123,
        ),
        branching=BranchingConfig(),
        pull_request=PullRequestConfig(),
        commits=CommitConfig(),
        skip=SkipConfig(),
        dbt=DbtConfig(enabled=False),
        databricks=DatabricksConfig(enabled=False),
        ci=CIConfig(),
    )


@pytest.fixture
def mock_jira_client() -> AsyncMock:
    """Create mock Jira client."""
    client = AsyncMock()

    # Mock get_issue
    client.get_issue.return_value = {
        "key": "TEST-123",
        "fields": {
            "summary": "Add new column to model",
            "description": {
                "type": "doc",
                "content": [
                    {
                        "type": "paragraph",
                        "content": [{"type": "text", "text": "Please add a new column."}],
                    }
                ],
            },
            "status": {"name": "To Do"},
            "issuetype": {"name": "Task"},
            "priority": {"name": "Medium"},
            "labels": [],
            "assignee": None,
        },
    }

    # Mock get_issue_comments
    client.get_issue_comments.return_value = []

    # Mock add_comment
    client.add_comment.return_value = {"id": "12345"}

    return client


@pytest.fixture
def mock_github_client() -> AsyncMock:
    """Create mock GitHub client."""
    client = AsyncMock()

    # Mock create_pull_request
    client.create_pull_request.return_value = {
        "number": 456,
        "html_url": "https://github.com/test-org/test-repo/pull/456",
        "state": "open",
        "head": {"sha": "abc123", "ref": "feat/TEST-123-new-column"},
        "base": {"ref": "develop"},
    }

    # Mock get_pull_request
    client.get_pull_request.return_value = {
        "number": 456,
        "state": "open",
        "mergeable": True,
        "mergeable_state": "clean",
        "draft": False,
        "html_url": "https://github.com/test-org/test-repo/pull/456",
        "head": {"sha": "abc123", "ref": "feat/TEST-123-new-column"},
    }

    # Mock get_check_runs
    client.get_check_runs.return_value = [
        {
            "id": 1,
            "name": "pre-commit",
            "status": "completed",
            "conclusion": "success",
        },
    ]

    return client


@pytest.fixture
def temp_repo(tmp_path: Path) -> Path:
    """Create a temporary git repository for testing."""
    import subprocess

    repo_path = tmp_path / "test-repo"
    repo_path.mkdir()

    # Initialize git repo
    subprocess.run(["git", "init"], cwd=repo_path, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=repo_path,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=repo_path,
        capture_output=True,
    )

    # Create initial commit
    readme = repo_path / "README.md"
    readme.write_text("# Test Repo\n")
    subprocess.run(["git", "add", "."], cwd=repo_path, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"],
        cwd=repo_path,
        capture_output=True,
    )

    return repo_path


@pytest.fixture
def sample_manifest() -> dict:
    """Create sample dbt manifest.json data."""
    return {
        "nodes": {
            "model.test.staging_orders": {
                "resource_type": "model",
                "name": "staging_orders",
                "original_file_path": "models/staging/staging_orders.sql",
                "schema": "staging",
                "database": "analytics",
                "config": {"materialized": "view"},
                "depends_on": {"nodes": ["source.test.raw_orders"]},
                "columns": {
                    "order_id": {"name": "order_id", "description": "Primary key"},
                    "customer_id": {"name": "customer_id", "description": "FK to customers"},
                },
            },
            "model.test.mart_orders": {
                "resource_type": "model",
                "name": "mart_orders",
                "original_file_path": "models/mart/mart_orders.sql",
                "schema": "mart",
                "database": "analytics",
                "config": {"materialized": "table"},
                "depends_on": {"nodes": ["model.test.staging_orders"]},
                "columns": {},
            },
        },
        "sources": {
            "source.test.raw_orders": {
                "name": "raw_orders",
                "source_name": "raw",
                "schema": "raw",
                "database": "raw_db",
            },
        },
        "child_map": {
            "model.test.staging_orders": ["model.test.mart_orders"],
        },
    }
