"""Tests for Git tools."""

import pytest
from pathlib import Path

from jira_agent.tools.git_tools import (
    GitTools,
    sanitize_branch_name,
    format_branch_name,
)


class TestSanitizeBranchName:
    """Tests for branch name sanitization."""

    def test_lowercase_conversion(self):
        """Test that names are converted to lowercase."""
        assert sanitize_branch_name("AddNewFeature") == "addnewfeature"

    def test_space_replacement(self):
        """Test that spaces are replaced with hyphens."""
        assert sanitize_branch_name("add new feature") == "add-new-feature"

    def test_special_char_replacement(self):
        """Test that special characters are replaced."""
        assert sanitize_branch_name("add_new@feature!") == "add-new-feature-"

    def test_consecutive_hyphens(self):
        """Test that consecutive hyphens are collapsed."""
        assert sanitize_branch_name("add---new---feature") == "add-new-feature"

    def test_leading_trailing_hyphens(self):
        """Test that leading/trailing hyphens are removed."""
        assert sanitize_branch_name("-add-feature-") == "add-feature"

    def test_length_limit(self):
        """Test that names are limited to 50 characters."""
        long_name = "a" * 100
        result = sanitize_branch_name(long_name)
        assert len(result) <= 50


class TestFormatBranchName:
    """Tests for branch name formatting."""

    def test_default_pattern(self):
        """Test formatting with default pattern."""
        result = format_branch_name(
            pattern="{type}/{ticket_key}-{description}",
            ticket_key="TEST-123",
            description="Add new column",
            branch_type="feat",
        )
        assert result == "feat/TEST-123-add-new-column"

    def test_fix_branch_type(self):
        """Test with fix branch type."""
        result = format_branch_name(
            pattern="{type}/{ticket_key}-{description}",
            ticket_key="TEST-456",
            description="Fix bug",
            branch_type="fix",
        )
        assert result == "fix/TEST-456-fix-bug"


class TestGitTools:
    """Tests for GitTools class."""

    def test_init(self, tmp_path: Path):
        """Test GitTools initialization."""
        tools = GitTools(tmp_path)
        assert tools.workspace_dir == tmp_path
        assert tools._repo is None

    def test_set_repo_path(self, temp_repo: Path):
        """Test setting repository path."""
        tools = GitTools(temp_repo.parent)
        tools.set_repo_path(temp_repo)

        assert tools.repo_path == temp_repo
        assert tools.repo is not None

    def test_get_current_branch(self, temp_repo: Path):
        """Test getting current branch name."""
        tools = GitTools(temp_repo.parent)
        tools.set_repo_path(temp_repo)

        # Default branch after git init
        branch = tools.get_current_branch()
        assert branch in ("main", "master")

    def test_create_branch(self, temp_repo: Path):
        """Test creating a new branch."""
        tools = GitTools(temp_repo.parent)
        tools.set_repo_path(temp_repo)

        tools.checkout_branch("test-branch", create=True)
        assert tools.get_current_branch() == "test-branch"

    def test_has_changes_false(self, temp_repo: Path):
        """Test has_changes returns False when clean."""
        tools = GitTools(temp_repo.parent)
        tools.set_repo_path(temp_repo)

        assert tools.has_changes() is False

    def test_has_changes_true(self, temp_repo: Path):
        """Test has_changes returns True when dirty."""
        tools = GitTools(temp_repo.parent)
        tools.set_repo_path(temp_repo)

        # Create a new file
        (temp_repo / "new_file.txt").write_text("test content")

        assert tools.has_changes() is True

    def test_get_diff_files(self, temp_repo: Path):
        """Test getting list of changed files."""
        tools = GitTools(temp_repo.parent)
        tools.set_repo_path(temp_repo)

        # Create new files
        (temp_repo / "file1.txt").write_text("content1")
        (temp_repo / "file2.txt").write_text("content2")

        diff_files = tools.get_diff_files()
        assert "file1.txt" in diff_files
        assert "file2.txt" in diff_files

    def test_stage_and_commit(self, temp_repo: Path):
        """Test staging and committing files."""
        tools = GitTools(temp_repo.parent)
        tools.set_repo_path(temp_repo)

        # Create a file
        (temp_repo / "test.txt").write_text("test")

        # Stage and commit
        tools.stage_files()
        sha = tools.commit("test: add test file")

        assert sha is not None
        assert len(sha) == 40  # Full SHA
        assert tools.has_changes() is False

    def test_run_command(self, temp_repo: Path):
        """Test running shell commands."""
        tools = GitTools(temp_repo.parent)
        tools.set_repo_path(temp_repo)

        code, stdout, stderr = tools.run_command(["git", "status"])

        assert code == 0
        assert "On branch" in stdout

    def test_repo_not_set_error(self, tmp_path: Path):
        """Test error when repo not set."""
        tools = GitTools(tmp_path)

        with pytest.raises(ValueError, match="No repository set"):
            _ = tools.repo

        with pytest.raises(ValueError, match="No repository set"):
            _ = tools.repo_path
