"""Tests for repository configuration."""

import pytest
from pathlib import Path

from jira_agent.repo_config.schema import RepoConfig, RepoIdentification, JiraConfig
from jira_agent.repo_config.loader import ConfigLoader


class TestRepoConfig:
    """Tests for RepoConfig schema."""

    def test_full_repo_name(self, sample_repo_config: RepoConfig):
        """Test full_repo_name property."""
        assert sample_repo_config.full_repo_name == "test-org/test-repo"

    def test_minimal_config(self):
        """Test creating config with minimal required fields."""
        config = RepoConfig(
            repo=RepoIdentification(owner="org", name="repo"),
            jira=JiraConfig(project_key="PROJ"),
        )

        assert config.repo.owner == "org"
        assert config.repo.name == "repo"
        assert config.repo.default_branch == "main"  # Default
        assert config.jira.project_key == "PROJ"
        assert config.dbt.enabled is False  # Default

    def test_branching_pattern(self, sample_repo_config: RepoConfig):
        """Test default branching pattern."""
        pattern = sample_repo_config.branching.pattern
        assert "{type}" in pattern
        assert "{ticket_key}" in pattern


class TestConfigLoader:
    """Tests for ConfigLoader."""

    def test_load_from_file(self, tmp_path: Path):
        """Test loading config from YAML file."""
        config_file = tmp_path / "test-config.yaml"
        config_file.write_text("""
repo:
  owner: test-org
  name: test-repo

jira:
  project_key: TEST
  board_id: 123
""")

        loader = ConfigLoader(tmp_path)
        config = loader.load_from_file(config_file)

        assert config.repo.owner == "test-org"
        assert config.repo.name == "test-repo"
        assert config.jira.project_key == "TEST"
        assert config.jira.board_id == 123

    def test_load_for_repo(self, tmp_path: Path):
        """Test loading config by repo name."""
        config_file = tmp_path / "test-org-test-repo.yaml"
        config_file.write_text("""
repo:
  owner: test-org
  name: test-repo

jira:
  project_key: TEST
""")

        loader = ConfigLoader(tmp_path)
        config = loader.load_for_repo("test-org/test-repo")

        assert config.repo.owner == "test-org"

    def test_load_for_repo_not_found(self, tmp_path: Path):
        """Test error when config not found."""
        loader = ConfigLoader(tmp_path)

        with pytest.raises(FileNotFoundError):
            loader.load_for_repo("nonexistent/repo")

    def test_list_configs(self, tmp_path: Path):
        """Test listing available configs."""
        (tmp_path / "org1-repo1.yaml").write_text("repo:\n  owner: org1\n  name: repo1\njira:\n  project_key: P1")
        (tmp_path / "org2-repo2.yaml").write_text("repo:\n  owner: org2\n  name: repo2\njira:\n  project_key: P2")
        (tmp_path / "example.yaml").write_text("# Example")  # Should be excluded

        loader = ConfigLoader(tmp_path)
        configs = loader.list_configs()

        assert len(configs) == 2
        assert "org1/repo1" in configs
        assert "org2/repo2" in configs

    def test_config_caching(self, tmp_path: Path):
        """Test that configs are cached."""
        config_file = tmp_path / "cached-org-cached-repo.yaml"
        config_file.write_text("""
repo:
  owner: cached-org
  name: cached-repo

jira:
  project_key: CACHE
""")

        loader = ConfigLoader(tmp_path)

        # Load twice
        config1 = loader.load_for_repo("cached-org/cached-repo")
        config2 = loader.load_for_repo("cached-org/cached-repo")

        # Should be same object (cached)
        assert config1 is config2

        # Clear cache
        loader.clear_cache()
        config3 = loader.load_for_repo("cached-org/cached-repo")

        # Should be different object after cache clear
        assert config1 is not config3
