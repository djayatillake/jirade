"""Pydantic schemas for repository configuration."""

from typing import Literal

from pydantic import BaseModel, Field


class RepoIdentification(BaseModel):
    """Repository identification settings."""

    owner: str = Field(..., description="GitHub repository owner/organization")
    name: str = Field(..., description="GitHub repository name")
    default_branch: str = Field("main", description="Branch to create feature branches from")
    pr_target_branch: str = Field("main", description="Branch PRs should target")


class JiraConfig(BaseModel):
    """Jira project configuration."""

    base_url: str = Field("", description="Jira instance URL (e.g., https://your-org.atlassian.net)")
    project_key: str = Field(..., description="Jira project key (e.g., AENG)")
    board_id: int | None = Field(None, description="Jira board ID for ticket fetching")


class BranchingTypes(BaseModel):
    """Branch type prefixes."""

    feature: str = Field("feat", description="Prefix for feature branches")
    bugfix: str = Field("fix", description="Prefix for bugfix branches")
    refactor: str = Field("refactor", description="Prefix for refactor branches")


class BranchingConfig(BaseModel):
    """Branch naming configuration."""

    pattern: str = Field(
        "{type}/{ticket_key}-{description}",
        description="Branch name pattern. Available vars: {type}, {ticket_key}, {description}",
    )
    types: BranchingTypes = Field(default_factory=BranchingTypes)


class PullRequestConfig(BaseModel):
    """Pull request configuration."""

    title_pattern: str = Field(
        "{type}({scope}): {description} ({ticket_key})",
        description="PR title pattern. Available vars: {type}, {scope}, {description}, {ticket_key}",
    )
    template_path: str | None = Field(
        ".github/PULL_REQUEST_TEMPLATE.md",
        description="Path to PR template in target repo",
    )
    contributing_path: str | None = Field(
        ".github/CONTRIBUTING.md",
        description="Path to contributing guide in target repo",
    )


class CommitConfig(BaseModel):
    """Commit message configuration."""

    style: Literal["conventional", "angular", "custom"] = Field(
        "conventional",
        description="Commit message style",
    )
    scope_required: bool = Field(False, description="Whether scope is required in commit messages")
    ticket_in_message: bool = Field(True, description="Whether to include ticket key in commit message")


class SkipConfig(BaseModel):
    """Skip conditions configuration."""

    comment_phrase: str = Field("[AGENT-SKIP]", description="Phrase in comments that signals agent to skip ticket")
    labels: list[str] = Field(
        default_factory=lambda: ["no-automation", "manual-only"],
        description="Labels that prevent agent from working ticket",
    )


class DbtProject(BaseModel):
    """Configuration for a single dbt project."""

    path: str = Field(..., description="Path to dbt project relative to repo root")
    manifest_path: str = Field("target/manifest.json", description="Path to manifest.json relative to dbt project")
    profile: str | None = Field(None, description="dbt profile name")


class DbtConfig(BaseModel):
    """dbt configuration."""

    enabled: bool = Field(False, description="Whether dbt tools are enabled")
    projects: list[DbtProject] = Field(default_factory=list, description="List of dbt projects in repo")


class DatabricksConfig(BaseModel):
    """Databricks configuration."""

    enabled: bool = Field(False, description="Whether Databricks tools are enabled")


class CIConfig(BaseModel):
    """CI configuration."""

    system: Literal["circleci", "github_actions", "jenkins", "other"] = Field(
        "github_actions",
        description="CI system used by the repo",
    )
    auto_fix: list[str] = Field(
        default_factory=lambda: ["pre-commit"],
        description="Auto-fix strategies to apply on CI failure",
    )


class AgentTriggerConfig(BaseModel):
    """Agent trigger configuration."""

    status: str = Field(
        "Ready for Agent",
        description="Jira status that triggers the agent to work on a ticket",
    )
    done_status: str = Field(
        "Done",
        description="Jira status to transition to after PR is merged",
    )
    in_progress_status: str = Field(
        "In Progress",
        description="Jira status to set while agent is working on ticket",
    )


class LearningConfig(BaseModel):
    """Learning capture configuration for this repository."""

    enabled: bool = Field(
        True,
        description="Enable learning capture for this repository",
    )
    categories: list[str] = Field(
        default_factory=lambda: ["ci-failure", "code-pattern", "error-resolution"],
        description="Categories of learnings to capture",
    )


class RepoConfig(BaseModel):
    """Complete repository configuration."""

    repo: RepoIdentification
    jira: JiraConfig
    branching: BranchingConfig = Field(default_factory=BranchingConfig)
    pull_request: PullRequestConfig = Field(default_factory=PullRequestConfig)
    commits: CommitConfig = Field(default_factory=CommitConfig)
    skip: SkipConfig = Field(default_factory=SkipConfig)
    dbt: DbtConfig = Field(default_factory=DbtConfig)
    databricks: DatabricksConfig = Field(default_factory=DatabricksConfig)
    ci: CIConfig = Field(default_factory=CIConfig)
    agent: AgentTriggerConfig = Field(default_factory=AgentTriggerConfig)
    learning: LearningConfig = Field(default_factory=LearningConfig)

    @property
    def full_repo_name(self) -> str:
        """Get full repository name (owner/name)."""
        return f"{self.repo.owner}/{self.repo.name}"
