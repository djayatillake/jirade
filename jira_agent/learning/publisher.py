"""Publish learnings to jira-agent repo via PR."""

import logging
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from ..tools.git_tools import GitTools
from .models import Learning, LearningCategory
from .storage import LearningStorage

logger = logging.getLogger(__name__)

# Knowledge base structure in jira-agent repo
KNOWLEDGE_BASE_DIR = "knowledge"
CATEGORY_DIRS = {
    LearningCategory.CI_FAILURE: "ci-failures",
    LearningCategory.CODE_PATTERN: "code-patterns",
    LearningCategory.ERROR_RESOLUTION: "error-resolutions",
}


class LearningPublisher:
    """Publishes accumulated learnings to jira-agent repo via PR."""

    def __init__(
        self,
        github_token: str,
        jira_agent_repo: str = "djayatillake/jira-agent",
        workspace_dir: Path | None = None,
    ):
        """Initialize publisher.

        Args:
            github_token: GitHub token for authentication.
            jira_agent_repo: Repository for jira-agent (owner/name).
            workspace_dir: Directory where repos are cloned.
        """
        self.github_token = github_token
        self.jira_agent_repo = jira_agent_repo
        self.workspace_dir = workspace_dir or Path("/tmp/jira-agent")

        # Parse owner/name
        if "/" in jira_agent_repo:
            self.repo_owner, self.repo_name = jira_agent_repo.split("/", 1)
        else:
            self.repo_owner = "djayatillake"
            self.repo_name = jira_agent_repo

    def collect_learnings(self, target_repos: list[Path] | None = None) -> list[Learning]:
        """Collect learnings from target repos.

        Args:
            target_repos: List of repo paths to collect from. If None, scans workspace.

        Returns:
            List of Learning objects.
        """
        storage = LearningStorage(self.workspace_dir)

        if target_repos:
            all_learnings = []
            for repo_path in target_repos:
                learnings = storage.load_from_target_repo(repo_path)
                all_learnings.extend(learnings)
            return all_learnings

        return storage.collect_from_workspace()

    def deduplicate(
        self,
        learnings: list[Learning],
        existing_ids: set[str] | None = None,
    ) -> list[Learning]:
        """Remove duplicate learnings.

        Args:
            learnings: List of learnings to deduplicate.
            existing_ids: Set of IDs already in knowledge base.

        Returns:
            Deduplicated list.
        """
        seen_ids = existing_ids or set()
        unique = []

        for learning in learnings:
            if learning.id not in seen_ids:
                seen_ids.add(learning.id)
                unique.append(learning)

        logger.info(f"Deduplicated {len(learnings)} -> {len(unique)} learnings")
        return unique

    def get_existing_learning_ids(self, kb_path: Path) -> set[str]:
        """Get IDs of learnings already in the knowledge base.

        Args:
            kb_path: Path to knowledge base directory.

        Returns:
            Set of existing learning IDs.
        """
        existing_ids = set()

        if not kb_path.exists():
            return existing_ids

        # Scan all markdown files in category directories
        for category_dir in CATEGORY_DIRS.values():
            cat_path = kb_path / category_dir
            if not cat_path.exists():
                continue

            for md_file in cat_path.glob("*.md"):
                if md_file.name == "README.md":
                    continue

                content = md_file.read_text()
                # Extract ID from frontmatter
                match = re.search(r"^id:\s*(\S+)", content, re.MULTILINE)
                if match:
                    existing_ids.add(match.group(1))

        return existing_ids

    def merge_learnings_into_kb(
        self,
        learnings: list[Learning],
        kb_path: Path,
    ) -> dict[str, str]:
        """Merge new learnings into knowledge base files.

        Args:
            learnings: Learnings to merge.
            kb_path: Path to knowledge base directory.

        Returns:
            Dict of file_path -> content for files that were created/modified.
        """
        storage = LearningStorage()
        changes: dict[str, str] = {}

        for learning in learnings:
            # Determine category directory
            category_dir = CATEGORY_DIRS.get(
                learning.category, "error-resolutions"
            )

            # Generate filename (no ticket reference - use ID and subcategory only)
            timestamp_str = learning.timestamp.strftime("%Y-%m-%d")
            safe_subcategory = re.sub(r"[^a-zA-Z0-9-]", "", learning.subcategory)
            filename = f"{timestamp_str}-{learning.id}-{safe_subcategory}.md"

            file_path = kb_path / category_dir / filename
            # Anonymize when publishing to jira-agent repo
            content = storage.render_markdown(learning, anonymize=True)

            # Ensure parent directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)

            changes[str(file_path)] = content
            logger.debug(f"Will create/update: {file_path}")

        return changes

    def publish(
        self,
        learnings: list[Learning] | None = None,
        dry_run: bool = False,
        branch_prefix: str = "learn",
    ) -> dict:
        """Publish learnings to jira-agent repo via PR.

        Args:
            learnings: Learnings to publish. If None, collects from workspace.
            dry_run: If True, don't create PR, just return what would be done.
            branch_prefix: Prefix for the branch name.

        Returns:
            Result dict with status and PR URL if created.
        """
        # Collect learnings if not provided
        if learnings is None:
            learnings = self.collect_learnings()

        if not learnings:
            return {"status": "no_learnings", "message": "No learnings to publish"}

        # Clone jira-agent repo
        git = GitTools(self.workspace_dir, self.github_token)
        repo_path = git.clone_repo(self.repo_owner, self.repo_name)

        kb_path = repo_path / KNOWLEDGE_BASE_DIR

        # Get existing learning IDs for deduplication
        existing_ids = self.get_existing_learning_ids(kb_path)

        # Deduplicate
        new_learnings = self.deduplicate(learnings, existing_ids)

        if not new_learnings:
            return {"status": "all_duplicates", "message": "All learnings already exist"}

        # Merge into knowledge base
        changes = self.merge_learnings_into_kb(new_learnings, kb_path)

        if not changes:
            return {"status": "no_changes", "message": "No changes to make"}

        if dry_run:
            return {
                "status": "dry_run",
                "learnings_count": len(new_learnings),
                "files_to_create": list(changes.keys()),
            }

        # Create branch
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        branch_name = f"{branch_prefix}/add-learnings-{timestamp}"

        git.set_repo_path(repo_path)
        git.create_branch_from(branch_name, "main")

        # Write files
        for file_path, content in changes.items():
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            Path(file_path).write_text(content)

        # Commit and push
        git.stage_files()
        commit_msg = f"docs: add {len(new_learnings)} learnings from agent execution"
        git.commit(commit_msg)
        git.push(branch_name)

        # Create PR
        pr_url = self._create_pr(
            branch_name,
            new_learnings,
        )

        return {
            "status": "success",
            "pr_url": pr_url,
            "learnings_count": len(new_learnings),
            "files_created": list(changes.keys()),
        }

    def _create_pr(
        self,
        branch_name: str,
        learnings: list[Learning],
    ) -> str:
        """Create a pull request to jira-agent repo.

        Args:
            branch_name: Branch with changes.
            learnings: Learnings being added.

        Returns:
            PR URL.
        """
        import httpx

        # Build PR body
        body_lines = [
            "## Summary",
            "",
            f"This PR adds {len(learnings)} new learnings from agent execution.",
            "",
            "### Learnings Added",
            "",
        ]

        for learning in learnings[:10]:  # Show first 10
            body_lines.append(f"- **{learning.title}** ({learning.category.value}/{learning.subcategory})")

        if len(learnings) > 10:
            body_lines.append(f"- ... and {len(learnings) - 10} more")

        body_lines.extend([
            "",
            "### Categories",
            "",
        ])

        # Count by category
        category_counts: dict[str, int] = {}
        for learning in learnings:
            cat = learning.category.value
            category_counts[cat] = category_counts.get(cat, 0) + 1

        for cat, count in sorted(category_counts.items()):
            body_lines.append(f"- {cat}: {count}")

        body = "\n".join(body_lines)

        # Create PR via GitHub API
        url = f"https://api.github.com/repos/{self.repo_owner}/{self.repo_name}/pulls"
        headers = {
            "Authorization": f"Bearer {self.github_token}",
            "Accept": "application/vnd.github.v3+json",
        }
        data = {
            "title": f"docs: add {len(learnings)} agent learnings",
            "body": body,
            "head": branch_name,
            "base": "main",
        }

        with httpx.Client() as client:
            response = client.post(url, json=data, headers=headers)
            response.raise_for_status()
            pr_data = response.json()

        pr_url = pr_data.get("html_url", "")
        logger.info(f"Created PR: {pr_url}")
        return pr_url

    def cleanup_published_learnings(self, repo_paths: list[Path]) -> int:
        """Remove published learnings from target repos.

        Args:
            repo_paths: List of repo paths to clean up.

        Returns:
            Number of files removed.
        """
        removed_count = 0

        for repo_path in repo_paths:
            learnings_dir = repo_path / ".jira-agent" / "learnings"
            if not learnings_dir.exists():
                continue

            for md_file in learnings_dir.glob("*.md"):
                md_file.unlink()
                removed_count += 1
                logger.debug(f"Removed: {md_file}")

        return removed_count
