"""Storage for learnings in markdown files."""

import logging
import re
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .models import Learning, LearningCategory, LearningConfidence

logger = logging.getLogger(__name__)

# Directory name for learnings in target repos
LEARNINGS_DIR = ".jira-agent/learnings"


class LearningStorage:
    """Manages learning storage in markdown files."""

    def __init__(self, workspace_dir: Path | None = None):
        """Initialize storage.

        Args:
            workspace_dir: Base workspace directory where repos are cloned.
        """
        self.workspace_dir = workspace_dir or Path("/tmp/jira-agent")

    def save_to_target_repo(self, learning: Learning, repo_path: Path) -> Path:
        """Save a learning to the target repo's learnings directory.

        Args:
            learning: The learning to save.
            repo_path: Path to the target repository.

        Returns:
            Path to the saved markdown file.
        """
        learnings_dir = repo_path / LEARNINGS_DIR
        learnings_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename from timestamp and ticket
        timestamp_str = learning.timestamp.strftime("%Y-%m-%d")
        safe_ticket = re.sub(r"[^a-zA-Z0-9-]", "", learning.ticket)
        safe_subcategory = re.sub(r"[^a-zA-Z0-9-]", "", learning.subcategory)
        filename = f"{timestamp_str}-{safe_ticket}-{safe_subcategory}.md"

        file_path = learnings_dir / filename
        content = self.render_markdown(learning)
        file_path.write_text(content)

        logger.info(f"Saved learning to {file_path}")
        return file_path

    def load_from_target_repo(self, repo_path: Path) -> list[Learning]:
        """Load all learnings from a target repo.

        Args:
            repo_path: Path to the target repository.

        Returns:
            List of Learning objects.
        """
        learnings_dir = repo_path / LEARNINGS_DIR
        if not learnings_dir.exists():
            return []

        learnings = []
        for md_file in learnings_dir.glob("*.md"):
            try:
                learning = self.parse_markdown(md_file)
                if learning:
                    learnings.append(learning)
            except Exception as e:
                logger.warning(f"Failed to parse learning from {md_file}: {e}")

        return learnings

    def collect_from_workspace(self) -> list[Learning]:
        """Collect learnings from all repos in the workspace.

        Returns:
            List of Learning objects from all repos.
        """
        all_learnings = []

        if not self.workspace_dir.exists():
            return all_learnings

        # Scan all directories in workspace
        for repo_dir in self.workspace_dir.iterdir():
            if repo_dir.is_dir():
                learnings = self.load_from_target_repo(repo_dir)
                all_learnings.extend(learnings)

        return all_learnings

    def render_markdown(self, learning: Learning, anonymize: bool = False) -> str:
        """Render a learning as markdown.

        Args:
            learning: The learning to render.
            anonymize: If True, remove org-specific data (for publishing to jira-agent).

        Returns:
            Markdown string.
        """
        # Build frontmatter
        frontmatter = {
            "id": learning.id,
            "timestamp": learning.timestamp.isoformat(),
            "category": learning.category.value,
            "subcategory": learning.subcategory,
            "confidence": learning.confidence.value,
        }

        # Only include ticket/repo if not anonymizing (for local storage)
        if not anonymize:
            frontmatter["ticket"] = learning.ticket
            frontmatter["repo"] = learning.repo

        # Build markdown content
        lines = [
            "---",
            yaml.dump(frontmatter, default_flow_style=False).strip(),
            "---",
            "",
            f"# {learning.title}",
            "",
            "## Problem",
            learning.problem,
            "",
        ]

        if learning.error_output:
            lines.extend([
                "```",
                learning.error_output,
                "```",
                "",
            ])

        lines.extend([
            "## Solution",
            learning.solution,
            "",
        ])

        if learning.files_affected:
            lines.extend([
                "## Files Affected",
                *[f"- {f}" for f in learning.files_affected],
                "",
            ])

        if learning.code_diff:
            lines.extend([
                "## Code Changes",
                "```diff",
                learning.code_diff,
                "```",
                "",
            ])

        lines.extend([
            "## Applicability",
            learning.applicability,
            "",
        ])

        return "\n".join(lines)

    def parse_markdown(self, file_path: Path) -> Learning | None:
        """Parse a learning from a markdown file.

        Args:
            file_path: Path to the markdown file.

        Returns:
            Learning object or None if parsing fails.
        """
        content = file_path.read_text()

        # Extract frontmatter
        frontmatter_match = re.match(r"^---\n(.*?)\n---\n", content, re.DOTALL)
        if not frontmatter_match:
            logger.warning(f"No frontmatter found in {file_path}")
            return None

        try:
            frontmatter = yaml.safe_load(frontmatter_match.group(1))
        except yaml.YAMLError as e:
            logger.warning(f"Invalid YAML frontmatter in {file_path}: {e}")
            return None

        body = content[frontmatter_match.end():]

        # Extract title
        title_match = re.search(r"^# (.+)$", body, re.MULTILINE)
        title = title_match.group(1) if title_match else "Unknown"

        # Extract sections
        problem = self._extract_section(body, "Problem")
        solution = self._extract_section(body, "Solution")
        applicability = self._extract_section(body, "Applicability")
        code_diff = self._extract_code_block(body, "Code Changes")
        error_output = self._extract_code_block(body, "Problem")

        # Extract files affected
        files_section = self._extract_section(body, "Files Affected")
        files_affected = re.findall(r"^- (.+)$", files_section, re.MULTILINE) if files_section else []

        # Parse timestamp
        timestamp_str = frontmatter.get("timestamp", "")
        try:
            timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            timestamp = datetime.now(timezone.utc)

        return Learning(
            id=frontmatter.get("id", ""),
            timestamp=timestamp,
            ticket=frontmatter.get("ticket", ""),
            category=LearningCategory(frontmatter.get("category", "error-resolution")),
            subcategory=frontmatter.get("subcategory", ""),
            repo=frontmatter.get("repo", ""),
            title=title,
            problem=problem or "",
            error_output=error_output,
            solution=solution or "",
            files_affected=files_affected,
            code_diff=code_diff,
            applicability=applicability or "",
            confidence=LearningConfidence(frontmatter.get("confidence", "medium")),
        )

    def _extract_section(self, body: str, section_name: str) -> str | None:
        """Extract content from a markdown section.

        Args:
            body: Markdown body content.
            section_name: Name of the section (without ##).

        Returns:
            Section content or None.
        """
        pattern = rf"^## {re.escape(section_name)}\n(.*?)(?=^## |\Z)"
        match = re.search(pattern, body, re.MULTILINE | re.DOTALL)
        if match:
            content = match.group(1).strip()
            # Remove code blocks from the extracted content
            content = re.sub(r"```.*?```", "", content, flags=re.DOTALL).strip()
            return content
        return None

    def _extract_code_block(self, body: str, section_name: str) -> str | None:
        """Extract a code block from a section.

        Args:
            body: Markdown body content.
            section_name: Name of the section containing the code block.

        Returns:
            Code block content or None.
        """
        pattern = rf"^## {re.escape(section_name)}\n.*?```(?:\w+)?\n(.*?)```"
        match = re.search(pattern, body, re.MULTILINE | re.DOTALL)
        if match:
            return match.group(1).strip()
        return None
