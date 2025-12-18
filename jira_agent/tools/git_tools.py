"""Git operations tools for the agent."""

import logging
import re
import subprocess
from pathlib import Path

from git import Repo
from git.exc import GitCommandError

logger = logging.getLogger(__name__)


class GitTools:
    """Git operations for repository management."""

    def __init__(self, workspace_dir: Path, github_token: str | None = None):
        """Initialize Git tools.

        Args:
            workspace_dir: Base directory for cloned repositories.
            github_token: GitHub token for authenticated operations.
        """
        self.workspace_dir = workspace_dir
        self.github_token = github_token
        self._repo: Repo | None = None
        self._repo_path: Path | None = None

    def _get_auth_url(self, repo_url: str) -> str:
        """Add authentication to repository URL.

        Args:
            repo_url: Original repository URL.

        Returns:
            URL with embedded token for auth.
        """
        if not self.github_token:
            return repo_url

        # Convert HTTPS URL to include token
        if repo_url.startswith("https://github.com/"):
            return repo_url.replace(
                "https://github.com/",
                f"https://x-access-token:{self.github_token}@github.com/",
            )
        return repo_url

    def clone_repo(self, owner: str, name: str) -> Path:
        """Clone a repository to the workspace.

        Args:
            owner: Repository owner.
            name: Repository name.

        Returns:
            Path to cloned repository.
        """
        repo_path = self.workspace_dir / f"{owner}-{name}"
        repo_url = f"https://github.com/{owner}/{name}.git"
        auth_url = self._get_auth_url(repo_url)

        if repo_path.exists():
            logger.info(f"Repository already exists at {repo_path}, pulling latest")
            self._repo = Repo(repo_path)
            self._repo_path = repo_path
            # Fetch all and pull current branch
            self._repo.remotes.origin.fetch()
            return repo_path

        logger.info(f"Cloning {owner}/{name} to {repo_path}")
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

        self._repo = Repo.clone_from(auth_url, repo_path)
        self._repo_path = repo_path

        return repo_path

    def set_repo_path(self, repo_path: Path) -> None:
        """Set the repository path for operations.

        Args:
            repo_path: Path to existing repository.
        """
        self._repo = Repo(repo_path)
        self._repo_path = repo_path

    @property
    def repo(self) -> Repo:
        """Get the current repository.

        Returns:
            Git repository object.

        Raises:
            ValueError: If no repository is set.
        """
        if self._repo is None:
            raise ValueError("No repository set. Call clone_repo or set_repo_path first.")
        return self._repo

    @property
    def repo_path(self) -> Path:
        """Get the current repository path.

        Returns:
            Path to repository.

        Raises:
            ValueError: If no repository is set.
        """
        if self._repo_path is None:
            raise ValueError("No repository set. Call clone_repo or set_repo_path first.")
        return self._repo_path

    def checkout_branch(self, branch_name: str, create: bool = False) -> None:
        """Checkout a branch.

        Args:
            branch_name: Branch name to checkout.
            create: If True, create the branch if it doesn't exist.
        """
        if create:
            # Check if branch exists locally
            if branch_name in [b.name for b in self.repo.branches]:
                self.repo.git.checkout(branch_name)
            else:
                self.repo.git.checkout("-b", branch_name)
        else:
            self.repo.git.checkout(branch_name)

        logger.info(f"Checked out branch: {branch_name}")

    def create_branch_from(
        self,
        branch_name: str,
        base_branch: str = "main",
        fetch_first: bool = True,
    ) -> None:
        """Create a new branch from a base branch.

        Args:
            branch_name: Name for the new branch.
            base_branch: Branch to create from.
            fetch_first: Whether to fetch latest from origin first.
        """
        if fetch_first:
            logger.info(f"Fetching origin/{base_branch}")
            self.repo.remotes.origin.fetch(base_branch)

        # Create branch from origin/base_branch
        self.repo.git.checkout("-b", branch_name, f"origin/{base_branch}")
        logger.info(f"Created branch {branch_name} from origin/{base_branch}")

    def get_current_branch(self) -> str:
        """Get the current branch name.

        Returns:
            Current branch name.
        """
        return self.repo.active_branch.name

    def stage_files(self, paths: list[str] | None = None) -> None:
        """Stage files for commit.

        Args:
            paths: Specific file paths to stage. If None, stages all changes.
        """
        if paths:
            self.repo.index.add(paths)
        else:
            self.repo.git.add("-A")

        logger.info(f"Staged {len(paths) if paths else 'all'} files")

    def commit(self, message: str, skip_hooks: bool = False) -> str:
        """Create a commit.

        Args:
            message: Commit message.
            skip_hooks: If True, bypass pre-commit hooks (--no-verify).

        Returns:
            Commit SHA.
        """
        if skip_hooks:
            # Use git command directly to skip hooks
            self.repo.git.commit("-m", message, "--no-verify")
        else:
            self.repo.index.commit(message)
        sha = self.repo.head.commit.hexsha
        logger.info(f"Created commit: {sha[:8]} - {message[:50]}")
        return sha

    def push(self, branch_name: str | None = None, force: bool = False) -> None:
        """Push branch to origin.

        Args:
            branch_name: Branch to push. If None, pushes current branch.
            force: Whether to force push.
        """
        branch = branch_name or self.get_current_branch()

        push_args = ["-u", "origin", branch]
        if force:
            push_args.insert(0, "--force")

        self.repo.git.push(*push_args)
        logger.info(f"Pushed branch {branch} to origin")

    def has_changes(self) -> bool:
        """Check if there are uncommitted changes.

        Returns:
            True if there are staged or unstaged changes.
        """
        return self.repo.is_dirty(untracked_files=True)

    def get_diff_files(self) -> list[str]:
        """Get list of changed files.

        Returns:
            List of file paths that have changes.
        """
        # Staged changes
        staged = [item.a_path for item in self.repo.index.diff("HEAD")]

        # Unstaged changes
        unstaged = [item.a_path for item in self.repo.index.diff(None)]

        # Untracked files
        untracked = self.repo.untracked_files

        return list(set(staged + unstaged + untracked))

    def get_changed_files_from_branch(self, base_branch: str) -> list[str]:
        """Get list of files changed compared to a base branch.

        Args:
            base_branch: Branch to compare against (e.g., 'develop', 'main').

        Returns:
            List of file paths that differ from the base branch.
        """
        try:
            # Fetch the base branch to ensure we have latest
            self.repo.git.fetch("origin", base_branch)
            base_ref = f"origin/{base_branch}"

            # Get diff between current HEAD and base branch
            diff_output = self.repo.git.diff("--name-only", base_ref)
            if diff_output:
                return diff_output.strip().split("\n")
            return []
        except GitCommandError as e:
            logger.warning(f"Could not get changed files from {base_branch}: {e}")
            return []

    def reset_hard(self, ref: str = "HEAD") -> None:
        """Hard reset to a reference.

        Args:
            ref: Git reference to reset to.
        """
        self.repo.git.reset("--hard", ref)
        logger.info(f"Reset to {ref}")

    def run_command(self, command: list[str], cwd: Path | None = None) -> tuple[int, str, str]:
        """Run a shell command in the repository.

        Args:
            command: Command and arguments.
            cwd: Working directory. Defaults to repo path.

        Returns:
            Tuple of (return_code, stdout, stderr).
        """
        work_dir = cwd or self.repo_path

        result = subprocess.run(
            command,
            cwd=work_dir,
            capture_output=True,
            text=True,
        )

        return result.returncode, result.stdout, result.stderr

    def run_pre_commit(self) -> tuple[bool, str]:
        """Run pre-commit hooks.

        Returns:
            Tuple of (success, output).
        """
        code, stdout, stderr = self.run_command(["pre-commit", "run", "--all-files"])
        output = stdout + stderr
        success = code == 0

        if not success:
            logger.warning("Pre-commit failed, attempting to stage fixes")
            # Pre-commit may have fixed files, stage them
            self.stage_files()

        return success, output


def sanitize_branch_name(name: str) -> str:
    """Sanitize a string for use as a branch name.

    Args:
        name: Input string.

    Returns:
        Sanitized branch name.
    """
    # Convert to lowercase
    name = name.lower()

    # Replace spaces and special chars with hyphens
    name = re.sub(r"[^a-z0-9-]", "-", name)

    # Remove consecutive hyphens
    name = re.sub(r"-+", "-", name)

    # Remove leading/trailing hyphens
    name = name.strip("-")

    # Limit length
    return name[:50]


def format_branch_name(
    pattern: str,
    ticket_key: str,
    description: str,
    branch_type: str = "feat",
) -> str:
    """Format a branch name from a pattern.

    Args:
        pattern: Branch name pattern with placeholders.
        ticket_key: Jira ticket key.
        description: Short description.
        branch_type: Type prefix (feat, fix, refactor).

    Returns:
        Formatted branch name.
    """
    sanitized_desc = sanitize_branch_name(description)

    return pattern.format(
        type=branch_type,
        ticket_key=ticket_key,
        description=sanitized_desc,
    )
