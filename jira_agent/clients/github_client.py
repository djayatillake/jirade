"""GitHub REST API client."""

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class GitHubClient:
    """Client for GitHub REST API."""

    def __init__(self, token: str, owner: str, repo: str):
        """Initialize GitHub client.

        Args:
            token: GitHub access token.
            owner: Repository owner.
            repo: Repository name.
        """
        self.token = token
        self.owner = owner
        self.repo = repo
        self.base_url = "https://api.github.com"
        self.headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        self._client = httpx.AsyncClient(headers=self.headers, timeout=30.0)

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()

    async def __aenter__(self) -> "GitHubClient":
        return self

    async def __aexit__(self, *args) -> None:
        await self.close()

    @property
    def repo_url(self) -> str:
        """Get the repository API URL."""
        return f"{self.base_url}/repos/{self.owner}/{self.repo}"

    async def _request(
        self,
        method: str,
        url: str,
        **kwargs,
    ) -> dict[str, Any] | list[Any]:
        """Make an HTTP request.

        Args:
            method: HTTP method.
            url: Full URL.
            **kwargs: Additional request arguments.

        Returns:
            JSON response.
        """
        response = await self._client.request(method, url, **kwargs)
        response.raise_for_status()

        if response.status_code == 204:
            return {}

        return response.json()

    async def create_pull_request(
        self,
        title: str,
        body: str,
        head: str,
        base: str,
        draft: bool = False,
    ) -> dict[str, Any]:
        """Create a pull request.

        Args:
            title: PR title.
            body: PR description.
            head: Source branch.
            base: Target branch.
            draft: Create as draft PR.

        Returns:
            Created PR data.
        """
        url = f"{self.repo_url}/pulls"
        data = {
            "title": title,
            "body": body,
            "head": head,
            "base": base,
            "draft": draft,
        }

        result = await self._request("POST", url, json=data)
        logger.info(f"Created PR #{result['number']}: {result['html_url']}")
        return result

    async def get_pull_request(self, pr_number: int) -> dict[str, Any]:
        """Get pull request details.

        Args:
            pr_number: PR number.

        Returns:
            PR data.
        """
        url = f"{self.repo_url}/pulls/{pr_number}"
        return await self._request("GET", url)

    async def list_pull_requests(
        self,
        state: str = "open",
        sort: str = "updated",
        direction: str = "desc",
        per_page: int = 30,
    ) -> list[dict[str, Any]]:
        """List pull requests for the repository.

        Args:
            state: Filter by state: open, closed, all.
            sort: Sort by: created, updated, popularity, long-running.
            direction: Sort direction: asc, desc.
            per_page: Results per page (max 100).

        Returns:
            List of PRs.
        """
        url = f"{self.repo_url}/pulls"
        params = {
            "state": state,
            "sort": sort,
            "direction": direction,
            "per_page": per_page,
        }
        return await self._request("GET", url, params=params)

    async def get_pr_reviews(self, pr_number: int) -> list[dict[str, Any]]:
        """Get reviews on a PR.

        Args:
            pr_number: PR number.

        Returns:
            List of reviews.
        """
        url = f"{self.repo_url}/pulls/{pr_number}/reviews"
        return await self._request("GET", url)

    async def get_pr_review_comments(self, pr_number: int) -> list[dict[str, Any]]:
        """Get review comments on a PR.

        Args:
            pr_number: PR number.

        Returns:
            List of review comments.
        """
        url = f"{self.repo_url}/pulls/{pr_number}/comments"
        return await self._request("GET", url)

    async def reply_to_review_comment(
        self,
        pr_number: int,
        comment_id: int,
        body: str,
    ) -> dict[str, Any]:
        """Reply to a review comment.

        Args:
            pr_number: PR number.
            comment_id: Comment ID to reply to.
            body: Reply body.

        Returns:
            Created reply data.
        """
        url = f"{self.repo_url}/pulls/{pr_number}/comments/{comment_id}/replies"
        return await self._request("POST", url, json={"body": body})

    async def add_pr_comment(self, pr_number: int, body: str) -> dict[str, Any]:
        """Add a comment to a PR (issue comment).

        Args:
            pr_number: PR number.
            body: Comment body.

        Returns:
            Created comment data.
        """
        url = f"{self.repo_url}/issues/{pr_number}/comments"
        return await self._request("POST", url, json={"body": body})

    async def get_check_runs(self, ref: str) -> list[dict[str, Any]]:
        """Get check runs for a reference.

        Args:
            ref: Git reference (SHA, branch, tag).

        Returns:
            List of check runs.
        """
        url = f"{self.repo_url}/commits/{ref}/check-runs"
        data = await self._request("GET", url)
        return data.get("check_runs", [])

    async def get_check_run_annotations(self, check_run_id: int) -> list[dict[str, Any]]:
        """Get annotations for a check run.

        Args:
            check_run_id: Check run ID.

        Returns:
            List of annotations.
        """
        url = f"{self.repo_url}/check-runs/{check_run_id}/annotations"
        return await self._request("GET", url)

    async def get_combined_status(self, ref: str) -> dict[str, Any]:
        """Get combined commit status.

        Args:
            ref: Git reference.

        Returns:
            Combined status data.
        """
        url = f"{self.repo_url}/commits/{ref}/status"
        return await self._request("GET", url)

    async def get_file_content(
        self,
        path: str,
        ref: str | None = None,
    ) -> str | None:
        """Get file content from repository.

        Args:
            path: File path.
            ref: Git reference (defaults to default branch).

        Returns:
            File content or None if not found.
        """
        url = f"{self.repo_url}/contents/{path}"
        params = {}
        if ref:
            params["ref"] = ref

        try:
            data = await self._request("GET", url, params=params)
            if data.get("encoding") == "base64":
                import base64

                return base64.b64decode(data["content"]).decode("utf-8")
            return data.get("content")
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            raise

    async def get_prs_for_branch(self, branch: str) -> list[dict[str, Any]]:
        """Get PRs for a branch.

        Args:
            branch: Branch name.

        Returns:
            List of PRs.
        """
        url = f"{self.repo_url}/pulls"
        params = {"head": f"{self.owner}:{branch}", "state": "all"}
        return await self._request("GET", url, params=params)

    async def get_suggested_reviewers(self, pr_number: int) -> list[dict[str, Any]]:
        """Get suggested reviewers for a PR.

        Uses GitHub's suggested reviewers API based on blame info and past reviews.

        Args:
            pr_number: PR number.

        Returns:
            List of suggested reviewer users.
        """
        url = f"{self.repo_url}/pulls/{pr_number}/requested_reviewers"
        try:
            # First get current requested reviewers
            current = await self._request("GET", url)
            current_logins = {u["login"] for u in current.get("users", [])}

            # Get collaborators as potential reviewers
            collab_url = f"{self.repo_url}/collaborators"
            collaborators = await self._request("GET", collab_url, params={"per_page": 50})

            # Filter out current reviewers and the PR author
            pr = await self.get_pull_request(pr_number)
            author = pr.get("user", {}).get("login", "")

            suggested = []
            for collab in collaborators:
                login = collab.get("login", "")
                if login and login != author and login not in current_logins:
                    suggested.append({
                        "login": login,
                        "avatar_url": collab.get("avatar_url", ""),
                        "type": collab.get("type", "User"),
                    })

            return suggested[:10]  # Limit to 10 suggestions

        except httpx.HTTPStatusError:
            return []

    async def request_reviewers(
        self,
        pr_number: int,
        reviewers: list[str],
        team_reviewers: list[str] | None = None,
    ) -> dict[str, Any]:
        """Request reviewers for a PR.

        Args:
            pr_number: PR number.
            reviewers: List of user logins to request review from.
            team_reviewers: Optional list of team slugs.

        Returns:
            Updated PR data.
        """
        url = f"{self.repo_url}/pulls/{pr_number}/requested_reviewers"
        data = {"reviewers": reviewers}
        if team_reviewers:
            data["team_reviewers"] = team_reviewers

        return await self._request("POST", url, json=data)

    async def get_pr_comments(self, pr_number: int) -> list[dict[str, Any]]:
        """Get all comments on a PR (issue comments, not review comments).

        Args:
            pr_number: PR number.

        Returns:
            List of comments.
        """
        url = f"{self.repo_url}/issues/{pr_number}/comments"
        return await self._request("GET", url)

    async def get_workflow_runs(
        self,
        branch: str | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get workflow runs.

        Args:
            branch: Filter by branch.
            status: Filter by status.

        Returns:
            List of workflow runs.
        """
        url = f"{self.repo_url}/actions/runs"
        params = {}
        if branch:
            params["branch"] = branch
        if status:
            params["status"] = status

        data = await self._request("GET", url, params=params)
        return data.get("workflow_runs", [])

    async def get_workflow_run_logs(self, run_id: int) -> bytes:
        """Get logs for a workflow run.

        Args:
            run_id: Workflow run ID.

        Returns:
            Compressed log data (ZIP format).
        """
        url = f"{self.repo_url}/actions/runs/{run_id}/logs"

        # Need to follow redirect for logs
        response = await self._client.get(url, follow_redirects=True)
        response.raise_for_status()
        return response.content


def format_pr_status(
    pr: dict[str, Any],
    checks: list[dict[str, Any]],
    statuses: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Format PR status for agent consumption.

    Args:
        pr: PR data.
        checks: Check run data.
        statuses: Combined commit status data (for CircleCI, etc.).

    Returns:
        Formatted status summary.
    """
    # Check runs (GitHub Actions, etc.)
    failed_checks = [c["name"] for c in checks if c.get("conclusion") == "failure"]
    pending_checks = [c["name"] for c in checks if c.get("status") != "completed"]

    # Commit statuses (CircleCI, etc.)
    failed_statuses = []
    pending_statuses = []
    if statuses and statuses.get("statuses"):
        for s in statuses["statuses"]:
            context = s.get("context", "unknown")
            state = s.get("state")
            if state == "failure" or state == "error":
                failed_statuses.append(context)
            elif state == "pending":
                pending_statuses.append(context)

    all_failed = failed_checks + failed_statuses
    all_pending = pending_checks + pending_statuses
    all_passed = len(all_failed) == 0 and len(all_pending) == 0

    return {
        "number": pr["number"],
        "state": pr["state"],
        "mergeable": pr.get("mergeable"),
        "mergeable_state": pr.get("mergeable_state"),
        "ci_status": "success" if all_passed else "pending" if all_pending else "failure",
        "failed_checks": all_failed,
        "pending_checks": all_pending,
        "draft": pr.get("draft", False),
        "html_url": pr["html_url"],
    }
