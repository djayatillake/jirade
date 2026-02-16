"""GitHub tool handlers for MCP server."""

import asyncio
import logging
from typing import Any

from ...auth.manager import AuthManager
from ...clients.github_client import GitHubClient, format_pr_status
from ...config import get_settings

logger = logging.getLogger(__name__)


async def get_github_client(owner: str, repo: str) -> tuple[GitHubClient, AuthManager]:
    """Get an authenticated GitHub client.

    Args:
        owner: Repository owner.
        repo: Repository name.

    Returns:
        Tuple of (GitHubClient, AuthManager).

    Raises:
        RuntimeError: If not authenticated.
    """
    settings = get_settings()
    auth = AuthManager(settings)

    # Try settings token first (from gh CLI or env), then token store
    token = settings.github_token
    if not token and auth.github.is_authenticated():
        token = auth.github.get_access_token()

    if not token:
        raise RuntimeError("Not authenticated with GitHub. Set GITHUB_TOKEN env var or run 'jirade auth login github'.")

    client = GitHubClient(token=token, owner=owner, repo=repo)
    return client, auth


async def handle_github_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Handle a GitHub tool call.

    Args:
        name: Tool name.
        arguments: Tool arguments.

    Returns:
        Tool result.
    """
    owner = arguments.get("owner", "")
    repo = arguments.get("repo", "")

    if not owner or not repo:
        raise ValueError("Missing required arguments: owner and repo")

    client, auth = await get_github_client(owner, repo)

    try:
        if name == "jirade_list_prs":
            return await list_prs(client, arguments)
        elif name == "jirade_get_pr":
            return await get_pr(client, arguments)
        elif name == "jirade_get_ci_status":
            return await get_ci_status(client, arguments)
        elif name == "jirade_watch_pr":
            return await watch_pr(client, arguments)
        else:
            raise ValueError(f"Unknown GitHub tool: {name}")
    finally:
        await client.close()


async def list_prs(client: GitHubClient, arguments: dict[str, Any]) -> dict[str, Any]:
    """List pull requests for a repository.

    Args:
        client: GitHub client.
        arguments: Tool arguments with optional 'state' and 'limit'.

    Returns:
        List of PRs.
    """
    state = arguments.get("state", "open")
    limit = arguments.get("limit", 30)

    prs = await client.list_pull_requests(state=state, per_page=limit)

    results = []
    for pr in prs:
        results.append(
            {
                "number": pr["number"],
                "title": pr["title"],
                "state": pr["state"],
                "draft": pr.get("draft", False),
                "author": pr.get("user", {}).get("login"),
                "head_branch": pr.get("head", {}).get("ref"),
                "base_branch": pr.get("base", {}).get("ref"),
                "created_at": pr.get("created_at"),
                "updated_at": pr.get("updated_at"),
                "html_url": pr.get("html_url"),
            }
        )

    return {
        "total": len(results),
        "pull_requests": results,
    }


async def get_pr(client: GitHubClient, arguments: dict[str, Any]) -> dict[str, Any]:
    """Get details for a specific pull request.

    Args:
        client: GitHub client.
        arguments: Tool arguments with 'number'.

    Returns:
        PR details.
    """
    number = arguments["number"]

    pr = await client.get_pull_request(number)
    reviews = await client.get_pr_reviews(number)
    comments = await client.get_pr_comments(number)

    # Format reviews
    formatted_reviews = []
    for review in reviews:
        formatted_reviews.append(
            {
                "author": review.get("user", {}).get("login"),
                "state": review.get("state"),
                "body": review.get("body", ""),
                "submitted_at": review.get("submitted_at"),
            }
        )

    # Format comments
    formatted_comments = []
    for comment in comments:
        formatted_comments.append(
            {
                "author": comment.get("user", {}).get("login"),
                "body": comment.get("body", ""),
                "created_at": comment.get("created_at"),
            }
        )

    return {
        "number": pr["number"],
        "title": pr["title"],
        "body": pr.get("body", ""),
        "state": pr["state"],
        "draft": pr.get("draft", False),
        "mergeable": pr.get("mergeable"),
        "mergeable_state": pr.get("mergeable_state"),
        "author": pr.get("user", {}).get("login"),
        "head_branch": pr.get("head", {}).get("ref"),
        "head_sha": pr.get("head", {}).get("sha"),
        "base_branch": pr.get("base", {}).get("ref"),
        "created_at": pr.get("created_at"),
        "updated_at": pr.get("updated_at"),
        "html_url": pr.get("html_url"),
        "additions": pr.get("additions", 0),
        "deletions": pr.get("deletions", 0),
        "changed_files": pr.get("changed_files", 0),
        "reviews": formatted_reviews,
        "comments": formatted_comments,
    }


async def get_ci_status(client: GitHubClient, arguments: dict[str, Any]) -> dict[str, Any]:
    """Get CI/CD check status for a pull request.

    Args:
        client: GitHub client.
        arguments: Tool arguments with 'pr_number'.

    Returns:
        CI status details.
    """
    pr_number = arguments["pr_number"]

    # Get PR to find head SHA
    pr = await client.get_pull_request(pr_number)
    head_sha = pr.get("head", {}).get("sha")

    if not head_sha:
        return {
            "error": "Could not determine PR head SHA",
            "pr_number": pr_number,
        }

    # Get check runs (GitHub Actions, etc.)
    check_runs = await client.get_check_runs(head_sha)

    # Get commit statuses (CircleCI, dbt Cloud, etc.)
    combined_status = await client.get_combined_status(head_sha)

    # Format check runs
    formatted_checks = []
    for check in check_runs:
        formatted_checks.append(
            {
                "name": check.get("name"),
                "status": check.get("status"),
                "conclusion": check.get("conclusion"),
                "started_at": check.get("started_at"),
                "completed_at": check.get("completed_at"),
                "details_url": check.get("details_url"),
            }
        )

    # Format commit statuses
    formatted_statuses = []
    for status in combined_status.get("statuses", []):
        formatted_statuses.append(
            {
                "context": status.get("context"),
                "state": status.get("state"),
                "description": status.get("description"),
                "target_url": status.get("target_url"),
                "created_at": status.get("created_at"),
            }
        )

    # Calculate overall status
    status_summary = format_pr_status(pr, check_runs, combined_status)

    return {
        "pr_number": pr_number,
        "head_sha": head_sha,
        "overall_status": status_summary["ci_status"],
        "mergeable": status_summary["mergeable"],
        "mergeable_state": status_summary["mergeable_state"],
        "failed_checks": status_summary["failed_checks"],
        "pending_checks": status_summary["pending_checks"],
        "check_runs": formatted_checks,
        "commit_statuses": formatted_statuses,
    }


async def watch_pr(client: GitHubClient, arguments: dict[str, Any]) -> dict[str, Any]:
    """Watch a PR's CI status until all checks complete.

    Args:
        client: GitHub client.
        arguments: Tool arguments with 'pr_number', optional 'interval' and 'timeout'.

    Returns:
        Final CI status when complete or timed out.
    """
    pr_number = arguments["pr_number"]
    interval = arguments.get("interval", 30)
    timeout = arguments.get("timeout", 1800)  # 30 minutes default

    elapsed = 0
    iteration = 0

    while elapsed < timeout:
        iteration += 1
        logger.info(f"Watch iteration {iteration} for PR #{pr_number} (elapsed: {elapsed}s)")

        # Get PR to find head SHA
        pr = await client.get_pull_request(pr_number)
        head_sha = pr.get("head", {}).get("sha")

        if not head_sha:
            return {
                "error": "Could not determine PR head SHA",
                "pr_number": pr_number,
            }

        # Get check runs and commit statuses
        check_runs = await client.get_check_runs(head_sha)
        combined_status = await client.get_combined_status(head_sha)

        # Calculate overall status
        status_summary = format_pr_status(pr, check_runs, combined_status)

        overall_status = status_summary["ci_status"]
        failed_checks = status_summary["failed_checks"]
        pending_checks = status_summary["pending_checks"]

        # Check if we're done
        if overall_status in ("success", "failure"):
            # Fetch reviews and comments so the caller knows about pending feedback
            pr_author = pr.get("user", {}).get("login", "")
            reviews = await client.get_pr_reviews(pr_number)
            comments = await client.get_pr_comments(pr_number)
            review_comments = await client.get_pr_review_comments(pr_number)

            # Filter to actionable reviews (not the PR author, not bot-generated)
            actionable_reviews = [
                {
                    "author": r.get("user", {}).get("login"),
                    "state": r.get("state"),
                    "body": r.get("body", ""),
                }
                for r in reviews
                if r.get("user", {}).get("login") != pr_author
                and r.get("state") in ("CHANGES_REQUESTED", "COMMENTED")
                and r.get("body", "").strip()
            ]

            # Filter to non-author, non-bot issue comments
            actionable_comments = [
                {
                    "author": c.get("user", {}).get("login"),
                    "body": c.get("body", ""),
                }
                for c in comments
                if c.get("user", {}).get("login") != pr_author
                and c.get("user", {}).get("type") != "Bot"
            ]

            # Include inline review comments from non-author
            actionable_review_comments = [
                {
                    "author": c.get("user", {}).get("login"),
                    "path": c.get("path", ""),
                    "body": c.get("body", ""),
                }
                for c in review_comments
                if c.get("user", {}).get("login") != pr_author
            ]

            result = {
                "pr_number": pr_number,
                "head_sha": head_sha,
                "status": overall_status,
                "elapsed_seconds": elapsed,
                "iterations": iteration,
                "mergeable": status_summary["mergeable"],
            }

            if overall_status == "success":
                result["message"] = "All CI checks passed"
            else:
                result["message"] = f"CI checks failed: {', '.join(failed_checks)}"
                result["failed_checks"] = failed_checks

            if actionable_reviews:
                result["reviews"] = actionable_reviews
            if actionable_comments:
                result["comments"] = actionable_comments
            if actionable_review_comments:
                result["review_comments"] = actionable_review_comments

            return result

        # Still pending, wait and retry
        logger.info(f"PR #{pr_number} has {len(pending_checks)} pending checks, waiting {interval}s...")
        await asyncio.sleep(interval)
        elapsed += interval

    # Timeout reached
    return {
        "pr_number": pr_number,
        "status": "timeout",
        "message": f"Timed out after {timeout} seconds",
        "pending_checks": pending_checks,
        "elapsed_seconds": elapsed,
        "iterations": iteration,
    }
