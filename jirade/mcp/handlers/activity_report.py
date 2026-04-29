"""Activity report handler — pulls all the raw data needed to write a jirade activity report.

Intentionally a data-collection tool, not a classifier. Reports are written
weekly or monthly and the shape evolves over time, so this tool returns
structured PR + ticket data and lets the calling agent synthesise the
narrative, funnel buckets, and prose each run. The agent then publishes
the resulting markdown via `jirade_publish_confluence_page` if desired.

What it pulls:
  - Self-authored PRs in the window (with merged/closed/open state)
  - PRs the user is involved in (commenter / reviewer / mentioned)
  - PRs by other users that mention 'jirade' (cross-user discovery)
  - For each non-self-authored PR: reviews + commits, so the agent can tell
    "review only" from "review + cleanup commit on someone else's branch"
  - Jira tickets with the `jirade` label
  - Jira tickets where the user is assignee/reporter (current or past)
  - Jira tickets with comments matching jirade signature phrases
"""

import json
import logging
import subprocess
from datetime import datetime, timedelta, timezone
from typing import Any

from ...auth.manager import AuthManager
from ...clients.jira_client import JiraClient
from ...config import get_settings

logger = logging.getLogger(__name__)


# ============================================================
# Subprocess helpers (shell out to gh)
# ============================================================


def _gh_json(args: list[str]) -> Any:
    """Run a gh command and return parsed JSON. Returns [] on failure."""
    cmd = ["gh"] + args
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if proc.returncode != 0:
            logger.warning("gh failed: %s | %s", " ".join(cmd), proc.stderr.strip())
            return []
        return json.loads(proc.stdout) if proc.stdout.strip() else []
    except (subprocess.TimeoutExpired, json.JSONDecodeError) as e:
        logger.warning("gh error (%s): %s", e, " ".join(cmd))
        return []


def _detect_gh_user() -> str | None:
    """Detect the authenticated gh user."""
    try:
        proc = subprocess.run(
            ["gh", "api", "user", "--jq", ".login"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if proc.returncode == 0:
            return proc.stdout.strip() or None
    except Exception:
        pass
    return None


# ============================================================
# Main entry
# ============================================================


async def handle_activity_report_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Pull the raw data needed for a jirade activity report.

    Returns structured data — does NOT classify or render markdown.
    The agent does that each run so the report shape can evolve freely.
    """
    if name != "jirade_activity_report":
        raise ValueError(f"Unknown activity-report tool: {name}")

    since_arg = arguments.get("since")
    if since_arg:
        since = since_arg
    else:
        since = (datetime.now(timezone.utc) - timedelta(days=90)).date().isoformat()

    repo = arguments.get("repo", "algolia/data")
    projects = arguments.get("projects", ["AENG", "DATASD", "DATA"])
    user = arguments.get("user") or _detect_gh_user()
    if not user:
        raise RuntimeError("Could not auto-detect GitHub user. Pass `user` explicitly or run `gh auth login`.")

    settings = get_settings()
    auth = AuthManager(settings)
    if not auth.jira.is_authenticated():
        raise RuntimeError("Not authenticated with Atlassian. Run 'jirade auth login --service=jira'.")

    access_token = auth.jira.get_access_token()
    cloud_id = auth.jira.get_cloud_id()
    jira_client = JiraClient(cloud_id=cloud_id, access_token=access_token)

    try:
        gh_data = _collect_github(repo, user, since)
        jira_data = await _collect_jira(jira_client, since, projects)
    finally:
        await jira_client.close()

    return {
        "window": {"since": since, "until": datetime.now(timezone.utc).date().isoformat()},
        "repo": repo,
        "user": user,
        "projects": projects,
        "github": gh_data,
        "jira": jira_data,
        "guidance": (
            "This is raw data — synthesize the report narrative yourself each run. "
            "Suggested funnel buckets per PR: initiated→merged, initiated→in-flight, "
            "initiated→abandoned (closed without merge), reviewed-only (no commit pushed by user), "
            "reviewed+commit (commit pushed to other author's branch). For tickets: groomed only "
            "(no PR by user), incident investigation (DATASD type=Incident with substantial findings comment), "
            "or end-to-end (matches a self-authored PR by ticket key). Split the report into "
            "Part 1 (the caller) and Part 2 (other users found via the cross-user 'jirade' search). "
            "Publish via jirade_publish_confluence_page when ready."
        ),
    }


# ============================================================
# GitHub collection
# ============================================================


def _collect_github(repo: str, user: str, since: str) -> dict[str, Any]:
    """Pull all relevant GitHub PRs and add review/commit metadata for non-self-authored ones."""
    self_authored = (
        _gh_json(
            [
                "search", "prs",
                "--repo", repo,
                "--author", user,
                "--created", f">={since}",
                "--limit", "100",
                "--json", "number,title,author,state,createdAt,closedAt,mergedAt,headRefName,url",
            ]
        )
        or []
    )

    involves = (
        _gh_json(
            [
                "search", "prs",
                "--repo", repo,
                "--involves", user,
                "--created", f">={since}",
                "--limit", "100",
                "--json", "number,title,author,state,createdAt,closedAt,url",
            ]
        )
        or []
    )

    jirade_mentions = (
        _gh_json(
            [
                "search", "prs",
                "--repo", repo,
                "jirade",
                "--created", f">={since}",
                "--limit", "100",
                "--json", "number,title,author,state,createdAt,closedAt,url",
            ]
        )
        or []
    )

    self_numbers = {p["number"] for p in self_authored}
    other_involved = [p for p in involves if p["number"] not in self_numbers]
    other_jirade = [
        p for p in jirade_mentions if p["number"] not in self_numbers and p["author"]["login"] != user
    ]

    # Enrich other-involved PRs with review+commit metadata so the agent can
    # tell review-only from review-and-commit
    owner, repo_name = repo.split("/", 1)
    enriched: list[dict[str, Any]] = []
    for pr in other_involved:
        n = pr["number"]
        commits = _gh_json(
            [
                "api", f"repos/{owner}/{repo_name}/pulls/{n}/commits",
                "--jq", "[.[] | {sha, login: .author.login, message: .commit.message}]",
            ]
        ) or []
        reviews = _gh_json(
            [
                "api", f"repos/{owner}/{repo_name}/pulls/{n}/reviews",
                "--jq", "[.[] | {state, user: .user.login, body, submittedAt: .submitted_at}]",
            ]
        ) or []
        user_commits = [c for c in commits if c.get("login") == user]
        user_reviews = [r for r in reviews if r.get("user") == user]
        enriched.append(
            {
                **pr,
                "commits_by_user": user_commits,
                "reviews_by_user": user_reviews,
                "user_committed_to_branch": bool(user_commits),
            }
        )

    return {
        "self_authored": self_authored,
        "other_involved": enriched,
        "other_jirade_authors": other_jirade,
    }


# ============================================================
# Jira collection
# ============================================================


async def _collect_jira(client: JiraClient, since: str, projects: list[str]) -> dict[str, Any]:
    """Pull tickets via several JQL angles, deduped, with the source labels preserved."""
    project_clause = " OR ".join(f'project = "{p}"' for p in projects)

    queries: list[tuple[str, str]] = [
        ("jirade_labeled", f'labels = "jirade" AND updated >= "{since}" ORDER BY updated DESC'),
        (
            "self_involved",
            f'({project_clause}) AND (assignee = currentUser() OR reporter = currentUser() OR assignee was currentUser()) AND updated >= "{since}" ORDER BY updated DESC',
        ),
        ("comment_jirade_grooming", f'comment ~ "jirade grooming" AND updated >= "{since}" ORDER BY updated DESC'),
        ("comment_via_claude_code", f'comment ~ "via Claude Code" AND updated >= "{since}" ORDER BY updated DESC'),
        ("comment_implemented_by_jirade", f'comment ~ "Implemented by Jirade" AND updated >= "{since}" ORDER BY updated DESC'),
    ]

    by_key: dict[str, dict[str, Any]] = {}
    for source_label, jql in queries:
        try:
            issues = await client.search_issues(
                jql,
                max_results=100,
                fields=["key", "summary", "status", "issuetype", "priority", "labels", "assignee", "updated"],
            )
        except Exception as e:
            logger.warning("Jira query '%s' failed: %s", source_label, e)
            issues = []

        for issue in issues:
            key = issue["key"]
            fields = issue.get("fields", {})
            entry = by_key.setdefault(
                key,
                {
                    "key": key,
                    "summary": fields.get("summary"),
                    "status": fields.get("status", {}).get("name"),
                    "type": fields.get("issuetype", {}).get("name"),
                    "priority": fields.get("priority", {}).get("name"),
                    "labels": fields.get("labels", []),
                    "assignee": (fields.get("assignee") or {}).get("displayName"),
                    "updated": fields.get("updated"),
                    "detected_via": [],
                },
            )
            if source_label not in entry["detected_via"]:
                entry["detected_via"].append(source_label)

    return {"tickets": list(by_key.values())}
