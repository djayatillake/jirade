"""Tool definitions for the MCP server."""

from typing import Any

# Tool definitions as JSON schemas for MCP
TOOLS: list[dict[str, Any]] = [
    # =========== Jira Tools ===========
    {
        "name": "jirade_search_jira",
        "description": "Search Jira issues using JQL (Jira Query Language). Returns matching issues with key, summary, status, and other fields.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "jql": {
                    "type": "string",
                    "description": "JQL query string (e.g., 'project = AENG AND status = \"In Progress\"', 'assignee = currentUser()')",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum results to return (default 20)",
                    "default": 20,
                },
            },
            "required": ["jql"],
        },
    },
    {
        "name": "jirade_get_issue",
        "description": "Get full details for a specific Jira issue including description, status, comments, and custom fields.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": "The Jira issue key (e.g., 'AENG-1234', 'DATA-567')",
                },
            },
            "required": ["key"],
        },
    },
    {
        "name": "jirade_add_comment",
        "description": "Add a comment to a Jira issue.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": "The Jira issue key",
                },
                "comment": {
                    "type": "string",
                    "description": "The comment text to add",
                },
            },
            "required": ["key", "comment"],
        },
    },
    {
        "name": "jirade_transition_issue",
        "description": "Change the status of a Jira issue (e.g., move to 'In Progress', 'Done', 'Code Review').",
        "inputSchema": {
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": "The Jira issue key",
                },
                "status": {
                    "type": "string",
                    "description": "The target status name (e.g., 'In Progress', 'Done', 'Ready for QA')",
                },
            },
            "required": ["key", "status"],
        },
    },
    # =========== GitHub Tools ===========
    {
        "name": "jirade_list_prs",
        "description": "List pull requests for a GitHub repository.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "owner": {
                    "type": "string",
                    "description": "Repository owner (e.g., 'algolia')",
                },
                "repo": {
                    "type": "string",
                    "description": "Repository name (e.g., 'data')",
                },
                "state": {
                    "type": "string",
                    "enum": ["open", "closed", "all"],
                    "description": "Filter by PR state (default: open)",
                    "default": "open",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of PRs to return (default 30)",
                    "default": 30,
                },
            },
            "required": ["owner", "repo"],
        },
    },
    {
        "name": "jirade_get_pr",
        "description": "Get details for a specific pull request including status, reviews, and comments.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "owner": {
                    "type": "string",
                    "description": "Repository owner",
                },
                "repo": {
                    "type": "string",
                    "description": "Repository name",
                },
                "number": {
                    "type": "integer",
                    "description": "The PR number",
                },
            },
            "required": ["owner", "repo", "number"],
        },
    },
    {
        "name": "jirade_get_ci_status",
        "description": "Get CI/CD check status for a pull request (GitHub Actions, CircleCI, dbt Cloud, etc.).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "owner": {
                    "type": "string",
                    "description": "Repository owner",
                },
                "repo": {
                    "type": "string",
                    "description": "Repository name",
                },
                "pr_number": {
                    "type": "integer",
                    "description": "The PR number to check",
                },
            },
            "required": ["owner", "repo", "pr_number"],
        },
    },
    # =========== dbt Cloud Tools ===========
    {
        "name": "jirade_dbt_list_jobs",
        "description": "List dbt Cloud jobs in the configured account. Can filter by project.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "project_id": {
                    "type": "integer",
                    "description": "Optional dbt Cloud project ID to filter jobs",
                },
            },
        },
    },
    {
        "name": "jirade_dbt_trigger_run",
        "description": "Trigger a dbt Cloud CI job run for a pull request. Updates event-time dates for microbatch models.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "job_id": {
                    "type": "integer",
                    "description": "dbt Cloud job ID to trigger",
                },
                "pr_number": {
                    "type": "integer",
                    "description": "GitHub PR number",
                },
                "git_sha": {
                    "type": "string",
                    "description": "Git commit SHA to build",
                },
                "git_branch": {
                    "type": "string",
                    "description": "Git branch name (optional)",
                },
            },
            "required": ["job_id", "pr_number", "git_sha"],
        },
    },
    {
        "name": "jirade_dbt_get_run",
        "description": "Get status and details of a dbt Cloud run, including errors if failed.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "run_id": {
                    "type": "integer",
                    "description": "dbt Cloud run ID",
                },
                "include_errors": {
                    "type": "boolean",
                    "description": "Include detailed error information if run failed (default: true)",
                    "default": True,
                },
            },
            "required": ["run_id"],
        },
    },
    {
        "name": "jirade_dbt_trigger_ci_for_pr",
        "description": "Trigger a dbt Cloud CI run for a PR with file-based model selection. Automatically detects changed dbt models from the PR and runs only those models and their downstream dependencies.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "owner": {
                    "type": "string",
                    "description": "Repository owner (e.g., 'algolia')",
                },
                "repo": {
                    "type": "string",
                    "description": "Repository name (e.g., 'data')",
                },
                "pr_number": {
                    "type": "integer",
                    "description": "GitHub PR number",
                },
                "job_id": {
                    "type": "integer",
                    "description": "dbt Cloud CI job ID (optional, uses configured default if not provided)",
                },
                "dbt_project_subdirectory": {
                    "type": "string",
                    "description": "Subdirectory containing dbt project (e.g., 'dbt-databricks'). Optional if configured in .jirade.yaml",
                },
            },
            "required": ["owner", "repo", "pr_number"],
        },
    },
]


def get_tools() -> list[dict[str, Any]]:
    """Return the list of tools for the MCP server."""
    return TOOLS
