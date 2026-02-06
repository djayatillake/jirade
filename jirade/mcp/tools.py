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
                    "description": "JQL query string (e.g., 'project = PROJ AND status = \"In Progress\"', 'assignee = currentUser()')",
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
                    "description": "The Jira issue key (e.g., 'PROJ-1234')",
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
                    "description": "Repository owner (e.g., 'my-org')",
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
    {
        "name": "jirade_watch_pr",
        "description": "Watch a PR's CI status until all checks complete. Polls every interval seconds and returns when all checks pass, any check fails, or timeout is reached.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "owner": {
                    "type": "string",
                    "description": "Repository owner (e.g., 'my-org')",
                },
                "repo": {
                    "type": "string",
                    "description": "Repository name (e.g., 'data')",
                },
                "pr_number": {
                    "type": "integer",
                    "description": "The PR number to watch",
                },
                "interval": {
                    "type": "integer",
                    "description": "Polling interval in seconds (default: 30)",
                    "default": 30,
                },
                "timeout": {
                    "type": "integer",
                    "description": "Maximum time to wait in seconds (default: 1800 = 30 minutes)",
                    "default": 1800,
                },
            },
            "required": ["owner", "repo", "pr_number"],
        },
    },
    # =========== dbt Diff Tools ===========
    {
        "name": "jirade_run_dbt_diff",
        "description": "Run dbt model diff for a PR. Compiles models using existing dbt, then runs SQL directly on DuckDB with agent-provided fixtures. Compares outputs between base and PR branches. Returns a detailed diff report.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "owner": {
                    "type": "string",
                    "description": "Repository owner (e.g., 'my-org')",
                },
                "repo": {
                    "type": "string",
                    "description": "Repository name (e.g., 'data')",
                },
                "pr_number": {
                    "type": "integer",
                    "description": "GitHub PR number",
                },
                "repo_path": {
                    "type": "string",
                    "description": "Local path to the repository (defaults to current directory)",
                },
                "dbt_project_subdir": {
                    "type": "string",
                    "description": "Subdirectory containing dbt project (default: 'dbt-databricks')",
                },
                "models": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Specific model names to diff. If not provided, auto-detects from changed files.",
                },
                "fixtures": {
                    "type": "object",
                    "description": """Fixtures for testing. Format: {model_name: {table_name: fixture_data}}.
Supports multiple formats:
- SQL statements: ["CREATE TABLE...", "INSERT INTO..."] or "CREATE TABLE...; INSERT INTO..."
- CSV content: "col1,col2\\nval1,val2"
- Use "_sql" as table_name to execute raw SQL setup statements.
Example: {"my_model": {"_sql": ["CREATE SCHEMA dashboard", "CREATE TABLE dashboard.users (...)"], "dashboard.users": "INSERT INTO..."}}""",
                    "additionalProperties": {
                        "type": "object",
                        "additionalProperties": {},
                    },
                },
            },
            "required": ["owner", "repo", "pr_number"],
        },
    },
    {
        "name": "jirade_post_diff_report",
        "description": "Post or update a dbt diff report as a PR comment. If a previous diff report exists, updates it in place.",
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
                    "description": "GitHub PR number",
                },
                "report": {
                    "type": "string",
                    "description": "Markdown report to post",
                },
                "update_existing": {
                    "type": "boolean",
                    "description": "If true, updates existing diff comment instead of creating new (default: true)",
                    "default": True,
                },
            },
            "required": ["owner", "repo", "pr_number", "report"],
        },
    },
    {
        "name": "jirade_run_dbt_ci",
        "description": "Run dbt CI for a PR on Databricks. Drops any existing CI schemas for this PR (clean slate), detects changed models and seeds, loads changed seeds via `dbt seed`, builds modified models +1 dependents in isolated schemas (jirade_ci_{pr_number}_*), compares results against production tables using metadata queries (no raw data exposed), and returns a diff report. CI tables are kept for inspection - use jirade_cleanup_ci to remove them after the PR is merged.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "owner": {
                    "type": "string",
                    "description": "Repository owner (e.g., 'my-org')",
                },
                "repo": {
                    "type": "string",
                    "description": "Repository name (e.g., 'data')",
                },
                "pr_number": {
                    "type": "integer",
                    "description": "GitHub PR number",
                },
                "repo_path": {
                    "type": "string",
                    "description": "Local path to the repository (defaults to current directory)",
                },
                "dbt_project_subdir": {
                    "type": "string",
                    "description": "Subdirectory containing dbt project (default: 'dbt-databricks')",
                },
                "models": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Specific model names to build and diff. If not provided, auto-detects from changed files.",
                },
                "lookback_days": {
                    "type": "integer",
                    "description": "Number of days back for event-time-start (for microbatch models). Default: 3",
                    "default": 3,
                },
                "post_to_pr": {
                    "type": "boolean",
                    "description": "If true, automatically posts the diff report as a PR comment (default: true)",
                    "default": True,
                },
            },
            "required": ["owner", "repo", "pr_number"],
        },
    },
    {
        "name": "jirade_analyze_deprecation",
        "description": "Analyze the impact of deprecating a table or column. Parses dbt manifest.json to find downstream models that reference the table, flagging marts/dims as user-exposed. Returns list of affected model files for agent to verify column-level usage.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "table_name": {
                    "type": "string",
                    "description": "The table name to analyze for deprecation (e.g., 'stg_salesforce__accounts')",
                },
                "column_name": {
                    "type": "string",
                    "description": "Optional column name to check for deprecation impact",
                },
                "repo_path": {
                    "type": "string",
                    "description": "Local path to the repository (defaults to current directory)",
                },
                "dbt_project_subdir": {
                    "type": "string",
                    "description": "Subdirectory containing dbt project (default: 'dbt-databricks')",
                },
            },
            "required": ["table_name"],
        },
    },
    {
        "name": "jirade_cleanup_ci",
        "description": """Clean up CI schemas for a merged PR.

**IMPORTANT: Call this tool after a PR is merged, at the same time as closing the Jira ticket.**

This removes all CI schemas created by jirade_run_dbt_ci for the specified PR (e.g., jirade_ci_3690_*).
CI tables are intentionally kept after CI runs so users can inspect them, but should be cleaned up
once the PR is merged and no longer needed.

Typical workflow:
1. Run jirade_run_dbt_ci to build and diff models
2. PR gets reviewed and merged
3. Call jirade_cleanup_ci to remove CI schemas
4. Close the Jira ticket""",
        "inputSchema": {
            "type": "object",
            "properties": {
                "pr_number": {
                    "type": "integer",
                    "description": "The PR number whose CI schemas should be cleaned up",
                },
            },
            "required": ["pr_number"],
        },
    },
]


def get_tools() -> list[dict[str, Any]]:
    """Return the list of tools for the MCP server."""
    return TOOLS
