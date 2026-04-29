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
    {
        "name": "jirade_log_adhoc_work",
        "description": (
            "Create a completed ticket in the current AENG sprint for ad-hoc work that was done without a pre-existing ticket. "
            "Automatically finds the active sprint, creates the task, marks it Done, and applies the 'jirade' label. "
            "Use this at the end of any ad-hoc request where no Jira ticket existed upfront."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "Short title for the ticket (e.g., 'Expand IQ score to all committed accounts')",
                },
                "description": {
                    "type": "string",
                    "description": "What was done and why — include the requester, the change made, and the outcome",
                },
                "pr_url": {
                    "type": "string",
                    "description": "GitHub PR URL to link in the ticket description (optional)",
                },
            },
            "required": ["summary", "description"],
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
        "description": "Watch a PR's CI status until all checks complete. Polls every interval seconds and returns when all checks pass, any check fails, or timeout is reached. On failure, automatically checks whether the failing checks also fail on the base branch — if so, sets rebase_suggested=true and rebase_reason explaining that rebasing may pick up fixes from the base branch.",
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
    # =========== dbt CI Tools ===========
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
        "name": "jirade_generate_schema_docs",
        "description": """Generate intelligent schema documentation context for dbt models.

Reads the model's SQL, its upstream dependencies' SQL, and manifest metadata to provide
full lineage context. Use this before writing schema.yml entries so descriptions explain
derivation logic and business meaning — not just literal column names.

Returns the model SQL, upstream model SQL, existing column definitions, and guidance
for writing descriptions that trace column lineage through CTEs and refs.""",
        "inputSchema": {
            "type": "object",
            "properties": {
                "models": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of model names to generate documentation context for (e.g., ['rpt_customer', 'dim_application'])",
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
            "required": ["models"],
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
    # =========== UAT Report Tools ===========
    {
        "name": "jirade_uat_report",
        "description": (
            "Generate a stakeholder-facing UAT data impact report from CI tables and post it to both "
            "the Jira ticket and the GitHub PR. Executes analytical aggregate queries against existing "
            "CI tables (built by jirade_run_dbt_ci) and formats the results as readable tables.\n\n"
            "The caller provides SQL queries that compare CI data (e.g., new vs old columns, value "
            "distributions, time deltas). All queries must reference CI tables only "
            "(catalog.jirade_ci_{pr_number}_* schema). No raw data is returned — only aggregates.\n\n"
            "Typical workflow:\n"
            "1. Run jirade_run_dbt_ci to build CI tables\n"
            "2. Call jirade_uat_report with analytical queries and a description\n"
            "3. Report is posted to both the PR and the linked Jira ticket"
        ),
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
                    "description": "GitHub PR number (CI tables must already exist)",
                },
                "description": {
                    "type": "string",
                    "description": (
                        "Natural language description of the change and its impact, written for "
                        "non-engineer stakeholders. This becomes the report header."
                    ),
                },
                "queries": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "label": {
                                "type": "string",
                                "description": "Display label for this result section (e.g., 'Field Population', 'Time Delta Analysis')",
                            },
                            "sql": {
                                "type": "string",
                                "description": "SQL query to execute. Must reference only CI tables (fully qualified: catalog.jirade_ci_*_schema.table). Only aggregate queries allowed.",
                            },
                        },
                        "required": ["label", "sql"],
                    },
                    "description": "List of analytical queries to execute. Each produces a labeled table in the report.",
                },
                "allowed_prod_tables": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Production tables the queries are allowed to reference for before/after "
                        "comparisons (e.g., ['reverse_etl.salesforce.contacts', 'staging.dashboard.users']). "
                        "Queries can always reference CI tables; this parameter whitelists additional "
                        "production tables needed for the analysis."
                    ),
                },
                "jira_ticket_key": {
                    "type": "string",
                    "description": "Jira ticket key (e.g., 'AENG-1937'). Auto-detected from PR title/branch if not provided.",
                },
            },
            "required": ["owner", "repo", "pr_number", "description", "queries"],
        },
    },
    # =========== Airflow Tools ===========
    {
        "name": "jirade_test_airflow_dag",
        "description": """Test Airflow DAG SQL statements against Databricks.

Parses an Airflow DAG file, extracts SQL from CustomDatabricksSqlOperator tasks,
rewrites catalog/schema references to target a CI schema, executes each statement,
and validates results including row counts and idempotency. Cleans up after testing.

This validates that CREATE TABLE, INSERT INTO, and other DDL/DML statements work
correctly before deploying the DAG to production Airflow.""",
        "inputSchema": {
            "type": "object",
            "properties": {
                "dag_path": {
                    "type": "string",
                    "description": "Absolute path to the Airflow DAG Python file to test",
                },
                "ci_schema": {
                    "type": "string",
                    "description": "Name for the CI test schema (default: 'jirade_airflow_test')",
                    "default": "jirade_airflow_test",
                },
                "cleanup": {
                    "type": "boolean",
                    "description": "Whether to drop the CI schema after testing (default: true)",
                    "default": True,
                },
                "test_idempotency": {
                    "type": "boolean",
                    "description": "Whether to re-run INSERT statements to verify idempotency (default: true)",
                    "default": True,
                },
                "post_to_pr": {
                    "type": "boolean",
                    "description": "If true, posts the test report as a comment on the specified PR (default: false)",
                    "default": False,
                },
                "owner": {
                    "type": "string",
                    "description": "Repository owner for PR comment posting (e.g., 'algolia')",
                },
                "repo": {
                    "type": "string",
                    "description": "Repository name for PR comment posting (e.g., 'data')",
                },
                "pr_number": {
                    "type": "integer",
                    "description": "PR number to post the test report to",
                },
            },
            "required": ["dag_path"],
        },
    },
    # =========== Confluence Tools ===========
    {
        "name": "jirade_publish_confluence_page",
        "description": (
            "Create or update a Confluence page from markdown content. Idempotent — if a page "
            "with the same title already exists in the space (and same parent if given), it's "
            "updated; otherwise a new page is created. Markdown is converted to Confluence "
            "storage format inline (supports headings, paragraphs, lists, GFM tables, fenced "
            "code blocks, and inline formatting).\n\n"
            "Requires Atlassian OAuth scopes: read:confluence-content.all, write:confluence-content. "
            "These were added in jirade v0.6.0 — re-run 'jirade auth login --service=jira' if "
            "you authorized before that."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "space_key": {
                    "type": "string",
                    "description": "Confluence space key (e.g. 'AENG' for the AENG space).",
                },
                "title": {
                    "type": "string",
                    "description": "Page title. Acts as the upsert key — same title in the same space updates the existing page.",
                },
                "body_markdown": {
                    "type": "string",
                    "description": "Page content in markdown. Headings, lists, tables, and code blocks all carry through.",
                },
                "parent_title": {
                    "type": "string",
                    "description": "Title of an existing parent page to nest under. Mutually exclusive with parent_id.",
                },
                "parent_id": {
                    "type": "string",
                    "description": "Parent page ID (alternative to parent_title). Omit to create at space root.",
                },
            },
            "required": ["space_key", "title", "body_markdown"],
        },
    },
    {
        "name": "jirade_get_confluence_page",
        "description": "Fetch a Confluence page by ID, or by space + title. Returns title, version, body in storage format, and the public URL.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "page_id": {"type": "string", "description": "Page ID. Use this OR (space_key, title)."},
                "space_key": {"type": "string", "description": "Space key, used with title."},
                "title": {"type": "string", "description": "Exact page title, used with space_key."},
            },
        },
    },
    {
        "name": "jirade_search_confluence",
        "description": "Search Confluence content using CQL (Confluence Query Language). Example: 'space = AENG AND title ~ \"Jirade\"'.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "cql": {
                    "type": "string",
                    "description": "CQL query string (e.g. 'space = AENG AND type = page AND title ~ \"audit\"').",
                },
                "limit": {"type": "integer", "description": "Max results (default 25)", "default": 25},
            },
            "required": ["cql"],
        },
    },
    # =========== Activity Report Data Pull ===========
    {
        "name": "jirade_activity_report",
        "description": (
            "Pull all the raw data needed to write a jirade activity report — PRs the user "
            "authored, reviewed, or commented on; PRs by OTHER users running jirade tools (cross-user "
            "discovery via 'jirade' text search); Jira tickets matching the jirade label, the user's "
            "assignment history, or jirade-signature comments ('jirade grooming', 'via Claude Code', "
            "'Implemented by Jirade'). For non-self-authored PRs it also pulls reviews + commits so "
            "the agent can distinguish 'reviewed only' from 'reviewed + cleanup commit pushed'.\n\n"
            "This tool is intentionally a data collector, not a classifier or renderer. The agent "
            "writes the report narrative each run so its shape can evolve. Publish the resulting "
            "markdown via jirade_publish_confluence_page."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "since": {
                    "type": "string",
                    "description": "ISO date (YYYY-MM-DD) — start of the audit window. Defaults to 90 days ago.",
                },
                "repo": {
                    "type": "string",
                    "description": "GitHub repo slug (default: 'algolia/data').",
                    "default": "algolia/data",
                },
                "projects": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Jira project keys to scan (default: ['AENG', 'DATASD', 'DATA']).",
                },
                "user": {
                    "type": "string",
                    "description": "GitHub username to audit. Defaults to the authenticated `gh` user.",
                },
            },
        },
    },
]


def get_tools() -> list[dict[str, Any]]:
    """Return the list of tools for the MCP server."""
    return TOOLS
