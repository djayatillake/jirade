# jirade

MCP server that gives Claude Code tools for Jira, GitHub, and dbt CI on Databricks. Also includes a standalone CLI for autonomous ticket processing.

## What it does

jirade exposes 13 tools via the [Model Context Protocol](https://modelcontextprotocol.io/) that let Claude Code:

- **Search and manage Jira tickets** -- query with JQL, read details, add comments, transition status
- **Monitor GitHub PRs** -- list PRs, check CI status, watch until checks pass
- **Run dbt CI on Databricks** -- build models in isolated schemas, compare against production using metadata-only queries, post diff reports to PRs
- **Analyze dbt deprecation impact** -- find downstream models affected by deprecating a table or column

No raw data is ever exposed. The Databricks client enforces a strict SQL whitelist -- only aggregated metadata queries (counts, schemas, NULLs, distributions) are allowed.

## Setup

### Install

```bash
# From source (recommended for development)
git clone https://github.com/djayatillake/jirade.git
cd jirade
poetry install

# Or via pipx
pipx install git+https://github.com/djayatillake/jirade.git
```

### Configure Claude Code

Add jirade as an MCP server in your Claude Code settings (`~/.claude/settings.json` or project `.claude/settings.json`):

```json
{
  "mcpServers": {
    "jirade": {
      "command": "jirade-mcp",
      "env": {}
    }
  }
}
```

If installed via poetry (not pipx), use the full path:

```json
{
  "mcpServers": {
    "jirade": {
      "command": "/path/to/jirade/.venv/bin/jirade-mcp",
      "env": {}
    }
  }
}
```

### Environment variables

**Required for Jira tools:**

```bash
JIRADE_JIRA_OAUTH_CLIENT_ID="your-client-id"
JIRADE_JIRA_OAUTH_CLIENT_SECRET="your-client-secret"
```

**Required for GitHub tools:**

```bash
# Option 1: gh CLI (recommended -- auto-detected, no env var needed)
gh auth login

# Option 2: manual token
JIRADE_GITHUB_TOKEN="ghp_..."
```

**Required for dbt CI tools:**

```bash
JIRADE_DATABRICKS_HOST="dbc-xxxxx.cloud.databricks.com"
JIRADE_DATABRICKS_HTTP_PATH="/sql/1.0/warehouses/abc123"
JIRADE_DATABRICKS_AUTH_TYPE="oauth"           # default, uses Databricks CLI creds
JIRADE_DATABRICKS_CI_CATALOG="development_yourname_metadata"  # catalog for CI schemas
```

**Optional:**

| Variable | Default | Description |
|----------|---------|-------------|
| `JIRADE_DATABRICKS_TOKEN` | -- | Databricks PAT (if `auth_type=token`) |
| `JIRADE_DATABRICKS_CATALOG` | -- | Default catalog for production lookups |
| `JIRADE_DBT_EVENT_TIME_LOOKBACK_DAYS` | `3` | Days of data for incremental CI builds |
| `JIRADE_DBT_CI_SCHEMA_PREFIX` | `jirade_ci` | Prefix for CI schema names |
| `JIRADE_LOG_LEVEL` | `INFO` | Logging level |
| `ANTHROPIC_API_KEY` | -- | Required only for CLI agent mode |
| `JIRADE_CLAUDE_MODEL` | `claude-opus-4-5-20251101` | Model for CLI agent mode |
| `JIRADE_WORKSPACE_DIR` | `/tmp/jirade` | Where repos are cloned (CLI mode) |

### Authenticate

```bash
jirade auth login                    # all services
jirade auth login --service=jira     # just Jira (opens browser for OAuth)
jirade auth login --service=databricks  # validate Databricks connection
jirade health                        # verify everything works
```

## MCP tools

These tools are available to Claude Code when jirade is configured as an MCP server.

### Jira

| Tool | Description |
|------|-------------|
| `jirade_search_jira` | Search issues with JQL (e.g., `project = PROJ AND status = "In Progress"`) |
| `jirade_get_issue` | Get full issue details -- description, status, comments, transitions |
| `jirade_add_comment` | Add a comment to an issue |
| `jirade_transition_issue` | Change issue status (e.g., move to "Done"). Auto-tags with `jirade` label |

### GitHub

| Tool | Description |
|------|-------------|
| `jirade_list_prs` | List PRs for a repository |
| `jirade_get_pr` | Get PR details including reviews and comments |
| `jirade_get_ci_status` | Get CI check status for a PR |
| `jirade_watch_pr` | Poll CI status until all checks pass or fail (default: 30s interval, 30min timeout) |

### dbt

| Tool | Description |
|------|-------------|
| `jirade_run_dbt_ci` | Build models on Databricks in isolated CI schemas, compare against prod, post report to PR |
| `jirade_run_dbt_diff` | Local diff using DuckDB with agent-provided fixtures (no Databricks needed) |
| `jirade_post_diff_report` | Post or update a diff report as a PR comment |
| `jirade_analyze_deprecation` | Find downstream models affected by deprecating a table or column |
| `jirade_cleanup_ci` | Drop CI schemas after a PR is merged |

## How dbt CI works

`jirade_run_dbt_ci` is the main CI tool. When invoked:

1. Checks out the PR branch
2. Detects changed models and seeds from the git diff
3. Loads changed seeds via `dbt seed`
4. Builds modified models +1 dependents in isolated schemas (`jirade_ci_{pr_number}_{catalog}_{schema}`)
5. Uses `--defer --state --favor-state` so upstream models resolve to production
6. Compares **all** built models (changed + downstream) against production using metadata queries
7. For incremental/microbatch models with `event_time`, date-filters comparisons to the CI lookback window
8. Skips comparison for downstream models whose upstream is time-limited (CI data inherently incomplete)
9. Posts a diff report to the PR

`dbt run` and `dbt test` are separate steps so test failures don't skip downstream model builds. If some models fail but others succeed, you still get a report with a "Build Failures" section.

CI tables persist after the run for manual inspection. Use `jirade_cleanup_ci` after the PR is merged.

### What the agent can see on Databricks

The `DatabricksMetadataClient` enforces a strict regex whitelist on every SQL query:

- `DESCRIBE TABLE`, `SHOW COLUMNS` -- column names and types
- `SELECT COUNT(*)` -- row counts (with optional WHERE for date filtering)
- `SELECT COUNT(*) WHERE col IS NULL` -- null counts
- `SELECT COUNT(DISTINCT col)` -- cardinality
- `SELECT col, COUNT(*) GROUP BY col` -- value distributions
- `SELECT MIN/MAX(col)` -- numeric ranges
- `CREATE/DROP SCHEMA`, `DROP TABLE` -- CI lifecycle

Everything else is rejected. No `SELECT *`, no raw rows, no freeform SQL.

### Macros required in your dbt project

Your dbt project needs `generate_schema_name` and `generate_database_name` macros that check the `DBT_JIRADE_CI` environment variable to redirect models into CI schemas.

## CLI

jirade also has a standalone CLI for autonomous ticket processing, powered by Claude.

### Autonomous ticket processing

```bash
# Process a specific Jira ticket (analyzes, implements, creates PR)
jirade process-ticket PROJ-123 --config .jirade.yaml

# Process multiple tickets by status
jirade process --config .jirade.yaml --status="Ready for Agent" --limit=5

# Watch mode -- poll for tickets and auto-close when PRs merge
jirade watch --config .jirade.yaml --interval=60
```

### Interactive REPL

```bash
# Start a chat session with Claude that has access to Jira, GitHub, and Git tools
jirade chat --config .jirade.yaml
```

### Other CLI commands

```bash
jirade list-tickets --config .jirade.yaml           # List Jira tickets
jirade list-tickets --config .jirade.yaml -i         # Interactive selection
jirade list-prs --config .jirade.yaml                # List GitHub PRs
jirade check-pr 123 --config .jirade.yaml            # Check PR status
jirade fix-ci 123 --config .jirade.yaml              # Auto-fix CI failures
jirade health                                         # Test all connections
jirade auth status                                    # Show auth status
jirade config validate .jirade.yaml                   # Validate config
jirade env check --config .jirade.yaml               # Check environment
jirade learn status                                   # Show pending learnings
```

### Repository config

The CLI requires a `.jirade.yaml` config file. Generate one with:

```bash
jirade init
```

## Jira OAuth setup

1. Go to [Atlassian Developer Console](https://developer.atlassian.com/console/myapps/)
2. Create an OAuth 2.0 integration
3. Add Jira API permissions: `read:jira-work`, `write:jira-work`, `read:jira-user`, `offline_access`
4. Set callback URL to `http://localhost:8888/callback`
5. Copy Client ID and Secret to environment variables
6. Run `jirade auth login --service=jira`

## License

MIT
