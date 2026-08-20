# jirade

MCP server that gives Claude Code tools for GitHub and dbt CI on Databricks.

> **v0.10.0**: Jira and Confluence tools were removed. Use the **Atlassian Rovo MCP connector**
> in Claude Code alongside jirade — authenticate it with `/mcp`. No Atlassian OAuth app,
> scopes, or `JIRADE_JIRA_OAUTH_*` env vars are needed any more.

## What it does

jirade exposes tools via the [Model Context Protocol](https://modelcontextprotocol.io/) that let Claude Code:

- **Monitor GitHub PRs** -- list PRs, check CI status, watch until checks pass
- **Run dbt CI on Databricks** -- build models in isolated schemas, compare against production using metadata-only queries, post diff reports to PRs
- **Generate UAT data impact reports** -- run analytical aggregate queries against CI tables and post the results to the GitHub PR (the agent posts the same markdown to the Jira ticket via Rovo)
- **Audit jirade activity** -- pull a quarter's worth of PR data (plus the JQL queries for the agent to run via Rovo) so the agent can write a funnel-style activity report
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

**Atlassian (Jira + Confluence):** no jirade configuration. Enable and authenticate the
**Atlassian Rovo MCP connector** in Claude Code (`/mcp` → "Atlassian Rovo") — it provides
JQL/CQL search, issue reads/comments/transitions, and Confluence page create/update,
acting as the authenticated user.

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
jirade auth login                    # all services (GitHub + Databricks)
jirade auth login --service=databricks  # validate Databricks connection
jirade health                        # verify everything works
```

## MCP tools

These tools are available to Claude Code when jirade is configured as an MCP server.

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
| `jirade_analyze_deprecation` | Find downstream models affected by deprecating a table or column |
| `jirade_generate_schema_docs` | Read model + upstream SQL from manifest for writing lineage-aware schema descriptions |
| `jirade_cleanup_ci` | Drop CI schemas after a PR is merged |
| `jirade_uat_report` | Run analytical aggregate queries against CI tables and post the report to the PR (returns the markdown for the agent to post to Jira via Rovo) |
| `jirade_test_airflow_dag` | Validate an Airflow DAG's SQL by running it in a CI schema and checking idempotency |

### Activity audits

| Tool | Description |
|------|-------------|
| `jirade_activity_report` | Pull the PR data needed for a jirade activity audit, plus the JQL queries for the agent to run via Rovo. Surfaces self-authored PRs, other-author PRs the user reviewed or committed to, and other users running jirade tools (cross-user discovery). Returns structured data — agent writes the narrative each run. Designed for weekly/monthly cadence. |

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

```bash
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

## License

MIT
