# Changelog

## v0.4.0 - Local Databricks CI

### Breaking Changes

dbt Cloud integration removed entirely. The following MCP tools are gone: `jirade_dbt_list_jobs`, `jirade_dbt_trigger_run`, `jirade_dbt_get_run`, `jirade_dbt_trigger_ci_for_pr`.

### Why

dbt Cloud CI didn't play well with our develop > master flow - deferral was awkward and unreliable. After building the local DuckDB diff tool, it was clear we could just run CI ourselves against Databricks and get better results with less complexity.

### What replaced it

**`jirade_run_dbt_ci`** runs dbt locally against Databricks. It checks out the PR branch, builds modified models +1 dependents into isolated CI schemas (`jirade_ci_{pr_number}_{catalog}_{schema}`) in your dev catalog, compares CI tables against production using metadata queries, and posts a diff report to the PR. Uses `--defer --state --favor-state` so upstream models resolve to prod.

`dbt run` and `dbt test` are separate steps so test failures don't skip downstream model builds. If some models fail but others succeed, you still get a report with a "Build Failures" section.

**`jirade_cleanup_ci`** drops CI schemas after a PR is merged. Tables are kept after CI runs for inspection.

**`jirade_analyze_deprecation`** parses `manifest.json` to find downstream models affected by deprecating a table or column, flagging marts/dims as user-exposed.

### What the agent can see on Databricks

The `DatabricksMetadataClient` enforces a strict regex whitelist on every SQL query. Allowed:

- `DESCRIBE TABLE`, `SHOW COLUMNS` - column names and types
- `SELECT COUNT(*)` - row counts
- `SELECT COUNT(*) WHERE col IS NULL` - null counts
- `SELECT COUNT(DISTINCT col)` - cardinality
- `SELECT col, COUNT(*) GROUP BY col` - value distributions
- `SELECT MIN/MAX(col)` - numeric ranges
- `CREATE/DROP SCHEMA`, `DROP TABLE` - CI lifecycle

Everything else is rejected. No `SELECT *`, no raw rows, no freeform SQL. The agent never sees actual data - only aggregated metadata. The dbt build itself runs through dbt's adapter (unrestricted), but the comparison step uses only whitelisted queries.

### Setup

Set these environment variables:

```bash
JIRADE_DATABRICKS_HOST=dbc-xxxxx.cloud.databricks.com
JIRADE_DATABRICKS_HTTP_PATH=/sql/1.0/warehouses/abc123
JIRADE_DATABRICKS_AUTH_TYPE=oauth  # default, uses existing Databricks CLI creds
JIRADE_DATABRICKS_CI_CATALOG=development_yourname_metadata  # your dev catalog
```

Then validate with `jirade auth login --service=databricks` and `jirade health`.

### Other changes

- dbt build progress streams line-by-line via MCP progress notifications
- Test failure names enriched from `manifest.json` - shows `test_type(model.column)` instead of hash IDs
- Git branch checkout ensures CI builds with the PR's code regardless of your current branch
- CLI auth/health updated to use Databricks SQL client with OAuth support
