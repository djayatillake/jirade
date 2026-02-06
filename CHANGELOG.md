# Changelog

## v0.4.2 - Diff all built models & date-filtered comparisons

### Diff all built models

Previously only changed models were compared against production — downstream models just got a "built successfully" line. Now all built models (changed + downstream) are compared, with the report split into "Changed Models" and "Downstream Models" sections, both with full diff tables (row counts, schema changes, NULL changes).

### Date-filtered comparisons for incremental models

CI builds incremental/microbatch models with `--event-time-start` / `--event-time-end`, so the CI table only has data for the lookback window. Comparing unfiltered `COUNT(*)` against prod's full history is meaningless. Now:

- The manifest is parsed for incremental/microbatch models with `config.event_time`
- `compare_tables` accepts a `date_filter` parameter that applies a WHERE clause to both row counts and NULL counts
- The report shows a calendar emoji and a note explaining the filtered date range

### Skip comparison for time-limited descendants

If a downstream model depends on a time-limited incremental parent, its CI data is inherently incomplete (only reflects the lookback window of upstream data). These models now skip comparison entirely, with the report explaining why: "Upstream model is incremental/microbatch — CI was built with only N days of data, so this downstream model's row counts are not comparable to production."

### Changes

- `jirade/clients/databricks_client.py`: `compare_tables` gains `date_filter` param, `get_null_count` gains `where_clause` param, whitelist regex updated for `NULL AND ...` queries
- `jirade/mcp/handlers/dbt_diff.py`: Expand `models_to_compare` to all built models, parse manifest for incremental configs, walk DAG for time-limited descendants, skip comparison for affected downstream models, extract `_format_model_summary_row` and `_format_model_detail_section` helpers, update report with downstream diffs and skip reasons

## v0.4.1 - Seed support & Jira labeling

### Seed support in CI

`jirade_run_dbt_ci` now detects changed seed files (`.csv`) in PRs. Changed seeds are loaded via `dbt seed` before `dbt run`, so downstream models that `ref()` seeds resolve to the CI version instead of deferring to production. Seed failures are tracked and reported separately.

The `generate_schema_name` macro (in the data repo) was also fixed to handle seeds correctly - seeds use `node.config.database` for catalog resolution instead of parsing `node.name` with `__` delimiters.

### Jira labeling

Tickets transitioned to "Done" via `jirade_transition_issue` are automatically tagged with a `jirade` label. This is non-blocking - if labeling fails, the transition still succeeds.

### Changes

- `jirade/mcp/handlers/dbt_diff.py`: Detect changed seeds, run `dbt seed` step, build seed descendants, report seed results
- `jirade/clients/jira_client.py`: Add `add_label()` method (idempotent via Jira update operation)
- `jirade/mcp/handlers/jira.py`: Tag "jirade" label on Done transitions
- `jirade/mcp/tools.py`: Updated `jirade_run_dbt_ci` description to mention seed support

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
