# Changelog

## v0.7.2 - Attribute EXCEPT row diffs to specific columns

The whole-row `EXCEPT` comparison in CI tells you *that* rows differ between
prod and CI, but not *which* columns moved. When you change a single column's
logic, the row diff lights up for every changed row and gives no signal about
whether the change stayed contained or leaked into other columns.

Requested by Afraz: when changing a column, run `SELECT * EXCEPT(col) … EXCEPT
SELECT * EXCEPT(col) …` both ways to confirm nothing *else* changed. This
release does that idea one better — instead of requiring the reviewer to name
the changed column up front, `compare_tables` now **attributes the row diff to
the columns that actually changed**:

- When the whole-row `EXCEPT` finds differing rows **and the row counts match**
  (a value-only change), each comparable column is probed on its own with the
  existing single-column `EXCEPT` count — ci-vs-prod first (catches added
  values), then prod-vs-ci (catches a value set that strictly shrank). The
  columns that differ are reported as `changed_columns`.
- This keeps full coverage of the changed column (it appears in the list,
  confirming the intended change landed) **and** surfaces any collateral
  changes to other columns — the actual question Afraz was after.
- When row counts differ, attribution is skipped with a note (added/removed
  rows make per-column value-set attribution ambiguous).
- Probing is capped by `JIRADE_DBT_CHANGED_COLUMN_MAX_PROBES` (default 100,
  `0` disables); if a wide table exceeds the cap the report says how many
  columns went unchecked. No extra queries run when rows match.

No new query shape and no new whitelist entry — each probe reuses the
already-whitelisted single-column `EXCEPT COUNT(*)` query, so the
metadata-only / no-raw-rows security model is unchanged.

The PR diff report grows a **"Columns with changed values"** line under the
EXCEPT section, and the summary row appends `· N cols` to the row-diff cell.

### Files

- `jirade/config.py`: new `dbt_changed_column_max_probes` setting
- `jirade/clients/databricks_client.py`: new `_attribute_except_diff()` helper; `compare_tables()` takes `max_column_probes` and populates `changed_columns`
- `jirade/mcp/handlers/dbt_diff.py`: passes the setting through; summary row + detail section render the changed columns
- `tests/test_changed_column_attribution.py`: whitelist check for the reused single-column EXCEPT query, plus attribution behaviour (changed-column flagging, row-count-mismatch skip, no-diff no-op, disabled, probe-limit truncation)

## v0.7.1 - Fix metric_view lookup key

v0.7.0 stored detected metric views in a dict keyed by the manifest's full
prefixed name (e.g. `mart__sales__mv_opportunity`), but the comparison loop
looked them up by `model_short_name` (the last `__`-separated segment —
e.g. `mv_opportunity`). The dict lookup always missed, the metric_view
branch never fired, and metric-view models silently fell through to the
regular table-comparison path — which then fails on the metric view's
`MEASURE()`-only columns because `COUNT(*) WHERE measure IS NULL` errors
with `MISSING_ATTRIBUTES.RESOLVED_ATTRIBUTE_MISSING_FROM_INPUT`.

Fix: look up `metric_view_models[model]` (full prefixed name, which matches
both the manifest `name` and the model identifier produced by the dbt build
step from `run_results.json`). Discovered by running v0.7.0 against
algolia/data#4203, Jeremy's metric-view fix PR — the diff report came back
with a NEW model error instead of the expected smoke-test section.

The same lookup mismatch exists in the older `model_short_name in model_configs`
check for incremental/microbatch date filtering — left as-is for now because
its only consequence is "no date filter applied" (incrementals still build
correctly). Worth a follow-up fix but not blocking.

## v0.7.0 - UC Metric View smoke testing in `dbt_run_dbt_ci`

dbt-databricks 1.12 (May 2026) shipped `materialized='metric_view'`, but jirade's CI flow only knew about `table` / `view` / `incremental` materializations. Running CI on a metric_view PR would either crash on the table-comparison path or — worse — silently report `:white_check_mark:` for models that fail at deploy time. The class of bug `dbt compile` misses (YAML body syntax, column refs that don't resolve, etc.) only surfaces when the view is actually queried.

This release adds metric-view awareness to the diff pipeline:

- Manifest pass picks up `materialized: metric_view` models alongside the existing incremental/microbatch detection. Measure names are extracted from the model's compiled YAML body and stashed per-model.
- In the comparison loop, metric views route to a new `smoke_query_metric_view()` path on `DatabricksMetadataClient` instead of `compare_tables()`. For each declared measure, the client runs `SELECT MEASURE(<m>) FROM <ci_view>` and records pass/fail.
- The `SELECT MEASURE(<id>) [AS <id>] FROM <fqn>` shape was added to `ALLOWED_PATTERNS` — bare aggregates only, no WHERE clause, no raw columns. Matches the rest of the whitelist's security model.
- The PR report grows a "Metric View Smoke Test" section per metric view: a probe table showing each measure with `:white_check_mark:` / `:x:` and the error text when a probe fails. The summary row uses `n/m measures :test_tube:` instead of row-count diffs.

This catches the exact two failures from PR #4203 in algolia/data (Jeremy's fix to `mart__sales__mv_opportunity`): the SQL-style `--` comment in the YAML body, and the `SUM(arr_expansion)` measure referencing a column that doesn't exist on `fact_opportunity`.

### Files

- `jirade/clients/databricks_client.py`: added `MEASURE()` pattern to `ALLOWED_PATTERNS`; new `smoke_query_metric_view()` method
- `jirade/mcp/handlers/dbt_diff.py`: new `_extract_metric_view_measures()` helper; manifest parsing collects `metric_view_models`; comparison loop branches on `is_metric_view`; summary row + detail section formatters render the smoke test results
- `tests/test_metric_view_smoke.py`: covers the whitelist regex and the YAML-extraction helper (well-formed, empty, malformed, no-measures, missing-name cases)

## v0.6.2 - Add granular Confluence OAuth scopes for v2 API

The v0.6.1 migration to Confluence REST API v2 surfaced an Atlassian quirk: v2 endpoints reject classic scopes with `401 Unauthorized — scope does not match`. v2 was introduced with a parallel "granular" scope naming convention (`read:page:confluence` instead of `read:confluence-content.all`, etc.) and the classic scopes are not accepted on v2 endpoints.

Added the three granular scopes the v2 client needs to `JiraOAuth.SCOPES`:

- `read:space:confluence` — `GET /wiki/api/v2/spaces`
- `read:page:confluence` — `GET /wiki/api/v2/pages` (find by title and read by ID; parent-page traversal happens through page IDs which this scope covers)
- `write:page:confluence` — `POST/PUT /wiki/api/v2/pages` (create + update + parent nesting)

The classic scopes are still in the SCOPES list because `search:confluence` and the CQL search endpoint at `/wiki/rest/api/search` haven't been migrated to v2 yet. Both sets coexist on a single token.

Existing users must add the three granular scopes to their OAuth app at https://developer.atlassian.com/console/myapps and re-run `jirade auth login --service=jira`. README, login flow error message, and CHANGELOG updated to walk through both sets.

### Files

- `jirade/auth/jira_auth.py`: SCOPES list extended with the four granular scopes; comments updated to explain classic vs granular bifurcation
- `jirade/auth/manager.py`: console output during login error walks through both scope sets
- `README.md`: scope table now shows classic + granular sections

## v0.6.1 - Migrate Confluence client to REST API v2

The Atlassian Confluence Cloud REST API v1 endpoints used in v0.6.0 (`/wiki/rest/api/content`) were retired during rollout and now return 410 Gone. Migrated to v2 (`/wiki/api/v2/`):

- Added `get_space_id(space_key)` with caching — v2 uses numeric `space-id` instead of string `spaceKey`
- `find_page_by_title` / `create_page` / `update_page` / `get_page` rewritten against v2 payload shapes (`spaceId`, `parentId`, `body.value`, `body.representation`, `version.number`)
- `_page_url` updated for v2 `_links.webui` shape
- CQL search retained at v1 (`/wiki/rest/api/search`) — that endpoint is the one v1 path that hasn't been migrated yet

### Files

- `jirade/clients/confluence_client.py`: rewritten against v2 API
- `jirade/mcp/handlers/confluence.py`: `_page_url` helper updated for v2 link shape

## v0.6.0 - Confluence support + activity report tool

### Confluence native integration

The Atlassian OAuth flow now requests Confluence scopes alongside Jira:

- `read:confluence-content.all` — read page bodies
- `read:confluence-content.summary` — find pages by title
- `read:confluence-space.summary` — resolve space keys
- `write:confluence-content` — create/update pages
- `search:confluence` — CQL endpoint (required for `jirade_search_confluence`)

A single OAuth access token is reused for both Jira and Confluence APIs (Atlassian Cloud issues one token per cloud_id). Existing users must re-run `jirade auth login --service=jira` to pick up the new scopes — `jirade auth status` will show a `⚠ Authenticated (Jira only — re-login for Confluence)` warning until that's done. The auth manager also detects `invalid_scope` errors and points at the developer console to add the scopes to the OAuth app.

### New MCP tools

Three Confluence tools and one activity-report tool:

| Tool | What it does |
|------|--------------|
| `jirade_publish_confluence_page` | Create-or-update a page from markdown. Idempotent on (space_key, title). Markdown → Confluence storage format inline (headings, lists, GFM tables, fenced code, inline formatting). Supports `parent_title` or `parent_id` for nesting. |
| `jirade_get_confluence_page` | Fetch a page by ID or by space+title. Returns body in storage format and the public URL. |
| `jirade_search_confluence` | CQL search (e.g. `space = AENG AND title ~ "audit"`). |
| `jirade_activity_report` | Pulls raw PR + ticket data for jirade activity audits. Surfaces self-authored PRs, other-author PRs the user reviewed/committed-to, other users running jirade tools (cross-user discovery via `'jirade'` text search), and jirade-signature Jira tickets. **Returns structured data, not a rendered report** — the calling agent writes the narrative each run so the report shape can evolve. Designed for weekly/monthly cadence. |

### Files

- `jirade/auth/jira_auth.py`: extend `JiraOAuth.SCOPES` with three Confluence scopes; class is now Atlassian-wide (Jira + Confluence) but keeps the `JiraOAuth` name for backwards compatibility with the token store. Add `has_confluence_scopes()` JWT-decode helper.
- `jirade/auth/manager.py`: login error path detects scope failures and points users at the developer console; status output shows Jira-only vs Jira+Confluence.
- `jirade/clients/confluence_client.py`: new — async REST client over `httpx`, `find_page_by_title`, `get_page`, `create_page`, `update_page`, `upsert_page`, `search_cql`, plus a self-contained `markdown_to_storage()` converter.
- `jirade/mcp/handlers/confluence.py`: new — wraps `ConfluenceClient` for the three Confluence tools, raises a clear `RuntimeError` if the token lacks Confluence scopes.
- `jirade/mcp/handlers/activity_report.py`: new — shells to `gh search prs` for the three GitHub queries (`--author`, `--involves`, `'jirade'`), enriches non-self-authored PRs with reviews+commits via `gh api`, queries Jira via `JiraClient` with five JQL angles (label, assignment, three signature-comment phrases), returns deduped + structured data.
- `jirade/mcp/tools.py`: register the four new tool definitions.
- `jirade/mcp/handlers/__init__.py`: dispatch the new tool prefixes.

### Onboarding changes

`jirade auth login` now mentions Confluence in console output. The error path for OAuth scope mismatches walks the user through adding scopes in the Atlassian developer console. README updated with Confluence scope list and re-auth instructions.

## v0.4.3 - Security hardening: remove DuckDB diff, tighten Databricks constraints

### Remove DuckDB diff path

The DuckDB-based local diff tools (`jirade_run_dbt_diff`, `jirade_post_diff_report`) have been removed. These were superseded by Databricks CI (`jirade_run_dbt_ci`) since v0.4.0. Removing ~1,200 lines eliminates value distribution leaks and reduces attack surface.

### Tighten Databricks query whitelist

- Removed `GROUP BY` pattern — was only used by `get_value_distribution()` which leaked actual data values
- Removed `MIN/MAX` pattern — had no callers

### Remove unsafe methods

- Deleted `execute_unsafe_query()` — zero callers, eliminated accidental bypass risk
- Deleted `get_value_distribution()` — zero callers, was the primary data leak vector

### Identifier validation

Added `_validate_identifier()` that rejects SQL identifiers containing anything other than `[a-zA-Z0-9_.`"]`. Applied to all methods that interpolate identifiers into SQL via f-strings: `get_table_schema`, `get_row_count`, `get_null_count`, `get_distinct_count`, `create_ci_schema`, `drop_ci_schema`, `list_tables_in_schema`, `drop_table`, and `compare_tables` date filter column.

### Changes

- `jirade/clients/databricks_client.py`: Remove GROUP BY/MIN/MAX whitelist patterns, delete `execute_unsafe_query()` and `get_value_distribution()`, add `_validate_identifier()` with enforcement in all f-string SQL methods
- `jirade/mcp/handlers/dbt_diff.py`: Remove `DbtDiffRunner` class, `format_diff_report()`, `run_dbt_diff()`, and handler dispatch branches (~1,240 lines)
- `jirade/mcp/tools.py`: Remove `jirade_run_dbt_diff` and `jirade_post_diff_report` tool definitions
- `jirade/mcp/server.py`: Update workflow instructions
- `README.md`: Remove deleted tools from table

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

The `generate_schema_name` macro (in the target dbt project) was also fixed to handle seeds correctly - seeds use `node.config.database` for catalog resolution instead of parsing `node.name` with `__` delimiters.

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
