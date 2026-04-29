# Changelog

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
