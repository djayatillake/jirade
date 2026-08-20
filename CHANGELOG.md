# Changelog

## v0.11.1 - Zoom bot removed

- **Deleted the Zoom meeting bot** (`jirade/zoom_bot/`, the `jirade zoom`
  subcommands, the `jirade-zoom` entry point, and the Recall.ai integration).
  It was the last feature that genuinely required an `ANTHROPIC_API_KEY`.
- Dropped now-unused dependencies: `fastapi`, `uvicorn`.
- Fixed the `jirade health` Anthropic probe to use a current model
  (`claude-haiku-4-5-20251001` — the retired `claude-3-haiku-20240307`
  404'd).

jirade's integration surface is now: **GitHub + Databricks** (required),
**Anthropic API** (optional, advisor auto-suggestions only). Atlassian is
handled by the Rovo MCP connector in Claude Code.

## v0.11.0 - Anthropic API key now optional; SDK agent loop removed

Nobody runs jirade with its own Anthropic API key — Claude Code is the
harness. This release removes everything that required one:

- **Deleted the standalone SDK agent** (`jirade/agent.py`) and the `check-pr` /
  `fix-ci` CLI commands built on it. The MCP tools (`jirade_get_pr`,
  `jirade_watch_pr`, `jirade_run_dbt_ci`) cover those flows from Claude Code.
- **Deleted the learning capture module** (`learning/capture.py`) — it only ran
  inside the SDK agent loop. `jirade learn status/publish/list` still work on
  existing workspace learnings.
- **Advisors degrade gracefully without a key**: `jirade_advise_permissions_for_pr`
  and `jirade_advise_tags_for_pr` still call the Anthropic API to auto-suggest
  classifications/tags for unclassified tables *when a key is set*; without one
  they leave those rows as needs_llm/needs_suggestion for the calling agent to
  fill in.
- `init` / `health` / `config show` treat the key as optional.
- The zoom bot is the one feature that still genuinely needs
  `ANTHROPIC_API_KEY` (it answers in meetings without a Claude Code session);
  `jirade zoom serve` checks for it at startup.

True integration dependencies after v0.10.0 + v0.11.0: **GitHub** (gh CLI or
PAT) and **Databricks** (CLI OAuth or PAT + CI catalog). Atlassian is handled
by the Rovo MCP connector in Claude Code. Anthropic API + Recall.ai are
optional (zoom bot / advisor auto-suggestions).

## v0.10.0 - Atlassian integration removed — use the Rovo MCP connector

**Breaking: jirade no longer talks to Jira or Confluence.** If you upgrade,
enable and authenticate the **Atlassian Rovo MCP connector** in Claude Code
(`/mcp` → "Atlassian Rovo"). It replaces everything jirade's own integration
did — JQL/CQL search, issue reads, comments, transitions, issue creation,
Confluence page create/update — acting as you, with no OAuth app to set up.

Removed:

- **MCP tools**: `jirade_search_jira`, `jirade_get_issue`, `jirade_add_comment`,
  `jirade_transition_issue`, `jirade_log_adhoc_work`,
  `jirade_publish_confluence_page`, `jirade_get_confluence_page`,
  `jirade_search_confluence`. Use the Rovo equivalents
  (`searchJiraIssuesUsingJql`, `getJiraIssue`, `addCommentToJiraIssue`,
  `transitionJiraIssue`, `createJiraIssue`, `searchConfluenceUsingCql`,
  `getConfluencePage`, `createConfluencePage`/`updateConfluencePage`).
- **Atlassian clients + OAuth**: `jira_client.py`, `confluence_client.py`,
  `jira_auth.py` — including the 17-scope bring-your-own OAuth app,
  `JIRADE_JIRA_OAUTH_CLIENT_ID`/`_SECRET`, the localhost:8888 login flow, and
  ~390 lines of hand-rolled markdown→ADF/storage-format conversion (Rovo
  accepts markdown directly).
- **CLI commands built on the Jira API**: `list-tickets`, `process`,
  `process-ticket`, `watch`, `serve` (Jira webhook server), and `chat`
  (the interactive REPL — unused, and its Jira half was already broken).
  `list-prs`, `check-pr`, `fix-ci`, `init`, `health`, `auth`, `config`,
  `learn`, `env`, and `zoom` remain.

Changed:

- `jirade_uat_report` no longer dual-posts to Jira. It posts to the PR and
  returns the report markdown plus a `next_step` telling the agent to post it
  to the ticket via Rovo (`addCommentToJiraIssue`, contentFormat=markdown).
- `jirade_activity_report` no longer queries Jira. It returns the GitHub data
  plus `jira_jql_to_run` — the same five provenance JQL queries, for the agent
  to run via Rovo and dedupe by ticket key.
- Closing tickets / applying the `jirade` label after a merge is now an
  explicit agent step via Rovo instead of a jirade side-effect.
- `jirade init` no longer prompts for Jira auth, boards, or agent trigger
  statuses; `.jirade.yaml` keeps `jira.project_key` (still used to extract
  ticket keys from PR titles/branches).

Also deleted along the way: the dead `page_url()` duplicate, the
`get_boards()` call that never existed on the client, the REPL Jira tools
that called nonexistent methods, and `jirade_get_issue`'s `customfield_*`
field selector that the Jira v3 API never supported.

## v0.9.10 - REST path: handle DDL statements without result manifests

- `_execute_rest_query` no longer KeyErrors on statements that return no
  result manifest (DROP/CREATE SCHEMA in jirade_cleanup_ci broke when REST
  became the default in v0.9.8). Empty row set returned instead.

## v0.9.9 - NULL-probe cap for modified-model comparison

- compare_tables skips per-column NULL probing when the prod table exceeds
  JIRADE_DBT_COLUMN_STATS_MAX_ROWS (same 10M default as new-table stats).
  Each probe full-scans BOTH tables — on a wide billions-row fact that is
  ~90s x 10 columns x 2 sides per model (observed live on
  fact_search_aggregates_ssot). Row count + schema diff still run;
  `null_probes_skipped: true` marks the result.

## v0.9.8 - Laptop-network hardening: no silent hangs by construction

Running CI from a laptop means long-lived connections die silently
(NAT/idle timeouts). Three layers so silence is structurally impossible:

- **REST metadata is now the default** (JIRADE_METADATA_REST=0 opts back
  into thrift). Stateless submit/poll with a hard per-query deadline.
- **Comparison wall-clock budget** (JIRADE_DBT_COMPARE_BUDGET_MINUTES,
  default 60): on breach the remaining models report comparison_skipped and
  the diff report still posts — partial report beats no report.
- **dbt temp profile hardening**: connect_retries=5, connect_timeout=60,
  connection_parameters.socket_timeout=900 — a dead thrift socket in the
  build phase becomes a bounded retry instead of a zombie.

## v0.9.7 - REST metadata execution path (thrift zombie fix)

- `JIRADE_METADATA_REST=1` routes execute_metadata_query through the SQL
  statement REST API (submit + poll by statement id) instead of the thrift
  connector. The thrift long-poll loses operation handles on warehouse
  suspend / retry-ceiling breach, leaving CI zombie-polling dead queries
  (observed 4x in one day); the stateless REST path survives all of it.
  Bearer token from the configured PAT or the Databricks CLI's cached OAuth
  (non-interactive — fails fast instead of opening a browser). Numeric
  result values coerced to int/float to match thrift behavior.

## v0.9.6 - JIRADE_DBT_VARS passthrough

- `JIRADE_DBT_VARS` (JSON) is passed through to `dbt run --vars` in CI. Lets
  CI shrink var-parameterized scan windows — e.g. a rolling-30d event
  pre-agg tagged databricks_compute='high' that times out a small CI
  warehouse at full width builds a 3-day slice instead. Prod uses defaults.

## v0.9.5 - Schema-only comparison for view materializations

- CI comparison no longer runs COUNT(*)/column stats/EXCEPT against
  view-materialized models — a view is a pass-through, so those queries scan
  its upstream sources (observed: an hour-long COUNT(*) on a staging view
  over raw Segment events that zombied the CI client). Views now get a
  schema-only diff (columns added/removed vs prod); row-level diff signal is
  carried by the table-materialized models downstream.

## v0.9.4 - Cap per-column stats on large tables

- `get_table_metadata` skips per-column NULL/distinct stats when row_count
  exceeds `JIRADE_DBT_COLUMN_STATS_MAX_ROWS` (default 10M) — each column
  costs 2 full scans, which reliably times out a 2X-Small warehouse on big
  new facts (observed: 1,054 stat queries in one CI run, cascading into the
  900s thrift retry ceiling, warehouse auto-suspend, and zombie CI clients).
  Report shows schema + row count with `column_stats_skipped: true`.

## v0.9.3 - CI exclude_models escape hatch

- New `exclude_models` param on `jirade_run_dbt_ci`: passed through to
  `dbt run --exclude` and noted prominently in the diff report ("their
  downstream diffs were NOT exercised"). Escape hatch for pathological
  +1-downstream bystanders — e.g. a 41 GB fact that is a direct dependent of
  dim_account but cannot be affected by the PR's column addition, and whose
  rebuild times out CI on a 2X-Small warehouse.

## v0.9.2 - Zero-scan metric-view smoke tests + dimension coverage

CI smoke tests for UC metric views are now near-instant and cover more.

- **Zero-scan probes.** The combined probe now carries `WHERE 1=0` — column
  resolution happens at plan time, so every measure/dimension ref is
  validated without scanning the underlying table. Previously each probe was
  an unbounded full scan (minutes on a large snapshot fact); now seconds.
  Verified live against `mart.sales.mv_accounts`.
- **Dimensions are probed too.** A broken dimension expr also only fails at
  query time; the combined probe now selects every declared dimension
  alongside `MEASURE()` calls with `GROUP BY ALL`. Probe results carry
  `field` + `kind` (measure|dimension) instead of `measure`.
- **Whitelist tightened, not loosened.** Dimension probes REQUIRE the
  constant-false predicate — a bare `SELECT dim FROM mv` stays blocked, so
  the no-raw-data guarantee holds. MEASURE()-only shape unchanged (aggregate,
  safe unfiltered) for backward compatibility.
- Per-field fallback attribution (on combined-probe failure) also uses
  zero-scan queries.
- Trade-off documented: data-dependent runtime errors (div/0, casts) are no
  longer incidentally caught by the smoke test — that class belongs to data
  tests, not deploy-blocking column-resolution checks.

## v0.9.1 - Advisor robustness: empty-scope + base-branch config

Bug fixes to both advisors, found dry-running v0.9.0 against live PRs.

- **Empty in-scope short-circuit.** A PR with no in-scope models now renders the
  friendly "nothing to do" comment *before* loading any config — previously a PR
  with zero mart/analytics models still tried to load the allowlist / governance
  file and failed if it was absent.
- **Base-branch config fallback.** Global governance config
  (`governed_tags.yaml`, `governance_state.yaml`, `capability_matrix.csv`,
  `dum.yaml`) is now read at the PR head **then the base branch** if absent.
  Stale branches — and every open PR the moment a brand-new config file first
  lands on `develop` — keep working instead of 404-ing. Model evidence is still
  read at head. `dum.yaml` grants are only committed when the head branch has
  the file (a base-only copy still powers the read-only drift check).

## v0.9.0 - Tag Advisor + Permission Advisor grants, drift & hardening

New **Tag Advisor** MCP tool **`jirade_advise_tags_for_pr`**, plus the
Permission Advisor now applies grants to `dum.yaml`, runs a division-drift
health check, and picks up post-review hardening. (Minor bump: new MCP tool.)

Note: this also bumps `pyproject.toml` from `0.7.3` → `0.9.0`; the v0.8.0
Permission Advisor PR updated the changelog but not the package version.

Permission Advisor / shared client hardening:

- **`get_pr_files` now paginates** (`per_page=100`, follows pages). Previously
  it read only the first 30 files, so the advisor silently missed new tables on
  large PRs — exactly the data-remodel PRs it targets.
- **Idempotent no-op re-runs are now real.** The handler fetches the existing
  advisor comment and skips the write entirely when the content hash matches
  (`comment_unchanged` was previously dead code); no PATCH, no notification.
- **Applies grants to `dum.yaml`.** The advisor now closes the loop to where
  table access is actually applied: with `apply_dum_edit=true` it writes
  `catalog.schema.table: read` under the matching `group-division-*` blocks of
  `infra/deployments/databricks_user_management/dum.yaml` and commits the edit
  to the PR branch (model + who-gets-access reviewed together). Only
  **high-confidence** classifications are written (deterministic mv-inherit +
  high-confidence LLM); low-confidence ones are noted for a human. Edits use
  `ruamel.yaml` round-trip so comments and `&anchors` survive (added-lines-only
  diff). Divisions with no `group-division-*` block are reported, never
  invented. Default is dry-run (proposal shown in the comment).
- **Division-drift health check (every run).** The advisor compares the
  governance division universe (`capability_lookup[*].allowed_divisions`)
  against the `group-division-*` blocks in `dum.yaml`. Divisions governance can
  emit but that have no dum block — grants to them would be silently skipped —
  are reported in the tool result (`division_drift.missing_in_dum`) and flagged
  in the PR comment. Unused dum blocks are reported as informational only.
- **Tags are read from `schema.yml`** (`config.databricks_tags`) where
  production models actually declare them, falling back to SQL-embedded
  `databricks_tags` for the metric-view `auto_config()` style. Sibling-schema
  lookup now also matches `.yaml`, not just `.yml`.
- **Algolia-specific conventions collected into one named-constants block**
  (repo layout, model-name separator, org/division vocabulary, pipeline +
  governance file names) instead of being scattered across the parser, prompt,
  and comment builder. The default Claude model is a named constant; the core
  signature defaults to `None` and resolves from settings via the handler.

New **Tag Advisor** — MCP tool `jirade_advise_tags_for_pr` over a pure-logic
core (`jirade/tools/tag_advisor.py`). It comments on new/changed mart/analytics
models missing a governed `domain` tag (or carrying `tbd`/`unclassified`):

- `parse_governed_tags` reads the terraform-applied allowlist
  `infra/deployments/databricks_governed_tags/governed_tags.yaml` (the same
  file `main.tf` feeds into `databricks_tag_policy`).
- `assess_tag_gap` flags models whose `domain` tag is missing or a placeholder
  (`tbd` / `unclassified`).
- `classify_tags_with_claude` proposes a governed `domain` (+ optional
  `sub_domain`), constrained to the allowlist. A value that isn't governed yet
  becomes a gated `governed_tags.yaml` addition (a one-line YAML append —
  terraform derives the policy from it, no `.tf` edit) requiring sign-off.
- `build_tag_comment` renders an idempotent PR comment with a copy-pasteable
  `schema.yml` block per model. Idempotency helpers are shared with the
  Permission Advisor (`append_content_hash` / `content_unchanged`).

## v0.8.0 - Permission Advisor for dbt PRs

New MCP tool **`jirade_advise_permissions_for_pr`** that comments on every
dbt PR in `algolia/data` adding tables under `mart` or `analytics`:

- Filters to `A`-status `*.sql` paths in scope (mart/analytics only).
- Parses each new file at PR head SHA — extracts `databricks_tags`,
  `{{ ref() }}` driving tables, and any sibling YAML description.
- Loads `dbt-databricks/seeds/governance_state.yaml` from the same SHA and:
  - **Skips** tables already in `TABLE_OVERRIDES` (read-only respect for
    curated state).
  - **Inherits** caps for `mv_*` from non-core driving tables when possible —
    no Claude call.
  - Falls back to Claude (Opus 4.5) for net-new tables, filtering hallucinated
    cap IDs against the capability matrix.
- Resolves proposed caps → `allowed_divisions` via the OCL.
- Renders an idempotent markdown comment with a stable marker
  (`<!-- jirade:permission-advisor:v1 -->`) — re-runs on the same content are
  no-ops via the trailing content hash.
- Comment is upserted only when `post_comment=true` (default dry-run safe).

The advisor never modifies `process_permissions.py` or any governance state;
it only proposes. Disagreements are surfaced via PR review or a follow-up
`governance_state.yaml` change.

Core logic in `jirade/tools/permission_advisor.py` is pure functions —
unit-tested end-to-end against fixtures without any network.

## v0.7.3 - Single-scan metric view smoke tests

CI's metric-view smoke test ran one `SELECT MEASURE(<m>) FROM <mv>` query per
declared measure, sequentially. On the worksheet-parity PR (#4238 in
algolia/data) that meant 162 full aggregation scans for the marketing batch
alone — tens of minutes of wall clock for views over large facts.

`smoke_query_metric_view` now probes **all measures in a single combined
query** — `SELECT MEASURE(m1), MEASURE(m2), … FROM <mv>` — one scan proving
the view is queryable and every measure's column references resolve. Only when
the combined query fails does it fall back to the per-measure loop to
attribute the failure to the specific broken measure(s). The metadata-query
whitelist regex was extended to accept the multi-MEASURE form.

Result shape is unchanged (one probe entry per measure), so reports and
callers are unaffected. Green-path cost drops from N queries to 1 per view.

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
