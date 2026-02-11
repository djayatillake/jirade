"""dbt CI tool handlers for MCP server.

This module provides functionality for running dbt CI on Databricks
and comparing model outputs against production using metadata-only queries.

The Databricks CI approach:
- Runs dbt build with modified models +1 dependents in isolated schema
- Compares CI tables against production using metadata queries only
- No raw data is exposed to the agent
- CI tables are kept for inspection; use cleanup_ci to remove after merge
"""

import asyncio
import json
import logging
import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import yaml

from ...clients.databricks_client import DatabricksMetadataClient
from ...clients.github_client import GitHubClient
from ...config import get_settings

logger = logging.getLogger(__name__)

# Marker for identifying dbt diff comments
DBT_DIFF_MARKER = "<!-- dbt-diff-report -->"



async def handle_dbt_diff_tool(
    name: str,
    arguments: dict[str, Any],
    progress_cb: Any | None = None,
) -> dict[str, Any]:
    """Handle a dbt diff tool call.

    Args:
        name: Tool name.
        arguments: Tool arguments.
        progress_cb: Optional callback for sending progress notifications.

    Returns:
        Tool result.
    """
    settings = get_settings()

    if not settings.github_token:
        raise RuntimeError("GitHub token not configured")

    if name == "jirade_run_dbt_ci":
        owner = arguments["owner"]
        repo = arguments["repo"]
        pr_number = arguments["pr_number"]
        repo_path = arguments.get("repo_path", os.getcwd())
        dbt_project_subdir = arguments.get("dbt_project_subdir", "dbt-databricks")
        changed_models = arguments.get("models")
        lookback_days = arguments.get("lookback_days", settings.dbt_event_time_lookback_days)
        post_to_pr = arguments.get("post_to_pr", True)

        return await run_dbt_ci(
            owner=owner,
            repo=repo,
            pr_number=pr_number,
            github_token=settings.github_token,
            repo_path=repo_path,
            dbt_project_subdir=dbt_project_subdir,
            changed_models=changed_models,
            lookback_days=lookback_days,
            post_to_pr=post_to_pr,
            progress_cb=progress_cb,
        )

    elif name == "jirade_analyze_deprecation":
        table_name = arguments["table_name"]
        column_name = arguments.get("column_name")
        repo_path = arguments.get("repo_path", os.getcwd())
        dbt_project_subdir = arguments.get("dbt_project_subdir", "dbt-databricks")

        return await analyze_deprecation(
            table_name=table_name,
            column_name=column_name,
            repo_path=repo_path,
            dbt_project_subdir=dbt_project_subdir,
        )

    elif name == "jirade_cleanup_ci":
        pr_number = arguments["pr_number"]

        return await cleanup_ci_schemas(pr_number=pr_number)

    else:
        raise ValueError(f"Unknown dbt diff tool: {name}")


# =============================================================================
# Databricks CI Functions
# =============================================================================


async def post_diff_report(
    owner: str,
    repo: str,
    pr_number: int,
    report: str,
    github_token: str,
    update_existing: bool = True,
) -> dict[str, Any]:
    """Post or update a diff report as a PR comment.

    Args:
        owner: Repository owner.
        repo: Repository name.
        pr_number: PR number.
        report: Markdown report to post.
        github_token: GitHub access token.
        update_existing: If True, updates existing comment instead of creating new.

    Returns:
        Result with comment URL.
    """
    github = GitHubClient(token=github_token, owner=owner, repo=repo)

    try:
        if update_existing:
            result = await github.upsert_pr_comment(pr_number, report, DBT_DIFF_MARKER)
        else:
            result = await github.add_pr_comment(pr_number, report)

        return {
            "success": True,
            "comment_id": result.get("id"),
            "comment_url": result.get("html_url"),
            "updated": update_existing and DBT_DIFF_MARKER in str(result.get("body", "")),
        }

    finally:
        await github.close()


async def run_dbt_ci(
    owner: str,
    repo: str,
    pr_number: int,
    github_token: str,
    repo_path: str,
    dbt_project_subdir: str = "dbt-databricks",
    changed_models: list[str] | None = None,
    lookback_days: int = 3,
    post_to_pr: bool = True,
    progress_cb: Any | None = None,
) -> dict[str, Any]:
    """Run dbt CI on Databricks for a PR.

    Builds modified models +1 dependents in an isolated CI schema,
    compares against production tables using metadata queries only.

    Args:
        owner: Repository owner.
        repo: Repository name.
        pr_number: PR number.
        github_token: GitHub access token.
        repo_path: Local path to the repository.
        dbt_project_subdir: Subdirectory containing dbt project.
        changed_models: List of model names to build. If None, auto-detects.
        lookback_days: Days back for event-time-start (microbatch models).
        post_to_pr: If True, automatically posts the diff report to the PR.
        progress_cb: Optional callback for sending progress notifications.

    Returns:
        CI results including model comparisons and report.
    """
    settings = get_settings()

    if not settings.has_databricks:
        raise RuntimeError(
            "Databricks not configured. Set JIRADE_DATABRICKS_HOST and "
            "JIRADE_DATABRICKS_HTTP_PATH (and JIRADE_DATABRICKS_TOKEN if using token auth)."
        )

    if not settings.databricks_ci_catalog:
        raise RuntimeError(
            "CI catalog not configured. Set JIRADE_DATABRICKS_CI_CATALOG to a catalog "
            "where you have CREATE SCHEMA permission (e.g., your dev catalog like "
            "'development_yourname_metadata')."
        )

    async def _notify(progress: float, total: float, message: str) -> None:
        if progress_cb:
            try:
                await progress_cb(progress, total, message)
            except Exception:
                pass  # Don't fail the build over a notification error

    github = GitHubClient(token=github_token, owner=owner, repo=repo)
    ci_schema = f"{settings.dbt_ci_schema_prefix}_{pr_number}"

    try:
        # Get PR details
        await _notify(1, 100, "Fetching PR details...")
        pr = await github.get_pull_request(pr_number)
        base_branch = pr.get("base", {}).get("ref", "develop")
        head_sha = pr.get("head", {}).get("sha", "")
        head_branch = pr.get("head", {}).get("ref", "")

        # Auto-detect changed models and seeds if not provided
        await _notify(5, 100, "Detecting changed models and seeds...")
        changed_seeds: list[str] = []
        if changed_models is None:
            pr_files = await github.get_pr_files(pr_number)
            changed_models = []
            for f in pr_files:
                filename = f.get("filename", "")
                if filename.startswith(f"{dbt_project_subdir}/models/") and filename.endswith(".sql"):
                    model_name = Path(filename).stem
                    changed_models.append(model_name)
                elif filename.startswith(f"{dbt_project_subdir}/seeds/") and filename.endswith(".csv"):
                    seed_name = Path(filename).stem
                    changed_seeds.append(seed_name)

            if not changed_models and not changed_seeds:
                return {
                    "success": True,
                    "message": "No dbt models or seeds changed in this PR",
                    "model_results": [],
                    "report": None,
                }

        logger.info(f"Running Databricks CI for models: {changed_models}, seeds: {changed_seeds}")
        await _notify(10, 100, f"Found {len(changed_models)} changed model(s), {len(changed_seeds)} changed seed(s)")

        # Checkout the PR branch to build with the correct code
        await _notify(12, 100, f"Checking out PR branch: {head_branch}...")
        original_branch = None
        repo_root = Path(repo_path)
        try:
            result = await asyncio.create_subprocess_exec(
                "git", "rev-parse", "--abbrev-ref", "HEAD",
                cwd=str(repo_root), stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await result.communicate()
            original_branch = stdout.decode().strip()

            # Fetch latest and checkout PR branch
            await (await asyncio.create_subprocess_exec(
                "git", "fetch", "origin", head_branch,
                cwd=str(repo_root), stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )).wait()
            proc = await asyncio.create_subprocess_exec(
                "git", "checkout", head_branch,
                cwd=str(repo_root), stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await proc.communicate()
            if proc.returncode != 0:
                # Try checkout as detached HEAD from origin
                proc2 = await asyncio.create_subprocess_exec(
                    "git", "checkout", f"origin/{head_branch}",
                    cwd=str(repo_root), stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                )
                await proc2.wait()

            logger.info(f"Checked out PR branch: {head_branch}")
        except Exception as e:
            logger.warning(f"Failed to checkout PR branch {head_branch}: {e}")

        # Clean up any existing CI schemas for this PR (ensures clean slate)
        await _notify(15, 100, "Dropping existing CI schemas (clean slate)...")
        ci_schema_prefix = f"jirade_ci_{pr_number}_"
        if settings.databricks_ci_catalog:
            logger.info(f"Cleaning up existing CI schemas matching {ci_schema_prefix}*")
            with DatabricksMetadataClient(
                host=settings.databricks_host,
                http_path=settings.databricks_http_path,
                auth_type=settings.databricks_auth_type,
                token=settings.databricks_token if settings.databricks_auth_type == "token" else None,
                catalog=settings.databricks_ci_catalog,
            ) as db_client:
                try:
                    schemas_result = db_client.execute_metadata_query(
                        f"SHOW SCHEMAS IN {settings.databricks_ci_catalog} LIKE 'jirade_ci_{pr_number}_*'"
                    )
                    ci_schemas = [r.get("databaseName", r.get("namespace", "")) for r in schemas_result]
                    for schema in ci_schemas:
                        if schema.startswith(ci_schema_prefix):
                            full_schema = f"{settings.databricks_ci_catalog}.{schema}"
                            try:
                                db_client.drop_ci_schema(full_schema)
                                logger.info(f"Dropped existing CI schema: {full_schema}")
                            except Exception as e:
                                logger.warning(f"Failed to drop CI schema {full_schema}: {e}")
                except Exception as e:
                    logger.warning(f"Failed to list/clean CI schemas: {e}")

        # Build model selector with +1 for dependents
        model_selectors = [f"{model}+1" for model in changed_models]
        # Seeds: select +1 downstream models (same as models, only direct dependents)
        seed_descendant_selectors = [f"{seed}+1" for seed in changed_seeds]
        selector_str = " ".join(model_selectors + seed_descendant_selectors)

        # Calculate event time dates
        today = datetime.now().date()
        start_date = today - timedelta(days=lookback_days)

        # Run dbt build on Databricks
        await _notify(20, 100, "Building dbt models on Databricks (this may take a few minutes)...")
        project_dir = Path(repo_path) / dbt_project_subdir
        dbt_build_result = await _run_dbt_build_databricks(
            project_dir=project_dir,
            ci_schema=ci_schema,
            pr_number=pr_number,
            selector=selector_str,
            event_time_start=start_date.isoformat(),
            event_time_end=today.isoformat(),
            progress_cb=progress_cb,
            changed_seeds=changed_seeds if changed_seeds else None,
        )

        if not dbt_build_result["success"]:
            return {
                "success": False,
                "error": dbt_build_result["error"],
                "model_results": [],
                "report": None,
            }

        # Get list of models that were built (for logging)
        built_models = dbt_build_result.get("built_models", changed_models)
        model_build_failures = dbt_build_result.get("model_failures", [])
        test_failures = dbt_build_result.get("test_failures", [])
        seed_failures = dbt_build_result.get("seed_failures", [])
        logger.info(f"Built {len(built_models)} models (changed + downstream)")
        if seed_failures:
            logger.warning(f"{len(seed_failures)} seed(s) failed to load: {seed_failures}")
        if model_build_failures:
            logger.warning(f"{len(model_build_failures)} model(s) failed to build: {model_build_failures}")
        if test_failures:
            logger.warning(f"{len(test_failures)} test(s) failed: {test_failures}")
        await _notify(70, 100, f"Built {len(built_models)} models. Comparing against production...")

        # Compare all built models (changed + downstream) against prod
        models_to_compare = built_models
        logger.info(f"Comparing {len(models_to_compare)} models ({len(changed_models)} changed + {len(built_models) - len(changed_models)} downstream)")

        # Parse manifest for incremental/microbatch model configs (for date filtering)
        # and find downstream models whose data is limited by time-filtered ancestors
        model_configs = {}
        time_limited_descendants: set[str] = set()
        manifest_path = project_dir / "target" / "manifest.json"
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text())
                nodes = manifest.get("nodes", {})

                # First pass: find incremental/microbatch models with event_time
                time_limited_node_ids: set[str] = set()
                node_id_to_name: dict[str, str] = {}
                for node_id, node in nodes.items():
                    if node.get("resource_type") != "model":
                        continue
                    model_name = node.get("name", "")
                    node_id_to_name[node_id] = model_name
                    config = node.get("config", {})
                    materialized = config.get("materialized", "")
                    if materialized in ("incremental", "microbatch"):
                        event_time = config.get("event_time", "")
                        if event_time:
                            # Strip backticks that manifest may include
                            event_time = event_time.strip("`")
                            if model_name:
                                model_configs[model_name] = {
                                    "materialized": materialized,
                                    "event_time": event_time,
                                }
                                time_limited_node_ids.add(node_id)

                # Second pass: walk the DAG to find all descendants of time-limited models
                # Build parent -> children map from depends_on.nodes
                if time_limited_node_ids:
                    children_map: dict[str, list[str]] = {}
                    for node_id, node in nodes.items():
                        if node.get("resource_type") != "model":
                            continue
                        for parent_id in node.get("depends_on", {}).get("nodes", []):
                            children_map.setdefault(parent_id, []).append(node_id)

                    # BFS from time-limited models to find all transitive descendants
                    queue = list(time_limited_node_ids)
                    visited: set[str] = set(time_limited_node_ids)
                    while queue:
                        current = queue.pop(0)
                        for child_id in children_map.get(current, []):
                            if child_id not in visited:
                                visited.add(child_id)
                                child_name = node_id_to_name.get(child_id, "")
                                # Only flag descendants that aren't themselves time-limited
                                # (those get their own date filter)
                                if child_name and child_name not in model_configs:
                                    time_limited_descendants.add(child_name)
                                queue.append(child_id)

                    if time_limited_descendants:
                        logger.info(f"Found {len(time_limited_descendants)} downstream models with time-limited ancestors: {sorted(time_limited_descendants)}")

                if model_configs:
                    logger.info(f"Found {len(model_configs)} incremental/microbatch models with event_time: {list(model_configs.keys())}")
            except Exception as e:
                logger.warning(f"Failed to parse manifest for incremental configs: {e}")

        # Compare CI tables against prod using metadata client
        changed_models_set = set(changed_models)
        with DatabricksMetadataClient(
            host=settings.databricks_host,
            http_path=settings.databricks_http_path,
            auth_type=settings.databricks_auth_type,
            token=settings.databricks_token if settings.databricks_auth_type == "token" else None,
            catalog=settings.databricks_catalog or None,
        ) as db_client:

            model_results = []

            for model in models_to_compare:
                try:
                    # Get table names (CI and prod)
                    ci_table = _get_ci_table_name(model, pr_number, settings.databricks_ci_catalog, project_dir)
                    prod_table = _get_prod_table_name(model, project_dir)
                    is_downstream = model not in changed_models_set

                    # Extract the short name for manifest lookup
                    # Model name in manifest is just the short name (e.g. "my_model"),
                    # but models_to_compare uses dbt unique_id format (catalog__schema__table).
                    model_short_name = model.split("__")[-1] if "__" in model else model

                    # Skip comparison for downstream models whose upstream is time-limited.
                    # CI only has data for the lookback window, so non-incremental descendants
                    # will have incomplete data — comparing against prod is meaningless.
                    if is_downstream and model_short_name in time_limited_descendants:
                        logger.info(f"Skipping comparison for {model}: downstream of time-limited model, CI data is incomplete")
                        model_results.append({
                            "model": model,
                            "change_type": "MODIFIED",
                            "is_downstream": True,
                            "comparison_skipped": True,
                            "skip_reason": "Upstream model is incremental/microbatch — CI was built with only "
                                           f"{lookback_days} days of data, so this downstream model's row counts "
                                           "are not comparable to production.",
                            "has_diff": False,
                        })
                        continue

                    # Build date filter for incremental models
                    date_filter = None
                    if model_short_name in model_configs:
                        mc = model_configs[model_short_name]
                        date_filter = {
                            "column": mc["event_time"],
                            "start": start_date.isoformat(),
                            "end": today.isoformat(),
                        }

                    if not prod_table:
                        # New model, get metadata only
                        metadata = db_client.get_new_table_metadata(ci_table)
                        result = {
                            "model": model,
                            "change_type": "NEW",
                            "has_diff": True,
                            "is_downstream": is_downstream,
                            "row_count": {"ci": metadata["row_count"], "base": 0, "diff": metadata["row_count"]},
                            "schema_changes": [
                                {"column": s["column"], "change": "ADDED", "type": s["type"]}
                                for s in metadata.get("column_stats", [])
                            ],
                            "column_stats": metadata.get("column_stats", []),
                        }
                        model_results.append(result)
                    else:
                        # Compare CI vs prod
                        comparison = db_client.compare_tables(prod_table, ci_table, date_filter=date_filter)
                        comparison["model"] = model
                        comparison["change_type"] = "MODIFIED"
                        comparison["is_downstream"] = is_downstream
                        if date_filter:
                            comparison["is_incremental"] = True
                            comparison["date_filter"] = date_filter
                        model_results.append(comparison)

                except Exception as e:
                    logger.exception(f"Error comparing model {model}: {e}")
                    model_results.append({
                        "model": model,
                        "change_type": "MODIFIED",
                        "is_downstream": model not in changed_models_set,
                        "error": str(e),
                        "has_diff": True,
                    })

            # Note: CI schemas are NOT cleaned up here - they remain available
            # for manual inspection until the PR is merged. Cleanup should be
            # triggered separately (e.g., via webhook on PR merge).

        # Split results into changed and downstream
        changed_model_results = [r for r in model_results if not r.get("is_downstream")]
        downstream_model_results = [r for r in model_results if r.get("is_downstream")]

        # Generate report with CI catalog for table references
        await _notify(85, 100, "Generating diff report...")
        report = format_ci_diff_report(
            pr_number=pr_number,
            base_branch=base_branch,
            head_sha=head_sha,
            model_results=changed_model_results,
            downstream_model_results=downstream_model_results,
            model_build_failures=model_build_failures,
            test_failures=test_failures,
            ci_catalog=settings.databricks_ci_catalog,
            changed_seeds=changed_seeds if changed_seeds else None,
            seed_failures=seed_failures if seed_failures else None,
        )

        # Post report to PR if requested
        posted_to_pr = False
        if post_to_pr and report:
            await _notify(90, 100, "Posting diff report to PR...")
            try:
                post_result = await post_diff_report(
                    owner=owner,
                    repo=repo,
                    pr_number=pr_number,
                    report=report,
                    github_token=github_token,
                    update_existing=True,
                )
                posted_to_pr = post_result.get("success", False)
                if posted_to_pr:
                    logger.info(f"Posted diff report to PR #{pr_number}")
                else:
                    logger.warning(f"Failed to post diff report: {post_result.get('error')}")
            except Exception as e:
                logger.warning(f"Failed to post diff report to PR: {e}")

        await _notify(100, 100, "CI complete")

        return {
            "success": len(model_build_failures) == 0,
            "model_results": model_results,
            "model_build_failures": model_build_failures,
            "report": report,
            "posted_to_pr": posted_to_pr,
            "models_analyzed": len(model_results),
            "models_with_diffs": sum(1 for r in model_results if r.get("has_diff")),
            "changed_models_compared": len(changed_model_results),
            "downstream_models_compared": len(downstream_model_results),
            "ci_catalog": settings.databricks_ci_catalog,
            "ci_schema_prefix": f"jirade_ci_{pr_number}_",
            "cleanup_pending": True,  # Cleanup happens on PR merge
        }

    finally:
        # Restore original branch
        if original_branch:
            try:
                await (await asyncio.create_subprocess_exec(
                    "git", "checkout", original_branch,
                    cwd=str(repo_root), stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                )).wait()
                logger.info(f"Restored original branch: {original_branch}")
            except Exception as e:
                logger.warning(f"Failed to restore branch {original_branch}: {e}")
        await github.close()


async def _run_dbt_build_databricks(
    project_dir: Path,
    ci_schema: str,
    pr_number: int,
    selector: str,
    event_time_start: str,
    event_time_end: str,
    progress_cb: Any | None = None,
    changed_seeds: list[str] | None = None,
) -> dict[str, Any]:
    """Run dbt build targeting Databricks CI schema.

    Args:
        project_dir: Path to dbt project.
        ci_schema: Target schema for CI tables.
        pr_number: GitHub PR number (used for CI schema isolation).
        selector: dbt model selector string.
        event_time_start: Start date for event-time filtering.
        event_time_end: End date for event-time filtering.
        progress_cb: Optional callback for sending progress notifications.

    Returns:
        Build result with success status and any errors.
    """
    from ...config import get_settings

    settings = get_settings()

    # Create temporary profiles.yml with OAuth auth
    temp_profiles_dir = project_dir / ".jirade_profiles"
    temp_profiles_dir.mkdir(exist_ok=True)
    profiles_file = temp_profiles_dir / "profiles.yml"

    # Determine auth config
    if settings.databricks_auth_type == "oauth":
        auth_config = "auth_type: oauth"
    else:
        auth_config = f"token: \"{settings.databricks_token}\""

    # Read profile name from dbt_project.yml
    project_file = project_dir / "dbt_project.yml"
    with open(project_file) as f:
        project_config = yaml.safe_load(f)
    profile_name = project_config.get("profile", "default")

    profiles_content = f"""{profile_name}:
  target: ci
  outputs:
    ci:
      type: databricks
      host: "{settings.databricks_host}"
      http_path: "{settings.databricks_http_path}"
      {auth_config}
      catalog: "{settings.databricks_catalog or 'hive_metastore'}"
      schema: "{ci_schema}"
      threads: 4
"""
    profiles_file.write_text(profiles_content)

    # Set environment variables to trigger CI macros in generate_database_name/generate_schema_name
    env = {
        **os.environ,
        "JIRADE_CI_SCHEMA": ci_schema,
        "DBT_JIRADE_CI": "true",  # Triggers CI mode in generate_database_name/generate_schema_name
        "DBT_JIRADE_PR_ID": str(pr_number),  # Used in generate_schema_name for isolation
        "DBT_JIRADE_CI_CATALOG": settings.databricks_ci_catalog,  # Catalog for CI tables
    }

    # Run dbt seed first if there are changed seeds (must load before dbt run so ref() resolves to CI version)
    seed_failures: list[str] = []
    if changed_seeds:
        seed_cmd = [
            "poetry", "run", "dbt", "seed",
            "--profiles-dir", str(temp_profiles_dir),
            "--select", " ".join(changed_seeds),
        ]
        logger.info(f"Running dbt seed: {' '.join(seed_cmd)}")

        seed_proc = await asyncio.create_subprocess_exec(
            *seed_cmd,
            cwd=str(project_dir),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=env,
        )

        seed_output_lines = []
        async for line in seed_proc.stdout:
            decoded_line = line.decode()
            seed_output_lines.append(decoded_line)
            logger.info(f"[dbt seed] {decoded_line.rstrip()}")

        await seed_proc.wait()

        # Parse seed results from run_results.json
        run_results_path = project_dir / "target" / "run_results.json"
        if run_results_path.exists():
            try:
                with open(run_results_path) as f:
                    seed_run_results = json.load(f)
                for result in seed_run_results.get("results", []):
                    unique_id = result.get("unique_id", "")
                    status = result.get("status", "")
                    if unique_id.startswith("seed.") and status == "error":
                        seed_name = unique_id.split(".")[-1]
                        seed_failures.append(seed_name)
            except Exception as e:
                logger.warning(f"Failed to parse seed run_results.json: {e}")

        if seed_failures:
            logger.warning(f"Seed failures: {seed_failures}")

    # Use dbt run (not build) so test failures don't skip downstream models
    # Use --defer --state to reference production tables for upstream models not in the PR
    state_dir = project_dir / "target_lock"
    cmd = [
        "poetry", "run", "dbt", "run",
        "--profiles-dir", str(temp_profiles_dir),
        "--select", selector,
        "--event-time-start", event_time_start,
        "--event-time-end", event_time_end,
        "--defer",
        "--state", str(state_dir),
    ]
    # Only use --favor-state when there are no changed seeds. With --favor-state,
    # dbt defers unselected nodes to the state manifest without checking the database.
    # Seeds are filtered from SELECTED_RESOURCES by dbt run's ResourceTypeSelector,
    # so --favor-state would cause ref('seed') to resolve to production even if we
    # just loaded the CI version via dbt seed. Without it, dbt checks the database
    # first - finding the seed we loaded - and uses the CI version correctly.
    if not changed_seeds:
        cmd.append("--favor-state")

    logger.info(f"Running dbt build: {' '.join(cmd)}")

    # Create log file for streaming output
    log_file = project_dir / ".jirade_dbt_ci.log"

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(project_dir),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,  # Merge stderr into stdout
            env=env,
        )

        # Stream output line by line to log file and collect for result
        output_lines = []
        with open(log_file, "w") as f:
            f.write(f"=== dbt CI build started ===\n")
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"Log file: {log_file}\n")
            f.write(f"{'=' * 50}\n\n")
            f.flush()

            line_count = 0
            recent_lines: list[str] = []
            async for line in proc.stdout:
                decoded_line = line.decode()
                output_lines.append(decoded_line)
                f.write(decoded_line)
                f.flush()  # Flush immediately for real-time viewing
                # Also log to jirade logger for visibility
                stripped = decoded_line.rstrip()
                logger.info(f"[dbt] {stripped}")
                # Send progress notification with last 10 lines as context
                if progress_cb:
                    line_count += 1
                    recent_lines.append(stripped)
                    if len(recent_lines) > 10:
                        recent_lines.pop(0)
                    try:
                        await progress_cb(line_count, None, "\n".join(recent_lines))
                    except Exception:
                        pass

        await proc.wait()

        full_output = "".join(output_lines)

        # Parse run_results.json for model build results
        run_results_path = project_dir / "target" / "run_results.json"
        built_models = []
        model_failures = []

        if run_results_path.exists():
            try:
                with open(run_results_path) as f:
                    run_results = json.load(f)
                for result in run_results.get("results", []):
                    unique_id = result.get("unique_id", "")
                    status = result.get("status", "")
                    if unique_id.startswith("model."):
                        model_name = unique_id.split(".")[-1]
                        if status in ("success", "pass"):
                            built_models.append(model_name)
                        elif status == "error":
                            model_failures.append(model_name)
            except Exception as e:
                logger.warning(f"Failed to parse run_results.json: {e}")

        # If ALL models failed (nothing built at all), return early
        if not built_models and (model_failures or proc.returncode != 0):
            return {
                "success": False,
                "error": f"dbt run failed (see log: {log_file}): {full_output[-1000:]}",
                "log_file": str(log_file),
            }

        # Run tests separately so they don't block model builds
        test_failures = []
        test_cmd = [
            "poetry", "run", "dbt", "test",
            "--profiles-dir", str(temp_profiles_dir),
            "--select", selector,
            "--indirect-selection=cautious",
            "--exclude", "test_name:no_missing_date*",
            "--defer",
            "--state", str(state_dir),
        ]
        if not changed_seeds:
            test_cmd.append("--favor-state")

        logger.info(f"Running dbt test: {' '.join(test_cmd)}")
        with open(log_file, "a") as f:
            f.write(f"\n{'=' * 50}\n")
            f.write(f"=== dbt test started ===\n")
            f.write(f"Command: {' '.join(test_cmd)}\n")
            f.write(f"{'=' * 50}\n\n")
            f.flush()

            test_proc = await asyncio.create_subprocess_exec(
                *test_cmd,
                cwd=str(project_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env=env,
            )

            async for line in test_proc.stdout:
                decoded_line = line.decode()
                f.write(decoded_line)
                f.flush()
                stripped = decoded_line.rstrip()
                logger.info(f"[dbt test] {stripped}")
                if progress_cb:
                    line_count += 1
                    recent_lines.append(stripped)
                    if len(recent_lines) > 10:
                        recent_lines.pop(0)
                    try:
                        await progress_cb(line_count, None, "\n".join(recent_lines))
                    except Exception:
                        pass

            await test_proc.wait()

        # Parse test results from run_results.json (overwritten by dbt test)
        if run_results_path.exists():
            try:
                with open(run_results_path) as f:
                    test_run_results = json.load(f)
                for result in test_run_results.get("results", []):
                    unique_id = result.get("unique_id", "")
                    status = result.get("status", "")
                    if unique_id.startswith("test."):
                        if status == "error" or status == "fail":
                            message = result.get("message", "").strip()[:300]
                            test_failures.append({"unique_id": unique_id, "message": message})
            except Exception as e:
                logger.warning(f"Failed to parse test run_results.json: {e}")

        # Enrich test failures with manifest metadata for clean names
        if test_failures:
            manifest_path = project_dir / "target" / "manifest.json"
            if manifest_path.exists():
                try:
                    with open(manifest_path) as f:
                        manifest = json.load(f)
                    nodes = manifest.get("nodes", {})
                    for tf in test_failures:
                        node = nodes.get(tf["unique_id"], {})
                        meta = node.get("test_metadata", {})
                        if meta:
                            test_type = meta.get("name", "test")
                            column = node.get("column_name", "")
                            deps = node.get("depends_on", {}).get("nodes", [])
                            model_ref = ""
                            for dep in deps:
                                if dep.startswith("model."):
                                    model_ref = dep.split(".")[-1]
                                    break
                            if model_ref and column:
                                tf["name"] = f"{test_type}({model_ref}.{column})"
                            elif model_ref:
                                tf["name"] = f"{test_type}({model_ref})"
                            else:
                                tf["name"] = test_type
                        else:
                            parts = tf["unique_id"].split(".", 2)
                            tf["name"] = parts[2] if len(parts) > 2 else tf["unique_id"]
                except Exception as e:
                    logger.warning(f"Failed to enrich test failures from manifest: {e}")
                    for tf in test_failures:
                        if "name" not in tf:
                            parts = tf["unique_id"].split(".", 2)
                            tf["name"] = parts[2] if len(parts) > 2 else tf["unique_id"]

        return {
            "success": True,
            "built_models": built_models,
            "model_failures": model_failures,
            "test_failures": test_failures,
            "seed_failures": seed_failures,
            "output": full_output[-2000:],  # Last 2000 chars of output
            "log_file": str(log_file),
        }
    finally:
        # Cleanup temp profiles directory
        if temp_profiles_dir.exists():
            import shutil
            shutil.rmtree(temp_profiles_dir, ignore_errors=True)


def _get_ci_table_name(model: str, pr_number: int, ci_catalog: str, project_dir: Path) -> str:
    """Get fully qualified CI table name for a model.

    The CI schema is generated using the jirade pattern:
    jirade_ci_{pr_number}_{original_catalog}_{original_schema}

    Args:
        model: Model name (format: catalog__schema__table_name).
        pr_number: PR number for CI schema isolation.
        ci_catalog: Catalog where CI tables are created.
        project_dir: Path to dbt project.

    Returns:
        Fully qualified table name (ci_catalog.ci_schema.table_name).
    """
    # Parse model name to extract original catalog, schema, and table name
    parts = model.split("__")
    if len(parts) >= 3:
        original_catalog = parts[0]
        original_schema = parts[1]
        table_name = "_".join(parts[2:])  # Handle multi-part table names
    elif len(parts) == 2:
        original_catalog = parts[0]
        original_schema = parts[0]
        table_name = parts[1]
    else:
        original_catalog = "default"
        original_schema = "default"
        table_name = model

    # Construct CI schema using jirade pattern
    ci_schema = f"jirade_ci_{pr_number}_{original_catalog}_{original_schema}"

    return f"{ci_catalog}.{ci_schema}.{table_name}"


def _get_prod_table_name(model: str, project_dir: Path) -> str | None:
    """Get fully qualified production table name for a model.

    Derives production table name from model naming convention:
    catalog__schema__table_name -> catalog.schema.table_name

    Args:
        model: Model name (format: catalog__schema__table_name).
        project_dir: Path to dbt project (unused, kept for compatibility).

    Returns:
        Fully qualified table name or None if model name cannot be parsed.
    """
    # Parse model name to extract original catalog, schema, and table name
    parts = model.split("__")
    if len(parts) >= 3:
        original_catalog = parts[0]
        original_schema = parts[1]
        table_name = "_".join(parts[2:])  # Handle multi-part table names
        return f"{original_catalog}.{original_schema}.{table_name}"
    elif len(parts) == 2:
        # Assume catalog and schema are the same for 2-part names
        original_catalog = parts[0]
        original_schema = parts[0]
        table_name = parts[1]
        return f"{original_catalog}.{original_schema}.{table_name}"

    # Can't parse model name, return None (treated as new model)
    return None


def _format_model_summary_row(
    result: dict[str, Any],
    pr_number: int,
    ci_catalog: str,
) -> str:
    """Format a single model result as a summary table row.

    Args:
        result: Comparison result dict for a model.
        pr_number: PR number (for CI table name generation).
        ci_catalog: Catalog where CI tables are created.

    Returns:
        Markdown table row string.
    """
    model = result.get("model", "unknown")
    change_type = result.get("change_type", "MODIFIED")
    row_count = result.get("row_count", {})
    schema_changes = len(result.get("schema_changes", []))

    # Generate CI table name from model name
    parts = model.split("__")
    if len(parts) >= 3:
        orig_catalog, orig_schema = parts[0], parts[1]
        table_name = "_".join(parts[2:])
    elif len(parts) == 2:
        orig_catalog, orig_schema = parts[0], parts[0]
        table_name = parts[1]
    else:
        orig_catalog, orig_schema, table_name = "default", "default", model

    ci_schema = f"jirade_ci_{pr_number}_{orig_catalog}_{orig_schema}"
    ci_table = f"{ci_catalog}.{ci_schema}.{table_name}"

    # Handle skipped comparisons
    if result.get("comparison_skipped"):
        return f"| `{table_name}` | `{ci_table}` | _skipped_ | _skipped_ | :fast_forward: |"

    # Format row count
    diff = row_count.get("diff", 0)
    pct = row_count.get("pct_change")
    if change_type == "NEW":
        row_str = f"+{row_count.get('ci', 0):,}"
    elif diff == 0:
        row_str = "No change"
    elif diff > 0:
        row_str = f"+{diff:,} (+{pct:.1f}%)" if pct else f"+{diff:,}"
    else:
        row_str = f"{diff:,} ({pct:.1f}%)" if pct else f"{diff:,}"

    # Date-filtered indicator
    if result.get("date_filtered"):
        row_str += " :calendar:"

    # Format schema changes
    if schema_changes == 0:
        schema_str = "No changes"
    else:
        schema_str = f"{schema_changes} change{'s' if schema_changes > 1 else ''}"

    # Check for any errors
    has_error = result.get("error") or result.get("row_count_error") or result.get("schema_error")

    # Status
    if has_error:
        status = ":x:"
    elif change_type == "NEW":
        status = ":new:"
    elif result.get("has_diff"):
        status = ":warning:"
    else:
        status = ":white_check_mark:"

    return f"| `{table_name}` | `{ci_table}` | {row_str} | {schema_str} | {status} |"


def _format_model_detail_section(result: dict[str, Any]) -> list[str]:
    """Format detailed diff section for a single model result.

    Args:
        result: Comparison result dict for a model.

    Returns:
        List of markdown lines for the detail section.
    """
    lines: list[str] = []
    has_error = result.get("error") or result.get("row_count_error") or result.get("schema_error")
    if not result.get("has_diff") and not has_error and not result.get("comparison_skipped"):
        return lines

    model = result.get("model", "unknown")
    change_type = result.get("change_type", "MODIFIED")

    lines.append("<details>")
    lines.append(f"<summary><b>{model}</b> ({change_type})</summary>")
    lines.append("")

    if result.get("comparison_skipped"):
        lines.append(f":fast_forward: **Comparison skipped:** {result.get('skip_reason', 'Unknown reason')}")
        lines.append("")
        lines.append("</details>")
        lines.append("")
        return lines

    if has_error:
        error_msg = result.get("error") or result.get("row_count_error") or result.get("schema_error")
        if "TABLE_OR_VIEW_NOT_FOUND" in str(error_msg):
            error_msg = f"CI table not found: `{result.get('ci_table', 'unknown')}` - model may not have been built"
        lines.append(f"**Error:** {error_msg}")
    else:
        # Date filter note for incremental models
        date_filter = result.get("date_filter")
        if date_filter:
            col = date_filter.get("column", "")
            start = date_filter.get("start", "")
            end = date_filter.get("end", "")
            lines.append(f"> :calendar: Row counts filtered to `{col}` >= `{start}` AND < `{end}` (incremental model)")
            lines.append("")

        # Row count
        rc = result.get("row_count", {})
        if change_type == "NEW":
            lines.append("#### New Model")
            lines.append(f"- Rows: {rc.get('ci', 0):,}")
        elif rc.get("diff", 0) != 0:
            lines.append("#### Row Count")
            lines.append(f"- Production: {rc.get('base', 0):,}")
            lines.append(f"- CI: {rc.get('ci', 0):,}")
            diff = rc.get("diff", 0)
            pct = rc.get("pct_change", 0) or 0
            lines.append(f"- Diff: {diff:+,} ({pct:+.1f}%)")

        lines.append("")

        # Schema changes
        schema_changes = result.get("schema_changes", [])
        if schema_changes:
            lines.append("#### Schema Changes")
            lines.append("")
            lines.append("| Column | Change | Type |")
            lines.append("|--------|--------|------|")
            for change in schema_changes:
                col = change.get("column", "")
                chg = change.get("change", "")
                if chg == "TYPE_CHANGED":
                    type_str = f"{change.get('base_type')} -> {change.get('ci_type')}"
                else:
                    type_str = change.get("type", "")
                lines.append(f"| `{col}` | {chg} | {type_str} |")
            lines.append("")

        # NULL changes
        null_changes = result.get("null_changes", [])
        if null_changes:
            lines.append("#### NULL Count Changes")
            lines.append("")
            lines.append("| Column | Prod NULLs | CI NULLs | Diff |")
            lines.append("|--------|------------|----------|------|")
            for change in null_changes:
                col = change.get("column", "")
                base_n = change.get("base_nulls", 0)
                ci_n = change.get("ci_nulls", 0)
                diff = change.get("diff", 0)
                lines.append(f"| `{col}` | {base_n:,} | {ci_n:,} | {diff:+,} |")
            lines.append("")

        # Column stats for new models
        col_stats = result.get("column_stats", [])
        if col_stats and change_type == "NEW":
            lines.append("#### Column Statistics")
            lines.append("")
            lines.append("| Column | Type | NULLs | NULL% | Distinct | Uniqueness% |")
            lines.append("|--------|------|-------|-------|----------|-------------|")
            for stat in col_stats:
                if stat.get("error"):
                    continue
                lines.append(
                    f"| `{stat.get('column')}` | {stat.get('type')} | "
                    f"{stat.get('null_count', 0):,} | {stat.get('null_pct', 0):.1f}% | "
                    f"{stat.get('distinct_count', 0):,} | {stat.get('uniqueness', 0):.1f}% |"
                )
            lines.append("")

    lines.append("</details>")
    lines.append("")

    return lines


def format_ci_diff_report(
    pr_number: int,
    base_branch: str,
    head_sha: str,
    model_results: list[dict[str, Any]],
    downstream_model_results: list[dict[str, Any]] | None = None,
    model_build_failures: list[str] | None = None,
    test_failures: list[str] | None = None,
    ci_catalog: str = "",
    changed_seeds: list[str] | None = None,
    seed_failures: list[str] | None = None,
) -> str:
    """Format CI comparison results as a markdown report.

    Args:
        pr_number: PR number.
        base_branch: Base branch name.
        head_sha: Head commit SHA.
        model_results: List of comparison results for changed models.
        downstream_model_results: List of comparison results for downstream models.
        model_build_failures: List of model names that failed to build.
        test_failures: List of test names that failed during the build.
        ci_catalog: Catalog where CI tables are created.
        changed_seeds: List of seed names that were updated in this PR.
        seed_failures: List of seed names that failed to load.

    Returns:
        Markdown formatted report.
    """
    downstream_model_results = downstream_model_results or []
    model_build_failures = model_build_failures or []
    test_failures = test_failures or []
    changed_seeds = changed_seeds or []
    seed_failures = seed_failures or []

    downstream_skipped = sum(1 for r in downstream_model_results if r.get("comparison_skipped"))
    total_compared = len(model_results) + len(downstream_model_results) - downstream_skipped

    header_parts = [f"{len(model_results)} changed"]
    downstream_compared = len(downstream_model_results) - downstream_skipped
    if downstream_compared:
        header_parts.append(f"{downstream_compared} downstream")
    if downstream_skipped:
        header_parts.append(f"{downstream_skipped} skipped")

    lines = [
        DBT_DIFF_MARKER,
        "## dbt CI Diff Report",
        "",
        f"**PR #{pr_number}** | **Base:** `{base_branch}` | **Models compared:** {total_compared} ({', '.join(header_parts)})",
        "",
        "> Models were built on Databricks in an isolated CI schema,",
        "> then compared against production using metadata queries (no raw data exposed).",
        "> CI tables remain available for inspection until the PR is merged.",
        "",
        "### Changed Models",
        "",
        "| Model | CI Table | Row Count | Schema | Status |",
        "|-------|----------|-----------|--------|--------|",
    ]

    for result in model_results:
        lines.append(_format_model_summary_row(result, pr_number, ci_catalog))

    # Add changed seeds section if any
    if changed_seeds:
        lines.append("")
        lines.append("### Changed Seeds")
        lines.append("")
        successful_seeds = [s for s in changed_seeds if s not in seed_failures]
        if successful_seeds:
            lines.append(f":seedling: **{len(successful_seeds)} seed(s) loaded successfully**")
            lines.append("")
            for seed in successful_seeds:
                lines.append(f"- `{seed}`")
            lines.append("")
        if seed_failures:
            lines.append(f":x: **{len(seed_failures)} seed(s) failed to load**")
            lines.append("")
            for seed in seed_failures:
                lines.append(f"- `{seed}`")
            lines.append("")

    lines.append("")
    lines.append("---")
    lines.append("")

    # Detailed sections for changed models
    for result in model_results:
        lines.extend(_format_model_detail_section(result))

    # Downstream models section with full diff tables
    if downstream_model_results:
        skipped_count = sum(1 for r in downstream_model_results if r.get("comparison_skipped"))
        compared_count = len(downstream_model_results) - skipped_count
        lines.append("### Downstream Models")
        lines.append("")
        if skipped_count and compared_count:
            lines.append(f"**{compared_count} downstream model(s) compared, {skipped_count} skipped**")
        elif skipped_count:
            lines.append(f"**{len(downstream_model_results)} downstream model(s) built, {skipped_count} comparison(s) skipped**")
        else:
            lines.append(f"**{len(downstream_model_results)} downstream model(s) compared against production**")
        lines.append("")
        lines.append("| Model | CI Table | Row Count | Schema | Status |")
        lines.append("|-------|----------|-----------|--------|--------|")
        for result in downstream_model_results:
            lines.append(_format_model_summary_row(result, pr_number, ci_catalog))
        lines.append("")

        # Detailed sections for downstream models (collapsed)
        for result in downstream_model_results:
            lines.extend(_format_model_detail_section(result))

    # Add model build failures section if any
    if model_build_failures:
        lines.append("### Build Failures")
        lines.append("")
        lines.append(f":x: **{len(model_build_failures)} model(s) failed to build**")
        lines.append("")
        for model in model_build_failures:
            lines.append(f"- `{model}`")
        lines.append("")

    # Add test failures section if any
    if test_failures:
        lines.append("### Test Failures")
        lines.append("")
        lines.append(f":x: **{len(test_failures)} test(s) failed** (models built successfully)")
        lines.append("")
        lines.append("| Test | Error |")
        lines.append("|------|-------|")
        for test in test_failures:
            if isinstance(test, dict):
                name = test.get("name", "unknown")
                message = test.get("message", "")
                lines.append(f"| `{name}` | {message} |")
            else:
                lines.append(f"| `{test}` | |")
        lines.append("")

    lines.append("---")
    lines.append(f":robot: Generated by jirade CI | Commit: `{head_sha[:7]}`")
    lines.append("")
    lines.append(f"> :broom: CI tables in `{ci_catalog}` will be cleaned up when this PR is merged.")

    return "\n".join(lines)


# =============================================================================
# CI Cleanup Functions
# =============================================================================

async def cleanup_ci_schemas(
    pr_number: int,
) -> dict[str, Any]:
    """Clean up CI schemas for a merged PR.

    This should be called after a PR is merged and the Jira ticket is being closed.
    It removes all CI schemas created for the PR (jirade_ci_{pr_number}_*).

    Args:
        pr_number: The PR number whose CI schemas should be cleaned up.

    Returns:
        Cleanup results including list of dropped schemas.
    """
    from ...config import get_settings

    settings = get_settings()

    if not settings.has_databricks:
        return {
            "success": False,
            "error": "Databricks not configured",
        }

    if not settings.databricks_ci_catalog:
        return {
            "success": False,
            "error": "CI catalog not configured (JIRADE_DATABRICKS_CI_CATALOG)",
        }

    dropped_schemas = []
    errors = []

    with DatabricksMetadataClient(
        host=settings.databricks_host,
        http_path=settings.databricks_http_path,
        auth_type=settings.databricks_auth_type,
        token=settings.databricks_token if settings.databricks_auth_type == "token" else None,
        catalog=settings.databricks_ci_catalog,
    ) as db_client:

        ci_schema_prefix = f"jirade_ci_{pr_number}_"

        try:
            # List all schemas in CI catalog matching our prefix
            schemas_result = db_client.execute_metadata_query(
                f"SHOW SCHEMAS IN {settings.databricks_ci_catalog} LIKE 'jirade_ci_{pr_number}_*'"
            )
            ci_schemas = [r.get("databaseName", r.get("namespace", "")) for r in schemas_result]

            for schema in ci_schemas:
                if schema.startswith(ci_schema_prefix):
                    full_schema = f"{settings.databricks_ci_catalog}.{schema}"
                    try:
                        db_client.drop_ci_schema(full_schema)
                        dropped_schemas.append(full_schema)
                        logger.info(f"Dropped CI schema: {full_schema}")
                    except Exception as e:
                        errors.append(f"{full_schema}: {str(e)}")
                        logger.warning(f"Failed to drop CI schema {full_schema}: {e}")

        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to list CI schemas: {str(e)}",
            }

    return {
        "success": len(errors) == 0,
        "pr_number": pr_number,
        "dropped_schemas": dropped_schemas,
        "schemas_dropped_count": len(dropped_schemas),
        "errors": errors if errors else None,
        "message": f"Cleaned up {len(dropped_schemas)} CI schema(s) for PR #{pr_number}",
    }


# =============================================================================
# Deprecation Analysis Functions
# =============================================================================

async def analyze_deprecation(
    table_name: str,
    column_name: str | None = None,
    repo_path: str = ".",
    dbt_project_subdir: str = "dbt-databricks",
) -> dict[str, Any]:
    """Analyze the impact of deprecating a table or column.

    Parses dbt manifest.json to find downstream models that reference the table.
    For column-level analysis, the agent should read the returned model files
    to verify actual column usage.

    Args:
        table_name: The table/model name to analyze.
        column_name: Optional specific column to check.
        repo_path: Local path to the repository.
        dbt_project_subdir: Subdirectory containing dbt project.

    Returns:
        Impact report with downstream dependencies.
    """
    project_dir = Path(repo_path) / dbt_project_subdir
    manifest_path = project_dir / "target" / "manifest.json"

    if not manifest_path.exists():
        # Try to generate manifest
        logger.info("Manifest not found, running dbt parse...")
        proc = await asyncio.create_subprocess_exec(
            "dbt", "parse",
            cwd=str(project_dir),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()

        if not manifest_path.exists():
            return {
                "success": False,
                "error": "Could not find or generate manifest.json. Run 'dbt parse' first.",
            }

    try:
        with open(manifest_path) as f:
            manifest = json.load(f)
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to parse manifest.json: {e}",
        }

    # Find the source model
    source_node = None
    source_node_id = None

    for node_id, node in manifest.get("nodes", {}).items():
        if node.get("name") == table_name:
            source_node = node
            source_node_id = node_id
            break

    # Also check sources
    if not source_node:
        for source_id, source in manifest.get("sources", {}).items():
            if source.get("name") == table_name:
                source_node = source
                source_node_id = source_id
                break

    if not source_node:
        return {
            "success": False,
            "error": f"Table '{table_name}' not found in manifest. Check the name or run 'dbt parse'.",
        }

    # Find all downstream dependencies
    downstream = _find_downstream_models(manifest, source_node_id)

    # Categorize models by type (mart, dim, fact, staging, intermediate)
    categorized = {
        "exposed_to_users": [],  # mart_, dim_, fact_
        "intermediate": [],       # int_
        "staging": [],            # stg_
        "other": [],
    }

    models_to_verify = []

    for node_id in downstream:
        node = manifest.get("nodes", {}).get(node_id, {})
        name = node.get("name", "")
        path = node.get("original_file_path", "")
        full_path = str(project_dir / path) if path else ""

        model_info = {
            "name": name,
            "path": full_path,
            "resource_type": node.get("resource_type", ""),
        }

        # Categorize
        if name.startswith(("mart_", "dim_", "fact_", "fct_")):
            categorized["exposed_to_users"].append(model_info)
        elif name.startswith("int_"):
            categorized["intermediate"].append(model_info)
        elif name.startswith("stg_"):
            categorized["staging"].append(model_info)
        else:
            categorized["other"].append(model_info)

        if full_path:
            models_to_verify.append(full_path)

    # Generate report
    report_lines = [
        f"## Deprecation Impact: `{table_name}`" + (f".`{column_name}`" if column_name else ""),
        "",
    ]

    if categorized["exposed_to_users"]:
        report_lines.append("### :warning: User-Exposed Models (BREAKING)")
        report_lines.append("")
        report_lines.append("These models are marts/dims/facts that end users may depend on:")
        report_lines.append("")
        for m in categorized["exposed_to_users"]:
            report_lines.append(f"- `{m['name']}` - {m['path']}")
        report_lines.append("")

    if categorized["intermediate"]:
        report_lines.append("### Intermediate Models")
        report_lines.append("")
        for m in categorized["intermediate"]:
            report_lines.append(f"- `{m['name']}` - {m['path']}")
        report_lines.append("")

    if categorized["staging"]:
        report_lines.append("### Staging Models")
        report_lines.append("")
        for m in categorized["staging"]:
            report_lines.append(f"- `{m['name']}` - {m['path']}")
        report_lines.append("")

    if categorized["other"]:
        report_lines.append("### Other Models")
        report_lines.append("")
        for m in categorized["other"]:
            report_lines.append(f"- `{m['name']}` - {m['path']}")
        report_lines.append("")

    if column_name:
        report_lines.append("### Next Steps")
        report_lines.append("")
        report_lines.append(f"To verify column `{column_name}` usage, read the model files above and check:")
        report_lines.append("1. Is the column selected/referenced in the model?")
        report_lines.append("2. Is it passed through to the output?")
        report_lines.append("3. Is it used in joins/filters but not exposed?")
        report_lines.append("")

    return {
        "success": True,
        "table_name": table_name,
        "column_name": column_name,
        "downstream_count": len(downstream),
        "exposed_to_users": len(categorized["exposed_to_users"]),
        "categorized": categorized,
        "models_to_verify": models_to_verify,
        "report": "\n".join(report_lines),
    }


def _find_downstream_models(manifest: dict[str, Any], source_node_id: str) -> list[str]:
    """Find all downstream models that depend on a source node.

    Uses the parent_map in manifest to trace dependencies.

    Args:
        manifest: Parsed manifest.json.
        source_node_id: Node ID of the source model/table.

    Returns:
        List of downstream node IDs.
    """
    downstream = set()
    child_map = manifest.get("child_map", {})

    # If child_map is available, use it directly
    if child_map:
        to_visit = list(child_map.get(source_node_id, []))
        while to_visit:
            node_id = to_visit.pop()
            if node_id not in downstream:
                downstream.add(node_id)
                to_visit.extend(child_map.get(node_id, []))
        return list(downstream)

    # Fallback: build reverse map from parent_map
    parent_map = manifest.get("parent_map", {})
    reverse_map: dict[str, list[str]] = {}

    for node_id, parents in parent_map.items():
        for parent in parents:
            if parent not in reverse_map:
                reverse_map[parent] = []
            reverse_map[parent].append(node_id)

    to_visit = list(reverse_map.get(source_node_id, []))
    while to_visit:
        node_id = to_visit.pop()
        if node_id not in downstream:
            downstream.add(node_id)
            to_visit.extend(reverse_map.get(node_id, []))

    return list(downstream)
