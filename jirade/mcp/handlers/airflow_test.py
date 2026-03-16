"""Airflow DAG SQL testing tool handler.

Parses Airflow DAG files to extract SQL from CustomDatabricksSqlOperator tasks,
then executes them against a CI schema on Databricks to validate they work.
SQL catalog/schema references are rewritten to target the CI catalog.
"""

import logging
import re
import sys
import types
from typing import Any
from unittest.mock import MagicMock

from databricks import sql as databricks_sql

from ...clients.github_client import GitHubClient
from ...config import get_settings

logger = logging.getLogger(__name__)


async def handle_airflow_test_tool(
    name: str,
    arguments: dict[str, Any],
    progress_cb: Any | None = None,
) -> dict[str, Any]:
    """Handle an Airflow test tool call."""
    if name == "jirade_test_airflow_dag":
        return await _test_airflow_dag(arguments, progress_cb)
    else:
        raise ValueError(f"Unknown tool: {name}")


def _extract_sql_operators(dag_path: str) -> list[dict[str, Any]]:
    """Extract SQL from CustomDatabricksSqlOperator tasks by executing the DAG module.

    Uses mock Airflow objects to capture operator instantiations. The DAG file is
    executed in a sandboxed namespace where Airflow imports are replaced with mocks,
    but CustomDatabricksSqlOperator captures its task_id and sql arguments.

    Returns:
        List of dicts with 'task_id' and 'sql' (list of SQL strings).
    """
    captured_operators: list[dict[str, Any]] = []

    class FakeOperator:
        """Mock operator that captures task_id and sql kwargs."""

        def __init__(self, *args, **kwargs):
            task_id = kwargs.get("task_id")
            sql = kwargs.get("sql")
            if task_id and sql:
                if isinstance(sql, str):
                    sql = [sql]
                captured_operators.append({"task_id": task_id, "sql": sql})

        def __rshift__(self, other):
            return other

        def __rrshift__(self, other):
            return self

    class FakeSensor(FakeOperator):
        pass

    class FakeDAG:
        def __init__(self, *args, **kwargs):
            self.dag_id = kwargs.get("dag_id", "")
            self.default_args = kwargs.get("default_args", {})
            self.tags = []

    # Build a mock module system for Airflow imports
    def _make_mock_module(name: str) -> types.ModuleType:
        mod = types.ModuleType(name)
        # Return a flexible mock for any attribute access
        mod.__dict__["__getattr__"] = lambda attr: MagicMock()
        return mod

    # Save original modules to restore later
    saved_modules = {}
    mock_modules = [
        "airflow", "airflow.models", "airflow.models.dag", "airflow.operators",
        "airflow.operators.empty", "airflow.operators.python",
        "airflow.utils", "airflow.utils.task_group", "airflow.utils.trigger_rule",
        "airflow_metaplane",
        "pendulum",
        "utils", "utils.airflow_tags", "utils.airflow_patterns",
        "utils.dag_manifest", "utils.variables", "utils.dag_manifest_databricks",
        "utils.dbt_manifest",
        "dags", "dags.commons",
        "operators", "operators.databricks_sql_operator",
        "operators.copied_from_v2_8", "operators.copied_from_v2_8.sensors_external_task",
        "operators.dbt_databricks_ecs_run_task_operator",
        "services", "services.slack", "services.slack.task_failure",
        "services.census",
    ]

    for mod_name in mock_modules:
        saved_modules[mod_name] = sys.modules.get(mod_name)
        sys.modules[mod_name] = _make_mock_module(mod_name)

    # Wire up specific objects that the DAG code uses
    sys.modules["airflow.models.dag"].DAG = FakeDAG
    sys.modules["airflow.operators.empty"].EmptyOperator = FakeOperator
    sys.modules["operators.databricks_sql_operator"].CustomDatabricksSqlOperator = FakeOperator
    sys.modules["operators.copied_from_v2_8.sensors_external_task"].ExternalTaskSensor = FakeSensor

    # pendulum.datetime needs to return something callable
    sys.modules["pendulum"].datetime = lambda *a, **kw: MagicMock()

    # utils.variables constants
    sys.modules["utils.variables"].DATABRICKS_CONN_ID = "mock_conn"
    sys.modules["utils.variables"].DATABRICKS_ENDPOINT_HTTP_PATH = "/mock"
    sys.modules["utils.variables"].DATABRICKS_CATALOG_PREFIX = ""

    # utils.dag_manifest
    sys.modules["utils.dag_manifest"].get_dag_schedule_interval = lambda dag_id: "@daily"

    # utils.airflow_tags
    sys.modules["utils.airflow_tags"].AirflowTagDagType = MagicMock()
    sys.modules["utils.airflow_tags"].AirflowTagOwner = MagicMock()
    sys.modules["utils.airflow_tags"].determine_airflow_schedule_mode = lambda dag: "scheduled"
    sys.modules["utils.airflow_tags"].generate_airflow_tags = lambda **kw: []

    # dags.commons
    sys.modules["dags.commons"].MAX_POKE_INTERVAL_IN_SECONDS = 600
    sys.modules["dags.commons"].MIN_POKE_INTERVAL_IN_SECONDS = 240
    sys.modules["dags.commons"].add_timedelta_to_datetime = lambda **kw: None

    # services.slack.task_failure
    sys.modules["services.slack"].task_failure = MagicMock()

    try:
        with open(dag_path) as f:
            source = f.read()

        exec(compile(source, dag_path, "exec"), {"__builtins__": __builtins__, "__file__": dag_path})
    except Exception as e:
        logger.warning(f"Error executing DAG file (may be partial): {e}")
    finally:
        # Restore original modules
        for mod_name, original in saved_modules.items():
            if original is None:
                sys.modules.pop(mod_name, None)
            else:
                sys.modules[mod_name] = original

    return captured_operators


def _rewrite_sql_for_ci(
    sql_statements: list[str],
    original_catalog: str,
    original_schema: str,
    ci_catalog: str,
    ci_schema: str,
) -> list[str]:
    """Rewrite SQL statements to target a CI schema instead of production.

    Only rewrites snapshot/target table references (the tables being created/inserted into).
    Source table references (production reads) are left unchanged so CI tests read real data.
    """
    rewritten = []
    old_prefix = f"{original_catalog}.{original_schema}."
    new_prefix = f"{ci_catalog}.{ci_schema}."

    for stmt in sql_statements:
        # Find all table references and determine which are targets vs sources
        new_stmt = _rewrite_target_tables(stmt, old_prefix, new_prefix)
        rewritten.append(new_stmt)

    return rewritten


def _rewrite_target_tables(sql: str, old_prefix: str, new_prefix: str) -> str:
    """Rewrite only target (written-to) tables, leaving source (read-from) tables alone.

    For CREATE TABLE: rewrite only the table being created, not the source in AS SELECT
    For INSERT INTO: rewrite only the INSERT target and the NOT EXISTS subquery target,
                     but not the FROM source table
    """
    normalized = " ".join(sql.split()).upper()

    if normalized.strip().startswith("CREATE TABLE"):
        # CREATE TABLE IF NOT EXISTS <target> AS SELECT ... FROM <source> ...
        # Rewrite only the target (before AS SELECT)
        as_pos = sql.upper().find(" AS ")
        if as_pos > 0:
            before_as = sql[:as_pos].replace(old_prefix, new_prefix)
            after_as = sql[as_pos:]  # Leave source table references intact
            return before_as + after_as
        return sql.replace(old_prefix, new_prefix)

    elif normalized.strip().startswith("INSERT"):
        # INSERT INTO <target> SELECT ... FROM <source> WHERE NOT EXISTS (SELECT 1 FROM <target>)
        # Need to rewrite <target> references but not the <source> in FROM
        # Strategy: replace all, then restore the FROM source
        #
        # Actually, for snapshot patterns the target has _snapshot suffix while source doesn't.
        # The safest approach: find snapshot tables and only rewrite those.
        result = sql
        # Find all table refs with the old prefix
        pattern = re.compile(re.escape(old_prefix) + r"(\w+)")
        tables = set(m.group(1) for m in pattern.finditer(sql))

        # Snapshot/target tables end with _snapshot; source tables don't
        for table in tables:
            if table.endswith("_snapshot"):
                result = result.replace(f"{old_prefix}{table}", f"{new_prefix}{table}")
        return result

    return sql.replace(old_prefix, new_prefix)


def _extract_table_refs(sql_statements: list[str]) -> set[tuple[str, str, str]]:
    """Extract fully-qualified table references (catalog.schema.table) from SQL."""
    refs = set()
    pattern = re.compile(r"(\w+)\.(\w+)\.(\w+)")
    for stmt in sql_statements:
        for match in pattern.finditer(stmt):
            refs.add((match.group(1), match.group(2), match.group(3)))
    return refs


async def _test_airflow_dag(
    arguments: dict[str, Any],
    progress_cb: Any | None = None,
) -> dict[str, Any]:
    """Test Airflow DAG SQL statements against Databricks CI schema.

    1. Parses the DAG file to extract SQL from CustomDatabricksSqlOperator
    2. Rewrites target table SQL to use a CI schema (source tables read from production)
    3. Executes each statement and records results
    4. Tests idempotency by re-running INSERT statements
    5. Cleans up the CI schema
    """
    settings = get_settings()

    async def _notify(progress: float, total: float, message: str) -> None:
        if progress_cb:
            try:
                await progress_cb(progress, total, message)
            except Exception:
                pass

    dag_path = arguments["dag_path"]
    ci_schema_name = arguments.get("ci_schema", "jirade_airflow_test")
    cleanup = arguments.get("cleanup", True)
    test_idempotency = arguments.get("test_idempotency", True)
    post_to_pr = arguments.get("post_to_pr", False)
    pr_owner = arguments.get("owner")
    pr_repo = arguments.get("repo")
    pr_number = arguments.get("pr_number")

    if not settings.has_databricks:
        return {
            "success": False,
            "error": "Databricks not configured. Set JIRADE_DATABRICKS_HOST and JIRADE_DATABRICKS_HTTP_PATH.",
        }

    if not settings.databricks_ci_catalog:
        return {
            "success": False,
            "error": "No CI catalog configured. Set JIRADE_DATABRICKS_CI_CATALOG.",
        }

    # Step 1: Parse the DAG
    await _notify(5, 100, "Parsing DAG file...")
    try:
        operators = _extract_sql_operators(dag_path)
    except Exception as e:
        return {"success": False, "error": f"Failed to parse DAG: {e}"}

    if not operators:
        return {"success": False, "error": "No CustomDatabricksSqlOperator tasks found in DAG."}

    # Detect which catalog/schema the SQL targets
    all_sql = [stmt for op in operators for stmt in op["sql"]]
    table_refs = _extract_table_refs(all_sql)

    if not table_refs:
        return {"success": False, "error": "No table references found in SQL statements."}

    # Find the most common catalog.schema pair (the production target)
    from collections import Counter

    schema_counts = Counter((cat, schema) for cat, schema, _ in table_refs)
    prod_catalog, prod_schema = schema_counts.most_common(1)[0][0]

    ci_catalog = settings.databricks_ci_catalog
    ci_schema = ci_schema_name

    await _notify(10, 100, f"Detected production target: {prod_catalog}.{prod_schema}")
    await _notify(12, 100, f"CI target: {ci_catalog}.{ci_schema}")

    # Step 2: Connect to Databricks
    await _notify(15, 100, "Connecting to Databricks...")
    try:
        conn_kwargs = {
            "server_hostname": settings.databricks_host,
            "http_path": settings.databricks_http_path,
        }
        if settings.databricks_auth_type == "token":
            conn_kwargs["access_token"] = settings.databricks_token
        else:
            conn_kwargs["auth_type"] = "databricks-oauth"

        conn = databricks_sql.connect(**conn_kwargs)
        cursor = conn.cursor()
    except Exception as e:
        return {"success": False, "error": f"Failed to connect to Databricks: {e}"}

    results = {
        "success": True,
        "dag_path": dag_path,
        "production_target": f"{prod_catalog}.{prod_schema}",
        "ci_target": f"{ci_catalog}.{ci_schema}",
        "operators": [],
    }

    try:
        # Step 3: Create CI schema
        await _notify(20, 100, f"Creating CI schema: {ci_catalog}.{ci_schema}")
        cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {ci_catalog}.{ci_schema}")

        # Step 4: Execute each operator's SQL
        total_ops = len(operators)
        for i, op in enumerate(operators):
            op_progress = 25 + (i / total_ops) * 40
            await _notify(op_progress, 100, f"Testing operator: {op['task_id']}")

            op_result = {
                "task_id": op["task_id"],
                "statements": [],
            }

            # Rewrite SQL for CI (only target tables, source tables read from prod)
            ci_sql = _rewrite_sql_for_ci(
                op["sql"], prod_catalog, prod_schema, ci_catalog, ci_schema
            )

            for stmt in ci_sql:
                stmt_result = {"sql": stmt.strip(), "status": "ok"}
                try:
                    cursor.execute(stmt)

                    # Get row count for CI target table
                    table_match = re.search(
                        rf"{re.escape(ci_catalog)}\.{re.escape(ci_schema)}\.(\w+)",
                        stmt,
                    )
                    if table_match:
                        table_name = f"{ci_catalog}.{ci_schema}.{table_match.group(1)}"
                        try:
                            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                            row = cursor.fetchone()
                            stmt_result["row_count"] = row[0]
                        except Exception:
                            pass

                except Exception as e:
                    stmt_result["status"] = "error"
                    stmt_result["error"] = str(e)
                    results["success"] = False

                op_result["statements"].append(stmt_result)

            results["operators"].append(op_result)

        # Step 5: Idempotency test
        if test_idempotency and results["success"]:
            await _notify(70, 100, "Testing idempotency (re-running INSERT statements)...")
            idempotency_results = []

            for op in operators:
                ci_sql = _rewrite_sql_for_ci(
                    op["sql"], prod_catalog, prod_schema, ci_catalog, ci_schema
                )

                for stmt in ci_sql:
                    normalized = " ".join(stmt.split()).upper()
                    if not normalized.strip().startswith("INSERT"):
                        continue

                    table_match = re.search(
                        rf"{re.escape(ci_catalog)}\.{re.escape(ci_schema)}\.(\w+)",
                        stmt,
                    )
                    if not table_match:
                        continue

                    table_name = f"{ci_catalog}.{ci_schema}.{table_match.group(1)}"
                    try:
                        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                        before_count = cursor.fetchone()[0]

                        cursor.execute(stmt)

                        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                        after_count = cursor.fetchone()[0]

                        idempotency_results.append({
                            "table": table_name,
                            "before": before_count,
                            "after": after_count,
                            "idempotent": before_count == after_count,
                        })

                        if before_count != after_count:
                            results["success"] = False

                    except Exception as e:
                        idempotency_results.append({
                            "table": table_name,
                            "error": str(e),
                            "idempotent": False,
                        })
                        results["success"] = False

            results["idempotency"] = idempotency_results

        # Step 6: Cleanup
        if cleanup:
            await _notify(90, 100, f"Cleaning up CI schema: {ci_catalog}.{ci_schema}")
            try:
                cursor.execute(f"DROP SCHEMA IF EXISTS {ci_catalog}.{ci_schema} CASCADE")
                results["cleaned_up"] = True
            except Exception as e:
                results["cleaned_up"] = False
                results["cleanup_error"] = str(e)
        else:
            results["cleaned_up"] = False
            results["ci_schema_kept"] = f"{ci_catalog}.{ci_schema}"

    except Exception as e:
        results["success"] = False
        results["error"] = str(e)
    finally:
        try:
            cursor.close()
            conn.close()
        except Exception:
            pass

    await _notify(95, 100, "Generating report...")

    # Build human-readable summary
    results["summary"] = _format_summary(results)

    # Post to PR if requested
    if post_to_pr and pr_owner and pr_repo and pr_number:
        await _notify(97, 100, f"Posting report to PR #{pr_number}...")
        try:
            gh = GitHubClient(owner=pr_owner, repo=pr_repo, token=settings.github_token)
            marker = "<!-- airflow-dag-test-report -->"
            body = f"{marker}\n{results['summary']}"
            await gh.upsert_pr_comment(pr_number, body, marker=marker)
            results["posted_to_pr"] = pr_number
        except Exception as e:
            results["post_to_pr_error"] = str(e)

    await _notify(100, 100, "Done!")
    return results


def _format_summary(results: dict[str, Any]) -> str:
    """Format results into a human-readable summary."""
    lines = []
    status = "PASS" if results["success"] else "FAIL"
    lines.append(f"## Airflow DAG SQL Test: {status}")
    lines.append(f"- **DAG**: `{results['dag_path']}`")
    lines.append(f"- **Production target**: `{results['production_target']}`")
    lines.append(f"- **CI target**: `{results['ci_target']}`")
    lines.append("")

    lines.append("### SQL Execution")
    lines.append("| Task ID | Statement | Status | Rows |")
    lines.append("|---|---|---|---|")

    for op in results.get("operators", []):
        for stmt in op.get("statements", []):
            sql_type = stmt["sql"].strip().split()[0].upper()
            row_count = stmt.get("row_count", "-")
            if isinstance(row_count, int):
                row_count = f"{row_count:,}"
            status_icon = "OK" if stmt["status"] == "ok" else f"ERROR: {stmt.get('error', '')[:60]}"
            lines.append(f"| `{op['task_id']}` | {sql_type} | {status_icon} | {row_count} |")

    if "idempotency" in results:
        lines.append("")
        lines.append("### Idempotency Test")
        lines.append("| Table | Before | After | Idempotent |")
        lines.append("|---|---|---|---|")
        for item in results["idempotency"]:
            if "error" in item:
                lines.append(f"| `{item['table']}` | - | - | ERROR: {item['error'][:50]} |")
            else:
                icon = "Yes" if item["idempotent"] else "No"
                lines.append(f"| `{item['table']}` | {item['before']:,} | {item['after']:,} | {icon} |")

    return "\n".join(lines)
