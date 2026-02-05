"""dbt diff tool handlers for MCP server.

This module provides functionality for comparing dbt model outputs between
a base branch and a PR branch. Supports two approaches:

1. DuckDB local diff: Uses DuckDB with agent-generated fixtures for quick testing
2. Databricks CI: Runs dbt build on Databricks and compares against prod (metadata only)

The Databricks CI approach:
- Runs dbt build with modified models +1 dependents in isolated schema
- Compares CI tables against production using metadata queries only
- No raw data is exposed to the agent
- Always cleans up CI schema after completion
"""

import asyncio
import json
import logging
import os
import re
import shutil
import tempfile
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


class DbtDiffRunner:
    """Runs dbt model diffs between base and PR branches.

    Uses dbt compile to generate SQL, then executes directly on DuckDB.
    No dbt-duckdb adapter needed.
    """

    def __init__(
        self,
        repo_path: str,
        dbt_project_subdir: str = "dbt-databricks",
        work_dir: str | None = None,
    ):
        """Initialize the diff runner.

        Args:
            repo_path: Path to the local git repository.
            dbt_project_subdir: Subdirectory containing the dbt project.
            work_dir: Working directory for temp files. Created if not provided.
        """
        self.repo_path = Path(repo_path)
        self.dbt_project_subdir = dbt_project_subdir
        self.work_dir = Path(work_dir) if work_dir else None
        self._temp_dir: tempfile.TemporaryDirectory | None = None
        self._venv_path: Path | None = None

    async def __aenter__(self) -> "DbtDiffRunner":
        """Set up the diff environment including isolated venv with dbt."""
        if self.work_dir is None:
            self._temp_dir = tempfile.TemporaryDirectory(prefix="dbt_diff_")
            self.work_dir = Path(self._temp_dir.name)

        # Create subdirectories
        (self.work_dir / "base").mkdir(parents=True, exist_ok=True)
        (self.work_dir / "pr").mkdir(parents=True, exist_ok=True)
        (self.work_dir / "fixtures").mkdir(parents=True, exist_ok=True)

        # Set up isolated virtual environment with dbt (cached for speed)
        await self._setup_venv()

        return self

    def _get_cache_dir(self) -> Path:
        """Get the cache directory for persistent venv storage."""
        cache_dir = Path.home() / ".cache" / "jirade" / "dbt-diff"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    async def _get_cached_packages(self, project_dir: Path) -> Path | None:
        """Get cached dbt_packages for a project based on packages.yml hash.

        Args:
            project_dir: Path to the dbt project.

        Returns:
            Path to cached packages or None if not cached.
        """
        import hashlib

        packages_file = project_dir / "packages.yml"
        if not packages_file.exists():
            return None

        # Hash the packages.yml content
        content = packages_file.read_bytes()
        packages_hash = hashlib.md5(content).hexdigest()[:12]

        cached_path = self._get_cache_dir() / f"packages_{packages_hash}"
        if cached_path.exists():
            return cached_path

        return None

    async def _cache_packages(self, project_dir: Path) -> None:
        """Cache dbt_packages for a project.

        Args:
            project_dir: Path to the dbt project with dbt_packages installed.
        """
        import hashlib

        packages_file = project_dir / "packages.yml"
        dbt_packages = project_dir / "dbt_packages"

        if not packages_file.exists() or not dbt_packages.exists():
            return

        # Hash the packages.yml content
        content = packages_file.read_bytes()
        packages_hash = hashlib.md5(content).hexdigest()[:12]

        cached_path = self._get_cache_dir() / f"packages_{packages_hash}"
        if not cached_path.exists():
            logger.info(f"Caching dbt_packages to {cached_path}")
            shutil.copytree(dbt_packages, cached_path)

    async def _find_compatible_python(self) -> str:
        """Find a Python version compatible with dbt (3.9-3.12)."""
        # dbt-databricks supports Python 3.9-3.12, try in order of preference
        candidates = ["python3.12", "python3.11", "python3.10", "python3.9", "python3"]

        for python in candidates:
            proc = await asyncio.create_subprocess_exec(
                "which", python,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()

            if proc.returncode == 0:
                python_path = stdout.decode().strip()

                # Verify version is compatible (3.9-3.12)
                proc = await asyncio.create_subprocess_exec(
                    python_path, "--version",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await proc.communicate()
                version_str = stdout.decode().strip()  # "Python 3.11.14"

                try:
                    version = version_str.split()[1]  # "3.11.14"
                    major, minor = map(int, version.split(".")[:2])
                    if major == 3 and 9 <= minor <= 12:
                        logger.info(f"Found compatible Python: {python_path} ({version_str})")
                        return python_path
                except (IndexError, ValueError):
                    continue

        raise RuntimeError(
            "No compatible Python found. dbt requires Python 3.9-3.12. "
            "Please install one of: python3.12, python3.11, python3.10, python3.9"
        )

    async def _setup_venv(self) -> None:
        """Set up an isolated virtual environment with dbt-databricks installed.

        Uses a cached venv in ~/.cache/jirade/dbt-diff/ for speed.
        The venv is only recreated if it doesn't exist or dbt is broken.
        """
        cache_dir = self._get_cache_dir()
        self._venv_path = cache_dir / "venv"
        dbt_path = self._venv_path / "bin" / "dbt"

        # Check if cached venv exists and works
        if self._venv_path.exists() and dbt_path.exists():
            # Verify dbt works
            proc = await asyncio.create_subprocess_exec(
                str(dbt_path), "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.communicate()

            if proc.returncode == 0:
                logger.info(f"Using cached dbt venv at {self._venv_path}")
                return

            # dbt is broken, recreate venv
            logger.info("Cached venv is broken, recreating...")
            shutil.rmtree(self._venv_path)

        # Find a compatible Python version
        python_path = await self._find_compatible_python()

        logger.info(f"Creating virtual environment at {self._venv_path}")

        # Create venv with compatible Python
        proc = await asyncio.create_subprocess_exec(
            python_path, "-m", "venv", str(self._venv_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"Failed to create venv: {stderr.decode()}")

        # Install dbt-duckdb (for local compilation without network calls)
        pip_path = self._venv_path / "bin" / "pip"
        logger.info("Installing dbt-duckdb in isolated environment (first run, may take ~20s)...")

        proc = await asyncio.create_subprocess_exec(
            str(pip_path), "install", "--quiet", "dbt-duckdb>=1.9.0",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            raise RuntimeError(f"Failed to install dbt-duckdb: {stderr.decode()}")

        logger.info("dbt-duckdb installed successfully")

    def _get_dbt_path(self) -> str:
        """Get path to dbt in the isolated venv."""
        return str(self._venv_path / "bin" / "dbt")

    def _create_compile_profile(self, project_dir: Path) -> Path:
        """Create a temporary profiles.yml for compilation.

        Reads dbt_project.yml to get the profile name, then creates a minimal
        profile that allows compilation without real credentials.

        Args:
            project_dir: Path to the dbt project directory.

        Returns:
            Path to the profiles directory.
        """
        # Read dbt_project.yml to get profile name
        project_file = project_dir / "dbt_project.yml"
        with open(project_file) as f:
            project_config = yaml.safe_load(f)

        profile_name = project_config.get("profile", "default")

        # Create profiles directory
        profiles_dir = self.work_dir / "profiles"
        profiles_dir.mkdir(exist_ok=True)

        # Create duckdb profile for local compilation (no network calls)
        profiles_content = f"""
{profile_name}:
  target: compile
  outputs:
    compile:
      type: duckdb
      path: ":memory:"
      threads: 4
"""
        profiles_path = profiles_dir / "profiles.yml"
        profiles_path.write_text(profiles_content)

        return profiles_dir

    async def __aexit__(self, *args) -> None:
        """Clean up the diff environment."""
        if self._temp_dir:
            self._temp_dir.cleanup()

    async def checkout_branch(self, branch: str, target_dir: str) -> None:
        """Checkout a branch to a target directory using git archive.

        Args:
            branch: Git branch name or commit SHA.
            target_dir: Either "base" or "pr" to indicate which directory.
        """
        target_path = self.work_dir / target_dir / self.dbt_project_subdir

        if target_path.exists():
            shutil.rmtree(target_path)

        logger.info(f"Extracting {branch} to {target_path}")
        target_path.parent.mkdir(parents=True, exist_ok=True)

        # Use git archive to extract the dbt project from a specific branch
        proc = await asyncio.create_subprocess_exec(
            "git", "archive", "--format=tar", branch, "--", self.dbt_project_subdir,
            cwd=str(self.repo_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        tar_data, stderr = await proc.communicate()

        if proc.returncode != 0:
            raise RuntimeError(f"Failed to archive branch {branch}: {stderr.decode()}")

        # Extract the tar data
        proc = await asyncio.create_subprocess_exec(
            "tar", "-xf", "-",
            cwd=str(self.work_dir / target_dir),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate(input=tar_data)

        # Strip Databricks-specific configs that DuckDB doesn't understand
        self._strip_databricks_configs(target_path)

    def _strip_databricks_configs(self, project_dir: Path) -> None:
        """Strip Databricks-specific configs from yml files for DuckDB compatibility.

        Args:
            project_dir: Path to the dbt project directory.
        """
        import re

        for yml_file in project_dir.rglob("*.yml"):
            try:
                content = yml_file.read_text()
                if "catalog:" in content:
                    # Remove catalog: lines (Databricks Unity Catalog specific)
                    content = re.sub(r"^\s*\+?catalog:.*$", "", content, flags=re.MULTILINE)
                    yml_file.write_text(content)
            except Exception:
                pass  # Skip files that can't be read/written

    def model_exists(self, model_name: str, branch_dir: str) -> bool:
        """Check if a model exists in the given branch.

        Args:
            model_name: Name of the model.
            branch_dir: Either "base" or "pr".

        Returns:
            True if model exists, False otherwise.
        """
        project_dir = self.work_dir / branch_dir / self.dbt_project_subdir
        models_dir = project_dir / "models"

        if not models_dir.exists():
            return False

        # Search for the model file
        sql_files = list(models_dir.rglob(f"{model_name}.sql"))
        return len(sql_files) > 0

    async def compile_model(self, model_name: str, branch_dir: str) -> dict[str, Any]:
        """Compile a dbt model to get the SQL.

        Uses the existing dbt installation (dbt-databricks) to compile.

        Args:
            model_name: Name of the model to compile.
            branch_dir: Either "base" or "pr".

        Returns:
            Dict with compiled SQL and any errors.
        """
        project_dir = self.work_dir / branch_dir / self.dbt_project_subdir

        # Check if model exists first
        if not self.model_exists(model_name, branch_dir):
            return {
                "success": False,
                "exists": False,
                "error": f"Model {model_name} does not exist in {branch_dir} branch",
                "sql": None,
            }

        # Create a temporary profile for compilation
        profiles_dir = self._create_compile_profile(project_dir)
        env = {**os.environ, "DBT_PROFILES_DIR": str(profiles_dir)}

        # Use cached dbt_packages if available, otherwise run dbt deps
        dbt_packages_dir = project_dir / "dbt_packages"
        cached_packages = await self._get_cached_packages(project_dir)

        if cached_packages and not dbt_packages_dir.exists():
            logger.info("Using cached dbt_packages")
            shutil.copytree(cached_packages, dbt_packages_dir)
        elif not dbt_packages_dir.exists():
            logger.info("Running dbt deps (~3-5s)...")
            proc = await asyncio.create_subprocess_exec(
                self._get_dbt_path(), "deps",
                cwd=str(project_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            stdout, stderr = await proc.communicate()

            if proc.returncode != 0:
                error_msg = stdout.decode() or stderr.decode()
                return {
                    "success": False,
                    "exists": True,
                    "error": f"dbt deps failed: {error_msg[:500]}",
                    "sql": None,
                }

            await self._cache_packages(project_dir)
            logger.info("dbt deps completed")

        # Run dbt compile for the specific model using isolated venv
        logger.info(f"Compiling {model_name} (~10-15s)...")
        proc = await asyncio.create_subprocess_exec(
            self._get_dbt_path(), "compile", "--select", model_name,
            cwd=str(project_dir),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            # dbt outputs errors to stdout, not stderr
            error_msg = stdout.decode() or stderr.decode()
            return {
                "success": False,
                "exists": True,
                "error": f"Compile failed: {error_msg[:500]}",
                "sql": None,
            }

        logger.info(f"Compiled {model_name}")

        # Find the compiled SQL file
        compiled_path = project_dir / "target" / "compiled"
        sql_files = list(compiled_path.rglob(f"{model_name}.sql"))

        if not sql_files:
            return {
                "success": False,
                "exists": True,
                "error": f"Compiled SQL not found for {model_name}",
                "sql": None,
            }

        sql = sql_files[0].read_text()

        return {
            "success": True,
            "exists": True,
            "sql": sql,
            "path": str(sql_files[0]),
        }

    def load_fixtures_to_duckdb(
        self,
        conn: Any,
        fixtures: dict[str, str | Path | list[str]],
    ) -> None:
        """Load fixtures into DuckDB as tables.

        Supports multiple fixture formats:
        - CSV file path (Path or str path that exists)
        - CSV content as string (auto-detected by header row)
        - SQL statements as list of strings ["CREATE TABLE...", "INSERT INTO..."]
        - SQL statements as single string (multiple statements separated by ;)

        Args:
            conn: DuckDB connection.
            fixtures: Dict mapping table names to fixture data.
                      Table names can be schema-qualified (e.g., "dashboard.users").
                      For SQL fixtures, use "_sql" as the table name to execute raw SQL.
        """
        created_schemas: set[str] = set()

        for table_name, source in fixtures.items():
            # Special case: raw SQL statements (use "_sql" as key)
            if table_name == "_sql":
                if isinstance(source, list):
                    for stmt in source:
                        conn.execute(stmt)
                        logger.info(f"Executed SQL fixture statement")
                elif isinstance(source, str):
                    # Split by semicolon and execute each statement
                    for stmt in source.split(";"):
                        stmt = stmt.strip()
                        if stmt:
                            conn.execute(stmt)
                    logger.info(f"Executed SQL fixture statements")
                continue

            # Handle schema-qualified table names (e.g., "dashboard.users")
            if "." in table_name:
                schema, table = table_name.split(".", 1)
                if schema not in created_schemas:
                    conn.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')
                    created_schemas.add(schema)
                qualified_name = f'"{schema}"."{table}"'
                safe_filename = f"{schema}_{table}"
            else:
                qualified_name = f'"{table_name}"'
                safe_filename = table_name

            # Detect fixture type and load accordingly
            if isinstance(source, list):
                # List of SQL statements for this table
                for stmt in source:
                    conn.execute(stmt)
                logger.info(f"Loaded fixture table via SQL: {qualified_name}")

            elif isinstance(source, Path) or (isinstance(source, str) and os.path.exists(source)):
                # Load from CSV file
                conn.execute(f"CREATE TABLE {qualified_name} AS SELECT * FROM read_csv_auto('{source}')")
                logger.info(f"Loaded fixture table from CSV file: {qualified_name}")

            elif isinstance(source, str) and source.strip().upper().startswith(("CREATE ", "INSERT ")):
                # SQL string - execute directly
                for stmt in source.split(";"):
                    stmt = stmt.strip()
                    if stmt:
                        conn.execute(stmt)
                logger.info(f"Loaded fixture table via SQL: {qualified_name}")

            else:
                # CSV content as string - write to temp file first
                temp_csv = self.work_dir / "fixtures" / f"{safe_filename}.csv"
                temp_csv.write_text(source)
                conn.execute(f"CREATE TABLE {qualified_name} AS SELECT * FROM read_csv_auto('{temp_csv}')")
                logger.info(f"Loaded fixture table from CSV content: {qualified_name}")

    def adapt_sql_for_duckdb(self, sql: str) -> str:
        """Adapt Databricks SQL to DuckDB syntax.

        Handles common differences between Databricks and DuckDB SQL.

        Args:
            sql: Databricks-compiled SQL.

        Returns:
            DuckDB-compatible SQL.
        """
        import re

        adapted = sql

        # Remove 3-part catalog references (catalog.schema.table -> schema.table)
        # Handle double-quoted identifiers: "catalog"."schema"."table"
        adapted = re.sub(r'"[^"]+"\."([^"]+)"\."([^"]+)"', r'"\1"."\2"', adapted)
        # Handle backtick identifiers: `catalog`.`schema`.`table`
        adapted = re.sub(r'`[^`]+`\.`([^`]+)`\.`([^`]+)`', r'`\1`.`\2`', adapted)
        # Handle unquoted: catalog.schema.table
        adapted = re.sub(r'\b\w+\.(\w+)\.(\w+)\b', r'\1.\2', adapted)

        # Remove remaining backticks (DuckDB uses double quotes)
        adapted = adapted.replace('`', '"')

        # Handle TIMESTAMP type differences
        adapted = re.sub(r'TIMESTAMP_NTZ', 'TIMESTAMP', adapted, flags=re.IGNORECASE)

        # Handle ARRAY type differences
        adapted = re.sub(r'ARRAY<(\w+)>', r'\1[]', adapted, flags=re.IGNORECASE)

        # Databricks/Spark function adaptations for DuckDB
        # get(array, index) -> array[index+1] (DuckDB is 1-indexed)
        adapted = re.sub(r'get\(split\(([^,]+),\s*([^)]+)\),\s*(\d+)\)', r'split_part(\1, \2, \3+1)', adapted)
        # General get(array, idx) -> list_extract(array, idx+1)
        adapted = re.sub(r'\bget\(([^,]+),\s*(\d+)\)', r'list_extract(\1, \2+1)', adapted)

        # regexp_extract(str, pattern, group) -> regexp_extract(str, pattern, group)
        # DuckDB has regexp_extract but syntax differs slightly - keep as is for now

        # get_json_object(json, path) -> json_extract_string(json, path)
        # DuckDB path format is slightly different: $.key -> '$.key'
        adapted = re.sub(r"get_json_object\(([^,]+),\s*'([^']+)'\)", r"json_extract_string(\1, '\2')", adapted)

        return adapted

    async def run_sql_on_duckdb(
        self,
        sql: str,
        fixtures: dict[str, str | Path] | None = None,
        db_path: str | None = None,
    ) -> dict[str, Any]:
        """Execute SQL on DuckDB with optional fixtures.

        Args:
            sql: SQL to execute.
            fixtures: Optional dict of table_name -> CSV source.
            db_path: Optional path for persistent DB, else uses in-memory.

        Returns:
            Query results and metadata.
        """
        import duckdb

        conn = duckdb.connect(db_path or ":memory:")

        try:
            # Load fixtures if provided
            if fixtures:
                self.load_fixtures_to_duckdb(conn, fixtures)

            # Adapt SQL for DuckDB
            adapted_sql = self.adapt_sql_for_duckdb(sql)

            # Execute as a CREATE TABLE to materialize results
            result_table = "diff_result"
            create_sql = f"CREATE TABLE {result_table} AS {adapted_sql}"

            try:
                conn.execute(create_sql)
            except Exception as e:
                return {
                    "success": False,
                    "error": f"SQL execution failed: {str(e)[:500]}",
                    "sql": adapted_sql,
                }

            # Get results
            row_count = conn.execute(f"SELECT COUNT(*) FROM {result_table}").fetchone()[0]
            schema = conn.execute(f"DESCRIBE {result_table}").fetchall()
            sample = conn.execute(f"SELECT * FROM {result_table} LIMIT 10").fetchall()

            return {
                "success": True,
                "row_count": row_count,
                "schema": [{"column": row[0], "type": row[1]} for row in schema],
                "sample_rows": sample,
                "connection": conn,  # Keep connection open for comparison
            }

        except Exception as e:
            conn.close()
            raise

    async def compare_results(
        self,
        base_conn: Any,
        pr_conn: Any,
        model_name: str,
    ) -> dict[str, Any]:
        """Compare model outputs between base and PR DuckDB connections.

        Args:
            base_conn: DuckDB connection with base results.
            pr_conn: DuckDB connection with PR results.
            model_name: Name of the model being compared.

        Returns:
            Comparison results.
        """
        results = {
            "model": model_name,
            "row_count": {},
            "schema_changes": [],
            "null_changes": [],
            "value_distribution_changes": [],
            "has_diff": False,
        }

        result_table = "diff_result"

        try:
            # 1. Row count comparison
            base_count = base_conn.execute(f"SELECT COUNT(*) FROM {result_table}").fetchone()[0]
            pr_count = pr_conn.execute(f"SELECT COUNT(*) FROM {result_table}").fetchone()[0]

            results["row_count"] = {
                "base": base_count,
                "pr": pr_count,
                "diff": pr_count - base_count,
                "pct_change": round((pr_count - base_count) * 100.0 / base_count, 2) if base_count > 0 else None,
            }

            # 2. Schema comparison
            base_schema = base_conn.execute(f"DESCRIBE {result_table}").fetchall()
            pr_schema = pr_conn.execute(f"DESCRIBE {result_table}").fetchall()

            base_cols = {row[0]: row[1] for row in base_schema}
            pr_cols = {row[0]: row[1] for row in pr_schema}

            for col, dtype in pr_cols.items():
                if col not in base_cols:
                    results["schema_changes"].append({
                        "column": col,
                        "change": "ADDED",
                        "type": dtype,
                    })
                elif base_cols[col] != dtype:
                    results["schema_changes"].append({
                        "column": col,
                        "change": "TYPE_CHANGED",
                        "base_type": base_cols[col],
                        "pr_type": dtype,
                    })

            for col in base_cols:
                if col not in pr_cols:
                    results["schema_changes"].append({
                        "column": col,
                        "change": "REMOVED",
                        "type": base_cols[col],
                    })

            # 3. NULL count comparison
            common_cols = set(base_cols.keys()) & set(pr_cols.keys())
            for col in common_cols:
                try:
                    base_nulls = base_conn.execute(
                        f'SELECT COUNT(*) FROM {result_table} WHERE "{col}" IS NULL'
                    ).fetchone()[0]
                    pr_nulls = pr_conn.execute(
                        f'SELECT COUNT(*) FROM {result_table} WHERE "{col}" IS NULL'
                    ).fetchone()[0]

                    if base_nulls != pr_nulls:
                        results["null_changes"].append({
                            "column": col,
                            "base_nulls": base_nulls,
                            "pr_nulls": pr_nulls,
                            "diff": pr_nulls - base_nulls,
                        })
                except Exception:
                    pass

            # 4. Value distribution for low-cardinality columns
            for col, dtype in pr_cols.items():
                if col not in common_cols:
                    continue
                if "VARCHAR" not in dtype.upper() and "TEXT" not in dtype.upper():
                    continue

                try:
                    cardinality = pr_conn.execute(
                        f'SELECT COUNT(DISTINCT "{col}") FROM {result_table}'
                    ).fetchone()[0]

                    if cardinality <= 20:
                        base_dist = dict(base_conn.execute(
                            f'SELECT "{col}", COUNT(*) FROM {result_table} GROUP BY "{col}"'
                        ).fetchall())
                        pr_dist = dict(pr_conn.execute(
                            f'SELECT "{col}", COUNT(*) FROM {result_table} GROUP BY "{col}"'
                        ).fetchall())

                        all_values = set(base_dist.keys()) | set(pr_dist.keys())
                        changes = []
                        for val in all_values:
                            base_cnt = base_dist.get(val, 0)
                            pr_cnt = pr_dist.get(val, 0)
                            if base_cnt != pr_cnt:
                                changes.append({
                                    "value": str(val) if val is not None else "NULL",
                                    "base_count": base_cnt,
                                    "pr_count": pr_cnt,
                                    "diff": pr_cnt - base_cnt,
                                })

                        if changes:
                            results["value_distribution_changes"].append({
                                "column": col,
                                "changes": sorted(changes, key=lambda x: abs(x["diff"]), reverse=True)[:10],
                            })
                except Exception:
                    pass

            # Determine if there are meaningful diffs
            results["has_diff"] = (
                results["row_count"]["diff"] != 0 or
                len(results["schema_changes"]) > 0 or
                len(results["null_changes"]) > 0 or
                len(results["value_distribution_changes"]) > 0
            )

        except Exception as e:
            logger.exception(f"Error comparing results: {e}")
            results["error"] = str(e)

        return results


def format_diff_report(
    pr_number: int,
    base_branch: str,
    head_sha: str,
    model_results: list[dict[str, Any]],
) -> str:
    """Format comparison results as a markdown report.

    Args:
        pr_number: PR number.
        base_branch: Base branch name.
        head_sha: Head commit SHA.
        model_results: List of comparison results per model.

    Returns:
        Markdown formatted report.
    """
    lines = [
        DBT_DIFF_MARKER,
        "## dbt Model Diff Report",
        "",
        f"**PR #{pr_number}** | **Base:** `{base_branch}` | **Models:** {len(model_results)} analyzed",
        "",
        "> **How this works:** Models are compiled from both branches and executed locally on DuckDB",
        "> with agent-generated test fixtures. This validates that code changes produce expected results.",
        "",
        "### Summary",
        "",
        "| Model | Change | Rows | Schema | Data | Status |",
        "|-------|--------|------|--------|------|--------|",
    ]

    for result in model_results:
        model = result["model"]
        change_type = result.get("change_type", "MODIFIED")
        row_diff = result.get("row_count", {}).get("diff", 0)
        pct_change = result.get("row_count", {}).get("pct_change")
        schema_changes = len(result.get("schema_changes", []))
        null_changes = len(result.get("null_changes", []))
        dist_changes = len(result.get("value_distribution_changes", []))
        data_changes = null_changes + dist_changes

        # Format change type with emoji
        if change_type == "NEW":
            change_str = ":sparkles: NEW"
        elif change_type == "DELETED":
            change_str = ":wastebasket: DELETED"
        else:
            change_str = ":pencil2: MODIFIED"

        # Format row change
        if change_type == "NEW":
            row_str = f"+{row_diff}" if row_diff else "0"
        elif change_type == "DELETED":
            row_str = f"{row_diff}" if row_diff else "0"
        elif row_diff == 0:
            row_str = "No change"
        elif row_diff > 0:
            row_str = f"+{row_diff}"
            if pct_change:
                row_str += f" (+{pct_change}%)"
        else:
            row_str = f"{row_diff}"
            if pct_change:
                row_str += f" ({pct_change}%)"

        # Format schema changes
        if schema_changes == 0:
            schema_str = "No changes"
        elif change_type == "NEW":
            schema_str = f"{schema_changes} column{'s' if schema_changes > 1 else ''}"
        elif change_type == "DELETED":
            schema_str = f"{schema_changes} column{'s' if schema_changes > 1 else ''}"
        else:
            schema_str = f"{schema_changes} change{'s' if schema_changes > 1 else ''}"

        # Format data changes
        if data_changes == 0:
            data_str = "-" if change_type in ("NEW", "DELETED") else "No changes"
        else:
            data_str = f"{data_changes} change{'s' if data_changes > 1 else ''}"

        # Status icon
        if result.get("error"):
            status = ":x:"
        elif change_type == "NEW":
            status = ":new:"
        elif change_type == "DELETED":
            status = ":warning:"
        elif result.get("has_diff"):
            status = ":warning:"
        else:
            status = ":white_check_mark:"

        lines.append(f"| `{model}` | {change_str} | {row_str} | {schema_str} | {data_str} | {status} |")

    lines.append("")
    lines.append("---")
    lines.append("")

    # Add detailed sections for each model with changes
    for result in model_results:
        if not result.get("has_diff") and not result.get("error"):
            continue

        model = result["model"]
        change_type = result.get("change_type", "MODIFIED")
        lines.append("<details>")
        lines.append(f"<summary><b>{model}</b> ({change_type})</summary>")
        lines.append("")

        if result.get("error"):
            lines.append(f"**Error:** {result['error']}")
        else:
            # Row count details
            rc = result.get("row_count", {})
            if change_type == "NEW":
                lines.append("#### New Model")
                lines.append("")
                lines.append(f"- Rows: {rc.get('pr', 0):,}")
                lines.append("")
            elif change_type == "DELETED":
                lines.append("#### Deleted Model")
                lines.append("")
                lines.append(f"- Rows (was): {rc.get('base', 0):,}")
                lines.append("")
            elif rc.get("diff", 0) != 0:
                lines.append("#### Row Count")
                lines.append("")
                lines.append(f"- Base: {rc.get('base', 'N/A'):,}")
                lines.append(f"- PR: {rc.get('pr', 'N/A'):,}")
                pct = rc.get('pct_change', 0) or 0
                lines.append(f"- Diff: {rc.get('diff', 0):+,} ({pct:+.2f}%)")
                lines.append("")

            # Schema changes
            schema_changes = result.get("schema_changes", [])
            if schema_changes:
                if change_type == "NEW":
                    lines.append("#### Schema")
                elif change_type == "DELETED":
                    lines.append("#### Schema (was)")
                else:
                    lines.append("#### Schema Changes")
                lines.append("")
                lines.append("| Column | Change | Type |")
                lines.append("|--------|--------|------|")
                for change in schema_changes:
                    col = change["column"]
                    col_change_type = change["change"]
                    if col_change_type == "TYPE_CHANGED":
                        type_str = f"{change['base_type']} → {change['pr_type']}"
                    else:
                        type_str = change.get("type", "")
                    lines.append(f"| `{col}` | {col_change_type} | {type_str} |")
                lines.append("")

            # NULL changes (only for MODIFIED)
            null_changes = result.get("null_changes", [])
            if null_changes and change_type == "MODIFIED":
                lines.append("#### NULL Count Changes")
                lines.append("")
                lines.append("| Column | Base NULLs | PR NULLs | Diff |")
                lines.append("|--------|------------|----------|------|")
                for change in null_changes:
                    col = change["column"]
                    base_n = change["base_nulls"]
                    pr_n = change["pr_nulls"]
                    diff = change["diff"]
                    lines.append(f"| `{col}` | {base_n:,} | {pr_n:,} | {diff:+,} |")
                lines.append("")

            # Value distribution changes (only for MODIFIED)
            dist_changes = result.get("value_distribution_changes", [])
            if dist_changes and change_type == "MODIFIED":
                lines.append("#### Value Distribution Changes")
                lines.append("")
                for dist in dist_changes:
                    col = dist["column"]
                    lines.append(f"**`{col}`**")
                    lines.append("")
                    lines.append("| Value | Base | PR | Diff |")
                    lines.append("|-------|------|-----|------|")
                    for change in dist["changes"]:
                        val = change["value"]
                        base_c = change["base_count"]
                        pr_c = change["pr_count"]
                        diff = change["diff"]
                        lines.append(f"| {val} | {base_c:,} | {pr_c:,} | {diff:+,} |")
                    lines.append("")

        lines.append("</details>")
        lines.append("")

    # Footer
    lines.append("---")
    lines.append(f":robot: Generated by dbt-diff with agent-generated test fixtures | Commit: `{head_sha[:7]}`")

    return "\n".join(lines)


async def run_dbt_diff(
    owner: str,
    repo: str,
    pr_number: int,
    github_token: str,
    repo_path: str,
    dbt_project_subdir: str = "dbt-databricks",
    changed_models: list[str] | None = None,
    fixtures: dict[str, dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Run a full dbt diff for a PR.

    Compiles models using existing dbt, then runs SQL on DuckDB with fixtures.

    Args:
        owner: Repository owner.
        repo: Repository name.
        pr_number: PR number.
        github_token: GitHub access token.
        repo_path: Local path to the repository.
        dbt_project_subdir: Subdirectory containing dbt project.
        changed_models: List of model names to diff. If None, auto-detects.
        fixtures: Optional dict of model_name -> {table_name: csv_content}.

    Returns:
        Diff results including model comparisons and report markdown.
    """
    github = GitHubClient(token=github_token, owner=owner, repo=repo)

    try:
        # Get PR details
        pr = await github.get_pull_request(pr_number)
        base_branch = pr.get("base", {}).get("ref", "develop")
        head_sha = pr.get("head", {}).get("sha", "")
        head_branch = pr.get("head", {}).get("ref", "")

        # Auto-detect changed models if not provided
        if changed_models is None:
            pr_files = await github.get_pr_files(pr_number)
            changed_models = []
            for f in pr_files:
                filename = f.get("filename", "")
                if filename.startswith(f"{dbt_project_subdir}/models/") and filename.endswith(".sql"):
                    model_name = Path(filename).stem
                    changed_models.append(model_name)

            if not changed_models:
                return {
                    "success": True,
                    "message": "No dbt models changed in this PR",
                    "model_results": [],
                    "report": None,
                }

        logger.info(f"Running diff for models: {changed_models}")

        async with DbtDiffRunner(repo_path, dbt_project_subdir) as runner:
            # Checkout both branches
            await runner.checkout_branch(base_branch, "base")
            await runner.checkout_branch(head_branch, "pr")

            model_results = []

            for model in changed_models:
                logger.info(f"Processing model: {model}")

                # Get fixtures for this model (if provided)
                model_fixtures = fixtures.get(model) if fixtures else None

                # Check if model exists in each branch
                base_exists = runner.model_exists(model, "base")
                pr_exists = runner.model_exists(model, "pr")

                # Handle new model (only in PR)
                if not base_exists and pr_exists:
                    pr_compile = await runner.compile_model(model, "pr")
                    if not pr_compile["success"]:
                        model_results.append({
                            "model": model,
                            "change_type": "NEW",
                            "error": pr_compile["error"],
                            "has_diff": True,
                        })
                        continue

                    pr_result = await runner.run_sql_on_duckdb(pr_compile["sql"], model_fixtures)
                    if not pr_result["success"]:
                        model_results.append({
                            "model": model,
                            "change_type": "NEW",
                            "error": f"PR SQL failed: {pr_result['error']}",
                            "has_diff": True,
                        })
                        continue

                    # Report new model info
                    schema = pr_result.get("schema", [])
                    model_results.append({
                        "model": model,
                        "change_type": "NEW",
                        "has_diff": True,
                        "row_count": {"pr": pr_result["row_count"], "base": 0, "diff": pr_result["row_count"]},
                        "schema_changes": [{"column": s["column"], "change": "ADDED", "type": s["type"]} for s in schema],
                    })
                    pr_result["connection"].close()
                    continue

                # Handle deleted model (only in base)
                if base_exists and not pr_exists:
                    base_compile = await runner.compile_model(model, "base")
                    if not base_compile["success"]:
                        model_results.append({
                            "model": model,
                            "change_type": "DELETED",
                            "error": base_compile["error"],
                            "has_diff": True,
                        })
                        continue

                    base_result = await runner.run_sql_on_duckdb(base_compile["sql"], model_fixtures)
                    if not base_result["success"]:
                        model_results.append({
                            "model": model,
                            "change_type": "DELETED",
                            "error": f"Base SQL failed: {base_result['error']}",
                            "has_diff": True,
                        })
                        continue

                    # Report deleted model info
                    schema = base_result.get("schema", [])
                    model_results.append({
                        "model": model,
                        "change_type": "DELETED",
                        "has_diff": True,
                        "row_count": {"base": base_result["row_count"], "pr": 0, "diff": -base_result["row_count"]},
                        "schema_changes": [{"column": s["column"], "change": "REMOVED", "type": s["type"]} for s in schema],
                    })
                    base_result["connection"].close()
                    continue

                # Handle modified model (exists in both)
                base_compile = await runner.compile_model(model, "base")
                if not base_compile["success"]:
                    model_results.append({
                        "model": model,
                        "change_type": "MODIFIED",
                        "error": base_compile["error"],
                        "has_diff": True,
                    })
                    continue

                pr_compile = await runner.compile_model(model, "pr")
                if not pr_compile["success"]:
                    model_results.append({
                        "model": model,
                        "change_type": "MODIFIED",
                        "error": pr_compile["error"],
                        "has_diff": True,
                    })
                    continue

                # Run both SQLs on DuckDB with fixtures
                base_result = await runner.run_sql_on_duckdb(base_compile["sql"], model_fixtures)
                if not base_result["success"]:
                    model_results.append({
                        "model": model,
                        "change_type": "MODIFIED",
                        "error": f"Base SQL failed: {base_result['error']}",
                        "has_diff": True,
                    })
                    continue

                pr_result = await runner.run_sql_on_duckdb(pr_compile["sql"], model_fixtures)
                if not pr_result["success"]:
                    base_result["connection"].close()
                    model_results.append({
                        "model": model,
                        "change_type": "MODIFIED",
                        "error": f"PR SQL failed: {pr_result['error']}",
                        "has_diff": True,
                    })
                    continue

                # Compare results
                comparison = await runner.compare_results(
                    base_result["connection"],
                    pr_result["connection"],
                    model,
                )
                comparison["change_type"] = "MODIFIED"
                model_results.append(comparison)

                # Clean up connections
                base_result["connection"].close()
                pr_result["connection"].close()

            # Generate report
            report = format_diff_report(pr_number, base_branch, head_sha, model_results)

            return {
                "success": True,
                "model_results": model_results,
                "report": report,
                "models_analyzed": len(model_results),
                "models_with_diffs": sum(1 for r in model_results if r.get("has_diff")),
            }

    finally:
        await github.close()


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

    if name == "jirade_run_dbt_diff":
        owner = arguments["owner"]
        repo = arguments["repo"]
        pr_number = arguments["pr_number"]
        repo_path = arguments.get("repo_path", os.getcwd())
        dbt_project_subdir = arguments.get("dbt_project_subdir", "dbt-databricks")
        changed_models = arguments.get("models")
        fixtures = arguments.get("fixtures")

        return await run_dbt_diff(
            owner=owner,
            repo=repo,
            pr_number=pr_number,
            github_token=settings.github_token,
            repo_path=repo_path,
            dbt_project_subdir=dbt_project_subdir,
            changed_models=changed_models,
            fixtures=fixtures,
        )

    elif name == "jirade_post_diff_report":
        owner = arguments["owner"]
        repo = arguments["repo"]
        pr_number = arguments["pr_number"]
        report = arguments["report"]
        update_existing = arguments.get("update_existing", True)

        return await post_diff_report(
            owner=owner,
            repo=repo,
            pr_number=pr_number,
            report=report,
            github_token=settings.github_token,
            update_existing=update_existing,
        )

    elif name == "jirade_run_dbt_ci":
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

        # Auto-detect changed models if not provided
        await _notify(5, 100, "Detecting changed models...")
        if changed_models is None:
            pr_files = await github.get_pr_files(pr_number)
            changed_models = []
            for f in pr_files:
                filename = f.get("filename", "")
                if filename.startswith(f"{dbt_project_subdir}/models/") and filename.endswith(".sql"):
                    model_name = Path(filename).stem
                    changed_models.append(model_name)

            if not changed_models:
                return {
                    "success": True,
                    "message": "No dbt models changed in this PR",
                    "model_results": [],
                    "report": None,
                }

        logger.info(f"Running Databricks CI for models: {changed_models}")
        await _notify(10, 100, f"Found {len(changed_models)} changed model(s)")

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
        selector_str = " ".join(model_selectors)

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
        logger.info(f"Built {len(built_models)} models (changed + downstream)")
        if model_build_failures:
            logger.warning(f"{len(model_build_failures)} model(s) failed to build: {model_build_failures}")
        if test_failures:
            logger.warning(f"{len(test_failures)} test(s) failed: {test_failures}")
        await _notify(70, 100, f"Built {len(built_models)} models. Comparing against production...")

        # Only compare the models that were actually changed in the PR
        # (downstream models are built just to verify they still work)
        models_to_compare = changed_models
        logger.info(f"Comparing {len(models_to_compare)} changed models")

        # Compare CI tables against prod using metadata client
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

                    if not prod_table:
                        # New model, get metadata only
                        metadata = db_client.get_new_table_metadata(ci_table)
                        model_results.append({
                            "model": model,
                            "change_type": "NEW",
                            "has_diff": True,
                            "row_count": {"ci": metadata["row_count"], "base": 0, "diff": metadata["row_count"]},
                            "schema_changes": [
                                {"column": s["column"], "change": "ADDED", "type": s["type"]}
                                for s in metadata.get("column_stats", [])
                            ],
                            "column_stats": metadata.get("column_stats", []),
                        })
                    else:
                        # Compare CI vs prod
                        comparison = db_client.compare_tables(prod_table, ci_table)
                        comparison["model"] = model
                        comparison["change_type"] = "MODIFIED"
                        model_results.append(comparison)

                except Exception as e:
                    logger.exception(f"Error comparing model {model}: {e}")
                    model_results.append({
                        "model": model,
                        "change_type": "MODIFIED",
                        "error": str(e),
                        "has_diff": True,
                    })

            # Note: CI schemas are NOT cleaned up here - they remain available
            # for manual inspection until the PR is merged. Cleanup should be
            # triggered separately (e.g., via webhook on PR merge).

        # Calculate downstream models (built but not changed)
        downstream_models = [m for m in built_models if m not in changed_models]

        # Generate report with CI catalog for table references
        await _notify(85, 100, "Generating diff report...")
        report = format_ci_diff_report(
            pr_number=pr_number,
            base_branch=base_branch,
            head_sha=head_sha,
            model_results=model_results,
            downstream_models=downstream_models,
            model_build_failures=model_build_failures,
            test_failures=test_failures,
            ci_catalog=settings.databricks_ci_catalog,
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
            "downstream_models_built": len(downstream_models),
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

    profiles_content = f"""algolia_databricks:
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
        "--favor-state",
    ]

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
            "--favor-state",
        ]

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


def format_ci_diff_report(
    pr_number: int,
    base_branch: str,
    head_sha: str,
    model_results: list[dict[str, Any]],
    downstream_models: list[str] | None = None,
    model_build_failures: list[str] | None = None,
    test_failures: list[str] | None = None,
    ci_catalog: str = "",
) -> str:
    """Format CI comparison results as a markdown report.

    Args:
        pr_number: PR number.
        base_branch: Base branch name.
        head_sha: Head commit SHA.
        model_results: List of comparison results per model.
        downstream_models: List of downstream model names that were built successfully.
        model_build_failures: List of model names that failed to build.
        test_failures: List of test names that failed during the build.
        ci_catalog: Catalog where CI tables are created.

    Returns:
        Markdown formatted report.
    """
    downstream_models = downstream_models or []
    model_build_failures = model_build_failures or []
    test_failures = test_failures or []

    lines = [
        DBT_DIFF_MARKER,
        "## dbt CI Diff Report",
        "",
        f"**PR #{pr_number}** | **Base:** `{base_branch}` | **Changed models:** {len(model_results)}",
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

        # Format schema changes
        if schema_changes == 0:
            schema_str = "No changes"
        else:
            schema_str = f"{schema_changes} change{'s' if schema_changes > 1 else ''}"

        # Check for any errors (direct or from comparison sub-queries)
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

        lines.append(f"| `{table_name}` | `{ci_table}` | {row_str} | {schema_str} | {status} |")

    lines.append("")
    lines.append("---")
    lines.append("")

    # Detailed sections
    for result in model_results:
        has_error = result.get("error") or result.get("row_count_error") or result.get("schema_error")
        if not result.get("has_diff") and not has_error:
            continue

        model = result.get("model", "unknown")
        change_type = result.get("change_type", "MODIFIED")

        lines.append("<details>")
        lines.append(f"<summary><b>{model}</b> ({change_type})</summary>")
        lines.append("")

        if has_error:
            # Show the most specific error
            error_msg = result.get("error") or result.get("row_count_error") or result.get("schema_error")
            # Clean up long Databricks error messages
            if "TABLE_OR_VIEW_NOT_FOUND" in str(error_msg):
                error_msg = f"CI table not found: `{result.get('ci_table', 'unknown')}` - model may not have been built"
            lines.append(f"**Error:** {error_msg}")
        else:
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

    # Add downstream models section if any
    if downstream_models:
        lines.append("### Downstream Models")
        lines.append("")
        lines.append(f":white_check_mark: **{len(downstream_models)} downstream model(s) built successfully**")
        lines.append("")
        lines.append("<details>")
        lines.append("<summary>View downstream models (click to expand)</summary>")
        lines.append("")
        lines.append("| Model | CI Table |")
        lines.append("|-------|----------|")
        for model in downstream_models:
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
            ci_table_full = f"{ci_catalog}.{ci_schema}.{table_name}"
            lines.append(f"| `{table_name}` | `{ci_table_full}` |")
        lines.append("")
        lines.append("</details>")
        lines.append("")

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
