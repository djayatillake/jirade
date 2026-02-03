"""dbt diff tool handlers for MCP server.

This module provides functionality for comparing dbt model outputs between
a base branch and a PR branch using DuckDB for local execution.

Approach:
1. Use existing dbt to compile models (generates SQL)
2. Run compiled SQL directly on DuckDB with agent-generated fixtures
3. Compare outputs between base and PR branches
"""

import asyncio
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import yaml

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
        fixtures: dict[str, str | Path],
    ) -> None:
        """Load fixture CSVs into DuckDB as tables.

        Args:
            conn: DuckDB connection.
            fixtures: Dict mapping table names to CSV file paths or CSV content strings.
        """
        for table_name, source in fixtures.items():
            if isinstance(source, Path) or (isinstance(source, str) and os.path.exists(source)):
                # Load from CSV file
                conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM read_csv_auto('{source}')")
            else:
                # Source is CSV content as string - write to temp file first
                temp_csv = self.work_dir / "fixtures" / f"{table_name}.csv"
                temp_csv.write_text(source)
                conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM read_csv_auto('{temp_csv}')")

            logger.info(f"Loaded fixture table: {table_name}")

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

        # Remove catalog prefixes (e.g., `catalog.schema.table` -> `schema.table`)
        # DuckDB doesn't use 3-part names by default
        adapted = re.sub(r'`?\w+`?\.`?(\w+)`?\.`?(\w+)`?', r'\1.\2', adapted)

        # Remove remaining backticks (DuckDB uses double quotes)
        adapted = adapted.replace('`', '"')

        # Handle TIMESTAMP type differences
        adapted = re.sub(r'TIMESTAMP_NTZ', 'TIMESTAMP', adapted, flags=re.IGNORECASE)

        # Handle ARRAY type differences
        adapted = re.sub(r'ARRAY<(\w+)>', r'\1[]', adapted, flags=re.IGNORECASE)

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
    lines.append(f":robot: Generated by dbt-diff | Commit: `{head_sha[:7]}`")

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


async def handle_dbt_diff_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Handle a dbt diff tool call.

    Args:
        name: Tool name.
        arguments: Tool arguments.

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

    else:
        raise ValueError(f"Unknown dbt diff tool: {name}")
