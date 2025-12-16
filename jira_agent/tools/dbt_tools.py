"""dbt tools for model discovery and validation."""

import json
import logging
import subprocess
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class DbtTools:
    """Tools for working with dbt projects."""

    def __init__(self, repo_path: Path, dbt_projects: list[dict]):
        """Initialize dbt tools.

        Args:
            repo_path: Path to repository root.
            dbt_projects: List of dbt project configurations.
        """
        self.repo_path = repo_path
        self.dbt_projects = dbt_projects
        self._manifests: dict[str, dict] = {}

    def get_project_path(self, project_name: str) -> Path:
        """Get path to a dbt project.

        Args:
            project_name: Project path/name from config.

        Returns:
            Full path to project.
        """
        return self.repo_path / project_name

    def load_manifest(self, project_path: str) -> dict[str, Any]:
        """Load dbt manifest.json for a project.

        Args:
            project_path: Project path from config.

        Returns:
            Manifest data.
        """
        if project_path in self._manifests:
            return self._manifests[project_path]

        # Find manifest path from config
        manifest_rel_path = None
        for proj in self.dbt_projects:
            if proj["path"] == project_path:
                manifest_rel_path = proj.get("manifest_path", "target/manifest.json")
                break

        if not manifest_rel_path:
            manifest_rel_path = "target/manifest.json"

        manifest_path = self.repo_path / project_path / manifest_rel_path

        if not manifest_path.exists():
            # Try target_lock directory
            manifest_path = self.repo_path / project_path / "target_lock" / "manifest.json"

        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found for project: {project_path}")

        with open(manifest_path) as f:
            manifest = json.load(f)

        self._manifests[project_path] = manifest
        return manifest

    def find_models(
        self,
        pattern: str,
        project_path: str | None = None,
    ) -> list[dict[str, Any]]:
        """Find dbt models matching a pattern.

        Args:
            pattern: Model name pattern (supports * wildcard).
            project_path: Specific project to search, or None for all.

        Returns:
            List of matching model info.
        """
        results = []
        projects = [project_path] if project_path else [p["path"] for p in self.dbt_projects]

        for proj in projects:
            try:
                manifest = self.load_manifest(proj)
            except FileNotFoundError:
                logger.warning(f"Could not load manifest for {proj}")
                continue

            for node_id, node in manifest.get("nodes", {}).items():
                if node.get("resource_type") != "model":
                    continue

                name = node.get("name", "")
                # Simple wildcard matching
                if "*" in pattern:
                    import fnmatch

                    if not fnmatch.fnmatch(name, pattern):
                        continue
                elif pattern.lower() not in name.lower():
                    continue

                results.append({
                    "project": proj,
                    "unique_id": node_id,
                    "name": name,
                    "path": node.get("original_file_path"),
                    "schema": node.get("schema"),
                    "database": node.get("database"),
                    "materialized": node.get("config", {}).get("materialized"),
                })

        return results

    def get_model_dependencies(
        self,
        model_name: str,
        project_path: str,
    ) -> dict[str, list[str]]:
        """Get upstream and downstream dependencies for a model.

        Args:
            model_name: Model name.
            project_path: Project containing the model.

        Returns:
            Dict with 'upstream' and 'downstream' lists.
        """
        manifest = self.load_manifest(project_path)

        # Find the model node
        model_id = None
        for node_id, node in manifest.get("nodes", {}).items():
            if node.get("name") == model_name and node.get("resource_type") == "model":
                model_id = node_id
                break

        if not model_id:
            raise ValueError(f"Model not found: {model_name}")

        model_node = manifest["nodes"][model_id]

        # Get upstream (dependencies)
        upstream = model_node.get("depends_on", {}).get("nodes", [])

        # Get downstream (dependents)
        downstream = []
        for node_id, node in manifest.get("nodes", {}).items():
            deps = node.get("depends_on", {}).get("nodes", [])
            if model_id in deps:
                downstream.append(node_id)

        # Also check child_map if available
        child_map = manifest.get("child_map", {})
        if model_id in child_map:
            downstream.extend(child_map[model_id])

        return {
            "upstream": list(set(upstream)),
            "downstream": list(set(downstream)),
        }

    def get_model_columns(
        self,
        model_name: str,
        project_path: str,
    ) -> list[dict[str, Any]]:
        """Get column information for a model.

        Args:
            model_name: Model name.
            project_path: Project containing the model.

        Returns:
            List of column info.
        """
        manifest = self.load_manifest(project_path)

        # Find the model node
        for node in manifest.get("nodes", {}).values():
            if node.get("name") == model_name and node.get("resource_type") == "model":
                columns = node.get("columns", {})
                return [
                    {
                        "name": col_name,
                        "description": col_info.get("description", ""),
                        "data_type": col_info.get("data_type"),
                        "tests": [t.get("test_name") for t in col_info.get("tests", [])],
                    }
                    for col_name, col_info in columns.items()
                ]

        return []

    def get_model_sql(
        self,
        model_name: str,
        project_path: str,
    ) -> str | None:
        """Get the SQL for a model.

        Args:
            model_name: Model name.
            project_path: Project containing the model.

        Returns:
            Model SQL or None.
        """
        manifest = self.load_manifest(project_path)

        for node in manifest.get("nodes", {}).values():
            if node.get("name") == model_name and node.get("resource_type") == "model":
                file_path = self.repo_path / project_path / node.get("original_file_path", "")
                if file_path.exists():
                    return file_path.read_text()
                # Fallback to raw_code from manifest
                return node.get("raw_code")

        return None

    def compile_model(
        self,
        model_name: str,
        project_path: str,
    ) -> tuple[bool, str]:
        """Compile a dbt model to check for errors.

        Args:
            model_name: Model name.
            project_path: Project containing the model.

        Returns:
            Tuple of (success, output).
        """
        dbt_path = self.repo_path / project_path

        result = subprocess.run(
            ["dbt", "compile", "--select", model_name],
            cwd=dbt_path,
            capture_output=True,
            text=True,
        )

        output = result.stdout + result.stderr
        success = result.returncode == 0

        return success, output

    def get_sources(self, project_path: str) -> list[dict[str, Any]]:
        """Get all sources defined in a project.

        Args:
            project_path: Project path.

        Returns:
            List of source definitions.
        """
        manifest = self.load_manifest(project_path)

        sources = []
        for source_id, source in manifest.get("sources", {}).items():
            sources.append({
                "unique_id": source_id,
                "name": source.get("name"),
                "source_name": source.get("source_name"),
                "schema": source.get("schema"),
                "database": source.get("database"),
                "identifier": source.get("identifier"),
            })

        return sources

    def find_model_file(
        self,
        model_name: str,
        project_path: str | None = None,
    ) -> Path | None:
        """Find the file path for a model.

        Args:
            model_name: Model name.
            project_path: Specific project to search.

        Returns:
            Path to model file or None.
        """
        projects = [project_path] if project_path else [p["path"] for p in self.dbt_projects]

        for proj in projects:
            # Search for .sql files matching model name
            models_dir = self.repo_path / proj / "models"
            if not models_dir.exists():
                continue

            for sql_file in models_dir.rglob(f"*{model_name}*.sql"):
                if sql_file.stem == model_name:
                    return sql_file

        return None


def parse_dbt_error(error_output: str) -> dict[str, Any]:
    """Parse dbt error output to extract useful information.

    Args:
        error_output: Raw dbt error output.

    Returns:
        Parsed error info.
    """
    error_info = {
        "type": "unknown",
        "message": error_output,
        "model": None,
        "line": None,
    }

    lines = error_output.split("\n")

    for line in lines:
        # Look for compilation error
        if "Compilation Error" in line:
            error_info["type"] = "compilation"

        # Look for model name in error
        if "in model" in line.lower():
            import re

            match = re.search(r"model\s+(\w+)", line, re.IGNORECASE)
            if match:
                error_info["model"] = match.group(1)

        # Look for line number
        if "line" in line.lower():
            import re

            match = re.search(r"line\s+(\d+)", line, re.IGNORECASE)
            if match:
                error_info["line"] = int(match.group(1))

    return error_info
