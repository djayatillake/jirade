"""dbt Cloud API client for CI job management."""

import asyncio
import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


# dbt Cloud run status codes
class RunStatus:
    QUEUED = 1
    STARTING = 2
    RUNNING = 3
    SUCCESS = 10
    ERROR = 20
    CANCELLED = 30

    @classmethod
    def is_complete(cls, status: int) -> bool:
        """Check if run has completed (success, error, or cancelled)."""
        return status in (cls.SUCCESS, cls.ERROR, cls.CANCELLED)

    @classmethod
    def is_success(cls, status: int) -> bool:
        """Check if run completed successfully."""
        return status == cls.SUCCESS

    @classmethod
    def to_string(cls, status: int) -> str:
        """Convert status code to string."""
        status_map = {
            cls.QUEUED: "queued",
            cls.STARTING: "starting",
            cls.RUNNING: "running",
            cls.SUCCESS: "success",
            cls.ERROR: "error",
            cls.CANCELLED: "cancelled",
        }
        return status_map.get(status, f"unknown({status})")


class DbtCloudClient:
    """Client for dbt Cloud Administrative API."""

    def __init__(
        self,
        api_token: str,
        account_id: str,
        base_url: str = "https://cloud.getdbt.com",
    ):
        """Initialize dbt Cloud client.

        Args:
            api_token: dbt Cloud API token (service account or personal).
            account_id: dbt Cloud account ID.
            base_url: dbt Cloud API base URL (default for multi-tenant).
        """
        self.api_token = api_token
        self.account_id = account_id
        self.base_url = base_url.rstrip("/")
        self.api_base = f"{self.base_url}/api/v2/accounts/{account_id}"
        self.api_base_v3 = f"{self.base_url}/api/v3/accounts/{account_id}"
        self.headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json",
        }
        self._client = httpx.AsyncClient(headers=self.headers, timeout=60.0)

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()

    async def __aenter__(self) -> "DbtCloudClient":
        return self

    async def __aexit__(self, *args) -> None:
        await self.close()

    async def _request(
        self,
        method: str,
        endpoint: str,
        **kwargs,
    ) -> dict[str, Any]:
        """Make an API request.

        Args:
            method: HTTP method.
            endpoint: API endpoint (relative to account base).
            **kwargs: Additional request arguments.

        Returns:
            JSON response data.
        """
        url = f"{self.api_base}/{endpoint.lstrip('/')}"
        response = await self._client.request(method, url, **kwargs)
        response.raise_for_status()

        if response.status_code == 204:
            return {}

        result = response.json()
        return result.get("data", result)

    # -------------------------------------------------------------------------
    # Job Management
    # -------------------------------------------------------------------------

    async def get_job(self, job_id: int) -> dict[str, Any]:
        """Get job details.

        Args:
            job_id: dbt Cloud job ID.

        Returns:
            Job data.
        """
        return await self._request("GET", f"jobs/{job_id}/")

    async def list_jobs(
        self,
        project_id: int | None = None,
        environment_id: int | None = None,
    ) -> list[dict[str, Any]]:
        """List jobs in the account.

        Args:
            project_id: Optional project filter.
            environment_id: Optional environment filter.

        Returns:
            List of jobs.
        """
        params = {}
        if project_id:
            params["project_id"] = project_id
        if environment_id:
            params["environment_id"] = environment_id

        result = await self._request("GET", "jobs/", params=params)
        return result if isinstance(result, list) else []

    async def find_ci_job(self, project_id: int) -> dict[str, Any] | None:
        """Find the CI job for a project.

        Args:
            project_id: dbt Cloud project ID.

        Returns:
            CI job data or None if not found.
        """
        jobs = await self.list_jobs(project_id=project_id)
        for job in jobs:
            # CI jobs have job_type "ci" or have triggers.github_webhook enabled
            if job.get("job_type") == "ci":
                return job
            triggers = job.get("triggers", {})
            if triggers.get("github_webhook") or triggers.get("git_provider_webhook"):
                return job
        return None

    async def update_job(
        self,
        job_id: int,
        **updates,
    ) -> dict[str, Any]:
        """Update a job's configuration.

        Args:
            job_id: dbt Cloud job ID.
            **updates: Fields to update (execute_steps, name, etc.)

        Returns:
            Updated job data.
        """
        # Get current job to preserve required fields
        job = await self.get_job(job_id)

        payload = {
            "account_id": int(self.account_id),
            "project_id": job["project_id"],
            "environment_id": job["environment_id"],
            "name": job["name"],
            **updates,
        }

        result = await self._request("POST", f"jobs/{job_id}/", json=payload)
        logger.info(f"Updated dbt Cloud job {job_id}")
        return result

    async def set_job_env_var_override(
        self,
        job_id: int,
        project_id: int,
        env_var_name: str,
        env_var_value: str,
    ) -> dict[str, Any]:
        """Set a job-level environment variable override.

        This creates or updates an environment variable override that applies
        only to the specified job.

        Args:
            job_id: dbt Cloud job ID.
            project_id: dbt Cloud project ID.
            env_var_name: Environment variable name (e.g., 'DBT_CLOUD_INVOCATION_CONTEXT').
            env_var_value: Value to set.

        Returns:
            API response data.
        """
        # Use v3 API for environment variable job overrides
        url = f"{self.api_base_v3}/projects/{project_id}/environment-variables/job/"

        payload = {
            "name": env_var_name,
            "type": "job",
            "job_definition_id": job_id,
            "project_id": project_id,
            "account_id": int(self.account_id),
            "raw_value": env_var_value,
        }

        response = await self._client.post(url, json=payload)
        response.raise_for_status()

        result = response.json()
        logger.info(f"Set job env var override {env_var_name}={env_var_value} for job {job_id}")
        return result.get("data", result)

    async def update_ci_job_event_time_dates(
        self,
        job_id: int,
        lookback_days: int = 3,
    ) -> dict[str, Any]:
        """Update CI job execute_steps with fresh event-time dates.

        This updates the --event-time-start and --event-time-end flags
        to limit microbatch models to recent data during CI runs.

        Args:
            job_id: CI job ID.
            lookback_days: Number of days back from today for start date.

        Returns:
            Updated job data.
        """
        from datetime import datetime, timedelta

        today = datetime.now().date()
        start_date = today - timedelta(days=lookback_days)

        execute_steps = [
            f"dbt build --select state:modified+1 --exclude test_name:no_missing_date* "
            f"--event-time-start {start_date.isoformat()} --event-time-end {today.isoformat()}"
        ]

        result = await self.update_job(job_id, execute_steps=execute_steps)
        logger.info(
            f"Updated CI job {job_id} event-time dates: {start_date} to {today}"
        )
        return result

    # -------------------------------------------------------------------------
    # Run Triggering
    # -------------------------------------------------------------------------

    async def trigger_job(
        self,
        job_id: int,
        cause: str = "Triggered by jirade",
        git_sha: str | None = None,
        git_branch: str | None = None,
        github_pull_request_id: int | None = None,
        schema_override: str | None = None,
        steps_override: list[str] | None = None,
        env_var_overrides: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Trigger a job run.

        Args:
            job_id: dbt Cloud job ID.
            cause: Reason for triggering (shown in UI).
            git_sha: Specific commit to checkout.
            git_branch: Branch to checkout.
            github_pull_request_id: PR number for CI jobs.
            schema_override: Override target schema.
            steps_override: Override job steps.
            env_var_overrides: Environment variable overrides for the run.

        Returns:
            Run data including run_id.
        """
        payload: dict[str, Any] = {"cause": cause}

        if git_sha:
            payload["git_sha"] = git_sha
        if git_branch:
            payload["git_branch"] = git_branch
        if github_pull_request_id:
            payload["github_pull_request_id"] = github_pull_request_id
        if schema_override:
            payload["schema_override"] = schema_override
        if steps_override:
            payload["steps_override"] = steps_override
        if env_var_overrides:
            payload["env_var_overrides"] = env_var_overrides

        result = await self._request("POST", f"jobs/{job_id}/run/", json=payload)
        logger.info(f"Triggered dbt Cloud job {job_id}, run_id: {result.get('id')}")
        return result

    async def trigger_ci_run(
        self,
        job_id: int,
        pr_number: int,
        git_sha: str,
        git_branch: str | None = None,
    ) -> dict[str, Any]:
        """Trigger a CI job run for a pull request.

        Args:
            job_id: CI job ID.
            pr_number: GitHub PR number.
            git_sha: Commit SHA to build.
            git_branch: Branch name (optional).

        Returns:
            Run data.
        """
        return await self.trigger_job(
            job_id=job_id,
            cause=f"jirade CI run for PR #{pr_number}",
            git_sha=git_sha,
            git_branch=git_branch,
            github_pull_request_id=pr_number,
            env_var_overrides={"DBT_CLOUD_INVOCATION_CONTEXT": "ci"},
        )

    async def trigger_ci_run_with_selectors(
        self,
        job_id: int,
        pr_number: int,
        git_branch: str,
        model_selectors: list[str],
        lookback_days: int = 3,
    ) -> dict[str, Any]:
        """Trigger a CI job run with explicit model selectors.

        This bypasses state:modified comparison and runs only the specified
        models and their direct downstream dependencies (1 level).

        Args:
            job_id: CI job ID.
            pr_number: GitHub PR number.
            git_branch: Branch name.
            model_selectors: List of model names with +1 suffix for 1 level of downstream deps.
            lookback_days: Days of data for event-time filtering.

        Returns:
            Run data.
        """
        from datetime import datetime, timedelta

        today = datetime.now().date()
        start_date = today - timedelta(days=lookback_days)

        # Build the selector string
        selector_str = " ".join(model_selectors)

        # Build the dbt command with explicit selectors
        steps_override = [
            f"dbt build --select {selector_str} "
            f"--event-time-start {start_date.isoformat()} --event-time-end {today.isoformat()}"
        ]

        logger.info(f"Triggering CI for PR #{pr_number} with selectors: {selector_str}")

        return await self.trigger_job(
            job_id=job_id,
            cause=f"jirade CI run for PR #{pr_number} (file-based selection)",
            git_branch=git_branch,
            github_pull_request_id=pr_number,
            steps_override=steps_override,
            env_var_overrides={"DBT_CLOUD_INVOCATION_CONTEXT": "ci"},
        )

    # -------------------------------------------------------------------------
    # Run Status & Monitoring
    # -------------------------------------------------------------------------

    async def get_run(
        self,
        run_id: int,
        include_related: list[str] | None = None,
    ) -> dict[str, Any]:
        """Get run details.

        Args:
            run_id: dbt Cloud run ID.
            include_related: Related data to include (job, trigger, debug_logs).

        Returns:
            Run data.
        """
        params = {}
        if include_related:
            params["include_related"] = ",".join(include_related)

        return await self._request("GET", f"runs/{run_id}/", params=params)

    async def list_runs(
        self,
        job_id: int | None = None,
        project_id: int | None = None,
        status: int | None = None,
        limit: int = 50,
        offset: int = 0,
        order_by: str = "-id",
    ) -> list[dict[str, Any]]:
        """List runs with optional filters.

        Args:
            job_id: Filter by job.
            project_id: Filter by project.
            status: Filter by status code.
            limit: Max results.
            offset: Results offset.
            order_by: Sort order (prefix with - for descending).

        Returns:
            List of runs.
        """
        params: dict[str, Any] = {
            "limit": limit,
            "offset": offset,
            "order_by": order_by,
        }
        if job_id:
            params["job_definition_id"] = job_id
        if project_id:
            params["project_id"] = project_id
        if status:
            params["status"] = status

        result = await self._request("GET", "runs/", params=params)
        return result if isinstance(result, list) else []

    async def get_runs_for_pr(
        self,
        job_id: int,
        pr_number: int,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Get CI runs for a specific pull request.

        dbt Cloud CI runs use schema naming: dbt_cloud_pr_<job_id>_<pr_id>

        Args:
            job_id: CI job ID.
            pr_number: GitHub PR number.

        Returns:
            List of runs for this PR.
        """
        runs = await self.list_runs(job_id=job_id, limit=limit)
        pr_runs = []

        for run in runs:
            run_id = run.get("id")

            # The list endpoint doesn't include trigger data, so fetch it
            try:
                full_run = await self.get_run(run_id, include_related=["trigger"])
                trigger = full_run.get("trigger") or {}

                # Check github_pull_request_id in trigger
                if trigger.get("github_pull_request_id") == pr_number:
                    pr_runs.append(full_run)
                    continue

                # Also check schema_override pattern (dbt_cloud_pr_<job_id>_<pr_id>)
                schema_override = trigger.get("schema_override", "")
                expected_pattern = f"dbt_cloud_pr_{job_id}_{pr_number}"
                if expected_pattern in schema_override:
                    pr_runs.append(full_run)
            except Exception as e:
                logger.debug(f"Failed to fetch run {run_id} details: {e}")

        return pr_runs

    async def poll_run_until_complete(
        self,
        run_id: int,
        poll_interval: int = 30,
        timeout: int = 3600,
    ) -> dict[str, Any]:
        """Poll a run until it completes.

        Args:
            run_id: Run ID to poll.
            poll_interval: Seconds between polls.
            timeout: Maximum wait time in seconds.

        Returns:
            Final run data.

        Raises:
            TimeoutError: If run doesn't complete within timeout.
        """
        elapsed = 0
        while elapsed < timeout:
            run = await self.get_run(run_id)
            status = run.get("status")

            if RunStatus.is_complete(status):
                logger.info(f"Run {run_id} completed with status: {RunStatus.to_string(status)}")
                return run

            logger.debug(f"Run {run_id} status: {RunStatus.to_string(status)}, waiting...")
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

        raise TimeoutError(f"Run {run_id} did not complete within {timeout} seconds")

    async def cancel_run(self, run_id: int) -> dict[str, Any]:
        """Cancel a running job.

        Args:
            run_id: Run ID to cancel.

        Returns:
            Updated run data.
        """
        return await self._request("POST", f"runs/{run_id}/cancel/")

    # -------------------------------------------------------------------------
    # Artifacts & Results
    # -------------------------------------------------------------------------

    async def list_artifacts(self, run_id: int) -> list[str]:
        """List available artifacts for a run.

        Args:
            run_id: Run ID.

        Returns:
            List of artifact paths.
        """
        result = await self._request("GET", f"runs/{run_id}/artifacts/")
        return result if isinstance(result, list) else []

    async def get_artifact(
        self,
        run_id: int,
        path: str,
        step: int | None = None,
    ) -> dict[str, Any]:
        """Get a specific artifact from a run.

        Args:
            run_id: Run ID.
            path: Artifact path (run_results.json, manifest.json, catalog.json).
            step: Step index (defaults to last step).

        Returns:
            Artifact content.
        """
        params = {}
        if step is not None:
            params["step"] = step

        return await self._request("GET", f"runs/{run_id}/artifacts/{path}", params=params)

    async def get_run_results(self, run_id: int) -> dict[str, Any]:
        """Get run_results.json artifact.

        Args:
            run_id: Run ID.

        Returns:
            Run results data including model statuses and errors.
        """
        return await self.get_artifact(run_id, "run_results.json")

    async def get_manifest(self, run_id: int) -> dict[str, Any]:
        """Get manifest.json artifact.

        Args:
            run_id: Run ID.

        Returns:
            Manifest data with model definitions.
        """
        return await self.get_artifact(run_id, "manifest.json")

    # -------------------------------------------------------------------------
    # Error Extraction
    # -------------------------------------------------------------------------

    async def get_run_errors(self, run_id: int) -> list[dict[str, Any]]:
        """Extract error information from a failed run.

        Args:
            run_id: Run ID.

        Returns:
            List of error details with model name, error message, and SQL.
        """
        errors = []

        try:
            run_results = await self.get_run_results(run_id)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.warning(f"No run_results.json found for run {run_id}")
                return errors
            raise

        results = run_results.get("results", [])

        for result in results:
            status = result.get("status")
            if status in ("error", "fail"):
                error_info = {
                    "unique_id": result.get("unique_id", "unknown"),
                    "status": status,
                    "message": result.get("message", ""),
                    "failures": result.get("failures"),
                    "execution_time": result.get("execution_time"),
                }

                # Extract adapter response for more details
                adapter_response = result.get("adapter_response", {})
                if adapter_response:
                    error_info["adapter_response"] = adapter_response

                # Get timing info
                timing = result.get("timing", [])
                if timing:
                    error_info["timing"] = timing

                errors.append(error_info)

        return errors

    async def get_run_debug_logs(self, run_id: int) -> str:
        """Get debug logs for a run.

        Args:
            run_id: Run ID.

        Returns:
            Debug log content.
        """
        run = await self.get_run(run_id, include_related=["debug_logs"])
        return run.get("debug_logs", "")

    # -------------------------------------------------------------------------
    # Health Check
    # -------------------------------------------------------------------------

    async def health_check(self) -> dict[str, Any]:
        """Verify API connectivity and authentication.

        Returns:
            Dict with status and account info.
        """
        try:
            # Try to list jobs as a connectivity test
            jobs = await self.list_jobs()
            return {
                "status": "ok",
                "account_id": self.account_id,
                "job_count": len(jobs),
            }
        except httpx.HTTPStatusError as e:
            return {
                "status": "error",
                "error": str(e),
                "status_code": e.response.status_code,
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
            }


def build_model_selectors_from_files(
    changed_files: list[str],
    dbt_project_subdirectory: str = "",
) -> list[str]:
    """Build dbt model selectors from changed file paths.

    Extracts model names from changed SQL files in the models directory
    and adds the +1 suffix for 1 level of downstream dependencies.

    Args:
        changed_files: List of changed file paths.
        dbt_project_subdirectory: Subdirectory containing dbt project (e.g., 'dbt-databricks').

    Returns:
        List of model selectors like ['model_name+1', 'other_model+1'].
    """
    import re

    selectors = []
    models_path = f"{dbt_project_subdirectory}/models/" if dbt_project_subdirectory else "models/"

    for file_path in changed_files:
        # Only process SQL files in models directory
        if not file_path.endswith(".sql"):
            continue
        if models_path not in file_path:
            continue

        # Extract model name from file path
        # e.g., "dbt-databricks/models/analytics/dim_user.sql" -> "dim_user"
        match = re.search(rf"{re.escape(models_path)}.*?([^/]+)\.sql$", file_path)
        if match:
            model_name = match.group(1)
            # Add +1 for 1 level of downstream dependencies
            selectors.append(f"{model_name}+1")

    return selectors


def format_run_errors_for_prompt(errors: list[dict[str, Any]]) -> str:
    """Format run errors for inclusion in a Claude prompt.

    Args:
        errors: List of error dicts from get_run_errors.

    Returns:
        Formatted error string.
    """
    if not errors:
        return "No errors found in run results."

    lines = ["## dbt Cloud CI Errors\n"]

    for i, error in enumerate(errors, 1):
        unique_id = error.get("unique_id", "unknown")
        # Extract model name from unique_id (format: model.project.model_name)
        model_name = unique_id.split(".")[-1] if "." in unique_id else unique_id

        lines.append(f"### Error {i}: `{model_name}`")
        lines.append(f"**Full ID:** `{unique_id}`")
        lines.append(f"**Status:** {error.get('status', 'unknown')}")

        message = error.get("message", "")
        if message:
            lines.append(f"\n**Error Message:**\n```\n{message}\n```")

        failures = error.get("failures")
        if failures:
            lines.append(f"\n**Failures:** {failures}")

        lines.append("")

    return "\n".join(lines)
