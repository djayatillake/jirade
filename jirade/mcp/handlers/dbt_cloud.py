"""dbt Cloud tool handlers for MCP server."""

import logging
from typing import Any

from ...clients.dbt_cloud_client import DbtCloudClient, RunStatus, format_run_errors_for_prompt
from ...config import get_settings

logger = logging.getLogger(__name__)


async def get_dbt_cloud_client() -> DbtCloudClient:
    """Get an authenticated dbt Cloud client.

    Returns:
        DbtCloudClient instance.

    Raises:
        RuntimeError: If not configured.
    """
    settings = get_settings()

    if not settings.has_dbt_cloud:
        raise RuntimeError("dbt Cloud not configured. Set JIRADE_DBT_CLOUD_API_TOKEN and JIRADE_DBT_CLOUD_ACCOUNT_ID.")

    return DbtCloudClient(
        api_token=settings.dbt_cloud_api_token,
        account_id=settings.dbt_cloud_account_id,
        base_url=settings.dbt_cloud_base_url,
    )


async def handle_dbt_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Handle a dbt Cloud tool call.

    Args:
        name: Tool name.
        arguments: Tool arguments.

    Returns:
        Tool result.
    """
    client = await get_dbt_cloud_client()

    try:
        if name == "jirade_dbt_list_jobs":
            return await list_jobs(client, arguments)
        elif name == "jirade_dbt_trigger_run":
            return await trigger_run(client, arguments)
        elif name == "jirade_dbt_get_run":
            return await get_run(client, arguments)
        else:
            raise ValueError(f"Unknown dbt Cloud tool: {name}")
    finally:
        await client.close()


async def list_jobs(client: DbtCloudClient, arguments: dict[str, Any]) -> dict[str, Any]:
    """List dbt Cloud jobs.

    Args:
        client: dbt Cloud client.
        arguments: Tool arguments with optional 'project_id'.

    Returns:
        List of jobs.
    """
    project_id = arguments.get("project_id")

    jobs = await client.list_jobs(project_id=project_id)

    results = []
    for job in jobs:
        triggers = job.get("triggers", {})
        results.append(
            {
                "id": job.get("id"),
                "name": job.get("name"),
                "project_id": job.get("project_id"),
                "environment_id": job.get("environment_id"),
                "job_type": job.get("job_type"),
                "is_ci": job.get("job_type") == "ci"
                or triggers.get("github_webhook")
                or triggers.get("git_provider_webhook"),
                "triggers": {
                    "github_webhook": triggers.get("github_webhook", False),
                    "git_provider_webhook": triggers.get("git_provider_webhook", False),
                    "schedule": triggers.get("schedule", False),
                },
                "state": job.get("state"),
            }
        )

    return {
        "total": len(results),
        "jobs": results,
    }


async def trigger_run(client: DbtCloudClient, arguments: dict[str, Any]) -> dict[str, Any]:
    """Trigger a dbt Cloud CI run for a pull request.

    Args:
        client: dbt Cloud client.
        arguments: Tool arguments with 'job_id', 'pr_number', 'git_sha', and optional 'git_branch'.

    Returns:
        Triggered run info.
    """
    job_id = arguments["job_id"]
    pr_number = arguments["pr_number"]
    git_sha = arguments["git_sha"]
    git_branch = arguments.get("git_branch")

    settings = get_settings()

    # Optionally update event-time dates for microbatch models
    lookback_days = settings.dbt_cloud_event_time_lookback_days
    if lookback_days > 0:
        try:
            await client.update_ci_job_event_time_dates(job_id, lookback_days=lookback_days)
            logger.info(f"Updated CI job {job_id} event-time dates with {lookback_days} day lookback")
        except Exception as e:
            logger.warning(f"Failed to update event-time dates: {e}")

    # Trigger the run
    run = await client.trigger_ci_run(
        job_id=job_id,
        pr_number=pr_number,
        git_sha=git_sha,
        git_branch=git_branch,
    )

    return {
        "success": True,
        "run_id": run.get("id"),
        "job_id": job_id,
        "pr_number": pr_number,
        "git_sha": git_sha,
        "status": RunStatus.to_string(run.get("status", 0)),
        "href": run.get("href"),
    }


async def get_run(client: DbtCloudClient, arguments: dict[str, Any]) -> dict[str, Any]:
    """Get status and details of a dbt Cloud run.

    Args:
        client: dbt Cloud client.
        arguments: Tool arguments with 'run_id' and optional 'include_errors'.

    Returns:
        Run details.
    """
    run_id = arguments["run_id"]
    include_errors = arguments.get("include_errors", True)

    run = await client.get_run(run_id, include_related=["job", "trigger"])

    status_code = run.get("status", 0)
    status_str = RunStatus.to_string(status_code)
    is_complete = RunStatus.is_complete(status_code)
    is_success = RunStatus.is_success(status_code)

    result = {
        "run_id": run_id,
        "job_id": run.get("job_definition_id"),
        "job_name": run.get("job", {}).get("name"),
        "status": status_str,
        "status_code": status_code,
        "is_complete": is_complete,
        "is_success": is_success,
        "created_at": run.get("created_at"),
        "started_at": run.get("started_at"),
        "finished_at": run.get("finished_at"),
        "duration_humanized": run.get("duration_humanized"),
        "href": run.get("href"),
    }

    # Include trigger info if available
    trigger = run.get("trigger", {})
    if trigger:
        result["trigger"] = {
            "cause": trigger.get("cause"),
            "git_branch": trigger.get("git_branch"),
            "git_sha": trigger.get("git_sha"),
            "github_pull_request_id": trigger.get("github_pull_request_id"),
        }

    # Include errors if run failed and requested
    if include_errors and is_complete and not is_success:
        try:
            errors = await client.get_run_errors(run_id)
            result["errors"] = errors
            result["error_summary"] = format_run_errors_for_prompt(errors)
        except Exception as e:
            logger.warning(f"Failed to fetch run errors: {e}")
            result["errors"] = []
            result["error_summary"] = f"Failed to fetch errors: {e}"

    return result
