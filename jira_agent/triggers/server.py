"""Webhook server for Jira and GitHub events."""

import asyncio
import hashlib
import hmac
import logging
import re
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel

from ..agent import JiraAgent
from ..config import AgentSettings
from ..repo_config.loader import ConfigLoader

logger = logging.getLogger(__name__)


def extract_ticket_key(pr_title: str, branch_name: str, project_key: str) -> str | None:
    """Extract Jira ticket key from PR title or branch name.

    Args:
        pr_title: PR title (e.g., "feat(dbt): add column (AENG-1234)").
        branch_name: Branch name (e.g., "feat/AENG-1234-add-column").
        project_key: Expected Jira project key (e.g., "AENG").

    Returns:
        Ticket key if found, None otherwise.
    """
    # Pattern to match ticket key like AENG-1234
    pattern = rf"\b({re.escape(project_key)}-\d+)\b"

    # Try PR title first
    match = re.search(pattern, pr_title, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Try branch name
    match = re.search(pattern, branch_name, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    return None

app = FastAPI(title="Jira Agent Webhook Server")

# Global state (set by run_server)
_settings: AgentSettings | None = None
_config_loader: ConfigLoader | None = None


class WebhookResponse(BaseModel):
    """Response for webhook endpoints."""

    status: str
    message: str


def verify_github_signature(payload: bytes, signature: str, secret: str) -> bool:
    """Verify GitHub webhook signature.

    Args:
        payload: Raw request body.
        signature: X-Hub-Signature-256 header value.
        secret: Webhook secret.

    Returns:
        True if signature is valid.
    """
    if not signature.startswith("sha256="):
        return False

    expected = hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
    actual = signature[7:]  # Remove 'sha256=' prefix

    return hmac.compare_digest(expected, actual)


def verify_jira_signature(payload: bytes, signature: str, secret: str) -> bool:
    """Verify Jira webhook signature.

    Args:
        payload: Raw request body.
        signature: X-Atlassian-Webhook-Signature header value.
        secret: Webhook secret.

    Returns:
        True if signature is valid.
    """
    # Jira uses SHA256 HMAC
    expected = hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)


@app.post("/webhook/jira", response_model=WebhookResponse)
async def handle_jira_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    x_atlassian_webhook_signature: str | None = Header(None),
) -> WebhookResponse:
    """Handle Jira webhook events.

    Triggers on:
    - Issue assignment to agent
    - @mention in comments
    - Status change to trigger status
    """
    payload = await request.body()

    # Verify signature if secret is configured
    if _settings and _settings.webhook_secret:
        if not x_atlassian_webhook_signature:
            raise HTTPException(status_code=401, detail="Missing signature")
        if not verify_jira_signature(payload, x_atlassian_webhook_signature, _settings.webhook_secret):
            raise HTTPException(status_code=401, detail="Invalid signature")

    data = await request.json()
    event_type = data.get("webhookEvent", "")
    issue_data = data.get("issue", {})
    issue_key = issue_data.get("key")

    logger.info(f"Received Jira webhook: {event_type} for {issue_key}")

    if not issue_key:
        return WebhookResponse(status="ignored", message="No issue key in payload")

    # Check if this is an event we care about
    should_process = False
    reason = ""

    if event_type == "jira:issue_updated":
        # Check for assignment change
        changelog = data.get("changelog", {})
        for item in changelog.get("items", []):
            if item.get("field") == "assignee":
                new_assignee_id = item.get("to")
                if _settings and new_assignee_id == _settings.agent_jira_account_id:
                    should_process = True
                    reason = "Assigned to agent"

    elif event_type == "comment_created":
        # Check for @mention
        comment = data.get("comment", {})
        comment_body = comment.get("body", "")
        # Check if agent is mentioned (would need agent's Jira account name)
        if _settings and _settings.agent_jira_user_id:
            if f"@{_settings.agent_jira_user_id}" in str(comment_body):
                should_process = True
                reason = "Mentioned in comment"

    if not should_process:
        return WebhookResponse(status="ignored", message="Event not actionable")

    # Find config for this project
    project_key = issue_data.get("fields", {}).get("project", {}).get("key")
    repo_config = _find_config_for_jira_project(project_key)

    if not repo_config:
        return WebhookResponse(
            status="error",
            message=f"No config found for Jira project: {project_key}",
        )

    # Process in background
    background_tasks.add_task(process_jira_ticket, issue_key, repo_config, reason)

    return WebhookResponse(
        status="accepted",
        message=f"Processing {issue_key}: {reason}",
    )


@app.post("/webhook/github", response_model=WebhookResponse)
async def handle_github_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    x_github_event: str = Header(...),
    x_hub_signature_256: str | None = Header(None),
) -> WebhookResponse:
    """Handle GitHub webhook events.

    Triggers on:
    - PR review with requested changes
    - Check run failure
    """
    payload = await request.body()

    # Verify signature if secret is configured
    if _settings and _settings.webhook_secret:
        if not x_hub_signature_256:
            raise HTTPException(status_code=401, detail="Missing signature")
        if not verify_github_signature(payload, x_hub_signature_256, _settings.webhook_secret):
            raise HTTPException(status_code=401, detail="Invalid signature")

    data = await request.json()

    logger.info(f"Received GitHub webhook: {x_github_event}")

    # Get repository info
    repo_data = data.get("repository", {})
    repo_full_name = repo_data.get("full_name", "")

    if not repo_full_name:
        return WebhookResponse(status="ignored", message="No repository in payload")

    # Find config for this repo
    try:
        repo_config = _config_loader.load_for_repo(repo_full_name)
    except FileNotFoundError:
        return WebhookResponse(
            status="ignored",
            message=f"No config for repo: {repo_full_name}",
        )

    should_process = False
    action_type = ""
    action_data: dict[str, Any] = {}

    if x_github_event == "pull_request_review":
        review = data.get("review", {})
        if review.get("state") == "changes_requested":
            should_process = True
            action_type = "review_changes"
            action_data = {
                "pr_number": data.get("pull_request", {}).get("number"),
                "review_id": review.get("id"),
                "body": review.get("body"),
            }

    elif x_github_event == "check_run":
        action = data.get("action")
        check_run = data.get("check_run", {})

        if action == "completed" and check_run.get("conclusion") == "failure":
            # Only process if this is for a PR
            prs = check_run.get("pull_requests", [])
            if prs:
                should_process = True
                action_type = "ci_failure"
                action_data = {
                    "pr_number": prs[0].get("number"),
                    "check_name": check_run.get("name"),
                    "check_run_id": check_run.get("id"),
                }

    elif x_github_event == "pull_request":
        action = data.get("action")
        pr = data.get("pull_request", {})

        # Handle PR merge - transition Jira ticket to Done
        if action == "closed" and pr.get("merged"):
            should_process = True
            action_type = "pr_merged"
            action_data = {
                "pr_number": pr.get("number"),
                "pr_title": pr.get("title", ""),
                "pr_branch": pr.get("head", {}).get("ref", ""),
                "merge_commit": pr.get("merge_commit_sha"),
            }

    if not should_process:
        return WebhookResponse(status="ignored", message="Event not actionable")

    # Process in background
    background_tasks.add_task(
        process_github_event,
        action_type,
        action_data,
        repo_config,
    )

    return WebhookResponse(
        status="accepted",
        message=f"Processing {action_type} for {repo_full_name}",
    )


@app.get("/health")
async def health_check() -> dict:
    """Health check endpoint."""
    return {"status": "healthy"}


def _find_config_for_jira_project(project_key: str):
    """Find repo config for a Jira project key."""
    if not _config_loader:
        return None

    for repo_name in _config_loader.list_configs():
        try:
            config = _config_loader.load_for_repo(repo_name)
            if config.jira.project_key == project_key:
                return config
        except Exception:
            continue

    return None


async def process_jira_ticket(
    ticket_key: str,
    repo_config,
    reason: str,
) -> None:
    """Process a Jira ticket in the background.

    Args:
        ticket_key: Jira ticket key.
        repo_config: Repository configuration.
        reason: Reason for processing.
    """
    logger.info(f"Background processing: {ticket_key} ({reason})")

    try:
        agent = JiraAgent(_settings, repo_config)
        result = await agent.process_single_ticket(ticket_key)
        logger.info(f"Completed {ticket_key}: {result['status']}")
    except Exception as e:
        logger.error(f"Failed to process {ticket_key}: {e}")


async def process_github_event(
    action_type: str,
    action_data: dict,
    repo_config,
) -> None:
    """Process a GitHub event in the background.

    Args:
        action_type: Type of action (review_changes, ci_failure, pr_merged).
        action_data: Action-specific data.
        repo_config: Repository configuration.
    """
    logger.info(f"Background processing GitHub event: {action_type}")

    try:
        agent = JiraAgent(_settings, repo_config)

        if action_type == "ci_failure":
            result = await agent.fix_ci_failures(action_data["pr_number"])
            logger.info(f"CI fix result: {result}")

        elif action_type == "review_changes":
            # TODO: Implement review response
            logger.info(f"Would respond to review on PR #{action_data['pr_number']}")

        elif action_type == "pr_merged":
            # Extract ticket key and transition to Done
            ticket_key = extract_ticket_key(
                action_data["pr_title"],
                action_data["pr_branch"],
                repo_config.jira.project_key,
            )
            if ticket_key:
                result = await agent.transition_ticket_to_done(ticket_key)
                logger.info(f"Transitioned {ticket_key} to Done: {result}")
            else:
                logger.info("No ticket key found in PR title/branch, skipping transition")

    except Exception as e:
        logger.error(f"Failed to process GitHub event: {e}")


def run_server(
    host: str = "0.0.0.0",
    port: int = 8080,
    config_dir: Path | None = None,
    settings: AgentSettings | None = None,
) -> None:
    """Run the webhook server.

    Args:
        host: Server host.
        port: Server port.
        config_dir: Directory containing repo configs.
        settings: Agent settings.
    """
    import uvicorn

    global _settings, _config_loader

    _settings = settings or AgentSettings()
    _config_loader = ConfigLoader(config_dir)

    logger.info(f"Starting webhook server on {host}:{port}")
    logger.info(f"Config directory: {config_dir}")
    logger.info(f"Available configs: {_config_loader.list_configs()}")

    uvicorn.run(app, host=host, port=port)
