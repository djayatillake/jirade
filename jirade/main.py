"""Jirade - Jira Data Engineer CLI.

An autonomous agent that processes Jira tickets and implements code changes.
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional

import typer
from typing_extensions import Annotated

from .config import get_settings
from .repo_config.loader import ConfigLoader, find_repo_config, get_git_remote_info
from .utils.logger import setup_logging

__version__ = "0.1.0"

# Main app
app = typer.Typer(
    name="jirade",
    help="Jira Data Engineer - Autonomous Jira ticket agent using Claude",
    no_args_is_help=True,
    rich_markup_mode="rich",
)

# Subcommand groups
auth_app = typer.Typer(help="Manage OAuth authentication")
config_app = typer.Typer(help="Show or validate configuration")
learn_app = typer.Typer(help="Manage agent learnings")
env_app = typer.Typer(help="Check and setup environment")

app.add_typer(auth_app, name="auth")
app.add_typer(config_app, name="config")
app.add_typer(learn_app, name="learn")
app.add_typer(env_app, name="env")


def version_callback(value: bool):
    if value:
        print(f"jirade {__version__}")
        raise typer.Exit()


@app.callback()
def main_callback(
    version: Annotated[
        Optional[bool],
        typer.Option("--version", "-v", callback=version_callback, is_eager=True, help="Show version"),
    ] = None,
):
    """Jirade - Jira Data Engineer CLI."""
    pass


def load_config_with_fallback(config_path: Optional[str], required: bool = True):
    """Load repo config with auto-detection fallback."""
    loader = ConfigLoader()

    if config_path:
        return loader.load_from_file(config_path)

    auto_config = loader.auto_detect()
    if auto_config:
        return auto_config

    if required:
        print("Error: No config found.")
        print("Either:")
        print("  1. Run 'jirade init' to create .jirade.yaml in this repo")
        print("  2. Specify --config=<path> to a config file")
        raise typer.Exit(1)

    return None


# ============================================================
# Main Commands
# ============================================================


@app.command()
def init(
    output: Annotated[
        str, typer.Option("--output", "-o", help="Output path for config")
    ] = ".jirade.yaml",
):
    """Initialize jirade for a repository (interactive setup)."""
    settings = get_settings()
    setup_logging(settings.log_level)
    args = {"--output": output}
    raise typer.Exit(handle_init(args, settings))


@app.command()
def chat(
    config: Annotated[Optional[str], typer.Option("--config", "-c", help="Path to config file")] = None,
    model: Annotated[str, typer.Option("--model", "-m", help="Claude model to use")] = "claude-sonnet-4-20250514",
):
    """Start an interactive chat session with the jirade agent."""
    settings = get_settings()
    setup_logging(settings.log_level)
    args = {"--config": config, "--model": model}
    raise typer.Exit(asyncio.run(handle_chat(args, settings)))


@app.command("list-tickets")
def list_tickets(
    config: Annotated[Optional[str], typer.Option("--config", "-c", help="Path to config file")] = None,
    status: Annotated[Optional[str], typer.Option("--status", "-s", help="Filter by Jira status")] = None,
    limit: Annotated[int, typer.Option("--limit", "-l", help="Maximum tickets to show")] = 20,
    interactive: Annotated[bool, typer.Option("--interactive", "-i", help="Interactive selection mode")] = False,
):
    """List tickets from a Jira board."""
    settings = get_settings()
    setup_logging(settings.log_level)
    args = {"--config": config, "--status": status, "--limit": limit, "--interactive": interactive}
    raise typer.Exit(asyncio.run(handle_list_tickets(args, settings)))


@app.command("list-prs")
def list_prs(
    config: Annotated[Optional[str], typer.Option("--config", "-c", help="Path to config file")] = None,
    state: Annotated[str, typer.Option("--state", help="PR state: open, closed, all")] = "open",
):
    """List open PRs for the repository."""
    settings = get_settings()
    setup_logging(settings.log_level)
    args = {"--config": config, "--state": state}
    raise typer.Exit(asyncio.run(handle_list_prs(args, settings)))


@app.command()
def watch(
    config: Annotated[Optional[str], typer.Option("--config", "-c", help="Path to config file")] = None,
    interval: Annotated[int, typer.Option("--interval", help="Polling interval in seconds")] = 60,
):
    """Watch for tickets and auto-transition when PRs merge."""
    settings = get_settings()
    setup_logging(settings.log_level)
    args = {"--config": config, "--interval": interval}
    raise typer.Exit(asyncio.run(handle_watch(args, settings)))


@app.command()
def process(
    config: Annotated[Optional[str], typer.Option("--config", "-c", help="Path to config file")] = None,
    status: Annotated[Optional[str], typer.Option("--status", "-s", help="Filter by Jira status")] = None,
    limit: Annotated[int, typer.Option("--limit", "-l", help="Maximum tickets to process")] = 10,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Preview without making changes")] = False,
):
    """Process multiple tickets from a Jira board."""
    settings = get_settings()
    setup_logging(settings.log_level)
    args = {"--config": config, "--status": status, "--limit": limit, "--dry-run": dry_run}
    raise typer.Exit(asyncio.run(handle_process(args, settings)))


@app.command("process-ticket")
def process_ticket(
    ticket_key: Annotated[str, typer.Argument(help="Jira ticket key (e.g., PROJ-123)")],
    config: Annotated[Optional[str], typer.Option("--config", "-c", help="Path to config file")] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Preview without making changes")] = False,
):
    """Process a specific ticket by key."""
    settings = get_settings()
    setup_logging(settings.log_level)
    args = {"<ticket_key>": ticket_key, "--config": config, "--dry-run": dry_run}
    result = asyncio.run(handle_process_ticket(args, settings))
    if result.get("post_pr_info"):
        info = result["post_pr_info"]
        exit_code = _post_pr_flow(
            info["pr_number"],
            info["pr_url"],
            info["repo_config"],
            settings,
            info["ticket_key"],
            args,
        )
        raise typer.Exit(exit_code)
    raise typer.Exit(result["exit_code"])


@app.command("check-pr")
def check_pr(
    pr_number: Annotated[int, typer.Argument(help="PR number to check")],
    config: Annotated[Optional[str], typer.Option("--config", "-c", help="Path to config file")] = None,
):
    """Check PR status and pending feedback."""
    settings = get_settings()
    setup_logging(settings.log_level)
    args = {"<pr_number>": pr_number, "--config": config}
    raise typer.Exit(asyncio.run(handle_check_pr(args, settings)))


@app.command("fix-ci")
def fix_ci(
    pr_number: Annotated[int, typer.Argument(help="PR number to fix")],
    config: Annotated[Optional[str], typer.Option("--config", "-c", help="Path to config file")] = None,
):
    """Attempt to fix CI failures on a PR."""
    settings = get_settings()
    setup_logging(settings.log_level)
    args = {"<pr_number>": pr_number, "--config": config}
    raise typer.Exit(asyncio.run(handle_fix_ci(args, settings)))


@app.command()
def serve(
    port: Annotated[int, typer.Option("--port", "-p", help="Server port")] = 8080,
    host: Annotated[str, typer.Option("--host", help="Server host")] = "0.0.0.0",
    config_dir: Annotated[str, typer.Option("--config-dir", help="Config directory")] = "./configs",
):
    """Start webhook server for Jira/GitHub events."""
    settings = get_settings()
    setup_logging(settings.log_level)
    args = {"--port": port, "--host": host, "--config-dir": config_dir}
    raise typer.Exit(handle_serve(args, settings))


@app.command()
def health(
    config: Annotated[Optional[str], typer.Option("--config", "-c", help="Path to config file")] = None,
):
    """Test all service connections."""
    settings = get_settings()
    setup_logging(settings.log_level)
    args = {"--config": config}
    raise typer.Exit(asyncio.run(handle_health(args, settings)))


# ============================================================
# Auth Subcommands
# ============================================================


@auth_app.command("login")
def auth_login(
    service: Annotated[str, typer.Option("--service", "-s", help="Service: jira, github, databricks, or all")] = "all",
):
    """Login to services."""
    settings = get_settings()
    setup_logging(settings.log_level)
    args = {"login": True, "status": False, "logout": False, "--service": service}
    raise typer.Exit(handle_auth(args, settings))


@auth_app.command("status")
def auth_status():
    """Show authentication status."""
    settings = get_settings()
    setup_logging(settings.log_level)
    args = {"login": False, "status": True, "logout": False, "--service": None}
    raise typer.Exit(handle_auth(args, settings))


@auth_app.command("logout")
def auth_logout(
    service: Annotated[str, typer.Option("--service", "-s", help="Service: jira, github, databricks, or all")] = "all",
):
    """Logout from services."""
    settings = get_settings()
    setup_logging(settings.log_level)
    args = {"login": False, "status": False, "logout": True, "--service": service}
    raise typer.Exit(handle_auth(args, settings))


# ============================================================
# Config Subcommands
# ============================================================


@config_app.command("show")
def config_show():
    """Show current configuration."""
    settings = get_settings()
    setup_logging(settings.log_level)
    args = {"show": True, "validate": False}
    raise typer.Exit(handle_config_command(args, settings))


@config_app.command("validate")
def config_validate(
    config_path: Annotated[str, typer.Argument(help="Path to config file to validate")],
):
    """Validate a config file."""
    settings = get_settings()
    setup_logging(settings.log_level)
    args = {"show": False, "validate": True, "<config_path>": config_path}
    raise typer.Exit(handle_config_command(args, settings))


# ============================================================
# Learn Subcommands
# ============================================================


@learn_app.command("status")
def learn_status():
    """Show pending learnings in workspace."""
    settings = get_settings()
    setup_logging(settings.log_level)
    args = {"status": True, "publish": False, "list": False}
    raise typer.Exit(handle_learn(args, settings))


@learn_app.command("publish")
def learn_publish(
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Preview without creating PR")] = False,
    jirade_repo: Annotated[str, typer.Option("--jirade-repo", help="Target repo for learnings")] = "djayatillake/jirade",
):
    """Publish learnings to jirade repo via PR."""
    settings = get_settings()
    setup_logging(settings.log_level)
    args = {"status": False, "publish": True, "list": False, "--dry-run": dry_run, "--jirade-repo": jirade_repo}
    raise typer.Exit(handle_learn(args, settings))


@learn_app.command("list")
def learn_list(
    category: Annotated[Optional[str], typer.Option("--category", help="Filter by category")] = None,
):
    """List learnings in the knowledge base."""
    settings = get_settings()
    setup_logging(settings.log_level)
    args = {"status": False, "publish": False, "list": True, "--category": category}
    raise typer.Exit(handle_learn(args, settings))


# ============================================================
# Env Subcommands
# ============================================================


@env_app.command("check")
def env_check(
    config: Annotated[Optional[str], typer.Option("--config", "-c", help="Path to config file")] = None,
    repo_path: Annotated[Optional[str], typer.Option("--repo-path", help="Path to repository")] = None,
):
    """Check environment for a repository."""
    settings = get_settings()
    setup_logging(settings.log_level)
    args = {"check": True, "setup": False, "--config": config, "--repo-path": repo_path}
    raise typer.Exit(handle_env(args, settings))


@env_app.command("setup")
def env_setup(
    config: Annotated[Optional[str], typer.Option("--config", "-c", help="Path to config file")] = None,
    repo_path: Annotated[Optional[str], typer.Option("--repo-path", help="Path to repository")] = None,
):
    """Auto-install missing dependencies."""
    settings = get_settings()
    setup_logging(settings.log_level)
    args = {"check": False, "setup": True, "--config": config, "--repo-path": repo_path}
    raise typer.Exit(handle_env(args, settings))


# ============================================================
# Handler Functions (preserved from original)
# ============================================================


def handle_auth(args: dict, settings) -> int:
    """Handle authentication commands."""
    from .auth import AuthManager

    auth_manager = AuthManager(settings)

    if args["login"]:
        service = args["--service"] or "all"
        if service == "all":
            auth_manager.login_all()
        else:
            auth_manager.login(service)
        return 0
    elif args["status"]:
        auth_manager.print_status()
        return 0
    elif args["logout"]:
        service = args["--service"] or "all"
        if service == "all":
            auth_manager.logout_all()
        else:
            auth_manager.logout(service)
        return 0
    return 1


def handle_config_command(args: dict, settings) -> int:
    """Handle config commands."""
    if args["show"]:
        print("Current Configuration:")
        print("-" * 40)
        print(f"Claude Model: {settings.claude_model}")
        print(f"Anthropic API Key: {'*' * 8 if settings.has_anthropic_key else 'Not set'}")
        print(f"Jira OAuth: {'Configured' if settings.has_jira_oauth else 'Not set'}")
        print(f"GitHub Token: {'Configured' if settings.has_github_token else 'Not set'}")
        print(f"Databricks: {'Configured' if settings.has_databricks else 'Not set'}")
        print(f"Workspace Dir: {settings.workspace_dir}")
        print(f"Log Level: {settings.log_level}")
        return 0
    elif args["validate"]:
        from .repo_config.loader import ConfigLoader

        config_path = args["<config_path>"]
        try:
            loader = ConfigLoader()
            config = loader.load_from_file(config_path)
            print(f"Config valid: {config.full_repo_name}")
            print(f"  Jira Project: {config.jira.project_key}")
            print(f"  Default Branch: {config.repo.default_branch}")
            print(f"  PR Target: {config.repo.pr_target_branch}")
            print(f"  dbt Enabled: {config.dbt.enabled}")
            return 0
        except Exception as e:
            print(f"Config validation failed: {e}")
            return 1
    return 1


async def handle_chat(args: dict, settings) -> int:
    """Start an interactive chat session with the jirade agent."""
    from pathlib import Path

    from .repl import REPLAgent

    model = args.get("--model", "claude-sonnet-4-20250514")
    repo_config = load_config_with_fallback(args.get("--config"))

    # Validate authentication
    if not settings.has_anthropic_key:
        print("Error: Anthropic API key not configured")
        return 1

    if not settings.has_github_token:
        print("Error: GitHub token not configured")
        return 1

    # Get repo path (current directory)
    repo_path = Path.cwd()

    # Create and run the REPL agent
    agent = REPLAgent(
        settings=settings,
        repo_config=repo_config,
        repo_path=repo_path,
        model=model,
    )

    try:
        await agent.run()
    except KeyboardInterrupt:
        print("\nGoodbye!")

    return 0


async def handle_list_tickets(args: dict, settings) -> int:
    """List tickets from a Jira board."""
    import questionary

    from .auth import AuthManager
    from .clients.jira_client import JiraClient

    status_filter = args["--status"]
    limit = int(args["--limit"]) if args["--limit"] else 20
    interactive = args["--interactive"]

    repo_config = load_config_with_fallback(args.get("--config"))

    auth_manager = AuthManager(settings)
    if not auth_manager.jira.is_authenticated():
        print("Error: Not authenticated with Jira. Run: jirade auth login --service=jira")
        return 1

    access_token = auth_manager.jira.get_access_token()
    cloud_id = auth_manager.jira.get_cloud_id()

    if not cloud_id:
        print("Error: Could not get Jira cloud ID. Try re-authenticating.")
        return 1

    jira_client = JiraClient(
        cloud_id=cloud_id,
        access_token=access_token,
    )

    print(f"Fetching tickets from {repo_config.jira.project_key}...")
    if status_filter:
        print(f"Filtering by status: {status_filter}")
    print()

    try:
        jql = f"project = {repo_config.jira.project_key}"
        if status_filter:
            jql += f' AND status = "{status_filter}"'
        jql += " ORDER BY updated DESC"
        tickets = await jira_client.search_issues(
            jql=jql,
            max_results=limit,
            fields=["summary", "status", "issuetype", "priority", "assignee"],
        )

        if not tickets:
            print("No tickets found.")
            return 0

        if interactive:
            return await _interactive_ticket_selection(tickets, settings, repo_config)

        print(f"{'Key':<12} {'Status':<20} {'Type':<12} {'Summary'}")
        print("-" * 80)

        for ticket in tickets:
            key = ticket.get("key", "")
            fields = ticket.get("fields", {})
            status = fields.get("status", {}).get("name", "Unknown")
            issue_type = fields.get("issuetype", {}).get("name", "Unknown")
            summary = fields.get("summary", "")

            if len(summary) > 40:
                summary = summary[:37] + "..."

            print(f"{key:<12} {status:<20} {issue_type:<12} {summary}")

        print()
        print(f"Total: {len(tickets)} tickets")
        print()
        print("Tip: Use --interactive to browse and select a ticket to process")

    except Exception as e:
        print(f"Error fetching tickets: {e}")
        return 1

    return 0


async def _interactive_ticket_selection(
    tickets: list,
    settings,
    repo_config,
) -> int:
    """Interactive ticket selection with arrow keys."""
    import questionary
    from questionary import Style

    from .agent import JiraAgent

    custom_style = Style([
        ("qmark", "fg:cyan bold"),
        ("question", "fg:white bold"),
        ("answer", "fg:green bold"),
        ("pointer", "fg:cyan bold"),
        ("highlighted", "fg:cyan bold"),
        ("selected", "fg:green"),
    ])

    choices = []
    for ticket in tickets:
        key = ticket.get("key", "")
        fields = ticket.get("fields", {})
        status = fields.get("status", {}).get("name", "Unknown")
        summary = fields.get("summary", "")

        if len(summary) > 45:
            summary = summary[:42] + "..."

        label = f"{key:<12} [{status:<15}] {summary}"
        choices.append(questionary.Choice(title=label, value=ticket))

    choices.append(questionary.Choice(title="Cancel", value=None))

    print(f"Found {len(tickets)} tickets. Use arrow keys to navigate, Enter to select:\n")

    selected = questionary.select(
        "Select a ticket to process:",
        choices=choices,
        style=custom_style,
        use_shortcuts=False,
        use_arrow_keys=True,
    ).ask()

    if selected is None:
        print("Cancelled.")
        return 0

    ticket_key = selected.get("key")
    fields = selected.get("fields", {})
    summary = fields.get("summary", "")
    status = fields.get("status", {}).get("name", "Unknown")

    print()
    print(f"Selected: {ticket_key}")
    print(f"Summary:  {summary}")
    print(f"Status:   {status}")
    print()

    action = questionary.select(
        "What would you like to do?",
        choices=[
            questionary.Choice(title="Process this ticket (create PR)", value="process"),
            questionary.Choice(title="Process with dry-run (preview only)", value="dry-run"),
            questionary.Choice(title="View ticket details", value="details"),
            questionary.Choice(title="Cancel", value="cancel"),
        ],
        style=custom_style,
    ).ask()

    if action == "cancel" or action is None:
        print("Cancelled.")
        return 0

    if action == "details":
        print()
        print("=" * 60)
        print(f"Ticket: {ticket_key}")
        print("=" * 60)
        print(f"Summary:  {summary}")
        print(f"Type:     {fields.get('issuetype', {}).get('name', 'Unknown')}")
        print(f"Status:   {status}")
        print(f"Priority: {fields.get('priority', {}).get('name', 'Unknown')}")
        assignee = fields.get("assignee")
        print(f"Assignee: {assignee.get('displayName') if assignee else 'Unassigned'}")
        print()
        print("To process this ticket, run:")
        print(f"  jirade process-ticket {ticket_key}")
        return 0

    dry_run = action == "dry-run"

    if dry_run:
        print("Starting dry-run (no changes will be made)...")
    else:
        confirm = questionary.confirm(
            f"Process {ticket_key} and create a PR?",
            default=True,
            style=custom_style,
        ).ask()

        if not confirm:
            print("Cancelled.")
            return 0

        print(f"Processing {ticket_key}...")

    print()

    agent = JiraAgent(settings, repo_config, dry_run=dry_run)
    result = await agent.process_single_ticket(ticket_key)

    status_icon = "✓" if result["status"] == "completed" else "○" if result["status"] == "skipped" else "✗"
    print(f"{status_icon} {result['ticket']}: {result['status']}")
    if result.get("pr_url"):
        print(f"  PR: {result['pr_url']}")
    if result.get("error"):
        print(f"  Error: {result['error']}")

    return 0 if result["status"] in ("completed", "skipped") else 1


async def handle_list_prs(args: dict, settings) -> int:
    """List PRs created by jirade for the repository."""
    import re

    from .clients.github_client import GitHubClient
    from .pr_tracker import PRTracker

    state = args["--state"] or "open"

    repo_config = load_config_with_fallback(args.get("--config"))

    if not settings.has_github_token:
        print("Error: GitHub token not configured")
        return 1

    github = GitHubClient(
        settings.github_token,
        repo_config.repo.owner,
        repo_config.repo.name,
    )

    tracker = PRTracker()

    print(f"Fetching jirade PRs from {repo_config.full_repo_name}...")
    print()

    try:
        # Fetch PRs from GitHub and filter by [jirade] tag in title
        prs = await github.list_pull_requests(state=state, per_page=100)
        jirade_prs = [pr for pr in prs if "[jirade]" in pr.get("title", "")]

        if not jirade_prs:
            print("No PRs created by jirade found.")
            return 0

        print(f"{'#':<6} {'State':<8} {'Ticket':<12} {'Title'}")
        print("-" * 80)

        ticket_pattern = rf"\b({re.escape(repo_config.jira.project_key)}-\d+)\b"

        for pr in jirade_prs:
            number = pr.get("number", "")
            pr_state = pr.get("state", "")
            title = pr.get("title", "")
            merged = pr.get("merged_at") is not None

            # Extract ticket from title or branch
            branch = pr.get("head", {}).get("ref", "")
            match = re.search(ticket_pattern, f"{title} {branch}", re.IGNORECASE)
            ticket = match.group(1).upper() if match else "-"

            if merged:
                pr_state = "merged"

            # Remove [jirade] tag for display
            display_title = title.replace(" [jirade]", "").replace("[jirade] ", "")
            if len(display_title) > 45:
                display_title = display_title[:42] + "..."

            print(f"#{number:<5} {pr_state:<8} {ticket:<12} {display_title}")

            # Update local tracker if not already tracked
            if not tracker.get_pr(repo_config.full_repo_name, number):
                tracker.add_pr(
                    pr_number=number,
                    pr_url=pr.get("html_url", ""),
                    repo=repo_config.full_repo_name,
                    ticket_key=ticket,
                    branch=branch,
                )

        print()
        print(f"Total: {len(jirade_prs)} PRs created by jirade")

    except Exception as e:
        print(f"Error fetching PRs: {e}")
        return 1
    finally:
        await github.close()

    return 0


async def handle_watch(args: dict, settings) -> int:
    """Watch for trigger status tickets, merged PRs, CI failures, and review comments."""
    import re
    from datetime import datetime, timezone

    from .agent import JiraAgent
    from .auth import AuthManager
    from .clients.dbt_cloud_client import DbtCloudClient, RunStatus
    from .clients.github_client import GitHubClient
    from .clients.jira_client import JiraClient
    from .pr_tracker import PRTracker

    interval = int(args["--interval"] or 60)

    repo_config = load_config_with_fallback(args.get("--config"))

    if not settings.has_github_token:
        print("Error: GitHub token not configured")
        return 1

    auth_manager = AuthManager(settings)
    if not auth_manager.jira.is_authenticated():
        print("Error: Not authenticated with Jira. Run: jirade auth login --service=jira")
        return 1

    # Set up dbt Cloud client if configured
    dbt_cloud_client: DbtCloudClient | None = None
    dbt_cloud_ci_job_id: str | None = None

    if settings.has_dbt_cloud:
        dbt_cloud_client = DbtCloudClient(
            api_token=settings.dbt_cloud_api_token,
            account_id=settings.dbt_cloud_account_id,
            base_url=settings.dbt_cloud_base_url,
        )
        dbt_cloud_ci_job_id = settings.dbt_cloud_ci_job_id

        # Try to find CI job if not explicitly configured
        if not dbt_cloud_ci_job_id and settings.dbt_cloud_project_id:
            try:
                ci_job = await dbt_cloud_client.find_ci_job(int(settings.dbt_cloud_project_id))
                if ci_job:
                    dbt_cloud_ci_job_id = str(ci_job["id"])
            except Exception as e:
                logger.warning(f"Failed to find dbt Cloud CI job: {e}")

    tracker = PRTracker()
    tracked_prs = tracker.get_open_prs(repo_config.full_repo_name)

    # Scan for existing jirade PRs and add them to tracking
    print("Scanning for existing jirade PRs...", flush=True)
    try:
        scan_github = GitHubClient(
            settings.github_token,
            repo_config.repo.owner,
            repo_config.repo.name,
        )
        open_prs = await scan_github.list_pull_requests(state="open", per_page=50)
        jirade_prs_found = 0
        for pr in open_prs:
            title = pr.get("title", "")
            if "[jirade]" in title:
                pr_number = pr["number"]
                # Check if already tracked
                existing = [t for t in tracked_prs if t.pr_number == pr_number]
                if not existing:
                    # Extract ticket key from title or branch
                    branch = pr.get("head", {}).get("ref", "")
                    ticket_match = re.search(rf"({re.escape(repo_config.jira.project_key)}-\d+)", f"{title} {branch}", re.IGNORECASE)
                    ticket_key = ticket_match.group(1).upper() if ticket_match else "UNKNOWN"
                    tracker.add_pr(
                        pr_number=pr_number,
                        pr_url=pr["html_url"],
                        repo=repo_config.full_repo_name,
                        ticket_key=ticket_key,
                        branch=branch,
                    )
                    jirade_prs_found += 1
        await scan_github.close()
        if jirade_prs_found > 0:
            print(f"  Found {jirade_prs_found} existing jirade PR(s)", flush=True)
        tracked_prs = tracker.get_open_prs(repo_config.full_repo_name)
    except Exception as e:
        logger.warning(f"Failed to scan for existing PRs: {e}")

    print(f"Watching {repo_config.full_repo_name}...", flush=True)
    print(f"Polling interval: {interval} seconds", flush=True)
    print(f"Jira project: {repo_config.jira.project_key}", flush=True)
    print(f"Trigger status: \"{repo_config.agent.status}\"", flush=True)
    print(f"Done status: \"{repo_config.agent.done_status}\"", flush=True)
    if dbt_cloud_client:
        job_info = f" (job ID: {dbt_cloud_ci_job_id})" if dbt_cloud_ci_job_id else ""
        print(f"dbt Cloud: enabled{job_info}", flush=True)
    if tracked_prs:
        print(f"Tracking {len(tracked_prs)} open PRs for CI/review monitoring", flush=True)
    print(flush=True)
    print("Press Ctrl+C to stop", flush=True)
    print("-" * 50, flush=True)

    processed_prs: set[int] = set()
    processed_comments: set[int] = set()
    processing_tickets: set[str] = set()
    ticket_pattern = rf"\b({re.escape(repo_config.jira.project_key)}-\d+)\b"

    agent = JiraAgent(settings, repo_config)

    def timestamp() -> str:
        return datetime.now(timezone.utc).strftime("%H:%M:%S")

    try:
        while True:
            try:
                access_token = auth_manager.jira.get_access_token()
                cloud_id = auth_manager.jira.get_cloud_id()
                jira = JiraClient(cloud_id, access_token)

                jql = f'project = {repo_config.jira.project_key} AND status = "{repo_config.agent.status}"'
                tickets = await jira.search_issues(jql, max_results=10)

                for ticket in tickets:
                    ticket_key = ticket.get("key")

                    if ticket_key in processing_tickets:
                        continue

                    print(f"[{timestamp()}] Found ticket {ticket_key} in \"{repo_config.agent.status}\"", flush=True)
                    processing_tickets.add(ticket_key)

                    print(f"[{timestamp()}] Processing {ticket_key}...", flush=True)
                    result = await agent.process_single_ticket(ticket_key)

                    if result.get("status") == "completed":
                        print(f"[{timestamp()}] ✓ {ticket_key} -> PR created: {result.get('pr_url', 'N/A')}", flush=True)
                        pr_match = re.search(r"/pull/(\d+)", result.get("pr_url", ""))
                        if pr_match:
                            tracker.add_pr(
                                pr_number=int(pr_match.group(1)),
                                pr_url=result["pr_url"],
                                repo=repo_config.full_repo_name,
                                ticket_key=ticket_key,
                                branch=f"feat/{ticket_key}",
                            )
                    elif result.get("status") == "skipped":
                        print(f"[{timestamp()}] ○ {ticket_key} skipped: {result.get('reason', 'N/A')}", flush=True)
                        processing_tickets.discard(ticket_key)
                    else:
                        print(f"[{timestamp()}] ✗ {ticket_key} failed: {result.get('error', 'Unknown error')}", flush=True)
                        processing_tickets.discard(ticket_key)

                await jira.close()

                github = GitHubClient(
                    settings.github_token,
                    repo_config.repo.owner,
                    repo_config.repo.name,
                )

                tracked_prs = tracker.get_open_prs(repo_config.full_repo_name)

                for tracked in tracked_prs:
                    pr_number = tracked.pr_number

                    try:
                        pr = await github.get_pull_request(pr_number)

                        if pr.get("state") == "closed":
                            if pr.get("merged_at"):
                                print(f"[{timestamp()}] PR #{pr_number} was merged", flush=True)
                                tracker.update_pr(repo_config.full_repo_name, pr_number, status="merged")

                                result = await agent.transition_ticket_to_done(tracked.ticket_key)
                                if result.get("success"):
                                    print(f"[{timestamp()}] ✓ {tracked.ticket_key} transitioned to Done", flush=True)
                                processing_tickets.discard(tracked.ticket_key)
                            else:
                                print(f"[{timestamp()}] PR #{pr_number} was closed without merge", flush=True)
                                tracker.update_pr(repo_config.full_repo_name, pr_number, status="closed")
                            continue

                        checks = await github.get_check_runs(pr["head"]["sha"])
                        failed_checks = [c for c in checks if c.get("conclusion") == "failure"]

                        # Also check dbt Cloud for CI run status
                        dbt_cloud_failed = False
                        if dbt_cloud_client and dbt_cloud_ci_job_id:
                            try:
                                dbt_runs = await dbt_cloud_client.get_runs_for_pr(
                                    int(dbt_cloud_ci_job_id), pr_number, limit=3
                                )
                                for run in dbt_runs:
                                    if run.get("status") == RunStatus.ERROR:
                                        dbt_cloud_failed = True
                                        # Add a synthetic "dbt Cloud CI" failure if not already in GitHub checks
                                        if not any("dbt" in c.get("name", "").lower() for c in failed_checks):
                                            failed_checks.append({"name": "dbt Cloud CI", "conclusion": "failure"})
                                        break
                            except Exception as e:
                                logger.debug(f"Failed to check dbt Cloud runs: {e}")

                        if failed_checks:
                            current_ci = tracked.ci_status
                            if current_ci != "failure":
                                failure_names = [c['name'] if isinstance(c, dict) else c for c in failed_checks]
                                print(f"[{timestamp()}] ⚠️  PR #{pr_number} has CI failures: {failure_names}", flush=True)
                                tracker.update_pr(repo_config.full_repo_name, pr_number, ci_status="failure")

                                print(f"[{timestamp()}] 🔧 Attempting to fix CI failures...", flush=True)
                                fix_result = await agent.fix_ci_failures(pr_number)

                                if fix_result.get("fixed"):
                                    print(f"[{timestamp()}] ✓ CI fix pushed: {fix_result.get('strategy', 'auto')}", flush=True)
                                    tracker.update_pr(repo_config.full_repo_name, pr_number, ci_status="pending")
                                else:
                                    print(f"[{timestamp()}] ✗ Could not auto-fix: {fix_result.get('error', 'unknown')}", flush=True)
                        else:
                            pending_checks = [c for c in checks if c.get("status") != "completed"]
                            if not pending_checks and tracked.ci_status != "success":
                                print(f"[{timestamp()}] ✓ PR #{pr_number} CI passed", flush=True)
                                tracker.update_pr(repo_config.full_repo_name, pr_number, ci_status="success")

                        # Check both review comments (inline) and issue comments (general)
                        review_comments = await github.get_pr_review_comments(pr_number)
                        issue_comments = await github.get_pr_comments(pr_number)

                        # Filter review comments
                        new_review = [c for c in review_comments if c["id"] not in processed_comments]
                        actionable_review = [
                            c for c in new_review
                            if not c.get("user", {}).get("login", "").endswith("[bot]")
                            and "[jirade]" not in c.get("body", "")
                        ]

                        # Filter issue comments (use different ID namespace to avoid collision)
                        new_issue = [c for c in issue_comments if f"issue_{c['id']}" not in processed_comments]
                        actionable_issue = [
                            c for c in new_issue
                            if not c.get("user", {}).get("login", "").endswith("[bot]")
                            and "[jirade]" not in c.get("body", "")
                        ]

                        all_actionable = actionable_review + actionable_issue

                        if all_actionable and not tracked.feedback_addressed:
                            comment_type = "review" if actionable_review else "general"
                            print(f"[{timestamp()}] 💬 PR #{pr_number} has {len(all_actionable)} new {comment_type} comment(s)", flush=True)
                            tracker.update_pr(repo_config.full_repo_name, pr_number, has_feedback=True)

                            for comment in all_actionable[:3]:
                                body = comment.get("body", "")[:80]
                                user = comment.get("user", {}).get("login", "unknown")
                                print(f"           @{user}: {body}...", flush=True)

                            # Address the comments
                            print(f"[{timestamp()}] 🔧 Attempting to address comments...", flush=True)
                            address_result = await agent.address_review_comments(
                                pr_number, all_actionable, is_issue_comment=(len(actionable_issue) > 0)
                            )

                            if address_result.get("success"):
                                addressed_count = address_result.get("addressed", 0)
                                print(f"[{timestamp()}] ✓ Addressed {addressed_count} comment(s)", flush=True)
                                tracker.update_pr(repo_config.full_repo_name, pr_number, feedback_addressed=True)
                            else:
                                print(f"[{timestamp()}] ✗ Could not address comments: {address_result.get('error', 'unknown')}", flush=True)

                            # Mark all as processed regardless of outcome
                            for comment in actionable_review:
                                processed_comments.add(comment["id"])
                            for comment in actionable_issue:
                                processed_comments.add(f"issue_{comment['id']}")

                    except Exception as e:
                        print(f"[{timestamp()}] Error checking PR #{pr_number}: {e}", flush=True)

                prs = await github.list_pull_requests(state="closed")

                for pr in prs:
                    pr_number = pr.get("number")
                    merged_at = pr.get("merged_at")

                    if not merged_at or pr_number in processed_prs:
                        continue

                    title = pr.get("title", "")

                    # Only process PRs created by jirade
                    if "[jirade]" not in title:
                        processed_prs.add(pr_number)
                        continue

                    branch = pr.get("head", {}).get("ref", "")
                    match = re.search(ticket_pattern, f"{title} {branch}", re.IGNORECASE)

                    if not match:
                        processed_prs.add(pr_number)
                        continue

                    ticket_key = match.group(1).upper()

                    print(f"[{timestamp()}] PR #{pr_number} merged -> transitioning {ticket_key} to \"{repo_config.agent.done_status}\"", flush=True)

                    result = await agent.transition_ticket_to_done(ticket_key)

                    if result.get("success"):
                        print(f"[{timestamp()}] ✓ {ticket_key} transitioned to {result.get('transition', 'Done')}", flush=True)
                        processing_tickets.discard(ticket_key)
                    else:
                        print(f"[{timestamp()}] ✗ Failed to transition {ticket_key}: {result.get('error')}", flush=True)

                    processed_prs.add(pr_number)

                await github.close()

            except Exception as e:
                print(f"[{timestamp()}] Error during poll: {e}", flush=True)

            await asyncio.sleep(interval)

    except KeyboardInterrupt:
        print("\nStopping watch...", flush=True)
        await agent.close()
        if dbt_cloud_client:
            await dbt_cloud_client.close()
        return 0


async def handle_process(args: dict, settings) -> int:
    """Process tickets from a Jira board."""
    from .agent import JiraAgent

    status_filter = args["--status"]
    limit = int(args["--limit"])
    dry_run = args["--dry-run"]

    repo_config = load_config_with_fallback(args.get("--config"))

    agent = JiraAgent(settings, repo_config, dry_run=dry_run)
    results = await agent.process_tickets(status_filter=status_filter, limit=limit)

    print(f"\nProcessed {len(results)} tickets:")
    for result in results:
        status_icon = "✓" if result["status"] == "completed" else "○" if result["status"] == "skipped" else "✗"
        print(f"  {status_icon} {result['ticket']}: {result['status']}")
        if result.get("pr_url"):
            print(f"    PR: {result['pr_url']}")
        if result.get("error"):
            print(f"    Error: {result['error']}")

    return 0


def _post_pr_flow(
    pr_number: int,
    pr_url: str,
    repo_config,
    settings,
    ticket_key: str,
    args: dict,
) -> int:
    """Handle post-PR setup: reviewer selection and watch offer."""
    import questionary
    from questionary import Style

    from .clients.github_client import GitHubClient
    from .pr_tracker import PRTracker

    custom_style = Style([
        ("qmark", "fg:cyan bold"),
        ("question", "fg:white bold"),
        ("answer", "fg:green bold"),
        ("pointer", "fg:cyan bold"),
        ("highlighted", "fg:cyan bold"),
    ])

    tracker = PRTracker()
    tracker.add_pr(
        pr_number=pr_number,
        pr_url=pr_url,
        repo=repo_config.full_repo_name,
        ticket_key=ticket_key,
        branch=f"feat/{ticket_key}",
    )

    print()
    print("─" * 50)
    print("Post-PR Setup")
    print("─" * 50)

    add_reviewers = questionary.confirm(
        "Add reviewers to this PR?",
        default=True,
        style=custom_style,
    ).ask()

    if add_reviewers:
        github = GitHubClient(
            settings.github_token,
            repo_config.repo.owner,
            repo_config.repo.name,
        )

        async def fetch_and_add_reviewers():
            try:
                print("Fetching suggested reviewers...")
                suggested = await github.get_suggested_reviewers(pr_number)
                return suggested
            finally:
                await github.close()

        async def add_reviewers_to_pr(reviewers):
            github2 = GitHubClient(
                settings.github_token,
                repo_config.repo.owner,
                repo_config.repo.name,
            )
            try:
                await github2.request_reviewers(pr_number, reviewers)
            finally:
                await github2.close()

        try:
            suggested = asyncio.run(fetch_and_add_reviewers())

            if suggested:
                choices = [
                    questionary.Choice(
                        title=f"@{r['login']}",
                        value=r["login"],
                    )
                    for r in suggested
                ]

                selected = questionary.checkbox(
                    "Select reviewers (space to select, enter to confirm):",
                    choices=choices,
                    style=custom_style,
                ).ask()

                if selected:
                    asyncio.run(add_reviewers_to_pr(selected))
                    print(f"✓ Added reviewers: {', '.join(selected)}")
            else:
                print("No suggested reviewers found.")
                manual = questionary.text(
                    "Enter reviewer usernames (comma-separated, or leave empty):",
                    style=custom_style,
                ).ask()
                if manual:
                    reviewers = [r.strip().lstrip("@") for r in manual.split(",")]
                    asyncio.run(add_reviewers_to_pr(reviewers))
                    print(f"✓ Added reviewers: {', '.join(reviewers)}")

        except Exception as e:
            print(f"Could not add reviewers: {e}")

    print()
    watch_now = questionary.confirm(
        "Watch this PR for CI failures and review comments?",
        default=True,
        style=custom_style,
    ).ask()

    if watch_now:
        print()
        print("Starting watch mode...")
        print("The agent will monitor for CI failures and respond to feedback.")
        print("Press Ctrl+C to stop watching.")
        print()

        watch_args = {
            "--config": args.get("--config"),
            "--interval": 30,
        }
        return asyncio.run(handle_watch(watch_args, settings))
    else:
        print()
        print("You can watch later with: jirade watch")
        print("The agent will find your PRs and respond to any that need attention.")

    return 0


async def handle_process_ticket(args: dict, settings) -> dict:
    """Process a single ticket."""
    import re

    from .agent import JiraAgent

    ticket_key = args["<ticket_key>"]
    dry_run = args["--dry-run"]

    repo_config = load_config_with_fallback(args.get("--config"))

    agent = JiraAgent(settings, repo_config, dry_run=dry_run)
    result = await agent.process_single_ticket(ticket_key)

    status_icon = "✓" if result["status"] == "completed" else "○" if result["status"] == "skipped" else "✗"
    print(f"{status_icon} {result['ticket']}: {result['status']}")
    if result.get("pr_url"):
        print(f"  PR: {result['pr_url']}")
    if result.get("error"):
        print(f"  Error: {result['error']}")

    exit_code = 0 if result["status"] in ("completed", "skipped") else 1

    if result.get("status") == "completed" and result.get("pr_url") and not dry_run:
        pr_url = result["pr_url"]
        pr_match = re.search(r"/pull/(\d+)", pr_url)
        if pr_match:
            return {
                "exit_code": exit_code,
                "post_pr_info": {
                    "pr_number": int(pr_match.group(1)),
                    "pr_url": pr_url,
                    "repo_config": repo_config,
                    "ticket_key": ticket_key,
                },
            }

    return {"exit_code": exit_code}


async def handle_check_pr(args: dict, settings) -> int:
    """Check PR status."""
    from .agent import JiraAgent

    pr_number = int(args["<pr_number>"])

    repo_config = load_config_with_fallback(args.get("--config"))

    agent = JiraAgent(settings, repo_config)
    status = await agent.check_pr_status(pr_number)

    print(f"PR #{pr_number} Status:")
    print(f"  State: {status.get('state', 'unknown')}")
    print(f"  Mergeable: {status.get('mergeable', 'unknown')}")
    print(f"  CI Status: {status.get('ci_status', 'unknown')}")
    if status.get("pending_reviews"):
        print(f"  Pending Reviews: {len(status['pending_reviews'])}")
    if status.get("failed_checks"):
        print(f"  Failed Checks: {', '.join(status['failed_checks'])}")

    return 0


async def handle_fix_ci(args: dict, settings) -> int:
    """Attempt to fix CI failures on a PR."""
    from .agent import JiraAgent

    pr_number = int(args["<pr_number>"])

    repo_config = load_config_with_fallback(args.get("--config"))

    agent = JiraAgent(settings, repo_config)
    result = await agent.fix_ci_failures(pr_number)

    if result["fixed"]:
        print(f"✓ CI issues fixed for PR #{pr_number}")
        if result.get("commit_sha"):
            print(f"  New commit: {result['commit_sha']}")
    else:
        print(f"✗ Could not fix CI issues for PR #{pr_number}")
        if result.get("error"):
            print(f"  Error: {result['error']}")

    return 0 if result["fixed"] else 1


def handle_serve(args: dict, settings) -> int:
    """Start the webhook server."""
    from .triggers.server import run_server

    port = int(args["--port"])
    host = args["--host"]
    config_dir = Path(args["--config-dir"])

    print(f"Starting webhook server on {host}:{port}")
    print(f"Config directory: {config_dir}")
    run_server(host=host, port=port, config_dir=config_dir, settings=settings)
    return 0


def handle_init(args: dict, settings) -> int:
    """Interactive setup for jirade in a repository."""
    import questionary
    from questionary import Style

    from .auth import AuthManager
    from .repo_config.loader import REPO_CONFIG_FILENAME

    custom_style = Style([
        ("qmark", "fg:cyan bold"),
        ("question", "fg:white bold"),
        ("answer", "fg:green bold"),
        ("pointer", "fg:cyan bold"),
        ("highlighted", "fg:cyan bold"),
    ])

    print("=" * 50)
    print("  Jirade - Repository Setup")
    print("=" * 50)
    print()

    print("Checking required credentials...")
    print("-" * 30)

    auth_manager = AuthManager(settings)
    all_credentials_ok = True

    if settings.has_anthropic_key:
        print("✓ Anthropic API key: configured")
    else:
        print("✗ Anthropic API key: NOT SET")
        print()
        print("  The Anthropic API key is required for the Claude AI agent.")
        print("  Get your API key at: https://console.anthropic.com/settings/keys")
        print()
        all_credentials_ok = False

        enter_key = questionary.confirm(
            "Would you like to enter your Anthropic API key now?",
            default=True,
            style=custom_style,
        ).ask()

        if enter_key:
            api_key = questionary.password(
                "Enter your Anthropic API key:",
                style=custom_style,
            ).ask()

            if api_key and api_key.strip():
                from .auth.token_store import TokenStore

                store = TokenStore()
                store.save("anthropic", {"api_key": api_key.strip()})

                if store._use_keyring:
                    import platform
                    system = platform.system()
                    if system == "Darwin":
                        location = "macOS Keychain"
                    elif system == "Linux":
                        location = "system keyring (Secret Service)"
                    elif system == "Windows":
                        location = "Windows Credential Manager"
                    else:
                        location = "system keyring"
                    print(f"✓ Anthropic API key: saved to {location}")
                else:
                    print(f"✓ Anthropic API key: saved to {store.fallback_dir}/anthropic_tokens.json")

                from .config import get_settings
                settings = get_settings()
                all_credentials_ok = settings.has_anthropic_key
            else:
                print("No key entered.")
        else:
            continue_anyway = questionary.confirm(
                "Continue setup without Anthropic API key?",
                default=False,
                style=custom_style,
            ).ask()
            if not continue_anyway:
                print("\nSetup cancelled. Please provide an Anthropic API key and try again.")
                return 1

    if settings.has_github_token:
        print("✓ GitHub token: configured")
    else:
        print("✗ GitHub token: NOT SET")
        print()
        print("  GitHub token is required for creating PRs and accessing repos.")
        print("  You can either:")
        print()
        print("  1. Install and authenticate with GitHub CLI (recommended):")
        print("     brew install gh")
        print("     gh auth login")
        print()
        print("  2. Or set a personal access token:")
        print("     export JIRADE_GITHUB_TOKEN='your-token-here'")
        print()
        all_credentials_ok = False

        setup_gh = questionary.confirm(
            "Would you like to authenticate with GitHub CLI now?",
            default=True,
            style=custom_style,
        ).ask()

        if setup_gh:
            import subprocess
            print("\nLaunching 'gh auth login'...")
            result = subprocess.run(["gh", "auth", "login"], capture_output=False)
            if result.returncode == 0:
                from .config import get_settings
                settings = get_settings()
                if settings.has_github_token:
                    print("✓ GitHub token: now configured")
                    all_credentials_ok = True
        else:
            continue_anyway = questionary.confirm(
                "Continue setup without GitHub token?",
                default=False,
                style=custom_style,
            ).ask()
            if not continue_anyway:
                print("\nSetup cancelled. Please configure GitHub access and try again.")
                return 1

    jira_authenticated = auth_manager.jira.is_authenticated()
    if jira_authenticated:
        print("✓ Jira: authenticated")
    else:
        print("✗ Jira: NOT AUTHENTICATED")
        print()
        print("  Jira OAuth is required for reading tickets and updating status.")
        print()

        login_now = questionary.confirm(
            "Login to Jira now?",
            default=True,
            style=custom_style,
        ).ask()

        if login_now:
            auth_manager.login("jira")
            jira_authenticated = auth_manager.jira.is_authenticated()
            if jira_authenticated:
                print("✓ Jira: now authenticated")
        else:
            all_credentials_ok = False
            continue_anyway = questionary.confirm(
                "Continue setup without Jira authentication?",
                default=False,
                style=custom_style,
            ).ask()
            if not continue_anyway:
                print("\nSetup cancelled. Please authenticate with Jira and try again.")
                return 1

    print()
    if all_credentials_ok:
        print("All required credentials are configured!")
    else:
        print("Some credentials are missing - agent may not work fully.")
    print()

    print("Repository Configuration")
    print("-" * 30)

    existing_config = find_repo_config()
    if existing_config:
        overwrite = questionary.confirm(
            f"Config already exists at {existing_config}. Overwrite?",
            default=False,
            style=custom_style,
        ).ask()
        if not overwrite:
            print("Setup cancelled.")
            return 0

    repo_info = get_git_remote_info()
    if repo_info:
        owner, name = repo_info
        print(f"Detected repository: {owner}/{name}")
        use_detected = questionary.confirm(
            "Use this repository?",
            default=True,
            style=custom_style,
        ).ask()
        if not use_detected:
            repo_info = None

    if not repo_info:
        repo_str = questionary.text(
            "Repository (owner/name):",
            style=custom_style,
        ).ask()
        if not repo_str or "/" not in repo_str:
            print("Error: Repository must be in owner/name format")
            return 1
        owner, name = repo_str.split("/", 1)

    default_branch = questionary.text(
        "Default branch:",
        default="main",
        style=custom_style,
    ).ask()

    print("Jira Project Configuration")
    print("-" * 30)

    project_key = questionary.text(
        "Jira project key (e.g., AENG):",
        style=custom_style,
    ).ask()

    if not project_key:
        print("Error: Project key is required")
        return 1

    project_key = project_key.upper()

    board_id = None
    if jira_authenticated:
        list_boards = questionary.confirm(
            "List available Jira boards?",
            default=True,
            style=custom_style,
        ).ask()

        if list_boards:
            try:
                from .clients.jira_client import JiraClient

                async def fetch_boards():
                    access_token = auth_manager.jira.get_access_token()
                    cloud_id = auth_manager.jira.get_cloud_id()
                    jira = JiraClient(cloud_id, access_token)
                    boards = await jira.get_boards(project_key=project_key)
                    await jira.close()
                    return boards

                boards = asyncio.run(fetch_boards())

                if boards:
                    board_choices = [
                        questionary.Choice(
                            title=f"{b['name']} (ID: {b['id']})",
                            value=b["id"],
                        )
                        for b in boards
                    ]
                    board_choices.append(questionary.Choice(title="Enter manually", value=None))

                    board_id = questionary.select(
                        "Select a board:",
                        choices=board_choices,
                        style=custom_style,
                    ).ask()
            except Exception as e:
                print(f"Could not list boards: {e}")

    if board_id is None:
        board_id_str = questionary.text(
            "Jira board ID (optional, press Enter to skip):",
            style=custom_style,
        ).ask()
        board_id = int(board_id_str) if board_id_str else None

    print()

    print("Agent Trigger Configuration")
    print("-" * 30)

    trigger_status = questionary.text(
        "Status that triggers the agent:",
        default="Ready for Agent",
        style=custom_style,
    ).ask()

    done_status = questionary.text(
        "Status after PR is merged:",
        default="Done",
        style=custom_style,
    ).ask()

    in_progress_status = questionary.text(
        "Status while agent is working:",
        default="In Progress",
        style=custom_style,
    ).ask()

    print()

    print("Repository Features (Auto-detected)")
    print("-" * 30)

    cwd = Path.cwd()
    dbt_projects = list(cwd.glob("**/dbt_project.yml"))
    dbt_enabled = len(dbt_projects) > 0

    if dbt_enabled:
        print(f"Detected {len(dbt_projects)} dbt project(s)")
        dbt_enabled = questionary.confirm(
            "Enable dbt tools?",
            default=True,
            style=custom_style,
        ).ask()

    ci_system = "github_actions"
    if (cwd / ".circleci").exists():
        ci_system = "circleci"
    elif (cwd / "Jenkinsfile").exists():
        ci_system = "jenkins"

    print(f"Detected CI system: {ci_system}")

    output_path = args.get("--output") or REPO_CONFIG_FILENAME

    config_content = f"""# Jirade configuration for {owner}/{name}
# Generated by: jirade init

repo:
  owner: "{owner}"
  name: "{name}"
  default_branch: "{default_branch}"
  pr_target_branch: "{default_branch}"

jira:
  project_key: "{project_key}"
  board_id: {board_id if board_id else 'null'}

agent:
  status: "{trigger_status}"
  done_status: "{done_status}"
  in_progress_status: "{in_progress_status}"

branching:
  pattern: "{{type}}/{{ticket_key}}-{{description}}"
  types:
    feature: "feat"
    bugfix: "fix"
    refactor: "refactor"

pull_request:
  title_pattern: "{{type}}({{scope}}): {{description}} ({{ticket_key}})"

commits:
  style: "conventional"
  ticket_in_message: true

skip:
  comment_phrase: "[AGENT-SKIP]"
  labels:
    - "no-automation"
    - "manual-only"

dbt:
  enabled: {str(dbt_enabled).lower()}

ci:
  system: "{ci_system}"
  auto_fix:
    - "pre-commit"

learning:
  enabled: true
"""

    output = Path(output_path)
    output.write_text(config_content)

    print()
    print("=" * 50)
    print(f"✓ Config created: {output_path}")
    print("=" * 50)
    print()
    print("Next steps:")
    print("  1. Review and customize the config if needed")
    print("  2. Run 'jirade auth login' if not authenticated")
    print("  3. Run 'jirade health' to test connections")
    print("  4. Run 'jirade list-tickets' to see available tickets")
    print()

    return 0


async def handle_health(args: dict, settings) -> int:
    """Test all service connections."""
    import httpx
    from anthropic import Anthropic

    from .auth import AuthManager

    print("Health Check")
    print("=" * 50)

    all_ok = True
    config_path = args.get("--config")
    repo_config = None

    if config_path:
        from .repo_config.loader import ConfigLoader

        try:
            loader = ConfigLoader()
            repo_config = loader.load_from_file(config_path)
            print(f"Config: {repo_config.full_repo_name}")
        except Exception as e:
            print(f"Config: FAILED - {e}")
            all_ok = False

    print()

    print("Anthropic API:")
    if settings.has_anthropic_key:
        try:
            client = Anthropic(api_key=settings.anthropic_api_key)
            response = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=10,
                messages=[{"role": "user", "content": "Say OK"}],
            )
            print("  Status: OK")
            print(f"  Model configured: {settings.claude_model}")
        except Exception as e:
            print(f"  Status: FAILED - {e}")
            all_ok = False
    else:
        print("  Status: NOT CONFIGURED (ANTHROPIC_API_KEY not set)")
        all_ok = False

    print()

    print("Jira:")
    auth_manager = AuthManager(settings)
    jira_token = auth_manager.jira.get_access_token() if auth_manager.jira.is_authenticated() else None
    if jira_token:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "https://api.atlassian.com/oauth/token/accessible-resources",
                    headers={"Authorization": f"Bearer {jira_token}"},
                )
                if response.status_code == 200:
                    resources = response.json()
                    if resources:
                        print("  Status: OK")
                        print(f"  Accessible sites: {len(resources)}")
                        for r in resources[:3]:
                            print(f"    - {r.get('name', 'Unknown')} ({r.get('url', '')})")

                        if repo_config and repo_config.jira.project_key:
                            cloud_id = resources[0]["id"]
                            project_response = await client.get(
                                f"https://api.atlassian.com/ex/jira/{cloud_id}/rest/api/3/project/{repo_config.jira.project_key}",
                                headers={"Authorization": f"Bearer {jira_token}"},
                            )
                            if project_response.status_code == 200:
                                project = project_response.json()
                                print(f"  Project {repo_config.jira.project_key}: OK ({project.get('name', '')})")
                            else:
                                print(f"  Project {repo_config.jira.project_key}: FAILED (status {project_response.status_code})")
                                all_ok = False
                    else:
                        print("  Status: WARNING - No accessible Jira sites")
                else:
                    print(f"  Status: FAILED - Token invalid or expired (status {response.status_code})")
                    all_ok = False
        except Exception as e:
            print(f"  Status: FAILED - {e}")
            all_ok = False
    else:
        print("  Status: NOT AUTHENTICATED")
        print("  Run: jirade auth login --service=jira")
        all_ok = False

    print()

    print("GitHub:")
    if settings.has_github_token:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "https://api.github.com/user",
                    headers={
                        "Authorization": f"Bearer {settings.github_token}",
                        "Accept": "application/vnd.github.v3+json",
                    },
                )
                if response.status_code == 200:
                    user = response.json()
                    print("  Status: OK")
                    print(f"  User: {user.get('login', 'Unknown')}")

                    if repo_config:
                        repo_response = await client.get(
                            f"https://api.github.com/repos/{repo_config.repo.owner}/{repo_config.repo.name}",
                            headers={
                                "Authorization": f"Bearer {settings.github_token}",
                                "Accept": "application/vnd.github.v3+json",
                            },
                        )
                        if repo_response.status_code == 200:
                            repo = repo_response.json()
                            perms = repo.get("permissions", {})
                            print(f"  Repo {repo_config.full_repo_name}: OK")
                            print(f"    Push access: {'Yes' if perms.get('push') else 'No'}")
                        else:
                            print(f"  Repo {repo_config.full_repo_name}: FAILED (status {repo_response.status_code})")
                            all_ok = False
                else:
                    print(f"  Status: FAILED - Token invalid (status {response.status_code})")
                    all_ok = False
        except Exception as e:
            print(f"  Status: FAILED - {e}")
            all_ok = False
    else:
        print("  Status: NOT CONFIGURED")
        print("  Either run 'gh auth login' or set JIRADE_GITHUB_TOKEN")
        all_ok = False

    print()

    print("Databricks:")
    if settings.has_databricks:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{settings.databricks_host}/api/2.0/clusters/list",
                    headers={"Authorization": f"Bearer {settings.databricks_token}"},
                )
                if response.status_code == 200:
                    print("  Status: OK")
                    print(f"  Host: {settings.databricks_host}")
                else:
                    print(f"  Status: FAILED - (status {response.status_code})")
                    all_ok = False
        except Exception as e:
            print(f"  Status: FAILED - {e}")
            all_ok = False
    else:
        print("  Status: NOT CONFIGURED (optional)")

    print()

    print("dbt Cloud:")
    if settings.has_dbt_cloud:
        try:
            from .clients.dbt_cloud_client import DbtCloudClient

            dbt_client = DbtCloudClient(
                api_token=settings.dbt_cloud_api_token,
                account_id=settings.dbt_cloud_account_id,
                base_url=settings.dbt_cloud_base_url,
            )
            result = await dbt_client.health_check()
            await dbt_client.close()

            if result.get("status") == "ok":
                print("  Status: OK")
                print(f"  Account ID: {settings.dbt_cloud_account_id}")
                print(f"  Jobs available: {result.get('job_count', 0)}")
                if settings.dbt_cloud_ci_job_id:
                    print(f"  CI Job ID: {settings.dbt_cloud_ci_job_id}")
                elif settings.dbt_cloud_project_id:
                    # Try to find CI job
                    dbt_client2 = DbtCloudClient(
                        api_token=settings.dbt_cloud_api_token,
                        account_id=settings.dbt_cloud_account_id,
                        base_url=settings.dbt_cloud_base_url,
                    )
                    ci_job = await dbt_client2.find_ci_job(int(settings.dbt_cloud_project_id))
                    await dbt_client2.close()
                    if ci_job:
                        print(f"  CI Job ID: {ci_job['id']} (auto-detected)")
                    else:
                        print("  CI Job: Not found (set JIRADE_DBT_CLOUD_CI_JOB_ID)")
            else:
                print(f"  Status: FAILED - {result.get('error', 'Unknown error')}")
        except Exception as e:
            print(f"  Status: FAILED - {e}")
    else:
        print("  Status: NOT CONFIGURED (optional)")
        print("  Set JIRADE_DBT_CLOUD_API_TOKEN and JIRADE_DBT_CLOUD_ACCOUNT_ID to enable")

    print()
    print("=" * 50)
    if all_ok:
        print("All required services are healthy!")
        return 0
    else:
        print("Some services have issues. Please fix them before using the agent.")
        return 1


def handle_learn(args: dict, settings) -> int:
    """Handle learn commands."""
    if args.get("status"):
        return handle_learn_status(args, settings)
    elif args.get("publish"):
        return handle_learn_publish(args, settings)
    elif args.get("list"):
        return handle_learn_list(args, settings)
    return 1


def handle_learn_status(args: dict, settings) -> int:
    """Show pending learnings in workspace."""
    from .learning import LearningStorage

    storage = LearningStorage(settings.workspace_dir)
    learnings = storage.collect_from_workspace()

    if not learnings:
        print("No pending learnings found in workspace.")
        print(f"Workspace: {settings.workspace_dir}")
        return 0

    print(f"Found {len(learnings)} pending learnings:")
    print("-" * 60)

    by_repo: dict[str, list] = {}
    for learning in learnings:
        by_repo.setdefault(learning.repo, []).append(learning)

    for repo, repo_learnings in sorted(by_repo.items()):
        print(f"\n{repo}:")
        for learning in repo_learnings:
            print(f"  - [{learning.category.value}] {learning.title}")
            print(f"    Ticket: {learning.ticket}, Subcategory: {learning.subcategory}")

    print()
    print("Run 'jirade learn publish' to create a PR with these learnings.")
    return 0


def handle_learn_publish(args: dict, settings) -> int:
    """Publish learnings to jirade repo."""
    from .learning import LearningPublisher

    dry_run = args.get("--dry-run", False)
    jirade_repo = args.get("--jirade-repo") or getattr(
        settings, "jirade_repo", "djayatillake/jirade"
    )

    if not settings.has_github_token:
        print("Error: GitHub token not configured")
        print("Either run 'gh auth login' or set JIRADE_GITHUB_TOKEN")
        return 1

    print(f"Publishing learnings to {jirade_repo}...")
    if dry_run:
        print("(dry-run mode)")
    print()

    publisher = LearningPublisher(
        github_token=settings.github_token,
        jirade_repo=jirade_repo,
        workspace_dir=settings.workspace_dir,
    )

    result = publisher.publish(dry_run=dry_run)

    if result["status"] == "no_learnings":
        print("No learnings to publish.")
        return 0

    if result["status"] == "all_duplicates":
        print("All learnings already exist in knowledge base.")
        return 0

    if result["status"] == "dry_run":
        print(f"Would publish {result['learnings_count']} learnings:")
        for file_path in result.get("files_to_create", []):
            print(f"  - {file_path}")
        return 0

    if result["status"] == "success":
        print(f"Successfully published {result['learnings_count']} learnings!")
        print(f"PR: {result['pr_url']}")
        return 0

    print(f"Failed to publish: {result.get('message', 'Unknown error')}")
    return 1


def handle_learn_list(args: dict, settings) -> int:
    """List learnings in the knowledge base."""
    from .learning import LearningCategory
    from .learning.publisher import CATEGORY_DIRS, KNOWLEDGE_BASE_DIR
    from .learning.storage import LearningStorage

    category_filter = args.get("--category")

    kb_paths = [
        Path.cwd() / KNOWLEDGE_BASE_DIR,
        settings.workspace_dir / "djayatillake-jirade" / KNOWLEDGE_BASE_DIR,
        Path(__file__).parent.parent / KNOWLEDGE_BASE_DIR,
    ]

    kb_path = None
    for path in kb_paths:
        if path.exists():
            kb_path = path
            break

    if not kb_path:
        print("Knowledge base not found.")
        print("Searched in:")
        for path in kb_paths:
            print(f"  - {path}")
        return 1

    print(f"Knowledge base: {kb_path}")
    print("-" * 60)

    storage = LearningStorage()
    total_count = 0

    for category, dir_name in CATEGORY_DIRS.items():
        if category_filter and category.value != category_filter:
            continue

        cat_path = kb_path / dir_name
        if not cat_path.exists():
            continue

        md_files = list(cat_path.glob("*.md"))
        md_files = [f for f in md_files if f.name != "README.md"]

        if not md_files:
            continue

        print(f"\n{category.value} ({len(md_files)} learnings):")

        for md_file in sorted(md_files)[:10]:
            learning = storage.parse_markdown(md_file)
            if learning:
                print(f"  - {learning.title}")
                print(f"    {learning.subcategory} | {learning.ticket}")

        if len(md_files) > 10:
            print(f"  ... and {len(md_files) - 10} more")

        total_count += len(md_files)

    if total_count == 0:
        print("\nNo learnings found in knowledge base.")
    else:
        print(f"\nTotal: {total_count} learnings")

    return 0


def handle_env(args: dict, settings) -> int:
    """Handle environment commands."""
    if args.get("check"):
        return handle_env_check(args, settings, auto_install=False)
    elif args.get("setup"):
        return handle_env_check(args, settings, auto_install=True)
    return 1


def handle_env_check(args: dict, settings, auto_install: bool = False) -> int:
    """Check environment for a repository."""
    from .environment import EnvironmentChecker, PackageInstaller
    from .environment.requirements import RequirementsParser
    from .tools.git_tools import GitTools

    repo_path_str = args.get("--repo-path")
    config_path = args.get("--config")

    if repo_path_str:
        repo_path = Path(repo_path_str)
        if not repo_path.exists():
            print(f"Error: Repository path does not exist: {repo_path}")
            return 1
        repo_config = None
    elif config_path:
        from .repo_config.loader import ConfigLoader

        loader = ConfigLoader()
        repo_config = loader.load_from_file(config_path)

        if not settings.has_github_token:
            print("Error: GitHub token required to clone repository")
            return 1

        print(f"Cloning {repo_config.full_repo_name}...")
        git = GitTools(settings.workspace_dir, settings.github_token)
        repo_path = git.clone_repo(repo_config.repo.owner, repo_config.repo.name)
        print(f"Repository at: {repo_path}")
        print()
    else:
        repo_path = Path.cwd()
        repo_config = load_config_with_fallback(args.get("--config"), required=False)

    print("System Tools")
    print("=" * 50)

    checker = EnvironmentChecker()
    report = checker.check_for_repo(repo_path, repo_config)

    for tool in report.tools:
        status = "✓" if tool.installed else "✗"
        req = "(required)" if tool.required else "(optional)"

        if tool.installed:
            version_str = f"v{tool.version}" if tool.version else ""
            print(f"  {status} {tool.name:<15} {version_str:<12} {req}")
        else:
            print(f"  {status} {tool.name:<15} {'NOT FOUND':<12} {req}")
            if tool.install_hint:
                print(f"      → {tool.install_hint}")

    print()

    print("Repository Requirements")
    print("=" * 50)

    parser = RequirementsParser(repo_path)
    reqs = parser.parse_all()

    if reqs.python_packages:
        installed = [r for r in reqs.python_packages if r.installed]
        missing = [r for r in reqs.python_packages if not r.installed]
        print(f"\nPython: {len(installed)} installed, {len(missing)} missing")

        if missing and not auto_install:
            print("  Missing packages:")
            for req in missing[:5]:
                print(f"    - {req.name}")
            if len(missing) > 5:
                print(f"    ... and {len(missing) - 5} more")

    if reqs.node_packages:
        installed = [r for r in reqs.node_packages if r.installed]
        missing = [r for r in reqs.node_packages if not r.installed]
        print(f"\nNode.js: {len(installed)} installed, {len(missing)} missing")

    if reqs.setup_commands:
        print("\nDetected Setup Commands:")
        for cmd in reqs.setup_commands:
            print(f"  $ {cmd}")

    print()

    if auto_install:
        print("Installing Dependencies")
        print("=" * 50)

        if report.missing_required:
            installer = PackageInstaller(repo_path, auto_confirm=True)
            for tool in report.missing_required:
                print(f"Installing {tool}...")
                result = installer.install_system_tool(tool)
                if result.success:
                    print(f"  ✓ {tool} installed")
                else:
                    print(f"  ✗ {tool} failed: {result.error}")

        missing_python, missing_node = parser.get_missing_packages()
        if missing_python or missing_node or reqs.setup_commands:
            installer = PackageInstaller(repo_path, auto_confirm=True)
            results = installer.install_repo_requirements()

            for result in results:
                if result.success:
                    print(f"  ✓ {result.package}")
                else:
                    print(f"  ✗ {result.package}: {result.error}")

        print()

    if report.missing_required:
        print(f"✗ Missing required tools: {', '.join(report.missing_required)}")
        return 1

    missing_python, missing_node = parser.get_missing_packages()
    if (missing_python or missing_node) and not auto_install:
        print("✗ Missing packages. Run 'jirade env setup' to install.")
        return 1

    print("✓ Environment is ready!")
    return 0


def main():
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
