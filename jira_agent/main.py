"""Jira Ticket Automation Agent CLI.

Usage:
    jira-agent init [--output=<path>]
    jira-agent list-tickets [--config=<path>] [--status=<status>] [--limit=<n>] [--interactive]
    jira-agent list-prs [--config=<path>] [--state=<state>]
    jira-agent watch [--config=<path>] [--interval=<seconds>]
    jira-agent process [--config=<path>] [--status=<status>] [--limit=<n>] [--dry-run]
    jira-agent process-ticket <ticket_key> [--config=<path>] [--dry-run]
    jira-agent check-pr <pr_number> [--config=<path>]
    jira-agent fix-ci <pr_number> [--config=<path>]
    jira-agent serve [--port=<port>] [--host=<host>] [--config-dir=<dir>]
    jira-agent auth login [--service=<service>]
    jira-agent auth status
    jira-agent auth logout [--service=<service>]
    jira-agent config show
    jira-agent config validate <config_path>
    jira-agent health [--config=<path>]
    jira-agent learn status
    jira-agent learn publish [--dry-run] [--jira-agent-repo=<repo>]
    jira-agent learn list [--category=<cat>]
    jira-agent env check [--config=<path>] [--repo-path=<path>]
    jira-agent env setup [--config=<path>] [--repo-path=<path>]
    jira-agent --help
    jira-agent --version

Commands:
    init            Initialize jira-agent for a repository (interactive setup)
    list-tickets    List tickets from a Jira board
    list-prs        List open PRs for the repository
    watch           Poll for merged PRs and auto-transition tickets to Done
    process         Process tickets from a Jira board
    process-ticket  Process a specific ticket by key
    check-pr        Check PR status and pending feedback
    fix-ci          Attempt to fix CI failures on a PR
    serve           Start webhook server for Jira/GitHub events
    auth            Manage OAuth authentication
    config          Show or validate configuration
    health          Test all service connections (Anthropic, Jira, GitHub, Databricks)
    learn           Manage agent learnings (captured from resolved failures)
    env             Check and setup environment (system tools, repo requirements)

Options:
    -h --help                Show this help message
    --version                Show version
    --config=<path>          Path to repo config file (auto-detects .jira-agent.yaml if not specified)
    --status=<status>        Filter tickets by Jira status (e.g., "To Do", "Ready for Dev")
    --state=<state>          Filter PRs by state: open, closed, all [default: open]
    --limit=<n>              Maximum tickets to process [default: 10]
    --interactive            Interactive mode: select ticket with arrow keys to process
    --dry-run                Preview actions without making changes
    --interval=<seconds>     Polling interval in seconds [default: 60]
    --port=<port>            Webhook server port [default: 8080]
    --host=<host>            Webhook server host [default: 0.0.0.0]
    --config-dir=<dir>       Directory containing repo config files [default: ./configs]
    --output=<path>          Output path for generated config [default: .jira-agent.yaml]
    --service=<service>      Service to authenticate: jira, github, databricks, or all [default: all]
    --jira-agent-repo=<repo> GitHub repo for jira-agent [default: djayatillake/jira-agent]
    --category=<cat>         Filter learnings by category: ci-failure, code-pattern, error-resolution
    --repo-path=<path>       Path to local repository to check/setup

Environment Variables:
    ANTHROPIC_API_KEY           Anthropic API key (or enter during 'init')
    JIRA_AGENT_JIRA_OAUTH_CLIENT_ID      Jira OAuth app client ID
    JIRA_AGENT_JIRA_OAUTH_CLIENT_SECRET  Jira OAuth app client secret
    JIRA_AGENT_GITHUB_TOKEN              GitHub personal access token
    JIRA_AGENT_DATABRICKS_HOST           Databricks workspace URL
    JIRA_AGENT_DATABRICKS_TOKEN          Databricks personal access token
    JIRA_AGENT_WEBHOOK_SECRET            Secret for webhook validation

Credential Storage:
    API keys entered during 'init' are stored securely in:
    - macOS: Keychain
    - Linux: Secret Service (GNOME Keyring/KWallet)
    - Windows: Windows Credential Manager
    - Fallback: ~/.jira-agent/tokens/ (with 0600 permissions)

Examples:
    # GETTING STARTED - Initialize jira-agent in your repo
    cd ~/repos/my-data-repo
    jira-agent init
    # Prompts for Jira project key, board ID, etc.
    # Creates .jira-agent.yaml in repo root

    # Once initialized, all commands auto-detect config from .jira-agent.yaml
    jira-agent list-tickets                    # Lists tickets for this repo
    jira-agent process-ticket AENG-1234        # Process a specific ticket
    jira-agent process --status="Ready for Dev" --limit=5  # Process multiple

    # Or specify a config file explicitly
    jira-agent list-tickets --config configs/my-repo.yaml

    # Authenticate with services (required before using agent)
    jira-agent auth login          # Login to all services
    jira-agent auth status         # Check authentication status

    # Test all service connections
    jira-agent health              # Test connections
    jira-agent health --config .jira-agent.yaml  # Test with specific config

    # Interactive mode: browse and select a ticket to process
    jira-agent list-tickets --interactive
    jira-agent list-tickets --status="To Do" --interactive

    # List open PRs
    jira-agent list-prs

    # Watch for merged PRs and auto-transition tickets
    jira-agent watch                        # Poll every 60s
    jira-agent watch --interval=120         # Poll every 2 minutes

    # Start webhook server (for CI/CD integration)
    jira-agent serve --port 8080 --config-dir ./configs

    # Agent learnings (captured from resolved failures)
    jira-agent learn status                 # View pending learnings
    jira-agent learn publish                # Create PR with learnings
    jira-agent learn publish --dry-run      # Preview only
    jira-agent learn list --category=ci-failure

    # Environment management
    jira-agent env check                    # Check current repo environment
    jira-agent env setup                    # Auto-install missing dependencies
"""

import asyncio
import logging
import sys
from pathlib import Path

from docopt import docopt

from .config import get_settings
from .repo_config.loader import ConfigLoader, find_repo_config, get_git_remote_info
from .utils.logger import setup_logging

__version__ = "0.1.0"


def load_config_with_fallback(args: dict, required: bool = True):
    """Load repo config with auto-detection fallback.

    Tries in order:
    1. Explicit --config path
    2. Auto-detect .jira-agent.yaml in current directory

    Args:
        args: CLI arguments dict.
        required: If True, raise error when no config found.

    Returns:
        RepoConfig or None.

    Raises:
        SystemExit: If required and no config found.
    """
    loader = ConfigLoader()
    config_path = args.get("--config")

    if config_path:
        return loader.load_from_file(config_path)

    # Try auto-detection
    auto_config = loader.auto_detect()
    if auto_config:
        return auto_config

    if required:
        print("Error: No config found.")
        print("Either:")
        print("  1. Run 'jira-agent init' to create .jira-agent.yaml in this repo")
        print("  2. Specify --config=<path> to a config file")
        sys.exit(1)

    return None


def main() -> int:
    """Main entry point for the CLI."""
    args = docopt(__doc__, version=f"jira-agent {__version__}")

    settings = get_settings()
    setup_logging(settings.log_level)
    logger = logging.getLogger(__name__)

    try:
        if args["list-tickets"]:
            return asyncio.run(handle_list_tickets(args, settings))
        elif args["list-prs"]:
            return asyncio.run(handle_list_prs(args, settings))
        elif args["watch"]:
            return asyncio.run(handle_watch(args, settings))
        elif args["auth"]:
            return handle_auth(args, settings)
        elif args["config"]:
            return handle_config_command(args, settings)
        elif args["process"]:
            return asyncio.run(handle_process(args, settings))
        elif args["process-ticket"]:
            return asyncio.run(handle_process_ticket(args, settings))
        elif args["check-pr"]:
            return asyncio.run(handle_check_pr(args, settings))
        elif args["fix-ci"]:
            return asyncio.run(handle_fix_ci(args, settings))
        elif args["serve"]:
            return handle_serve(args, settings)
        elif args["init"]:
            return asyncio.run(handle_init(args, settings))
        elif args["health"]:
            return asyncio.run(handle_health(args, settings))
        elif args["learn"]:
            return handle_learn(args, settings)
        elif args["env"]:
            return handle_env(args, settings)
        else:
            print(__doc__)
            return 1
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return 1


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


async def handle_list_tickets(args: dict, settings) -> int:
    """List tickets from a Jira board."""
    import questionary

    from .auth import AuthManager
    from .clients.jira_client import JiraClient

    status_filter = args["--status"]
    limit = int(args["--limit"]) if args["--limit"] else 20
    interactive = args["--interactive"]

    repo_config = load_config_with_fallback(args)

    auth_manager = AuthManager(settings)
    if not auth_manager.jira.is_authenticated():
        print("Error: Not authenticated with Jira. Run: jira-agent auth login --service=jira")
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
        # Use JQL search (works with standard scopes)
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

        # Interactive mode: use questionary for selection
        if interactive:
            return await _interactive_ticket_selection(tickets, settings, repo_config)

        # Non-interactive: just print the table
        print(f"{'Key':<12} {'Status':<20} {'Type':<12} {'Summary'}")
        print("-" * 80)

        for ticket in tickets:
            key = ticket.get("key", "")
            fields = ticket.get("fields", {})
            status = fields.get("status", {}).get("name", "Unknown")
            issue_type = fields.get("issuetype", {}).get("name", "Unknown")
            summary = fields.get("summary", "")

            # Truncate summary if too long
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

    # Custom style for the selection
    custom_style = Style([
        ("qmark", "fg:cyan bold"),
        ("question", "fg:white bold"),
        ("answer", "fg:green bold"),
        ("pointer", "fg:cyan bold"),
        ("highlighted", "fg:cyan bold"),
        ("selected", "fg:green"),
    ])

    # Build choices with ticket info
    choices = []
    for ticket in tickets:
        key = ticket.get("key", "")
        fields = ticket.get("fields", {})
        status = fields.get("status", {}).get("name", "Unknown")
        issue_type = fields.get("issuetype", {}).get("name", "Unknown")
        summary = fields.get("summary", "")

        # Truncate summary
        if len(summary) > 45:
            summary = summary[:42] + "..."

        # Format: KEY [Status] Summary
        label = f"{key:<12} [{status:<15}] {summary}"
        choices.append(questionary.Choice(title=label, value=ticket))

    # Add cancel option
    choices.append(questionary.Choice(title="Cancel", value=None))

    # Show selection prompt
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

    # Ask what to do
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
        # Show full ticket details
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
        print(f"  jira-agent process-ticket {ticket_key}")
        return 0

    # Process the ticket
    dry_run = action == "dry-run"

    if dry_run:
        print("Starting dry-run (no changes will be made)...")
    else:
        # Confirm before processing
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
    """List PRs for the repository."""
    import re

    from .clients.github_client import GitHubClient

    state = args["--state"] or "open"

    repo_config = load_config_with_fallback(args)

    if not settings.has_github_token:
        print("Error: GitHub token not configured")
        return 1

    github = GitHubClient(
        settings.github_token,
        repo_config.repo.owner,
        repo_config.repo.name,
    )

    print(f"Fetching {state} PRs from {repo_config.full_repo_name}...")
    print()

    try:
        prs = await github.list_pull_requests(state=state)

        if not prs:
            print("No PRs found.")
            return 0

        print(f"{'#':<6} {'State':<8} {'Ticket':<12} {'Title'}")
        print("-" * 80)

        ticket_pattern = rf"\b({re.escape(repo_config.jira.project_key)}-\d+)\b"

        for pr in prs:
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

            # Truncate title
            if len(title) > 45:
                title = title[:42] + "..."

            print(f"#{number:<5} {pr_state:<8} {ticket:<12} {title}")

        print()
        print(f"Total: {len(prs)} PRs")

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
    from .clients.github_client import GitHubClient
    from .clients.jira_client import JiraClient
    from .pr_tracker import PRTracker

    interval = int(args["--interval"] or 60)

    repo_config = load_config_with_fallback(args)

    if not settings.has_github_token:
        print("Error: GitHub token not configured")
        return 1

    auth_manager = AuthManager(settings)
    if not auth_manager.jira.is_authenticated():
        print("Error: Not authenticated with Jira. Run: jira-agent auth login --service=jira")
        return 1

    # Load tracked PRs
    tracker = PRTracker()
    tracked_prs = tracker.get_open_prs(repo_config.full_repo_name)

    print(f"Watching {repo_config.full_repo_name}...", flush=True)
    print(f"Polling interval: {interval} seconds", flush=True)
    print(f"Jira project: {repo_config.jira.project_key}", flush=True)
    print(f"Trigger status: \"{repo_config.agent.status}\"", flush=True)
    print(f"Done status: \"{repo_config.agent.done_status}\"", flush=True)
    if tracked_prs:
        print(f"Tracking {len(tracked_prs)} open PRs for CI/review monitoring", flush=True)
    print(flush=True)
    print("Press Ctrl+C to stop", flush=True)
    print("-" * 50, flush=True)

    # Track processed items to avoid duplicates
    processed_prs: set[int] = set()
    processed_comments: set[int] = set()  # Comment IDs already handled
    processing_tickets: set[str] = set()  # Tickets currently being processed
    ticket_pattern = rf"\b({re.escape(repo_config.jira.project_key)}-\d+)\b"

    # Initialize agent
    agent = JiraAgent(settings, repo_config)

    def timestamp() -> str:
        return datetime.now(timezone.utc).strftime("%H:%M:%S")

    try:
        while True:
            try:
                # === Poll Jira for tickets in trigger status ===
                access_token = auth_manager.jira.get_access_token()
                cloud_id = auth_manager.jira.get_cloud_id()
                jira = JiraClient(cloud_id, access_token)

                jql = f'project = {repo_config.jira.project_key} AND status = "{repo_config.agent.status}"'
                tickets = await jira.search_issues(jql, max_results=10)

                for ticket in tickets:
                    ticket_key = ticket.get("key")

                    # Skip if already processing
                    if ticket_key in processing_tickets:
                        continue

                    print(f"[{timestamp()}] Found ticket {ticket_key} in \"{repo_config.agent.status}\"", flush=True)
                    processing_tickets.add(ticket_key)

                    # Process ticket (this creates a PR)
                    print(f"[{timestamp()}] Processing {ticket_key}...", flush=True)
                    result = await agent.process_single_ticket(ticket_key)

                    if result.get("status") == "completed":
                        print(f"[{timestamp()}] ✓ {ticket_key} -> PR created: {result.get('pr_url', 'N/A')}", flush=True)
                        # Track the new PR
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

                # === Poll GitHub for our tracked PRs ===
                github = GitHubClient(
                    settings.github_token,
                    repo_config.repo.owner,
                    repo_config.repo.name,
                )

                # Refresh tracked PRs list
                tracked_prs = tracker.get_open_prs(repo_config.full_repo_name)

                for tracked in tracked_prs:
                    pr_number = tracked.pr_number

                    try:
                        pr = await github.get_pull_request(pr_number)

                        # Check if PR was merged or closed
                        if pr.get("state") == "closed":
                            if pr.get("merged_at"):
                                print(f"[{timestamp()}] PR #{pr_number} was merged", flush=True)
                                tracker.update_pr(repo_config.full_repo_name, pr_number, status="merged")

                                # Transition ticket to done
                                result = await agent.transition_ticket_to_done(tracked.ticket_key)
                                if result.get("success"):
                                    print(f"[{timestamp()}] ✓ {tracked.ticket_key} transitioned to Done", flush=True)
                                processing_tickets.discard(tracked.ticket_key)
                            else:
                                print(f"[{timestamp()}] PR #{pr_number} was closed without merge", flush=True)
                                tracker.update_pr(repo_config.full_repo_name, pr_number, status="closed")
                            continue

                        # Check CI status
                        checks = await github.get_check_runs(pr["head"]["sha"])
                        failed_checks = [c for c in checks if c.get("conclusion") == "failure"]

                        if failed_checks:
                            current_ci = tracked.ci_status
                            if current_ci != "failure":
                                print(f"[{timestamp()}] ⚠️  PR #{pr_number} has CI failures: {[c['name'] for c in failed_checks]}", flush=True)
                                tracker.update_pr(repo_config.full_repo_name, pr_number, ci_status="failure")

                                # Attempt to fix CI
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

                        # Check for new review comments
                        comments = await github.get_pr_review_comments(pr_number)
                        new_comments = [c for c in comments if c["id"] not in processed_comments]

                        if new_comments:
                            # Filter for comments that aren't from the bot/agent
                            actionable = [c for c in new_comments if not c.get("user", {}).get("login", "").endswith("[bot]")]

                            if actionable and not tracked.feedback_addressed:
                                print(f"[{timestamp()}] 💬 PR #{pr_number} has {len(actionable)} new review comment(s)", flush=True)
                                tracker.update_pr(repo_config.full_repo_name, pr_number, has_feedback=True)

                                # TODO: In future, agent can analyze and respond to comments
                                # For now, just log and mark as needing attention
                                for comment in actionable[:3]:  # Show first 3
                                    body = comment.get("body", "")[:80]
                                    user = comment.get("user", {}).get("login", "unknown")
                                    print(f"           @{user}: {body}...", flush=True)
                                    processed_comments.add(comment["id"])

                    except Exception as e:
                        print(f"[{timestamp()}] Error checking PR #{pr_number}: {e}", flush=True)

                # === Also check for merged PRs we might have missed ===
                prs = await github.list_pull_requests(state="closed")

                for pr in prs:
                    pr_number = pr.get("number")
                    merged_at = pr.get("merged_at")

                    # Skip if not merged or already processed
                    if not merged_at or pr_number in processed_prs:
                        continue

                    # Extract ticket key
                    title = pr.get("title", "")
                    branch = pr.get("head", {}).get("ref", "")
                    match = re.search(ticket_pattern, f"{title} {branch}", re.IGNORECASE)

                    if not match:
                        processed_prs.add(pr_number)
                        continue

                    ticket_key = match.group(1).upper()

                    print(f"[{timestamp()}] PR #{pr_number} merged -> transitioning {ticket_key} to \"{repo_config.agent.done_status}\"", flush=True)

                    # Transition ticket to done
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

            # Wait for next poll
            await asyncio.sleep(interval)

    except KeyboardInterrupt:
        print("\nStopping watch...", flush=True)
        await agent.close()
        return 0


async def handle_process(args: dict, settings) -> int:
    """Process tickets from a Jira board."""
    from .agent import JiraAgent

    status_filter = args["--status"]
    limit = int(args["--limit"])
    dry_run = args["--dry-run"]

    repo_config = load_config_with_fallback(args)

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


async def handle_process_ticket(args: dict, settings) -> int:
    """Process a single ticket."""
    import re

    import questionary
    from questionary import Style

    from .agent import JiraAgent
    from .clients.github_client import GitHubClient
    from .pr_tracker import PRTracker

    ticket_key = args["<ticket_key>"]
    dry_run = args["--dry-run"]

    repo_config = load_config_with_fallback(args)

    agent = JiraAgent(settings, repo_config, dry_run=dry_run)
    result = await agent.process_single_ticket(ticket_key)

    status_icon = "✓" if result["status"] == "completed" else "○" if result["status"] == "skipped" else "✗"
    print(f"{status_icon} {result['ticket']}: {result['status']}")
    if result.get("pr_url"):
        print(f"  PR: {result['pr_url']}")
    if result.get("error"):
        print(f"  Error: {result['error']}")

    # Post-PR flow: reviewer selection and watch offer
    if result.get("status") == "completed" and result.get("pr_url") and not dry_run:
        pr_url = result["pr_url"]

        # Extract PR number from URL
        pr_match = re.search(r"/pull/(\d+)", pr_url)
        if pr_match:
            pr_number = int(pr_match.group(1))

            custom_style = Style([
                ("qmark", "fg:cyan bold"),
                ("question", "fg:white bold"),
                ("answer", "fg:green bold"),
                ("pointer", "fg:cyan bold"),
                ("highlighted", "fg:cyan bold"),
            ])

            # Track the PR
            tracker = PRTracker()
            tracker.add_pr(
                pr_number=pr_number,
                pr_url=pr_url,
                repo=repo_config.full_repo_name,
                ticket_key=ticket_key,
                branch=f"feat/{ticket_key}",  # Best guess, could extract from PR
            )

            print()
            print("─" * 50)
            print("Post-PR Setup")
            print("─" * 50)

            # Reviewer selection
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

                try:
                    print("Fetching suggested reviewers...")
                    suggested = await github.get_suggested_reviewers(pr_number)

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
                            await github.request_reviewers(pr_number, selected)
                            print(f"✓ Added reviewers: {', '.join(selected)}")
                    else:
                        print("No suggested reviewers found.")
                        manual = questionary.text(
                            "Enter reviewer usernames (comma-separated, or leave empty):",
                            style=custom_style,
                        ).ask()
                        if manual:
                            reviewers = [r.strip().lstrip("@") for r in manual.split(",")]
                            await github.request_reviewers(pr_number, reviewers)
                            print(f"✓ Added reviewers: {', '.join(reviewers)}")

                except Exception as e:
                    print(f"Could not add reviewers: {e}")

                finally:
                    await github.close()

            # Offer to watch
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

                # Call watch with this specific PR focus
                watch_args = {
                    "--config": args.get("--config"),
                    "--interval": "30",  # Check every 30 seconds for active watching
                }
                return await handle_watch(watch_args, settings)
            else:
                print()
                print("You can watch later with: jira-agent watch")
                print("The agent will find your PRs and respond to any that need attention.")

    return 0 if result["status"] in ("completed", "skipped") else 1


async def handle_check_pr(args: dict, settings) -> int:
    """Check PR status."""
    from .agent import JiraAgent

    pr_number = int(args["<pr_number>"])

    repo_config = load_config_with_fallback(args)

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

    repo_config = load_config_with_fallback(args)

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


async def handle_init(args: dict, settings) -> int:
    """Interactive setup for jira-agent in a repository."""
    import questionary
    from questionary import Style

    from .auth import AuthManager
    from .clients.jira_client import JiraClient
    from .repo_config.loader import REPO_CONFIG_FILENAME

    custom_style = Style([
        ("qmark", "fg:cyan bold"),
        ("question", "fg:white bold"),
        ("answer", "fg:green bold"),
        ("pointer", "fg:cyan bold"),
        ("highlighted", "fg:cyan bold"),
    ])

    print("=" * 50)
    print("  Jira Agent - Repository Setup")
    print("=" * 50)
    print()

    # ========================================
    # Step 1: Check required credentials
    # ========================================
    print("Checking required credentials...")
    print("-" * 30)

    auth_manager = AuthManager(settings)
    all_credentials_ok = True

    # Check Anthropic API key
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
                # Store the key securely
                from .auth.token_store import TokenStore

                store = TokenStore()
                store.save("anthropic", {"api_key": api_key.strip()})
                print("✓ Anthropic API key: saved securely")

                # Refresh settings to pick up the new key
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

    # Check GitHub token
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
        print("     export JIRA_AGENT_GITHUB_TOKEN='your-token-here'")
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
                # Refresh settings to pick up new token
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

    # Check Jira OAuth
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

    # ========================================
    # Step 2: Repository configuration
    # ========================================
    print("Repository Configuration")
    print("-" * 30)

    # Check if config already exists
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

    # Detect repo from git remote
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

    # Get default branch
    default_branch = questionary.text(
        "Default branch:",
        default="main",
        style=custom_style,
    ).ask()

    # ========================================
    # Step 3: Jira project configuration
    # ========================================
    print("Jira Project Configuration")
    print("-" * 30)

    # Get Jira project key
    project_key = questionary.text(
        "Jira project key (e.g., AENG):",
        style=custom_style,
    ).ask()

    if not project_key:
        print("Error: Project key is required")
        return 1

    project_key = project_key.upper()

    # Get board ID - optionally list boards
    board_id = None
    if jira_authenticated:
        list_boards = questionary.confirm(
            "List available Jira boards?",
            default=True,
            style=custom_style,
        ).ask()

        if list_boards:
            try:
                access_token = auth_manager.jira.get_access_token()
                cloud_id = auth_manager.jira.get_cloud_id()
                jira = JiraClient(cloud_id, access_token)

                boards = await jira.get_boards(project_key=project_key)
                await jira.close()

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

    # ========================================
    # Step 4: Agent trigger configuration
    # ========================================
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

    # ========================================
    # Step 5: Auto-detect repository features
    # ========================================
    print("Repository Features (Auto-detected)")
    print("-" * 30)

    # Detect dbt projects
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

    # Detect CI system
    ci_system = "github_actions"
    if (cwd / ".circleci").exists():
        ci_system = "circleci"
    elif (cwd / "Jenkinsfile").exists():
        ci_system = "jenkins"

    print(f"Detected CI system: {ci_system}")

    # Build config
    output_path = args.get("--output") or REPO_CONFIG_FILENAME

    config_content = f"""# Jira Agent configuration for {owner}/{name}
# Generated by: jira-agent init

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

    # Write config
    output = Path(output_path)
    output.write_text(config_content)

    print()
    print("=" * 50)
    print(f"✓ Config created: {output_path}")
    print("=" * 50)
    print()
    print("Next steps:")
    print("  1. Review and customize the config if needed")
    print("  2. Run 'jira-agent auth login' if not authenticated")
    print("  3. Run 'jira-agent health' to test connections")
    print("  4. Run 'jira-agent list-tickets' to see available tickets")
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

    # Load repo config if provided
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

    # Test Anthropic API
    print("Anthropic API:")
    if settings.has_anthropic_key:
        try:
            client = Anthropic(api_key=settings.anthropic_api_key)
            # Make a minimal API call to verify the key
            response = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=10,
                messages=[{"role": "user", "content": "Say OK"}],
            )
            print(f"  Status: OK")
            print(f"  Model configured: {settings.claude_model}")
        except Exception as e:
            print(f"  Status: FAILED - {e}")
            all_ok = False
    else:
        print("  Status: NOT CONFIGURED (ANTHROPIC_API_KEY not set)")
        all_ok = False

    print()

    # Test Jira connection
    print("Jira:")
    auth_manager = AuthManager(settings)
    jira_token = auth_manager.jira.get_access_token() if auth_manager.jira.is_authenticated() else None
    if jira_token:
        try:
            async with httpx.AsyncClient() as client:
                # Get accessible resources to find cloud ID
                response = await client.get(
                    "https://api.atlassian.com/oauth/token/accessible-resources",
                    headers={"Authorization": f"Bearer {jira_token}"},
                )
                if response.status_code == 200:
                    resources = response.json()
                    if resources:
                        print(f"  Status: OK")
                        print(f"  Accessible sites: {len(resources)}")
                        for r in resources[:3]:  # Show up to 3
                            print(f"    - {r.get('name', 'Unknown')} ({r.get('url', '')})")

                        # If we have a config, test access to the specific project
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
                        print(f"  Status: WARNING - No accessible Jira sites")
                else:
                    print(f"  Status: FAILED - Token invalid or expired (status {response.status_code})")
                    all_ok = False
        except Exception as e:
            print(f"  Status: FAILED - {e}")
            all_ok = False
    else:
        print("  Status: NOT AUTHENTICATED")
        print("  Run: jira-agent auth login --service=jira")
        all_ok = False

    print()

    # Test GitHub connection
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
                    print(f"  Status: OK")
                    print(f"  User: {user.get('login', 'Unknown')}")

                    # If we have a config, test access to the repo
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
        print("  Either run 'gh auth login' or set JIRA_AGENT_GITHUB_TOKEN")
        all_ok = False

    print()

    # Test Databricks connection (optional)
    print("Databricks:")
    if settings.has_databricks:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{settings.databricks_host}/api/2.0/clusters/list",
                    headers={"Authorization": f"Bearer {settings.databricks_token}"},
                )
                if response.status_code == 200:
                    print(f"  Status: OK")
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
    print("=" * 50)
    if all_ok:
        print("All required services are healthy!")
        return 0
    else:
        print("Some services have issues. Please fix them before using the agent.")
        return 1


def handle_learn(args: dict, settings) -> int:
    """Handle learn commands."""
    if args["status"]:
        return handle_learn_status(args, settings)
    elif args["publish"]:
        return handle_learn_publish(args, settings)
    elif args["list"]:
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

    # Group by repo
    by_repo: dict[str, list] = {}
    for learning in learnings:
        by_repo.setdefault(learning.repo, []).append(learning)

    for repo, repo_learnings in sorted(by_repo.items()):
        print(f"\n{repo}:")
        for learning in repo_learnings:
            print(f"  - [{learning.category.value}] {learning.title}")
            print(f"    Ticket: {learning.ticket}, Subcategory: {learning.subcategory}")

    print()
    print(f"Run 'jira-agent learn publish' to create a PR with these learnings.")
    return 0


def handle_learn_publish(args: dict, settings) -> int:
    """Publish learnings to jira-agent repo."""
    from .learning import LearningPublisher

    dry_run = args.get("--dry-run", False)
    jira_agent_repo = args.get("--jira-agent-repo") or getattr(
        settings, "jira_agent_repo", "djayatillake/jira-agent"
    )

    if not settings.has_github_token:
        print("Error: GitHub token not configured")
        print("Either run 'gh auth login' or set JIRA_AGENT_GITHUB_TOKEN")
        return 1

    print(f"Publishing learnings to {jira_agent_repo}...")
    if dry_run:
        print("(dry-run mode)")
    print()

    publisher = LearningPublisher(
        github_token=settings.github_token,
        jira_agent_repo=jira_agent_repo,
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
    from pathlib import Path

    from .learning import LearningCategory
    from .learning.publisher import CATEGORY_DIRS, KNOWLEDGE_BASE_DIR
    from .learning.storage import LearningStorage

    category_filter = args.get("--category")

    # Try to find knowledge base in current directory or jira-agent clone
    kb_paths = [
        Path.cwd() / KNOWLEDGE_BASE_DIR,
        settings.workspace_dir / "djayatillake-jira-agent" / KNOWLEDGE_BASE_DIR,
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

        for md_file in sorted(md_files)[:10]:  # Show first 10
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
    if args["check"]:
        return handle_env_check(args, settings, auto_install=False)
    elif args["setup"]:
        return handle_env_check(args, settings, auto_install=True)
    return 1


def handle_env_check(args: dict, settings, auto_install: bool = False) -> int:
    """Check environment for a repository."""
    from pathlib import Path

    from .environment import EnvironmentChecker, PackageInstaller
    from .environment.requirements import RequirementsParser
    from .tools.git_tools import GitTools

    repo_path_str = args.get("--repo-path")
    config_path = args.get("--config")

    # Determine repo path
    if repo_path_str:
        repo_path = Path(repo_path_str)
        if not repo_path.exists():
            print(f"Error: Repository path does not exist: {repo_path}")
            return 1
        repo_config = None
    elif config_path:
        # Clone the repo first
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
        # Use current directory and try to auto-detect config
        repo_path = Path.cwd()
        repo_config = load_config_with_fallback(args, required=False)

    # Check system tools
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

    # Check repository requirements
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

    # Auto-install if requested
    if auto_install:
        print("Installing Dependencies")
        print("=" * 50)

        # Install missing system tools
        if report.missing_required:
            installer = PackageInstaller(repo_path, auto_confirm=True)
            for tool in report.missing_required:
                print(f"Installing {tool}...")
                result = installer.install_system_tool(tool)
                if result.success:
                    print(f"  ✓ {tool} installed")
                else:
                    print(f"  ✗ {tool} failed: {result.error}")

        # Install repo dependencies
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

    # Final status
    if report.missing_required:
        print(f"✗ Missing required tools: {', '.join(report.missing_required)}")
        return 1

    missing_python, missing_node = parser.get_missing_packages()
    if (missing_python or missing_node) and not auto_install:
        print("✗ Missing packages. Run 'jira-agent env setup' to install.")
        return 1

    print("✓ Environment is ready!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
