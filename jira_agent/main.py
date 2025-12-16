"""Jira Ticket Automation Agent CLI.

Usage:
    jira-agent process [--config=<path>] [--repo=<repo>] [--status=<status>] [--limit=<n>] [--dry-run]
    jira-agent process-ticket <ticket_key> [--config=<path>] [--repo=<repo>] [--dry-run]
    jira-agent check-pr <pr_number> [--config=<path>] [--repo=<repo>]
    jira-agent fix-ci <pr_number> [--config=<path>] [--repo=<repo>]
    jira-agent serve [--port=<port>] [--host=<host>] [--config-dir=<dir>]
    jira-agent init-config <repo> [--output=<path>]
    jira-agent auth login [--service=<service>]
    jira-agent auth status
    jira-agent auth logout [--service=<service>]
    jira-agent config show
    jira-agent config validate <config_path>
    jira-agent --help
    jira-agent --version

Commands:
    process         Process tickets from a Jira board
    process-ticket  Process a specific ticket by key
    check-pr        Check PR status and pending feedback
    fix-ci          Attempt to fix CI failures on a PR
    serve           Start webhook server for Jira/GitHub events
    init-config     Generate a config file for a new repository
    auth            Manage OAuth authentication
    config          Show or validate configuration

Options:
    -h --help                Show this help message
    --version                Show version
    --config=<path>          Path to repo config file
    --repo=<repo>            Repository in owner/name format (e.g., acme/data)
    --status=<status>        Filter tickets by Jira status (e.g., "To Do", "Ready for Dev")
    --limit=<n>              Maximum tickets to process [default: 10]
    --dry-run                Preview actions without making changes
    --port=<port>            Webhook server port [default: 8080]
    --host=<host>            Webhook server host [default: 0.0.0.0]
    --config-dir=<dir>       Directory containing repo config files [default: ./configs]
    --output=<path>          Output path for generated config
    --service=<service>      Service to authenticate: jira, github, databricks, or all [default: all]

Environment Variables:
    ANTHROPIC_API_KEY           Required for Claude Agent SDK
    JIRA_AGENT_JIRA_OAUTH_CLIENT_ID      Jira OAuth app client ID
    JIRA_AGENT_JIRA_OAUTH_CLIENT_SECRET  Jira OAuth app client secret
    JIRA_AGENT_GITHUB_TOKEN              GitHub personal access token
    JIRA_AGENT_DATABRICKS_HOST           Databricks workspace URL
    JIRA_AGENT_DATABRICKS_TOKEN          Databricks personal access token
    JIRA_AGENT_WEBHOOK_SECRET            Secret for webhook validation

Examples:
    # Process all "Ready for Dev" tickets for acme/data
    jira-agent process --config configs/acme-data.yaml --status="Ready for Dev" --limit=5

    # Process a specific ticket
    jira-agent process-ticket AENG-1234 --config configs/acme-data.yaml

    # Start webhook server
    jira-agent serve --port 8080 --config-dir ./configs

    # Generate config for a new repo
    jira-agent init-config acme/new-repo --output configs/acme-new-repo.yaml

    # Authenticate with all services
    jira-agent auth login

    # Check authentication status
    jira-agent auth status
"""

import asyncio
import logging
import sys
from pathlib import Path

from docopt import docopt

from .config import get_settings
from .utils.logger import setup_logging

__version__ = "0.1.0"


def main() -> int:
    """Main entry point for the CLI."""
    args = docopt(__doc__, version=f"jira-agent {__version__}")

    settings = get_settings()
    setup_logging(settings.log_level)
    logger = logging.getLogger(__name__)

    try:
        if args["auth"]:
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
        elif args["init-config"]:
            return handle_init_config(args, settings)
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


async def handle_process(args: dict, settings) -> int:
    """Process tickets from a Jira board."""
    from .agent import JiraAgent
    from .repo_config.loader import ConfigLoader

    config_path = args["--config"]
    repo_name = args["--repo"]
    status_filter = args["--status"]
    limit = int(args["--limit"])
    dry_run = args["--dry-run"]

    if not config_path and not repo_name:
        print("Error: Either --config or --repo must be specified")
        return 1

    loader = ConfigLoader()
    if config_path:
        repo_config = loader.load_from_file(config_path)
    else:
        repo_config = loader.load_for_repo(repo_name)

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
    from .agent import JiraAgent
    from .repo_config.loader import ConfigLoader

    ticket_key = args["<ticket_key>"]
    config_path = args["--config"]
    repo_name = args["--repo"]
    dry_run = args["--dry-run"]

    if not config_path and not repo_name:
        print("Error: Either --config or --repo must be specified")
        return 1

    loader = ConfigLoader()
    if config_path:
        repo_config = loader.load_from_file(config_path)
    else:
        repo_config = loader.load_for_repo(repo_name)

    agent = JiraAgent(settings, repo_config, dry_run=dry_run)
    result = await agent.process_single_ticket(ticket_key)

    status_icon = "✓" if result["status"] == "completed" else "○" if result["status"] == "skipped" else "✗"
    print(f"{status_icon} {result['ticket']}: {result['status']}")
    if result.get("pr_url"):
        print(f"  PR: {result['pr_url']}")
    if result.get("error"):
        print(f"  Error: {result['error']}")

    return 0 if result["status"] in ("completed", "skipped") else 1


async def handle_check_pr(args: dict, settings) -> int:
    """Check PR status."""
    from .agent import JiraAgent
    from .repo_config.loader import ConfigLoader

    pr_number = int(args["<pr_number>"])
    config_path = args["--config"]
    repo_name = args["--repo"]

    if not config_path and not repo_name:
        print("Error: Either --config or --repo must be specified")
        return 1

    loader = ConfigLoader()
    if config_path:
        repo_config = loader.load_from_file(config_path)
    else:
        repo_config = loader.load_for_repo(repo_name)

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
    from .repo_config.loader import ConfigLoader

    pr_number = int(args["<pr_number>"])
    config_path = args["--config"]
    repo_name = args["--repo"]

    if not config_path and not repo_name:
        print("Error: Either --config or --repo must be specified")
        return 1

    loader = ConfigLoader()
    if config_path:
        repo_config = loader.load_from_file(config_path)
    else:
        repo_config = loader.load_for_repo(repo_name)

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


def handle_init_config(args: dict, settings) -> int:
    """Generate a config file for a new repository."""
    repo = args["<repo>"]
    output_path = args["--output"]

    if "/" not in repo:
        print("Error: Repository must be in owner/name format (e.g., acme/data)")
        return 1

    owner, name = repo.split("/", 1)

    if not output_path:
        output_path = f"configs/{owner}-{name}.yaml"

    template = f"""# Configuration for {repo}
repo:
  owner: "{owner}"
  name: "{name}"
  default_branch: "main"
  pr_target_branch: "main"

jira:
  base_url: "https://your-org.atlassian.net"
  project_key: "PROJ"
  board_id: null  # Set your board ID

branching:
  pattern: "{{type}}/{{ticket_key}}-{{description}}"
  types:
    feature: "feat"
    bugfix: "fix"
    refactor: "refactor"

pull_request:
  title_pattern: "{{type}}({{scope}}): {{description}} ({{ticket_key}})"
  template_path: ".github/PULL_REQUEST_TEMPLATE.md"
  contributing_path: ".github/CONTRIBUTING.md"

commits:
  style: "conventional"
  scope_required: false
  ticket_in_message: true

skip:
  comment_phrase: "[AGENT-SKIP]"
  labels:
    - "no-automation"
    - "manual-only"

dbt:
  enabled: false
  projects: []

databricks:
  enabled: false

ci:
  system: "github_actions"
  auto_fix:
    - "pre-commit"
"""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(template)

    print(f"Generated config at: {output_path}")
    print("Please edit the file to customize for your repository.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
