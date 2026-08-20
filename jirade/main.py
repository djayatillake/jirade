"""Jirade CLI.

Agentic data-engineering tooling driven from Claude Code: dbt CI on Databricks,
GitHub PR tooling, and environment/learning utilities. Jira/Confluence
interaction moved to the Atlassian Rovo MCP connector in v0.10.0.
"""

import asyncio
import logging
import sys
from importlib.metadata import version
from pathlib import Path
from typing import Optional

import typer
from typing_extensions import Annotated

from .config import get_settings
from .repo_config.loader import ConfigLoader, find_repo_config, get_git_remote_info
from .utils.logger import setup_logging

__version__ = version("jirade")

# Main app
app = typer.Typer(
    name="jirade",
    help="Jirade - agentic data engineering tools (dbt CI, GitHub PRs) for Claude Code",
    no_args_is_help=True,
    rich_markup_mode="rich",
)

# Subcommand groups
auth_app = typer.Typer(help="Manage OAuth authentication")
config_app = typer.Typer(help="Show or validate configuration")
learn_app = typer.Typer(help="Manage agent learnings")
env_app = typer.Typer(help="Check and setup environment")
zoom_app = typer.Typer(help="Zoom meeting bot (join meetings and respond to questions)")

app.add_typer(auth_app, name="auth")
app.add_typer(config_app, name="config")
app.add_typer(learn_app, name="learn")
app.add_typer(env_app, name="env")
app.add_typer(zoom_app, name="zoom")


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
    """Jirade CLI."""
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
    service: Annotated[str, typer.Option("--service", "-s", help="Service: github, databricks, or all")] = "all",
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
    service: Annotated[str, typer.Option("--service", "-s", help="Service: github, databricks, or all")] = "all",
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
# Zoom Subcommands
# ============================================================


@zoom_app.command("serve")
def zoom_serve(
    host: Annotated[str, typer.Option("--host", help="Webhook server host")] = "0.0.0.0",
    port: Annotated[int, typer.Option("--port", "-p", help="Webhook server port")] = 8090,
):
    """Start the Zoom bot webhook server."""
    settings = get_settings()
    setup_logging(settings.log_level)

    from .zoom_bot.config import get_zoom_settings
    from .zoom_bot.server import run_server

    zoom_settings = get_zoom_settings()

    if not zoom_settings.has_recall_api:
        print("Error: Recall.ai API key not configured")
        print("Set JIRADE_ZOOM_RECALL_API_KEY in your environment or .env file")
        raise typer.Exit(1)

    if not settings.has_anthropic_key:
        print("Error: Anthropic API key not configured")
        raise typer.Exit(1)

    print(f"Starting jirade Zoom bot webhook server on {host}:{port}")
    print(f"Wake words: {', '.join(zoom_settings.wake_words)}")
    print(f"Response mode: {zoom_settings.response_mode}")
    if zoom_settings.webhook_url:
        print(f"Webhook URL: {zoom_settings.webhook_url} (static)")
    elif zoom_settings.auto_tunnel:
        print(f"Webhook URL: auto-tunnel via {zoom_settings.tunnel_host} (URL assigned on startup)")
    else:
        print("WARNING: No webhook URL and auto-tunnel disabled - Recall.ai won't know where to send events")
    print(f"TTS: {'enabled' if zoom_settings.has_tts else 'disabled (install ffmpeg to enable)'}")
    print()
    print("Endpoints:")
    print(f"  POST /webhook/recall  - Recall.ai webhook receiver")
    print(f"  POST /api/join        - Join a meeting ({{\"meeting_url\": \"...\"}}")
    print(f"  POST /api/leave/{{id}} - Leave a meeting")
    print(f"  GET  /api/status/{{id}} - Bot status")
    print(f"  GET  /api/bots        - List bots")
    print(f"  GET  /api/tunnel      - Tunnel status")
    print(f"  GET  /health          - Health check")
    print()
    import tempfile
    print(f"Query notifications: tail -f {tempfile.gettempdir()}/jirade-zoom-queries.jsonl")
    print()

    run_server(host=host, port=port)


@zoom_app.command("join")
def zoom_join(
    meeting_url: Annotated[str, typer.Argument(help="Zoom meeting URL")],
    server_url: Annotated[str, typer.Option("--server", "-s", help="Zoom bot server URL")] = "http://localhost:8090",
):
    """Make the bot join a Zoom meeting (requires the server to be running)."""
    import httpx

    try:
        response = httpx.post(
            f"{server_url}/api/join",
            json={"meeting_url": meeting_url},
            timeout=30.0,
        )
        response.raise_for_status()
        data = response.json()
        print(f"Bot joining meeting: {meeting_url}")
        print(f"Bot ID: {data.get('bot_id')}")
        print()
        print(f"Leave with: jirade zoom leave {data.get('bot_id')}")
    except httpx.ConnectError:
        print(f"Error: Cannot connect to server at {server_url}")
        print("Make sure the server is running: jirade zoom serve")
        raise typer.Exit(1)
    except Exception as e:
        print(f"Error: {e}")
        raise typer.Exit(1)


@zoom_app.command("leave")
def zoom_leave(
    bot_id: Annotated[str, typer.Argument(help="Bot ID to remove from meeting")],
    server_url: Annotated[str, typer.Option("--server", "-s", help="Zoom bot server URL")] = "http://localhost:8090",
):
    """Make a bot leave its meeting."""
    import httpx

    try:
        response = httpx.post(
            f"{server_url}/api/leave/{bot_id}",
            timeout=30.0,
        )
        response.raise_for_status()
        print(f"Bot {bot_id} leaving meeting")
    except httpx.ConnectError:
        print(f"Error: Cannot connect to server at {server_url}")
        raise typer.Exit(1)
    except Exception as e:
        print(f"Error: {e}")
        raise typer.Exit(1)


@zoom_app.command("status")
def zoom_status(
    bot_id: Annotated[Optional[str], typer.Argument(help="Bot ID (omit to list all bots)")] = None,
    server_url: Annotated[str, typer.Option("--server", "-s", help="Zoom bot server URL")] = "http://localhost:8090",
):
    """Check bot status or list all bots."""
    import httpx

    try:
        if bot_id:
            response = httpx.get(f"{server_url}/api/status/{bot_id}", timeout=30.0)
            response.raise_for_status()
            data = response.json()
            status_code = data.get("status_changes", [{}])[-1].get("code", "unknown") if data.get("status_changes") else "unknown"
            meeting_url = data.get("meeting_url", {}).get("meeting_url", "N/A") if isinstance(data.get("meeting_url"), dict) else data.get("meeting_url", "N/A")
            print(f"Bot ID:  {bot_id}")
            print(f"Status:  {status_code}")
            print(f"Meeting: {meeting_url}")
        else:
            response = httpx.get(f"{server_url}/api/bots", timeout=30.0)
            response.raise_for_status()
            bots = response.json().get("bots", [])
            if not bots:
                print("No active bots")
                return

            print(f"{'Bot ID':<40} {'Status':<15} {'Meeting'}")
            print("-" * 80)
            for bot in bots:
                bot_id = bot.get("id", "")
                status_changes = bot.get("status_changes", [])
                status = status_changes[-1].get("code", "unknown") if status_changes else "unknown"
                meeting = bot.get("meeting_url", "N/A")
                if isinstance(meeting, dict):
                    meeting = meeting.get("meeting_url", "N/A")
                print(f"{bot_id:<40} {status:<15} {meeting}")

    except httpx.ConnectError:
        print(f"Error: Cannot connect to server at {server_url}")
        print("Make sure the server is running: jirade zoom serve")
        raise typer.Exit(1)
    except Exception as e:
        print(f"Error: {e}")
        raise typer.Exit(1)


@zoom_app.command("listen")
def zoom_listen(
    interval: Annotated[float, typer.Option("--interval", "-i", help="Poll interval in seconds")] = 5.0,
    server_url: Annotated[str, typer.Option("--server", "-s", help="Zoom bot server URL")] = "http://localhost:8090",
):
    """Poll the server for pending queries and print them as JSONL.

    Watches for wake-word-triggered queries in relay mode and outputs each
    new query as a JSON line to stdout. Useful for piping into other tools
    or for Claude Code to monitor and respond to meeting questions.

    Example: jirade zoom listen | while read line; do echo "$line"; done
    """
    import json
    import time

    import httpx

    seen_ids: set[int] = set()
    try:
        while True:
            try:
                response = httpx.get(f"{server_url}/api/pending", timeout=10.0)
                response.raise_for_status()
                queries = response.json().get("queries", [])
                for q in queries:
                    qid = q.get("id", 0)
                    if qid not in seen_ids:
                        seen_ids.add(qid)
                        print(json.dumps(q), flush=True)
            except httpx.ConnectError:
                pass  # Server not ready yet, keep polling
            except Exception as e:
                print(json.dumps({"error": str(e)}), flush=True)
            time.sleep(interval)
    except KeyboardInterrupt:
        pass


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
        print(f"Anthropic API Key: {'*' * 8 if settings.has_anthropic_key else 'Not set (optional — only for zoom bot + advisor auto-suggestions)'}")
        print("Atlassian: via Rovo MCP connector (no jirade config needed)")
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

    # Anthropic API key is optional since v0.11.0 — Claude Code is the harness.
    # It is only used by the zoom bot and the advisors' auto-suggestions.
    if settings.has_anthropic_key:
        print("✓ Anthropic API key: configured (optional)")
    else:
        print("- Anthropic API key: not set (fine — only needed for zoom bot / advisor auto-suggestions)")

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
        "Jira project key (e.g., PROJ):",
        style=custom_style,
    ).ask()

    if not project_key:
        print("Error: Project key is required")
        return 1

    project_key = project_key.upper()
    board_id = None

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
    print("  4. For Jira/Confluence, authenticate the Atlassian Rovo MCP connector in Claude Code (/mcp)")
    print()

    return 0


async def handle_health(args: dict, settings) -> int:
    """Test all service connections."""
    import httpx

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

    print("Anthropic API (optional — zoom bot + advisor auto-suggestions only):")
    if settings.has_anthropic_key:
        try:
            from anthropic import Anthropic

            client = Anthropic(api_key=settings.anthropic_api_key)
            client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=10,
                messages=[{"role": "user", "content": "Say OK"}],
            )
            print("  Status: OK")
            print(f"  Model configured: {settings.claude_model}")
        except Exception as e:
            print(f"  Status: FAILED - {e}")
    else:
        print("  Status: not set (fine — Claude Code is the harness)")

    print()

    print("Atlassian: handled by the Rovo MCP connector (authenticate via /mcp in Claude Code)")
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
            import logging as _logging
            _logging.getLogger("databricks").setLevel(_logging.WARNING)

            from .clients.databricks_client import DatabricksMetadataClient

            with DatabricksMetadataClient(
                host=settings.databricks_host,
                http_path=settings.databricks_http_path,
                auth_type=settings.databricks_auth_type,
                token=settings.databricks_token if settings.databricks_auth_type == "token" else None,
                catalog=settings.databricks_catalog or None,
            ) as db_client:
                db_client.execute_metadata_query("SHOW SCHEMAS")
                print("  Status: OK")
                print(f"  Host: {settings.databricks_host}")
                print(f"  Auth: {settings.databricks_auth_type}")
                if settings.databricks_ci_catalog:
                    print(f"  CI catalog: {settings.databricks_ci_catalog}")
                    print(f"  Schema prefix: {settings.dbt_ci_schema_prefix}")
                else:
                    print("  WARNING: JIRADE_DATABRICKS_CI_CATALOG not set")
                    print("  Set it to your dev catalog (e.g., development_yourname_metadata)")
        except Exception as e:
            print(f"  Status: FAILED - {e}")
            all_ok = False
    else:
        print("  Status: NOT CONFIGURED")
        print("  Set JIRADE_DATABRICKS_HOST and JIRADE_DATABRICKS_HTTP_PATH to enable")

    print()

    print("Zoom Bot (Recall.ai):")
    try:
        from .zoom_bot.config import get_zoom_settings

        zoom_settings = get_zoom_settings()
        if zoom_settings.has_recall_api:
            try:
                import httpx as _httpx

                resp = _httpx.get(
                    f"{zoom_settings.recall_api_url}/bot",
                    headers={"Authorization": f"Token {zoom_settings.recall_api_key}"},
                    params={"limit": 1},
                    timeout=10.0,
                )
                if resp.status_code == 200:
                    print("  Status: OK")
                    print(f"  API: {zoom_settings.recall_api_url}")
                    print(f"  Response mode: {zoom_settings.response_mode}")
                    if zoom_settings.webhook_url:
                        print(f"  Webhook URL: {zoom_settings.webhook_url}")
                    else:
                        print("  WARNING: No webhook URL configured (JIRADE_ZOOM_WEBHOOK_URL)")
                        print("  Set up a tunnel (e.g., localhost.run) and configure the URL")
                else:
                    print(f"  Status: FAILED - API returned {resp.status_code}")
                    print("  Check JIRADE_ZOOM_RECALL_API_KEY is valid")
            except Exception as e:
                print(f"  Status: FAILED - {e}")
        else:
            print("  Status: NOT CONFIGURED")
            print("  Set JIRADE_ZOOM_RECALL_API_KEY to enable the Zoom meeting bot")
            print("  Sign up at https://recall.ai to get an API key")
    except Exception:
        print("  Status: NOT CONFIGURED")

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
