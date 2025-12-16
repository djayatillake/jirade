# Jira Agent

An autonomous agent that processes Jira tickets and implements code changes using Claude Opus 4.5.

## Features

- **Autonomous Code Changes**: Analyzes Jira tickets and implements required changes
- **Multi-Repository Support**: Configure the agent to work with any GitHub repository
- **dbt Integration**: Special support for dbt model changes with dependency tracking
- **PR Automation**: Creates pull requests following repository conventions
- **CI Response**: Automatically fixes CI failures when possible
- **Multiple Triggers**: CLI, Jira webhooks (assignment/mention), GitHub webhooks (reviews/CI)

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/jira-agent
cd jira-agent

# Install dependencies
poetry install

# Authenticate with services
jira-agent auth login
```

### Configuration

1. Set up environment variables:

```bash
# Required
export ANTHROPIC_API_KEY="your-api-key"

# Jira OAuth (create app at https://developer.atlassian.com/console/myapps/)
export JIRA_AGENT_JIRA_OAUTH_CLIENT_ID="your-client-id"
export JIRA_AGENT_JIRA_OAUTH_CLIENT_SECRET="your-client-secret"

# GitHub (create token at https://github.com/settings/tokens)
export JIRA_AGENT_GITHUB_TOKEN="your-github-token"

# Optional: Databricks
export JIRA_AGENT_DATABRICKS_HOST="https://your-workspace.databricks.com"
export JIRA_AGENT_DATABRICKS_TOKEN="your-databricks-token"
```

2. Create a repository configuration:

```bash
# Generate config template
jira-agent init-config your-org/your-repo --output configs/your-repo.yaml

# Edit the config file to match your repository conventions
```

### Usage

#### Process Tickets from CLI

```bash
# Process tickets with a specific status
jira-agent process --config configs/your-repo.yaml --status="Ready for Dev" --limit=5

# Process a specific ticket
jira-agent process-ticket PROJ-1234 --config configs/your-repo.yaml

# Dry run (preview without making changes)
jira-agent process-ticket PROJ-1234 --config configs/your-repo.yaml --dry-run
```

#### Start Webhook Server

```bash
# For local development
jira-agent serve --port 8080 --config-dir ./configs

# Use ngrok for external access
ngrok http 8080
```

#### Check/Fix PRs

```bash
# Check PR status
jira-agent check-pr 123 --repo your-org/your-repo

# Attempt to fix CI failures
jira-agent fix-ci 123 --repo your-org/your-repo
```

## Repository Configuration

Each target repository needs a configuration file. See `configs/example.yaml` for a complete template.

### Key Settings

```yaml
repo:
  owner: "your-org"
  name: "your-repo"
  default_branch: "main"        # Branch to create features from
  pr_target_branch: "develop"   # Branch PRs should target

jira:
  base_url: "https://your-org.atlassian.net"
  project_key: "PROJ"
  board_id: 123

branching:
  pattern: "{type}/{ticket_key}-{description}"

pull_request:
  title_pattern: "{type}({scope}): {description} ({ticket_key})"

skip:
  comment_phrase: "[AGENT-SKIP]"
  labels: ["no-automation"]
```

### dbt Configuration

For repositories with dbt projects:

```yaml
dbt:
  enabled: true
  projects:
    - path: "dbt"
      manifest_path: "target/manifest.json"
```

## Triggers

### CLI Trigger
Run manually or via cron to process tickets.

### Jira Webhook
Configure a webhook in Jira to trigger on:
- Issue assignment to the agent user
- @mention in comments

### GitHub Webhook
Configure a webhook in GitHub to trigger on:
- PR review with requested changes
- Check run failures

## Architecture

```
jira_agent/
├── main.py              # CLI entry point
├── agent.py             # Core Claude agent orchestration
├── config.py            # Global settings
├── repo_config/         # Repository configuration
├── triggers/            # Webhook handlers
├── auth/                # OAuth authentication
├── tools/               # Git, dbt tools
└── clients/             # Jira, GitHub API clients
```

## Development

### Running Tests

```bash
poetry run pytest
```

### Code Style

```bash
poetry run black .
poetry run isort .
poetry run mypy .
```

## License

MIT
