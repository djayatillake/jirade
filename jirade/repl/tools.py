"""Tool definitions for the REPL agent."""

from typing import Any

# Tool definitions as JSON schemas for Claude
TOOLS: list[dict[str, Any]] = [
    # =========== Jira Tools ===========
    {
        "name": "list_jira_tickets",
        "description": "List Jira tickets from the project board. Use this to see what tickets are available to work on.",
        "input_schema": {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "description": "Filter by status (e.g., 'Ready for Agent', 'In Progress', 'To Do'). Leave empty for all statuses.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of tickets to return",
                    "default": 20,
                },
            },
        },
    },
    {
        "name": "get_jira_ticket",
        "description": "Get full details for a specific Jira ticket including description, acceptance criteria, and comments.",
        "input_schema": {
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": "The Jira ticket key (e.g., 'AENG-1234')",
                },
            },
            "required": ["key"],
        },
    },
    {
        "name": "transition_ticket",
        "description": "Change the status of a Jira ticket (e.g., move to 'In Progress' or 'Done').",
        "input_schema": {
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": "The Jira ticket key",
                },
                "status": {
                    "type": "string",
                    "description": "The target status name",
                },
            },
            "required": ["key", "status"],
        },
    },
    {
        "name": "add_ticket_comment",
        "description": "Add a comment to a Jira ticket.",
        "input_schema": {
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": "The Jira ticket key",
                },
                "comment": {
                    "type": "string",
                    "description": "The comment text to add",
                },
            },
            "required": ["key", "comment"],
        },
    },
    {
        "name": "search_tickets",
        "description": "Search for Jira tickets using JQL (Jira Query Language).",
        "input_schema": {
            "type": "object",
            "properties": {
                "jql": {
                    "type": "string",
                    "description": "JQL query string (e.g., 'project = AENG AND status = \"In Progress\"')",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum results to return",
                    "default": 20,
                },
            },
            "required": ["jql"],
        },
    },
    # =========== GitHub Tools ===========
    {
        "name": "list_pull_requests",
        "description": "List pull requests for the repository. Can filter to show only jirade-created PRs.",
        "input_schema": {
            "type": "object",
            "properties": {
                "state": {
                    "type": "string",
                    "enum": ["open", "closed", "all"],
                    "description": "Filter by PR state",
                    "default": "open",
                },
                "jirade_only": {
                    "type": "boolean",
                    "description": "Only show PRs created by jirade (with [jirade] tag)",
                    "default": False,
                },
            },
        },
    },
    {
        "name": "get_pull_request",
        "description": "Get details for a specific pull request including status, reviews, and CI checks.",
        "input_schema": {
            "type": "object",
            "properties": {
                "number": {
                    "type": "integer",
                    "description": "The PR number",
                },
            },
            "required": ["number"],
        },
    },
    {
        "name": "check_ci_status",
        "description": "Get the CI/CD status for a pull request (GitHub Actions, CircleCI, etc.).",
        "input_schema": {
            "type": "object",
            "properties": {
                "pr_number": {
                    "type": "integer",
                    "description": "The PR number to check",
                },
            },
            "required": ["pr_number"],
        },
    },
    # =========== File Tools ===========
    {
        "name": "read_file",
        "description": "Read the contents of a file in the repository.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file relative to repository root",
                },
                "max_lines": {
                    "type": "integer",
                    "description": "Maximum lines to read (default 500)",
                    "default": 500,
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Create or overwrite a file in the repository.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file relative to repository root",
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file",
                },
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "edit_file",
        "description": "Edit a file by replacing specific text. The old_string must match exactly.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file relative to repository root",
                },
                "old_string": {
                    "type": "string",
                    "description": "The exact text to find and replace",
                },
                "new_string": {
                    "type": "string",
                    "description": "The text to replace it with",
                },
            },
            "required": ["path", "old_string", "new_string"],
        },
    },
    {
        "name": "list_directory",
        "description": "List files and directories in a path.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to list (relative to repository root, or '.' for root)",
                    "default": ".",
                },
            },
        },
    },
    {
        "name": "search_files",
        "description": "Search for files matching a glob pattern.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern (e.g., '**/*.sql', 'dbt/models/**/*.yml')",
                },
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "search_content",
        "description": "Search for content in files using a regex pattern.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regex pattern to search for",
                },
                "file_pattern": {
                    "type": "string",
                    "description": "Optional glob pattern to filter files (e.g., '*.sql')",
                },
            },
            "required": ["pattern"],
        },
    },
    # =========== Git Tools ===========
    {
        "name": "git_status",
        "description": "Show the current git status including branch, changes, and untracked files.",
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "create_branch",
        "description": "Create a new git branch and check it out.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Branch name (e.g., 'feat/AENG-1234-add-metrics')",
                },
                "from_branch": {
                    "type": "string",
                    "description": "Base branch to create from (defaults to default branch)",
                },
            },
            "required": ["name"],
        },
    },
    {
        "name": "commit_changes",
        "description": "Stage all changes and create a commit.",
        "input_schema": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "Commit message (use conventional commits format)",
                },
            },
            "required": ["message"],
        },
    },
    {
        "name": "push_branch",
        "description": "Push the current branch to the remote repository.",
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "create_pull_request",
        "description": "Create a pull request for the current branch.",
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "PR title",
                },
                "body": {
                    "type": "string",
                    "description": "PR description/body",
                },
            },
            "required": ["title", "body"],
        },
    },
    # =========== Process Tools ===========
    {
        "name": "process_ticket",
        "description": "Fully process a Jira ticket: analyze requirements, make code changes, and create a PR. This is an autonomous operation that may take several minutes.",
        "input_schema": {
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": "The Jira ticket key to process (e.g., 'AENG-1234')",
                },
            },
            "required": ["key"],
        },
    },
    {
        "name": "fix_ci",
        "description": "Attempt to automatically fix CI failures on a pull request.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pr_number": {
                    "type": "integer",
                    "description": "The PR number with CI failures",
                },
            },
            "required": ["pr_number"],
        },
    },
    # =========== Utility Tools ===========
    {
        "name": "run_command",
        "description": "Run a shell command in the repository directory. Use for dbt, pre-commit, or other CLI tools.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The command to run (e.g., 'dbt compile', 'pre-commit run --all-files')",
                },
            },
            "required": ["command"],
        },
    },
]


def get_tools() -> list[dict[str, Any]]:
    """Return the list of tools for the REPL agent."""
    return TOOLS
