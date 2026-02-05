"""MCP server for jirade tools.

This server exposes jirade's Jira, GitHub, and dbt Cloud tools
via the Model Context Protocol (MCP) for use with Claude Code.
"""

import asyncio
import json
import logging
import sys
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import GetPromptResult, Prompt, PromptMessage, TextContent, Tool

from .handlers import dispatch_tool
from .tools import get_tools

# Instructions for using jirade with Claude Code
JIRADE_INSTRUCTIONS = """
## jirade Usage Instructions

When working on Jira tickets using jirade tools, follow these conventions:

### PR Title Format
When creating PRs for Jira tickets, always include `[jirade]` in the PR title:
- Format: `<type>(<scope>): <description> [jirade] (<TICKET-KEY>)`
- Example: `feat(segment): normalize Courier messages [jirade] (AENG-1885)`

### Code Formatting Requirements
- **SQL files must be formatted with sqlfmt** to pass CI pre-commit checks
- Run `sqlfmt <file.sql>` or `pre-commit run sqlfmt --files <file.sql>` before committing
- If CI fails on `pre-commit-python-3-11`, check sqlfmt formatting first

### Workflow
1. Get the Jira ticket details with `jirade_get_issue`
2. Transition the ticket to "In Progress" with `jirade_transition_issue`
3. Create your feature branch and make changes
4. **Format SQL files with sqlfmt before committing**
5. Create a PR with `[jirade]` in the title
6. Run `jirade_run_dbt_diff` to validate dbt model changes with test fixtures
7. Post the diff report with `jirade_post_diff_report`
8. **IMPORTANT: After successful local diff, ALWAYS trigger dbt Cloud CI with `jirade_dbt_trigger_ci_for_pr`**
9. Use `jirade_watch_pr` to monitor CI status until completion

### Automated CI Triggering
After running `jirade_run_dbt_diff`:
- If the local diff **succeeds** (models compile and execute correctly): Automatically trigger `jirade_dbt_trigger_ci_for_pr`
- If the local diff **fails**: Fix the issues first, do NOT trigger dbt Cloud CI

### PR Body
Include a link to jirade in the PR body:
```
🤖 Generated with [jirade](https://github.com/djayatillake/jirade)
```
"""

# Configure logging to stderr (stdout is reserved for MCP JSON-RPC)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

# Create the MCP server
server = Server("jirade")


@server.list_prompts()
async def list_prompts() -> list[Prompt]:
    """List available prompts.

    Returns:
        List of MCP Prompt objects.
    """
    return [
        Prompt(
            name="jirade-instructions",
            description="Instructions for using jirade tools with Claude Code",
        )
    ]


@server.get_prompt()
async def get_prompt(name: str) -> GetPromptResult:
    """Get a prompt by name.

    Args:
        name: Prompt name.

    Returns:
        GetPromptResult with the prompt content.
    """
    if name == "jirade-instructions":
        return GetPromptResult(
            messages=[
                PromptMessage(
                    role="user",
                    content=TextContent(type="text", text=JIRADE_INSTRUCTIONS),
                )
            ]
        )
    raise ValueError(f"Unknown prompt: {name}")


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools.

    Returns:
        List of MCP Tool objects.
    """
    tools = get_tools()
    return [
        Tool(
            name=tool["name"],
            description=tool["description"],
            inputSchema=tool["inputSchema"],
        )
        for tool in tools
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Handle a tool call.

    Args:
        name: Tool name.
        arguments: Tool arguments.

    Returns:
        List containing a single TextContent with the JSON result.
    """
    logger.info(f"Tool call: {name} with arguments: {arguments}")

    try:
        result = await dispatch_tool(name, arguments)
        return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
    except Exception as e:
        logger.exception(f"Tool {name} failed")
        error_result = {
            "error": True,
            "message": str(e),
            "tool": name,
        }
        return [TextContent(type="text", text=json.dumps(error_result, indent=2))]


async def run_server() -> None:
    """Run the MCP server."""
    logger.info("Starting jirade MCP server...")

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


def main() -> None:
    """Main entry point for jirade-mcp command."""
    try:
        asyncio.run(run_server())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.exception(f"Server error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
