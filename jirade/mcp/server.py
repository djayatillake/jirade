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
from mcp.types import TextContent, Tool

from .handlers import dispatch_tool
from .tools import get_tools

# Configure logging to stderr (stdout is reserved for MCP JSON-RPC)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

# Create the MCP server
server = Server("jirade")


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
