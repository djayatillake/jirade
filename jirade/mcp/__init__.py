"""MCP (Model Context Protocol) server for jirade.

This module exposes jirade's Jira, GitHub, and dbt Cloud tools
as an MCP server for use with Claude Code and other MCP clients.
"""

from .server import main

__all__ = ["main"]
