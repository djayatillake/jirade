"""Claude agent for the Zoom meeting bot.

Handles queries detected by the transcript handler by calling Claude API
with jirade MCP tools, then returns concise responses suitable for
Zoom chat or TTS.
"""

import json
import logging
from typing import Any

from anthropic import Anthropic

from ..config import get_settings
from ..mcp.handlers import dispatch_tool
from ..mcp.tools import get_tools

logger = logging.getLogger(__name__)

# Convert MCP tool schemas to Anthropic API format
def _mcp_tools_to_anthropic() -> list[dict[str, Any]]:
    """Convert jirade MCP tool definitions to Anthropic tool_use format."""
    tools = []
    for tool in get_tools():
        tools.append({
            "name": tool["name"],
            "description": tool["description"],
            "input_schema": tool["inputSchema"],
        })
    return tools


SYSTEM_PROMPT = """You are jirade, a data engineering assistant participating in a Zoom meeting.
You have access to tools for Jira, GitHub, and dbt CI. Use them proactively
whenever a question could be answered with live data.

## Your tools

### Jira
- jirade_search_jira: Search with JQL
- jirade_get_issue: Get full issue details by key (e.g., "DE-1234")
- jirade_add_comment: Add a comment to an issue
- jirade_transition_issue: Move an issue to a new status

### GitHub
- jirade_list_prs: List PRs for a repo
- jirade_get_pr: Get PR details including reviews and comments
- jirade_get_ci_status: Check CI/CD status for a PR
- jirade_watch_pr: Watch a PR until CI completes

### dbt CI
- jirade_run_dbt_ci: Build changed dbt models in isolated CI schemas and diff against production
- jirade_analyze_deprecation: Find downstream impact of deprecating a table/column
- jirade_generate_schema_docs: Generate documentation context for dbt models
- jirade_cleanup_ci: Clean up CI schemas after a PR is merged

## Behavior
- You are jirade. Be helpful, thoughtful, and direct.
- Keep responses concise since they go to Zoom chat (2-4 sentences typically).
- Use plain text, not markdown formatting (no ** or ## etc).
- When asked about tickets, PRs, or CI status, USE THE TOOLS - don't guess.
- When asked about "your" PRs or what "you" are working on, search for PRs that have the "jirade" label or that were created by the jirade bot. Only claim ownership of PRs that are tagged or created by jirade.
- Always include the full GitHub URL for PRs (e.g., https://github.com/owner/repo/pull/123) so users can click them.
- Always include the full Jira URL for tickets so users can click them.
- If you don't have enough info (e.g., need a ticket key), say so and ask.
- You can chain multiple tool calls to answer complex questions.

## Meeting context
You may receive recent transcript context showing what people have been discussing.
Use it to understand what's being talked about, but focus on answering the specific question asked.
"""


class ZoomBotAgent:
    """Agent that processes queries from the Zoom meeting bot."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        max_response_tokens: int = 1024,
    ):
        settings = get_settings()
        self.claude = Anthropic(api_key=settings.anthropic_api_key)
        self.model = model
        self.max_response_tokens = max_response_tokens
        self._tools = _mcp_tools_to_anthropic()

    async def answer_query(
        self,
        speaker: str,
        query: str,
        transcript_context: str = "",
    ) -> str:
        """Process a query and return a response.

        Args:
            speaker: Name of the person who asked the question.
            query: The question or command text.
            transcript_context: Recent meeting transcript for context.

        Returns:
            Response text suitable for Zoom chat.
        """
        logger.info(f"Processing query from {speaker}: {query}")

        user_message = f"{speaker} asked: {query}"
        if transcript_context:
            user_message = f"Recent meeting transcript:\n{transcript_context}\n\n---\n\n{user_message}"

        messages: list[dict[str, Any]] = [{"role": "user", "content": user_message}]

        # Agentic loop (max 10 iterations for quick responses)
        for iteration in range(10):
            response = self.claude.messages.create(
                model=self.model,
                max_tokens=self.max_response_tokens,
                system=SYSTEM_PROMPT,
                tools=self._tools,
                messages=messages,
            )

            # If Claude is done, extract text response
            if response.stop_reason == "end_turn":
                text_parts = []
                for block in response.content:
                    if hasattr(block, "text") and block.text:
                        text_parts.append(block.text)
                result = " ".join(text_parts).strip()
                logger.info(f"Response ({iteration + 1} iterations): {result[:100]}...")
                return result

            # If Claude wants to use tools, execute them
            if response.stop_reason == "tool_use":
                messages.append({"role": "assistant", "content": response.content})
                tool_results = []

                for block in response.content:
                    if block.type == "tool_use":
                        logger.info(f"Tool call: {block.name}({json.dumps(block.input)[:100]}...)")
                        try:
                            result = await dispatch_tool(block.name, block.input)
                            result_str = json.dumps(result) if isinstance(result, dict) else str(result)
                        except Exception as e:
                            logger.error(f"Tool {block.name} failed: {e}")
                            result_str = f"Error: {e}"

                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result_str,
                        })

                messages.append({"role": "user", "content": tool_results})

        return "Sorry, I took too long processing that. Could you ask again more specifically?"
