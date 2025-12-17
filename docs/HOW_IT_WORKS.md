# How the Jira Agent Works

This document explains the internal architecture of the Jira Agent - an autonomous system that processes Jira tickets and implements code changes using Claude as the reasoning engine.

> **Looking to get started?** See the [README](../README.md) for installation instructions, prerequisites, and setup guides.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Data Flow](#data-flow)
- [Prompt Engineering](#prompt-engineering)
- [The Agentic Loop](#the-agentic-loop)
- [Tools](#tools)
- [Deterministic vs LLM-Generated](#deterministic-vs-llm-generated)
- [Example Workflow](#example-workflow)
- [Configuration](#configuration)
- [Feedback Loop: PR Comments and CI Failures](#feedback-loop-pr-comments-and-ci-failures)
- [Interactive Mode](#interactive-mode)

---

## Overview

The Jira Agent automates the software development workflow:

1. **Monitors Jira** for tickets in a configurable status (e.g., "Ready for Agent")
2. **Analyzes tickets** to understand what changes are needed
3. **Implements changes** by reading, modifying, and creating files
4. **Creates pull requests** following repository conventions
5. **Closes the loop** by transitioning tickets to "Done" when PRs are merged

The agent uses Claude (Opus 4.5) as its reasoning engine, giving it the ability to understand natural language requirements and make intelligent decisions about code changes.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              JIRA AGENT                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐                 │
│  │   Triggers   │     │    Agent     │     │   Clients    │                 │
│  ├──────────────┤     ├──────────────┤     ├──────────────┤                 │
│  │ • CLI        │────▶│ • Prompts    │────▶│ • Jira API   │                 │
│  │ • Watch loop │     │ • Tools      │     │ • GitHub API │                 │
│  │ • Webhooks   │     │ • Agentic    │     │ • Git        │                 │
│  └──────────────┘     │   Loop       │     └──────────────┘                 │
│                       └──────────────┘                                       │
│                              │                                               │
│                              ▼                                               │
│                       ┌──────────────┐                                       │
│                       │  Claude API  │                                       │
│                       │  (Opus 4.5)  │                                       │
│                       └──────────────┘                                       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Components

| Component | Purpose |
|-----------|---------|
| **Triggers** | Start ticket processing (CLI commands, polling loop, webhooks) |
| **Agent Core** | Orchestrates the agentic loop, manages prompts and tools |
| **Clients** | Communicate with external services (Jira, GitHub, Git) |
| **Claude API** | The LLM that reasons about tasks and decides actions |

---

## Data Flow

```
┌─────────────┐      ┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│   Jira      │      │   Agent     │      │   Claude    │      │   GitHub    │
│   Ticket    │─────▶│   Core      │◀────▶│   (LLM)     │─────▶│   PR        │
└─────────────┘      └─────────────┘      └─────────────┘      └─────────────┘
      │                    │                    │                    │
      │                    │                    │                    │
      ▼                    ▼                    ▼                    ▼
 ┌─────────┐         ┌─────────┐         ┌─────────┐         ┌─────────┐
 │Metadata │         │ Prompts │         │ Tool    │         │ PR URL  │
 │extracted│         │ built   │         │ calls   │         │ posted  │
 └─────────┘         └─────────┘         └─────────┘         └─────────┘
```

### Step-by-Step Flow

1. **Ticket Detection**: Agent finds ticket in "Ready for Agent" status
2. **Metadata Extraction**: Fetch ticket details from Jira API
3. **Skip Check**: Verify no `[AGENT-SKIP]` comment or existing PR
4. **Repository Setup**: Clone/update the target repository
5. **Prompt Construction**: Build system + user prompts from metadata
6. **Agentic Loop**: Claude reasons and calls tools iteratively
7. **PR Creation**: Push branch and create pull request
8. **Jira Update**: Post PR link as comment on ticket
9. **Status Transition**: Move ticket to "Done" when PR merges

---

## Prompt Engineering

The agent constructs two prompts that guide Claude's behavior:

### System Prompt

The system prompt provides context about the repository and conventions:

```
You are an autonomous software engineering agent specializing in data engineering.
Your task is to implement changes to the repository based on Jira ticket requirements.

## Repository Information
- Owner: acme
- Name: data
- Path: /tmp/jira-agent/acme-data
- Default Branch: main
- PR Target: develop

## Conventions
- Branch naming: {type}/{ticket_key}-{description}
- Commit style: conventional
- PR title pattern: {type}({scope}): {description} ({ticket_key})

## dbt Information
dbt is enabled for this repo. Projects: ['dbt-databricks', 'dbt']

## Workflow
1. Analyze the ticket requirements
2. Search the codebase for relevant files
3. Create a feature branch from main
4. Make the necessary code changes
5. Run any validation (dbt compile, pre-commit)
6. Commit changes with a conventional commit message
7. Push the branch and create a PR
8. Return the PR URL

## Important Rules
- Always create a new branch from main
- Follow existing code patterns in the repository
- Write clear, descriptive commit messages
- Create PRs targeting develop
- If you encounter errors, try to fix them before giving up
```

### User Prompt (Task)

The user prompt contains the specific ticket to implement:

```
Please implement the following Jira ticket:

**Ticket Key:** AENG-1234
**Summary:** Add customer_lifetime_value column to orders model
**Type:** Story
**Priority:** Medium

**Description:**
We need to add a customer_lifetime_value column to the orders model.
Calculate it as the sum of all previous orders for that customer.

Please:
1. Analyze what changes are needed
2. Search for relevant files in the repository
3. Create a feature branch
4. Implement the required changes
5. Commit and push the changes
6. Create a pull request

Start by exploring the codebase to understand the changes needed.
```

### Metadata Sources

| Data | Source | Example |
|------|--------|---------|
| Ticket key | Jira API | `AENG-1234` |
| Summary | Jira API | `Add customer_lifetime_value column` |
| Description | Jira API (ADF→text) | Plain text requirements |
| Type | Jira API | `Story`, `Bug`, `Task` |
| Priority | Jira API | `High`, `Medium`, `Low` |
| Repo owner/name | Config YAML | `acme/data` |
| Branch pattern | Config YAML | `{type}/{ticket_key}-{description}` |
| PR target | Config YAML | `develop` |
| dbt projects | Config YAML | `['dbt-databricks']` |

---

## The Agentic Loop

The core of the agent is an iterative loop where Claude reasons about the task and takes actions:

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           AGENTIC LOOP                                    │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│   messages = [user_prompt]                                                │
│   iteration = 0                                                           │
│                                                                           │
│   while iteration < 50:  ◄─────────────────────────────────────────┐     │
│       │                                                             │     │
│       ▼                                                             │     │
│   ┌───────────────────────────────────────┐                         │     │
│   │        Claude API Call                 │                         │     │
│   │  ┌─────────────────────────────────┐  │                         │     │
│   │  │ Input:                          │  │                         │     │
│   │  │  • system_prompt                │  │                         │     │
│   │  │  • tools (10 available)         │  │                         │     │
│   │  │  • messages (conversation)      │  │                         │     │
│   │  └─────────────────────────────────┘  │                         │     │
│   │  ┌─────────────────────────────────┐  │                         │     │
│   │  │ Output:                         │  │                         │     │
│   │  │  • Text response                │  │                         │     │
│   │  │  • Tool calls (optional)        │  │                         │     │
│   │  │  • stop_reason                  │  │                         │     │
│   │  └─────────────────────────────────┘  │                         │     │
│   └───────────────────────────────────────┘                         │     │
│       │                                                             │     │
│       ▼                                                             │     │
│   stop_reason == "end_turn"? ───yes───▶ Extract PR URL & return     │     │
│       │                                                             │     │
│       no                                                            │     │
│       │                                                             │     │
│       ▼                                                             │     │
│   stop_reason == "tool_use"?                                        │     │
│       │                                                             │     │
│       yes                                                           │     │
│       │                                                             │     │
│       ▼                                                             │     │
│   ┌───────────────────────────────────────┐                         │     │
│   │     Execute Tool Calls                 │                         │     │
│   │  ┌─────────────────────────────────┐  │                         │     │
│   │  │ For each tool_use in response:  │  │                         │     │
│   │  │  • read_file(path)              │  │                         │     │
│   │  │  • write_file(path, content)    │  │                         │     │
│   │  │  • create_branch(name)          │  │                         │     │
│   │  │  • etc.                         │  │                         │     │
│   │  └─────────────────────────────────┘  │                         │     │
│   └───────────────────────────────────────┘                         │     │
│       │                                                             │     │
│       ▼                                                             │     │
│   Append assistant response + tool results to messages              │     │
│       │                                                             │     │
│       └─────────────────────────────────────────────────────────────┘     │
│                                                                           │
└──────────────────────────────────────────────────────────────────────────┘
```

### Key Characteristics

- **Iterative**: Claude can take multiple steps to complete a task
- **Conversational**: Each iteration builds on previous context
- **Tool-augmented**: Claude can read files, make changes, run commands
- **Self-correcting**: If something fails, Claude can try a different approach
- **Bounded**: Maximum 50 iterations prevents infinite loops

---

## Tools

The agent provides Claude with 10 tools for interacting with the codebase:

### File Operations

| Tool | Description | LLM Provides | Returns |
|------|-------------|--------------|---------|
| `read_file` | Read file contents | `path` | File contents as string |
| `write_file` | Create/overwrite file | `path`, `content` | Success message |
| `list_directory` | List directory contents | `path` | `d`/`f` prefixed listing |
| `search_files` | Find files by glob pattern | `pattern` | Matching file paths |
| `search_content` | Search file contents | `pattern`, `file_pattern` | Files containing pattern |

### Git Operations

| Tool | Description | LLM Provides | Returns |
|------|-------------|--------------|---------|
| `create_branch` | Create and checkout branch | `branch_name` | Success message |
| `commit_changes` | Stage all and commit | `message` | Commit SHA |
| `push_branch` | Push to origin | (none) | Success message |

### GitHub Operations

| Tool | Description | LLM Provides | Returns |
|------|-------------|--------------|---------|
| `create_pull_request` | Create PR via API | `title`, `body` | PR number and URL |

### Shell Operations

| Tool | Description | LLM Provides | Returns |
|------|-------------|--------------|---------|
| `run_command` | Execute shell command | `command` | Exit code, stdout, stderr |

### Tool Schema Example

```json
{
  "name": "create_pull_request",
  "description": "Create a pull request for the current branch",
  "input_schema": {
    "type": "object",
    "properties": {
      "title": {
        "type": "string",
        "description": "PR title"
      },
      "body": {
        "type": "string",
        "description": "PR description"
      }
    },
    "required": ["title", "body"]
  }
}
```

---

## Deterministic vs LLM-Generated

Understanding what's controlled by code vs what Claude decides is crucial:

### Deterministic (Code-Controlled)

| Aspect | Implementation |
|--------|----------------|
| Ticket metadata extraction | Jira API calls |
| Repository cloning | GitPython library |
| System prompt template | Hard-coded in `agent.py` |
| Tool execution | `_execute_tool()` method |
| Git operations | GitPython commands |
| GitHub API calls | httpx requests |
| PR URL extraction | Regex on tool results |
| Jira status transitions | Jira API calls |
| Skip condition checks | Label/comment inspection |

### LLM-Generated (Claude Decides)

| Aspect | Guided By |
|--------|-----------|
| Which files to read/search | Ticket description |
| Code changes to make | Ticket requirements |
| Branch name | Pattern in system prompt |
| Commit message | Style in system prompt |
| PR title | Pattern in system prompt |
| PR body/description | Ticket context |
| Workflow order | Workflow in system prompt |
| Error recovery approach | "Try to fix" instruction |
| Tool call sequence | Task requirements |

### The Balance

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        DETERMINISTIC SCAFFOLDING                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                                                                        │  │
│  │   • Fetches ticket from Jira                                          │  │
│  │   • Clones repository                                                 │  │
│  │   • Builds prompts from templates                                     │  │
│  │   • Executes tool calls safely                                        │  │
│  │   • Posts results back to Jira/GitHub                                 │  │
│  │                                                                        │  │
│  │   ┌────────────────────────────────────────────────────────────────┐  │  │
│  │   │                    LLM REASONING CORE                          │  │  │
│  │   │                                                                │  │  │
│  │   │   • Understands natural language requirements                  │  │  │
│  │   │   • Explores codebase intelligently                           │  │  │
│  │   │   • Makes implementation decisions                            │  │  │
│  │   │   • Writes appropriate code                                   │  │  │
│  │   │   • Crafts meaningful commit messages                         │  │  │
│  │   │   • Handles errors and edge cases                             │  │  │
│  │   │                                                                │  │  │
│  │   └────────────────────────────────────────────────────────────────┘  │  │
│  │                                                                        │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Example Workflow

Here's a complete example of processing a ticket:

### Ticket

```
Key: AENG-1234
Summary: Add customer_lifetime_value column to orders model
Description: Calculate LTV as sum of all previous orders for each customer
```

### Agent Execution

```
ITERATION 1
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Claude thinks: "I need to find the orders model first"

Tool call: search_files
  Input:  { "pattern": "**/orders*.sql" }
  Output: "dbt-databricks/models/marts/orders.sql
           dbt-databricks/models/staging/stg_orders.sql"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ITERATION 2
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Claude thinks: "The marts/orders.sql is likely the main model"

Tool call: read_file
  Input:  { "path": "dbt-databricks/models/marts/orders.sql" }
  Output: "{{ config(materialized='table') }}

           SELECT
               order_id,
               customer_id,
               order_date,
               amount
           FROM {{ ref('stg_orders') }}"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ITERATION 3
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Claude thinks: "I'll create a branch and add the LTV calculation"

Tool call: create_branch
  Input:  { "branch_name": "feat/AENG-1234-add-customer-ltv" }
  Output: "Created and checked out branch: feat/AENG-1234-add-customer-ltv"

Tool call: write_file
  Input:  {
    "path": "dbt-databricks/models/marts/orders.sql",
    "content": "{{ config(materialized='table') }}

WITH order_history AS (
    SELECT
        customer_id,
        order_date,
        SUM(amount) OVER (
            PARTITION BY customer_id
            ORDER BY order_date
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ) AS customer_lifetime_value
    FROM {{ ref('stg_orders') }}
)

SELECT
    o.order_id,
    o.customer_id,
    o.order_date,
    o.amount,
    COALESCE(oh.customer_lifetime_value, 0) AS customer_lifetime_value
FROM {{ ref('stg_orders') }} o
LEFT JOIN order_history oh
    ON o.customer_id = oh.customer_id
    AND o.order_date = oh.order_date"
  }
  Output: "Successfully wrote to dbt-databricks/models/marts/orders.sql"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ITERATION 4
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Claude thinks: "Let me validate with dbt compile"

Tool call: run_command
  Input:  { "command": "cd dbt-databricks && dbt compile --select orders" }
  Output: "Exit code: 0
           stdout: Compiled node model.dbt_databricks.orders"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ITERATION 5
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Claude thinks: "Compilation passed, now commit and push"

Tool call: commit_changes
  Input:  { "message": "feat(dbt): add customer_lifetime_value to orders model (AENG-1234)" }
  Output: "Committed changes: a1b2c3d4"

Tool call: push_branch
  Input:  {}
  Output: "Pushed branch feat/AENG-1234-add-customer-ltv to origin"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ITERATION 6
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Claude thinks: "Creating the pull request"

Tool call: create_pull_request
  Input:  {
    "title": "feat(dbt): add customer_lifetime_value column (AENG-1234)",
    "body": "## Summary\n- Added `customer_lifetime_value` column to orders model\n- Calculates sum of all previous orders per customer\n- Uses window function for efficient computation\n\n## Test Plan\n- [ ] dbt compile passes\n- [ ] Verify LTV calculation logic\n\nCloses AENG-1234"
  }
  Output: "Created PR #567: https://github.com/acme/data/pull/567"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ITERATION 7
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Claude: "I've successfully implemented the ticket. Created PR #567."

stop_reason: "end_turn"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RESULT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Agent extracts PR URL from tool results: https://github.com/acme/data/pull/567
Agent posts comment to AENG-1234: "Created PR: https://github.com/acme/data/pull/567"

✓ Ticket processed successfully
```

---

## Configuration

The agent is configured through YAML files that define repository-specific settings:

### Example Configuration

```yaml
# configs/acme-data.yaml

repo:
  owner: "acme"
  name: "data"
  default_branch: "main"
  pr_target_branch: "develop"

jira:
  base_url: "https://acme.atlassian.net"
  project_key: "AENG"
  board_id: 296

branching:
  pattern: "{type}/{ticket_key}-{description}"
  types:
    feature: "feat"
    bugfix: "fix"
    refactor: "refactor"

pull_request:
  title_pattern: "{type}({scope}): {description} ({ticket_key})"
  template_path: ".github/PULL_REQUEST_TEMPLATE.md"

commits:
  style: "conventional"
  scope_required: true
  ticket_in_message: true

skip:
  comment_phrase: "[AGENT-SKIP]"
  labels:
    - "no-automation"
    - "manual-only"

dbt:
  enabled: true
  projects:
    - path: "dbt-databricks"
      manifest_path: "target_lock/manifest.json"

agent:
  status: "Ready for Agent"
  in_progress_status: "In Progress"
  done_status: "Done"
```

### Configuration Impact

| Setting | Affects |
|---------|---------|
| `repo.*` | Where code is cloned, PR targets |
| `jira.*` | Which tickets to fetch |
| `branching.pattern` | Branch names (in system prompt) |
| `pull_request.title_pattern` | PR titles (in system prompt) |
| `commits.style` | Commit messages (in system prompt) |
| `skip.*` | Which tickets to skip |
| `dbt.*` | dbt info in system prompt |
| `agent.status` | Which Jira status triggers processing |

---

## Feedback Loop: PR Comments and CI Failures

The agent doesn't just create PRs and walk away. It monitors for feedback and can respond to CI failures and review comments.

### Complete Lifecycle

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        COMPLETE TICKET LIFECYCLE                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌──────────────┐                                                          │
│   │    Jira      │                                                          │
│   │   Ticket     │                                                          │
│   │  "Ready for  │                                                          │
│   │   Agent"     │                                                          │
│   └──────┬───────┘                                                          │
│          │                                                                   │
│          ▼                                                                   │
│   ┌──────────────┐     ┌──────────────┐     ┌──────────────┐               │
│   │   Agent      │────▶│    PR        │────▶│    CI        │               │
│   │  Processes   │     │  Created     │     │   Runs       │               │
│   │   Ticket     │     │              │     │              │               │
│   └──────────────┘     └──────────────┘     └──────┬───────┘               │
│                                                     │                        │
│                              ┌──────────────────────┼──────────────────────┐│
│                              │                      │                      ││
│                              ▼                      ▼                      ▼│
│                        ┌──────────┐           ┌──────────┐          ┌────────┐
│                        │ CI Fails │           │ Review   │          │  CI    │
│                        │          │           │ Comments │          │ Passes │
│                        └────┬─────┘           └────┬─────┘          └───┬────┘
│                             │                      │                    │    │
│                             ▼                      ▼                    │    │
│                        ┌──────────┐           ┌──────────┐              │    │
│                        │  Agent   │           │  Agent   │              │    │
│                        │ Fixes CI │           │ Responds │              │    │
│                        └────┬─────┘           └────┬─────┘              │    │
│                             │                      │                    │    │
│                             └──────────┬───────────┘                    │    │
│                                        │                                │    │
│                                        ▼                                │    │
│                                  ┌──────────┐                           │    │
│                                  │  New     │◀──────────────────────────┘    │
│                                  │  Commit  │                                │
│                                  └────┬─────┘                                │
│                                       │                                      │
│                                       │ (loop until approved)                │
│                                       ▼                                      │
│                                 ┌──────────┐                                 │
│                                 │    PR    │                                 │
│                                 │  Merged  │                                 │
│                                 └────┬─────┘                                 │
│                                      │                                       │
│                                      ▼                                       │
│                               ┌──────────────┐                               │
│                               │    Jira      │                               │
│                               │   Ticket     │                               │
│                               │   "Done"     │                               │
│                               └──────────────┘                               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Trigger Methods

The agent can be triggered in two ways:

#### 1. Polling Mode (Recommended for Local Development)

```bash
jira-agent watch --config configs/acme-data.yaml --interval=60
```

The watch command polls both services:

```python
while True:
    # Poll Jira for tickets in trigger status
    tickets = await jira.search_issues(
        f'project = AENG AND status = "Ready for Agent"'
    )
    for ticket in tickets:
        await agent.process_single_ticket(ticket.key)

    # Poll GitHub for merged PRs
    prs = await github.list_pull_requests(state="closed")
    for pr in prs:
        if pr.merged_at and has_ticket_reference(pr):
            await agent.transition_ticket_to_done(ticket_key)

    await asyncio.sleep(60)
```

#### 2. Webhook Mode (For Production)

The webhook server listens for events:

| Source | Event | Trigger |
|--------|-------|---------|
| Jira | `jira:issue_updated` | Ticket assigned to agent |
| Jira | `comment_created` | Agent @mentioned in comment |
| GitHub | `check_run` (completed, failure) | CI check failed |
| GitHub | `pull_request_review` (changes_requested) | Reviewer requests changes |
| GitHub | `pull_request` (closed, merged) | PR merged |

```bash
jira-agent serve --port 8080 --config-dir ./configs
```

### CI Failure Handling

When CI fails, the agent attempts automatic fixes:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CI FAILURE FLOW                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   GitHub Webhook: check_run (conclusion: failure)                            │
│        │                                                                     │
│        ▼                                                                     │
│   ┌──────────────────────────────────────────────────┐                      │
│   │  1. Clone repo and checkout PR branch            │  DETERMINISTIC       │
│   │  2. Get failed check details from GitHub API     │                      │
│   └──────────────────────────────────────────────────┘                      │
│        │                                                                     │
│        ▼                                                                     │
│   ┌──────────────────────────────────────────────────┐                      │
│   │  Try auto-fix strategies (from config):          │                      │
│   │                                                  │                      │
│   │  for strategy in repo_config.ci.auto_fix:        │                      │
│   │      if strategy == "pre-commit":                │  DETERMINISTIC       │
│   │          run_pre_commit()                        │                      │
│   │          if changes_made:                        │                      │
│   │              commit("style: auto-fix")           │                      │
│   │              push()                              │                      │
│   │              return SUCCESS                      │                      │
│   │                                                  │                      │
│   │      if strategy == "dbt_compile":               │                      │
│   │          run_dbt_compile()                       │                      │
│   │          # Fix any detected issues               │                      │
│   └──────────────────────────────────────────────────┘                      │
│        │                                                                     │
│        ▼                                                                     │
│   Auto-fix failed? ──────────────────────────────────────────────┐          │
│        │                                                          │          │
│        ▼                                                          ▼          │
│   ┌──────────────────────────────────────────────────┐    ┌────────────┐   │
│   │  Invoke Claude to analyze and fix                │    │  Log error │   │
│   │  (Future enhancement - not yet implemented)      │    │  and skip  │   │
│   └──────────────────────────────────────────────────┘    └────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Configuration for Auto-Fix

```yaml
# configs/acme-data.yaml
ci:
  system: "circleci"
  auto_fix:
    - "pre-commit"    # Run pre-commit hooks to fix formatting
    - "dbt_compile"   # Re-compile dbt models to update manifest
```

#### Implementation

```python
async def fix_ci_failures(self, pr_number: int) -> dict[str, Any]:
    """Attempt to fix CI failures on a PR."""

    # Get PR and check status
    pr = await github.get_pull_request(pr_number)
    checks = await github.get_check_runs(pr["head"]["sha"])

    failed_checks = [c for c in checks if c["conclusion"] == "failure"]
    if not failed_checks:
        return {"fixed": True, "message": "No failed checks"}

    # Clone and checkout PR branch
    repo_path = git.clone_repo(owner, name)
    git.checkout_branch(pr["head"]["ref"])

    # Try auto-fix strategies from config
    for strategy in self.repo_config.ci.auto_fix:
        if strategy == "pre-commit":
            success, output = git.run_pre_commit()
            if not success and git.has_changes():
                git.stage_files()
                git.commit("style: auto-fix pre-commit issues")
                git.push()
                return {
                    "fixed": True,
                    "strategy": "pre-commit",
                    "commit_sha": git.repo.head.commit.hexsha,
                }

    return {"fixed": False, "error": "Could not auto-fix CI failures"}
```

### PR Review Comments

When a reviewer requests changes, the agent can respond:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      PR REVIEW RESPONSE FLOW                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   GitHub Webhook: pull_request_review (state: changes_requested)             │
│        │                                                                     │
│        ▼                                                                     │
│   ┌──────────────────────────────────────────────────┐                      │
│   │  1. Fetch PR details and review comments         │  DETERMINISTIC       │
│   │  2. Clone repo and checkout PR branch            │                      │
│   │  3. Read files mentioned in review               │                      │
│   └──────────────────────────────────────────────────┘                      │
│        │                                                                     │
│        ▼                                                                     │
│   ┌──────────────────────────────────────────────────┐                      │
│   │  Build prompt with:                              │                      │
│   │  • Original ticket context                       │                      │
│   │  • Current PR diff                               │  DETERMINISTIC       │
│   │  • Review comments and feedback                  │                      │
│   │  • Line-specific comments                        │                      │
│   └──────────────────────────────────────────────────┘                      │
│        │                                                                     │
│        ▼                                                                     │
│   ┌──────────────────────────────────────────────────┐                      │
│   │  Claude analyzes feedback and decides:           │                      │
│   │  • Which changes to make                         │  LLM-GENERATED       │
│   │  • How to address each comment                   │                      │
│   │  • Reply text for reviewer                       │                      │
│   └──────────────────────────────────────────────────┘                      │
│        │                                                                     │
│        ▼                                                                     │
│   ┌──────────────────────────────────────────────────┐                      │
│   │  Execute changes:                                │                      │
│   │  1. Modify files per feedback                    │  DETERMINISTIC       │
│   │  2. Commit with descriptive message              │  (execution)         │
│   │  3. Push to PR branch                            │                      │
│   │  4. Reply to review comments                     │  LLM-GENERATED       │
│   └──────────────────────────────────────────────────┘                      │  (content)
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Review Response Prompt Structure

```
You are responding to review feedback on a pull request.

## Original Ticket
Key: AENG-1234
Summary: Add customer_lifetime_value column
Description: [...]

## Current PR
Branch: feat/AENG-1234-add-ltv-column
Files changed: [...]

## Review Feedback
Reviewer: @john-smith
State: changes_requested

### Comment 1 (on dbt-databricks/models/marts/orders.sql:15)
"The window function should use ROWS BETWEEN UNBOUNDED PRECEDING
AND 1 PRECEDING, not CURRENT ROW, to exclude the current order
from the lifetime value calculation."

### Comment 2 (general)
"Please add a test for the LTV calculation."

## Instructions
1. Address each review comment
2. Make the necessary code changes
3. Reply to each comment explaining what you changed
```

### PR Merged → Ticket Done

When a PR is merged, the agent closes the loop:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PR MERGE FLOW                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   GitHub Webhook: pull_request (action: closed, merged: true)                │
│   OR: Polling detects merged PR                                              │
│        │                                                                     │
│        ▼                                                                     │
│   ┌──────────────────────────────────────────────────┐                      │
│   │  Extract ticket key from:                        │                      │
│   │  • PR title: "feat(dbt): add LTV (AENG-1234)"   │  DETERMINISTIC       │
│   │  • Branch: "feat/AENG-1234-add-ltv"            │  (regex matching)    │
│   └──────────────────────────────────────────────────┘                      │
│        │                                                                     │
│        ▼                                                                     │
│   ┌──────────────────────────────────────────────────┐                      │
│   │  Get available Jira transitions                  │  DETERMINISTIC       │
│   │  Find "Done" (or "Closed", "Complete")          │                      │
│   └──────────────────────────────────────────────────┘                      │
│        │                                                                     │
│        ▼                                                                     │
│   ┌──────────────────────────────────────────────────┐                      │
│   │  Transition ticket to Done                       │                      │
│   │  Add comment: "PR merged. Automatically          │  DETERMINISTIC       │
│   │               transitioned to Done."            │                      │
│   └──────────────────────────────────────────────────┘                      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Implementation

```python
async def transition_ticket_to_done(self, ticket_key: str) -> dict[str, Any]:
    """Transition a Jira ticket to Done status."""

    # Get available transitions
    transitions = await jira.get_issue_transitions(ticket_key)

    # Find "Done" transition
    done_names = ["done", "closed", "complete", "resolved"]
    done_transition = None
    for t in transitions:
        if t.get("name", "").lower() in done_names:
            done_transition = t
            break

    if not done_transition:
        return {"success": False, "error": f"No Done transition available"}

    # Perform transition
    await jira.transition_issue(ticket_key, done_transition["id"])

    # Add comment
    await jira.add_comment(
        ticket_key,
        f"PR merged. Automatically transitioned to {done_transition['name']}."
    )

    return {"success": True, "transition": done_transition["name"]}
```

### Event Summary Table

| Event | Source | Agent Action | Automated? |
|-------|--------|--------------|------------|
| Ticket in "Ready for Agent" | Jira | Process ticket, create PR | Yes |
| Ticket assigned to agent | Jira webhook | Process ticket, create PR | Yes |
| Agent @mentioned | Jira webhook | Process ticket/respond | Yes |
| CI check failure | GitHub webhook | Run auto-fix strategies | Partial |
| PR review: changes requested | GitHub webhook | Address feedback, push fix | Yes |
| PR merged | GitHub webhook/poll | Transition ticket to Done | Yes |

---

## Interactive Mode

For manual work, the agent provides an interactive CLI that lets you browse and select tickets with arrow keys.

### Usage

```bash
jira-agent list-tickets --config configs/your-repo.yaml --interactive
```

### How It Works

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        INTERACTIVE MODE FLOW                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   1. Fetch tickets from Jira (deterministic)                                │
│        │                                                                     │
│        ▼                                                                     │
│   ┌──────────────────────────────────────────────────────────────────────┐  │
│   │  ? Select a ticket to process:                                        │  │
│   │  ❯ PROJ-123     [To Do          ] Add user authentication endpoint    │  │
│   │    PROJ-124     [In Progress    ] Fix database connection timeout     │  │
│   │    PROJ-125     [Ready for Agent] Update API documentation            │  │
│   │    Cancel                                                             │  │
│   └──────────────────────────────────────────────────────────────────────┘  │
│        │                                                                     │
│        │ User selects with arrow keys + Enter                               │
│        ▼                                                                     │
│   ┌──────────────────────────────────────────────────────────────────────┐  │
│   │  ? What would you like to do?                                         │  │
│   │  ❯ Process this ticket (create PR)                                    │  │
│   │    Process with dry-run (preview only)                                │  │
│   │    View ticket details                                                │  │
│   │    Cancel                                                             │  │
│   └──────────────────────────────────────────────────────────────────────┘  │
│        │                                                                     │
│        │ If "Process" selected                                              │
│        ▼                                                                     │
│   ┌──────────────────────────────────────────────────────────────────────┐  │
│   │  ? Process PROJ-123 and create a PR? (Y/n)                           │  │
│   └──────────────────────────────────────────────────────────────────────┘  │
│        │                                                                     │
│        │ If confirmed                                                        │
│        ▼                                                                     │
│   Agent processes ticket using the agentic loop                             │
│   (same flow as automated processing)                                       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Features

| Feature | Description |
|---------|-------------|
| **Arrow Key Navigation** | Browse tickets without typing |
| **Any Status** | Process tickets regardless of Jira status |
| **Dry-Run Option** | Preview what would happen without making changes |
| **Ticket Details** | View ticket info before processing |
| **Confirmation** | Prompts before taking action |

### Comparison with Watch Mode

| Aspect | Interactive Mode | Watch Mode |
|--------|------------------|------------|
| **Invocation** | `list-tickets --interactive` | `watch` |
| **User Input** | Required (selection) | None |
| **Status Filter** | Any status | Only "Ready for Agent" |
| **Processing** | One ticket at a time | Continuous polling |
| **Use Case** | Manual work | Automation |

---

## Security Considerations

### What the Agent CAN Do
- Read any file in the cloned repository
- Write/modify files in the repository
- Run shell commands in the repository directory
- Create branches and push to origin
- Create pull requests

### What the Agent CANNOT Do
- Access files outside the workspace directory
- Push directly to protected branches (relies on GitHub branch protection)
- Merge pull requests (requires human approval)
- Access other repositories without explicit configuration
- Access secrets/credentials beyond what's needed for API calls

### Safeguards
- **Skip conditions**: `[AGENT-SKIP]` comment or labels prevent processing
- **Max iterations**: Loop bounded to 50 iterations
- **PR-based workflow**: All changes go through PR review
- **Branch protection**: Relies on GitHub's branch protection rules
- **Audit trail**: All actions logged, PRs link back to tickets

---

## Extending the Agent

### Adding New Tools

Add tool definition in `_get_agent_tools()`:

```python
{
    "name": "my_new_tool",
    "description": "Description for Claude",
    "input_schema": {
        "type": "object",
        "properties": {
            "param1": {"type": "string", "description": "..."},
        },
        "required": ["param1"],
    },
}
```

Add execution in `_execute_tool()`:

```python
elif tool_name == "my_new_tool":
    # Implementation here
    return "Result string"
```

### Modifying Prompts

System prompt is built in `_build_system_prompt()`. Add new sections for:
- Additional context about the repository
- Specific coding guidelines
- Custom workflow steps

### Adding Triggers

New triggers can be added by:
1. Creating a handler function in `main.py`
2. Adding CLI command in the docstring
3. Calling `agent.process_single_ticket()` or `agent.process_tickets()`

---

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Agent skips ticket | PR already exists or skip label | Check Jira comments/labels |
| Tool execution fails | File not found, permission error | Check repo clone, file paths |
| PR creation fails | Branch protection, permissions | Check GitHub token scopes |
| Max iterations reached | Task too complex | Simplify ticket requirements |
| Wrong branch target | Config mismatch | Check `pr_target_branch` in config |

### Debugging

Enable debug logging:
```bash
export LOG_LEVEL=DEBUG
jira-agent process-ticket AENG-1234 --config configs/acme-data.yaml
```

Check the agentic loop iterations in logs to see Claude's reasoning and tool calls.
