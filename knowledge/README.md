# Jira Agent Knowledge Base

This directory contains accumulated learnings from the jira-agent's operation across repositories.

## Structure

- `ci-failures/` - Patterns for resolving CI/CD failures (pre-commit, dbt compile, tests)
- `code-patterns/` - Coding patterns and conventions discovered during implementation
- `error-resolutions/` - Common error resolutions and troubleshooting guides

## How Learnings Are Captured

Learnings are automatically captured when the agent successfully resolves a failure:

1. **Failure Detection**: Agent encounters an error (e.g., pre-commit fails, dbt compile error)
2. **Fix Attempt**: Agent modifies code to address the issue
3. **Verification**: The same operation now succeeds (e.g., pre-commit passes)
4. **Capture**: Learning is saved with the failure context, solution, and affected files

Only **verified fixes** are captured - if a fix doesn't work, no learning is saved.

## Learning Format

Each learning is stored as a markdown file with frontmatter:

```markdown
---
id: abc123def456
timestamp: 2024-12-17T10:30:00Z
ticket: PROJ-1234
category: ci-failure
subcategory: pre-commit
repo: your-org/your-repo
confidence: high
---

# Title describing the learning

## Problem
Description of the error encountered.

## Solution
What was done to fix it.

## Files Affected
- file1.py
- file2.sql

## Applicability
When this learning should be applied.
```

## Usage

These learnings can be used to:

1. Provide context to the agent for similar future situations
2. Document common issues and their resolutions
3. Build institutional knowledge across repositories

## Contributing

Learnings are automatically added via PRs from the agent. To add a learning manually:

1. Create a markdown file in the appropriate category directory
2. Follow the format above with frontmatter
3. Use a unique ID (first 12 chars of SHA-256 hash of content)
