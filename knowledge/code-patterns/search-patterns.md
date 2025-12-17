---
id: search-patterns-001
timestamp: 2024-01-15T00:00:00Z
category: code-pattern
subcategory: search-tools
confidence: high
status: production-ready
---

# Common Search Patterns for Repository Navigation

> **Status: Production Ready** - These patterns have been validated through extensive use and are recommended for production workflows.

This document captures effective search patterns used by the Jira Agent when navigating and understanding codebases.

## Problem

When processing tickets, the agent needs to efficiently locate relevant files and understand code dependencies without manually browsing the entire repository.

## Solution

Use targeted search patterns based on repository type and file conventions.

### dbt Repository Patterns

| Goal | Pattern |
|------|---------|
| Find all models | `search_files: **/*.sql` |
| Find specific model | `search_files: **/*orders*.sql` |
| Find schema files | `search_files: **/schema.yml` |
| Find model refs | `search_content: ref\('model_name'\), *.sql` |
| Find source refs | `search_content: source\(', *.sql` |
| Find macros | `search_files: **/macros/**/*.sql` |
| Find tests | `search_files: **/tests/**/*.sql` |

### Python Repository Patterns

| Goal | Pattern |
|------|---------|
| Find all Python files | `search_files: **/*.py` |
| Find test files | `search_files: **/test_*.py` |
| Find class definition | `search_content: class ClassName, *.py` |
| Find function definition | `search_content: def function_name, *.py` |
| Find imports | `search_content: from module import, *.py` |
| Find configuration | `search_files: **/config*.py` |

### Configuration Patterns

| Goal | Pattern |
|------|---------|
| Find YAML configs | `search_files: **/*.yaml` |
| Find JSON configs | `search_files: **/*.json` |
| Find environment files | `search_files: **/.env*` |
| Find Docker files | `search_files: **/Dockerfile*` |
| Find CI configs | `search_files: **/.github/**/*.yml` |

## Workflow Strategy

### Step 1: Understand Repository Structure
```
list_directory: .
search_files: **/*.sql    # Check for dbt
search_files: **/*.py     # Check for Python
search_files: **/*.yaml   # Check for configs
```

### Step 2: Find Relevant Files
```
# Based on ticket requirements, search for relevant files
search_content: <keyword from ticket>, <file_type>
search_files: **/*<component>*
```

### Step 3: Trace Dependencies
```
# For SQL/dbt
search_content: ref\('<found_model>'\), *.sql

# For Python
search_content: from <module> import, *.py
```

## Files Affected
- All search operations

## Applicability
- When agent needs to navigate unfamiliar repositories
- When locating files related to ticket requirements
- When understanding code dependencies before making changes
