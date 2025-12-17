# Search Tools Documentation

This document provides comprehensive documentation for the Jira Agent's search capabilities, including file search and content search tools.

> **Status: Production Ready** ✅
>
> The search tools have been validated for production use and are a core component of the Jira Agent's codebase navigation capabilities.

## Overview

The Jira Agent provides two powerful search tools that enable it to navigate and understand codebases efficiently:

1. **`search_files`** - Find files by name/path patterns using glob syntax
2. **`search_content`** - Search within file contents using text or regex patterns

These tools are essential for the agent's ability to:
- Understand repository structure
- Find relevant code to modify
- Locate configuration files
- Identify dependencies between files

---

## Search Files Tool

### Purpose

The `search_files` tool allows the agent to locate files within a repository based on filename patterns using glob syntax.

### Usage

```json
{
  "name": "search_files",
  "input": {
    "pattern": "**/*.sql"
  }
}
```

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `pattern` | string | Yes | Glob pattern to match files |

### Glob Pattern Reference

| Pattern | Matches |
|---------|---------|
| `*` | Any sequence of characters within a path segment |
| `**` | Any sequence of characters across path segments (recursive) |
| `?` | Any single character |
| `[abc]` | Any character in the set |
| `[!abc]` | Any character not in the set |

### Examples

| Pattern | Description | Example Matches |
|---------|-------------|-----------------|
| `**/*.sql` | All SQL files | `models/orders.sql`, `dbt/models/marts/customers.sql` |
| `**/*.py` | All Python files | `src/main.py`, `tests/test_agent.py` |
| `**/test_*.py` | All Python test files | `tests/test_agent.py`, `src/tests/test_utils.py` |
| `models/*.sql` | SQL files directly in models | `models/orders.sql` (not `models/staging/orders.sql`) |
| `**/schema.yml` | All schema.yml files | `dbt/models/schema.yml`, `dbt/models/staging/schema.yml` |
| `*.md` | Markdown files in root | `README.md`, `CONTRIBUTING.md` |
| `configs/*.yaml` | YAML files in configs | `configs/prod.yaml`, `configs/dev.yaml` |
| `**/*orders*` | Files with "orders" in name | `models/orders.sql`, `src/orders_utils.py` |

### Output

Returns a newline-separated list of matching file paths (relative to repository root), limited to 100 results:

```
models/marts/orders.sql
models/staging/stg_orders.sql
models/intermediate/int_orders.sql
```

If no files match:
```
No files found matching pattern: **/*.xyz
```

### Best Practices

1. **Start broad, then narrow**: Begin with `**/*.sql` to understand the scope, then narrow down
2. **Use recursive patterns**: Prefer `**/*.sql` over `*.sql` to search all directories
3. **Be specific when possible**: `models/**/orders*.sql` is better than `**/*orders*`
4. **Consider file extensions**: Include extensions to avoid matching directories

---

## Search Content Tool

### Purpose

The `search_content` tool searches within file contents to find files containing specific text or patterns. It uses `grep` under the hood for efficient searching.

### Usage

```json
{
  "name": "search_content",
  "input": {
    "pattern": "customer_id",
    "file_pattern": "*.sql"
  }
}
```

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `pattern` | string | Yes | - | Text or regex pattern to search for |
| `file_pattern` | string | No | `*` | Glob pattern to filter which files to search |

### Pattern Syntax

The search pattern supports basic regex syntax:

| Pattern | Matches |
|---------|---------|
| `text` | Literal text |
| `text.*pattern` | "text" followed by any characters then "pattern" |
| `^text` | "text" at the start of a line |
| `text$` | "text" at the end of a line |
| `\bword\b` | Whole word "word" (word boundaries) |
| `[0-9]+` | One or more digits |
| `(foo\|bar)` | Either "foo" or "bar" |

### Examples

#### Find SQL files referencing a column
```json
{
  "pattern": "customer_lifetime_value",
  "file_pattern": "*.sql"
}
```

#### Find Python files importing a module
```json
{
  "pattern": "from anthropic import",
  "file_pattern": "*.py"
}
```

#### Find YAML files with a specific key
```json
{
  "pattern": "materialized:",
  "file_pattern": "*.yml"
}
```

#### Find all references to a model
```json
{
  "pattern": "ref\\('orders'\\)",
  "file_pattern": "*.sql"
}
```

#### Find dbt source references
```json
{
  "pattern": "source\\('",
  "file_pattern": "*.sql"
}
```

### Output

Returns a newline-separated list of files containing the pattern:

```
./models/marts/orders.sql
./models/staging/stg_customers.sql
```

If no matches found:
```
No matches found for pattern: nonexistent_column
```

### Best Practices

1. **Filter by file type**: Always use `file_pattern` to limit search scope
2. **Escape regex characters**: Use `\\` to escape special characters like `(`, `)`, `.`
3. **Use word boundaries**: `\bword\b` prevents matching partial words
4. **Case sensitivity**: Searches are case-sensitive by default

---

## Common Search Workflows

### 1. Finding a Model's Dependencies

```
Step 1: Find the model file
  search_files: **/*orders*.sql

Step 2: Read the model to see refs
  read_file: models/marts/orders.sql

Step 3: Find upstream models
  search_content: pattern="ref('stg_orders')", file_pattern="*.sql"
```

### 2. Understanding a Column's Usage

```
Step 1: Find all files mentioning the column
  search_content: pattern="customer_id", file_pattern="*.sql"

Step 2: Read each file to understand context
  read_file: <each matching file>
```

### 3. Finding Configuration Files

```
Step 1: Find all YAML configs
  search_files: **/*.yaml

Step 2: Find specific configuration
  search_content: pattern="database:", file_pattern="*.yaml"
```

### 4. Locating Test Files

```
Step 1: Find all test files
  search_files: **/test_*.py

Step 2: Find tests for specific functionality
  search_content: pattern="test_orders", file_pattern="test_*.py"
```

### 5. Exploring Repository Structure

```
Step 1: List root directory
  list_directory: .

Step 2: Search for common file types
  search_files: **/*.sql    (for dbt models)
  search_files: **/*.py     (for Python code)
  search_files: **/*.yaml   (for configuration)
```

---

## Troubleshooting

### No Results Returned

**Possible causes:**
- Pattern doesn't match any files
- Incorrect glob syntax (use `**` for recursive search)
- Files are in `.gitignore` directories

**Solutions:**
1. Start with a broader pattern: `**/*`
2. Check directory structure with `list_directory`
3. Verify file extensions are correct

### Too Many Results

**Possible causes:**
- Pattern is too broad
- Missing file type filter

**Solutions:**
1. Add file type restriction: `**/*.sql` instead of `**/*`
2. Be more specific in the pattern
3. Combine with content search to narrow down

### Pattern Not Matching Expected Files

**Possible causes:**
- Case sensitivity mismatch
- Special characters not escaped
- Pattern syntax error

**Solutions:**
1. Check exact filename/path casing
2. Escape special regex characters: `\.`, `\(`, `\)`
3. Test simpler patterns first

---

## Performance Considerations

1. **Limit recursive searches**: `**/*` searches entire repository, which can be slow for large repos
2. **Use file type filters**: `file_pattern` significantly improves content search speed
3. **Results are capped**: `search_files` returns maximum 100 results to prevent overflow
4. **Binary files**: Content search may not work well with binary files

---

## Integration with Other Tools

The search tools work best when combined with other agent tools:

| Tool | Use With Search |
|------|-----------------|
| `read_file` | Read files found by search |
| `list_directory` | Understand structure before searching |
| `write_file` | Modify files found by search |
| `run_command` | Run commands on files found |

### Example Workflow

```
1. search_files(**/*.sql) → Find all SQL files
2. search_content("customer_id", "*.sql") → Find relevant files
3. read_file(models/orders.sql) → Read the file
4. write_file(models/orders.sql, ...) → Make changes
5. run_command("dbt compile --select orders") → Validate
```

---

## Production Readiness

The search tools are production-ready and have been validated for use in automated ticket processing workflows.

### Reliability

| Aspect | Status | Notes |
|--------|--------|-------|
| File search | ✅ Stable | Uses Python's `pathlib.glob()` |
| Content search | ✅ Stable | Uses system `grep` command |
| Error handling | ✅ Complete | Returns clear error messages |
| Edge cases | ✅ Handled | Empty results, missing paths |

### Limitations

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| 100 result limit | Large repos may not show all matches | Use more specific patterns |
| Case-sensitive | May miss case-variant matches | Document expected casing |
| Binary files | Content search skips binary files | Use file search for binaries |
| Symlinks | May not follow symlinks | Use explicit paths |

### Monitoring

When using search tools in production:

1. **Log patterns used**: Track which search patterns are most effective
2. **Monitor failures**: Watch for "No matches found" in critical workflows
3. **Performance**: Large repositories may need pattern optimization

### Security Considerations

- Search is scoped to the cloned repository directory
- No access to files outside the workspace
- Pattern injection is not a concern (patterns are used as-is)

---

## Changelog

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2024-01-15 | Initial production release |
| - | - | Added `search_files` tool |
| - | - | Added `search_content` tool |
| - | - | Comprehensive documentation |

---

## See Also

- [HOW_IT_WORKS.md](./HOW_IT_WORKS.md) - Overall agent architecture
- [README.md](../README.md) - Getting started guide
- [CHANGELOG.md](../CHANGELOG.md) - Project changelog
