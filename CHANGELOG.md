# Changelog

All notable changes to the Jira Agent project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Search tools documentation improvements for production readiness
- CHANGELOG.md to track project changes

## [1.0.0] - 2024-01-15

### Added
- **Search Tools (Production Ready)**
  - `search_files` tool for finding files by glob patterns
  - `search_content` tool for searching within file contents using regex
  - Comprehensive documentation in `docs/SEARCH_TOOLS.md`
  - Search patterns knowledge base in `knowledge/code-patterns/search-patterns.md`

- **Core Agent Features**
  - Autonomous ticket processing from Jira
  - Claude Opus 4.5 integration for intelligent code generation
  - Interactive CLI for ticket selection and processing
  - Watch mode for continuous ticket monitoring

- **Git Operations**
  - Branch creation with configurable naming patterns
  - Commit with conventional commit message support
  - Push and pull request creation via GitHub API

- **Repository Operations**
  - File read/write operations
  - Directory listing
  - Shell command execution

- **CI/CD Integration**
  - Automatic CI failure detection
  - Pre-commit auto-fix strategies
  - PR status monitoring

- **Learning System**
  - Automatic capture of verified fixes
  - Knowledge base for CI failures, code patterns, and error resolutions

- **Authentication**
  - Jira OAuth 2.0 (3LO) support
  - GitHub token integration (via `gh` CLI or environment variable)
  - Secure credential storage in system keychain

### Documentation
- Comprehensive README with installation and setup guides
- `docs/HOW_IT_WORKS.md` - Detailed architecture documentation
- `docs/SEARCH_TOOLS.md` - Search tools user guide
- Knowledge base structure in `knowledge/`

### Infrastructure
- Docker support for containerized deployment
- Poetry-based dependency management
- Multi-repository configuration support
