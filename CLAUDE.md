# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Type

FastMCP-based Model Context Protocol server with RAG capabilities for Obsidian vault semantic search. Python 3.12+ with async/await and strict interface-based architecture.

## Essential Commands

```bash
# Run tests
uv run pytest test/test_file.py -v

# Type checking and linting
uvx pyright src/mcps/file.py
uvx ruff check src/mcps/file.py

# Run server
uv run mcps
```

## Obsidian Vault Specifics

The RAG system is optimized for Obsidian structure:
- Wikilinks `[[filename]]` extracted to chunk metadata for future graph-based search
- YAML frontmatter: `title`, `description`, `tags`, `source` properties indexed
- Semantic chunking splits by H2 headers (atomic information units in scientific essay structure)
- Tag filtering (exact match) and path filtering (LIKE patterns) supported in search
- Files organized in topic-based folders, with optional subfolders for narrow subjects

## Documentation

- Team code style and best practices: `rules/` folder
- Comprehensive project documentation: [docs/index.md](docs/index.md)
