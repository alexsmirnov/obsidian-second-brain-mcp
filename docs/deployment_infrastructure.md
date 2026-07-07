# Deployment and Infrastructure

Python package deployment using Hatchling build system and UV package manager. #deployment #infrastructure #python

## Deployment Method

The project is packaged and distributed as a **Python package** installable via pip or UV.

### Build System

**Hatchling** [pyproject.toml:33-39](../pyproject.toml#L33-L39):
```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/mcps"]
sources = ["src", "tests"]
```

### Console Script Entry Point

**Entry Point** [pyproject.toml:30-31](../pyproject.toml#L30-L31):
```toml
[project.scripts]
mcps = "mcps:main"
```

After installation, the `mcps` command executes [src/mcps/__init__.py](../src/mcps/__init__.py), which delegates to [src/mcps/server.py:192-235](../src/mcps/server.py#L192-L235).

## Package Management

### UV Tool

**UV Configuration** [pyproject.toml:27-28](../pyproject.toml#L27-L28):
```toml
[tool.uv]
package = true
```

UV manages dependencies and provides faster package installation than pip.

### Dependency Lock File

**uv.lock** - Lock file ensuring reproducible installations with pinned dependency versions.

## Runtime Requirements

### Python Version

**Python 3.13+** required [pyproject.toml:9](../pyproject.toml#L9):
```toml
requires-python = ">=3.13"
```

### Runtime Dependencies

Core dependencies [pyproject.toml:10-26](../pyproject.toml#L10-L26):
- fastmcp==3.4.2
- lancedb==0.25.3
- markdown>=3.10.0
- rank_bm25
- python-dotenv>=1.2.2
- python-frontmatter>=1.3.0
- html2text>=2025.4.15
- langgraph>=1.1.6
- langchain-openai>=1.1.11
- langchain-community>=0.4.1
- langchain-aws>=1.4.0
- lxml>=5.0.0
- arxiv-to-prompt>=0.13.3
- pymupdf>=1.25.0
- langchain-google-genai>=4.2.5

## Development Environment

### Development Dependencies

**Dependency Groups** [pyproject.toml:41-52](../pyproject.toml#L41-L52):

**dev group**:
- pytest==8.3.4
- pytest-asyncio==0.25.3
- pytest-httpx>=0.35.0
- datasets>=4.5.0

**lint group**:
- pyright>=1.1.407
- pyrefly>=0.15.2
- ruff>=0.11.10

### Code Quality Tools

**Pyright** [pyproject.toml:68-79](../pyproject.toml#L68-L79):
- Type checker with basic type checking mode
- Configured for src directory
- Virtual environment: .venv

**Ruff** [pyproject.toml:81-112](../pyproject.toml#L81-L112):
- Linter and formatter
- Target: Python 3.13
- Line length: 88 characters
- Enabled rules: E, F, B, I, N, UP, RUF
- Double quotes for strings, 4-space indentation

**Pyrefly** [pyproject.toml:113-121](../pyproject.toml#L113-L121):
- Project analysis tool
- Includes: src directory
- Excludes: node_modules, __pycache__

## Infrastructure Requirements

### File System

**Vault Directory**: Obsidian vault root directory containing markdown files
- Configured via `VAULT` environment variable
- Default skip patterns exclude: hidden files, node_modules, __pycache__, scripts, templates, prompts

**Log Directory**: `~/Library/Logs/Mcps/` [src/mcps/logs.py:10](../src/mcps/logs.py#L10)
- Log file: `mcps.log`
- Rotating file handler: 10MB max size, 5 backups

**Cache Directory**: Configurable via ServerConfig
- Default: `./cache` relative to project root

### API Services

External API services required for full functionality:

**Required**:
- OpenAI-compatible model router for chat/embedding access (used by RAG and web research)

**Optional**:
- Google Custom Search API key and search engine ID (web research alternative to DuckDuckGo)

API keys configured via environment variables (see [config_environment.md](config_environment.md)).

## Startup Sequence

### Server Initialization

**Main Entry** [src/mcps/server.py:192-235](../src/mcps/server.py#L192-L235):
1. Parse CLI arguments via `parse_args()`
2. Create configuration via `mcps.config.create_config()`
3. If `--reindex`: validate vault and run one-shot indexing
4. Otherwise create server via `mcps.server.create_server(config)`
5. Setup logging via `setup_logging()`
6. Run server asynchronously via `asyncio.run(server.start(...))`

**Server Factory** [src/mcps/server.py:134-146](../src/mcps/server.py#L134-L146):
1. Instantiate DevAutomationServer with composed lifespan
2. Register tools (`web_research` and Obsidian tools when vault is configured)
3. Return configured server instance

**Server Start** [src/mcps/server.py:125-131](../src/mcps/server.py#L125-L131):
- Forwards to FastMCP `run_async()`
- Lifespan context managers handle vault and HTTP client lifecycle
- Clean shutdown on exit

## Build and Package

### Building the Package

```bash
# Using UV
uv build

# Using Hatchling directly
python -m build
```

### Installing the Package

```bash
# Install from source
uv pip install .

# Install in editable mode for development
uv pip install -e .

# Install with development dependencies
uv pip install -e ".[dev,lint]"
```

## Environment Isolation

### Virtual Environment

The project uses Python virtual environments for isolation:
- Virtual environment directory: `.venv`
- Managed by UV or standard venv

### Container Support

**Multi-stage Dockerfile** [Dockerfile:1](../Dockerfile):
- Builder stage: `python:3.13-slim` with `uv`, installs locked dependencies via `uv sync --locked` into `/app/.venv`.
- Runtime stage: `python:3.13-slim`, non-root `mcps` user, copies app source and builder `.venv`, sets `PATH=/app/.venv/bin:$PATH` and `PYTHONUNBUFFERED=1`.
- Default command runs the server with streamable HTTP transport: `mcps --transport streamable-http --host 0.0.0.0 --port 8000`.
- `EXPOSE 8000` documents the container network port.

**Runtime contract**:
- Vault must be mounted explicitly: `-v <host-vault-path>:/vault -e VAULT=/vault`.
- Secrets/config supplied via `--env-file .env` (see [config_environment.md](config_environment.md)).
- MCP clients connect over streamable HTTP at `http://<host>:8000/mcp`.

**Build and run**:
```bash
docker build -t mcps:local .
docker run --rm --env-file .env -e VAULT=/vault -v <host-vault-path>:/vault -p 8000:8000 mcps:local
```

**`.dockerignore`** [.dockerignore:1](../.dockerignore) excludes VCS metadata, `.venv`, Python/test caches, and local tool/agent config directories from the build context.

## Logging Configuration

**Log Setup** [src/mcps/logs.py:6-44](../src/mcps/logs.py#L6-L44):
- Removes console handlers including Rich formatters
- Configures rotating file handler
- Reduces httpx logging noise to WARNING level
- Log format includes timestamp, level, module, and message
