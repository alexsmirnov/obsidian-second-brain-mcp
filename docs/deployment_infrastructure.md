# Deployment and Infrastructure

Python package deployment using Hatchling build system and UV package manager. #deployment #infrastructure #python

## Deployment Method

The project is packaged and distributed as a **Python package** installable via pip or UV.

### Build System

**Hatchling** [pyproject.toml:27-33](../pyproject.toml#L27-L33):
```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/mcps"]
```

### Console Script Entry Point

**Entry Point** [pyproject.toml:24-25](../pyproject.toml#L24-L25):
```toml
[project.scripts]
mcps = "mcps:main"
```

After installation, the `mcps` command executes [src/mcps/__init__.py:15-30](../src/mcps/__init__.py#L15-L30).

## Package Management

### UV Tool

**UV Configuration** [pyproject.toml:21-22](../pyproject.toml#L21-L22):
```toml
[tool.uv]
package = true
```

UV manages dependencies and provides faster package installation than pip.

### Dependency Lock File

**uv.lock** - Lock file ensuring reproducible installations with pinned dependency versions.

## Runtime Requirements

### Python Version

**Python 3.12+** required [pyproject.toml:9](../pyproject.toml#L9):
```toml
requires-python = ">=3.12"
```

### Runtime Dependencies

Core dependencies [pyproject.toml:10-20](../pyproject.toml#L10-L20):
- fastmcp >= 2.8.1
- lancedb > 0.23.0
- markdown >= 3.4.0
- rank_bm25
- python-dotenv >= 1.0.0
- python-frontmatter >= 1.1.0
- langchain-core / langchain-openai
- httpx

## Development Environment

### Development Dependencies

**Dependency Groups** [pyproject.toml:35-43](../pyproject.toml#L35-L43):

**dev group**:
- pytest == 8.3.4
- pytest-asyncio == 0.25.3

**lint group**:
- pyrefly >= 0.15.2
- ruff >= 0.11.10

### Code Quality Tools

**Pyright** [pyproject.toml:59-70](../pyproject.toml#L59-L70):
- Type checker with basic type checking mode
- Configured for src directory
- Virtual environment: .venv

**Ruff** [pyproject.toml:72-103](../pyproject.toml#L72-L103):
- Linter and formatter
- Target: Python 3.13
- Line length: 88 characters
- Enabled rules: E, F, B, I, N, UP, RUF
- Double quotes for strings, 4-space indentation

**Pyrefly** [pyproject.toml:104-112](../pyproject.toml#L104-L112):
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
- LiteLLM Router for model and embedding access

**Optional**:
- Serper API (web search tool)
- Tavily API (web search tool)

API keys configured via environment variables (see [config_environment.md](config_environment.md)).

## Startup Sequence

### Server Initialization

**Main Entry** [src/mcps/__init__.py:15-30](../src/mcps/__init__.py#L15-L30):
1. Create configuration via `mcps.config.create_config()`
2. Create server via `mcps.server.create_server(config)`
3. Setup logging via `setup_logging()`
4. Run server asynchronously via `asyncio.run(server.start())`

**Server Factory** [src/mcps/server.py:75-87](../src/mcps/server.py#L75-L87):
1. Instantiate DevAutomationServer
2. Register all tools and resources
3. Return configured server instance

**Server Start** [src/mcps/server.py:70-72](../src/mcps/server.py#L70-L72):
- Uses async context manager for vault lifecycle
- Vault initialized and updated on startup
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

No containerization currently implemented. Deployment as Python package in host environment or virtual environment.

## Logging Configuration

**Log Setup** [src/mcps/logs.py:6-44](../src/mcps/logs.py#L6-L44):
- Removes console handlers including Rich formatters
- Configures rotating file handler
- Reduces httpx logging noise to WARNING level
- Log format includes timestamp, level, module, and message
