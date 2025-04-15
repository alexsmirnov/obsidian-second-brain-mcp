# mcps/config.py
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict
from dotenv import load_dotenv
import os


@dataclass
class ServerConfig:
    prompts_dir: Path = field(default_factory=lambda: Path(__file__).parent / "prompts")
    cache_dir: Path = field(default_factory=lambda: Path(__file__).parent / "cache")
    tests_dir: Path = field(default_factory=lambda: Path(__file__).parent / "tests")
    library_docs: Dict[str, str] = field(default_factory=dict)
    project_paths: Dict[str, str] = field(default_factory=dict)
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    perplexity_api_key: str = ""

def create_config(
    prompts_dir: Path = Path("./prompts"),
    cache_dir: Path = Path("./cache"),
    tests_dir: Path = Path("./tests"),
    library_docs: Dict[str, str] | None = None,
    project_paths: Dict[str, str] | None = None,
) -> ServerConfig:
    """
    Creates a ServerConfig instance, ensuring directories exist and
    handling default values for library_docs and project_paths.
    """
    # Load environment variables from .env files
    for env_path in [
        Path(__file__).parent.parent.parent,
        Path.home()
    ]:
        dotenv_path = env_path / ".env"
        if dotenv_path.exists():
            load_dotenv(dotenv_path)

    # Use provided dictionaries or default to empty dictionaries
    library_docs = library_docs if library_docs is not None else {}
    project_paths = project_paths if project_paths is not None else {}

    return ServerConfig(
        prompts_dir=prompts_dir,
        cache_dir=cache_dir,
        tests_dir=tests_dir,
        library_docs=library_docs,
        project_paths=project_paths,
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
        perplexity_api_key=os.getenv("PERPLEXITY_API_KEY", ""),
    )
