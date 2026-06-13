# mcps/config.py
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from mcps.rag.document_processing import default_skip_patterns


@dataclass
class ServerConfig:
    prompts_dir: Path = field(default_factory=lambda: Path(__file__).parent / "prompts")
    cache_dir: Path = field(default_factory=lambda: Path(__file__).parent / "cache")
    tests_dir: Path = field(default_factory=lambda: Path(__file__).parent / "tests")
    library_docs: dict[str, str] = field(default_factory=dict)
    project_paths: dict[str, str] = field(default_factory=dict)
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    litellm_router: str = ""
    litellm_router_key: str = ""
    # Obsidian Vault configuration
    voyage_api_key: str = ""
    ollama_api_base: str = ""
    vault_dir: Path | None = None
    table_name: str = "documents"
    skip_patterns: list[str] = field(default_factory=list)
    batch_size: int = 8
    # Chunking configuration
    max_chunk_size: int = 4000
    
    ollama_embedding_model: str = "bge-m3:latest"
    voyage_embedding_model: str = "voyage-3-lite"
    
    ollama_reranker_model: str = "phi4-mini:latest"
    voyage_reranker_model: str = "rerank-2-lite"
    
    search_limit: int = 20

def create_config(
    prompts_dir: Path = Path("./prompts"),
    cache_dir: Path = Path("./cache"),
    tests_dir: Path = Path("./tests"),
    library_docs: dict[str, str] | None = None,
    project_paths: dict[str, str] | None = None,
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
        litellm_router=os.getenv("LITELLM_ROUTER", ""),
        litellm_router_key=os.getenv("LITELLM_ROUTER_KEY", ""),
        voyage_api_key=os.getenv("VOYAGE_API_KEY", ""),
        ollama_api_base=os.getenv("OLLAMA_API_BASE", ""),
        vault_dir=Path(os.getenv("VAULT","")) if os.getenv("VAULT") else None,
        skip_patterns=default_skip_patterns
    )
