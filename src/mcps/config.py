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
    litellm_router: str = ""
    litellm_router_key: str = ""
    # Obsidian Vault configuration
    vault_dir: Path | None = None
    table_name: str = "documents"
    skip_patterns: list[str] = field(default_factory=list)
    batch_size: int = 8
    # Chunking configuration
    max_chunk_size: int = 4000
    
    rag_embedding_model: str = "text-embedding-3-small"
    rag_embedding_dimensions: int = 1536
    rag_reranker_model: str = "" # Rerank model to use in ProxyReranker
    # Embeddings and inferrence models used by LlmReranker
    rag_reranker_embedding_model: str = ""
    rag_reranker_embedding_dimensions: int = 736
    rag_reranker_infer_model: str = ""
    
    search_limit: int = 20
    # Web deep research config
    google_api_key: str = ""
    google_search_id: str = ""
    research_fast_model: str = "" # used to generate queries and clean fetch results
    research_infer_model: str = "" # used for reflection and final result generation

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
        litellm_router=os.getenv("LITELLM_ROUTER", ""),
        litellm_router_key=os.getenv("LITELLM_API_KEY", ""),
        rag_embedding_model=os.getenv("RAG_EMBEDDING_MODEL", "text-embedding-3-small"),
        rag_embedding_dimensions=int(os.getenv("RAG_EMBEDDING_DIMENSIONS", "1536")),
        rag_reranker_model=os.getenv("RAG_RERANKER_MODEL", ""),
        vault_dir=Path(os.getenv("VAULT","")) if os.getenv("VAULT") else None,
        skip_patterns=default_skip_patterns,
        google_api_key=os.environ.get("GOOGLE_API_KEY",""),
        google_search_id=os.environ.get("GOOGLE_SEARCH_ID","")
    )
