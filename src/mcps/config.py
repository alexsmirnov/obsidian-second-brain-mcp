# mcps/config.py
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from mcps.rag.document_processing import default_skip_patterns

logger = logging.getLogger("mcps.config")


@dataclass
class ServerConfig:
    prompts_dir: Path = field(default_factory=lambda: Path(__file__).parent / "prompts")
    cache_dir: Path = field(default_factory=lambda: Path(__file__).parent / "cache")
    tests_dir: Path = field(default_factory=lambda: Path(__file__).parent / "tests")
    library_docs: dict[str, str] = field(default_factory=dict)
    project_paths: dict[str, str] = field(default_factory=dict)
    router_api_base: str = ""
    router_api_key: str = ""
    # Obsidian Vault configuration
    vault_dir: Path | None = None
    table_name: str = "documents"
    skip_patterns: list[str] = field(default_factory=list)
    batch_size: int = 8
    # Chunking configuration
    min_chunk_size: int = 500
    max_chunk_size: int = 1000

    # RAG models. All model values are intentionally empty by default and must
    # be supplied through environment variables (see env.example).
    rag_embedding_model: str = ""
    rag_embedding_dimensions: int = 0
    rag_reranker_model: str = ""  # Rerank model to use in ProxyReranker
    # Embeddings and inference models used by LlmReranker
    rag_reranker_embedding_model: str = ""
    rag_reranker_embedding_dimensions: int = 0
    rag_reranker_infer_model: str = ""
    rag_infer_model: str = ""
    rag_summary_model: str = ""

    search_limit: int = 30
    # Minimum reranker relevance score for a search result to be returned.
    # 0.0 keeps all results: reranker score scales differ (RRF ~0.016,
    # LLM/proxy rerankers 0..1), so any positive default would silently
    # drop all RRF-scored results.
    min_score: float = 0.0
    # Web deep research config
    google_api_key: str = ""
    google_search_id: str = ""
    # used to generate queries and clean fetch results
    research_fast_model: str = ""
    # used for reflection and final result generation
    research_infer_model: str = ""

def create_config(
    prompts_dir: Path = Path("./prompts"),
    cache_dir: Path = Path("./cache"),
    tests_dir: Path = Path("./tests"),
    library_docs: dict[str, str] | None = None,
    project_paths: dict[str, str] | None = None,
    vault_dir: Path | None = None,
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

    config = ServerConfig(
        prompts_dir=prompts_dir,
        cache_dir=cache_dir,
        tests_dir=tests_dir,
        library_docs=library_docs,
        project_paths=project_paths,
        router_api_base=os.getenv(
            (
                "ROUTER_DOCKER_API_BASE"
                if os.path.isfile("/.dockerenv")
                else "ROUTER_API_BASE"
            ),
            "",
        ),
        router_api_key=os.getenv("ROUTER_API_KEY", ""),
        rag_embedding_model=os.getenv("RAG_EMBEDDING_MODEL", ""),
        rag_embedding_dimensions=int(os.getenv("RAG_EMBEDDING_DIMENSIONS", "0")),
        rag_reranker_model=os.getenv("RAG_RERANKER_MODEL", ""),
        rag_reranker_embedding_model=os.getenv("RAG_RERANKER_EMBEDDING_MODEL", ""),
        rag_reranker_embedding_dimensions=int(
            os.getenv("RAG_RERANKER_EMBEDDING_DIMENSIONS", "0")
        ),
        rag_reranker_infer_model=os.getenv("RAG_RERANKER_INFER_MODEL", ""),
        rag_infer_model=os.getenv("RAG_INFER_MODEL", ""),
        rag_summary_model=os.getenv("RAG_SUMMARY_MODEL", ""),
        research_fast_model=os.getenv("RESEARCH_FAST_MODEL", ""),
        research_infer_model=os.getenv("RESEARCH_INFER_MODEL", ""),
        vault_dir=(
            vault_dir
            if vault_dir is not None
            else (Path(env_vault) if (env_vault := os.getenv("VAULT")) else None)
        ),
        skip_patterns=default_skip_patterns,
        min_chunk_size=int(os.environ.get("MIN_CHUNK_SIZE") or 500),
        max_chunk_size=int(os.environ.get("MAX_CHUNK_SIZE") or 1000),
        min_score=float(os.environ.get("MIN_SCORE") or 0.0),
        google_api_key=os.environ.get("GOOGLE_API_KEY", ""),
        google_search_id=os.environ.get("GOOGLE_SEARCH_ID", "")
    )
    validate_config(config)
    return config


def validate_config(config: ServerConfig) -> None:
    """Warn about configuration combinations that cannot work at runtime.

    Validation is intentionally non-fatal so the server can start with a
    subset of features enabled (e.g. no vault, no AI tools).
    """
    if config.min_chunk_size <= 0 or config.max_chunk_size <= 0:
        logger.warning(
            "min_chunk_size (%d) and max_chunk_size (%d) must be positive integers.",
            config.min_chunk_size,
            config.max_chunk_size,
        )
    elif config.min_chunk_size > config.max_chunk_size:
        logger.warning(
            "min_chunk_size (%d) is greater than max_chunk_size (%d). "
            "Undersized sections will be merged up to max_chunk_size.",
            config.min_chunk_size,
            config.max_chunk_size,
        )

    if not 0.0 <= config.min_score <= 1.0:
        logger.warning(
            "min_score (%.3f) is outside the reranker score range [0, 1]. "
            "Search will drop all or no results; check MIN_SCORE.",
            config.min_score,
        )

    if not config.router_api_base:
        if config.research_fast_model or config.research_infer_model:
            logger.warning(
                "Web research models are configured but ROUTER_API_BASE is empty. "
                "Web research will fail until a router URL is provided."
            )
        if config.rag_embedding_model:
            logger.warning(
                "RAG embedding model is configured but ROUTER_API_BASE is empty. "
                "Obsidian vault indexing and search will fail until a router URL "
                "is provided."
            )

    if config.rag_embedding_model and not config.rag_embedding_dimensions:
        logger.warning(
            "RAG_EMBEDDING_MODEL is set but RAG_EMBEDDING_DIMENSIONS is 0. "
            "LanceDB schema creation will fail; set the dimension to match "
            "the embedding model."
        )
