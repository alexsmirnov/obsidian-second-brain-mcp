# mcps/config.py
import logging
import os
from pathlib import Path
from typing import Annotated, Any, Literal, Self
from urllib.parse import urlparse

from dotenv import load_dotenv
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

__all__ = [
    "ENV_GROUPS",
    "BrowserSettings",
    "ChunkingSettings",
    "FetchSettings",
    "GoogleSettings",
    "RagSettings",
    "ResearchLimits",
    "ResearchSettings",
    "RouterSettings",
    "ScraperSettings",
    "SearchSettings",
    "ServerConfig",
    "VaultSettings",
    "create_config",
]

logger = logging.getLogger("mcps.config")

DOTENV_PATHS: tuple[Path, ...] = (
    Path(__file__).parents[2] / ".env",
    Path.home() / ".env",
)


def _is_valid_router_url(url: str) -> bool:
    parsed = urlparse(url)
    try:
        _ = parsed.port
    except ValueError:
        return False
    return (
        parsed.scheme in ("http", "https")
        and bool(parsed.hostname)
        and not parsed.username
        and not parsed.password
        and not parsed.query
        and not parsed.fragment
    )


class _EnvGroup(BaseSettings):
    """Base of every settings group: frozen, strict, empty env value = default."""

    model_config = SettingsConfigDict(
        env_ignore_empty=True,
        allow_inf_nan=False,
        extra="forbid",
        frozen=True,
    )


class RouterSettings(_EnvGroup):
    """OpenAI-compatible model router shared by every AI feature."""

    model_config = SettingsConfigDict(env_prefix="ROUTER_")

    api_base: str = ""
    # Used instead of ``api_base`` when running inside a Docker container.
    docker_api_base: str = ""
    api_key: str = ""

    @model_validator(mode="before")
    @classmethod
    def _prefer_docker_api_base(cls, data: Any) -> Any:
        if (
            isinstance(data, dict)
            and data.get("docker_api_base")
            and os.path.isfile("/.dockerenv")
        ):
            return {**data, "api_base": data["docker_api_base"]}
        return data

    @field_validator("api_base")
    @classmethod
    def _valid_router_url(cls, value: str) -> str:
        if value and not _is_valid_router_url(value):
            raise ValueError(
                "must be an http(s) URL with a host and no credentials, "
                f"query or fragment, got {value!r}"
            )
        return value


class RagSettings(_EnvGroup):
    """Models for Obsidian vault indexing, reranking and search.

    All model values are empty by default and must be supplied through
    environment variables (see env.example).
    """

    model_config = SettingsConfigDict(env_prefix="RAG_")

    embedding_model: str = ""
    embedding_dimensions: int = Field(default=0, ge=0)
    # Rerank model used by ProxyReranker.
    reranker_model: str = ""
    # Embeddings and inference models used by LlmReranker.
    reranker_embedding_model: str = ""
    reranker_embedding_dimensions: int = Field(default=0, ge=0)
    reranker_infer_model: str = ""
    infer_model: str = ""
    summary_model: str = ""

    @model_validator(mode="after")
    def _embedding_model_needs_dimensions(self) -> Self:
        if self.embedding_model and self.embedding_dimensions < 1:
            raise ValueError(
                "RAG_EMBEDDING_DIMENSIONS must match RAG_EMBEDDING_MODEL "
                "(LanceDB schema creation needs it)"
            )
        return self


class VaultSettings(_EnvGroup):
    """Obsidian vault location and agent memory storage."""

    model_config = SettingsConfigDict(env_prefix="VAULT_")

    dir: Path | None = Field(default=None, alias="VAULT")
    # Vault-relative folder where agent memories are stored.
    memory_dir: str = Field(default="Memory", alias="MEMORY_DIR")
    table_name: str = "documents"


class ChunkingSettings(_EnvGroup):
    """Semantic chunk size bounds, in characters."""

    min_size: int = Field(default=500, gt=0, alias="MIN_CHUNK_SIZE")
    max_size: int = Field(default=1000, gt=0, alias="MAX_CHUNK_SIZE")

    @model_validator(mode="after")
    def _min_not_above_max(self) -> Self:
        if self.min_size > self.max_size:
            raise ValueError(
                f"MIN_CHUNK_SIZE ({self.min_size}) must not exceed "
                f"MAX_CHUNK_SIZE ({self.max_size})"
            )
        return self


class SearchSettings(_EnvGroup):
    """Vault search result limits."""

    model_config = SettingsConfigDict(env_prefix="SEARCH_")

    limit: int = Field(default=30, ge=1)
    # Minimum reranker relevance score for a result to be returned. 0.0 keeps
    # all results: reranker score scales differ (RRF ~0.016, LLM/proxy
    # rerankers 0..1), so any positive default would silently drop all
    # RRF-scored results.
    min_score: float = Field(default=0.0, ge=0.0, le=1.0, alias="MIN_SCORE")


class GoogleSettings(_EnvGroup):
    """Google Custom Search credentials; Yahoo is used when absent."""

    model_config = SettingsConfigDict(env_prefix="GOOGLE_")

    api_key: str = ""
    search_id: str = ""


class BrowserSettings(_EnvGroup):
    """CDP browser pool, reconnect and cleanup settings."""

    model_config = SettingsConfigDict(env_prefix="BROWSER_")

    # CDP endpoint used to render generic pages; empty or unreachable disables
    # web_research.
    cdp_url: str = ""
    # Named tabs reused across fetches; also the cap on parallel renders.
    pool_size: int = Field(default=8, ge=1)
    # Attempts to start a replacement crawler after the connection is lost.
    reconnect_attempts: int = Field(default=5, ge=1)
    # Pause before each reconnect attempt.
    reconnect_delay_seconds: float = Field(default=2.0, ge=0)
    # Time limit for connecting one crawler to the CDP endpoint.
    connect_timeout_seconds: float = Field(default=30.0, gt=0)
    # Extra tries of a fetch whose browser connection was lost.
    fetch_retries: int = Field(default=1, ge=0)
    # Time limit for resetting a tab or closing a crawler.
    cleanup_timeout_seconds: float = Field(default=5.0, gt=0)


class ScraperSettings(_EnvGroup):
    """Commercial unblocking provider for blocked or unrenderable pages."""

    provider: Literal["", "scrape_do", "bright_data"] = Field(
        default="", alias="SCRAPER_PROVIDER"
    )
    scrape_do_token: str = Field(default="", alias="SCRAPE_DO_TOKEN")
    bright_data_api_key: str = Field(default="", alias="BRIGHT_DATA_API_KEY")
    bright_data_zone: str = Field(default="", alias="BRIGHT_DATA_ZONE")

    @model_validator(mode="after")
    def _provider_has_credentials(self) -> Self:
        if self.provider == "scrape_do" and not self.scrape_do_token:
            raise ValueError("SCRAPER_PROVIDER=scrape_do requires SCRAPE_DO_TOKEN")
        if self.provider == "bright_data" and not (
            self.bright_data_api_key and self.bright_data_zone
        ):
            raise ValueError(
                "SCRAPER_PROVIDER=bright_data requires BRIGHT_DATA_API_KEY "
                "and BRIGHT_DATA_ZONE"
            )
        return self


class FetchSettings(_EnvGroup):
    """Content filtering and fetch routing."""

    # Router decision model for per-fragment relevance. Empty selects local
    # lexical filtering.
    decision_model: str = Field(default="", alias="DECISION_MODEL")
    # Filters and cleans fetch results.
    fetch_model: str = Field(default="", alias="FETCH_MODEL")
    # Hostnames (and their subdomains) fetch must never visit.
    restricted_domains: Annotated[tuple[str, ...], NoDecode] = Field(
        default=(), alias="FETCH_RESTRICTED_DOMAINS"
    )

    @field_validator("restricted_domains", mode="before")
    @classmethod
    def _split_domains(cls, value: Any) -> tuple[str, ...]:
        parts = value.split(",") if isinstance(value, str) else value
        return tuple(part.strip().lower() for part in parts if part.strip())


class ResearchLimits(_EnvGroup):
    """Tunable bounds for one deep research run."""

    model_config = SettingsConfigDict(env_prefix="RESEARCH_")

    # Seconds since start; no new round begins after it.
    round_start_budget: float = Field(default=500.0, gt=0)
    max_pages_per_round: int = Field(default=10, ge=1)
    max_initial_queries: int = Field(default=3, ge=1)
    # Independent limit for search_queries and for links.
    max_follow_ups: int = Field(default=5, ge=1)
    # First call plus re-prompts on invalid structured output.
    max_parse_attempts: int = Field(default=3, ge=1)


class ResearchSettings(_EnvGroup):
    """Web research models and background job timing."""

    model_config = SettingsConfigDict(env_prefix="RESEARCH_")

    # Generates queries and cleans fetch results.
    fast_model: str = ""
    # Reflection and final result generation.
    infer_model: str = ""
    # web_research waits this long for a job before returning its name; keep it
    # below the MCP client timeout (30 s by default).
    wait_seconds: float = Field(default=25.0, gt=0)
    # Finished research jobs are dropped this long after their last update.
    job_ttl_seconds: float = Field(default=3600.0, gt=0)


ENV_GROUPS: tuple[type[BaseSettings], ...] = (
    RouterSettings,
    RagSettings,
    VaultSettings,
    ChunkingSettings,
    SearchSettings,
    GoogleSettings,
    BrowserSettings,
    ScraperSettings,
    FetchSettings,
    ResearchLimits,
    ResearchSettings,
)


class ServerConfig(BaseModel):
    """Server configuration: one validated settings group per feature."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    router: RouterSettings = Field(default_factory=RouterSettings)
    rag: RagSettings = Field(default_factory=RagSettings)
    vault: VaultSettings = Field(default_factory=VaultSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    search: SearchSettings = Field(default_factory=SearchSettings)
    google: GoogleSettings = Field(default_factory=GoogleSettings)
    browser: BrowserSettings = Field(default_factory=BrowserSettings)
    scraper: ScraperSettings = Field(default_factory=ScraperSettings)
    fetch: FetchSettings = Field(default_factory=FetchSettings)
    research: ResearchSettings = Field(default_factory=ResearchSettings)
    research_limits: ResearchLimits = Field(default_factory=ResearchLimits)

    @model_validator(mode="after")
    def _warn_about_models_without_router(self) -> Self:
        """Warn about feature combinations that cannot work at runtime.

        Non-fatal so the server can start with a subset of features enabled
        (e.g. no vault, no AI tools).
        """
        if self.router.api_base:
            return self
        if self.research.fast_model or self.research.infer_model:
            logger.warning(
                "Web research models are configured but ROUTER_API_BASE is empty. "
                "Web research will fail until a router URL is provided."
            )
        if self.fetch.decision_model:
            logger.warning(
                "DECISION_MODEL is configured but ROUTER_API_BASE is empty. "
                "Fetch filtering will use lexical ranking."
            )
        if self.rag.embedding_model:
            logger.warning(
                "RAG embedding model is configured but ROUTER_API_BASE is empty. "
                "Obsidian vault indexing and search will fail until a router URL "
                "is provided."
            )
        return self


def create_config(vault_dir: Path | None = None) -> ServerConfig:
    """Load ``.env`` files, then build the config from the environment.

    ``vault_dir`` (the ``--vault`` CLI option) wins over ``VAULT``. Real
    environment variables win over ``.env`` values.
    """
    for dotenv_path in DOTENV_PATHS:
        if dotenv_path.exists():
            load_dotenv(dotenv_path)

    if vault_dir is None:
        return ServerConfig()
    return ServerConfig(vault=VaultSettings(VAULT=vault_dir))
