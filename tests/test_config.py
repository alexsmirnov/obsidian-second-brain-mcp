from __future__ import annotations

import os
from operator import attrgetter
from pathlib import Path
from typing import Any

import dotenv
import pytest
from pydantic import ValidationError

from mcps.config import ResearchLimits, RouterSettings, ServerConfig, create_config

# (env name, attribute path, sample env value, parsed value, default, extra env)
ENV_FIELDS = [
    ("ROUTER_API_BASE", "router.api_base", "http://r:4000", "http://r:4000", "", {}),
    (
        "ROUTER_DOCKER_API_BASE",
        "router.docker_api_base",
        "http://d",
        "http://d",
        "",
        {},
    ),
    ("ROUTER_API_KEY", "router.api_key", "sk-test", "sk-test", "", {}),
    (
        "RAG_EMBEDDING_MODEL",
        "rag.embedding_model",
        "text-embedding-3-small",
        "text-embedding-3-small",
        "",
        {"RAG_EMBEDDING_DIMENSIONS": "8"},
    ),
    ("RAG_EMBEDDING_DIMENSIONS", "rag.embedding_dimensions", "1536", 1536, 0, {}),
    ("RAG_RERANKER_MODEL", "rag.reranker_model", "gpt-4o-mini", "gpt-4o-mini", "", {}),
    (
        "RAG_RERANKER_EMBEDDING_MODEL",
        "rag.reranker_embedding_model",
        "embed-2",
        "embed-2",
        "",
        {},
    ),
    (
        "RAG_RERANKER_EMBEDDING_DIMENSIONS",
        "rag.reranker_embedding_dimensions",
        "768",
        768,
        0,
        {},
    ),
    (
        "RAG_RERANKER_INFER_MODEL",
        "rag.reranker_infer_model",
        "custom-model",
        "custom-model",
        "",
        {},
    ),
    ("RAG_INFER_MODEL", "rag.infer_model", "search-model", "search-model", "", {}),
    ("RAG_SUMMARY_MODEL", "rag.summary_model", "summary", "summary", "", {}),
    ("VAULT", "vault.dir", "/tmp/vault", Path("/tmp/vault"), None, {}),
    ("MEMORY_DIR", "vault.memory_dir", "Notes/Mem", "Notes/Mem", "Memory", {}),
    ("VAULT_TABLE_NAME", "vault.table_name", "notes", "notes", "documents", {}),
    ("MIN_CHUNK_SIZE", "chunking.min_size", "800", 800, 500, {}),
    ("MAX_CHUNK_SIZE", "chunking.max_size", "1600", 1600, 1000, {}),
    ("SEARCH_LIMIT", "search.limit", "7", 7, 30, {}),
    ("MIN_SCORE", "search.min_score", "0.35", 0.35, 0.0, {}),
    ("GOOGLE_API_KEY", "google.api_key", "key", "key", "", {}),
    ("GOOGLE_SEARCH_ID", "google.search_id", "cse", "cse", "", {}),
    ("BROWSER_CDP_URL", "browser.cdp_url", "ws://b:9222", "ws://b:9222", "", {}),
    ("BROWSER_POOL_SIZE", "browser.pool_size", "5", 5, 8, {}),
    ("BROWSER_RECONNECT_ATTEMPTS", "browser.reconnect_attempts", "7", 7, 5, {}),
    (
        "BROWSER_RECONNECT_DELAY_SECONDS",
        "browser.reconnect_delay_seconds",
        "0.5",
        0.5,
        2.0,
        {},
    ),
    (
        "BROWSER_CONNECT_TIMEOUT_SECONDS",
        "browser.connect_timeout_seconds",
        "9.5",
        9.5,
        30.0,
        {},
    ),
    ("BROWSER_FETCH_RETRIES", "browser.fetch_retries", "3", 3, 1, {}),
    (
        "BROWSER_CLEANUP_TIMEOUT_SECONDS",
        "browser.cleanup_timeout_seconds",
        "1.5",
        1.5,
        5.0,
        {},
    ),
    (
        "SCRAPER_PROVIDER",
        "scraper.provider",
        "scrape_do",
        "scrape_do",
        "",
        {"SCRAPE_DO_TOKEN": "t"},
    ),
    ("SCRAPE_DO_TOKEN", "scraper.scrape_do_token", "t", "t", "", {}),
    ("BRIGHT_DATA_API_KEY", "scraper.bright_data_api_key", "k", "k", "", {}),
    ("BRIGHT_DATA_ZONE", "scraper.bright_data_zone", "z", "z", "", {}),
    ("DECISION_MODEL", "fetch.decision_model", "jev-latest", "jev-latest", "", {}),
    (
        "FETCH_RESTRICTED_DOMAINS",
        "fetch.restricted_domains",
        " A.com, b.org ,",
        ("a.com", "b.org"),
        (),
        {},
    ),
    ("RESEARCH_FAST_MODEL", "research.fast_model", "fast", "fast", "", {}),
    ("RESEARCH_INFER_MODEL", "research.infer_model", "infer", "infer", "", {}),
    ("RESEARCH_WAIT_SECONDS", "research.wait_seconds", "7", 7.0, 25.0, {}),
    ("RESEARCH_JOB_TTL_SECONDS", "research.job_ttl_seconds", "60", 60.0, 3600.0, {}),
]


@pytest.fixture
def config_warnings(caplog):
    """WARNING records logged by ``mcps.config``."""
    caplog.set_level("WARNING", logger="mcps.config")
    return lambda: [
        record
        for record in caplog.records
        if record.name == "mcps.config" and record.levelname == "WARNING"
    ]


class TestEnvFields:
    @pytest.mark.parametrize(
        ("name", "path", "sample", "expected", "default", "extra"), ENV_FIELDS
    )
    def test_env_var_is_read(
        self, monkeypatch, name, path, sample, expected, default, extra
    ):
        monkeypatch.setenv(name, sample)
        for extra_name, extra_value in extra.items():
            monkeypatch.setenv(extra_name, extra_value)

        assert attrgetter(path)(create_config()) == expected

    @pytest.mark.parametrize(
        ("name", "path", "sample", "expected", "default", "extra"), ENV_FIELDS
    )
    def test_default_when_unset(
        self, monkeypatch, name, path, sample, expected, default, extra
    ):
        assert attrgetter(path)(create_config()) == default

    @pytest.mark.parametrize(
        ("name", "path", "sample", "expected", "default", "extra"), ENV_FIELDS
    )
    def test_empty_env_value_uses_default(
        self, monkeypatch, name, path, sample, expected, default, extra
    ):
        monkeypatch.setenv(name, "")

        assert attrgetter(path)(create_config()) == default


class TestCreateConfigVaultPrecedence:
    def test_cli_vault_overrides_env_var(self, monkeypatch, tmp_path: Path):
        env_path = tmp_path / "env_vault"
        cli_path = tmp_path / "cli_vault"
        monkeypatch.setenv("VAULT", str(env_path))

        config = create_config(vault_dir=cli_path)

        assert config.vault.dir == cli_path

    def test_env_var_used_when_cli_vault_missing(self, monkeypatch, tmp_path: Path):
        env_path = tmp_path / "env_vault"
        monkeypatch.setenv("VAULT", str(env_path))

        config = create_config()

        assert config.vault.dir == env_path

    def test_vault_dir_is_none_when_cli_and_env_missing(self, monkeypatch):
        monkeypatch.setenv("VAULT", "")

        config = create_config()

        assert config.vault.dir is None


class TestCreateConfigRouter:
    def test_router_api_base_uses_host_url_outside_container(self, monkeypatch):
        monkeypatch.setenv("ROUTER_API_BASE", "http://host-router:4000")
        monkeypatch.setenv("ROUTER_DOCKER_API_BASE", "http://docker-router:4000")
        monkeypatch.setattr("mcps.config.os.path.isfile", lambda _: False)

        config = create_config()

        assert config.router.api_base == "http://host-router:4000"

    def test_router_api_base_uses_docker_url_inside_container(self, monkeypatch):
        monkeypatch.setenv("ROUTER_API_BASE", "http://host-router:4000")
        monkeypatch.setenv("ROUTER_DOCKER_API_BASE", "http://docker-router:4000")
        monkeypatch.setattr("mcps.config.os.path.isfile", lambda _: True)

        config = create_config()

        assert config.router.api_base == "http://docker-router:4000"

    def test_router_init_value_kept_inside_docker(self, monkeypatch):
        monkeypatch.setattr("mcps.config.os.path.isfile", lambda _: True)

        assert RouterSettings(api_base="http://init").api_base == "http://init"

        monkeypatch.setenv("ROUTER_DOCKER_API_BASE", "http://d")

        assert RouterSettings().api_base == "http://d"


    def test_router_uses_api_base_inside_container_without_docker_variable(
        self, monkeypatch
    ):
        monkeypatch.setenv("ROUTER_API_BASE", "http://host-router:4000")
        monkeypatch.setattr("mcps.config.os.path.isfile", lambda _: True)

        assert create_config().router.api_base == "http://host-router:4000"

    def test_unused_docker_router_value_is_not_validated_on_host(self, monkeypatch):
        monkeypatch.setenv("ROUTER_DOCKER_API_BASE", "ftp://docker-router")
        monkeypatch.setattr("mcps.config.os.path.isfile", lambda _: False)

        assert create_config().router.api_base == ""


class TestUndocumentedEnvNames:
    @pytest.mark.parametrize(
        ("name", "value"),
        [
            ("PROVIDER", "aws"),
            ("MIN_SIZE", "900"),
            ("MAX_SIZE", "950"),
            ("VAULT_DIR", "/elsewhere"),
            ("VAULT_MEMORY_DIR", "Elsewhere"),
            ("SEARCH_MIN_SCORE", "0.9"),
            ("RESTRICTED_DOMAINS", "a.com"),
        ],
    )
    def test_field_name_variables_are_ignored(self, monkeypatch, name, value):
        monkeypatch.setenv(name, value)

        config = create_config()

        assert (
            config.scraper.provider,
            config.chunking.min_size,
            config.chunking.max_size,
            config.vault.dir,
            config.vault.memory_dir,
            config.search.min_score,
            config.fetch.restricted_domains,
        ) == ("", 500, 1000, None, "Memory", 0.0, ())


class TestInvalidValues:
    @pytest.mark.parametrize(
        ("name", "value"),
        [
            ("BROWSER_POOL_SIZE", "0"),
            ("BROWSER_RECONNECT_ATTEMPTS", "0"),
            ("BROWSER_RECONNECT_DELAY_SECONDS", "-1"),
            ("BROWSER_CONNECT_TIMEOUT_SECONDS", "0"),
            ("BROWSER_FETCH_RETRIES", "-1"),
            ("BROWSER_CLEANUP_TIMEOUT_SECONDS", "0"),
            ("MIN_CHUNK_SIZE", "0"),
            ("MAX_CHUNK_SIZE", "0"),
            ("MIN_SCORE", "1.5"),
            ("MIN_SCORE", "-0.1"),
            ("SEARCH_LIMIT", "0"),
            ("SCRAPER_PROVIDER", "foo"),
            ("RAG_EMBEDDING_DIMENSIONS", "x"),
            ("RAG_EMBEDDING_DIMENSIONS", "-1"),
            ("ROUTER_API_BASE", "ftp://h"),
            ("ROUTER_API_BASE", "http://u:p@h"),
            ("ROUTER_API_BASE", "http://h:abc"),
            ("RESEARCH_WAIT_SECONDS", "inf"),
            ("BROWSER_RECONNECT_DELAY_SECONDS", "inf"),
            ("BROWSER_CONNECT_TIMEOUT_SECONDS", "nan"),
            ("RESEARCH_WAIT_SECONDS", "0"),
            ("RESEARCH_JOB_TTL_SECONDS", "0"),
            ("RESEARCH_ROUND_START_BUDGET", "0"),
            ("RESEARCH_MAX_PAGES_PER_ROUND", "0"),
            ("RESEARCH_MAX_INITIAL_QUERIES", "0"),
            ("RESEARCH_MAX_FOLLOW_UPS", "0"),
            ("RESEARCH_MAX_PARSE_ATTEMPTS", "0"),
        ],
    )
    def test_invalid_value_raises(self, monkeypatch, name, value):
        monkeypatch.setenv(name, value)

        with pytest.raises(ValidationError):
            create_config()

    def test_min_chunk_size_above_max_raises(self, monkeypatch):
        monkeypatch.setenv("MIN_CHUNK_SIZE", "900")
        monkeypatch.setenv("MAX_CHUNK_SIZE", "800")

        with pytest.raises(ValidationError):
            create_config()

    def test_embedding_model_without_dimensions_raises(self, monkeypatch):
        monkeypatch.setenv("RAG_EMBEDDING_MODEL", "m")

        with pytest.raises(ValidationError):
            create_config()

    @pytest.mark.parametrize(
        "env",
        [
            {"SCRAPER_PROVIDER": "scrape_do"},
            {"SCRAPER_PROVIDER": "bright_data", "BRIGHT_DATA_API_KEY": "k"},
            {"SCRAPER_PROVIDER": "bright_data", "BRIGHT_DATA_ZONE": "z"},
        ],
    )
    def test_scraper_provider_without_credentials_raises(self, monkeypatch, env):
        for name, value in env.items():
            monkeypatch.setenv(name, value)

        with pytest.raises(ValidationError):
            create_config()

    def test_bright_data_with_key_and_zone_loads(self, monkeypatch):
        monkeypatch.setenv("SCRAPER_PROVIDER", "bright_data")
        monkeypatch.setenv("BRIGHT_DATA_API_KEY", "k")
        monkeypatch.setenv("BRIGHT_DATA_ZONE", "z")

        assert create_config().scraper.provider == "bright_data"

    def test_unknown_kwarg_rejected(self):
        leftover_flat_kwarg: dict[str, Any] = {"vault_dir": Path("x")}

        with pytest.raises(ValidationError):
            ServerConfig(**leftover_flat_kwarg)


class TestRouterWarnings:
    @pytest.mark.parametrize(
        "env",
        [
            {"RESEARCH_FAST_MODEL": "gemini-flash-lite"},
            {"RESEARCH_FAST_MODEL": "f", "RESEARCH_INFER_MODEL": "i"},
            {"DECISION_MODEL": "jev-latest"},
            {"RAG_EMBEDDING_MODEL": "m", "RAG_EMBEDDING_DIMENSIONS": "8"},
        ],
    )
    def test_warns_when_model_set_without_router(
        self, monkeypatch, config_warnings, env
    ):
        for name, value in env.items():
            monkeypatch.setenv(name, value)

        create_config()

        (record,) = config_warnings()
        assert "ROUTER_API_BASE" in record.getMessage()

    def test_no_warning_when_router_set(self, monkeypatch, config_warnings):
        monkeypatch.setenv("ROUTER_API_BASE", "http://localhost:4000")
        monkeypatch.setenv("RESEARCH_FAST_MODEL", "f")
        monkeypatch.setenv("DECISION_MODEL", "d")

        create_config()

        assert config_warnings() == []

    def test_no_warnings_when_ai_disabled(self, config_warnings):
        create_config()

        assert config_warnings() == []


class TestBrowserDefaults:
    def test_browser_pool_size_defaults_to_eight(self):
        assert create_config().browser.pool_size == 8


class TestResearchLimits:
    def test_research_limits_read_env(self, monkeypatch):
        monkeypatch.setenv("RESEARCH_ROUND_START_BUDGET", "60.5")
        monkeypatch.setenv("RESEARCH_MAX_PAGES_PER_ROUND", "4")
        monkeypatch.setenv("RESEARCH_MAX_INITIAL_QUERIES", "2")
        monkeypatch.setenv("RESEARCH_MAX_FOLLOW_UPS", "6")
        monkeypatch.setenv("RESEARCH_MAX_PARSE_ATTEMPTS", "5")

        assert create_config().research_limits == ResearchLimits(
            round_start_budget=60.5,
            max_pages_per_round=4,
            max_initial_queries=2,
            max_follow_ups=6,
            max_parse_attempts=5,
        )

    def test_research_limits_default(self):
        assert create_config().research_limits == ResearchLimits(
            round_start_budget=180.0,
            max_pages_per_round=10,
            max_initial_queries=3,
            max_follow_ups=5,
            max_parse_attempts=3,
        )


class TestResearchJobs:
    def test_research_job_defaults_are_sane(self):
        research = create_config().research

        # The wait must stay below the MCP client's default 30 s timeout.
        assert 0 < research.wait_seconds < 30
        assert research.job_ttl_seconds > research.wait_seconds


@pytest.fixture
def real_dotenv(monkeypatch):
    """Let ``create_config`` load real ``.env`` files, then undo the process env."""
    monkeypatch.setattr("mcps.config.load_dotenv", dotenv.load_dotenv)
    yield
    # load_dotenv writes outside monkeypatch's bookkeeping.
    for name in ("MEMORY_DIR", "SEARCH_LIMIT", "MIN_SCORE"):
        os.environ.pop(name, None)


class TestDotenv:
    def test_real_env_beats_dotenv(self, monkeypatch, tmp_path: Path, real_dotenv):
        dotenv_file = tmp_path / ".env"
        dotenv_file.write_text("MEMORY_DIR=FromFile\nSEARCH_LIMIT=7\n")
        monkeypatch.setattr("mcps.config.DOTENV_PATHS", (dotenv_file,))
        monkeypatch.setenv("MEMORY_DIR", "FromEnv")

        config = create_config()

        assert config.vault.memory_dir == "FromEnv"
        assert config.search.limit == 7

    def test_project_dotenv_beats_home_dotenv(
        self, monkeypatch, tmp_path: Path, real_dotenv
    ):
        project_file = tmp_path / "project.env"
        project_file.write_text("MEMORY_DIR=Project\nSEARCH_LIMIT=7\n")
        home_file = tmp_path / "home.env"
        home_file.write_text("MEMORY_DIR=Home\nSEARCH_LIMIT=9\nMIN_SCORE=0.5\n")
        monkeypatch.setattr("mcps.config.DOTENV_PATHS", (project_file, home_file))

        config = create_config()

        assert config.vault.memory_dir == "Project"
        assert config.search.limit == 7
        assert config.search.min_score == 0.5
