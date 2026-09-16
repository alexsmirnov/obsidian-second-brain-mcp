from __future__ import annotations

from pathlib import Path

import pytest

from mcps.config import ServerConfig, create_config, validate_config


class TestCreateConfigVaultPrecedence:
    def test_cli_vault_overrides_env_var(self, monkeypatch, tmp_path: Path):
        env_path = tmp_path / "env_vault"
        cli_path = tmp_path / "cli_vault"
        monkeypatch.setenv("VAULT", str(env_path))

        config = create_config(vault_dir=cli_path)

        assert config.vault_dir == cli_path

    def test_env_var_used_when_cli_vault_missing(self, monkeypatch, tmp_path: Path):
        env_path = tmp_path / "env_vault"
        monkeypatch.setenv("VAULT", str(env_path))

        config = create_config()

        assert config.vault_dir == env_path

    def test_vault_dir_is_none_when_cli_and_env_missing(self, monkeypatch):
        monkeypatch.setenv("VAULT", "")

        config = create_config()

        assert config.vault_dir is None


class TestCreateConfigRouter:

    def test_router_api_base_uses_host_url_outside_container(self, monkeypatch):
        monkeypatch.setenv("ROUTER_API_BASE", "http://host-router:4000")
        monkeypatch.setenv("ROUTER_DOCKER_API_BASE", "http://docker-router:4000")
        monkeypatch.setattr("mcps.config.os.path.isfile", lambda _: False)

        config = create_config()

        assert config.router_api_base == "http://host-router:4000"

    def test_router_api_base_uses_docker_url_inside_container(self, monkeypatch):
        monkeypatch.setenv("ROUTER_API_BASE", "http://host-router:4000")
        monkeypatch.setenv("ROUTER_DOCKER_API_BASE", "http://docker-router:4000")
        monkeypatch.setattr("mcps.config.os.path.isfile", lambda _: True)

        config = create_config()

        assert config.router_api_base == "http://docker-router:4000"

    def test_router_api_key_reads_env_var(self, monkeypatch):
        monkeypatch.setenv("ROUTER_API_KEY", "sk-test")

        config = create_config()

        assert config.router_api_key == "sk-test"


class TestCreateConfigReranker:
    def test_rag_reranker_infer_model_defaults_to_empty(self, monkeypatch):
        monkeypatch.delenv("RAG_RERANKER_INFER_MODEL", raising=False)

        config = create_config()

        assert config.rag_reranker_infer_model == ""

    def test_rag_reranker_infer_model_reads_env_var(self, monkeypatch):
        monkeypatch.setenv("RAG_RERANKER_INFER_MODEL", "custom-model")

        config = create_config()

        assert config.rag_reranker_infer_model == "custom-model"


class TestCreateConfigSearchInference:
    def test_rag_infer_model_defaults_to_empty(self, monkeypatch):
        monkeypatch.delenv("RAG_INFER_MODEL", raising=False)

        config = create_config()

        assert config.rag_infer_model == ""

    def test_rag_infer_model_reads_env_var(self, monkeypatch):
        monkeypatch.setenv("RAG_INFER_MODEL", "search-model")

        config = create_config()

        assert config.rag_infer_model == "search-model"


class TestCreateConfigModelDefaults:
    def test_all_model_fields_default_to_empty_or_zero(self, monkeypatch):
        for var in (
            "RAG_EMBEDDING_MODEL",
            "RAG_RERANKER_MODEL",
            "RAG_RERANKER_EMBEDDING_MODEL",
            "RAG_RERANKER_INFER_MODEL",
            "RAG_INFER_MODEL",
            "RESEARCH_FAST_MODEL",
            "RESEARCH_INFER_MODEL",
        ):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.delenv("RAG_EMBEDDING_DIMENSIONS", raising=False)
        monkeypatch.delenv("RAG_RERANKER_EMBEDDING_DIMENSIONS", raising=False)

        config = create_config()

        assert config.rag_embedding_model == ""
        assert config.rag_embedding_dimensions == 0
        assert config.rag_reranker_model == ""
        assert config.rag_reranker_embedding_model == ""
        assert config.rag_reranker_embedding_dimensions == 0
        assert config.rag_reranker_infer_model == ""
        assert config.rag_infer_model == ""
        assert config.research_fast_model == ""
        assert config.research_infer_model == ""

    def test_model_env_vars_are_read(self, monkeypatch):
        monkeypatch.setenv("RAG_EMBEDDING_MODEL", "text-embedding-3-small")
        monkeypatch.setenv("RAG_EMBEDDING_DIMENSIONS", "1536")
        monkeypatch.setenv("RAG_RERANKER_MODEL", "gpt-4o-mini")
        monkeypatch.setenv("RAG_RERANKER_EMBEDDING_MODEL", "embed-2")
        monkeypatch.setenv("RAG_RERANKER_EMBEDDING_DIMENSIONS", "768")
        monkeypatch.setenv("RAG_RERANKER_INFER_MODEL", "gemini-flash-lite")
        monkeypatch.setenv("RAG_INFER_MODEL", "gpt-4o")
        monkeypatch.setenv("RESEARCH_FAST_MODEL", "gemini-flash-lite")
        monkeypatch.setenv("RESEARCH_INFER_MODEL", "gemini-flash")

        config = create_config()

        assert config.rag_embedding_model == "text-embedding-3-small"
        assert config.rag_embedding_dimensions == 1536
        assert config.rag_reranker_model == "gpt-4o-mini"
        assert config.rag_reranker_embedding_model == "embed-2"
        assert config.rag_reranker_embedding_dimensions == 768
        assert config.rag_reranker_infer_model == "gemini-flash-lite"
        assert config.rag_infer_model == "gpt-4o"
        assert config.research_fast_model == "gemini-flash-lite"
        assert config.research_infer_model == "gemini-flash"


class TestValidateConfig:
    def test_warns_when_research_model_without_router(self, caplog):
        config = ServerConfig(
            research_fast_model="gemini-flash-lite",
            research_infer_model="gemini-flash",
        )

        with caplog.at_level("WARNING", logger="mcps.config"):
            validate_config(config)

        assert "ROUTER_API_BASE is empty" in caplog.text
        assert "Web research" in caplog.text

    def test_warns_when_rag_embedding_model_without_router(self, caplog):
        config = ServerConfig(rag_embedding_model="text-embedding-3-small")

        with caplog.at_level("WARNING", logger="mcps.config"):
            validate_config(config)

        assert "ROUTER_API_BASE is empty" in caplog.text
        assert "Obsidian vault" in caplog.text

    def test_warns_when_embedding_dimensions_missing(self, caplog):
        config = ServerConfig(
            router_api_base="http://localhost:4000",
            rag_embedding_model="text-embedding-3-small",
        )

        with caplog.at_level("WARNING", logger="mcps.config"):
            validate_config(config)

        assert "RAG_EMBEDDING_DIMENSIONS is 0" in caplog.text

    def test_no_warnings_when_ai_disabled(self, caplog):
        config = ServerConfig()

        with caplog.at_level("WARNING", logger="mcps.config"):
            validate_config(config)

        assert caplog.text == ""

    @pytest.mark.parametrize(
        "dimensions",
        [768, 1536],
    )
    def test_no_warning_when_embedding_dimensions_present(self, caplog, dimensions):
        config = ServerConfig(
            router_api_base="http://localhost:4000",
            rag_embedding_model="text-embedding-3-small",
            rag_embedding_dimensions=dimensions,
        )

        with caplog.at_level("WARNING", logger="mcps.config"):
            validate_config(config)

        assert caplog.text == ""


class TestCreateConfigChunkSize:
    def test_config_chunk_size_defaults(self, monkeypatch):
        monkeypatch.delenv("MIN_CHUNK_SIZE", raising=False)
        monkeypatch.delenv("MAX_CHUNK_SIZE", raising=False)

        config = create_config()

        assert config.min_chunk_size == 1000
        assert config.max_chunk_size == 2000

    def test_config_reads_chunk_size_env_vars(self, monkeypatch):
        monkeypatch.setenv("MIN_CHUNK_SIZE", "800")
        monkeypatch.setenv("MAX_CHUNK_SIZE", "1600")

        config = create_config()

        assert config.min_chunk_size == 800
        assert config.max_chunk_size == 1600

    def test_config_empty_env_vars_fallback_to_defaults(self, monkeypatch):
        monkeypatch.setenv("MIN_CHUNK_SIZE", "")
        monkeypatch.setenv("MAX_CHUNK_SIZE", "")

        config = create_config()

        assert config.min_chunk_size == 1000
        assert config.max_chunk_size == 2000

    def test_validate_config_warns_when_min_chunk_size_greater_than_max(self, caplog):
        config = ServerConfig(min_chunk_size=2500, max_chunk_size=1000)

        with caplog.at_level("WARNING", logger="mcps.config"):
            validate_config(config)

        assert "min_chunk_size" in caplog.text
        assert "greater than max_chunk_size" in caplog.text

    def test_validate_config_warns_when_chunk_size_non_positive(self, caplog):
        config = ServerConfig(min_chunk_size=0, max_chunk_size=1000)

        with caplog.at_level("WARNING", logger="mcps.config"):
            validate_config(config)

        assert "must be positive integers" in caplog.text


class TestCreateConfigMinScore:
    def test_min_score_defaults_to_zero(self, monkeypatch):
        monkeypatch.delenv("MIN_SCORE", raising=False)

        config = create_config()

        assert config.min_score == 0.0

    def test_min_score_reads_env_var(self, monkeypatch):
        monkeypatch.setenv("MIN_SCORE", "0.35")

        config = create_config()

        assert config.min_score == 0.35

    def test_empty_min_score_env_var_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("MIN_SCORE", "")

        config = create_config()

        assert config.min_score == 0.0

    def test_validate_config_warns_when_min_score_out_of_range(self, caplog):
        config = ServerConfig(min_score=1.5)

        with caplog.at_level("WARNING", logger="mcps.config"):
            validate_config(config)

        assert "MIN_SCORE" in caplog.text

