from __future__ import annotations

from pathlib import Path

from mcps.config import create_config


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


class TestCreateConfigReranker:
    def test_rag_reranker_infer_model_defaults_to_gemini_flash_lite(
        self, monkeypatch
    ):
        monkeypatch.delenv("RAG_RERANKER_INFER_MODEL", raising=False)

        config = create_config()

        assert config.rag_reranker_infer_model == "gemini-flash-lite"

    def test_rag_reranker_infer_model_reads_env_var(self, monkeypatch):
        monkeypatch.setenv("RAG_RERANKER_INFER_MODEL", "")

        config = create_config()

        assert config.rag_reranker_infer_model == ""

    def test_rag_reranker_infer_model_env_override(self, monkeypatch):
        monkeypatch.setenv("RAG_RERANKER_INFER_MODEL", "custom-model")

        config = create_config()

        assert config.rag_reranker_infer_model == "custom-model"
