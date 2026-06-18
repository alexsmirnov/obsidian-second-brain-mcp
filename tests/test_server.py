from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from mcps.config import ServerConfig
from mcps.server import create_server, main, parse_args


class TestParseArgs:
    def test_parse_args_returns_vault_path_when_provided(self, tmp_path: Path):
        result = parse_args(["--vault", str(tmp_path)])

        assert result.vault == tmp_path

    def test_parse_args_returns_none_vault_when_not_provided(self):
        result = parse_args([])

        assert result.vault is None

    def test_parse_args_sets_reindex_flag(self):
        result = parse_args(["--reindex"])

        assert result.reindex is True


class TestMain:
    @patch("mcps.server.create_config")
    @patch("mcps.server.httpx.AsyncClient")
    @patch("mcps.server.create_vault")
    def test_main_reindex_runs_update_index_and_returns_zero(
        self,
        mock_create_vault,
        mock_async_client,
        mock_create_config,
        tmp_path: Path,
    ):
        mock_create_config.return_value = ServerConfig(vault_dir=tmp_path)
        mock_async_client.return_value.__aenter__ = AsyncMock(
            return_value=AsyncMock()
        )
        mock_async_client.return_value.__aexit__ = AsyncMock(return_value=False)
        fake_vault = AsyncMock()
        mock_create_vault.return_value.__aenter__ = AsyncMock(
            return_value=fake_vault
        )
        mock_create_vault.return_value.__aexit__ = AsyncMock(return_value=False)

        result = main(["--vault", str(tmp_path), "--reindex"])

        assert result == 0
        fake_vault.update_index.assert_awaited_once()

    @patch("mcps.server.create_config")
    def test_main_reindex_returns_nonzero_for_missing_vault_dir(
        self,
        mock_create_config,
        tmp_path: Path,
    ):
        missing_path = tmp_path / "missing"
        mock_create_config.return_value = ServerConfig(vault_dir=missing_path)

        result = main(["--reindex"])

        assert result != 0

    @patch("mcps.server.setup_logging")
    @patch("mcps.server.create_config")
    @patch("mcps.server.create_server")
    def test_main_server_mode_creates_and_starts_server(
        self,
        mock_create_server,
        mock_create_config,
        mock_setup_logging,
    ):
        config = ServerConfig(vault_dir=None)
        mock_create_config.return_value = config
        mock_server = mock_create_server.return_value
        mock_server.start = AsyncMock()

        result = main([])

        assert result == 0
        mock_create_server.assert_called_once_with(config)
        mock_server.start.assert_awaited_once()
