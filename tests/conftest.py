"""Session-wide test setup."""

from __future__ import annotations

import os
from pathlib import Path

import nltk
import pytest

# NLTK resource path -> downloader package name
_NLTK_RESOURCES = {
    "corpora/stopwords": "stopwords",
    "tokenizers/punkt_tab": "punkt_tab",
}


def _nltk_data_dir() -> Path:
    """``$NLTK_DATA``, else ``${XDG_CACHE_HOME:-$HOME}/nltk_data``."""
    if configured := os.environ.get("NLTK_DATA"):
        return Path(configured.split(os.pathsep)[0])
    cache = os.environ.get("XDG_CACHE_HOME") or Path.home()
    return Path(cache) / "nltk_data"


def _missing_nltk_resources() -> list[str]:
    missing = []
    for resource in _NLTK_RESOURCES:
        try:
            nltk.data.find(resource)
        except LookupError:
            missing.append(resource)
    return missing


def _ensure_nltk_resources() -> list[str]:
    """Download missing NLTK resources; return the ones still missing."""
    data_dir = _nltk_data_dir()
    if str(data_dir) not in nltk.data.path:
        nltk.data.path.append(str(data_dir))
    for resource in _missing_nltk_resources():
        nltk.download(_NLTK_RESOURCES[resource], download_dir=str(data_dir), quiet=True)
    return _missing_nltk_resources()


def _config_env_names() -> list[str]:
    """Every environment variable name the config schema reads."""
    from mcps.config import ENV_GROUPS

    names: list[str] = []
    for group in ENV_GROUPS:
        prefix = group.model_config.get("env_prefix", "")
        for name, field in group.model_fields.items():
            alias = field.validation_alias
            names.append(alias if isinstance(alias, str) else f"{prefix}{name}")
    return [name.upper() for name in names]


@pytest.fixture(autouse=True)
def clean_config_env(request, monkeypatch):
    """Hide the developer's environment and ``.env`` from config tests.

    Tests marked ``real_env`` read the real environment and are left alone.
    """
    if request.node.get_closest_marker("real_env"):
        return
    for name in _config_env_names():
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("mcps.config.load_dotenv", lambda *_args, **_kwargs: False)


if missing := _ensure_nltk_resources():
    # Fail once with a clear setup hint instead of separate failures in ranker tests.
    pytest.exit(
        f"NLTK data not found and download failed: {', '.join(missing)}.\n"
        "Download it once and point NLTK at it:\n"
        '  export NLTK_DATA="${XDG_CACHE_HOME:-$HOME}/nltk_data"\n'
        '  uv run python -m nltk.downloader -d "$NLTK_DATA" stopwords punkt_tab',
        returncode=3,
    )
