"""Session-wide test setup."""

from __future__ import annotations

import nltk
import pytest

_NLTK_RESOURCES = ("corpora/stopwords", "tokenizers/punkt_tab")


def _missing_nltk_resources() -> list[str]:
    missing = []
    for resource in _NLTK_RESOURCES:
        try:
            nltk.data.find(resource)
        except LookupError:
            missing.append(resource)
    return missing


if missing := _missing_nltk_resources():
    # Fail once with a clear setup hint instead of separate failures in ranker tests.
    pytest.exit(
        f"NLTK data not found: {', '.join(missing)}.\n"
        "Download it once and point NLTK at it:\n"
        '  export NLTK_DATA="${XDG_CACHE_HOME:-$HOME/.cache}/nltk_data"\n'
        '  uv run python -m nltk.downloader -d "$NLTK_DATA" stopwords punkt_tab',
        returncode=3,
    )
