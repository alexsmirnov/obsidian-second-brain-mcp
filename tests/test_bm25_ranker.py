"""Contract tests for the lexical BM25L ranker."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from mcps.research.tools.bm25_ranker import Bm25Ranker


@pytest.mark.parametrize("construct", [False, True], ids=["import", "construct"])
def test_ranker_without_nltk_data_imports_but_construction_fails(construct: bool):
    # Arrange: a fresh process cannot reuse already loaded NLTK corpora.
    script = textwrap.dedent(
        f"""\
        import importlib
        from unittest.mock import patch

        import nltk

        nltk.data.path = []
        with patch("nltk.download", side_effect=AssertionError("Download attempted")):
            module = importlib.import_module("mcps.research.tools.bm25_ranker")
            if {construct!r}:
                try:
                    module.Bm25Ranker()
                except LookupError as error:
                    assert "stopwords" in str(error)
                else:
                    raise AssertionError("Construction must require stopwords")
        """
    )

    # Act
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    # Assert
    assert result.returncode == 0, result.stderr

FRAGMENTS = [
    "Bread flour needs water and time.",
    "Quantum routing reduces latency in networks.",
    "Nothing relevant here at all.",
    "Quantum computers use routing_table identifiers.",
]


async def test_bm25_ranker_returns_only_fragments_overlapping_the_query():
    indices = await Bm25Ranker()(FRAGMENTS, "quantum routing")

    assert set(indices) == {1, 3}


async def test_bm25_ranker_orders_by_descending_score():
    indices = await Bm25Ranker()(FRAGMENTS, "quantum routing latency")

    assert indices[0] == 1


async def test_bm25_ranker_stopword_only_query_selects_nothing():
    assert await Bm25Ranker()(FRAGMENTS, "the and of") == []


async def test_bm25_ranker_identifiers_are_searchable():
    assert await Bm25Ranker()(FRAGMENTS, "routing_table") == [3]


def test_lexical_scores_returns_one_score_per_fragment_zero_without_overlap():
    scores = Bm25Ranker().lexical_scores(FRAGMENTS, "quantum")

    assert len(scores) == len(FRAGMENTS)
    assert scores[0] == 0.0 and scores[2] == 0.0
    assert scores[1] > 0.0 and scores[3] > 0.0
