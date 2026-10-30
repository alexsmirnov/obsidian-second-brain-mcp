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
    indices = await Bm25Ranker(threshold=0.5)(FRAGMENTS, "quantum routing")

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


async def test_bm25_ranker_threshold_above_all_scores_selects_nothing():
    assert await Bm25Ranker(threshold=1000.0)(FRAGMENTS, "quantum routing") == []


async def test_bm25_ranker_zero_threshold_keeps_every_overlapping_fragment():
    indices = await Bm25Ranker(threshold=0.0)(FRAGMENTS, "quantum")

    assert set(indices) == {1, 3}


def test_bm25_ranker_exposes_threshold():
    assert Bm25Ranker(threshold=2.5).threshold == 2.5


async def test_bm25_ranker_relative_cutoff_keeps_only_fragments_near_the_best_score():
    # Arrange
    ranker = Bm25Ranker(threshold=0.0, relative_cutoff=0.9)

    # Act
    indices = await ranker(FRAGMENTS, "quantum routing latency")

    # Assert
    assert indices == [1]


async def test_bm25_ranker_link_penalty_drops_citation_dense_fragment():
    # Arrange
    fragments = [
        "Quantum routing reduces latency in networks.",
        "[Quantum routing](https://doi.org/10.1/qr) (2020) https://example.org",
    ]
    ranker = Bm25Ranker(threshold=0.0, link_penalty=10.0)

    # Act
    indices = await ranker(fragments, "quantum routing")

    # Assert
    assert indices == [0]


async def test_bm25_ranker_smoothing_keeps_fragment_between_strong_neighbours():
    # Arrange
    fragments = [
        "Quantum routing reduces latency in networks.",
        "Nothing relevant here at all.",
        "Quantum routing lowers latency again.",
    ]
    ranker = Bm25Ranker(threshold=0.0, smoothing=0.2)

    # Act
    indices = await ranker(fragments, "quantum routing latency")

    # Assert
    assert indices == [0, 1, 2]


async def test_bm25_ranker_smoothing_selects_nothing_without_query_overlap():
    ranker = Bm25Ranker(threshold=0.0, smoothing=0.2, relative_cutoff=0.5)

    assert await ranker(FRAGMENTS, "the and of") == []


def test_bm25_ranker_exposes_tuning():
    # Arrange
    ranker = Bm25Ranker(relative_cutoff=0.2, smoothing=0.3, link_penalty=10.0)

    # Act
    tuning = (ranker.relative_cutoff, ranker.smoothing, ranker.link_penalty)

    # Assert
    assert tuning == (0.2, 0.3, 10.0)
