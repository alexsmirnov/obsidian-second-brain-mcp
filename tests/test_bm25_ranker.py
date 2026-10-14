"""Contract tests for the lexical BM25L ranker."""

from __future__ import annotations

from mcps.research.tools.bm25_ranker import Bm25Ranker, lexical_scores

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
    scores = lexical_scores(FRAGMENTS, "quantum")

    assert len(scores) == len(FRAGMENTS)
    assert scores[0] == 0.0 and scores[2] == 0.0
    assert scores[1] > 0.0 and scores[3] > 0.0
