import pytest

from mcps.rag.interfaces import Link
from mcps.rag.note_paths import (
    alias_map,
    is_malformed_reference,
    link_aliases,
    pick_note,
    reference_name,
    resolve_links,
    select_matching,
    to_note_path,
    to_source_path,
)

PATHS = [
    "/A/B",
    "/R/A/B",
    "/R/X/A/B",
    "/XA/B",
    "/A/B2",
    "/A/B/C",
    "/Q/B",
    "/Solo",
    "/D/Note.v2",
]


def test_to_note_path_strips_extension_and_adds_root():
    assert to_note_path("A/Note.v2.md") == "/A/Note.v2"


def test_to_source_path_inverts_note_path():
    assert to_source_path("/A/Note.v2") == "A/Note.v2.md"


def test_link_aliases_lists_every_suffix_and_full_path():
    assert link_aliases("/R/A/B") == ["B", "A/B", "R/A/B", "/R/A/B"]


def test_alias_map_lists_every_note_an_alias_resolves_to():
    result = alias_map(["/P/Dup", "/Q/Dup"])

    assert result["Dup"] == ["/P/Dup", "/Q/Dup"]
    assert result["P/Dup"] == ["/P/Dup"]
    assert result["/Q/Dup"] == ["/Q/Dup"]


@pytest.mark.parametrize(
    "reference",
    [
        "",
        "/",
        "A//B",
        "A/B/",
        "/A/B/",
        "B.md",
        "A\\B",
        "/../x",
        "A/../B",
        "/./A",
        "A/./B",
        "/.",
    ],
)
def test_is_malformed_reference_rejects_unresolvable_forms(reference):
    assert is_malformed_reference(reference)


@pytest.mark.parametrize("reference", ["B", "A/B", "/A/B", "/Note.v2"])
def test_is_malformed_reference_accepts_valid_forms(reference):
    assert not is_malformed_reference(reference)


def test_reference_name_is_last_segment():
    assert reference_name("/R/A/B") == "B"
    assert reference_name("A/B") == "B"
    assert reference_name("Solo") == "Solo"


@pytest.mark.parametrize("reference", ["", "/", "A//B", "B.md"])
def test_reference_name_is_none_for_malformed_reference(reference):
    assert reference_name(reference) is None


def test_full_path_matches_exactly():
    assert select_matching("/A/B", PATHS) == ["/A/B"]


def test_partial_path_matches_whole_segment_suffixes():
    assert select_matching("A/B", PATHS) == ["/A/B", "/R/A/B", "/R/X/A/B"]


def test_bare_name_matches_every_stem():
    assert select_matching("B", PATHS) == [
        "/A/B",
        "/Q/B",
        "/R/A/B",
        "/R/X/A/B",
        "/XA/B",
    ]


def test_unique_bare_name():
    assert select_matching("Solo", PATHS) == ["/Solo"]


@pytest.mark.parametrize("reference", ["Missing", "/B", "solo"])
def test_unknown_reference_matches_nothing(reference):
    assert select_matching(reference, PATHS) == []


@pytest.mark.parametrize("reference", ["", "/", "A//B", "A/B/", "B.md"])
def test_malformed_reference_matches_nothing(reference):
    assert select_matching(reference, PATHS) == []


def test_resolve_links_replaces_targets_and_keeps_unresolved():
    links = [
        Link(type="uses", target="X/A/B"),
        Link(type="related", target="img.png"),
    ]
    resolved = {"X/A/B": ["/R/X/A/B"]}

    assert resolve_links(links, resolved) == [
        Link(type="uses", target="/R/X/A/B"),
        Link(type="related", target="img.png"),
    ]


def test_resolve_links_points_ambiguous_link_to_the_picked_note():
    links = [Link(type="uses", target="A/B")]
    resolved = {"A/B": ["/R/A/B", "/A/B"]}

    assert resolve_links(links, resolved) == [Link(type="uses", target="/A/B")]


def test_resolve_links_prefers_note_closest_to_linking_note():
    links = [Link(type="uses", target="Dup")]
    resolved = {"Dup": ["/P/Dup", "/Q/Dup"]}

    assert resolve_links(links, resolved, from_note="/Q/Beta") == [
        Link(type="uses", target="/Q/Dup")
    ]


def test_resolve_links_dedupes_after_expansion():
    links = [
        Link(type="related", target="Solo"),
        Link(type="uses", target="/Solo"),
    ]
    resolved = {"Solo": ["/Solo"], "/Solo": ["/Solo"]}

    assert resolve_links(links, resolved) == [Link(type="uses", target="/Solo")]


def test_pick_note_prefers_shortest_path():
    assert pick_note(["/R/A/B", "/A/B", "/X/Y/A/B"]) == "/A/B"


def test_pick_note_breaks_length_tie_by_proximity_to_linking_note():
    matches = ["/P/Dup", "/Q/Dup"]

    assert pick_note(matches, from_note="/Q/Beta") == "/Q/Dup"


def test_pick_note_proximity_counts_shared_leading_folders():
    matches = ["/X/Y/Dup", "/X/Z/Dup"]

    assert pick_note(matches, from_note="/X/Z/Note") == "/X/Z/Dup"


def test_pick_note_shortest_path_beats_proximity():
    matches = ["/Dup", "/Q/Sub/Dup"]

    assert pick_note(matches, from_note="/Q/Sub/Note") == "/Dup"


def test_pick_note_falls_back_to_alphabetical_order():
    matches = ["/Q/Dup", "/P/Dup"]

    assert pick_note(matches) == "/P/Dup"
    assert pick_note(matches, from_note="/Z/Note") == "/P/Dup"
