"""Vault note paths and resolution of note references.

A note path is the vault-relative path with a leading ``/`` and no extension
(``Folder/Note.md`` -> ``/Folder/Note``). A reference is what a caller types:
a full path (``/Folder/Note``), a partial path (``Folder/Note``) matching any
note whose path ends with those segments, or a bare name (``Note``).
"""

from collections import defaultdict
from collections.abc import Iterable

from mcps.rag.interfaces import Link, dedupe_links

MARKDOWN_SUFFIX = ".md"


def to_note_path(source_path: str) -> str:
    """Convert a source path (``A/B.md``) to a note path (``/A/B``)."""
    return "/" + source_path.removesuffix(MARKDOWN_SUFFIX)


def to_source_path(note_path: str) -> str:
    """Convert a note path (``/A/B``) to a source path (``A/B.md``)."""
    return note_path.removeprefix("/") + MARKDOWN_SUFFIX


def link_aliases(note_path: str) -> list[str]:
    """Return every raw link target that resolves to the note.

    Shortest first: the bare name, each longer partial path, then the full
    note path.
    """
    segments = _segments(note_path)
    tails = [
        "/".join(segments[start:]) for start in range(len(segments) - 1, -1, -1)
    ]
    return [*tails, note_path]


def alias_map(note_paths: Iterable[str]) -> dict[str, list[str]]:
    """Map each raw link target to the given notes it can resolve to."""
    aliases: dict[str, list[str]] = defaultdict(list)
    for note_path in note_paths:
        for alias in link_aliases(note_path):
            aliases[alias].append(note_path)
    return aliases


def _segments(path: str) -> list[str]:
    return path.removeprefix("/").split("/")


def is_malformed_reference(reference: str) -> bool:
    """Return True for references that can never match a note."""
    if reference.endswith((MARKDOWN_SUFFIX, "/")) or "\\" in reference:
        return True
    return any(segment in {"", ".", ".."} for segment in _segments(reference))


def reference_name(reference: str) -> str | None:
    """Return the note name a reference ends with, None when malformed."""
    if is_malformed_reference(reference):
        return None
    return _segments(reference)[-1]


def select_matching(reference: str, note_paths: Iterable[str]) -> list[str]:
    """Return the sorted note paths a reference matches (case-sensitive).

    A full path matches itself; a partial path or bare name matches every
    note whose path ends with those segments.
    """
    if is_malformed_reference(reference):
        return []
    if reference.startswith("/"):
        return [path for path in note_paths if path == reference]
    wanted = _segments(reference)
    return sorted(
        path for path in note_paths if _segments(path)[-len(wanted):] == wanted
    )


def pick_note(matches: Iterable[str], from_note: str | None = None) -> str:
    """Pick the one note an ambiguous reference means, like Obsidian does.

    Rules in order: the shortest path, then the note closest to ``from_note``
    (most shared leading folders), then alphabetical order.
    """
    source_folders = _segments(from_note)[:-1] if from_note else []

    def rank(note_path: str) -> tuple[int, int, str]:
        segments = _segments(note_path)
        return (
            len(segments),
            -_shared_prefix_length(source_folders, segments[:-1]),
            note_path,
        )

    return min(matches, key=rank)


def _shared_prefix_length(first: list[str], second: list[str]) -> int:
    shared = 0
    for left, right in zip(first, second, strict=False):
        if left != right:
            break
        shared += 1
    return shared


def resolve_links(
    links: Iterable[Link],
    candidates: dict[str, list[str]],
    from_note: str | None = None,
) -> list[Link]:
    """Replace link targets with the note path each one resolves to.

    ``candidates`` maps raw link targets to every note path they match; the
    link points to the one picked by `pick_note` for the linking note
    ``from_note``. Targets matching no note stay unchanged. Deduplication runs
    afterwards so an untyped link is absorbed by a typed link to the same
    resolved note.
    """
    resolved: list[Link] = []
    for link in links:
        matches = candidates.get(link.target)
        if matches:
            resolved.append(
                Link(type=link.type, target=pick_note(matches, from_note))
            )
        else:
            resolved.append(link)
    return dedupe_links(resolved)
