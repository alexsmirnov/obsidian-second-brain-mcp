import asyncio
from collections.abc import Generator
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from mcps.config import ServerConfig
from mcps.rag.document_processing import SemanticChunker
from mcps.rag.interfaces import (
    Chunk,
    Document,
    IChunker,
    IDocumentProcessor,
    IFileTraversal,
    ISearchEngine,
    IVectorStore,
    Link,
    Metadata,
    NoteLinks,
    SearchQuery,
    SearchScope,
)
from mcps.rag.vault import Vault, _create_chunker


class FakeDocumentProcessor(IDocumentProcessor):
    async def process(self, file_path: Path) -> Document:
        return Document(
            id="doc",
            content="",
            metadata=Metadata(),
            source_path=file_path.name,
            modified_at=1234.567,
            wikilink_name="doc",
            file_size=10
        )


class FakeChunker(IChunker):
    def chunk(self, document: Document) -> Generator[Chunk]:
        yield from ()


class FakeVectorStore(IVectorStore):
    def __init__(self) -> None:
        # source path -> outgoing links of the indexed note
        self.notes: dict[str, list[Link]] = {}
        self.name_lookups: list[str] = []
        self.source_scans = 0

    async def initialize(self) -> None:
        pass

    async def store(self, chunks: list[Chunk]) -> None:
        pass

    async def search(
        self,
        query: str,
        hypotetical_document: str | None = None,
        tags: list[str] | None = None,
        file_path: str | None = None,
        scope: SearchScope = SearchScope.ALL,
        limit: int = 5,
    ) -> list[Chunk]:
        return []

    async def delete(self, source_paths: list[str]) -> None:
        pass

    async def reindex(self) -> None:
        pass

    async def sources(self) -> dict[str, float]:
        self.source_scans += 1
        return dict.fromkeys(self.notes, 0.0)

    async def get_chunks_by_ids(self, ids: list[str]) -> list[Chunk]:
        return []

    async def get_sources_by_name(self, wikilink_name: str) -> list[str]:
        self.name_lookups.append(wikilink_name)
        return sorted(
            path for path in self.notes if Path(path).stem == wikilink_name
        )

    async def get_notes_with_links(
        self, source_paths: list[str]
    ) -> list[NoteLinks]:
        return [
            NoteLinks(note=path, links=self.notes[path])
            for path in source_paths
            if path in self.notes
        ]

    async def get_notes_linking_to(
        self, targets: list[str], relation_types: list[str] | None = None
    ) -> list[NoteLinks]:
        result = []
        for path, links in self.notes.items():
            matching = [
                link
                for link in links
                if link.target in targets
                and (relation_types is None or link.type in relation_types)
            ]
            if matching:
                result.append(NoteLinks(note=path, links=matching))
        return result


class FakeSearchEngine(ISearchEngine):
    async def search(self, query: SearchQuery) -> list[Chunk]:
        return []


class FakeFileTraversal(IFileTraversal):
    def find_files(self) -> Generator[Path]:
        yield from ()


def make_vault(vault_path: Path) -> tuple[Vault, FakeVectorStore]:
    vector_store = FakeVectorStore()
    vault = Vault(
        vault_path=vault_path,
        file_traversal=FakeFileTraversal(),
        document_processor=FakeDocumentProcessor(),
        chunker=FakeChunker(),
        vector_store=vector_store,
        search_engine=FakeSearchEngine(),
    )
    vault._initialized = True
    return vault, vector_store


def write_note(vault_root: Path, source_path: str, content: str) -> None:
    note = vault_root / source_path
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text(content, encoding="utf-8")


async def test_get_file_full_path_reads_exact_file(tmp_path: Path) -> None:
    write_note(tmp_path, "A/B.md", "ab")
    write_note(tmp_path, "R/A/B.md", "rab")
    vault, vector_store = make_vault(tmp_path)
    vector_store.notes = {"A/B.md": [], "R/A/B.md": []}

    result = await vault.get_file("/A/B")

    assert result == "ab"


async def test_get_file_full_path_reads_file_missing_from_index(
    tmp_path: Path,
) -> None:
    write_note(tmp_path, "Excluded/Raw.md", "raw")
    vault, _ = make_vault(tmp_path)

    result = await vault.get_file("/Excluded/Raw")

    assert result == "raw"


async def test_get_file_full_path_cannot_escape_the_vault(tmp_path: Path) -> None:
    vault_root = tmp_path / "vault"
    vault_root.mkdir()
    (tmp_path / "secret.md").write_text("secret", encoding="utf-8")
    vault, _ = make_vault(vault_root)

    with pytest.raises(FileNotFoundError):
        await vault.get_file("/../secret")


async def test_get_file_unique_partial_path_reads_the_match(tmp_path: Path) -> None:
    write_note(tmp_path, "R/A/B.md", "rab")
    write_note(tmp_path, "Q/B.md", "qb")
    vault, vector_store = make_vault(tmp_path)
    vector_store.notes = {"R/A/B.md": [], "Q/B.md": []}

    result = await vault.get_file("A/B")

    assert result == "rab"


async def test_get_file_ambiguous_partial_path_reads_shortest_path(
    tmp_path: Path,
) -> None:
    write_note(tmp_path, "A/B.md", "short")
    write_note(tmp_path, "R/A/B.md", "long")
    vault, vector_store = make_vault(tmp_path)
    vector_store.notes = {"A/B.md": [], "R/A/B.md": [], "XA/B.md": []}

    assert await vault.get_file("A/B") == "short"


async def test_get_file_ambiguous_bare_name_breaks_tie_alphabetically(
    tmp_path: Path,
) -> None:
    write_note(tmp_path, "A/B.md", "first")
    write_note(tmp_path, "Q/B.md", "second")
    vault, vector_store = make_vault(tmp_path)
    vector_store.notes = {"A/B.md": [], "Q/B.md": [], "R/A/B.md": []}

    assert await vault.get_file("B") == "first"


async def test_get_file_reads_unique_bare_name(tmp_path: Path) -> None:
    write_note(tmp_path, "Folder/Note.md", "content")
    vault, vector_store = make_vault(tmp_path)
    vector_store.notes = {"Folder/Note.md": []}

    result = await vault.get_file("Note")

    assert result == "content"


async def test_get_file_applies_offset_and_limit_as_lines(
    tmp_path: Path,
) -> None:
    write_note(tmp_path, "Note.md", "line0\nline1\nline2\nline3\n")
    vault, vector_store = make_vault(tmp_path)
    vector_store.notes = {"Note.md": []}

    result = await vault.get_file("Note", offset=1, limit=2)

    assert result == "line1\nline2\n"


async def test_get_file_returns_empty_string_for_zero_limit(tmp_path: Path) -> None:
    write_note(tmp_path, "Note.md", "content")
    vault, vector_store = make_vault(tmp_path)
    vector_store.notes = {"Note.md": []}

    result = await vault.get_file("Note", limit=0)

    assert result == ""


async def test_get_file_returns_empty_string_for_offset_at_or_beyond_eof(
    tmp_path: Path,
) -> None:
    write_note(tmp_path, "Note.md", "content")
    vault, vector_store = make_vault(tmp_path)
    vector_store.notes = {"Note.md": []}

    result = await vault.get_file("Note", offset=20)

    assert result == ""


async def test_get_file_unknown_reference_raises_not_found(tmp_path: Path) -> None:
    vault, _ = make_vault(tmp_path)

    with pytest.raises(FileNotFoundError):
        await vault.get_file("Missing")


async def test_get_file_does_not_match_by_substring(tmp_path: Path) -> None:
    vault, vector_store = make_vault(tmp_path)
    vector_store.notes = {"Folder/Long Note.md": []}

    with pytest.raises(FileNotFoundError):
        await vault.get_file("Note")


async def test_resolve_references_selects_notes_matching_partial_path(
    tmp_path: Path,
) -> None:
    vault, vector_store = make_vault(tmp_path)
    vector_store.notes = {"Solo.md": [], "A/B.md": [], "R/A/B.md": [], "Q/B.md": []}

    result = await vault.resolve_references(["Solo", "A/B", "B", "/Q/B", "Nope"])

    assert result == {
        "Solo": ["/Solo"],
        "A/B": ["/A/B", "/R/A/B"],
        "B": ["/A/B", "/Q/B", "/R/A/B"],
        "/Q/B": ["/Q/B"],
        "Nope": [],
    }


async def test_resolve_references_queries_by_name_without_full_scan(
    tmp_path: Path,
) -> None:
    vault, vector_store = make_vault(tmp_path)
    vector_store.notes = {"A/B.md": []}

    await vault.resolve_references(["B", "A/B", "B", "/A/B", "A//B"])

    assert vector_store.name_lookups == ["B", "B", "B"]
    assert vector_store.source_scans == 0


def make_dup_store(vector_store: FakeVectorStore) -> None:
    vector_store.notes = {
        "Hub/Alpha.md": [Link(type="uses", target="Dup")],
        "P/Dup.md": [],
        "Q/Dup.md": [],
        "Z/Zulu.md": [Link(type="details", target="P/Dup")],
    }


async def test_get_backlinks_attributes_ambiguous_link_to_picked_note_only(
    tmp_path: Path,
) -> None:
    vault, vector_store = make_vault(tmp_path)
    make_dup_store(vector_store)

    result = await vault.get_backlinks(["/P/Dup", "/Q/Dup"])

    assert Link(type="uses", target="/Hub/Alpha") in result["/P/Dup"]
    assert result["/Q/Dup"] == []


async def test_get_backlinks_ambiguous_link_prefers_nearest_note(
    tmp_path: Path,
) -> None:
    vault, vector_store = make_vault(tmp_path)
    make_dup_store(vector_store)
    vector_store.notes["Q/Beta.md"] = [Link(type="uses", target="Dup")]

    result = await vault.get_backlinks(["/P/Dup", "/Q/Dup"])

    assert result["/Q/Dup"] == [Link(type="uses", target="/Q/Beta")]
    assert Link(type="uses", target="/Q/Beta") not in result["/P/Dup"]


async def test_get_backlinks_partial_link_hits_only_its_match(
    tmp_path: Path,
) -> None:
    vault, vector_store = make_vault(tmp_path)
    make_dup_store(vector_store)

    result = await vault.get_backlinks(["/P/Dup", "/Q/Dup"])

    assert Link(type="details", target="/Z/Zulu") in result["/P/Dup"]
    assert Link(type="details", target="/Z/Zulu") not in result["/Q/Dup"]


async def test_get_backlinks_returns_empty_buckets_for_unlinked_notes(
    tmp_path: Path,
) -> None:
    vault, vector_store = make_vault(tmp_path)
    make_dup_store(vector_store)

    result = await vault.get_backlinks(["/Z/Zulu"])

    assert result == {"/Z/Zulu": []}


async def test_list_files_lists_notes_and_folders_for_root(tmp_path: Path) -> None:
    vault_root = tmp_path / "vault"
    (vault_root / "Projects").mkdir(parents=True)
    (vault_root / "Note.md").write_text("content", encoding="utf-8")
    vault, _ = make_vault(vault_root)

    result = await vault.list_files("")

    assert result == ["Note", "Projects/"]


async def test_list_files_lists_subfolder_contents(tmp_path: Path) -> None:
    vault_root = tmp_path / "vault"
    (vault_root / "Projects").mkdir(parents=True)
    (vault_root / "Projects" / "Plan.md").write_text("content", encoding="utf-8")
    vault, _ = make_vault(vault_root)

    result = await vault.list_files("Projects")

    assert result == ["Plan"]


async def test_list_files_rejects_parent_traversal_escape(
    tmp_path: Path,
) -> None:
    vault_root = tmp_path / "vault"
    vault_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "Secret.md").write_text("secret", encoding="utf-8")
    vault, _ = make_vault(vault_root)

    with pytest.raises(ValueError, match="escapes the vault root"):
        await vault.list_files("../outside")


async def test_list_files_rejects_nested_parent_traversal_escape(
    tmp_path: Path,
) -> None:
    vault_root = tmp_path / "vault"
    (vault_root / "a").mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "Secret.md").write_text("secret", encoding="utf-8")
    vault, _ = make_vault(vault_root)

    with pytest.raises(ValueError, match="escapes the vault root"):
        await vault.list_files("a/../../outside")


async def test_list_files_rejects_symlink_escape(tmp_path: Path) -> None:
    vault_root = tmp_path / "vault"
    vault_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "Secret.md").write_text("secret", encoding="utf-8")
    (vault_root / "linked").symlink_to(outside, target_is_directory=True)
    vault, _ = make_vault(vault_root)

    with pytest.raises(ValueError, match="escapes the vault root"):
        await vault.list_files("linked")


async def test_search_does_not_call_update_index(tmp_path: Path) -> None:
    vault, _ = make_vault(tmp_path)
    vault.update_index = AsyncMock(side_effect=RuntimeError("update_index called"))  # type: ignore[method-assign]

    results = await vault.search("query")

    assert results == []


async def test_search_returns_immediately_when_index_update_is_slow(
    tmp_path: Path,
) -> None:
    vault, _ = make_vault(tmp_path)

    async def slow_update() -> None:
        await asyncio.sleep(60)

    vault.update_index = AsyncMock(side_effect=slow_update)  # type: ignore[method-assign]

    results = await asyncio.wait_for(vault.search("query"), timeout=0.1)

    assert results == []


def test_create_chunker_with_config_passes_bounds() -> None:
    config = ServerConfig(min_chunk_size=800, max_chunk_size=1600)
    chunker = _create_chunker(config)
    assert isinstance(chunker, SemanticChunker)
    assert chunker.min_chunk_size == 800
    assert chunker.max_chunk_size == 1600

