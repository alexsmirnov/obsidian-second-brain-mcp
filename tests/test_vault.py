from collections.abc import Generator
from datetime import datetime
from pathlib import Path

import pytest

from mcps.rag.interfaces import (
    Chunk,
    Document,
    IChunker,
    IDocumentProcessor,
    IFileTraversal,
    ISearchEngine,
    IVectorStore,
    Metadata,
    SearchQuery,
    SearchScope,
)
from mcps.rag.vault import Vault


class FakeDocumentProcessor(IDocumentProcessor):
    async def process(self, file_path: Path) -> Document:
        return Document(
            id="doc",
            content="",
            metadata=Metadata(),
            source_path=file_path.name,
            modified_at=datetime.now(),
        )


class FakeChunker(IChunker):
    def chunk(self, document: Document) -> Generator[Chunk]:
        yield from ()


class FakeVectorStore(IVectorStore):
    def __init__(self) -> None:
        self.sources_by_name: dict[str, list[str]] = {}

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

    async def sources(self) -> dict[str, datetime]:
        return {}

    async def get_sources_by_name(self, wikilink_name: str) -> list[str]:
        return self.sources_by_name.get(wikilink_name, [])


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


async def test_get_file_reads_exact_wikilink_markdown_file(tmp_path: Path) -> None:
    note = tmp_path / "Folder" / "Note.md"
    note.parent.mkdir()
    note.write_text("content", encoding="utf-8")
    vault, _ = make_vault(tmp_path)

    result = await vault.get_file("Folder/Note")

    assert result == "content"


async def test_get_file_applies_offset_and_limit_as_string_characters(
    tmp_path: Path,
) -> None:
    (tmp_path / "Note.md").write_text("a🙂bcdef", encoding="utf-8")
    vault, vector_store = make_vault(tmp_path)
    vector_store.sources_by_name = {"Note": ["Note.md"]}

    result = await vault.get_file("Note", offset=2, limit=3)

    assert result == "bcd"


async def test_get_file_returns_empty_string_for_zero_limit(tmp_path: Path) -> None:
    (tmp_path / "Note.md").write_text("content", encoding="utf-8")
    vault, vector_store = make_vault(tmp_path)
    vector_store.sources_by_name = {"Note": ["Note.md"]}

    result = await vault.get_file("Note", limit=0)

    assert result == ""


async def test_get_file_returns_empty_string_for_offset_at_or_beyond_eof(
    tmp_path: Path,
) -> None:
    (tmp_path / "Note.md").write_text("content", encoding="utf-8")
    vault, vector_store = make_vault(tmp_path)
    vector_store.sources_by_name = {"Note": ["Note.md"]}

    result = await vault.get_file("Note", offset=20)

    assert result == ""


async def test_get_file_returns_not_found_for_missing_exact_file(
    tmp_path: Path,
) -> None:
    vault, _ = make_vault(tmp_path)

    with pytest.raises(FileNotFoundError):
        await vault.get_file("Missing")


async def test_get_file_does_not_match_by_stem_or_substring(tmp_path: Path) -> None:
    (tmp_path / "Folder").mkdir()
    (tmp_path / "Folder" / "Long Note.md").write_text("content", encoding="utf-8")
    vault, _ = make_vault(tmp_path)

    with pytest.raises(FileNotFoundError):
        await vault.get_file("Note")


async def test_get_file_reads_unique_source_path_for_short_name(tmp_path: Path) -> None:
    (tmp_path / "Folder").mkdir()
    (tmp_path / "Folder" / "Note.md").write_text("content", encoding="utf-8")
    vault, vector_store = make_vault(tmp_path)
    vector_store.sources_by_name = {"Note": ["Folder/Note.md"]}

    result = await vault.get_file("Note")

    assert result == "content"


async def test_get_file_raises_value_error_for_ambiguous_short_name(
    tmp_path: Path,
) -> None:
    vault, vector_store = make_vault(tmp_path)
    vector_store.sources_by_name = {
        "Note": ["Archive/Note.md", "Projects/Note.md"],
    }

    with pytest.raises(ValueError, match=r"Archive/Note\.md, Projects/Note\.md"):
        await vault.get_file("Note")
