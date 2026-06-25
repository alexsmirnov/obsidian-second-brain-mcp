from collections.abc import Generator
from datetime import datetime
from pathlib import Path

import pytest

from mcps.rag.document_processing import SUMMARY_CHUNK_POSITION
from mcps.rag.interfaces import (
    Chunk,
    Document,
    IChunker,
    IDocumentProcessor,
    IDocumentSummaryGenerator,
    IFileTraversal,
    ISearchEngine,
    IVectorStore,
    Metadata,
    SearchQuery,
    SearchScope,
)
from mcps.rag.vault import Vault


class FakeDocumentProcessor(IDocumentProcessor):
    def __init__(self, document: Document) -> None:
        self.document = document

    async def process(self, file_path: Path) -> Document:
        return self.document


class FakeChunker(IChunker):
    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks

    def chunk(self, document: Document) -> Generator[Chunk]:
        yield from self.chunks


class FakeVectorStore(IVectorStore):
    def __init__(self) -> None:
        self.stored_chunks: list[Chunk] = []

    async def initialize(self) -> None:
        pass

    async def store(self, chunks: list[Chunk]) -> None:
        self.stored_chunks = chunks

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
        return []

    async def get_chunks_by_ids(self, ids: list[str]) -> list[Chunk]:
        return []


class FakeSearchEngine(ISearchEngine):
    async def search(self, query: SearchQuery) -> list[Chunk]:
        return []


class FakeFileTraversal(IFileTraversal):
    def find_files(self) -> Generator[Path]:
        yield from ()


class FakeSummaryGenerator(IDocumentSummaryGenerator):
    def __init__(
        self,
        summary: str | None = None,
        error: Exception | None = None,
    ) -> None:
        self.summary = summary
        self.error = error
        self.calls = 0

    async def generate(self, document: Document) -> str:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.summary or ""


@pytest.fixture
def document() -> Document:
    return Document(
        id="note-123",
        content="# Note\n\nBody with [[Whole Link]] and #whole-tag.",
        metadata=Metadata(title="Note", description="Description", source="source"),
        tags=["frontmatter"],
        source_path="folder/note.md",
        modified_at=datetime(2024, 1, 2, 3, 4, 5),
    )


@pytest.fixture
def normal_chunks(document: Document) -> list[Chunk]:
    return [
        Chunk(
            id=f"{document.id}_0",
            content="Normal chunk",
            title=document.metadata.title,
            description=document.metadata.description,
            source=document.metadata.source,
            outgoing_links=[],
            tags=document.tags,
            source_path=document.source_path,
            wikilink_name="folder/note",
            modified_at=document.modified_at,
            position=0,
            offset=0,
            size=len("Normal chunk"),
        )
    ]


def create_vault(
    document: Document,
    normal_chunks: list[Chunk],
    summary_generator: IDocumentSummaryGenerator | None,
) -> tuple[Vault, FakeVectorStore]:
    vector_store = FakeVectorStore()
    vault = Vault(
        vault_path=Path("/vault"),
        file_traversal=FakeFileTraversal(),
        document_processor=FakeDocumentProcessor(document),
        chunker=FakeChunker(normal_chunks),
        vector_store=vector_store,
        search_engine=FakeSearchEngine(),
        document_summary_generator=summary_generator,
    )
    return vault, vector_store


async def test_process_file_stores_summary_chunk_in_addition_to_semantic_chunks(
    document: Document,
    normal_chunks: list[Chunk],
) -> None:
    vault, vector_store = create_vault(
        document,
        normal_chunks,
        FakeSummaryGenerator("Generated summary"),
    )

    await vault._process_file(Path("/vault/folder/note.md"))

    assert [chunk.position for chunk in vector_store.stored_chunks] == [
        SUMMARY_CHUNK_POSITION,
        0,
    ]
    summary_chunk = vector_store.stored_chunks[0]
    assert summary_chunk.content == "Generated summary"
    assert set(summary_chunk.outgoing_links) == {"Whole Link"}
    assert set(summary_chunk.tags) == {"frontmatter", "whole-tag"}
    assert vector_store.stored_chunks[1:] == normal_chunks


async def test_process_file_without_summary_generator_stores_semantic_chunks_only(
    document: Document,
    normal_chunks: list[Chunk],
) -> None:
    vault, vector_store = create_vault(document, normal_chunks, None)

    await vault._process_file(Path("/vault/folder/note.md"))

    assert vector_store.stored_chunks == normal_chunks


async def test_process_file_skips_summary_chunk_when_document_content_is_blank(
    document: Document,
    normal_chunks: list[Chunk],
) -> None:
    blank_document = document.model_copy(update={"content": "   "})
    summary_generator = FakeSummaryGenerator("Generated summary")
    vault, vector_store = create_vault(
        blank_document,
        normal_chunks,
        summary_generator,
    )

    await vault._process_file(Path("/vault/folder/note.md"))

    assert vector_store.stored_chunks == normal_chunks
    assert summary_generator.calls == 0


async def test_process_file_skips_summary_chunk_when_summary_is_blank(
    document: Document,
    normal_chunks: list[Chunk],
) -> None:
    vault, vector_store = create_vault(
        document,
        normal_chunks,
        FakeSummaryGenerator("  "),
    )

    await vault._process_file(Path("/vault/folder/note.md"))

    assert vector_store.stored_chunks == normal_chunks


async def test_process_file_skips_summary_chunk_when_summary_generation_fails(
    document: Document,
    normal_chunks: list[Chunk],
) -> None:
    vault, vector_store = create_vault(
        document,
        normal_chunks,
        FakeSummaryGenerator(error=RuntimeError("model unavailable")),
    )

    await vault._process_file(Path("/vault/folder/note.md"))

    assert vector_store.stored_chunks == normal_chunks
