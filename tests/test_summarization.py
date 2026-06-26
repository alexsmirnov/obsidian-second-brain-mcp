from datetime import datetime
from types import SimpleNamespace

import pytest

from mcps.rag.interfaces import Document, Metadata
from mcps.rag.summarization import LangChainDocumentSummaryGenerator


class FakeChatModel:
    def __init__(self, content: object) -> None:
        self.content = content
        self.messages: list[object] | None = None

    async def ainvoke(self, messages: list[object]) -> SimpleNamespace:
        self.messages = messages
        return SimpleNamespace(content=self.content)


@pytest.fixture
def document() -> Document:
    content = "# Note\n\nWhole note body with [[Link]] and #tag."
    return Document(
        id="note-123",
        content=content,
        metadata=Metadata(title="Note Title", description="Note description"),
        tags=["frontmatter"],
        source_path="folder/note.md",
        wikilink_name="folder/note",
        file_size=len(content),
        modified_at=datetime(2024, 1, 2, 3, 4, 5),
    )


async def test_generate_sends_whole_document_to_model_and_returns_stripped_content(
    document: Document,
) -> None:
    model = FakeChatModel("\n  Concise summary.  \n")
    generator = LangChainDocumentSummaryGenerator(model)

    summary = await generator.generate(document)

    assert summary == "Concise summary."
    assert model.messages is not None
    prompt_text = "\n".join(str(message.content) for message in model.messages)
    assert document.content in prompt_text
    assert document.metadata.title in prompt_text
    assert document.metadata.description in prompt_text


async def test_generate_converts_non_string_response_content_to_string(
    document: Document,
) -> None:
    model = FakeChatModel(["summary", "parts"])
    generator = LangChainDocumentSummaryGenerator(model)

    summary = await generator.generate(document)

    assert summary == "['summary', 'parts']"
