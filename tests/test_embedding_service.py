"""Contract tests for LangChain-backed RAG embeddings."""

from langchain_core.embeddings import Embeddings

from mcps.rag.embeddings import LangChainEmbeddingService
from mcps.rag.interfaces import IEmbeddingService


class FakeEmbeddings(Embeddings):
    def __init__(self) -> None:
        self.document_calls: list[list[str]] = []
        self.query_calls: list[str] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_calls.append(texts)
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        self.query_calls.append(text)
        return [99.0, float(len(text))]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)

    @staticmethod
    def _vector(text: str) -> list[float]:
        return [float(len(text)), float(sum(ord(char) for char in text) % 100)]


async def test_generate_embeddings_empty_texts_returns_empty_list() -> None:
    embeddings = FakeEmbeddings()
    service: IEmbeddingService = LangChainEmbeddingService(embeddings, dimensions=2)

    result = await service.generate_embeddings([])

    assert result == []
    assert embeddings.document_calls == []
    assert embeddings.query_calls == []


async def test_generate_embeddings_documents_preserves_input_order() -> None:
    embeddings = FakeEmbeddings()
    service = LangChainEmbeddingService(embeddings, dimensions=2)

    result = await service.generate_embeddings(["first", "second"])

    assert result == [FakeEmbeddings._vector("first"), FakeEmbeddings._vector("second")]
    assert embeddings.document_calls == [["first", "second"]]


async def test_generate_embeddings_query_uses_query_embedding_contract() -> None:
    embeddings = FakeEmbeddings()
    service = LangChainEmbeddingService(embeddings, dimensions=2)

    result = await service.generate_embeddings(["question"], query=True)

    assert result == [[99.0, 8.0]]
    assert embeddings.query_calls == ["question"]
    assert embeddings.document_calls == []


async def test_ndims_returns_configured_embedding_dimensions() -> None:
    service = LangChainEmbeddingService(FakeEmbeddings(), dimensions=1536)

    assert service.ndims() == 1536
