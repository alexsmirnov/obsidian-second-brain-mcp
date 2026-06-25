"""Whole-document summarization for RAG indexing."""

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from .interfaces import Document, IDocumentSummaryGenerator

DOCUMENT_SUMMARY_SYSTEM_PROMPT = """Summarize the Obsidian markdown note below.
Write one concise paragraph. Focus on the note's core ideas,
claims, entities, and terminology. Do not include tags or wikilinks."""


class LangChainDocumentSummaryGenerator(IDocumentSummaryGenerator):
    """Generate whole-document summaries with a LangChain chat model."""

    def __init__(self, model: BaseChatModel) -> None:
        self.model = model

    async def generate(self, document: Document) -> str:
        messages = [
            SystemMessage(content=DOCUMENT_SUMMARY_SYSTEM_PROMPT),
            HumanMessage(content=self._create_prompt(document)),
        ]
        response = await self.model.ainvoke(messages)
        content = response.content
        if isinstance(content, str):
            return content.strip()
        return str(content).strip()

    @staticmethod
    def _create_prompt(document: Document) -> str:
        return f"""Title: {document.metadata.title or ""}
Description: {document.metadata.description or ""}
Source path: {document.source_path}

Document:
{document.content}

Summary:"""
