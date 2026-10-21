"""HTML to Markdown conversion shared by the browser and the httpx route."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import replace

from crawl4ai import DefaultMarkdownGenerator, PruningContentFilterLXML

from mcps.research.tools.common import (
    MIME_HTML,
    MIME_MARKDOWN,
    failure,
    textual_mime,
)
from mcps.research.tools.models import FetchResult, FetchStatus

__all__ = ["HtmlToMarkdown", "create_markdown_generator", "is_html"]

logger = logging.getLogger(__name__)


def create_markdown_generator() -> DefaultMarkdownGenerator:
    """Generator pruning boilerplate; ``fit_markdown`` holds the pruned text."""
    return DefaultMarkdownGenerator(
        content_filter=PruningContentFilterLXML(threshold=0.6),
        options={"ignore_links": False},
    )


def is_html(result: FetchResult) -> bool:
    return textual_mime(result.mime) == MIME_HTML


class HtmlToMarkdown:
    """Filter turning an HTML result into Markdown."""

    def __init__(self, generator: DefaultMarkdownGenerator | None = None) -> None:
        self._generator = generator or create_markdown_generator()

    async def __call__(
        self, result: FetchResult, query: str | None = None, /
    ) -> FetchResult:
        try:
            generated = await asyncio.to_thread(
                self._generator.generate_markdown,
                result.content,
                result.url,
                citations=False,
            )
        except Exception:
            logger.warning("HTML conversion failed for %s", result.url)
            return failure(result.url, FetchStatus.FILTER_FAILED)
        markdown = generated.fit_markdown or generated.raw_markdown or ""
        return replace(result, content=markdown.strip(), mime=MIME_MARKDOWN)
