"""Pure content preparation, source-mapped parsing, and selection rendering.

This module owns everything about a fetched document *except* relevance
scoring and model calls. It renders HTML once, keeps source character offsets
for Markdown and plain text, resolves hyperlinks against the requested URL, and
renders the selected source spans back into a string.

It deliberately does not import :mod:`filtering`, :mod:`combinators`, or
:mod:`fetch` so the dependency direction stays one-way.
"""

from __future__ import annotations

import html as html_lib
import re
from dataclasses import dataclass
from typing import Literal
from urllib.parse import urljoin, urlparse

from crawl4ai import DefaultMarkdownGenerator
from lxml import html as lxml_html
from markdown_it import MarkdownIt
from markdown_it.common.utils import normalizeReference
from markdown_it.token import Token

from mcps.research.tools.common import (
    MIME_HTML,
    MIME_MARKDOWN,
    MIME_PLAIN,
    textual_mime,
)
from mcps.research.tools.models import FetchResult

__all__ = [
    "ContentBlock",
    "ContentDocument",
    "ScoringWindow",
    "SourceEdit",
    "SourceSpan",
    "build_scoring_windows",
    "normalize_markdown_links",
    "parse_document",
    "prepare_document",
    "render_selection",
    "truncate_content",
]

BlockKind = Literal["heading", "prose", "atomic"]


@dataclass(frozen=True, slots=True)
class SourceSpan:
    """Half-open Python character range into prepared document text."""

    start: int
    end: int


@dataclass(frozen=True, slots=True)
class ContentBlock:
    """One source-mapped structural block of a document."""

    block_id: int
    span: SourceSpan
    kind: BlockKind
    heading_ids: tuple[int, ...]
    introduction_id: int | None
    reference_labels: tuple[str, ...]
    search_text: str
    trimmable: bool


@dataclass(frozen=True, slots=True)
class ContentDocument:
    """Prepared text plus its mapped blocks and reference definitions."""

    text: str
    mime: str
    blocks: tuple[ContentBlock, ...]
    references: dict[str, SourceSpan]


@dataclass(frozen=True, slots=True)
class ScoringWindow:
    """A searchable window belonging to a parent content block."""

    window_id: int
    block_id: int
    text: str


@dataclass(frozen=True, slots=True)
class SourceEdit:
    """A replacement of one source range; applied right-to-left."""

    start: int
    end: int
    replacement: str


_MD_OPTIONS = {"html": True, "store_labels": True}
_SAFE_SCHEMES = frozenset(["http", "https"])


def _build_markdown_parser() -> MarkdownIt:
    parser = MarkdownIt("commonmark", _MD_OPTIONS)
    parser.enable("table")
    return parser


def _line_offsets(text: str) -> list[int]:
    offsets = [0]
    for line in text.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    return offsets


def _rstrip_newlines(text: str, end: int) -> int:
    while end > 0 and text[end - 1] in "\r\n":
        end -= 1
    return end


def _map_span(token: Token, offsets: list[int], text: str) -> SourceSpan:
    lines = token.map or [0, 0]
    start = offsets[min(lines[0], len(offsets) - 1)]
    end = offsets[min(lines[1], len(offsets) - 1)]
    return SourceSpan(start, _rstrip_newlines(text, end))


def _matching_close(tokens: list[Token], start: int) -> int:
    depth = 0
    for index in range(start, len(tokens)):
        depth += tokens[index].nesting
        if depth == 0 and index > start:
            return index
    return len(tokens) - 1


def _squash(value: str) -> str:
    return " ".join(value.split())


def _visible_text(token: Token) -> str:
    """Return visible text of an inline token, dropping destinations/tags."""
    if token.type != "inline" or not token.children:
        return _squash(token.content)
    parts: list[str] = []
    for child in token.children:
        if child.type in ("text", "code_inline"):
            parts.append(child.content)
        elif child.type == "image":
            parts.append(child.content)
        elif child.type in ("softbreak", "hardbreak"):
            parts.append(" ")
    return _squash(" ".join(parts))


def _is_plain_inline(token: Token) -> bool:
    """True when an inline token contains only plain text and breaks."""
    if token.type != "inline" or not token.children:
        return True
    return all(
        child.type in ("text", "softbreak", "hardbreak")
        for child in token.children
    )


def _collect_labels(tokens: list[Token]) -> tuple[str, ...]:
    labels: list[str] = []
    for token in tokens:
        if token.type != "inline" or not token.children:
            continue
        for child in token.children:
            label = child.meta.get("label") if child.meta else None
            if label and label not in labels:
                labels.append(label)
    return tuple(labels)


def _visible_texts(tokens: list[Token]) -> str:
    parts = [
        _visible_text(token) for token in tokens if token.type == "inline"
    ]
    return _squash(" ".join(parts))


def _strip_html(value: str) -> str:
    return _squash(html_lib.unescape(re.sub(r"<[^>]+>", " ", value)))


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------


def _is_safe_destination(destination: str, base_url: str) -> str | None:
    """Resolve ``destination`` and return it only when it is an HTTP(S) URL."""
    candidate = destination.strip()
    if not candidate or "\\" in candidate:
        return None
    if any(ord(character) < 0x20 for character in candidate):
        return None
    resolved = urljoin(base_url, candidate)
    parsed = urlparse(resolved)
    if parsed.scheme.lower() not in _SAFE_SCHEMES or not parsed.hostname:
        return None
    return resolved


def _render_html(content: str, base_url: str) -> str:
    """Render HTML to Markdown once, dropping scripts/styles and bad targets."""
    parser = lxml_html.HTMLParser(recover=True, no_network=True)
    try:
        document = lxml_html.document_fromstring(
            content, parser=parser, base_url=base_url
        )
    except Exception:
        document = None
    if document is not None:
        for element in document.xpath("//script|//style"):
            parent = element.getparent()
            if parent is not None:
                if element.tail:
                    previous = element.getprevious()
                    if previous is not None:
                        previous.tail = (previous.tail or "") + element.tail
                    else:
                        parent.text = (parent.text or "") + element.tail
                parent.remove(element)
        for anchor in document.xpath("//a[@href]"):
            href = anchor.get("href", "")
            if _is_safe_destination(href, base_url) is None:
                del anchor.attrib["href"]
        rendered_source = str(lxml_html.tostring(document, encoding="unicode"))
    else:
        rendered_source = content
    generated = DefaultMarkdownGenerator().generate_markdown(
        rendered_source, base_url, citations=False
    )
    return (generated.raw_markdown or "").strip()


# ---------------------------------------------------------------------------
# Markdown link normalization
# ---------------------------------------------------------------------------


def _protected_mask(text: str, protected: list[SourceSpan]) -> list[bool]:
    mask = [False] * len(text)
    for span in protected:
        for index in range(max(span.start, 0), min(span.end, len(text))):
            mask[index] = True
    return mask


def _code_spans(text: str) -> list[SourceSpan]:
    """Protected ranges: fenced/indented code, HTML blocks, and code spans."""
    parser = _build_markdown_parser()
    try:
        tokens = parser.parse(text, {})
    except Exception:
        tokens = []
    offsets = _line_offsets(text)
    spans: list[SourceSpan] = []
    for token in tokens:
        if token.type in ("fence", "code_block", "html_block") and token.map:
            spans.append(_map_span(token, offsets, text))
    mask = _protected_mask(text, spans)
    index = 0
    while index < len(text):
        if mask[index]:
            index += 1
            continue
        if text[index] == "`":
            run = 1
            while index + run < len(text) and text[index + run] == "`":
                run += 1
            closing = _find_backtick_run(text, index + run, run)
            if closing is not None:
                for position in range(index, closing + run):
                    mask[position] = True
                index = closing + run
                continue
        index += 1
    return [
        SourceSpan(start, end)
        for start, end in _mask_to_spans(mask)
    ]


def _find_backtick_run(text: str, start: int, run: int) -> int | None:
    index = start
    while index < len(text):
        if text[index] == "`":
            length = 1
            while index + length < len(text) and text[index + length] == "`":
                length += 1
            if length == run:
                return index
            index += length
        else:
            index += 1
    return None


def _mask_to_spans(mask: list[bool]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(mask):
        if value and start is None:
            start = index
        elif not value and start is not None:
            spans.append((start, index))
            start = None
    if start is not None:
        spans.append((start, len(mask)))
    return spans


def _definition_ranges(
    text: str, references: dict[str, dict[str, object]]
) -> list[SourceSpan]:
    offsets = _line_offsets(text)
    spans: list[SourceSpan] = []
    for reference in references.values():
        lines = reference.get("map")
        if not isinstance(lines, list) or len(lines) < 2:
            continue
        start = offsets[min(int(lines[0]), len(offsets) - 1)]
        end = offsets[min(int(lines[1]), len(offsets) - 1)]
        spans.append(SourceSpan(start, _rstrip_newlines(text, end)))
    return spans


def _definition_destination_edit(
    text: str, span: SourceSpan, base_url: str
) -> SourceEdit | None:
    """Return an edit resolving a safe definition, or ``None`` when unsafe."""
    block = text[span.start : span.end]
    match = re.match(r"^\s*\[[^\]]*\]:\s*", block)
    if match is None:
        return None
    destination, start, end = _read_destination(text, span.start + match.end())
    if destination is None:
        return None
    resolved = _is_safe_destination(destination, base_url)
    if resolved is None:
        return None
    return SourceEdit(start, end, resolved)


def _read_destination(
    text: str, position: int
) -> tuple[str | None, int, int]:
    """Read a Markdown link destination; return ``(value, start, end)``."""
    index = position
    while index < len(text) and text[index] in " \t":
        index += 1
    if index >= len(text):
        return None, index, index
    if text[index] == "<":
        closing = text.find(">", index + 1)
        if closing == -1:
            return None, index, index
        return text[index + 1 : closing], index + 1, closing
    value_start = index
    while index < len(text) and text[index] not in " \t\r\n":
        index += 1
    return text[value_start:index], value_start, index


class _LinkScanner:
    """Collect source edits for links and reference usages in raw Markdown."""

    def __init__(
        self, text: str, base_url: str, unsafe_labels: set[str], mask: list[bool]
    ) -> None:
        self.text = text
        self.base_url = base_url
        self.unsafe_labels = unsafe_labels
        self.mask = mask
        self.edits: list[SourceEdit] = []

    def run(self) -> list[SourceEdit]:
        index = 0
        while index < len(self.text):
            if self.mask[index]:
                index += 1
                continue
            character = self.text[index]
            if character == "\\":
                index += 2
                continue
            if character == "[":
                index = self._scan_label(index)
            elif character == "<":
                index = self._scan_autolink(index)
            else:
                index += 1
        return self.edits

    def _scan_label(self, start: int) -> int:
        label_end = self._match_square(start)
        if label_end is None:
            return start + 1
        label_source = self.text[start + 1 : label_end]
        after = label_end + 1
        if after < len(self.text) and self.text[after] == "(":
            return self._scan_inline(start, label_end, after, label_source)
        if after < len(self.text) and self.text[after] == "[":
            reference_end = self._match_square(after)
            if reference_end is None:
                return label_end + 1
            raw_reference = self.text[after + 1 : reference_end]
            key = normalizeReference(raw_reference or label_source)
            if key in self.unsafe_labels:
                self.edits.append(
                    SourceEdit(start, reference_end + 1, label_source)
                )
            return reference_end + 1
        key = normalizeReference(label_source)
        if key in self.unsafe_labels:
            self.edits.append(SourceEdit(start, label_end + 1, label_source))
        return label_end + 1

    def _scan_inline(
        self, start: int, label_end: int, open_paren: int, label_source: str
    ) -> int:
        close_paren = self._match_paren(open_paren)
        if close_paren is None:
            return label_end + 1
        destination, destination_start, destination_end = _read_destination(
            self.text, open_paren + 1
        )
        if destination is None:
            return close_paren + 1
        resolved = _is_safe_destination(destination, self.base_url)
        if resolved is None:
            self.edits.append(
                SourceEdit(start, close_paren + 1, label_source)
            )
            return close_paren + 1
        if self.text[destination_start - 1 : destination_start] == "<":
            self.edits.append(
                SourceEdit(
                    destination_start - 1,
                    destination_end + 1,
                    f"<{resolved}>",
                )
            )
        else:
            self.edits.append(
                SourceEdit(destination_start, destination_end, resolved)
            )
        return close_paren + 1

    def _scan_autolink(self, start: int) -> int:
        closing = self.text.find(">", start + 1)
        if closing == -1:
            return start + 1
        inner = self.text[start + 1 : closing]
        if " " in inner or ":" not in inner:
            return start + 1
        if _is_safe_destination(inner, self.base_url) is not None:
            return closing + 1
        self.edits.append(SourceEdit(start, closing + 1, inner))
        return closing + 1

    def _match_square(self, start: int) -> int | None:
        depth = 0
        index = start
        while index < len(self.text):
            character = self.text[index]
            if character == "\\":
                index += 2
                continue
            if character == "[":
                depth += 1
            elif character == "]":
                depth -= 1
                if depth == 0:
                    return index
            index += 1
        return None

    def _match_paren(self, start: int) -> int | None:
        depth = 0
        index = start
        quote: str | None = None
        while index < len(self.text):
            character = self.text[index]
            if character == "\\":
                index += 2
                continue
            if quote is not None:
                if character == quote:
                    quote = None
            elif character in "\"'":
                quote = character
            elif character == "(":
                depth += 1
            elif character == ")":
                depth -= 1
                if depth == 0:
                    return index
            index += 1
        return None


def _apply_source_edits(text: str, edits: list[SourceEdit]) -> str:
    result = text
    for edit in sorted(edits, key=lambda item: item.start, reverse=True):
        result = result[: edit.start] + edit.replacement + result[edit.end :]
    return result


def normalize_markdown_links(text: str, base_url: str) -> str:
    """Resolve safe hyperlinks and neutralize unsafe ones in raw Markdown."""
    parser = _build_markdown_parser()
    environment: dict[str, object] = {}
    try:
        parser.parse(text, environment)
    except Exception:
        environment = {}
    references = environment.get("references", {})
    if not isinstance(references, dict):
        references = {}
    definition_spans = _definition_ranges(text, references)
    edits: list[SourceEdit] = []
    unsafe_labels: set[str] = set()
    for label, reference in references.items():
        href = reference.get("href") if isinstance(reference, dict) else None
        raw_href = href if isinstance(href, str) else ""
        if _is_safe_destination(raw_href, base_url) is None:
            unsafe_labels.add(label)
    for span in definition_spans:
        edit = _definition_destination_edit(text, span, base_url)
        if edit is not None:
            edits.append(edit)
    offsets = _line_offsets(text)
    for label in unsafe_labels:
        reference = references.get(label, {})
        lines = reference.get("map") if isinstance(reference, dict) else None
        if not isinstance(lines, list) or len(lines) < 2:
            continue
        start = offsets[min(int(lines[0]), len(offsets) - 1)]
        end = offsets[min(int(lines[1]), len(offsets) - 1)]
        edits.append(SourceEdit(start, end, ""))
    mask = _protected_mask(text, _code_spans(text) + definition_spans)
    edits.extend(_LinkScanner(text, base_url, unsafe_labels, mask).run())
    return _apply_source_edits(text, edits)


# ---------------------------------------------------------------------------
# Document parsing
# ---------------------------------------------------------------------------


def _heading_context(blocks: object, heading_ids: tuple[int, ...]) -> str:
    if not heading_ids:
        return ""
    return blocks[heading_ids[-1]].search_text  # type: ignore[index]


def _introduction_for(
    block: ContentBlock, previous: ContentBlock | None
) -> int | None:
    if block.kind != "atomic" or previous is None:
        return None
    if previous.kind != "prose" or previous.heading_ids != block.heading_ids:
        return None
    if previous.search_text.rstrip().endswith(":"):
        return previous.block_id
    return None


def _parse_markdown(text: str, mime: str, *, window_chars: int) -> ContentDocument:
    parser = _build_markdown_parser()
    environment: dict[str, object] = {}
    tokens = parser.parse(text, environment)
    offsets = _line_offsets(text)
    blocks: list[ContentBlock] = []
    heading_stack: list[tuple[int, int]] = []
    references: dict[str, SourceSpan] = {}

    raw_references = environment.get("references", {})
    if isinstance(raw_references, dict):
        for label, reference in raw_references.items():
            lines = reference.get("map") if isinstance(reference, dict) else None
            if not isinstance(lines, list) or len(lines) < 2:
                continue
            start = offsets[min(int(lines[0]), len(offsets) - 1)]
            end = offsets[min(int(lines[1]), len(offsets) - 1)]
            references[label] = SourceSpan(start, _rstrip_newlines(text, end))

    def add_block(
        span: SourceSpan,
        kind: BlockKind,
        heading_ids: tuple[int, ...],
        search_text: str,
        labels: tuple[str, ...] = (),
        trimmable: bool = False,
    ) -> ContentBlock:
        previous = blocks[-1] if blocks else None
        block = ContentBlock(
            block_id=len(blocks),
            span=span,
            kind=kind,
            heading_ids=heading_ids,
            introduction_id=None,
            reference_labels=labels,
            search_text=search_text,
            trimmable=trimmable,
        )
        block = ContentBlock(
            block_id=block.block_id,
            span=block.span,
            kind=block.kind,
            heading_ids=block.heading_ids,
            introduction_id=_introduction_for(block, previous),
            reference_labels=block.reference_labels,
            search_text=block.search_text,
            trimmable=block.trimmable,
        )
        blocks.append(block)
        return block

    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token.level != 0:
            index += 1
            continue
        if token.type == "heading_open":
            level = int(token.tag[1])
            inline = tokens[index + 1]
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_ids = tuple(block_id for _, block_id in heading_stack)
            block = add_block(
                _map_span(token, offsets, text),
                "heading",
                heading_ids,
                _visible_text(inline),
                _collect_labels([inline]),
            )
            heading_stack.append((level, block.block_id))
            index += 3
            continue
        if token.type == "paragraph_open":
            inline = tokens[index + 1]
            add_block(
                _map_span(token, offsets, text),
                "prose",
                tuple(block_id for _, block_id in heading_stack),
                _visible_text(inline),
                _collect_labels([inline]),
                trimmable=_is_plain_inline(inline),
            )
            index += 3
            continue
        if token.type in ("fence", "code_block"):
            add_block(
                _map_span(token, offsets, text),
                "atomic",
                tuple(block_id for _, block_id in heading_stack),
                _squash(token.content),
                _collect_labels([token]),
            )
            index += 1
            continue
        if token.type in (
            "bullet_list_open",
            "ordered_list_open",
            "blockquote_open",
            "table_open",
        ):
            close = _matching_close(tokens, index)
            inner = tokens[index : close + 1]
            add_block(
                _map_span(token, offsets, text),
                "atomic",
                tuple(block_id for _, block_id in heading_stack),
                _visible_texts(inner),
                _collect_labels(inner),
            )
            index = close + 1
            continue
        if token.type == "html_block":
            add_block(
                _map_span(token, offsets, text),
                "atomic",
                tuple(block_id for _, block_id in heading_stack),
                _strip_html(token.content),
            )
            index += 1
            continue
        if token.type == "hr":
            add_block(
                _map_span(token, offsets, text),
                "atomic",
                tuple(block_id for _, block_id in heading_stack),
                "",
            )
            index += 1
            continue
        index += 1

    return ContentDocument(
        text=text, mime=mime, blocks=tuple(blocks), references=references
    )


def _plain_blocks(text: str) -> list[SourceSpan]:
    spans: list[SourceSpan] = []
    offset = 0
    start: int | None = None
    end = 0
    for line in text.splitlines(keepends=True):
        content = line.rstrip("\r\n")
        if content.strip():
            if start is None:
                start = offset
            end = offset + len(content)
        elif start is not None:
            spans.append(SourceSpan(start, end))
            start = None
        offset += len(line)
    if start is not None:
        spans.append(SourceSpan(start, end))
    return spans


def _parse_plain(text: str, mime: str) -> ContentDocument:
    blocks = tuple(
        ContentBlock(
            block_id=index,
            span=span,
            kind="prose",
            heading_ids=(),
            introduction_id=None,
            reference_labels=(),
            search_text=_squash(text[span.start : span.end]),
            trimmable=True,
        )
        for index, span in enumerate(_plain_blocks(text))
    )
    return ContentDocument(text=text, mime=mime, blocks=blocks, references={})


def parse_document(
    text: str, mime: str, *, window_chars: int
) -> ContentDocument:
    """Parse prepared text into source-mapped blocks without converting it."""
    if window_chars <= 0:
        raise ValueError("window_chars must be positive")
    if mime == MIME_MARKDOWN:
        return _parse_markdown(text, mime, window_chars=window_chars)
    if mime == MIME_PLAIN:
        return _parse_plain(text, mime)
    return _parse_plain(text, MIME_PLAIN)


def prepare_document(
    result: FetchResult, *, max_input_chars: int, window_chars: int
) -> ContentDocument:
    """Prepare a fetch result for filtering, preserving native text structure."""
    if max_input_chars <= 0:
        raise ValueError("max_input_chars must be positive")
    source = result.content
    if len(source) > max_input_chars:
        raise ValueError("input exceeds max_input_chars")
    classified = textual_mime(result.mime)
    if classified is None:
        raise LookupError(result.mime)
    if classified == MIME_HTML:
        prepared = _render_html(source, result.url)
        if len(prepared) > max_input_chars:
            raise ValueError("rendered input exceeds max_input_chars")
        prepared = normalize_markdown_links(prepared, result.url)
        return parse_document(prepared, MIME_MARKDOWN, window_chars=window_chars)
    if classified == MIME_MARKDOWN:
        prepared = normalize_markdown_links(source, result.url)
        if len(prepared) > max_input_chars:
            raise ValueError("normalized input exceeds max_input_chars")
        return parse_document(prepared, MIME_MARKDOWN, window_chars=window_chars)
    return parse_document(source, MIME_PLAIN, window_chars=window_chars)


# ---------------------------------------------------------------------------
# Scoring windows and rendering
# ---------------------------------------------------------------------------


def _chunks(value: str, size: int, overlap: int) -> list[str]:
    if size <= 0:
        size = 1
    if not value:
        return [""]
    if len(value) <= size:
        return [value]
    step = max(1, size - overlap)
    parts: list[str] = []
    position = 0
    while position < len(value):
        parts.append(value[position : position + size])
        if position + size >= len(value):
            break
        position += step
    return parts


def build_scoring_windows(
    document: ContentDocument,
    *,
    window_chars: int,
    overlap_chars: int,
    max_windows: int,
) -> tuple[ScoringWindow, ...]:
    """Produce searchable windows for every non-heading block in source order."""
    if window_chars <= 0 or max_windows <= 0:
        raise ValueError("window limits must be positive")
    if not 0 <= overlap_chars < window_chars:
        raise ValueError("overlap_chars must be below window_chars")
    searchable = [block for block in document.blocks if block.kind != "heading"]
    if not searchable:
        searchable = list(document.blocks)
    windows: list[ScoringWindow] = []
    for block in searchable:
        heading = _heading_context(document.blocks, block.heading_ids)
        prefix = heading[: window_chars // 4]
        budget = window_chars - len(prefix) - (1 if prefix else 0)
        for chunk in _chunks(block.search_text, max(1, budget), overlap_chars):
            text = f"{prefix}\n{chunk}" if prefix else chunk
            windows.append(
                ScoringWindow(
                    window_id=len(windows), block_id=block.block_id, text=text
                )
            )
            if len(windows) > max_windows:
                raise ValueError("max_windows exceeded")
    return tuple(windows)


def render_selection(document: ContentDocument, block_ids: set[int]) -> str:
    """Render selected blocks with heading, introduction, and reference closure."""
    selected = set(block_ids)
    for block in document.blocks:
        if block.block_id in selected:
            selected.update(block.heading_ids)
            if block.introduction_id is not None:
                selected.add(block.introduction_id)
    ordered = [block for block in document.blocks if block.block_id in selected]
    parts = [document.text[block.span.start : block.span.end] for block in ordered]
    required = {
        label
        for block in ordered
        for label in block.reference_labels
        if label in document.references
    }
    for label, span in sorted(
        document.references.items(), key=lambda item: item[1].start
    ):
        if label in required:
            parts.append(document.text[span.start : span.end])
    return "\n\n".join(part for part in parts if part)


def truncate_content(
    content: str, mime: str, max_chars: int
) -> tuple[str, bool]:
    """Return a syntax-safe content prefix and whether anything was omitted."""
    if max_chars < 0:
        raise ValueError("max_chars must be nonnegative")
    if len(content) <= max_chars:
        return content, False
    if mime == MIME_MARKDOWN:
        prefix = _markdown_prefix(content, max_chars)
    else:
        prefix = _plain_prefix(content, max_chars)
    return prefix, True


_URL_RE = re.compile(r"https?://[^\s]+", re.IGNORECASE)


def _url_safe_cut(content: str, max_chars: int) -> int:
    """Return a cut position that never splits an HTTP(S) URL token."""
    for match in _URL_RE.finditer(content):
        if match.start() >= max_chars:
            break
        if match.end() > max_chars:
            return match.start()
    return max_chars


def _plain_prefix(content: str, max_chars: int) -> str:
    """Longest prefix bounded by a word/newline boundary, never splitting a URL."""
    if max_chars <= 0:
        return ""
    candidate = content[: _url_safe_cut(content, max_chars)]
    boundary = max(
        candidate.rfind("\n"),
        candidate.rfind(" "),
        candidate.rfind("\t"),
    )
    if boundary > 0:
        return candidate[:boundary]
    return candidate


def _render_blocks(
    document: ContentDocument, blocks: list[ContentBlock]
) -> str:
    """Render evidence spans plus exactly their required definitions."""
    parts = [
        document.text[block.span.start : block.span.end] for block in blocks
    ]
    required = {
        label
        for block in blocks
        for label in block.reference_labels
        if label in document.references
    }
    for label, span in sorted(
        document.references.items(), key=lambda item: item[1].start
    ):
        if label in required:
            parts.append(document.text[span.start : span.end])
    return "\n\n".join(part for part in parts if part)


def _markdown_prefix(content: str, max_chars: int) -> str:
    """Prefix of whole source blocks; drop orphan headings; keep definitions."""
    if max_chars <= 0:
        return ""
    document = _parse_markdown(
        content, MIME_MARKDOWN, window_chars=max(1, max_chars)
    )
    selected: list[ContentBlock] = []
    pending: list[ContentBlock] = []
    has_content = False
    for block in document.blocks:
        if block.kind == "heading":
            pending.append(block)
            continue
        has_content = True
        candidate = [*selected, *pending, block]
        if len(_render_blocks(document, candidate)) <= max_chars:
            selected = candidate
            pending = []
            continue
        if block.trimmable:
            base = _render_blocks(document, [*selected, *pending])
            separator = 2 if base else 0
            available = max_chars - len(base) - separator
            if available > 0:
                piece = _plain_prefix(
                    content[block.span.start : block.span.end], available
                )
                if piece:
                    return "\n\n".join(
                        part for part in (base, piece) if part
                    )
        break
    if not selected and pending and not has_content:
        rendered = _render_blocks(document, pending)
        if len(rendered) <= max_chars:
            return rendered
    return _render_blocks(document, selected)
