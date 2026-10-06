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
from dataclasses import dataclass, replace
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
    "build_selection_units",
    "normalize_markdown_links",
    "parse_document",
    "prepare_document",
    "render_selection",
    "truncate_content",
]

# "context" and "table_header" blocks exist only in selection-unit documents:
# they are rendered as dependencies of a unit but never scored on their own.
BlockKind = Literal["heading", "prose", "atomic", "context", "table_header"]


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


def _line_range(offsets: list[int], first: int, last: int) -> tuple[int, int]:
    limit = len(offsets) - 1
    return offsets[min(first, limit)], offsets[min(last, limit)]


def _map_span(token: Token, offsets: list[int], text: str) -> SourceSpan:
    first, last = token.map or [0, 0]
    start, end = _line_range(offsets, first, last)
    return SourceSpan(start, _rstrip_newlines(text, end))


def _reference_lines(reference: object) -> tuple[int, int] | None:
    lines = reference.get("map") if isinstance(reference, dict) else None
    if not isinstance(lines, list) or len(lines) < 2:
        return None
    return int(lines[0]), int(lines[1])


def _reference_spans(
    text: str, offsets: list[int], references: dict[str, object]
) -> dict[str, SourceSpan]:
    spans: dict[str, SourceSpan] = {}
    for label, reference in references.items():
        lines = _reference_lines(reference)
        if lines is None:
            continue
        start, end = _line_range(offsets, *lines)
        spans[label] = SourceSpan(start, _rstrip_newlines(text, end))
    return spans


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
        if child.type in ("text", "code_inline", "image"):
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
    labels = (
        child.meta.get("label")
        for token in tokens
        if token.type == "inline" and token.children
        for child in token.children
        if child.meta
    )
    return tuple(dict.fromkeys(label for label in labels if label))


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------


def _resolve_safe_destination(destination: str, base_url: str) -> str | None:
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


def _sanitize_html(content: str, base_url: str) -> str:
    """Drop scripts/styles and unsafe hrefs; unparseable HTML passes through."""
    parser = lxml_html.HTMLParser(recover=True, no_network=True)
    try:
        document = lxml_html.document_fromstring(
            content, parser=parser, base_url=base_url
        )
    except Exception:
        return content
    for element in document.xpath("//script|//style"):
        if element.getparent() is not None:
            element.drop_tree()
    for anchor in document.xpath("//a[@href]"):
        if _resolve_safe_destination(anchor.get("href", ""), base_url) is None:
            del anchor.attrib["href"]
    return str(lxml_html.tostring(document, encoding="unicode"))


def _render_html(content: str, base_url: str) -> str:
    generated = DefaultMarkdownGenerator().generate_markdown(
        _sanitize_html(content, base_url), base_url, citations=False
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


def _code_mask(text: str) -> list[bool]:
    """Mask fenced/indented code, HTML blocks, and inline code spans."""
    parser = _build_markdown_parser()
    try:
        tokens = parser.parse(text, {})
    except Exception:
        tokens = []
    offsets = _line_offsets(text)
    block_spans = [
        _map_span(token, offsets, text)
        for token in tokens
        if token.type in ("fence", "code_block", "html_block") and token.map
    ]
    mask = _protected_mask(text, block_spans)
    index = 0
    while index < len(text):
        if not mask[index] and text[index] == "`":
            run = _backtick_run_length(text, index)
            closing = _find_backtick_run(text, index + run, run)
            if closing is not None:
                mask[index : closing + run] = [True] * (closing + run - index)
                index = closing + run
                continue
        index += 1
    return mask


def _backtick_run_length(text: str, start: int) -> int:
    length = 1
    while start + length < len(text) and text[start + length] == "`":
        length += 1
    return length


def _find_backtick_run(text: str, start: int, run: int) -> int | None:
    index = start
    while index < len(text):
        if text[index] != "`":
            index += 1
            continue
        length = _backtick_run_length(text, index)
        if length == run:
            return index
        index += length
    return None


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
    resolved = _resolve_safe_destination(destination, base_url)
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
        if self.text.startswith("(", after):
            return self._scan_inline(start, label_end, after, label_source)
        if self.text.startswith("[", after):
            reference_end = self._match_square(after)
            if reference_end is None:
                return label_end + 1
            raw_reference = self.text[after + 1 : reference_end]
            key = normalizeReference(raw_reference or label_source)
            link_end = reference_end + 1
        else:
            key = normalizeReference(label_source)
            link_end = label_end + 1
        if key in self.unsafe_labels:
            self.edits.append(SourceEdit(start, link_end, label_source))
        return link_end

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
        resolved = _resolve_safe_destination(destination, self.base_url)
        if resolved is None:
            edit = SourceEdit(start, close_paren + 1, label_source)
        elif self.text[destination_start - 1] == "<":
            edit = SourceEdit(
                destination_start - 1, destination_end + 1, f"<{resolved}>"
            )
        else:
            edit = SourceEdit(destination_start, destination_end, resolved)
        self.edits.append(edit)
        return close_paren + 1

    def _scan_autolink(self, start: int) -> int:
        closing = self.text.find(">", start + 1)
        if closing == -1:
            return start + 1
        inner = self.text[start + 1 : closing]
        if " " in inner or ":" not in inner:
            return start + 1
        if _resolve_safe_destination(inner, self.base_url) is None:
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


def _reference_href(reference: object) -> str:
    href = reference.get("href") if isinstance(reference, dict) else None
    return href if isinstance(href, str) else ""


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
    offsets = _line_offsets(text)
    definition_spans = list(_reference_spans(text, offsets, references).values())
    unsafe_labels = {
        label
        for label, reference in references.items()
        if _resolve_safe_destination(_reference_href(reference), base_url) is None
    }
    edits = [
        edit
        for span in definition_spans
        if (edit := _definition_destination_edit(text, span, base_url))
    ]
    for label in unsafe_labels:
        lines = _reference_lines(references[label])
        if lines is not None:
            # Whole lines (including the newline) so no blank line remains.
            edits.append(SourceEdit(*_line_range(offsets, *lines), ""))
    mask = [
        in_code or in_definition
        for in_code, in_definition in zip(
            _code_mask(text),
            _protected_mask(text, definition_spans),
            strict=True,
        )
    ]
    edits.extend(_LinkScanner(text, base_url, unsafe_labels, mask).run())
    return _apply_source_edits(text, edits)


# ---------------------------------------------------------------------------
# Document parsing
# ---------------------------------------------------------------------------


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


_CONTAINER_OPENERS = frozenset(
    ["bullet_list_open", "ordered_list_open", "blockquote_open", "table_open"]
)


def _parse_markdown(text: str) -> ContentDocument:
    environment: dict[str, object] = {}
    tokens = _build_markdown_parser().parse(text, environment)
    offsets = _line_offsets(text)
    raw_references = environment.get("references", {})
    references = (
        _reference_spans(text, offsets, raw_references)
        if isinstance(raw_references, dict)
        else {}
    )
    blocks: list[ContentBlock] = []
    heading_stack: list[tuple[int, int]] = []

    def add_block(
        token: Token,
        kind: BlockKind,
        search_text: str,
        labels: tuple[str, ...] = (),
        trimmable: bool = False,
    ) -> None:
        block = ContentBlock(
            block_id=len(blocks),
            span=_map_span(token, offsets, text),
            kind=kind,
            heading_ids=tuple(block_id for _, block_id in heading_stack),
            introduction_id=None,
            reference_labels=labels,
            search_text=search_text,
            trimmable=trimmable,
        )
        previous = blocks[-1] if blocks else None
        introduction_id = _introduction_for(block, previous)
        blocks.append(replace(block, introduction_id=introduction_id))

    index = 0
    while index < len(tokens):
        token = tokens[index]
        next_index = index + 1
        if token.level != 0:
            pass
        elif token.type == "heading_open":
            level = int(token.tag[1])
            inline = tokens[index + 1]
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            add_block(
                token, "heading", _visible_text(inline), _collect_labels([inline])
            )
            heading_stack.append((level, blocks[-1].block_id))
            next_index = index + 3
        elif token.type == "paragraph_open":
            inline = tokens[index + 1]
            add_block(
                token,
                "prose",
                _visible_text(inline),
                _collect_labels([inline]),
                trimmable=_is_plain_inline(inline),
            )
            next_index = index + 3
        elif token.type in ("fence", "code_block"):
            add_block(
                token, "atomic", _squash(token.content), _collect_labels([token])
            )
        elif token.type in _CONTAINER_OPENERS:
            close = _matching_close(tokens, index)
            inner = tokens[index : close + 1]
            visible = " ".join(
                _visible_text(item) for item in inner if item.type == "inline"
            )
            add_block(token, "atomic", _squash(visible), _collect_labels(inner))
            next_index = close + 1
        elif token.type == "html_block":
            add_block(token, "atomic", _strip_html(token.content))
        elif token.type == "hr":
            add_block(token, "atomic", "")
        index = next_index

    return ContentDocument(
        text=text, mime=MIME_MARKDOWN, blocks=tuple(blocks), references=references
    )


def _strip_html(value: str) -> str:
    return _squash(html_lib.unescape(re.sub(r"<[^>]+>", " ", value)))


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


def _parse_plain(text: str) -> ContentDocument:
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
    return ContentDocument(
        text=text, mime=MIME_PLAIN, blocks=blocks, references={}
    )


def parse_document(
    text: str, mime: str, *, window_chars: int
) -> ContentDocument:
    """Parse prepared text into source-mapped blocks without converting it."""
    if window_chars <= 0:
        raise ValueError("window_chars must be positive")
    if mime == MIME_MARKDOWN:
        return _parse_markdown(text)
    return _parse_plain(text)


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


# ---------------------------------------------------------------------------
# Selection units
# ---------------------------------------------------------------------------

_SECTION_MAX_LEVEL = 3
_SENTENCE_BREAK = re.compile(r"[.!?][\"')\]]*(\s)")
_LIST_OPENERS = frozenset(["bullet_list_open", "ordered_list_open"])
_CONTEXT_KINDS = frozenset(["heading", "context", "table_header"])


@dataclass(frozen=True, slots=True)
class _Candidate:
    """A selection unit expressed in the parsed document's block ids."""

    span: SourceSpan
    kind: BlockKind
    heading_ids: tuple[int, ...]
    introduction_id: int | None
    table_header: SourceSpan | None
    reference_labels: tuple[str, ...]
    search_text: str


def _length(span: SourceSpan) -> int:
    return span.end - span.start


def _heading_level(source: str) -> int:
    stripped = source.lstrip()
    if stripped.startswith("#"):
        return len(stripped) - len(stripped.lstrip("#"))
    return 1 if source.rstrip().endswith("=") else 2


def _sections(document: ContentDocument) -> list[list[ContentBlock]]:
    """Group blocks into sections starting at each level 1-3 heading."""
    sections: list[list[ContentBlock]] = [[]]
    for block in document.blocks:
        starts_section = block.kind == "heading" and (
            _heading_level(document.text[block.span.start : block.span.end])
            <= _SECTION_MAX_LEVEL
        )
        if starts_section and sections[-1]:
            sections.append([])
        sections[-1].append(block)
    return [section for section in sections if section]


def _join_text(parts: list[str]) -> str:
    return " ".join(part for part in parts if part)


def _union_labels(groups: list[tuple[str, ...]]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(label for group in groups for label in group))


def _from_block(block: ContentBlock) -> _Candidate:
    return _Candidate(
        span=block.span,
        kind=block.kind,
        heading_ids=block.heading_ids,
        introduction_id=block.introduction_id,
        table_header=None,
        reference_labels=block.reference_labels,
        search_text=block.search_text,
    )


def _from_section(section: list[ContentBlock]) -> _Candidate:
    return _Candidate(
        span=SourceSpan(section[0].span.start, section[-1].span.end),
        kind="atomic",
        heading_ids=section[0].heading_ids,
        introduction_id=None,
        table_header=None,
        reference_labels=_union_labels([b.reference_labels for b in section]),
        search_text=_join_text([block.search_text for block in section]),
    )


def _merge(first: _Candidate, second: _Candidate) -> _Candidate:
    return replace(
        first,
        span=SourceSpan(first.span.start, second.span.end),
        kind="atomic",
        reference_labels=_union_labels(
            [first.reference_labels, second.reference_labels]
        ),
        search_text=_join_text([first.search_text, second.search_text]),
    )


def _merge_small(
    candidates: list[_Candidate], min_chars: int, max_chars: int
) -> list[_Candidate]:
    """Absorb following units into an undersized one, as SemanticChunker does.

    Callers pass the units of a single section, so merging never crosses a
    heading. A table piece never starts a merge tail: its header would be lost.
    """
    merged: list[_Candidate] = []
    index = 0
    while index < len(candidates):
        current = candidates[index]
        index += 1
        if _length(current.span) < min_chars:
            while (
                index < len(candidates)
                and candidates[index].table_header is None
                and candidates[index].span.end - current.span.start <= max_chars
            ):
                current = _merge(current, candidates[index])
                index += 1
        merged.append(current)
    return merged


def _syntax_mask(source: str) -> list[bool]:
    """Mark code and link syntax so a split never lands inside them."""
    mask = _code_mask(source)
    matcher = _LinkScanner(source, "", set(), mask)
    index = 0
    while index < len(source):
        label_end = (
            matcher._match_square(index)
            if source[index] == "[" and not mask[index]
            else None
        )
        if label_end is None:
            index += 1
            continue
        end = label_end + 1
        if source.startswith("(", end):
            close = matcher._match_paren(end)
            end = end if close is None else close + 1
        elif source.startswith("[", end):
            close = matcher._match_square(end)
            end = end if close is None else close + 1
        mask[index:end] = [True] * (end - index)
        index = end
    return mask


def _best_break(source: str, start: int, stop: int, mask: list[bool]) -> int | None:
    """Last unprotected break in ``(start, stop]``: line, sentence, then space."""
    for index in range(stop, start, -1):
        if source[index] == "\n" and not mask[index]:
            return index
    sentences = [
        match.start(1)
        for match in _SENTENCE_BREAK.finditer(source, start, stop + 1)
        if not mask[match.start(1)]
    ]
    if sentences:
        return sentences[-1]
    for index in range(stop, start, -1):
        if source[index].isspace() and not mask[index]:
            return index
    return None


def _next_break(source: str, start: int, mask: list[bool]) -> int | None:
    for index in range(start, len(source)):
        if source[index].isspace() and not mask[index]:
            return index
    return None


def _split_source(
    source: str, limit: int, mask: list[bool]
) -> list[tuple[int, int]]:
    """Split text into ``limit``-sized pieces at the best unprotected breaks.

    A run with no unprotected break (one huge link or word) stays oversized.
    """
    pieces: list[tuple[int, int]] = []
    start, end = 0, len(source)
    while end - start > limit:
        cut = _best_break(source, start, start + limit, mask)
        if cut is None:
            cut = _next_break(source, start + limit, mask)
        if cut is None:
            break
        piece_end = cut
        while piece_end > start and source[piece_end - 1].isspace():
            piece_end -= 1
        pieces.append((start, piece_end))
        start = cut
        while start < end and source[start].isspace():
            start += 1
    if start < end:
        pieces.append((start, end))
    return pieces


def _group_lines(
    source: str, maps: list[list[int]], budget: int
) -> list[tuple[int, int]]:
    """Pack consecutive line ranges (items or rows) into pieces of ``budget``."""
    offsets = _line_offsets(source)
    groups: list[tuple[int, int]] = []
    for first, last in maps:
        start, end = _line_range(offsets, first, last)
        end = _rstrip_newlines(source, end)
        if groups and end - groups[-1][0] <= budget:
            groups[-1] = (groups[-1][0], end)
        else:
            groups.append((start, end))
    return groups


def _structured_pieces(
    source: str, limit: int
) -> tuple[tuple[int, int] | None, list[tuple[int, int]]]:
    """Split a list by top-level item or a table by row; else no pieces.

    Tables return the header line range, which every row piece repeats.
    """
    tokens = _build_markdown_parser().parse(source, {})
    first = tokens[0].type if tokens else ""
    if first in _LIST_OPENERS:
        items = [
            token.map
            for token in tokens
            if token.type == "list_item_open" and token.level == 1 and token.map
        ]
        return None, _group_lines(source, items, limit)
    if first != "table_open":
        return None, []
    body = next(
        (token.map for token in tokens if token.type == "tbody_open" and token.map),
        None,
    )
    if body is None:
        return None, []
    header_end = _rstrip_newlines(source, _line_offsets(source)[body[0]])
    rows = [
        token.map
        for token in tokens
        if token.type == "tr_open" and token.map and token.map[0] >= body[0]
    ]
    budget = max(1, limit - header_end - 1)
    return (0, header_end), _group_lines(source, rows, budget)


def _markdown_search_text(source: str) -> str:
    tokens = _build_markdown_parser().parse(source, {})
    return _join_text([_visible_text(t) for t in tokens if t.type == "inline"])


def _split_block(
    document: ContentDocument, block: ContentBlock, limit: int
) -> list[_Candidate]:
    """Split an oversized block into source-slice pieces of at most ``limit``.

    Prose splits at line, sentence, then word boundaries outside link and
    code syntax; lists split between top-level items; tables split between
    rows and repeat the header. Code, quotes, and HTML stay whole.
    """
    if _length(block.span) <= limit:
        return [_from_block(block)]
    source = document.text[block.span.start : block.span.end]
    markdown = document.mime == MIME_MARKDOWN
    header: tuple[int, int] | None = None
    if block.kind == "prose":
        mask = _syntax_mask(source) if markdown else [False] * len(source)
        pieces = _split_source(source, limit, mask)
    elif markdown:
        header, pieces = _structured_pieces(source, limit)
    else:
        pieces = []
    if len(pieces) < 2:
        return [_from_block(block)]
    offset = block.span.start
    table_header = (
        SourceSpan(offset + header[0], offset + header[1]) if header else None
    )
    # ponytail: pieces inherit the whole block's reference labels, so a piece
    # may carry a definition it does not use; per-piece labels need a reparse
    # with the document's reference map.
    return [
        _Candidate(
            span=SourceSpan(offset + start, offset + end),
            kind=block.kind,
            heading_ids=block.heading_ids,
            introduction_id=block.introduction_id,
            table_header=table_header,
            reference_labels=block.reference_labels,
            search_text=(
                _markdown_search_text(source[start:end])
                if markdown
                else _squash(source[start:end])
            ),
        )
        for start, end in pieces
    ]


def _markdown_candidates(
    document: ContentDocument, unit_chars: int, min_unit_chars: int
) -> list[_Candidate]:
    """A section that fits is one unit; an oversized one yields its blocks,
    split when oversized and merged when undersized, within that section."""
    candidates: list[_Candidate] = []
    for section in _sections(document):
        span = SourceSpan(section[0].span.start, section[-1].span.end)
        if _length(span) <= unit_chars:
            candidates.append(_from_section(section))
            continue
        pieces = [
            piece
            for block in section
            if block.kind != "heading"
            for piece in _split_block(document, block, unit_chars)
        ]
        candidates.extend(_merge_small(pieces, min_unit_chars, unit_chars))
    return candidates


def _emit_units(
    document: ContentDocument, candidates: list[_Candidate]
) -> ContentDocument:
    """Materialize candidates plus the context blocks they depend on."""
    blocks: list[ContentBlock] = []
    contexts: dict[tuple[int, int, str], int] = {}

    def add(
        span: SourceSpan,
        kind: BlockKind,
        search_text: str,
        heading_ids: tuple[int, ...] = (),
        introduction_id: int | None = None,
        labels: tuple[str, ...] = (),
    ) -> int:
        blocks.append(
            ContentBlock(
                block_id=len(blocks),
                span=span,
                kind=kind,
                heading_ids=heading_ids,
                introduction_id=introduction_id,
                reference_labels=labels,
                search_text=search_text,
                trimmable=False,
            )
        )
        return blocks[-1].block_id

    def context(span: SourceSpan, kind: BlockKind, search_text: str) -> int:
        key = (span.start, span.end, kind)
        if key not in contexts:
            contexts[key] = add(span, kind, search_text)
        return contexts[key]

    def context_of(block_id: int, kind: BlockKind) -> int:
        source = document.blocks[block_id]
        return context(source.span, kind, source.search_text)

    for candidate in candidates:
        heading_ids = tuple(
            context_of(block_id, "heading") for block_id in candidate.heading_ids
        )
        if candidate.table_header is not None:
            header = candidate.table_header
            heading_ids += (
                context(
                    header,
                    "table_header",
                    _squash(document.text[header.start : header.end]),
                ),
            )
        introduction_id = (
            None
            if candidate.introduction_id is None
            else context_of(candidate.introduction_id, "context")
        )
        add(
            candidate.span,
            candidate.kind,
            candidate.search_text,
            heading_ids,
            introduction_id,
            candidate.reference_labels,
        )
    return replace(document, blocks=tuple(blocks))


def build_selection_units(
    document: ContentDocument, *, unit_chars: int, min_unit_chars: int
) -> ContentDocument:
    """Regroup parsed blocks into selection units of about ``unit_chars``.

    Markdown follows the RAG ``SemanticChunker`` rules, except that merging
    stays inside one section: a level 1-3 section that fits is one unit; an
    oversized section falls back to its blocks (split when oversized), and a
    block unit under ``min_unit_chars`` absorbs following units of the same
    section while the result fits. Plain text keeps paragraph units, splitting
    only oversized ones. Units remain exact source slices; headings, table
    headers, and introductions become render-only context blocks.
    """
    if unit_chars <= 0 or min_unit_chars <= 0:
        raise ValueError("unit sizes must be positive")
    if document.mime == MIME_MARKDOWN:
        candidates = _markdown_candidates(
            document, unit_chars, min(min_unit_chars, unit_chars)
        )
    else:
        candidates = [
            piece
            for block in document.blocks
            for piece in _split_block(document, block, unit_chars)
        ]
    return _emit_units(document, candidates)


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
    searchable = [
        block for block in document.blocks if block.kind not in _CONTEXT_KINDS
    ]
    if not searchable:
        searchable = list(document.blocks)
    windows: list[ScoringWindow] = []
    for block in searchable:
        headings = [
            document.blocks[block_id].search_text
            for block_id in block.heading_ids
            if document.blocks[block_id].kind == "heading"
        ]
        heading = headings[-1] if headings else ""
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
    return _render_blocks(document, ordered)


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
    boundary = max(candidate.rfind(separator) for separator in "\n \t")
    if boundary > 0:
        return candidate[:boundary]
    return candidate


def _render_blocks(
    document: ContentDocument, blocks: list[ContentBlock]
) -> str:
    """Render evidence spans plus exactly their required definitions.

    Spans render in source order; a span inside an already rendered one is
    skipped. Spans separated only by whitespace keep the original gap, and
    row pieces of one table join with a newline so the table stays intact.
    """
    text = document.text
    kept: list[ContentBlock] = []
    for block in sorted(blocks, key=lambda item: (item.span.start, -item.span.end)):
        if kept and block.span.end <= max(item.span.end for item in kept):
            continue
        kept.append(block)
    required = {label for block in kept for label in block.reference_labels}
    definitions = sorted(
        (
            span
            for label, span in document.references.items()
            if label in required
            and not any(
                item.span.start <= span.start and span.end <= item.span.end
                for item in kept
            )
        ),
        key=lambda span: span.start,
    )
    rendered = ""
    previous: ContentBlock | None = None
    for block in kept:
        part = text[block.span.start : block.span.end]
        if not part:
            continue
        if previous is None:
            rendered = part
        else:
            gap = text[previous.span.end : block.span.start]
            if gap.isspace():
                separator = gap
            elif _same_table(document, previous, block):
                separator = "\n"
            else:
                separator = "\n\n"
            rendered += separator + part
        previous = block
    parts = [rendered, *(text[span.start : span.end] for span in definitions)]
    return "\n\n".join(part for part in parts if part)


def _table_of(document: ContentDocument, block: ContentBlock) -> int | None:
    if block.kind == "table_header":
        return block.block_id
    if block.heading_ids:
        last = document.blocks[block.heading_ids[-1]]
        if last.kind == "table_header":
            return last.block_id
    return None


def _same_table(
    document: ContentDocument, first: ContentBlock, second: ContentBlock
) -> bool:
    table = _table_of(document, second)
    return table is not None and table == _table_of(document, first)


def _markdown_prefix(content: str, max_chars: int) -> str:
    """Prefix of whole source blocks; drop orphan headings; keep definitions."""
    if max_chars <= 0:
        return ""
    document = _parse_markdown(content)
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
