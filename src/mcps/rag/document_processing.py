"""Document processing, traversal, and chunking."""

import hashlib
import logging
import os
import re
from collections.abc import Generator
from pathlib import Path
from typing import override

import frontmatter
from yaml.parser import ParserError

from .interfaces import (
    Chunk,
    Document,
    IChunker,
    IDocumentProcessor,
    IFileTraversal,
    Metadata,
)

SUMMARY_CHUNK_POSITION = -1


def _leading_line_delta(text: str) -> int:
    stripped = text.lstrip()
    if not stripped:
        return 0
    leading_chars = len(text) - len(stripped)
    return text[:leading_chars].count("\n")


def extract_wikilinks(content: str) -> list[str]:
    """Extract wikilinks from markdown content.
    
    Handles the following wikilink formats:
    - Basic wikilinks: [[Note Name]]
    - Wikilinks with display text: [[Note Name|Display Text]]
    - Wikilinks with headers: [[Note Name#Header]]
    - Wikilinks with both: [[Note Name#Header|Display Text]]
    - Image wikilinks: ![[Note Name]]
    - Wikilinks with spaces and special characters
    - Wikilinks with nested brackets: [[Note [with] brackets]]
    
    Returns only the note name portion, without the header or display text.
    """
    wikilink_pattern = r'!?\[\[((?:[^\[\]]|\[[^\[\]]*\])*?)(?:[#|][^\]]*?)?\]\]'
    matches = re.findall(wikilink_pattern, content)
    # Filter out empty matches but preserve trailing spaces for compatibility
    filtered_matches = [match for match in matches if match.strip()]
    return list(set(filtered_matches))  # Remove duplicates


def extract_content_tags(text: str) -> list[str]:
    """Extract hashtags from markdown content."""
    # Pattern for #tag (hashtags)
    tag_pattern = r'#([a-zA-Z][a-zA-Z0-9_-]*)'
    matches = re.findall(tag_pattern, text)
    return list(set(matches))  # Remove duplicates


def create_chunk(
    document: Document,
    content: str,
    position: int,
    line_offset: int = 0,
) -> Chunk:
    """Create a chunk from document and content."""
    chunk_id = f"{document.id}_{position}"
    chunk_content = content.strip()
    is_summary_chunk = position == SUMMARY_CHUNK_POSITION
    content_offset = (
        0 if is_summary_chunk else line_offset + _leading_line_delta(content)
    )

    metadata_source_text = document.content if is_summary_chunk else content

    # Extract wikilinks from chunk content
    outgoing_links = extract_wikilinks(metadata_source_text)

    # Extract tags from chunk content and combine with document tags
    content_tags = extract_content_tags(metadata_source_text)
    combined_tags = list(set(document.tags + content_tags))

    return Chunk(
        id=chunk_id,
        content=chunk_content,
        title=document.metadata.title,
        description=document.metadata.description,
        source=document.metadata.source,
        outgoing_links=outgoing_links,
        tags=combined_tags,
        source_path=document.source_path,
        wikilink_name=document.wikilink_name,
        modified_at=document.modified_at,
        position=position,
        offset=content_offset,
        file_size=document.file_size,
    )

logger = logging.getLogger("mcps.documents")

default_skip_patterns = [
    r'^\..*',
    r'node_modules/',
    r'__pycache__/',
    r'/worktrees/',
    r'^scripts/',
    r'^templates/',
    r'^prompts/',
]


class MarkdownFileTraversal(IFileTraversal):
    """File traversal implementation for markdown files."""

    def __init__(
        self,
        base_path: Path,
        skip_patterns: list[str] = default_skip_patterns,
    ):
        self.base_path = base_path
        self.skip_patterns = skip_patterns

    def _is_path_allowed(self, path: Path) -> bool:
        """Check if the path is allowed based on skip patterns."""
        relative_path = str(path.relative_to(self.base_path))
        return not any(
            re.search(pattern, relative_path) for pattern in self.skip_patterns
        )

    @override
    def find_files(self) -> Generator[Path]:
        """Find markdown files to process."""

        if not self.base_path.exists():
            logger.warning(f"Search path does not exist: {self.base_path}")
            yield from ()
            return

        yield from (
            self._find_files_recursive(self.base_path,'.md')
        )

    def  _find_files_recursive(self,root_path: Path, extension: str) -> Generator[Path]:
        for entry in os.scandir(root_path):
            file_path = Path(entry.path)
            if self._is_path_allowed(file_path) :
                if entry.is_dir(follow_symlinks=False):
                    # Recursively yield results from subdirectories
                    yield from self._find_files_recursive(file_path, extension)
                elif entry.is_file():
                    if entry.name.endswith(extension):
                            yield file_path


class MarkdownProcessor(IDocumentProcessor):
    """Markdown document processor."""

    def __init__(self, base_path: Path):
        self.base_path = base_path

    @override
    async def process(self, file_path: Path) -> Document:
        """Process a single markdown document file."""
        try:
            with open(file_path, encoding='utf-8') as f:
                post = frontmatter.load(f)
        except ParserError:
            post = frontmatter.Post(
                content=file_path.read_text(encoding='utf-8', errors='replace')
            )
        if not post.content:
            logger.warning("File %s is empty",file_path)
        # Extract metadata
        metadata = Metadata(
            source=self._metadata_as_str(post.metadata, 'source'),
            title=self._metadata_as_str(post.metadata, 'title'),
            description=self._metadata_as_str(post.metadata, 'description'),
        )

        # Extract tags from frontmatter only
        tags = self._extract_frontmatter_tags(post)

        # Get file modification time
        stat = file_path.stat()
        modified_at = stat.st_mtime
        size = stat.st_size
        # Generate document ID
        doc_id = self._generate_document_id(file_path)

        return Document(
            id=doc_id,
            content=post.content,
            metadata=metadata,
            tags=tags,
            source_path=file_path.relative_to(self.base_path).as_posix(),
            file_size=size,
            wikilink_name=file_path.stem,
            modified_at=modified_at,
        )

    def _metadata_as_str(self, metadata: dict[str, object], field) -> str | None:
        """Convert metadata dictionary value to a string representation."""
        return str(metadata.get(field, '')) if field in metadata else None

    def _extract_frontmatter_tags(self, content: frontmatter.Post) -> list[str]:
        """Extract tags from frontmatter only."""
        fm_tags = content.metadata.get('tags', [])
        # convert string or list of tags to a set to avoid duplicates
        if isinstance(fm_tags, str):
            fm_tags = [fm_tags]
        elif not isinstance(fm_tags, list):
            fm_tags = []
        return list(set(fm_tags))  # Remove duplicates

    def _generate_document_id(self, file_path: Path) -> str:
        """Generate a unique document ID."""
        # Use file path hash for consistent ID
        path_str = str(file_path.absolute())
        return hashlib.md5(path_str.encode()).hexdigest()


class FixedSizeChunker(IChunker):
    """Fixed size text chunker with overlap."""

    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, document: Document) -> Generator[Chunk]:
        """Split a document into fixed-size chunks with overlap."""
        content = document.content

        # Simple character-based chunking
        start = 0
        start_line = 0
        position = 0
        chunk_count = 0

        while start < len(content):
            end = min(start + self.chunk_size, len(content))
            broke_at_word_boundary = False

            # Try to break at word boundaries
            if end < len(content):
                # Look for the last space within the chunk
                last_space = content.rfind(' ', start, end)
                if last_space > start:
                    end = last_space
                    broke_at_word_boundary = True

            chunk_content = content[start:end]

            if chunk_content.strip():  # Only create non-empty chunks
                chunk = create_chunk(
                    document,
                    chunk_content,
                    position,
                    start_line,
                )

                yield chunk
                position += 1
                chunk_count += 1

            # Move start position with overlap
            next_start = end if broke_at_word_boundary else max(
                start + self.chunk_size - self.overlap,
                end,
            )
            start_line += content[start:next_start].count("\n")
            start = next_start

        logger.debug(f"Created {chunk_count} chunks from document {document.id}")


class SemanticChunker(IChunker):
    """Semantic chunker that splits on markdown sections."""

    def __init__(self, max_chunk_size: int = 1000, min_chunk_size: int = 500):
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size

    def chunk(self, document: Document) -> Generator[Chunk]:
        """Split document into semantic chunks based on markdown structure."""
        content = document.content
        chunk_count = 0
        if not content.strip():
            yield from ()
        else:
            sections = self._split_by_headers(content)
            pieces = [
                piece
                for section, line_offset in sections
                for piece in self._split_section(section, line_offset)
            ]
            position = 0
            i = 0

            while i < len(pieces):
                current_piece, current_offset = pieces[i]

                if len(current_piece.strip()) < self.min_chunk_size:
                    merged_pieces = [current_piece]
                    j = i + 1

                    while j < len(pieces):
                        next_piece = pieces[j][0]
                        if (
                            self._joined_length([*merged_pieces, next_piece])
                            > self.max_chunk_size
                        ):
                            break
                        merged_pieces.append(next_piece)
                        j += 1

                    merged_content = "\n\n".join(merged_pieces)
                    if merged_content.strip():
                        chunk = create_chunk(
                            document,
                            merged_content,
                            position,
                            current_offset,
                        )
                        yield chunk
                        position += 1
                        chunk_count += 1
                    i = j
                    continue

                chunk = create_chunk(
                    document,
                    current_piece,
                    position,
                    current_offset,
                )
                yield chunk
                position += 1
                chunk_count += 1
                i += 1

        logger.info(
            "Created %s semantic chunks from document %s",
            chunk_count,
            document.source_path,
        )

    def _split_by_headers(self, content: str) -> list[tuple[str, int]]:
        """Split content by markdown headers."""
        # Split by headers while keeping the header with the content
        header_pattern = r'^(#{1,3}\s+.+)$'
        lines = content.splitlines(keepends=True)
        sections = []
        current_section = []
        current_line_offset = 0
        line_offset = 0

        for line in lines:
            if re.match(header_pattern, line) and current_section:
                # Start new section
                sections.append((''.join(current_section), current_line_offset))
                current_section = [line]
                current_line_offset = line_offset
            else:
                current_section.append(line)
            line_offset += 1

        if current_section:
            sections.append((''.join(current_section), current_line_offset))

        return sections

    def _split_section(self, section: str, line_offset: int) -> list[tuple[str, int]]:
        if len(section) <= self.max_chunk_size:
            return [(section, line_offset)]

        return self._split_large_section(section, line_offset)

    def _split_large_section(
        self,
        section: str,
        line_offset: int,
    ) -> list[tuple[str, int]]:
        """Split large sections into smaller chunks."""
        paragraphs = self._split_with_line_offsets(section, line_offset)
        units = [
            unit
            for paragraph, paragraph_line_offset in paragraphs
            for unit in self._split_large_paragraph(paragraph, paragraph_line_offset)
        ]
        chunks: list[tuple[str, int]] = []
        current_chunk: list[str] = []
        current_chunk_start_offset = 0

        for unit, unit_offset in units:
            candidate_chunk = (
                [*current_chunk, unit]
                if current_chunk
                else [unit]
            )

            if self._joined_length(candidate_chunk) <= self.max_chunk_size:
                if not current_chunk:
                    current_chunk_start_offset = unit_offset
                current_chunk = candidate_chunk
                continue

            chunks.append(("\n\n".join(current_chunk), current_chunk_start_offset))
            current_chunk = [unit]
            current_chunk_start_offset = unit_offset

        if current_chunk:
            chunks.append(("\n\n".join(current_chunk), current_chunk_start_offset))

        return chunks

    def _split_large_paragraph(
        self,
        paragraph: str,
        paragraph_line_offset: int,
    ) -> list[tuple[str, int]]:
        if len(paragraph) <= self.max_chunk_size:
            return [(paragraph, paragraph_line_offset)]

        lines = paragraph.splitlines(keepends=True)
        chunks: list[tuple[str, int]] = []
        current_lines: list[str] = []
        current_line_offset = paragraph_line_offset
        line_offset = paragraph_line_offset

        for line in lines:
            if len(line) > self.max_chunk_size:
                if current_lines:
                    chunks.append(("".join(current_lines), current_line_offset))
                    current_lines = []
                chunks.extend(
                    (part, line_offset + part_line_offset)
                    for part, part_line_offset in self._split_hard_cap(line)
                )
                current_line_offset = line_offset + line.count("\n")
            elif len("".join([*current_lines, line])) <= self.max_chunk_size:
                current_lines.append(line)
            else:
                chunks.append(("".join(current_lines), current_line_offset))
                current_lines = [line]
                current_line_offset = line_offset

            line_offset += line.count("\n")

        if current_lines:
            chunks.append(("".join(current_lines), current_line_offset))

        return chunks

    @staticmethod
    def _joined_length(parts: list[str], separator: str = "\n\n") -> int:
        if not parts:
            return 0
        return sum(len(part) for part in parts) + len(separator) * (len(parts) - 1)

    @staticmethod
    def _split_with_line_offsets(
        content: str,
        line_offset: int,
        separator: str = "\n\n",
    ) -> list[tuple[str, int]]:
        parts: list[tuple[str, int]] = []
        start = 0
        current_line_offset = line_offset
        separator_length = len(separator)

        while start < len(content):
            separator_index = content.find(separator, start)
            if separator_index == -1:
                parts.append((content[start:], current_line_offset))
                break

            part = content[start:separator_index]
            parts.append((part, current_line_offset))
            current_line_offset += part.count("\n") + separator.count("\n")
            start = separator_index + separator_length

        return parts

    def _split_hard_cap(self, text: str) -> list[tuple[str, int]]:
        if not text:
            return []

        pieces: list[tuple[str, int]] = []
        start = 0
        current_line_offset = 0

        while start < len(text):
            end = min(start + self.max_chunk_size, len(text))

            if end < len(text):
                last_space = text.rfind(" ", start, end)
                if last_space > start:
                    end = last_space

            if end == start:
                end = min(start + self.max_chunk_size, len(text))

            piece = text[start:end]
            pieces.append((piece, current_line_offset))
            current_line_offset += piece.count("\n")
            start = end
            while start < len(text) and text[start].isspace():
                if text[start] == "\n":
                    current_line_offset += 1
                start += 1

        return pieces
