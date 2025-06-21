"""
Document processing module containing file discovery, traversal, document processing, and chunking.
"""

import hashlib
import logging
import re
from collections.abc import Generator
from datetime import datetime
from pathlib import Path
from yaml.parser import ParserError

import frontmatter
from overrides import override, overrides

from .interfaces import Chunk, Document, IChunker, IDocumentProcessor, IFileTraversal

logger = logging.getLogger("mcps")
        # Default skip patterns
default_skip_patterns = [
            r'^\..*',
            r'node_modules/',
            r'__pycache__/',
            r'^scripts/',
            r'^templates/',
            r'^prompts/',
        ]


class MarkdownFileTraversal(IFileTraversal):
    """File traversal implementation for markdown files."""
    
    def __init__(self, base_path: Path, skip_patterns: list[str] = default_skip_patterns):
        self.base_path = base_path
        self.skip_patterns = skip_patterns

    def _is_path_allowed(self, path: Path) -> bool:
        """Check if the path is allowed based on skip patterns."""
        relative_path = str(path.relative_to(self.base_path))
        return not any(re.search(pattern, relative_path) for pattern in self.skip_patterns)
    
    @override
    def find_files(self) -> Generator[Path]:
        """Find markdown files to process."""
        
        if not self.base_path.exists():
            logger.warning(f"Search path does not exist: {self.base_path}")
            yield from ()
        
        yield from  (path for path in self.base_path.rglob("*.md") if self._is_path_allowed(path))


class MarkdownProcessor(IDocumentProcessor):
    """Markdown document processor."""
    
    @overrides
    async def process(self, file_path: Path) -> Document:
        """Process a single markdown document file."""
        try:
            with open(file_path, encoding='utf-8') as f:
                post = frontmatter.load(f)
        except ParserError as e:
                # Handle frontmatter parsing errors gracefully
                post = frontmatter.Post(content=file_path.read_text(encoding='utf-8', errors='replace'))
        # Extract metadata
        metadata = {
            "created_at": datetime.fromtimestamp(file_path.stat().st_ctime),
            "modified_at": datetime.fromtimestamp(file_path.stat().st_mtime),
            **{ k:str(v) for k,v in post.metadata.items() if k not in { 'tags' } }
        }
        
        # Extract wikilinks
        outgoing_links = self._extract_wikilinks(post.content)
        
        # Extract tags
        tags = self._extract_tags(post)
        
        # Get file stats
        stat = file_path.stat()
        created_at = datetime.fromtimestamp(stat.st_ctime)
        modified_at = datetime.fromtimestamp(stat.st_mtime)
        
        # Generate document ID
        doc_id = self._generate_document_id(file_path)
        
        return Document(
            id=doc_id,
            content=post.content,
            metadata=metadata,
            outgoing_links=outgoing_links,
            tags=tags,
            source_path=file_path,
            created_at=created_at,
            modified_at=modified_at
        )
    
    
    def _extract_wikilinks(self, content: str) -> list[str]:
        """Extract wikilinks from markdown content."""
        # Pattern for wikilinks: !?[[note name#heading|display text]]
        # Captures only the note name portion (before # or |)
        # Allows brackets inside link names but stops at # or |
        wikilink_pattern = r'!?\[\[([^#|]*?)(?:[#|][^\]]*)?\]\]'
        matches = re.findall(wikilink_pattern, content)
        # Filter out empty matches but preserve trailing spaces for compatibility
        filtered_matches = [match for match in matches if match.strip()]
        return list(set(filtered_matches))  # Remove duplicates
    
    def _extract_tags(self, content: frontmatter.Post) -> list[str]:
        """Extract tags from markdown content."""
        # Pattern for #tag (hashtags)
        tag_pattern = r'#([a-zA-Z][a-zA-Z0-9_-]*)'
        matches = re.findall(tag_pattern, content.content) 
        fm_tags = content.metadata.get('tags', [])
        # convert string or list of tags to a set to avoid duplicates
        if isinstance(fm_tags, str):
            fm_tags = [fm_tags]
        elif not isinstance(fm_tags, list):
            fm_tags = []
        return list(set(matches + fm_tags))  # Remove duplicates
    
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
    
    async def chunk(self, document: Document) -> list[Chunk]:
        """Split a document into fixed-size chunks with overlap."""
        content = document.content
        chunks = []
        
        # Simple character-based chunking
        start = 0
        position = 0
        
        while start < len(content):
            end = min(start + self.chunk_size, len(content))
            
            # Try to break at word boundaries
            if end < len(content):
                # Look for the last space within the chunk
                last_space = content.rfind(' ', start, end)
                if last_space > start:
                    end = last_space
            
            chunk_content = content[start:end].strip()
            
            if chunk_content:  # Only create non-empty chunks
                chunk_id = f"{document.id}_{position}"
                
                chunk = Chunk(
                    id=chunk_id,
                    content=chunk_content,
                    metadata=document.metadata.copy(),
                    outgoing_links=document.outgoing_links.copy(),
                    tags=document.tags.copy(),
                    source_path=document.source_path,
                    created_at=document.created_at,
                    modified_at=document.modified_at,
                    position=position
                )
                
                chunks.append(chunk)
                position += 1
            
            # Move start position with overlap
            start = max(start + self.chunk_size - self.overlap, end)
        
        logger.debug(f"Created {len(chunks)} chunks from document {document.id}")
        return chunks


class SemanticChunker(IChunker):
    """Semantic chunker that splits on markdown sections."""
    
    def __init__(self, max_chunk_size: int = 2000, min_chunk_size: int = 100):
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
    
    async def chunk(self, document: Document) -> list[Chunk]:
        """Split document into semantic chunks based on markdown structure."""
        content = document.content
        chunks = []
        
        # Split by headers (# ## ### etc.)
        sections = self._split_by_headers(content)
        
        position = 0
        for section in sections:
            if len(section.strip()) < self.min_chunk_size:
                continue
            
            # If section is too large, split it further
            if len(section) > self.max_chunk_size:
                sub_chunks = self._split_large_section(section)
                for sub_chunk in sub_chunks:
                    if len(sub_chunk.strip()) >= self.min_chunk_size:
                        chunk = self._create_chunk(document, sub_chunk, position)
                        chunks.append(chunk)
                        position += 1
            else:
                chunk = self._create_chunk(document, section, position)
                chunks.append(chunk)
                position += 1
        
        logger.debug(f"Created {len(chunks)} semantic chunks from document {document.id}")
        return chunks
    
    def _split_by_headers(self, content: str) -> list[str]:
        """Split content by markdown headers."""
        # Split by headers while keeping the header with the content
        header_pattern = r'^(#{1,6}\s+.+)$'
        lines = content.split('\n')
        sections = []
        current_section = []
        
        for line in lines:
            if re.match(header_pattern, line) and current_section:
                # Start new section
                sections.append('\n'.join(current_section))
                current_section = [line]
            else:
                current_section.append(line)
        
        if current_section:
            sections.append('\n'.join(current_section))
        
        return sections
    
    def _split_large_section(self, section: str) -> list[str]:
        """Split large sections into smaller chunks."""
        # Simple paragraph-based splitting for large sections
        paragraphs = section.split('\n\n')
        chunks = []
        current_chunk = []
        current_size = 0
        
        for paragraph in paragraphs:
            paragraph_size = len(paragraph)
            
            if current_size + paragraph_size > self.max_chunk_size and current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = [paragraph]
                current_size = paragraph_size
            else:
                current_chunk.append(paragraph)
                current_size += paragraph_size
        
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        
        return chunks
    
    def _create_chunk(self, document: Document, content: str, position: int) -> Chunk:
        """Create a chunk from document and content."""
        chunk_id = f"{document.id}_{position}"
        
        return Chunk(
            id=chunk_id,
            content=content.strip(),
            metadata=document.metadata.copy(),
            outgoing_links=document.outgoing_links.copy(),
            tags=document.tags.copy(),
            source_path=document.source_path,
            created_at=document.created_at,
            modified_at=document.modified_at,
            position=position
        )