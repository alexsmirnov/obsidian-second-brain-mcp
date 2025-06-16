"""
Document processing module containing file discovery, traversal, document processing, and chunking.
"""

import logging
import re
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from .interfaces import (
    Document, Chunk, IDocumentProcessor, IChunker, IFileTraversal
)

logger = logging.getLogger("mcps")


class MarkdownFileTraversal(IFileTraversal):
    """File traversal implementation for markdown files."""
    
    def __init__(self, base_path: Path):
        self.base_path = base_path
    
    async def find_files(self, start_folder: Optional[str] = None, skip_patterns: Optional[List[str]] = None) -> List[Path]:
        """Find markdown files to process."""
        if start_folder:
            search_path = self.base_path / start_folder
        else:
            search_path = self.base_path
        
        if not search_path.exists():
            logger.warning(f"Search path does not exist: {search_path}")
            return []
        
        # Default skip patterns
        default_skip_patterns = [
            r'\.git/',
            r'node_modules/',
            r'__pycache__/',
            r'\.vscode/',
            r'\.idea/',
            r'build/',
            r'dist/',
            r'cache/',
        ]
        
        skip_patterns = skip_patterns or []
        all_skip_patterns = default_skip_patterns + skip_patterns
        
        markdown_files = []
        
        for file_path in search_path.rglob("*.md"):
            # Check if file should be skipped
            relative_path = str(file_path.relative_to(self.base_path))
            should_skip = any(re.search(pattern, relative_path) for pattern in all_skip_patterns)
            
            if not should_skip:
                markdown_files.append(file_path)
        
        logger.info(f"Found {len(markdown_files)} markdown files")
        return markdown_files


class MarkdownProcessor(IDocumentProcessor):
    """Markdown document processor."""
    
    def supports_file_type(self, file_path: Path) -> bool:
        """Check if this processor supports the given file type."""
        return file_path.suffix.lower() == '.md'
    
    async def process(self, file_path: Path) -> Document:
        """Process a single markdown document file."""
        try:
            content = file_path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            logger.warning(f"Failed to read file with UTF-8 encoding: {file_path}")
            content = file_path.read_text(encoding='utf-8', errors='ignore')
        
        # Extract metadata
        metadata = self._extract_metadata(file_path, content)
        
        # Extract wikilinks
        outgoing_links = self._extract_wikilinks(content)
        
        # Extract tags
        tags = self._extract_tags(content)
        
        # Get file stats
        stat = file_path.stat()
        created_at = datetime.fromtimestamp(stat.st_ctime)
        modified_at = datetime.fromtimestamp(stat.st_mtime)
        
        # Generate document ID
        doc_id = self._generate_document_id(file_path)
        
        return Document(
            id=doc_id,
            content=content,
            metadata=metadata,
            outgoing_links=outgoing_links,
            tags=tags,
            source_path=file_path,
            created_at=created_at,
            modified_at=modified_at
        )
    
    def _extract_metadata(self, file_path: Path, content: str) -> Dict[str, Any]:
        """Extract metadata from markdown file."""
        metadata = {
            'filename': file_path.name,
            'file_size': len(content),
            'file_extension': file_path.suffix,
        }
        
        # Extract YAML frontmatter if present
        if content.startswith('---'):
            try:
                end_marker = content.find('---', 3)
                if end_marker != -1:
                    frontmatter = content[3:end_marker].strip()
                    # Simple key-value extraction (could use yaml library for full parsing)
                    for line in frontmatter.split('\n'):
                        if ':' in line:
                            key, value = line.split(':', 1)
                            metadata[key.strip()] = value.strip()
            except Exception as e:
                logger.warning(f"Failed to parse frontmatter in {file_path}: {e}")
        
        return metadata
    
    def _extract_wikilinks(self, content: str) -> List[str]:
        """Extract wikilinks from markdown content."""
        # Pattern for [[link]] or [[link|display text]]
        wikilink_pattern = r'\[\[([^\]|]+)(?:\|[^\]]+)?\]\]'
        matches = re.findall(wikilink_pattern, content)
        return list(set(matches))  # Remove duplicates
    
    def _extract_tags(self, content: str) -> List[str]:
        """Extract tags from markdown content."""
        # Pattern for #tag (hashtags)
        tag_pattern = r'#([a-zA-Z0-9_-]+)'
        matches = re.findall(tag_pattern, content)
        return list(set(matches))  # Remove duplicates
    
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
    
    async def chunk(self, document: Document) -> List[Chunk]:
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
    
    async def chunk(self, document: Document) -> List[Chunk]:
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
    
    def _split_by_headers(self, content: str) -> List[str]:
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
    
    def _split_large_section(self, section: str) -> List[str]:
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