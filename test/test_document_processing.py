"""
Tests for the document processing module, including MarkdownFileTraversal and MarkdownProcessor classes.
"""

import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Generator
from unittest.mock import patch

import frontmatter
import pytest

from mcps.rag.document_processing import (
    MarkdownFileTraversal,
    MarkdownProcessor,
    default_skip_patterns,
)
from mcps.rag.interfaces import Document


class TestMarkdownFileTraversal:
    """Test cases for MarkdownFileTraversal class."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def sample_directory_structure(self, temp_dir):
        """Create a sample directory structure with markdown files."""
        # Create directories
        (temp_dir / "docs").mkdir()
        (temp_dir / "docs" / "subdir").mkdir()
        (temp_dir / ".git").mkdir()
        (temp_dir / "node_modules").mkdir()
        (temp_dir / "__pycache__").mkdir()
        (temp_dir / ".vscode").mkdir()
        (temp_dir / "build").mkdir()
        (temp_dir / "cache").mkdir()
        (temp_dir / ".obsidian").mkdir()

        # Create markdown files
        (temp_dir / "README.md").write_text("# Main README")
        (temp_dir / "docs" / "guide.md").write_text("# User Guide")
        (temp_dir / "docs" / "subdir" / "advanced.md").write_text("# Advanced Topics")
        
        # Create files in skip directories
        (temp_dir / ".git" / "config.md").write_text("Git config")
        (temp_dir / "node_modules" / "package.md").write_text("Package docs")
        (temp_dir / "__pycache__" / "cache.md").write_text("Cache file")
        (temp_dir / ".vscode" / "settings.md").write_text("VS Code settings")
        (temp_dir / "build" / "output.md").write_text("Build output")
        (temp_dir / "cache" / "cached.md").write_text("Cached file")
        (temp_dir / ".obsidian" / "obsidian.md").write_text("Cached file")

        # Create non-markdown files
        (temp_dir / "script.py").write_text("# Python script")
        (temp_dir / "config.txt").write_text("Configuration file")

        return temp_dir

    def test_init_with_default_skip_patterns(self, temp_dir):
        """Test initialization with default skip patterns."""
        traversal = MarkdownFileTraversal(temp_dir)
        
        assert traversal.base_path == temp_dir
        assert traversal.skip_patterns == default_skip_patterns

    def test_init_with_custom_skip_patterns(self, temp_dir):
        """Test initialization with custom skip patterns."""
        custom_patterns = [r'\.custom/', r'temp/']
        traversal = MarkdownFileTraversal(temp_dir, custom_patterns)
        
        assert traversal.base_path == temp_dir
        assert traversal.skip_patterns == custom_patterns

    def test_is_path_allowed_with_allowed_paths(self, sample_directory_structure):
        """Test _is_path_allowed method with paths that should be allowed."""
        traversal = MarkdownFileTraversal(sample_directory_structure)
        
        # These paths should be allowed
        allowed_paths = [
            sample_directory_structure / "README.md",
            sample_directory_structure / "docs" / "guide.md",
            sample_directory_structure / "docs" / "subdir" / "advanced.md",
        ]
        
        for path in allowed_paths:
            assert traversal._is_path_allowed(path), f"Path should be allowed: {path}"

    def test_is_path_allowed_with_skipped_paths(self, sample_directory_structure):
        """Test _is_path_allowed method with paths that should be skipped."""
        traversal = MarkdownFileTraversal(sample_directory_structure)
        
        # These paths should be skipped
        skipped_paths = [
            sample_directory_structure / ".git" / "config.md",
            sample_directory_structure / "node_modules" / "package.md",
            sample_directory_structure / "__pycache__" / "cache.md",
            sample_directory_structure / ".vscode" / "settings.md",
            sample_directory_structure / ".obsidian" / "output.md",
        ]
        
        for path in skipped_paths:
            assert not traversal._is_path_allowed(path), f"Path should be skipped: {path}"

    def test_is_path_allowed_with_custom_patterns(self, temp_dir):
        """Test _is_path_allowed method with custom skip patterns."""
        # Create test structure
        (temp_dir / "allowed").mkdir()
        (temp_dir / "custom_skip").mkdir()
        
        custom_patterns = [r'custom_skip/']
        traversal = MarkdownFileTraversal(temp_dir, custom_patterns)
        
        allowed_path = temp_dir / "allowed" / "file.md"
        skipped_path = temp_dir / "custom_skip" / "file.md"
        
        # Create the files
        allowed_path.parent.mkdir(exist_ok=True)
        skipped_path.parent.mkdir(exist_ok=True)
        allowed_path.touch()
        skipped_path.touch()
        
        assert traversal._is_path_allowed(allowed_path)
        assert not traversal._is_path_allowed(skipped_path)

    def test_find_files_with_existing_directory(self, sample_directory_structure):
        """Test find_files method with existing directory containing markdown files."""
        traversal = MarkdownFileTraversal(sample_directory_structure)
        
        found_files = list(traversal.find_files())
        
        # Should find only the allowed markdown files
        expected_files = {
            "README.md",
            "docs/guide.md", 
            "docs/subdir/advanced.md",
            'build/output.md',
            'cache/cached.md'
        }
        
        found_relative_paths = {
            str(f.relative_to(sample_directory_structure)) for f in found_files
        }
        
        assert found_relative_paths == expected_files
        assert len(found_files) == 5

    def test_find_files_with_nonexistent_directory(self, temp_dir):
        """Test find_files method with non-existent directory."""
        non_existent_path = temp_dir / "does_not_exist"
        traversal = MarkdownFileTraversal(non_existent_path)
        
        found_files = list(traversal.find_files())
        assert len(found_files) == 0

    def test_find_files_empty_directory(self, temp_dir):
        """Test find_files method with empty directory."""
        traversal = MarkdownFileTraversal(temp_dir)
        
        found_files = list(traversal.find_files())
        
        assert len(found_files) == 0

    def test_find_files_directory_with_no_markdown_files(self, temp_dir):
        """Test find_files method with directory containing no markdown files."""
        # Create some non-markdown files
        (temp_dir / "script.py").write_text("# Python script")
        (temp_dir / "config.txt").write_text("Configuration file")
        (temp_dir / "data.json").write_text('{"key": "value"}')
        
        traversal = MarkdownFileTraversal(temp_dir)
        
        found_files = list(traversal.find_files())
        
        assert len(found_files) == 0

    def test_find_files_with_mixed_extensions(self, temp_dir):
        """Test find_files method with mixed file extensions."""
        # Create files with different extensions
        (temp_dir / "document.md").write_text("# Markdown Document")
        (temp_dir / "document.txt").write_text("Text Document")
        (temp_dir / "document.rst").write_text("RestructuredText Document")
        (temp_dir / "README.MD").write_text("# README in uppercase")  # Different case
        
        traversal = MarkdownFileTraversal(temp_dir)
        
        found_files = list(traversal.find_files())
        found_names = {f.name for f in found_files}
        
        # Should only find .md files (case-sensitive)
        expected_names = {"document.md"}
        assert found_names == expected_names

    def test_find_files_respects_skip_patterns_order(self, temp_dir):
        """Test that skip patterns are applied correctly regardless of order."""
        # Create nested structure
        (temp_dir / "project").mkdir()
        (temp_dir  / ".git").mkdir()
        (temp_dir  / ".git" / "hooks").mkdir()
        
        # Create markdown files
        (temp_dir / "project" / "README.md").write_text("# Project README")
        (temp_dir  / ".git" / "config.md").write_text("Git config")
        (temp_dir  / ".git" / "hooks" / "pre-commit.md").write_text("Pre-commit hook")
        
        traversal = MarkdownFileTraversal(temp_dir)
        
        found_files = list(traversal.find_files())
        found_relative_paths = {
            str(f.relative_to(temp_dir)) for f in found_files
        }
        
        # Should only find the project README, not the git files
        expected_paths = {"project/README.md"}
        assert found_relative_paths == expected_paths

    def test_find_files_generator_behavior(self, sample_directory_structure):
        """Test that find_files returns a generator and can be iterated multiple times."""
        traversal = MarkdownFileTraversal(sample_directory_structure)
        
        # Get the generator
        file_generator = traversal.find_files()
        
        # Convert to list
        files_list = list(file_generator)
        
        # Generator should be exhausted now
        remaining_files = list(file_generator)
        assert len(remaining_files) == 0
        
        # But we can call find_files again to get a new generator
        new_generator = traversal.find_files()
        new_files_list = list(new_generator)
        
        assert len(files_list) == len(new_files_list) == 5

    def test_skip_patterns_regex_matching(self, temp_dir):
        """Test that skip patterns work with regex matching."""
        # Create test structure
        (temp_dir / ".test_cache").mkdir()
        (temp_dir / ".cache_test").mkdir()
        (temp_dir / "templates").mkdir()
        (temp_dir / "normal_dir").mkdir()
        
        # Create markdown files
        (temp_dir / ".test_cache" / "file.md").write_text("# Cache test")
        (temp_dir / ".cache_test" / "file.md").write_text("# Cache test") 
        (temp_dir / "templates" / "file.md").write_text("# My cache")
        (temp_dir / "normal_dir" / "file.md").write_text("# Normal file")
        
        traversal = MarkdownFileTraversal(temp_dir)
        
        found_files = list(traversal.find_files())
        found_dirs = {f.parent.name for f in found_files}
        
        # Only normal_dir should be found since cache/ pattern should match directories with 'cache/'
        # The default pattern is r'cache/' which matches 'cache/' at any position
        expected_dirs = {"normal_dir"}
        assert found_dirs == expected_dirs

    @pytest.mark.parametrize("skip_pattern,test_path,should_be_allowed", [
        (r'\.git/', "project/.git/config.md", False),
        (r'\.git/', "project/git_info.md", True),
        (r'node_modules/', "project/node_modules/package.md", False),
        (r'node_modules/', "project/my_modules/file.md", True),
        (r'__pycache__/', "src/__pycache__/cache.md", False),
        (r'__pycache__/', "src/my_cache/file.md", True),
    ])
    def test_skip_patterns_parametrized(self, temp_dir, skip_pattern, test_path, should_be_allowed):
        """Parametrized test for various skip patterns."""
        # Create the test file
        full_path = temp_dir / test_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text("# Test file")
        
        traversal = MarkdownFileTraversal(temp_dir, [skip_pattern])
        
        result = traversal._is_path_allowed(full_path)
        assert result == should_be_allowed, f"Pattern '{skip_pattern}' with path '{test_path}' should be {'allowed' if should_be_allowed else 'skipped'}"


class TestMarkdownProcessor:
    """Test cases for MarkdownProcessor class."""

    @pytest.fixture
    def temp_file(self) -> Generator[Path, None, None]:
        """Create a temporary markdown file for testing."""
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False)
        yield Path(temp_file.name)
        Path(temp_file.name).unlink(missing_ok=True)

    @pytest.fixture
    def processor(self, temp_file):
        """Create a MarkdownProcessor instance."""
        return MarkdownProcessor(temp_file.parent)

    @pytest.fixture
    def sample_markdown_content(self):
        """Sample markdown content with frontmatter."""
        return """---
title: Test Document
author: Test Author
tags:
  - python
  - testing
date: 2023-01-01
---

# Test Document

This is a test document with [[wikilink]] and [[another link|display text]].

It also has some #hashtags and #python-related content.

## Section 2

More content with [[duplicate link]] and [[wikilink]] again.

#testing #automation
"""

    @pytest.fixture
    def simple_markdown_content(self):
        """Simple markdown content without frontmatter."""
        return """# Simple Document

This is a simple document with [[simple link]].

#simple #test
"""

    @pytest.fixture
    def empty_markdown_content(self):
        """Empty markdown content."""
        return ""

    @pytest.fixture
    def markdown_with_complex_wikilinks(self):
        """Markdown with complex wikilink patterns."""
        return """# Complex Links

Regular wikilink: [[Simple Link]]
Wikilink with display text: [[Actual Link|Display Text]]
Wikilink with spaces: [[Link With Spaces]]
Nested brackets: [[Link]] and [[Another [Special] Link]]
Multiple on same line: [[Link1]] and [[Link2]] here.
"""

    @pytest.fixture
    def markdown_with_various_tags(self):
        """Markdown with various tag patterns."""
        return """---
tags:
  - frontmatter-tag
  - multi-word-tag
---

# Document with Tags

Content with #inline-tag and #CamelCaseTag.
Also #tag_with_underscores and #tag-with-hyphens.
Numbers in tags: #tag123 and #123tag.

Invalid tags: # (space after hash), #-invalid (starts with hyphen).
"""

    async def test_process_basic_markdown_file(self, processor, temp_file, sample_markdown_content):
        """Test processing a basic markdown file with frontmatter."""
        # Write content to temp file
        temp_file.write_text(sample_markdown_content, encoding='utf-8')
        
        # Process the file
        document = await processor.process(temp_file)
        
        # Verify document structure
        assert isinstance(document, Document)
        assert document.id is not None
        assert len(document.id) == 32  # MD5 hash length
        assert document.source_path == temp_file.name
        
        # Verify content
        expected_content = sample_markdown_content.split('---\n')[2]  # Content after frontmatter
        assert document.content.strip() == expected_content.strip()
        
        # Verify metadata
        assert document.metadata.title == 'Test Document'
        assert isinstance(document.created_at, datetime)
        assert isinstance(document.modified_at, datetime)
        
        # Verify wikilinks
        expected_links = {'wikilink', 'another link', 'duplicate link'}
        assert set(document.outgoing_links) == expected_links
        
        # Verify tags (frontmatter + hashtags)
        expected_tags = {'python', 'testing', 'hashtags', 'python-related', 'automation'}
        assert set(document.tags) == expected_tags
        
        # Verify timestamps
        assert isinstance(document.created_at, datetime)
        assert isinstance(document.modified_at, datetime)

    async def test_process_simple_markdown_without_frontmatter(self, processor, temp_file, simple_markdown_content):
        """Test processing markdown file without frontmatter."""
        temp_file.write_text(simple_markdown_content, encoding='utf-8')
        
        document = await processor.process(temp_file)
        
        # Verify basic structure
        assert isinstance(document, Document)
        assert document.content.strip() == simple_markdown_content.strip()
        
        # Verify metadata (should only have timestamps)
        assert isinstance(document.created_at, datetime)
        assert isinstance(document.modified_at, datetime)
        assert document.metadata.title is None
        
        # Verify wikilinks
        assert document.outgoing_links == ['simple link']
        
        # Verify tags
        expected_tags = {'simple', 'test'}
        assert set(document.tags) == expected_tags

    async def test_process_empty_markdown_file(self, processor, temp_file, empty_markdown_content):
        """Test processing an empty markdown file."""
        temp_file.write_text(empty_markdown_content, encoding='utf-8')
        
        document = await processor.process(temp_file)
        
        assert isinstance(document, Document)
        assert document.content == ""
        assert document.outgoing_links == []
        assert document.tags == []
        assert isinstance(document.created_at, datetime)
        assert isinstance(document.modified_at, datetime)

    async def test_extract_wikilinks_various_patterns(self, processor, markdown_with_complex_wikilinks):
        """Test wikilink extraction with various patterns."""
        links = processor._extract_wikilinks(markdown_with_complex_wikilinks)
        
        expected_links = {
            'Simple Link',
            'Actual Link',
            'Link With Spaces',
            'Another [Special] Link',
            'Link',
            'Link1',
            'Link2'
        }
        assert set(links) == expected_links

    async def test_extract_wikilinks_duplicates_removed(self, processor):
        """Test that duplicate wikilinks are removed."""
        content = "[[Link1]] and [[Link2]] and [[Link1]] again."
        links = processor._extract_wikilinks(content)
        
        assert set(links) == {'Link1', 'Link2'}
        assert len(links) == 2

    async def test_extract_wikilinks_empty_content(self, processor):
        """Test wikilink extraction from empty content."""
        links = processor._extract_wikilinks("")
        assert links == []

    async def test_extract_wikilinks_no_links(self, processor):
        """Test wikilink extraction from content without links."""
        content = "This is regular markdown content without any wikilinks."
        links = processor._extract_wikilinks(content)
        assert links == []

    async def test_extract_tags_from_frontmatter_and_content(self, processor, markdown_with_various_tags):
        """Test tag extraction from both frontmatter and content."""
        # Parse the content using frontmatter
        post = frontmatter.loads(markdown_with_various_tags)
        tags = processor._extract_tags(post)
        
        expected_tags = {
            'frontmatter-tag',
            'multi-word-tag',
            'inline-tag',
            'CamelCaseTag',
            'tag_with_underscores',
            'tag-with-hyphens',
            'tag123'
        }
        assert set(tags) == expected_tags

    async def test_extract_tags_frontmatter_string_format(self, processor):
        """Test tag extraction when frontmatter tags is a string."""
        content = """---
tags: single-tag
---

# Content
"""
        post = frontmatter.loads(content)
        tags = processor._extract_tags(post)
        
        assert 'single-tag' in tags

    async def test_extract_tags_frontmatter_invalid_format(self, processor):
        """Test tag extraction when frontmatter tags is invalid format."""
        content = """---
tags: 123
---

# Content with #hashtag
"""
        post = frontmatter.loads(content)
        tags = processor._extract_tags(post)
        
        # Should only get hashtag from content, not the invalid frontmatter tag
        assert tags == ['hashtag']

    async def test_extract_tags_no_tags(self, processor):
        """Test tag extraction from content without tags."""
        content = "This is content without any tags."
        post = frontmatter.loads(content)
        tags = processor._extract_tags(post)
        
        assert tags == []

    async def test_extract_tags_duplicates_removed(self, processor):
        """Test that duplicate tags are removed."""
        content = """---
tags:
  - duplicate
---

# Content

#duplicate #unique #duplicate
"""
        post = frontmatter.loads(content)
        tags = processor._extract_tags(post)
        
        assert set(tags) == {'duplicate', 'unique'}

    async def test_generate_document_id_consistency(self, processor, temp_file):
        """Test that document ID generation is consistent for the same file."""
        temp_file.write_text("# Test", encoding='utf-8')
        
        id1 = processor._generate_document_id(temp_file)
        id2 = processor._generate_document_id(temp_file)
        
        assert id1 == id2
        assert len(id1) == 32  # MD5 hash length
        assert isinstance(id1, str)

    async def test_generate_document_id_different_files(self, processor):
        """Test that different files generate different document IDs."""
        temp_file1 = tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False)
        temp_file2 = tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False)
        
        try:
            path1 = Path(temp_file1.name)
            path2 = Path(temp_file2.name)
            
            id1 = processor._generate_document_id(path1)
            id2 = processor._generate_document_id(path2)
            
            assert id1 != id2
        finally:
            Path(temp_file1.name).unlink(missing_ok=True)
            Path(temp_file2.name).unlink(missing_ok=True)

    async def test_process_file_with_unicode_content(self, processor, temp_file):
        """Test processing file with unicode characters."""
        unicode_content = """---
title: Тест
author: 测试
---

# Unicode Test 🚀

Content with émojis and spëcial characters.
[[链接]] and #тег
"""
        temp_file.write_text(unicode_content, encoding='utf-8')
        
        document = await processor.process(temp_file)
        
        assert document.metadata.title == 'Тест'
        assert '链接' in document.outgoing_links

    async def test_process_file_io_error(self, processor):
        """Test handling of file I/O errors."""
        non_existent_file = Path("/non/existent/file.md")
        
        with pytest.raises(FileNotFoundError):
            await processor.process(non_existent_file)

    async def test_process_file_with_malformed_frontmatter(self, processor, temp_file):
        """Test processing file with malformed frontmatter."""
        malformed_content = """---
title: Test
invalid yaml: [unclosed bracket
---

# Content

Regular content here.
"""
        temp_file.write_text(malformed_content, encoding='utf-8')
        
        # Should still process the file, frontmatter library handles malformed YAML gracefully
        document = await processor.process(temp_file)
        assert isinstance(document, Document)

    @pytest.mark.parametrize("wikilink_content,expected_links", [
        ("[[Simple]]", ["Simple"]),
        ("[[Link|Display]]", ["Link"]),
        ("[[Multi Word Link]]", ["Multi Word Link"]),
        ("[[Link1]] and [[Link2]]", ["Link1", "Link2"]),
        ("No links here", []),
        ("[[]]", []),  # Empty wikilink
        ("[[Link with | pipe]]", ["Link with "]),  # Pipe in middle
    ])
    async def test_extract_wikilinks_parametrized(self, processor, wikilink_content, expected_links):
        """Parametrized test for wikilink extraction."""
        links = processor._extract_wikilinks(wikilink_content)
        assert set(links) == set(expected_links)

    @pytest.mark.parametrize("tag_content,expected_tags", [
        ("#simple", ["simple"]),
        ("#tag1 #tag2", ["tag1", "tag2"]),
        ("No tags here", []),
        ("#valid_tag #valid-tag", ["valid_tag", "valid-tag"]),
        ("# invalid", []),  # Space after hash
        ("#CamelCase", ["CamelCase"]),
    ])
    async def test_extract_tags_parametrized(self, processor, tag_content, expected_tags):
        """Parametrized test for hashtag extraction."""
        post = frontmatter.loads(tag_content)
        tags = processor._extract_tags(post)
        assert set(tags) == set(expected_tags)

    async def test_process_large_file(self, processor, temp_file):
        """Test processing a large markdown file."""
        # Create large content
        large_content = """---
title: Large Document
---

# Large Document

""" + "\n".join([f"## Section {i}\n\nContent for section {i} with [[link{i}]] and #tag{i}." for i in range(100)])
        
        temp_file.write_text(large_content, encoding='utf-8')
        
        document = await processor.process(temp_file)
        
        assert isinstance(document, Document)
        assert document.metadata.title == 'Large Document'
        assert len(document.outgoing_links) == 100  # link0 to link99
        assert len(document.tags) == 100  # tag0 to tag99

    async def test_process_preserves_file_timestamps(self, processor, temp_file):
        """Test that processing preserves original file timestamps."""
        temp_file.write_text("# Test", encoding='utf-8')
        
        # Get original timestamps
        stat = temp_file.stat()
        original_ctime = datetime.fromtimestamp(stat.st_ctime)
        original_mtime = datetime.fromtimestamp(stat.st_mtime)
        
        document = await processor.process(temp_file)
        
        # Timestamps should match (within a small tolerance for precision)
        assert abs((document.created_at - original_ctime).total_seconds()) < 1
        assert abs((document.modified_at - original_mtime).total_seconds()) < 1

    async def test_process_excludes_tags_from_metadata(self, processor, temp_file):
        """Test that 'tags' key is excluded from metadata when present in frontmatter."""
        content_with_tags = """---
title: Test
tags:
  - tag1
  - tag2
author: Test Author
description: Test Document
---

# Content
"""
        temp_file.write_text(content_with_tags, encoding='utf-8')
        
        document = await processor.process(temp_file)
        
        # Tags should not be in metadata (they're handled separately)
        assert 'tags' not in document.metadata
        assert document.metadata.title == 'Test'
        assert document.metadata.description == 'Test Document'
        
        # But tags should be in the tags field
        assert 'tag1' in document.tags
        assert 'tag2' in document.tags
