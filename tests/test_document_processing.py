"""
Tests for the document processing module, specifically MarkdownFileTraversal class.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
from typing import List

from src.mcps.rag.document_processing import MarkdownFileTraversal, default_skip_patterns


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
            sample_directory_structure / "build" / "output.md",
            sample_directory_structure / "cache" / "cached.md",
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
            "docs/subdir/advanced.md"
        }
        
        found_relative_paths = {
            str(f.relative_to(sample_directory_structure)) for f in found_files
        }
        
        assert found_relative_paths == expected_files
        assert len(found_files) == 3

    def test_find_files_with_nonexistent_directory(self, temp_dir):
        """Test find_files method with non-existent directory."""
        non_existent_path = temp_dir / "does_not_exist"
        traversal = MarkdownFileTraversal(non_existent_path)
        
        with patch('src.mcps.rag.document_processing.logger') as mock_logger:
            found_files = list(traversal.find_files())
            
            assert len(found_files) == 0
            mock_logger.warning.assert_called_once_with(
                f"Search path does not exist: {non_existent_path}"
            )

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
        (temp_dir / "project" / ".git").mkdir()
        (temp_dir / "project" / ".git" / "hooks").mkdir()
        
        # Create markdown files
        (temp_dir / "project" / "README.md").write_text("# Project README")
        (temp_dir / "project" / ".git" / "config.md").write_text("Git config")
        (temp_dir / "project" / ".git" / "hooks" / "pre-commit.md").write_text("Pre-commit hook")
        
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
        
        assert len(files_list) == len(new_files_list) == 3

    def test_skip_patterns_regex_matching(self, temp_dir):
        """Test that skip patterns work with regex matching."""
        # Create test structure
        (temp_dir / "test_cache").mkdir()
        (temp_dir / "cache_test").mkdir()
        (temp_dir / "my_cache_dir").mkdir()
        (temp_dir / "normal_dir").mkdir()
        
        # Create markdown files
        (temp_dir / "test_cache" / "file.md").write_text("# Cache test")
        (temp_dir / "cache_test" / "file.md").write_text("# Cache test") 
        (temp_dir / "my_cache_dir" / "file.md").write_text("# My cache")
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
