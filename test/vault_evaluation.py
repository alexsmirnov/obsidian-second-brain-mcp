"""
Comprehensive evaluation test for the Vault class.

This module provides evaluation tests for the Vault class's search functionality,
measuring precision, recall, and F-score based on expected and unwanted words
in search results.

The test assumes a test_content folder containing an Obsidian vault with rich content
about AI/ML, programming, projects, and personal knowledge management.
"""

import asyncio
from dotenv import load_dotenv, find_dotenv
import logging
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import pytest

from mcps.rag.vault import Vault


# Configure logging for detailed test output
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv(find_dotenv())
load_dotenv(find_dotenv(usecwd=True))


class VaultEvaluationTest:
    """
    Evaluation test class for the Vault search functionality.
    
    This class implements comprehensive evaluation metrics including:
    - Precision: percentage of expected words found in search results
    - Recall: percentage of unwanted words NOT found in search results  
    - F-score: harmonic mean of precision and recall
    """
    
    def __init__(self, vault_path: Path):
        """
        Initialize the evaluation test.
        
        Args:
            vault_path: Path to the test content vault directory
        """
        self.vault_path = vault_path
        self.vault = None
        self.test_cases = self._create_test_cases()
    
    def _create_test_cases(self) -> List[Dict[str, Any]]:
        """
        Create the 5 test cases based on available content.
        
        Returns:
            List of test case dictionaries with query, expected_words, and unwanted_words
        """
        return [
            {
                "name": "Python AI Libraries Query",
                "query": "Python artificial intelligence machine learning libraries",
                "expected_words": ["python", "AI", "artificial intelligence", "machine learning", "library", "libraries", "scikit", "tensorflow", "pytorch"],
                "unwanted_words": ["Java", "Salesforce", "camping", "3D printing", "quantum computing"]
            },
            {
                "name": "Salesforce Work Query", 
                "query": "Salesforce work project development",
                "expected_words": ["Salesforce", "work", "project", "development", "CRM", "apex", "lightning", "workflow"],
                "unwanted_words": ["personal", "hobby", "camping", "family", "vacation", "entertainment"]
            },
            {
                "name": "Machine Learning Query",
                "query": "machine learning deep learning neural networks LLM",
                "expected_words": ["machine learning", "deep learning", "neural network", "LLM", "model", "training", "algorithm", "embedding"],
                "unwanted_words": ["camping", "3D printing", "cooking", "travel", "music", "sports"]
            },
            {
                "name": "JavaScript Frameworks Query",
                "query": "JavaScript React Vue Angular frontend framework",
                "expected_words": ["JavaScript", "React", "Vue", "Angular", "frontend", "framework", "component", "DOM"],
                "unwanted_words": ["Python", "quantum", "biology", "chemistry", "physics", "geology"]
            },
            {
                "name": "Project Management Query",
                "query": "project management kanban agile scrum methodology",
                "expected_words": ["project", "management", "kanban", "agile", "scrum", "methodology", "workflow", "planning"],
                "unwanted_words": ["learning", "archive", "personal notes", "diary", "journal", "memories"]
            }
        ]
    
    async def setup_vault(self) -> None:
        """
        Set up and initialize the Vault with test content.
        
        Raises:
            RuntimeError: If vault setup fails
        """
        try:
            logger.info(f"Setting up Vault with path: {self.vault_path}")
            
            # Initialize Vault with optimized settings for testing
            self.vault = Vault(
                vault_path=self.vault_path,
                chunk_size=800,  # Smaller chunks for better precision
                chunk_overlap=100,
                db_table_name="evaluation_test",
                max_content_length=1500,
                include_metadata=True
            )
            
            # Initialize and update index
            await self.vault.initialize()
            logger.info("Vault initialized successfully")
            
            await self.vault.update_index()
            logger.info("Vault index updated successfully")
            
        except Exception as e:
            logger.error(f"Failed to setup vault: {e}")
            raise RuntimeError(f"Vault setup failed: {e}") from e
    
    def calculate_precision(self, search_results: str, expected_words: List[str]) -> float:
        """
        Calculate precision: percentage of expected words found in search results.
        
        Args:
            search_results: The formatted search results string
            expected_words: List of words that should appear in results
            
        Returns:
            Precision score as a float between 0 and 1
        """
        if not expected_words:
            return 1.0
        
        search_results_lower = search_results.lower()
        found_words = 0
        
        for word in expected_words:
            if word.lower() in search_results_lower:
                found_words += 1
        
        precision = found_words / len(expected_words)
        return precision
    
    def calculate_recall(self, search_results: str, unwanted_words: List[str]) -> float:
        """
        Calculate recall: percentage of unwanted words NOT found in search results.
        
        Args:
            search_results: The formatted search results string
            unwanted_words: List of words that should NOT appear in results
            
        Returns:
            Recall score as a float between 0 and 1
        """
        if not unwanted_words:
            return 1.0
        
        search_results_lower = search_results.lower()
        unwanted_found = 0
        
        for word in unwanted_words:
            if word.lower() in search_results_lower:
                unwanted_found += 1
        
        # Recall is the percentage of unwanted words NOT found
        recall = (len(unwanted_words) - unwanted_found) / len(unwanted_words)
        return recall
    
    def calculate_f_score(self, precision: float, recall: float) -> float:
        """
        Calculate F-score: harmonic mean of precision and recall.
        
        Args:
            precision: Precision score
            recall: Recall score
            
        Returns:
            F-score as a float between 0 and 1
        """
        if precision + recall == 0:
            return 0.0
        
        f_score = 2 * (precision * recall) / (precision + recall)
        return f_score
    
    async def run_single_test(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a single test case and calculate evaluation metrics.
        
        Args:
            test_case: Test case dictionary with query and expected/unwanted words
            
        Returns:
            Dictionary with test results and metrics
        """
        try:
            logger.info(f"Running test case: {test_case['name']}")
            logger.info(f"Query: {test_case['query']}")
            
            # Perform search
            search_results = await self.vault.search(test_case['query'])
            
            # Calculate metrics
            precision = self.calculate_precision(search_results, test_case['expected_words'])
            recall = self.calculate_recall(search_results, test_case['unwanted_words'])
            f_score = self.calculate_f_score(precision, recall)
            
            # Log detailed results
            logger.info(f"Search results length: {len(search_results)} characters")
            logger.info(f"Expected words: {test_case['expected_words']}")
            logger.info(f"Unwanted words: {test_case['unwanted_words']}")
            logger.info(f"Precision: {precision:.3f}")
            logger.info(f"Recall: {recall:.3f}")
            logger.info(f"F-score: {f_score:.3f}")
            
            return {
                'name': test_case['name'],
                'query': test_case['query'],
                'search_results': search_results,
                'expected_words': test_case['expected_words'],
                'unwanted_words': test_case['unwanted_words'],
                'precision': precision,
                'recall': recall,
                'f_score': f_score,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Test case '{test_case['name']}' failed: {e}")
            return {
                'name': test_case['name'],
                'query': test_case['query'],
                'error': str(e),
                'precision': 0.0,
                'recall': 0.0,
                'f_score': 0.0,
                'success': False
            }
    
    async def run_all_tests(self) -> List[Dict[str, Any]]:
        """
        Run all test cases and return comprehensive results.
        
        Returns:
            List of test results with evaluation metrics
        """
        logger.info("Starting comprehensive Vault evaluation tests")
        results = []
        
        for test_case in self.test_cases:
            result = await self.run_single_test(test_case)
            results.append(result)
            
            # Add separator between tests for readability
            logger.info("-" * 80)
        
        return results
    
    def print_summary(self, results: List[Dict[str, Any]]) -> None:
        """
        Print a comprehensive summary of all test results.
        
        Args:
            results: List of test results from run_all_tests()
        """
        logger.info("=" * 80)
        logger.info("VAULT EVALUATION SUMMARY")
        logger.info("=" * 80)
        
        successful_tests = [r for r in results if r['success']]
        failed_tests = [r for r in results if not r['success']]
        
        logger.info(f"Total tests: {len(results)}")
        logger.info(f"Successful tests: {len(successful_tests)}")
        logger.info(f"Failed tests: {len(failed_tests)}")
        
        if successful_tests:
            avg_precision = sum(r['precision'] for r in successful_tests) / len(successful_tests)
            avg_recall = sum(r['recall'] for r in successful_tests) / len(successful_tests)
            avg_f_score = sum(r['f_score'] for r in successful_tests) / len(successful_tests)
            
            logger.info(f"\nOverall Metrics:")
            logger.info(f"Average Precision: {avg_precision:.3f}")
            logger.info(f"Average Recall: {avg_recall:.3f}")
            logger.info(f"Average F-score: {avg_f_score:.3f}")
        
        logger.info(f"\nDetailed Results:")
        for result in results:
            status = "✓" if result['success'] else "✗"
            logger.info(f"{status} {result['name']}: F-score = {result['f_score']:.3f}")
            if not result['success']:
                logger.info(f"  Error: {result.get('error', 'Unknown error')}")
        
        if failed_tests:
            logger.info(f"\nFailed Tests:")
            for result in failed_tests:
                logger.info(f"- {result['name']}: {result.get('error', 'Unknown error')}")
    
    async def cleanup(self) -> None:
        """Clean up vault resources."""
        if self.vault:
            try:
                await self.vault.cleanup()
                logger.info("Vault cleanup completed")
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")


# Pytest fixtures and test functions
@pytest.fixture
def test_content_path() -> Path:
    """
    Fixture to provide the path to test content.
    
    This assumes a test_content folder exists at the same level as the test directory.
    If it doesn't exist, the test will be skipped.
    """
    # Look for test_content folder at project root level
    project_root = Path(__file__).parent.parent
    test_content_path = project_root / "test_content"
    
    if not test_content_path.exists():
        pytest.skip(f"Test content directory not found at {test_content_path}")
    
    return test_content_path


@pytest.fixture
async def vault_evaluator(test_content_path) -> VaultEvaluationTest:
    """
    Fixture to create and setup a VaultEvaluationTest instance.
    
    Args:
        test_content_path: Path to the test content directory
        
    Returns:
        Configured VaultEvaluationTest instance
    """
    evaluator = VaultEvaluationTest(test_content_path)
    await evaluator.setup_vault()
    
    yield evaluator
    
    # Cleanup
    await evaluator.cleanup()


@pytest.mark.asyncio
async def test_vault_evaluation_comprehensive(vault_evaluator):
    """
    Comprehensive evaluation test for Vault search functionality.
    
    This test runs all evaluation test cases and provides detailed metrics
    for precision, recall, and F-score.
    """
    # Run all evaluation tests
    results = await vault_evaluator.run_all_tests()
    
    # Print comprehensive summary
    vault_evaluator.print_summary(results)
    
    # Assert that we have results
    assert len(results) == 5, f"Expected 5 test cases, got {len(results)}"
    
    # Assert that at least some tests succeeded
    successful_tests = [r for r in results if r['success']]
    assert len(successful_tests) > 0, "No tests succeeded"
    
    # Assert reasonable performance thresholds
    if successful_tests:
        avg_f_score = sum(r['f_score'] for r in successful_tests) / len(successful_tests)
        assert avg_f_score > 0.1, f"Average F-score too low: {avg_f_score:.3f}"
        
        # Log final assessment
        logger.info(f"\nFINAL ASSESSMENT:")
        logger.info(f"Average F-score: {avg_f_score:.3f}")
        if avg_f_score >= 0.7:
            logger.info("EXCELLENT: Vault search performance is excellent")
        elif avg_f_score >= 0.5:
            logger.info("GOOD: Vault search performance is good")
        elif avg_f_score >= 0.3:
            logger.info("FAIR: Vault search performance is fair")
        else:
            logger.info("POOR: Vault search performance needs improvement")


@pytest.mark.asyncio
async def test_individual_test_cases(vault_evaluator):
    """
    Test individual test cases separately for detailed analysis.
    """
    for i, test_case in enumerate(vault_evaluator.test_cases):
        logger.info(f"\n{'='*60}")
        logger.info(f"INDIVIDUAL TEST {i+1}: {test_case['name']}")
        logger.info(f"{'='*60}")
        
        result = await vault_evaluator.run_single_test(test_case)
        
        # Assert individual test requirements
        assert 'precision' in result
        assert 'recall' in result
        assert 'f_score' in result
        
        # Log individual test assessment
        if result['success']:
            if result['f_score'] >= 0.5:
                logger.info(f"✓ PASS: {test_case['name']} (F-score: {result['f_score']:.3f})")
            else:
                logger.info(f"⚠ WEAK: {test_case['name']} (F-score: {result['f_score']:.3f})")
        else:
            logger.info(f"✗ FAIL: {test_case['name']} - {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    """
    Direct execution for manual testing.
    
    Usage:
        python test/vault_evaluation.py
    """
    async def main():
        # Setup test content path
        project_root = Path(__file__).parent.parent
        test_content_path = project_root / "test_content"
        
        if not test_content_path.exists():
            print(f"Error: Test content directory not found at {test_content_path}")
            print("Please create a test_content folder with Obsidian vault content.")
            return
        
        # Run evaluation
        evaluator = VaultEvaluationTest(test_content_path)
        
        try:
            await evaluator.setup_vault()
            results = await evaluator.run_all_tests()
            evaluator.print_summary(results)
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
        finally:
            await evaluator.cleanup()
    
    # Run the main function
    asyncio.run(main())