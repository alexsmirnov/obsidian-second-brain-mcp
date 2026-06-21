"""
Comprehensive evaluation test for the Vault class.

This module provides evaluation tests for the Vault class's search functionality,
measuring precision, recall, and F-score based on expected and unwanted words
in search results.

The test assumes a test_content folder containing an Obsidian vault with rich content
about AI/ML, programming, projects, and personal knowledge management.
"""

import asyncio
import logging
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

import httpx
import pytest
from dotenv import find_dotenv, load_dotenv

from mcps.config import ServerConfig, create_config
from mcps.rag.vault import Vault, create_vault

logger = logging.getLogger(__name__)


# Configure logging for detailed test output
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
load_dotenv(find_dotenv())
load_dotenv(find_dotenv(usecwd=True))


def _tc(
    query: str, expected_words: list[str], unwanted_words: list[str]
) -> dict[str, Any]:
    return {
        "query": query,
        "expected_words": expected_words,
        "unwanted_words": unwanted_words,
    }


class VaultEvaluationTest:
    """
    Evaluation test class for the Vault search functionality.

    This class implements comprehensive evaluation metrics including:
    - Precision: percentage of expected words found in search results
    - Recall: percentage of unwanted words NOT found in search results
    - F-score: harmonic mean of precision and recall
    """
    vault: Vault | None
    config: ServerConfig

    def __init__(self, vault_path: Path, config: ServerConfig):
        """
        Initialize the evaluation test.

        Args:
            vault_path: Path to the test content vault directory
            config: Server configuration produced by the config factory
        """
        self.vault_path = vault_path
        self.config = config
        self.vault = None
        self.test_cases = self._create_test_cases()

    def _create_test_cases(self) -> list[dict[str, Any]]:
        """
        Read test cases from a colon-separated CSV file.

        The CSV file should have 3 columns:
        - Query: The search query
        - Expected: Comma-separated list of expected words
        - Unwanted: Comma-separated list of unwanted words

        Returns:
            Test case dictionaries with query, expected_words,
            and unwanted_words.
        """
        test_cases_file = self.vault_path / "evaluation_tests.csv"

        if not test_cases_file.exists():
            raise FileNotFoundError(f"Test cases file not found at {test_cases_file}")

        test_cases = []
        try:
            with open(test_cases_file, encoding="utf-8") as f:
                # Skip header line
                next(f)

                for line in f:
                    # Split by colon and strip whitespace
                    parts = [part.strip() for part in line.split(":")]

                    if len(parts) != 3:
                        logger.warning(f"Skipping invalid line: {line}")
                        continue

                    query, expected, unwanted = parts

                    # Split expected and unwanted words by comma and strip whitespace
                    expected_words = [
                        word.strip() for word in expected.split(",") if word.strip()
                    ]
                    unwanted_words = [
                        word.strip() for word in unwanted.split(",") if word.strip()
                    ]

                    test_cases.append({
                        "query": query,
                        "expected_words": expected_words,
                        "unwanted_words": unwanted_words,
                    })

        except Exception as e:
            raise RuntimeError(f"Failed to read test cases: {e}") from e

        if not test_cases:
            raise ValueError("No valid test cases found in file")

        return test_cases

    def _chunks_to_text(self, chunks: list) -> str:
        """Concatenate chunk content fields into a single searchable string."""
        parts = []
        for c in chunks:
            if c.title:
                parts.append(c.title)
            if c.description:
                parts.append(c.description)
            parts.append(c.content)
        return "\n".join(parts)

    def calculate_precision(
        self, search_results: str, expected_words: list[str]
    ) -> float:
        """
        Calculate precision: percentage of expected words found in search results.

        Args:
            search_results: The combined search results text
            expected_words: List of words that should appear in results

        Returns:
            Precision score as a float between 0 and 1
        """
        if not expected_words:
            return 1.0

        search_results_lower = search_results.lower()
        found_words = sum(
            1 for word in expected_words if word.lower() in search_results_lower
        )

        return found_words / len(expected_words)

    def calculate_recall(self, search_results: str, unwanted_words: list[str]) -> float:
        """
        Calculate recall: percentage of unwanted words NOT found in search results.

        Args:
            search_results: The combined search results text
            unwanted_words: List of words that should NOT appear in results

        Returns:
            Recall score as a float between 0 and 1
        """
        if not unwanted_words:
            return 1.0

        search_results_lower = search_results.lower()
        unwanted_found = sum(
            1 for word in unwanted_words if word.lower() in search_results_lower
        )

        # Recall is the percentage of unwanted words NOT found
        return (len(unwanted_words) - unwanted_found) / len(unwanted_words)

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

        return 2 * (precision * recall) / (precision + recall)

    async def run_single_test(self, test_case: dict[str, Any]) -> dict[str, Any]:
        """
        Run a single test case and calculate evaluation metrics.

        Args:
            test_case: Test case dictionary with query and expected/unwanted words

        Returns:
            Dictionary with test results and metrics
        """
        if self.vault is None:
            raise RuntimeError("Vault has not been initialized")

        try:
            logger.info(f"Query: {test_case['query']}")

            # Perform search
            chunks = await self.vault.search(test_case["query"])
            search_results = self._chunks_to_text(chunks)
            logger.info(f"Search results: {search_results[:150]}...")
            # Calculate metrics
            precision = self.calculate_precision(
                search_results, test_case["expected_words"]
            )
            recall = self.calculate_recall(search_results, test_case["unwanted_words"])
            f_score = self.calculate_f_score(precision, recall)

            # Log detailed results
            logger.info(f"Search results length: {len(search_results)} characters")
            logger.info(f"Expected words: {test_case['expected_words']}")
            logger.info(f"Unwanted words: {test_case['unwanted_words']}")
            logger.info(f"Precision: {precision:.3f}")
            logger.info(f"Recall: {recall:.3f}")
            logger.info(f"F-score: {f_score:.3f}")

            return {
                "query": test_case["query"],
                "search_results": search_results,
                "expected_words": test_case["expected_words"],
                "unwanted_words": test_case["unwanted_words"],
                "precision": precision,
                "recall": recall,
                "f_score": f_score,
                "success": True,
            }

        except Exception as e:
            logger.error(f"Test case '{test_case['query']}' failed: {e}")
            return {
                "query": test_case["query"],
                "error": str(e),
                "precision": 0.0,
                "recall": 0.0,
                "f_score": 0.0,
                "success": False,
            }

    async def run_all_tests(self) -> list[dict[str, Any]]:
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

    def print_summary(self, results: list[dict[str, Any]]) -> None:
        """
        Print a comprehensive summary of all test results.

        Args:
            results: List of test results from run_all_tests()
        """
        logger.info("=" * 80)
        logger.info("VAULT EVALUATION SUMMARY")
        logger.info("=" * 80)

        successful_tests = [r for r in results if r["success"]]
        failed_tests = [r for r in results if not r["success"]]

        logger.info(f"Total tests: {len(results)}")
        logger.info(f"Successful tests: {len(successful_tests)}")
        logger.info(f"Failed tests: {len(failed_tests)}")

        if successful_tests:
            avg_precision = sum(r["precision"] for r in successful_tests) / len(
                successful_tests
            )
            avg_recall = sum(r["recall"] for r in successful_tests) / len(
                successful_tests
            )
            avg_f_score = sum(r["f_score"] for r in successful_tests) / len(
                successful_tests
            )

            logger.info("\nOverall Metrics:")
            logger.info(f"Average Precision: {avg_precision:.3f}")
            logger.info(f"Average Recall: {avg_recall:.3f}")
            logger.info(f"Average F-score: {avg_f_score:.3f}")

        logger.info("\nDetailed Results:")
        for result in results:
            status = "✓" if result["success"] else "✗"
            logger.info(
                f"{status} {result['query']}: F-score = {result['f_score']:.3f}"
            )
            if not result["success"]:
                logger.info(f"  Error: {result.get('error', 'Unknown error')}")

        if failed_tests:
            logger.info("\nFailed Tests:")
            for result in failed_tests:
                logger.info(
                    f"- {result['query']}: {result.get('error', 'Unknown error')}"
                )


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
async def vault_evaluator(
    test_content_path: Path,
) -> AsyncGenerator[VaultEvaluationTest]:
    """
    Fixture to create and set up a VaultEvaluationTest instance.

    Uses the production config and vault factories with real embeddings.
    Skips the test when the LiteLLM router is not configured.

    Args:
        test_content_path: Path to the test content directory

    Yields:
        Configured VaultEvaluationTest instance with an indexed vault
    """
    config = create_config(vault_dir=test_content_path)
    config.table_name = "evaluation_test"

    if not config.litellm_router or not config.litellm_router_key:
        pytest.skip("LITELLM router is not configured")

    evaluator = VaultEvaluationTest(test_content_path, config)

    async with httpx.AsyncClient() as http_client:
        async with create_vault(config, http_client) as vault:
            logger.info("Vault created; updating index for evaluation")
            await vault.update_index()
            evaluator.vault = vault
            yield evaluator


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
    successful_tests = [r for r in results if r["success"]]
    assert len(successful_tests) > 0, "No tests succeeded"

    # Assert reasonable performance thresholds
    if successful_tests:
        avg_f_score = sum(r["f_score"] for r in successful_tests) / len(
            successful_tests
        )
        assert avg_f_score > 0.1, f"Average F-score too low: {avg_f_score:.3f}"

        # Log final assessment
        logger.info("FINAL ASSESSMENT:")
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
        logger.info(f"INDIVIDUAL TEST {i+1}: {test_case['query']}")
        logger.info(f"{'='*60}")

        result = await vault_evaluator.run_single_test(test_case)

        # Assert individual test requirements
        assert "precision" in result
        assert "recall" in result
        assert "f_score" in result

        # Log individual test assessment
        if result["success"]:
            if result["f_score"] >= 0.5:
                logger.info(
                    f"✓ PASS: {test_case['query']} (F-score: {result['f_score']:.3f})"
                )
            else:
                logger.info(
                    f"⚠ WEAK: {test_case['query']} (F-score: {result['f_score']:.3f})"
                )
        else:
            logger.info(
                f"✗ FAIL: {test_case['query']} - {result.get('error', 'Unknown error')}"
            )


if __name__ == "__main__":
    """
    Direct execution for manual testing.

    Usage:
        python tests/vault_evaluation.py
    """

    async def main() -> None:
        # Setup test content path
        project_root = Path(__file__).parent.parent
        test_content_path = project_root / "test_content"

        if not test_content_path.exists():
            print(f"Error: Test content directory not found at {test_content_path}")
            print("Please create a test_content folder with Obsidian vault content.")
            return

        config = create_config(vault_dir=test_content_path)
        config.table_name = "evaluation_test"

        if not config.litellm_router or not config.litellm_router_key:
            print(
                "Error: LITELLM router is not configured; "
                "real embeddings are required for evaluation."
            )
            return

        evaluator = VaultEvaluationTest(test_content_path, config)

        try:
            async with httpx.AsyncClient() as http_client:
                async with create_vault(config, http_client) as vault:
                    evaluator.vault = vault
                    await vault.update_index()
                    results = await evaluator.run_all_tests()
                    evaluator.print_summary(results)
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")

    # Run the main function
    asyncio.run(main())
