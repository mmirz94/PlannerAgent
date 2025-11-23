"""Unit tests for PlannerAgent."""
import unittest

from tests.test_agent import (
    TestAgentInitialization,
    TestAgentResponse,
    TestKnowledgeBaseLoading,
    TestLLMCreation,
    TestSaveGraphDiagram,
    TestToolValidation,
)


def suite():
    """
    Create a test suite containing all test cases.

    This allows running tests programmatically or selectively running
    specific test classes.

    Returns:
        unittest.TestSuite: Suite containing all test cases
    """
    loader = unittest.TestLoader()
    test_suite = unittest.TestSuite()

    # Add all test classes using the modern approach
    test_suite.addTests(loader.loadTestsFromTestCase(TestKnowledgeBaseLoading))
    test_suite.addTests(loader.loadTestsFromTestCase(TestLLMCreation))
    test_suite.addTests(loader.loadTestsFromTestCase(TestSaveGraphDiagram))
    test_suite.addTests(loader.loadTestsFromTestCase(TestToolValidation))
    test_suite.addTests(loader.loadTestsFromTestCase(TestAgentInitialization))
    test_suite.addTests(loader.loadTestsFromTestCase(TestAgentResponse))

    return test_suite


# Allow running tests with: python -m tests
if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())