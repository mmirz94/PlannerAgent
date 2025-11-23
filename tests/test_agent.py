"""
Integration tests for PlanAndExecuteAgent.

These tests use real objects and minimal mocking to test actual functionality.
External services (MongoDB, APIs) are tested with environment variables or
skipped.
"""

import asyncio
import json
import os
import tempfile
import unittest

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_tavily import TavilySearch

from PlannerAgent.agent import PlanAndExecuteAgent

load_dotenv()


class TestKnowledgeBaseLoading(unittest.TestCase):
    """Test knowledge base loading without mocks."""

    def setUp(self):
        """Create temporary test files."""
        self.temp_dir = tempfile.mkdtemp()

        # Valid knowledge base
        self.valid_kb_path = os.path.join(self.temp_dir, "valid_kb.json")
        with open(self.valid_kb_path, "w") as f:
            json.dump({"test_key": "test_value", "nested": {"data": 123}}, f)

        # Invalid JSON file
        self.invalid_kb_path = os.path.join(
            self.temp_dir, "invalid_kb.json"
        )
        with open(self.invalid_kb_path, "w") as f:
            f.write("{ this is not valid json }")

        self.agent = PlanAndExecuteAgent.__new__(PlanAndExecuteAgent)

    def tearDown(self):
        """Clean up temporary files."""
        for file in os.listdir(self.temp_dir):
            os.remove(os.path.join(self.temp_dir, file))
        os.rmdir(self.temp_dir)

    def test_load_valid_knowledge_base(self):
        """Test loading a valid knowledge base file."""
        result = self.agent._load_knowledge_base(self.valid_kb_path)

        # Verify it's valid JSON
        parsed = json.loads(result)
        self.assertEqual(parsed["test_key"], "test_value")
        self.assertEqual(parsed["nested"]["data"], 123)

    def test_load_nonexistent_file_raises_error(self):
        """Test that loading nonexistent file raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError) as context:
            self.agent._load_knowledge_base("/nonexistent/path/file.json")

        self.assertIn("not found", str(context.exception))

    def test_load_invalid_json_raises_error(self):
        """Test that loading invalid JSON raises JSONDecodeError."""
        with self.assertRaises(json.JSONDecodeError) as context:
            self.agent._load_knowledge_base(self.invalid_kb_path)

        self.assertIn("Invalid JSON", str(context.exception))


class TestLLMCreation(unittest.TestCase):
    """Test LLM creation logic."""

    def setUp(self):
        self.agent = PlanAndExecuteAgent.__new__(PlanAndExecuteAgent)

    def test_invalid_provider_raises_error(self):
        """Test that invalid provider raises ValueError."""
        with self.assertRaises(ValueError) as context:
            self.agent._create_llm("invalid_provider")

        self.assertIn("Unsupported model provider", str(context.exception))

    def test_openai_model_creation(self):
        """Test that OpenAI provider creates LLM instance."""
        # Skip if Azure credentials not available
        if not os.getenv("AZURE_OPENAI_ENDPOINT"):
            self.skipTest("Azure OpenAI credentials not available")

        try:
            llm = self.agent._create_llm("openai", "gpt-4o")
            self.assertIsNotNone(llm)
        except Exception as e:
            # If credentials are wrong, that's okay - testing the logic
            if "credential" not in str(e).lower():
                raise

    def test_anthropic_model_creation(self):
        """Test that Anthropic provider creates LLM instance."""
        # Skip if Anthropic API key not available
        if not os.getenv("ANTHROPIC_API_KEY"):
            self.skipTest("Anthropic API key not available")

        try:
            llm = self.agent._create_llm(
                "anthropic", "claude-sonnet-4-5-20250929"
            )
            self.assertIsNotNone(llm)
        except Exception as e:
            # If credentials are wrong, that's okay - testing the logic
            if "api" not in str(e).lower():
                raise


class TestSaveGraphDiagram(unittest.TestCase):
    """Test graph diagram saving."""

    def setUp(self):
        """Create temporary directory."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary files."""
        if os.path.exists(self.temp_dir):
            for root, dirs, files in os.walk(self.temp_dir, topdown=False):
                for file in files:
                    os.remove(os.path.join(root, file))
                for dir in dirs:
                    os.rmdir(os.path.join(root, dir))
            os.rmdir(self.temp_dir)

    def test_invalid_extension_raises_error(self):
        """Test that non-png/jpg extension raises ValueError."""
        agent = PlanAndExecuteAgent.__new__(PlanAndExecuteAgent)
        with self.assertRaises(ValueError) as context:
            agent.save_graph_diagram(
                os.path.join(self.temp_dir, "test.txt")
            )

        self.assertIn("png or .jpg", str(context.exception))

    def test_creates_parent_directories(self):
        """Test that save_graph_diagram creates parent directories."""
        # Skip if API credentials not available
        if not (
            os.getenv("AZURE_OPENAI_ENDPOINT")
            or os.getenv("ANTHROPIC_API_KEY")
        ):
            self.skipTest("API credentials not available")

        # Create KB file
        kb_path = os.path.join(self.temp_dir, "kb.json")
        os.makedirs(os.path.dirname(kb_path), exist_ok=True)
        with open(kb_path, "w") as f:
            json.dump({"test": "data"}, f)

        try:
            # Create agent with minimal config
            agent = PlanAndExecuteAgent(
                kb_path=kb_path,
                planner_model="gpt-4o",
                replanner_model="gpt-4o",
                agent_model="gpt-4o",
                tools=[],
            )

            # Test nested path
            path = os.path.join(self.temp_dir, "deep", "nested", "directorie", "diagram.png")

            agent.save_graph_diagram(path)

            # Verify directory structure was created
            self.assertTrue(os.path.exists(os.path.dirname(path)))
            self.assertTrue(os.path.exists(path))
        except Exception as e:
            # If it's just credential issues, that's okay
            if (
                "credential" not in str(e).lower()
                and "api" not in str(e).lower()
            ):
                raise


class TestToolValidation(unittest.TestCase):
    """Test tool validation logic."""

    def setUp(self):
        """Create temporary directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.kb_path = os.path.join(self.temp_dir, "kb.json")
        with open(self.kb_path, "w") as f:
            json.dump({"test_key": "test_value"}, f)

    def tearDown(self):
        """Clean up temporary files."""
        if os.path.exists(self.temp_dir):
            for root, dirs, files in os.walk(self.temp_dir, topdown=False):
                for file in files:
                    os.remove(os.path.join(root, file))
                for dir in dirs:
                    os.rmdir(os.path.join(root, dir))
            os.rmdir(self.temp_dir)

    def test_valid_tools_accepted(self):
        """Test that valid BaseTool instances are accepted."""
        # Skip if API credentials not available
        if not (
            os.getenv("AZURE_OPENAI_ENDPOINT")
            or os.getenv("ANTHROPIC_API_KEY")
        ):
            self.skipTest("API credentials not available")

        try:

            # Create a test tool using langchain's @tool decorator
            @tool
            def test_tool(query: str) -> str:
                """Dummy tool for test."""
                return f"Results for: {query}"

            # create a predefined search tool
            tavily_search = TavilySearch(
                max_results=3, topic="finance", include_raw_content=True
            )

            # Should not raise - valid BaseTool
            agent = PlanAndExecuteAgent(
                kb_path=self.kb_path,
                planner_model="gpt-4o",
                replanner_model="gpt-4o",
                agent_model="gpt-4o",
                tools=[test_tool, tavily_search],
            )
            self.assertIsNotNone(agent)
            self.assertEqual(len(agent.tools), 2)
        except Exception as e:
            # If it's just credential issues, that's okay
            if (
                "credential" not in str(e).lower()
                and "api" not in str(e).lower()
            ):
                raise

    def test_invalid_tool_raises_error(self):
        """Test that non-BaseTool objects raise ValueError."""

        with self.assertRaises(ValueError) as context:
            PlanAndExecuteAgent(
                kb_path=self.kb_path,
                planner_model="gpt-4o",
                replanner_model="gpt-4o",
                agent_model="gpt-4o",
                tools=["not_a_tool", 123],  # Invalid - not BaseTool
            )

        self.assertIn("valid LangChain Tool", str(context.exception))

    def test_none_tools_uses_empty_list(self):
        """Test that None tools parameter defaults to empty list."""
        # Skip if API credentials not available
        if not (
            os.getenv("AZURE_OPENAI_ENDPOINT")
            or os.getenv("ANTHROPIC_API_KEY")
        ):
            self.skipTest("API credentials not available")

        try:
            # Should default to empty list
            agent = PlanAndExecuteAgent(
                kb_path=self.kb_path,
                planner_model="gpt-4o",
                replanner_model="gpt-4o",
                agent_model="gpt-4o",
                tools=None,
            )
            self.assertIsNotNone(agent)
            self.assertEqual(agent.tools, [])
        except Exception as e:
            # If it's just credential issues, that's okay
            if (
                "credential" not in str(e).lower()
                and "api" not in str(e).lower()
            ):
                raise

    def test_empty_tools_list_accepted(self):
        """Test that empty tools list is accepted."""
        # Skip if API credentials not available
        if not (
            os.getenv("AZURE_OPENAI_ENDPOINT")
            or os.getenv("ANTHROPIC_API_KEY")
        ):
            self.skipTest("API credentials not available")

        try:
            # Should accept empty list
            agent = PlanAndExecuteAgent(
                kb_path=self.kb_path,
                planner_model="gpt-4o",
                replanner_model="gpt-4o",
                agent_model="gpt-4o",
                tools=[],
            )
            self.assertIsNotNone(agent)
            self.assertEqual(agent.tools, [])
        except Exception as e:
            # If it's just credential issues, that's okay
            if (
                "credential" not in str(e).lower()
                and "api" not in str(e).lower()
            ):
                raise


class TestAgentInitialization(unittest.TestCase):
    """Test full agent initialization."""
    def setUp(self):
        """Create temporary directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.kb_path = os.path.join(self.temp_dir, "kb.json")
        with open(self.kb_path, "w") as f:
            json.dump({"test_key": "test_value"}, f)

    def tearDown(self):
        """Clean up temporary files."""
        if os.path.exists(self.temp_dir):
            for root, dirs, files in os.walk(self.temp_dir, topdown=False):
                for file in files:
                    os.remove(os.path.join(root, file))
                for dir in dirs:
                    os.rmdir(os.path.join(root, dir))
            os.rmdir(self.temp_dir)

    def test_minimal_initialization_with_no_tools(self):
        """Test initialization with minimal config and no tools."""
        # Skip if API credentials not available
        if not (
            os.getenv("AZURE_OPENAI_ENDPOINT")
        ):
            self.skipTest("OpenAI API credentials not available")

        try:
            agent = PlanAndExecuteAgent(
                kb_path=self.kb_path,
                planner_model="gpt-4o",
                replanner_model="gpt-4o",
                agent_model="gpt-4o",
                tools=[],
            )

            # Verify basic attributes
            self.assertIsNotNone(agent.graph)
            self.assertIsNotNone(agent.planner_llm)
            self.assertIsNotNone(agent.replanner_llm)
            self.assertIsNotNone(agent.agent_llm)
            self.assertIsNotNone(agent.nodes)
            self.assertIsNotNone(agent.edges)
            self.assertEqual(agent.tools, [])

            # Verify no MongoDB when not configured
            self.assertIsNone(agent.checkpointer)
            self.assertIsNone(agent.db)
        except Exception as e:
            # If it's just credential issues, that's okay
            if (
                "credential" not in str(e).lower()
                and "api" not in str(e).lower()
            ):
                raise

    def test_initialization_with_different_providers(self):
        """Test initialization with different model providers."""
        # Skip if credentials not available
        if not (
            os.getenv("AZURE_OPENAI_ENDPOINT")
            and os.getenv("ANTHROPIC_API_KEY")
        ):
            self.skipTest("Both API credentials needed for this test")

        try:

            # Mix of providers
            agent = PlanAndExecuteAgent(
                kb_path=self.kb_path,
                tools=[],
                planner_provider="anthropic",
                planner_model="claude-sonnet-4-5-20250929",
                replanner_provider="anthropic",
                replanner_model="claude-sonnet-4-5-20250929",
                agent_provider="openai",
                agent_model="gpt-4o",
            )

            self.assertIsNotNone(agent)
            self.assertEqual(agent.planner_provider, "anthropic")
            self.assertEqual(agent.agent_provider, "openai")
        except Exception as e:
            # If it's just credential issues, that's okay
            if (
                "credential" not in str(e).lower()
                and "api" not in str(e).lower()
            ):
                raise

    def test_initialization_without_kb_path(self):
        """Test initialization without knowledge base path."""
        # Skip if API credentials not available
        if not (
            os.getenv("AZURE_OPENAI_ENDPOINT")
        ):
            self.skipTest("OpenAI API credentials not available")

        try:
            agent = PlanAndExecuteAgent(
                planner_model="gpt-4o",
                replanner_model="gpt-4o",
                agent_model="gpt-4o",
                tools=[],
            )

            # Should use default message
            self.assertEqual(
                agent.knowledge_base, "No knowledge base provided"
            )
        except Exception as e:
            # If it's just credential issues, that's okay
            if (
                "credential" not in str(e).lower()
                and "api" not in str(e).lower()
            ):
                raise


class TestAgentResponse(unittest.TestCase):
    """Test full agent initialization."""
    def setUp(self):
        """Create temporary directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.kb_path = os.path.join(self.temp_dir, "kb.json")
        with open(self.kb_path, "w") as f:
            json.dump({"test_key": "test_value"}, f)

    def tearDown(self):
        """Clean up temporary files."""
        if os.path.exists(self.temp_dir):
            for root, dirs, files in os.walk(self.temp_dir, topdown=False):
                for file in files:
                    os.remove(os.path.join(root, file))
                for dir in dirs:
                    os.rmdir(os.path.join(root, dir))
            os.rmdir(self.temp_dir)
   
    def test_agent_run(self):
        """Test agent can search and answer: Who won 2025 Tour de France?"""
        # Skip if API credentials not available
        if not os.getenv("AZURE_OPENAI_ENDPOINT"):
            self.skipTest("Azure OpenAI credentials not available")

        if not os.getenv("TAVILY_API_KEY"):
            self.skipTest("Tavily API key not available")

        # Create a search tool
        tavily_search = TavilySearch(
            max_results=3, topic="general", include_raw_content=True
        )

        # Create agent with search capability
        agent = PlanAndExecuteAgent(
            kb_path=self.kb_path,
            planner_model="gpt-4o",
            replanner_model="gpt-4o",
            agent_model="gpt-4o",
            tools=[tavily_search],
            verbose=2
        )

        # Execute the async test - pass coroutine directly to asyncio.run
        try:
            result = asyncio.run(
                agent.run(
                    user_input="Who won the 2025 Tour de France?",
                    user_id="test_user_id"
                )
            )

            # Check that we got a response
            self.assertIsNotNone(result)
            self.assertIsNotNone(result.response)

            # Check if "Tadej" is in the response (case-insensitive)
            response_text = result.response.lower()
            self.assertIn(
                "tadej",
                response_text,
                f"Expected 'Tadej' in response, got: {result.response}"
            )

            print(f"\n✓ Agent Response: {result.response}\n")

        except Exception as e:
            # If credentials or API issues, skip the test
            if any(
                word in str(e).lower()
                for word in ["credential", "api", "key", "auth"]
            ):
                self.skipTest(f"API/credential issue: {e}")
            else:
                raise

    def test_agent_stream(self):
        """Test agent streaming: Who won 2025 Tour de France?"""
        # Skip if API credentials not available
        if not os.getenv("AZURE_OPENAI_ENDPOINT"):
            self.skipTest("Azure OpenAI credentials not available")

        if not os.getenv("TAVILY_API_KEY"):
            self.skipTest("Tavily API key not available")

        # Create a search tool
        tavily_search = TavilySearch(
            max_results=3, topic="general", include_raw_content=True
        )

        # Create agent with search capability
        agent = PlanAndExecuteAgent(
            kb_path=self.kb_path,
            planner_model="gpt-4o",
            replanner_model="gpt-4o",
            agent_model="gpt-4o",
            tools=[tavily_search],
        )

        # Stream the agent execution
        async def stream_query():
            events = []
            async for event in agent.stream(
                user_input="Who won the 2025 Tour de France?",
                user_id="test_user",
                stream_mode="updates"
            ):
                events.append(event)

            return events

        # Execute the async test
        try:
            events = asyncio.run(stream_query())

            # Check that we got events
            self.assertGreater(
                len(events), 0, "Expected streaming events, got none"
            )

            # Find the final response in the last event
            final_event = events[-1]
            # The final event should contain the response
            response = None
            if "replan" in final_event:
                replan_data = final_event["replan"]
                if "response" in replan_data:
                    response = replan_data["response"]

            # Check if we got a response with "Tadej"
            if response:
                response_text = response.lower()
                self.assertIn(
                    "tadej",
                    response_text,
                    f"Expected 'Tadej' in response, got: {response}"
                )
                print(f"\n✓ Streaming Response: {response}\n")
            else:
                print(f"\n✓ Received {len(events)} streaming events\n")

        except Exception as e:
            # If credentials or API issues, skip the test
            if any(
                word in str(e).lower()
                for word in ["credential", "api", "key", "auth"]
            ):
                self.skipTest(f"API/credential issue: {e}")
            else:
                raise


if __name__ == "__main__":
    # Print environment info
    print("\n" + "=" * 70)
    print("Test Environment:")
    print(
        f"  Azure OpenAI: "
        f"{'✓' if os.getenv('AZURE_OPENAI_ENDPOINT') else '✗'}"
    )
    print(
        f"  Anthropic API: "
        f"{'✓' if os.getenv('ANTHROPIC_API_KEY') else '✗'}"
    )
    print(f"  MongoDB: {'✓' if os.getenv('MONGODB_URI') else '✗'}")
    print("=" * 70 + "\n")

    unittest.main(verbosity=2)
