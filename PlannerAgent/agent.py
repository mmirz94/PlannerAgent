import json
import os
from typing import List, Literal, Optional

from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool
from langchain_openai import AzureChatOpenAI
from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.graph import END, START, StateGraph
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient

from .edges import Edges
from .models.actions import create_dynamic_models
from .models.state import InputState, OutputState, PlanExecuteState
from .nodes import Nodes
from .prompts import create_react_agent_system_prompt


class PlanAndExecuteAgent:
    """
    This agent first creates a plan to answer a user query, then executes
    each step iteratively, replanning after each step until the answer
    is complete.

    Each agent instance has its own LLM and components, allowing multiple
    agents with different configurations to coexist without conflicts.
    """

    def __init__(
        self,
        mongo_db_connection_string: Optional[str] = None,
        db_name: Optional[str] = None,
        kb_path: str = None,
        planner_model: Optional[str] = None,
        replanner_model: Optional[str] = None,
        agent_model: Optional[str] = None,
        planner_provider: Literal["openai", "anthropic"] = "openai",
        replanner_provider: Literal["openai", "anthropic"] = "openai",
        agent_provider: Literal["openai", "anthropic"] = "openai",
        tools: Optional[List[BaseTool]] = None,
        verbose: int = 0,
    ):
        """
        Initialize the Plan-and-Execute Agent with multi-model support.

        Args:
            mongo_db_connection_string: MongoDB connection string
            db_name: Database name
            planner_model: Model for planning (e.g., "claude-sonnet-4-5-20250929")
            replanner_model: Model for replanning (e.g., "claude-sonnet-4-5-20250929")
            agent_model: Model for execution (e.g., "gpt-4o-mini")
            planner_provider: Provider for planner (default: "openai")
            replanner_provider: Provider for replanner (default: "openai")
            agent_provider: Provider for agent (default: "openai")
            tools: List of LangChain tools available to the agent
            verbose: Logging level (0=no logging, 1=states only, 2=states+token usage)

        Example:
            >>> # Use powerful Claude for planning, fast GPT for execution
            >>> agent = PlanAndExecuteAgent(
            ...     planner_model="claude-sonnet-4-5-20250929",
            ...     planner_provider="anthropic",
            ...     replanner_model="claude-sonnet-4-5-20250929",
            ...     replanner_provider="anthropic",
            ...     agent_model="gpt-4o-mini",
            ...     agent_provider="openai"
            ... )
        """
        # Store model configurations
        self.planner_provider = planner_provider
        self.replanner_provider = replanner_provider
        self.agent_provider = agent_provider

        self.planner_model = planner_model
        self.replanner_model = replanner_model
        self.agent_model = agent_model

        # Initialize MongoDB connections if provided
        if mongo_db_connection_string and db_name:
            # Sync client for checkpointer
            mongo_client_sync = MongoClient(
                mongo_db_connection_string,
                compressors="zlib",
                connectTimeoutMS=5000,
            )
            self.checkpointer = MongoDBSaver(mongo_client_sync, db_name)

            # Async client for QueryTool
            mongo_client_async = AsyncIOMotorClient(mongo_db_connection_string)
            self.db = mongo_client_async[db_name]
        else:
            self.checkpointer = None
            self.db = None

        # Load knowledge base
        if kb_path:
            self.knowledge_base = self._load_knowledge_base(kb_path)
        else:
            self.knowledge_base = "No knowledge base provided"
        # Initialize LLMs for each stage
        self.planner_llm = self._create_llm(self.planner_provider, self.planner_model)
        self.replanner_llm = self._create_llm(
            self.replanner_provider, self.replanner_model
        )
        self.agent_llm = self._create_llm(self.agent_provider, self.agent_model)

        if tools is not None:
            invalid_tools = [t for t in tools if not isinstance(t, BaseTool)]
            if invalid_tools:
                raise ValueError("Tools must be a valid LangChain Tool")
        else:
            tools = []

        self.tools = tools

        # Create dynamic models based on available tools
        self.dynamic_models = create_dynamic_models(self.tools)

        # Create executer agent
        self.react_agent = self._create_react_agent()

        # Create Nodes instance with dependencies
        # Use agent_llm for both execution and summarization
        self.nodes = Nodes(
            planner=self.planner_llm,
            replanner=self.replanner_llm,
            react_agent=self.react_agent,
            knowledge_base=self.knowledge_base,
            llm=self.agent_llm,  # Use light weight agent LLM for summarization
            planner_provider=planner_provider,
            replanner_provider=replanner_provider,
            tools=self.tools,
            dynamic_models=self.dynamic_models,
            verbose=verbose,
        )

        # Create Edges instance
        self.edges = Edges()

        # Build the graph
        self.graph = self._build_graph()

    def _load_knowledge_base(self, knowledge_base_path: str) -> str:
        """
        Load and format the knowledge base for prompts.

        Args:
            knowledge_base_path: Path to knowledge_base.json

        Returns:
            Formatted knowledge base string

        Raises:
            FileNotFoundError: If knowledge base file doesn't exist
            json.JSONDecodeError: If knowledge base file contains invalid JSON
            PermissionError: If lacking permissions to read the file
        """
 
        try:
            with open(knowledge_base_path, "r") as f:
                kb = json.load(f)
            return json.dumps(kb, indent=2)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Knowledge base file not found at: {knowledge_base_path}"
            )
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"Invalid JSON in knowledge base file: {knowledge_base_path}",
                e.doc,
                e.pos,
            )
        except PermissionError:
            raise PermissionError(
                f"Permission denied when reading knowledge base file: {knowledge_base_path}"
            )

    def _create_llm(self, model_provider: str, model_name: Optional[str] = None):
        """
        Create LLM instance based on provider and optional model name.

        Args:
            model_provider: Either "openai" or "anthropic"
            model_name: Specific model name (optional, uses default if None)

        Returns:
            Configured LLM instance

        Raises:
            ValueError: If model_provider is not supported
        """
        if model_provider == "openai":
            # Use provided model_name or fall back to environment variable or default
            return AzureChatOpenAI(
                azure_deployment=model_name,
                model=model_name,
                api_version="2024-08-01-preview",
                temperature=0,
            )
        elif model_provider == "anthropic":
            # Use provided model_name or fall back to environment variable or default
            return ChatAnthropic(model=model_name, temperature=0, max_tokens=4000)
        else:
            raise ValueError(f"Unsupported model provider: {model_provider}")

    def _create_react_agent(self):
        """
        Create react agent for executing plan steps.

        Uses agent_llm which can be a faster/cheaper model.

        Returns:
            React agent with tools
        """
        # Create dynamic system prompt based on available tools
        system_prompt = create_react_agent_system_prompt(self.tools)
        return create_agent(
            self.agent_llm, self.tools, system_prompt=system_prompt
        )

    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph workflow.

        Returns:
            Compiled StateGraph
        """
        # Initialize the graph with states
        builder = StateGraph(
            PlanExecuteState, input_schema=InputState, output_schema=OutputState
        )

        # Add nodes - using instance methods
        builder.add_node("summarize_conversation", self.nodes.summarize_conversation)
        builder.add_node("planner", self.nodes.plan_step)
        builder.add_node("agent", self.nodes.execute_step)
        builder.add_node("replan", self.nodes.replan_step)

        # Add edges to the graph
        builder.add_edge(START, "summarize_conversation")
        builder.add_edge("summarize_conversation", "planner")
        builder.add_edge("planner", "agent")
        builder.add_edge("agent", "replan")

        # Add conditional edge for replanning
        builder.add_conditional_edges(
            "replan",
            self.edges.should_end,
            ["agent", END],
        )

        # Compile with checkpointer if available
        graph = builder.compile(checkpointer=self.checkpointer)
        return graph

    async def run(
        self, user_input: str, user_id: str, config: Optional[dict] = None
    ) -> OutputState:
        """
        Run the agent with the given user input.

        Args:
            user_input: User's query/prompt
            config: Optional configuration (e.g., recursion_limit)

        Returns:
            OutputState with final response
        """
        if config is None:
            config = {"recursion_limit": 50}

        # Invoke the graph
        output = await self.graph.ainvoke(
            InputState(messages=[HumanMessage(content=user_input)], user_id=user_id),
            config=config,
        )

        # Validate and return as OutputState
        return OutputState.model_validate(output)

    async def stream(
        self,
        user_input: str,
        user_id: str,
        config: Optional[dict] = None,
        stream_mode: Literal["values", "updates"] = "updates",
    ):
        """
        Stream the agent execution with status updates.

        Args:
            user_input: User's query/prompt
            config: Optional configuration (e.g., recursion_limit)
            stream_mode: "values" for full state, "updates" for deltas

        Yields:
            State updates during execution
        """
        if config is None:
            config = {"recursion_limit": 50}

        async for event in self.graph.astream(
            InputState(messages=[HumanMessage(content=user_input)], user_id=user_id),
            config=config,
            stream_mode=stream_mode,
        ):
            yield event

    def save_graph_diagram(self, save_path: str):
        """
        Save the graph diagram as an image.

        Args:
            save_path: Path to save the diagram (must end with .png or .jpg)

        Raises:
            ValueError: If save_path doesn't end with .png or .jpg
        """

        if not (save_path.endswith(".png") or save_path.endswith(".jpg")):
            raise ValueError(
                "Save path must be a valid file path with " ".png or .jpg extension"
            )

        # Create parent directories if they don't exist
        directory = os.path.dirname(save_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        with open(save_path, "wb") as f:
            f.write(self.graph.get_graph(xray=3).draw_mermaid_png())