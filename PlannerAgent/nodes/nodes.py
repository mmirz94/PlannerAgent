"""
Nodes for the plan-and-execute workflow.

This module contains the Nodes class that defines the
logic for each node in the LangGraph workflow.
"""

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage

from ..models.state import InputState, PlanExecuteState
from ..prompts import create_planner_messages, create_replanner_messages


class Nodes:

    """
    Nodes for the plan-and-execute workflow.

    This class contains the logic for each node in the LangGraph workflow:
    - summarize_conversation: Summarizes long conversations
    - plan_step: Creates initial execution plan
    - execute_step: Executes current plan step
    - replan_step: Updates plan or returns final answer
    """

    def __init__(
        self,
        planner,
        replanner,
        react_agent,
        knowledge_base,
        llm,
        planner_provider,
        replanner_provider,
        tools,
        dynamic_models,
        verbose=0,
    ):
        """
        Initialize nodes with dependencies.

        Args:
            planner: Planner chain for creating initial plans
            replanner: Replanner chain for updating plans
            react_agent: React agent for executing steps
            knowledge_base: Database schema knowledge base (JSON string)
            llm: Language model for summarization
            planner_provider: one of `openai` or `anthropic`,
            replanner_provider: one of `openai` or `anthropic`,
            tools: List of available tools for the agent
            dynamic_models: Tuple of (ToolType, PlanStep, Plan, Response, Act) dynamically generated models
            verbose: Logging level (0=no logging, 1=states only, 2=states+token usage)
        """
        self.planner = planner
        self.replanner = replanner
        self.react_agent = react_agent
        self.knowledge_base = knowledge_base
        self.llm = llm
        self.planner_provider = planner_provider
        self.replanner_provider = replanner_provider
        self.tools = tools
        self.verbose = verbose

        # Unpack dynamic models
        self.ToolType, self.PlanStep, self.Plan, self.Response, self.Act = dynamic_models

    async def summarize_conversation(self, state: InputState) -> dict:
        """
        Summarize and clean conversation every 4 message pairs (8 total messages).

        Args:
            state: Current agent state with messages

        Returns:
            State updates with summary and cleaned messages
        """

        # Check if we have 4 or more messages (3 exchanges)
        if len(state.messages) >= 6:
            # Get any existing summary
            summary = state.messages_summary

            # Create summarization prompt
            if summary:
                # Extend existing summary
                summary_message = (
                    f"This is summary of the conversation up to this point:\n{summary}\n\n"
                    "Extend the summary by taking into account the following messages."
                    "You are only allowed to used the provided conversation. You are NOT allowed to geenrate content that is not in the conversation"
                )
            else:
                # Create new summary
                summary_message = (
                    "Create a summary of the following conversation."
                    "You are only allowed to used the provided conversation. You are NOT allowed to geenrate content that is not in the conversation"
                )

            # Add prompt to messages (excluding last 2 to keep most recent context)
            messages = [HumanMessage(content=summary_message)] + state.messages[:-2]
            response = await self.llm.ainvoke(messages)

            # Delete all but the 2 most recent messages using RemoveMessage
            delete_messages = [RemoveMessage(id=m.id) for m in state.messages[:-2]]

            return {
                "messages": delete_messages,
                "messages_summary": response.content,
            }
        else:
            return {}

    async def plan_step(self, state: InputState) -> dict:
        """
        Create initial plan based on user input.

        Uses cached prompts if provider is Anthropic.

        Args:
            state: Current agent state

        Returns:
            State updates with plan and status message
        """

        if self.verbose >= 1:
            print("-" * 80)
            print("PLANNING STEP - State:")
            print(state)
            print("-" * 80)

        # Format messages_summary for the prompt
        if state.messages_summary:
            summary_context = (
                f"Summary of previous conversation: {state.messages_summary}"
            )
        else:
            summary_context = "No previous conversation summary."

        # Create messages (cacheable if provider is anthropic).
        # Openai automatically caches the prompts
        messages = create_planner_messages(
            knowledge_base=self.knowledge_base,
            messages_summary=summary_context,
            messages=str([m.content for m in state.messages]),
            tools=self.tools,
            provider=self.planner_provider,
        )

        planner_with_structure = self.planner.with_structured_output(
            self.Plan, include_raw=True
        )
        response = await planner_with_structure.ainvoke(messages)

        plan = response["parsed"]
        # log token usage
        if self.verbose >= 2:
            raw_message = response["raw"]
            token_usage = raw_message.usage_metadata
            print("*" * 80)
            print("PLANNER - Token Usage:")
            print(token_usage)
            print("*" * 80)

        return {
            "plan": plan.steps,
            "user_facing_status_message": plan.user_facing_status_message,
            "response": None,
            "past_steps": [],  # Clear past_steps for new plan
        }

    async def execute_step(self, state: PlanExecuteState) -> dict:
        """
        Execute the first step in the current plan.

        Args:
            state: Current agent state with plan

        Returns:
            State updates with executed step results
        """
        if self.verbose >= 1:
            print("-" * 80)
            print("EXECUTE STEP - State:")
            print(state)
            print("-" * 80)
        plan = state.plan
        plan_str = "\n".join(
            f"{i + 1}. {step.plan_step}" for i, step in enumerate(plan)
        )
        task = plan[0].plan_step
        tool_type = plan[0].tool_type

        # Build task prompt - include user_id for database query tasks
        # Compare with string value since ToolType is dynamic
        user_id_line = (
            f"\nUser ID: {state.user_id}" if tool_type.value == "query" else ""
        )

        task_formatted = f"""
For the following user query and previous interactions:
{[m.content for m in state.messages]}

and for the following plan:
{plan_str}

You are tasked with executing step 1: {task}



The `{tool_type}` tool is required for this task.
{user_id_line}
"""

        agent_response = await self.react_agent.ainvoke(
            {"messages": [("user", task_formatted)]}
        )

        # Manually append to past_steps (no longer using operator.add)
        new_step = (task, agent_response["messages"][-1].content)
        updated_past_steps = (state.past_steps or []) + [new_step]

        return {
            "past_steps": updated_past_steps,
        }

    async def replan_step(self, state: PlanExecuteState) -> dict:
        """
        Replan based on completed steps - either finish or continue.

        Uses cached prompts if provider is Anthropic.

        Args:
            state: Current agent state with past steps

        Returns:
            State updates with new plan or final response
        """

        if self.verbose >= 1:
            print("-" * 80)
            print("REPLAN STEP - State:")
            print(state)
            print("-" * 80)
        # Add knowledge_base and messages_summary to the state data

        # Format messages_summary for the prompt
        if state.messages_summary:
            summary_context = state.messages_summary
        else:
            summary_context = "No previous conversation summary."

        # Create messages (cacheable if provider is anthropic).
        # Openai automatically caches the prompts
        messages = create_replanner_messages(
            knowledge_base=self.knowledge_base,
            messages_summary=summary_context,
            messages=str([m.content for m in state.messages]),
            plan=str(state.plan),
            past_steps=str(state.past_steps),
            tools=self.tools,
            provider=self.replanner_provider,
        )

        replanner_with_structure = self.replanner.with_structured_output(
            self.Act, include_raw=True
        )
        response = await replanner_with_structure.ainvoke(messages)

        output = response["parsed"]
        # log token usage
        if self.verbose >= 2:
            raw_message = response["raw"]
            token_usage = raw_message.usage_metadata
            print("*" * 80)
            print("REPLANNER - Token Usage:")
            print(token_usage)
            print("*" * 80)

        if isinstance(output.action, self.Response):
            # We have a final answer
            return {
                "response": output.action.response,
                "user_facing_status_message": None,
                "messages": [AIMessage(content=output.action.response)],
            }
        else:
            # Continue with new plan
            return {
                "plan": output.action.steps,
                "user_facing_status_message": (
                    output.action.user_facing_status_message
                ),
            }
