"""
State models for the plan-and-execute agent workflow.

This module defines the state structures used by LangGraph:
- PlanExecuteState: Main workflow state
- InputState: Input schema for agent invocation
- OutputState: Output schema returned to user
"""

from typing import Annotated, Any, List, Optional, Tuple

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class PlanExecuteState(BaseModel):
    """
    Main state for the plan-and-execute agent workflow.

    This state is passed between nodes in the LangGraph workflow,
    accumulating information as the agent executes.
    """

    messages: Annotated[list[AnyMessage], add_messages] = Field(
        description="All messages in the conversation"
    )
    messages_summary: Optional[str] = Field(
        default=None, description="Summary of the conversation messages"
    )
    user_id: str = Field(description="Unique ID of the user asking the question")
    plan: Optional[List[Any]] = Field(
        default_factory=list,
        description=(
            "Agent's plan - list of steps to execute and the tool needed for each step. "
            "Uses dynamic PlanStep models created at runtime based on available tools."
        ),
    )
    past_steps: Optional[List[Tuple]] = Field(
        default_factory=list,
        description="Executed steps along with their results (cleared on new plan)",
    )
    user_facing_status_message: Optional[str] = Field(
        default="Thinking...", description="Status message shown to user during processing. used for streaming"
    )
    response: Optional[str] = Field(
        default=None, description="Final answer in markdown format"
    )


class InputState(BaseModel):
    """
    Input state for agent invocation.

    This is the minimal state required to start the agent workflow.
    """

    messages: Annotated[list[AnyMessage], add_messages] = Field(
        description="Initial messages to process"
    )
    messages_summary: Optional[str] = Field(
        default=None, description="Summary of previous conversation (if continuing)"
    )
    user_id: str = Field(description="Unique ID of the user asking the question")


class OutputState(BaseModel):
    """
    Output state returned to the user.

    This is what the agent returns after completing execution.
    """

    response: str = Field(description="Final answer in markdown format")
    user_facing_status_message: Optional[str] = Field(
        default=None, description="Final status message"
    )
