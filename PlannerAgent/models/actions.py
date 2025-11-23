"""
Dynamic action models for the plan-and-execute agent.

This module creates Pydantic models dynamically based on available tools,
allowing the agent to work with any set of custom tools.
"""

from enum import StrEnum
from typing import List, Union

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


# Create the models dynamically based on the tools that user passes
def create_dynamic_models(tools: List[BaseTool]):
    """
    Create dynamic Pydantic models based on available tools.

    This generates ToolType enum, PlanStep, Plan, and Act models
    with tool types matching the actual available tools.

    Args:
        tools: List of available tools

    Returns:
        Tuple of (ToolType, PlanStep, Plan, Response, Act) classes
    """
    # Extract tool names
    if tools is None:
        tool_names = ["llm"]
    else:
        tool_names = [tool.name for tool in tools]+["llm"]

    # Create dynamic ToolType enum (inherits from str and Enum)
    ToolType = StrEnum("ToolType", {name.upper(): name for name in tool_names})

    # Create PlanStep model with dynamic ToolType
    class PlanStep(BaseModel):
        """
        A single step in the execution plan.

        Attributes:
            plan_step: Description of the task to perform
            tool_type: Tool required to execute this step
        """

        plan_step: str = Field(description="Description of the task to perform")
        tool_type: ToolType = Field(  # type: ignore[valid-type]
            description=f"Tool required to execute this step. Must be one of: {', '.join(tool_names)}"
        )

    # Create Plan model
    class Plan(BaseModel):
        """
        Multi-step plan to achieve the user's objective.

        This is the structured output from the planner, containing a sequence
        of steps to execute and a status message to show the user.

        Attributes:
            steps: List of plan steps in execution order
            user_facing_status_message: Short status message for the user
        """

        steps: List[PlanStep] = Field(
            description=(
                "Different steps of the plan to follow (along with the type of tool "
                "required to execute the step), in sorted order"
            )
        )
        user_facing_status_message: str = Field(
            description=(
                "Very short summary (max 5 words) of the first task. "
                "Shown to user while waiting. Should be general, engaging, "
                "and contain no confidential info."
            )
        )

    # Create Response model (static, doesn't depend on tools)
    class Response(BaseModel):
        """
        Final response to the user.

        This is the structured output when the agent has completed all steps
        and is ready to return the final answer.

        Attributes:
            response: Final answer to the user's query (Markdown supported)
        """

        response: str = Field(description="Final answer to the user's query")

    # Create Act model
    class Act(BaseModel):
        """
        Action to perform during replanning.

        The replanner returns either:
        - Response: If the objective is complete
        - Plan: If more steps are needed

        Attributes:
            action: Either a Response (complete) or Plan (continue)
        """

        action: Union[Response, Plan] = Field(
            description=(
                "Action to perform. Use Response to answer user, "
                "or Plan to continue with more steps."
            )
        )

    return ToolType, PlanStep, Plan, Response, Act


# Static models for type hints and initialization
# These are used by state.py and other modules for type checking


class PlanStepInit(BaseModel):
    """
    A single step in the execution plan (static version for type hints).

    Attributes:
        plan_step: Description of the task to perform
        tool_type: Tool required to execute this step
    """

    plan_step: str = Field(description="Description of the task to perform")
    tool_type: str = Field(description="Name of the tool required to execute this step")