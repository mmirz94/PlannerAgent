from typing import Literal

from langgraph.graph import END

from ..models.state import PlanExecuteState


class Edges:
    """Edge decision functions for routing."""

    def should_end(self, state: PlanExecuteState) -> Literal["agent", END]:
        """
        Determine if workflow should end or continue to agent.

        Args:
            state: Current agent state

        Returns:
            "agent" to continue execution, END to finish
        """
        if state.response:
            return END
        else:
            return "agent"
