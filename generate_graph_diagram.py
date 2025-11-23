"""
Generate graph architecture diagram for README documentation.

This script creates a visual representation of the PlannerAgent workflow
and saves it as a PNG image for use in the README.
"""

import os

from dotenv import load_dotenv

from PlannerAgent.agent import PlanAndExecuteAgent

load_dotenv()

def generate_diagram():
    """Generate and save the agent workflow diagram."""

    # Create a minimal agent instance (no actual LLM calls needed)
    # We just need the graph structure
    agent = PlanAndExecuteAgent(
        kb_path=None,  # Not needed for diagram
        planner_model="gpt-4o",
        planner_provider="openai",
        replanner_model="gpt-4o",
        replanner_provider="openai",
        agent_model="gpt-4o",
        agent_provider="openai",
        verbose=0
    )

    # Create diagrams directory if it doesn't exist
    os.makedirs("diagrams", exist_ok=True)

    # Save the graph diagram
    output_path = "diagrams/agent_architecture.png"
    agent.save_graph_diagram(output_path)

    print(f"Graph diagram saved to: {output_path}")


if __name__ == "__main__":
    generate_diagram()
