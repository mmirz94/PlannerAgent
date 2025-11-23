from typing import List

from langchain_core.tools import BaseTool


def create_react_agent_system_prompt(tools: List[BaseTool]) -> str:
    """
    Create React agent system prompt with dynamic tool descriptions.

    Args:
        tools: List of available tools

    Returns:
        System prompt string with tool descriptions
    """
    # Generate tool descriptions dynamically
    tool_descriptions = []
    for tool in tools:
        tool_descriptions.append(f"- {tool.name}: {tool.description}")

    if tool_descriptions:
        available_tools_str = "\n".join(tool_descriptions)
    else:
        available_tools_str = "No tools available at this point!"

    return f"""You are a helpful assistant with access to the following tools:
{available_tools_str}

You will receive the task and the suggested tool that should be used for executing the task.

Execute tasks precisely and provide accurate information.
"""
