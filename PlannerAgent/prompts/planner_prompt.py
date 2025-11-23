
from typing import List, Literal

from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool

from ..utils.cache_utils import create_cached_messages


def create_planner_messages(
    knowledge_base: str,
    messages_summary: str,
    messages: str,
    tools: List[BaseTool],
    provider: Literal["anthropic", "openai"] = "openai",
) -> List[BaseMessage]:
    """
    Create planner messages with optional caching (Anthropic only).

    Static parts (instructions + knowledge base) are cached for Anthropic.
    Dynamic parts (summary, messages) are not cached.

    Args:
        knowledge_base: Database schema (static, cached if Anthropic)
        messages_summary: Earlier conversation summary (dynamic)
        messages: Current conversation messages (dynamic)
        tools: List of available tools (static, cached if Anthropic)
        provider: LLM provider ("anthropic" or "openai")

    Returns:
        List of messages for LLM invocation
    """

    # Generate tool descriptions dynamically
    tool_descriptions = []
    for tool in tools:
        tool_descriptions.append(f"- `{tool.name}`: {tool.description}")

    tool_descriptions.append("- `llm`: for combining data from previous steps (if applicable) and preparing the final answer using a Large Language Model")
    available_tools_str = "\n".join(tool_descriptions)
    tool_names = ", ".join([tool.name for tool in tools]+["llm"])
    # Static system instructions + knowledge base
    static_prompt = f"""For the given objective and chat history, come up with a simple step by step plan and the tool needed to execute each step.
This plan should involve individual tasks, that if executed correctly will yield the correct answer.
Do not add any superfluous steps.
The result of the final step should be the final answer.
Make sure that each step has all the information needed - do not skip steps.

## Database Schema (Knowledge Base):
{knowledge_base}

## Available Tools:
When designing Plan steps, use the appropriate tool from the available tools listed below.
{available_tools_str}

Each step should specify:
- plan_step: The task description
- tool_type: The tool to use ({tool_names})

Also provide a user_facing_status_message (max 5 words) summarizing the first task.

## General Rules:
- If the user is asking for a comparison and references something asked before, and you have enough information in the previous responses or the summary, do not perform the task again.\
    For example, if the user first asks about price of a company and later asks you to compare it with the price of another company, use the data previously extracted for the first company (if available)."""

    # Dynamic user context
    user_message = f"""## Earlier Conversation Summary:
{messages_summary}

## Your objective and a few past interactions are:
{messages}"""

    # Return messages based on provider
    if provider == "anthropic":
        # Use Anthropic caching

        return create_cached_messages(
            static_system_prompt=static_prompt, user_message=user_message
        )

    else:
        # Use regular ChatPromptTemplate for OpenAI and others
        prompt_template = ChatPromptTemplate.from_template(
            """{static_prompt}

{user_message}"""
        )
        return prompt_template.format_messages(
            static_prompt=static_prompt, user_message=user_message
        )
