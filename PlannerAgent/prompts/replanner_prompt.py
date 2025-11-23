from typing import List, Literal

from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool

from ..utils.cache_utils import create_cached_messages


def create_replanner_messages(
    knowledge_base: str,
    messages_summary: str,
    messages: str,
    plan: str,
    past_steps: str,
    tools: List[BaseTool],
    provider: Literal["anthropic", "openai"] = "openai",
) -> List[BaseMessage]:
    """
    Create replanner messages with optional caching (Anthropic only).

    Static parts (instructions + knowledge base) are cached for Anthropic.
    Dynamic parts (context, plan, past steps) are not cached.

    Args:
        knowledge_base: Database schema (static, cached if Anthropic)
        messages_summary: Earlier conversation summary (dynamic)
        messages: Current conversation messages (dynamic)
        plan: Original plan (dynamic)
        past_steps: Completed steps (dynamic)
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

## Database Schema and industry definitions (Knowledge Base):
{knowledge_base}

## Available Tools:
When designing remaining steps, use the appropriate tool from the available tools listed below.
{available_tools_str}

## Output format
You can either:
1. Return a Response with the final answer to the user.
    - In this case you are to use the information from previous steps and prepare a nicely represented MarkDown response based on the initial objective.
    - All related information to the objective must be included in the response. For example, if the user asks to see a table of their accounts and the previous step has extracted such a table it should be included in your final response.
    - When responding to the user avoid using terms like `user's request has been processed and here is the answer:`. This answer is directly shown to the user so the response should address the user.
2. Return a Plan with the remaining steps (each step should have plan_step and tool_type). Each step should specify:
    - plan_step: The task description
    - tool_type: The tool to use ({tool_names})"""

    # Dynamic user context
    user_message = f"""## Earlier Conversation Summary:
{messages_summary}

## Your objective and past interactions were these:
{messages}

## Your original plan was this:
{plan}

## You have currently done the following steps:
{past_steps}

Update your plan accordingly. If no more steps are needed and you can return to the user with the final answer, respond with that.
Otherwise, fill out the plan. Only add steps to the plan that still NEED to be done.
Do not return previously done steps as part of the plan."""

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
