"""
Cache Control Utilities for Anthropic Prompt Caching

This module provides helper functions to create messages with cache control
for Anthropic's prompt caching feature. Static parts of prompts are cached
to reduce token costs and improve response times.
"""

from typing import List, Union

from langchain_core.messages import HumanMessage, SystemMessage


def create_cached_system_message(content: str) -> SystemMessage:
    """
    Create a SystemMessage with cache control enabled.

    Static system instructions should be cached to reduce costs.
    Anthropic caches the last message block with cache_control set.

    Args:
        content: Static system prompt text

    Returns:
        SystemMessage with cache_control enabled

    Example:
        >>> system_msg = create_cached_system_message(
        ...     "You are a helpful assistant."
        ... )
        >>> # Cached by Anthropic for ~5 minutes
    """
    return SystemMessage(
        content=[
            {"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}
        ]
    )


def create_system_message(content: str) -> SystemMessage:
    """
    Create a regular SystemMessage without caching.

    Use this for dynamic content that changes frequently.

    Args:
        content: System prompt text

    Returns:
        SystemMessage without cache control
    """
    return SystemMessage(content=content)


def create_cached_messages(
    static_system_prompt: str,
    dynamic_context: str = None,
    user_message: str = None,
) -> List[Union[SystemMessage, HumanMessage]]:
    """
    Create a message sequence with cached static system prompt.

    Pattern:
    1. Static system instructions (CACHED)
    2. Dynamic context (NOT cached, optional)
    3. User message (NOT cached)

    Args:
        static_system_prompt: Static instructions to cache
        dynamic_context: Optional dynamic context (schema, history, etc.)
        user_message: Optional user query

    Returns:
        List of messages ready for LLM invocation

    Example:
        >>> messages = create_cached_messages(
        ...     static_system_prompt="You are a MongoDB expert.",
        ...     dynamic_context="Schema: {...}",
        ...     user_message="Find all users"
        ... )
        >>> response = llm.invoke(messages)
    """
    messages = []

    # Add static system prompt with caching
    messages.append(create_cached_system_message(static_system_prompt))

    # Add dynamic context as separate system message (not cached)
    if dynamic_context:
        messages.append(create_system_message(dynamic_context))

    # Add user message
    if user_message:
        messages.append(HumanMessage(content=user_message))

    return messages