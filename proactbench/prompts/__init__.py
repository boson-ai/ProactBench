"""Prompt templates for the runtime agents (User Agent / judge)."""

from .runtime import (
    USER_AGENT_SYSTEM_TEMPLATE,
    build_user_agent_eval_message,
)

__all__ = [
    "USER_AGENT_SYSTEM_TEMPLATE",
    "build_user_agent_eval_message",
]
