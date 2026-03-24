"""Unified token budget management for LLM context windows.

This module provides a centralized system for managing token budgets across
all phases of LLM interaction: prompt construction, RAG retrieval, conversation
history, tool calling, and response generation.
"""

import json
import logging
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage

from ols.constants import TOKEN_BUFFER_WEIGHT
from ols.utils.token_handler import TokenHandler

logger = logging.getLogger(__name__)


@dataclass
class BudgetPartition:
    """Represents how the context window is partitioned among consumers.

    All values are in tokens. The partition defines the maximum allowable
    usage for each category, while actual usage may be lower.
    """

    # Fixed allocations (reserved before any content is processed)
    max_response_tokens: int
    max_tool_tokens: int  # Total budget for all tool-related tokens

    # Variable allocations (computed from remaining space)
    available_for_prompt: int = 0

    # Actual usage tracking (populated during execution)
    prompt_tokens_used: int = 0
    rag_tokens_used: int = 0
    history_tokens_used: int = 0
    tool_tokens_used: int = 0

    def __post_init__(self):
        """Validate the partition configuration."""
        if self.max_response_tokens < 0 or self.max_tool_tokens < 0:
            raise ValueError("Token allocations must be non-negative")

    def get_remaining_tool_budget(self) -> int:
        """Get remaining token budget for tool operations."""
        return max(0, self.max_tool_tokens - self.tool_tokens_used)

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of budget allocation and usage."""
        return {
            "allocations": {
                "max_response": self.max_response_tokens,
                "max_tools": self.max_tool_tokens,
                "available_for_prompt": self.available_for_prompt,
            },
            "usage": {
                "prompt": self.prompt_tokens_used,
                "rag": self.rag_tokens_used,
                "history": self.history_tokens_used,
                "tools": self.tool_tokens_used,
            },
            "remaining": {
                "tools": self.get_remaining_tool_budget(),
            },
        }


class TokenBudget:
    """Centralized token budget manager for LLM context windows.

    This class provides a single source of truth for:
    - Context window partitioning
    - Token counting for all message types
    - Budget tracking across all phases
    - Auditable usage reporting
    """

    def __init__(
        self,
        context_window_size: int,
        max_tokens_for_response: int,
        max_tokens_for_tools: int = 0,
        token_handler: TokenHandler | None = None,
    ):
        """Initialize the token budget manager.

        Args:
            context_window_size: Total size of the LLM context window
            max_tokens_for_response: Tokens reserved for LLM response
            max_tokens_for_tools: Tokens reserved for tool operations (0 if no tools)
            token_handler: Optional TokenHandler instance (creates new if None)
        """
        self.context_window_size = context_window_size
        self.token_handler = token_handler or TokenHandler()

        # Create the budget partition
        self.partition = BudgetPartition(
            max_response_tokens=max_tokens_for_response,
            max_tool_tokens=max_tokens_for_tools,
        )

        logger.debug(
            "TokenBudget initialized: context_window=%d, response=%d, tools=%d",
            context_window_size,
            max_tokens_for_response,
            max_tokens_for_tools,
        )

    def count_message_tokens(self, message: BaseMessage) -> int:
        """Count tokens in a LangChain message, handling all message types.

        Properly accounts for:
        - Simple messages (HumanMessage, SystemMessage): content only
        - AIMessage with tool_calls: content + serialized tool_calls
        - ToolMessage: content only

        Args:
            message: A LangChain BaseMessage

        Returns:
            Token count including buffer
        """
        # Start with message type and content
        parts = [f"{message.type}: {message.content}"]

        # For AIMessage with tool_calls, include the tool call data
        if isinstance(message, AIMessage) and message.tool_calls:
            # Serialize tool_calls to match what's sent to the LLM
            tool_calls_json = json.dumps(message.tool_calls)
            parts.append(tool_calls_json)

        # Combine all parts
        message_text = "\n".join(parts)
        tokens = self.token_handler.text_to_tokens(message_text)
        return TokenHandler._get_token_count(tokens)

    def count_messages_tokens(self, messages: list[BaseMessage]) -> int:
        """Count total tokens across multiple messages.

        Args:
            messages: List of LangChain messages

        Returns:
            Total token count including newlines between messages
        """
        total = 0
        for message in messages:
            total += self.count_message_tokens(message)
            total += 1  # Newline separator
        return total

    def calculate_available_for_prompt(self, base_prompt_tokens: int) -> int:
        """Calculate tokens available for prompt augmentation (RAG + history).

        This should be called once during prompt preparation to establish
        the budget partition.

        Args:
            base_prompt_tokens: Token count of the base prompt template

        Returns:
            Number of tokens available for RAG and history

        Raises:
            ValueError: If base prompt is too large
        """
        self.partition.prompt_tokens_used = base_prompt_tokens

        available = (
            self.context_window_size
            - self.partition.max_response_tokens
            - self.partition.max_tool_tokens
            - base_prompt_tokens
        )

        if available < 0:
            limit = (
                self.context_window_size
                - self.partition.max_response_tokens
                - self.partition.max_tool_tokens
            )
            raise ValueError(
                f"Base prompt length {base_prompt_tokens} exceeds available "
                f"context window limit {limit} tokens"
            )

        self.partition.available_for_prompt = available

        logger.debug(
            "Available tokens for prompt augmentation: %d "
            "(context=%d, response=%d, tools=%d, base_prompt=%d)",
            available,
            self.context_window_size,
            self.partition.max_response_tokens,
            self.partition.max_tool_tokens,
            base_prompt_tokens,
        )

        return available

    def track_rag_usage(self, tokens_used: int) -> None:
        """Track tokens used by RAG context.

        Args:
            tokens_used: Number of tokens consumed by RAG context
        """
        self.partition.rag_tokens_used = tokens_used
        logger.debug("RAG tokens used: %d", tokens_used)

    def track_history_usage(self, tokens_used: int) -> None:
        """Track tokens used by conversation history.

        Args:
            tokens_used: Number of tokens consumed by history
        """
        self.partition.history_tokens_used = tokens_used
        logger.debug("History tokens used: %d", tokens_used)

    def track_tool_usage(self, tokens_used: int) -> None:
        """Track tokens used by tool operations.

        This includes tool definitions, AIMessage with tool_calls, and
        ToolMessage results.

        Args:
            tokens_used: Additional tokens consumed by tool operations
        """
        self.partition.tool_tokens_used += tokens_used
        logger.debug(
            "Tool tokens used: +%d (total: %d)",
            tokens_used,
            self.partition.tool_tokens_used,
        )

    def limit_conversation_history(
        self, history: list[BaseMessage], available_tokens: int
    ) -> tuple[list[BaseMessage], bool]:
        """Limit conversation history to fit within token budget.

        Properly handles all message types including AIMessage with tool_calls
        and ToolMessage.

        Args:
            history: List of conversation messages
            available_tokens: Maximum tokens allowed for history

        Returns:
            Tuple of (truncated_history, was_truncated)
        """
        total_length = 0
        index = 0

        for message in reversed(history):
            message_length = self.count_message_tokens(message)
            total_length += message_length + 1  # +1 for newline

            if total_length > available_tokens:
                logger.debug(
                    "History truncated: exceeds available %d tokens", available_tokens
                )
                truncated = history[len(history) - index :]
                self.track_history_usage(
                    self.count_messages_tokens(truncated) if truncated else 0
                )
                return truncated, True

            index += 1

        self.track_history_usage(total_length)
        return history, False

    def get_budget_summary(self) -> dict[str, Any]:
        """Get a comprehensive summary of budget allocation and usage.

        Returns:
            Dictionary with allocation, usage, and remaining budget information
        """
        summary = self.partition.get_summary()
        summary["context_window_size"] = self.context_window_size
        return summary

    def log_budget_summary(self, level: int = logging.DEBUG) -> None:
        """Log the current budget state.

        Args:
            level: Logging level (default DEBUG)
        """
        summary = self.get_budget_summary()
        logger.log(
            level,
            "Token budget summary:\n%s",
            json.dumps(summary, indent=2),
        )
