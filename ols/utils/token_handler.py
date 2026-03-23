"""Utility to handle tokens."""

import json
import logging
from dataclasses import dataclass, field
from math import ceil

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from llama_index.core.schema import NodeWithScore
from tiktoken import get_encoding

from ols.app.models.models import RagChunk
from ols.constants import (
    DEFAULT_TOKENIZER_MODEL,
    MINIMUM_CONTEXT_TOKEN_LIMIT,
    RAG_SIMILARITY_CUTOFF,
    TOKEN_BUFFER_WEIGHT,
)
from ols.src.prompts.prompt_generator import format_retrieved_chunk

logger = logging.getLogger(__name__)


class PromptTooLongError(Exception):
    """Prompt is too long."""


@dataclass
class ContextWindowBudget:
    """Track context window token budget across all consumers.

    A single object that flows through the full request lifecycle,
    recording how the context window is divided between prompt,
    RAG, history, response, and tools.
    """

    total: int
    response_reserve: int
    tool_reserve: int = 0

    prompt_tokens: int = field(default=0, init=False)
    rag_tokens: int = field(default=0, init=False)
    history_tokens: int = field(default=0, init=False)
    tool_tokens: int = field(default=0, init=False)

    @property
    def augmentation_available(self) -> int:
        """Return tokens available for RAG and history."""
        return (
            self.total
            - self.response_reserve
            - self.tool_reserve
            - self.prompt_tokens
            - self.rag_tokens
            - self.history_tokens
        )

    @property
    def tool_remaining(self) -> int:
        """Return remaining tool token budget."""
        return self.tool_reserve - self.tool_tokens

    def consume_prompt(self, tokens: int) -> None:
        """Record tokens consumed by the prompt skeleton."""
        self.prompt_tokens += tokens

    def consume_rag(self, tokens: int) -> None:
        """Record tokens consumed by RAG context."""
        self.rag_tokens += tokens

    def consume_history(self, tokens: int) -> None:
        """Record tokens consumed by conversation history."""
        self.history_tokens += tokens

    def consume_tools(self, tokens: int) -> None:
        """Record tokens consumed by tool operations."""
        self.tool_tokens += tokens

    def effective_per_tool_limit(self, max_per_tool: int) -> int:
        """Compute the effective per-tool output limit from remaining budget."""
        return min(max_per_tool, self.tool_remaining)

    def check_fits_or_raise(self) -> None:
        """Raise PromptTooLongError if the prompt exceeds available space."""
        if self.augmentation_available < 0:
            limit = self.total - self.response_reserve - self.tool_reserve
            raise PromptTooLongError(
                f"Prompt length {self.prompt_tokens} exceeds LLM "
                f"available context window limit {limit} tokens"
            )


class TokenHandler:
    """This class handles tokens.

    Convert text to tokens.
    Get rough estimation of token count.
    Truncate text based on token limit.
    """

    def __init__(self, encoding_name: str = DEFAULT_TOKENIZER_MODEL) -> None:
        """Initialize the class instance."""
        # Note: We need an approximate tokens count.
        # For different models, exact tokens may vary due to different tokenizer.
        # Also the provider may add model specific tags.
        self._encoder = get_encoding(encoding_name)

    def text_to_tokens(self, text: str) -> list[int]:
        """Convert text to tokens.

        Args:
            text: context text, ex: "This is my doc"

        Returns:
            List of tokens, ex: [1, 2, 3, 4]
        """
        return self._encoder.encode(text)

    def tokens_to_text(self, tokens: list) -> str:
        """Convert tokens to text.

        Args:
            tokens: ex: [1, 2, 3, 4]

        Returns:
            text: ex "This is my doc"
        """
        return self._encoder.decode(tokens)

    @staticmethod
    def _get_token_count(tokens: list[int]) -> int:
        """Get approximate tokens count."""
        # Note: As we get approximate tokens count, we want to have enough
        # buffer so that there is less chance of under-estimation.
        # We increase by certain percentage to nearest integer (ceil).
        return ceil(len(tokens) * TOKEN_BUFFER_WEIGHT)

    def calculate_and_check_available_tokens(
        self,
        prompt: str,
        context_window_size: int,
        max_tokens_for_response: int,
        max_tokens_for_tools: int = 0,
    ) -> int:
        """Get available tokens that can be used for prompt augmentation.

        Args:
            prompt: format prompt template to string before passing as arg
            context_window_size: context window size of LLM
            max_tokens_for_response: max tokens allowed for response (estimation)
            max_tokens_for_tools: tokens reserved for tool outputs (only when MCP
                servers are configured, default 0 means no reservation)

        Returns:
            available_tokens: int, tokens that can be used for augmentation.
        """
        logger.debug(
            "Context window size: %d, Max generated tokens: %d, Reserved for tools: %d",
            context_window_size,
            max_tokens_for_response,
            max_tokens_for_tools,
        )

        prompt_token_count = TokenHandler._get_token_count(self.text_to_tokens(prompt))
        logger.debug("Prompt tokens: %d", prompt_token_count)

        # The context_window_size is the maximum number of tokens that
        # can be used for a "conversation" in the LLM model. This
        # includes prompt AND response. Hence we need to subtract the
        # prompt tokens, max tokens for response, and reserved tool tokens
        # from the context window size.
        available_tokens = (
            context_window_size
            - max_tokens_for_response
            - max_tokens_for_tools
            - prompt_token_count
        )

        if available_tokens < 0:
            limit = context_window_size - max_tokens_for_response - max_tokens_for_tools
            raise PromptTooLongError(
                f"Prompt length {prompt_token_count} exceeds LLM "
                f"available context window limit {limit} tokens"
            )

        return available_tokens

    def truncate_rag_context(
        self, retrieved_nodes: list[NodeWithScore], max_tokens: int = 500
    ) -> tuple[list[RagChunk], int]:
        """Process retrieved node text and truncate if required.

        Args:
            retrieved_nodes: retrieved nodes object from index
            max_tokens: maximum tokens allowed for rag context

        Returns:
            list of `RagChunk` objects, available tokens after context usage
        """
        rag_chunks = []
        logger.info(
            "Processing %d retrieved nodes for RAG context", len(retrieved_nodes)
        )

        for idx, node in enumerate(retrieved_nodes):
            score = float(node.get_score(raise_error=False))
            doc_title = node.metadata.get("title", "unknown")
            doc_url = node.metadata.get("docs_url", "unknown")
            index_id = node.metadata.get("index_id", "")
            index_origin = node.metadata.get("index_origin", "")

            if score < RAG_SIMILARITY_CUTOFF:
                logger.info(
                    "Document #%d rejected: '%s' (index: %s) - "
                    "similarity score %.4f < threshold %.4f",
                    idx + 1,
                    doc_title,
                    index_origin or index_id or "unknown",
                    score,
                    RAG_SIMILARITY_CUTOFF,
                )
                break

            node_text = node.get_text()
            node_text = format_retrieved_chunk(node_text)
            tokens = self.text_to_tokens(node_text)
            tokens_count = TokenHandler._get_token_count(tokens)
            tokens_count += 1  # for new-line char
            logger.debug("RAG content tokens count: %d", tokens_count)

            available_tokens = min(tokens_count, max_tokens)
            logger.debug(
                "Tokens used for this chunk: %d, remaining budget: %d",
                available_tokens,
                max_tokens - available_tokens,
            )

            if available_tokens < MINIMUM_CONTEXT_TOKEN_LIMIT:
                logger.info(
                    "Document #%d rejected: '%s' (index: %s) - "
                    "insufficient tokens (%d < %d minimum)",
                    idx + 1,
                    doc_title,
                    index_origin or index_id or "unknown",
                    available_tokens,
                    MINIMUM_CONTEXT_TOKEN_LIMIT,
                )
                break

            logger.info(
                "Document #%d selected: title='%s', url='%s', index='%s', "
                "score=%.4f, tokens=%d, remaining_context=%d",
                idx + 1,
                doc_title,
                doc_url,
                index_origin or index_id or "unknown",
                score,
                available_tokens,
                max_tokens - available_tokens,
            )

            node_text = self.tokens_to_text(tokens[:available_tokens])
            rag_chunks.append(
                RagChunk(
                    text=node_text,
                    doc_url=doc_url,
                    doc_title=doc_title,
                )
            )

            max_tokens -= available_tokens

        logger.info(
            "Final selection: %d documents chosen for RAG context", len(rag_chunks)
        )
        return rag_chunks, max_tokens

    def count_message_tokens(self, message: BaseMessage) -> int:
        """Count tokens for a LangChain message, handling all message types.

        Properly serializes structured message types so that AIMessage
        with tool_calls and ToolMessage are not undercounted.

        Args:
            message: Any LangChain message.

        Returns:
            Buffered token count for the message.
        """
        content = message.content
        if isinstance(content, list):
            content_text = json.dumps(content)
        else:
            content_text = str(content)

        parts = [f"{message.type}: {content_text}"]

        if isinstance(message, AIMessage) and message.tool_calls:
            parts.append(json.dumps(message.tool_calls))

        if isinstance(message, ToolMessage) and message.tool_call_id:
            parts.insert(0, f"tool_call_id:{message.tool_call_id} ")

        return TokenHandler._get_token_count(self.text_to_tokens("".join(parts)))

    def limit_conversation_history(
        self, history: list[BaseMessage], limit: int = 0
    ) -> tuple[list[BaseMessage], bool]:
        """Limit conversation history to specified number of tokens."""
        total_length = 0
        index = 0

        for message in reversed(history):
            message_length = self.count_message_tokens(message)
            total_length += message_length + 1  # 1 for new-line char

            # if total length of already checked messages is higher than limit
            # then skip all remaining messages (we need to skip from top)
            if total_length > limit:
                logger.debug(
                    "History truncated, it exceeds available %d tokens.", limit
                )
                return history[len(history) - index :], True
            index += 1

        return history, False
