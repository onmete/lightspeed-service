"""Unit tests for the TokenBudget class."""

import json
from math import ceil
from unittest import TestCase, mock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from ols.constants import TOKEN_BUFFER_WEIGHT
from ols.utils.token_budget import BudgetPartition, TokenBudget
from ols.utils.token_handler import TokenHandler


class TestBudgetPartition(TestCase):
    """Test cases for BudgetPartition dataclass."""

    def test_budget_partition_initialization(self):
        """Test BudgetPartition initialization with valid values."""
        partition = BudgetPartition(
            max_response_tokens=1000,
            max_tool_tokens=500,
        )
        assert partition.max_response_tokens == 1000
        assert partition.max_tool_tokens == 500
        assert partition.available_for_prompt == 0
        assert partition.prompt_tokens_used == 0
        assert partition.rag_tokens_used == 0
        assert partition.history_tokens_used == 0
        assert partition.tool_tokens_used == 0

    def test_budget_partition_negative_values_rejected(self):
        """Test that negative token allocations are rejected."""
        with pytest.raises(ValueError, match="Token allocations must be non-negative"):
            BudgetPartition(max_response_tokens=-1, max_tool_tokens=500)

        with pytest.raises(ValueError, match="Token allocations must be non-negative"):
            BudgetPartition(max_response_tokens=1000, max_tool_tokens=-1)

    def test_get_remaining_tool_budget(self):
        """Test calculation of remaining tool budget."""
        partition = BudgetPartition(max_response_tokens=1000, max_tool_tokens=500)

        # Initially, full budget is available
        assert partition.get_remaining_tool_budget() == 500

        # After using some tokens
        partition.tool_tokens_used = 200
        assert partition.get_remaining_tool_budget() == 300

        # After using all tokens
        partition.tool_tokens_used = 500
        assert partition.get_remaining_tool_budget() == 0

        # After exceeding budget (should return 0, not negative)
        partition.tool_tokens_used = 600
        assert partition.get_remaining_tool_budget() == 0

    def test_get_summary(self):
        """Test budget partition summary generation."""
        partition = BudgetPartition(max_response_tokens=1000, max_tool_tokens=500)
        partition.available_for_prompt = 2000
        partition.prompt_tokens_used = 100
        partition.rag_tokens_used = 300
        partition.history_tokens_used = 200
        partition.tool_tokens_used = 150

        summary = partition.get_summary()

        assert summary["allocations"]["max_response"] == 1000
        assert summary["allocations"]["max_tools"] == 500
        assert summary["allocations"]["available_for_prompt"] == 2000
        assert summary["usage"]["prompt"] == 100
        assert summary["usage"]["rag"] == 300
        assert summary["usage"]["history"] == 200
        assert summary["usage"]["tools"] == 150
        assert summary["remaining"]["tools"] == 350


class TestTokenBudget(TestCase):
    """Test cases for TokenBudget class."""

    def setUp(self):
        """Set up test fixtures."""
        self.context_window = 4000
        self.max_response = 1000
        self.max_tools = 500
        self.token_budget = TokenBudget(
            context_window_size=self.context_window,
            max_tokens_for_response=self.max_response,
            max_tokens_for_tools=self.max_tools,
        )

    def test_initialization(self):
        """Test TokenBudget initialization."""
        assert self.token_budget.context_window_size == 4000
        assert self.token_budget.partition.max_response_tokens == 1000
        assert self.token_budget.partition.max_tool_tokens == 500
        assert isinstance(self.token_budget.token_handler, TokenHandler)

    def test_initialization_with_custom_token_handler(self):
        """Test TokenBudget initialization with custom TokenHandler."""
        custom_handler = TokenHandler()
        budget = TokenBudget(
            context_window_size=4000,
            max_tokens_for_response=1000,
            max_tokens_for_tools=500,
            token_handler=custom_handler,
        )
        assert budget.token_handler is custom_handler

    def test_count_message_tokens_human_message(self):
        """Test token counting for HumanMessage."""
        message = HumanMessage("What is Kubernetes?")
        token_count = self.token_budget.count_message_tokens(message)

        # Should count "human: What is Kubernetes?" with buffer
        assert token_count > 0
        # Verify it matches expected calculation
        handler = self.token_budget.token_handler
        text = f"{message.type}: {message.content}"
        tokens = handler.text_to_tokens(text)
        expected = TokenHandler._get_token_count(tokens)
        assert token_count == expected

    def test_count_message_tokens_ai_message_without_tool_calls(self):
        """Test token counting for AIMessage without tool_calls."""
        message = AIMessage("Kubernetes is a container orchestration platform.")
        token_count = self.token_budget.count_message_tokens(message)

        assert token_count > 0
        # Should only count type + content, no tool calls
        handler = self.token_budget.token_handler
        text = f"{message.type}: {message.content}"
        tokens = handler.text_to_tokens(text)
        expected = TokenHandler._get_token_count(tokens)
        assert token_count == expected

    def test_count_message_tokens_ai_message_with_tool_calls(self):
        """Test token counting for AIMessage with tool_calls."""
        tool_calls = [
            {
                "name": "search_docs",
                "args": {"query": "kubernetes pod"},
                "id": "call_123",
            }
        ]
        message = AIMessage(content="", tool_calls=tool_calls)
        token_count = self.token_budget.count_message_tokens(message)

        # Should count type, content, AND serialized tool_calls
        assert token_count > 0

        # Verify it includes tool_calls in the count
        handler = self.token_budget.token_handler
        parts = [f"{message.type}: {message.content}", json.dumps(tool_calls)]
        text = "\n".join(parts)
        tokens = handler.text_to_tokens(text)
        expected = TokenHandler._get_token_count(tokens)
        assert token_count == expected

    def test_count_message_tokens_tool_message(self):
        """Test token counting for ToolMessage."""
        message = ToolMessage(
            content="Search results: pod documentation found",
            tool_call_id="call_123",
        )
        token_count = self.token_budget.count_message_tokens(message)

        assert token_count > 0
        # Should count type + content
        handler = self.token_budget.token_handler
        text = f"{message.type}: {message.content}"
        tokens = handler.text_to_tokens(text)
        expected = TokenHandler._get_token_count(tokens)
        assert token_count == expected

    def test_count_messages_tokens(self):
        """Test counting tokens across multiple messages."""
        messages = [
            HumanMessage("first message"),
            AIMessage("first response"),
            HumanMessage("second message"),
            AIMessage("second response"),
        ]

        total = self.token_budget.count_messages_tokens(messages)

        # Should be sum of individual messages + newlines
        expected = sum(self.token_budget.count_message_tokens(m) for m in messages)
        expected += len(messages)  # One newline per message
        assert total == expected

    def test_count_messages_tokens_empty_list(self):
        """Test counting tokens for empty message list."""
        total = self.token_budget.count_messages_tokens([])
        assert total == 0

    def test_calculate_available_for_prompt_success(self):
        """Test successful calculation of available prompt tokens."""
        base_prompt_tokens = 100

        available = self.token_budget.calculate_available_for_prompt(base_prompt_tokens)

        # Should be: context - response - tools - base_prompt
        expected = 4000 - 1000 - 500 - 100
        assert available == expected
        assert self.token_budget.partition.prompt_tokens_used == base_prompt_tokens
        assert self.token_budget.partition.available_for_prompt == expected

    def test_calculate_available_for_prompt_overflow(self):
        """Test that oversized prompts raise ValueError."""
        # A prompt that's too large
        base_prompt_tokens = 10000

        with pytest.raises(ValueError, match="exceeds available context window limit"):
            self.token_budget.calculate_available_for_prompt(base_prompt_tokens)

    def test_calculate_available_for_prompt_exact_limit(self):
        """Test calculation when prompt exactly fills available space."""
        # Exactly fill the available space
        base_prompt_tokens = 4000 - 1000 - 500  # = 2500

        available = self.token_budget.calculate_available_for_prompt(base_prompt_tokens)

        assert available == 0
        assert self.token_budget.partition.prompt_tokens_used == base_prompt_tokens

    def test_track_rag_usage(self):
        """Test tracking RAG token usage."""
        self.token_budget.track_rag_usage(200)
        assert self.token_budget.partition.rag_tokens_used == 200

        # Multiple calls should replace, not accumulate
        self.token_budget.track_rag_usage(300)
        assert self.token_budget.partition.rag_tokens_used == 300

    def test_track_history_usage(self):
        """Test tracking history token usage."""
        self.token_budget.track_history_usage(150)
        assert self.token_budget.partition.history_tokens_used == 150

        # Multiple calls should replace, not accumulate
        self.token_budget.track_history_usage(250)
        assert self.token_budget.partition.history_tokens_used == 250

    def test_track_tool_usage(self):
        """Test tracking tool token usage."""
        self.token_budget.track_tool_usage(100)
        assert self.token_budget.partition.tool_tokens_used == 100

        # Multiple calls should accumulate
        self.token_budget.track_tool_usage(50)
        assert self.token_budget.partition.tool_tokens_used == 150

    @mock.patch("ols.utils.token_budget.TOKEN_BUFFER_WEIGHT", 1.05)
    def test_limit_conversation_history_no_truncation(self):
        """Test history limiting when no truncation is needed."""
        history = [
            HumanMessage("first message"),
            AIMessage("first response"),
        ]

        truncated, was_truncated = self.token_budget.limit_conversation_history(
            history, available_tokens=1000
        )

        assert truncated == history
        assert was_truncated is False
        assert self.token_budget.partition.history_tokens_used > 0

    @mock.patch("ols.utils.token_budget.TOKEN_BUFFER_WEIGHT", 1.05)
    def test_limit_conversation_history_with_truncation(self):
        """Test history limiting with truncation."""
        history = [
            HumanMessage("first message from human"),
            AIMessage("first answer from AI"),
            HumanMessage("second message from human"),
            AIMessage("second answer from AI"),
            HumanMessage("third message from human"),
            AIMessage("third answer from AI"),
        ]

        # Limit to fit only the last 2 messages
        truncated, was_truncated = self.token_budget.limit_conversation_history(
            history, available_tokens=20
        )

        assert len(truncated) < len(history)
        assert was_truncated is True
        # Should keep the most recent messages
        assert truncated == history[-len(truncated) :]

    @mock.patch("ols.utils.token_budget.TOKEN_BUFFER_WEIGHT", 1.05)
    def test_limit_conversation_history_with_tool_messages(self):
        """Test history limiting correctly counts AIMessage with tool_calls and ToolMessage."""
        tool_calls = [{"name": "search", "args": {"q": "test"}, "id": "1"}]
        history = [
            HumanMessage("search for something"),
            AIMessage(content="", tool_calls=tool_calls),
            ToolMessage(content="search results here", tool_call_id="1"),
            AIMessage("Here's what I found"),
        ]

        # With generous limit, should keep all
        truncated, was_truncated = self.token_budget.limit_conversation_history(
            history, available_tokens=10000
        )

        assert truncated == history
        assert was_truncated is False

        # The token count should properly account for tool_calls in AIMessage
        # Verify by checking that tracked usage is reasonable
        assert self.token_budget.partition.history_tokens_used > 0

    def test_limit_conversation_history_empty(self):
        """Test history limiting with empty history."""
        truncated, was_truncated = self.token_budget.limit_conversation_history(
            [], available_tokens=1000
        )

        assert truncated == []
        assert was_truncated is False
        assert self.token_budget.partition.history_tokens_used == 0

    def test_get_budget_summary(self):
        """Test budget summary generation."""
        # Set up some usage
        self.token_budget.calculate_available_for_prompt(100)
        self.token_budget.track_rag_usage(200)
        self.token_budget.track_history_usage(150)
        self.token_budget.track_tool_usage(300)

        summary = self.token_budget.get_budget_summary()

        assert summary["context_window_size"] == 4000
        assert summary["allocations"]["max_response"] == 1000
        assert summary["allocations"]["max_tools"] == 500
        assert summary["usage"]["prompt"] == 100
        assert summary["usage"]["rag"] == 200
        assert summary["usage"]["history"] == 150
        assert summary["usage"]["tools"] == 300
        assert summary["remaining"]["tools"] == 200  # 500 - 300

    def test_log_budget_summary(self):
        """Test budget summary logging."""
        with mock.patch("ols.utils.token_budget.logger") as mock_logger:
            self.token_budget.log_budget_summary()
            mock_logger.log.assert_called_once()
            # Verify it logs at DEBUG level by default
            call_args = mock_logger.log.call_args
            assert call_args[0][0] == 10  # logging.DEBUG = 10

    def test_integration_full_workflow(self):
        """Test complete workflow: prompt calculation, RAG, history, tools."""
        budget = TokenBudget(
            context_window_size=4000,
            max_tokens_for_response=1000,
            max_tokens_for_tools=500,
        )

        # Step 1: Calculate available tokens for prompt
        base_prompt_tokens = 200
        available = budget.calculate_available_for_prompt(base_prompt_tokens)
        assert available == 2300  # 4000 - 1000 - 500 - 200

        # Step 2: Use some for RAG
        rag_used = 800
        budget.track_rag_usage(rag_used)
        remaining_for_history = available - rag_used  # 1500

        # Step 3: Limit history
        history = [
            HumanMessage("msg1"),
            AIMessage("resp1"),
            HumanMessage("msg2"),
            AIMessage("resp2"),
        ]
        truncated_history, _ = budget.limit_conversation_history(
            history, remaining_for_history
        )
        # With 1500 tokens available, should keep all messages
        assert len(truncated_history) == len(history)

        # Step 4: Track tool usage
        budget.track_tool_usage(100)  # Tool definitions
        budget.track_tool_usage(200)  # Tool execution
        assert budget.partition.tool_tokens_used == 300
        assert budget.partition.get_remaining_tool_budget() == 200

        # Verify final summary
        summary = budget.get_budget_summary()
        assert summary["usage"]["prompt"] == 200
        assert summary["usage"]["rag"] == 800
        assert summary["usage"]["tools"] == 300
        assert summary["remaining"]["tools"] == 200
