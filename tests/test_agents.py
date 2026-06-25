"""
Tests for Research Agent and Synthesis Agent.
All LLM calls are mocked — no real API keys needed.
"""
import pytest
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))


def make_mock_gateway_response(
    content: str = "Test response",
    cache_hit: bool = False,
    cost_usd: float = 0.00015,
):
    """Create a mock LiteLLM gateway response including cost_usd."""
    return {
        "content": content,
        "model": "gpt-4o-mini",
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        },
        "cost_usd": cost_usd,
        "cache_hit": cache_hit,
        "latency_seconds": 0.5,
        "finish_reason": "stop",
    }


@pytest.fixture
def mock_gateway():
    gateway = AsyncMock()
    gateway.chat_completion = AsyncMock(
        return_value=make_mock_gateway_response(
            content=(
                "1. What is LiteLLM?\n"
                "2. How does caching work?\n"
                "3. What are the benefits?\n\n"
                "## Research Brief\n"
                "**Query Analysis**: User wants to understand LiteLLM\n"
                "**Key Findings**: LiteLLM provides unified API, caching, fallbacks\n"
                "**Context**: LiteLLM is widely used in production AI systems\n"
                "**Sources**: litellm.ai/docs\n"
                "**Confidence**: High"
            )
        )
    )
    return gateway


@pytest.fixture
def mock_search_tool():
    tool = AsyncMock()
    tool.search = AsyncMock(return_value={
        "query": "LiteLLM benefits",
        "results": [
            {
                "snippet": "LiteLLM provides semantic caching and fallbacks.",
                "source": "litellm.ai/docs",
                "relevance_score": 0.95,
            }
        ],
        "total_results": 1,
        "topic_matched": "llm gateway",
    })
    tool.summarize_results = AsyncMock(
        return_value="Search results:\n[1] LiteLLM provides semantic caching. Source: litellm.ai"
    )
    return tool


class TestResearchAgent:

    @pytest.mark.asyncio
    async def test_run_returns_expected_keys(self, mock_gateway, mock_search_tool):
        from app.agents.research_agent import ResearchAgent
        agent = ResearchAgent(gateway=mock_gateway, search_tool=mock_search_tool)
        result = await agent.run(
            query="What are the benefits of LiteLLM?",
            request_id="test-001",
        )
        assert "research_brief" in result
        assert "sources" in result
        assert "sub_queries" in result
        assert "token_usage" in result
        assert "cache_hits" in result
        assert "cost_usd" in result           # ← new: cost rolled up from gateway

    @pytest.mark.asyncio
    async def test_cost_usd_is_sum_of_calls(self, mock_gateway, mock_search_tool):
        """cost_usd in result must equal sum of all gateway call costs."""
        from app.agents.research_agent import ResearchAgent
        cost_per_call = 0.00015
        mock_gateway.chat_completion = AsyncMock(
            return_value=make_mock_gateway_response(cost_usd=cost_per_call)
        )
        agent = ResearchAgent(gateway=mock_gateway, search_tool=mock_search_tool)
        result = await agent.run(query="test", request_id="test-cost")
        # Two LLM calls: decompose + synthesis
        assert result["cost_usd"] == pytest.approx(cost_per_call * 2, rel=1e-3)

    @pytest.mark.asyncio
    async def test_extract_queries_parses_numbered_list(self, mock_gateway, mock_search_tool):
        from app.agents.research_agent import ResearchAgent
        agent = ResearchAgent(gateway=mock_gateway, search_tool=mock_search_tool)
        text = "1. What is LiteLLM?\n2. How does caching work?\n3. What are fallbacks?"
        queries = agent._extract_queries(text, "original query")
        assert len(queries) <= 3
        assert len(queries) >= 1

    @pytest.mark.asyncio
    async def test_extract_queries_falls_back_to_original(self, mock_gateway, mock_search_tool):
        from app.agents.research_agent import ResearchAgent
        agent = ResearchAgent(gateway=mock_gateway, search_tool=mock_search_tool)
        queries = agent._extract_queries("", "my original question")
        assert "my original question" in queries

    @pytest.mark.asyncio
    async def test_gateway_called_multiple_times(self, mock_gateway, mock_search_tool):
        from app.agents.research_agent import ResearchAgent
        agent = ResearchAgent(gateway=mock_gateway, search_tool=mock_search_tool)
        await agent.run(query="Explain RAGAS metrics", request_id="test-002")
        # decompose + synthesis = at least 2 gateway calls
        assert mock_gateway.chat_completion.call_count >= 2

    @pytest.mark.asyncio
    async def test_cache_hits_tracked(self, mock_search_tool):
        from app.agents.research_agent import ResearchAgent
        gateway = AsyncMock()
        gateway.chat_completion = AsyncMock(
            return_value=make_mock_gateway_response(cache_hit=True, cost_usd=0.0)
        )
        agent = ResearchAgent(gateway=gateway, search_tool=mock_search_tool)
        result = await agent.run(query="Cached query test", request_id="test-003")
        assert result["cache_hits"] >= 0
        assert result["cost_usd"] == 0.0   # cache hits cost nothing


class TestSynthesisAgent:

    @pytest.mark.asyncio
    async def test_run_returns_expected_keys(self, mock_gateway):
        from app.agents.synthesis_agent import SynthesisAgent
        agent = SynthesisAgent(gateway=mock_gateway)
        result = await agent.run(
            query="What is LiteLLM?",
            research_output={
                "research_brief": "## Research Brief\nLiteLLM is great.",
                "sources": ["litellm.ai"],
                "token_usage": {},
            },
            request_id="test-syn-001",
        )
        assert "final_response" in result
        assert "draft_response" in result
        assert "critique" in result
        assert "was_revised" in result
        assert "token_usage" in result
        assert "cache_hits" in result
        assert "cost_usd" in result           # ← new: cost rolled up from gateway

    @pytest.mark.asyncio
    async def test_cost_usd_is_sum_of_calls(self, mock_gateway):
        """cost_usd must equal draft + critique call costs."""
        from app.agents.synthesis_agent import SynthesisAgent
        cost_per_call = 0.00020
        mock_gateway.chat_completion = AsyncMock(
            return_value=make_mock_gateway_response(cost_usd=cost_per_call)
        )
        agent = SynthesisAgent(gateway=mock_gateway)
        result = await agent.run(
            query="test",
            research_output={"research_brief": "brief", "sources": []},
        )
        # Two LLM calls: draft + critique
        assert result["cost_usd"] == pytest.approx(cost_per_call * 2, rel=1e-3)

    @pytest.mark.asyncio
    async def test_gateway_called_for_draft_and_critique(self, mock_gateway):
        from app.agents.synthesis_agent import SynthesisAgent
        agent = SynthesisAgent(gateway=mock_gateway)
        await agent.run(
            query="Explain caching",
            research_output={"research_brief": "Caching is fast.", "sources": []},
            request_id="test-syn-002",
        )
        assert mock_gateway.chat_completion.call_count >= 2

    @pytest.mark.asyncio
    async def test_approved_critique_uses_draft(self):
        from app.agents.synthesis_agent import SynthesisAgent
        gateway = AsyncMock()
        gateway.chat_completion = AsyncMock(side_effect=[
            make_mock_gateway_response("This is the draft response about LiteLLM."),
            make_mock_gateway_response("APPROVED: The response is accurate and complete."),
        ])
        agent = SynthesisAgent(gateway=gateway)
        result = await agent.run(
            query="What is LiteLLM?",
            research_output={"research_brief": "Research brief.", "sources": []},
        )
        assert result["was_revised"] is False
        assert result["final_response"] == "This is the draft response about LiteLLM."

    @pytest.mark.asyncio
    async def test_cache_hit_response_costs_zero(self):
        from app.agents.synthesis_agent import SynthesisAgent
        gateway = AsyncMock()
        gateway.chat_completion = AsyncMock(
            return_value=make_mock_gateway_response(cache_hit=True, cost_usd=0.0)
        )
        agent = SynthesisAgent(gateway=gateway)
        result = await agent.run(
            query="cached",
            research_output={"research_brief": "brief", "sources": []},
        )
        assert result["cost_usd"] == 0.0
        assert result["cache_hits"] >= 1
