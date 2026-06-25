"""
Tests for RAGAS Evaluator
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.ragas_eval.evaluator import RAGASEvaluator


@pytest.fixture
def evaluator():
    return RAGASEvaluator()


class TestRAGASEvaluator:

    @pytest.mark.asyncio
    async def test_returns_all_metrics(self, evaluator):
        scores = await evaluator.evaluate(
            question="What is LiteLLM?",
            answer="LiteLLM is an open-source LLM gateway.",
            contexts=["LiteLLM is an open-source proxy for LLM providers."],
        )
        assert "faithfulness" in scores
        assert "answer_relevancy" in scores
        assert "context_precision" in scores
        assert "context_recall" in scores

    @pytest.mark.asyncio
    async def test_scores_in_range(self, evaluator):
        scores = await evaluator.evaluate(
            question="What is RAGAS?",
            answer="RAGAS evaluates RAG pipeline quality.",
            contexts=["RAGAS is a framework for evaluating RAG pipelines."],
        )
        for metric, score in scores.items():
            assert 0.0 <= score <= 1.0, f"{metric} score {score} out of range"

    @pytest.mark.asyncio
    async def test_empty_contexts_returns_zero(self, evaluator):
        scores = await evaluator.evaluate(
            question="What is X?",
            answer="X is something.",
            contexts=[],
        )
        # Context-based metrics should be 0 when no contexts provided
        assert scores["context_precision"] == 0.0
        assert scores["context_recall"] == 0.0

    @pytest.mark.asyncio
    async def test_highly_relevant_scores_high(self, evaluator):
        """A perfect answer grounded in context should score high."""
        scores = await evaluator.evaluate(
            question="What is semantic caching in LLMs?",
            answer=(
                "Semantic caching stores LLM responses indexed by embedding vectors. "
                "Similar queries are matched using cosine similarity and return cached responses."
            ),
            contexts=[
                "Semantic caching stores LLM responses indexed by embedding vectors.",
                "Cosine similarity is used to match new queries against cached query embeddings.",
                "When similarity exceeds the threshold, the cached LLM response is returned.",
            ],
        )
        # High-quality answer should have decent faithfulness
        assert scores["faithfulness"] >= 0.4

    @pytest.mark.asyncio
    async def test_batch_evaluate(self, evaluator):
        samples = [
            {
                "question": "What is A?",
                "answer": "A is the first letter.",
                "contexts": ["A is used as the first letter of the alphabet."],
            },
            {
                "question": "What is B?",
                "answer": "B is the second letter.",
                "contexts": ["B follows A in the alphabet sequence."],
            },
        ]
        results = await evaluator.batch_evaluate(samples)
        assert len(results) == 2
        for result in results:
            assert "ragas_scores" in result
            assert "faithfulness" in result["ragas_scores"]

    @pytest.mark.asyncio
    async def test_empty_answer_handled(self, evaluator):
        """Should not crash on empty answer."""
        scores = await evaluator.evaluate(
            question="What is X?",
            answer="",
            contexts=["X is a variable."],
        )
        assert isinstance(scores, dict)
        for score in scores.values():
            assert score == 0.0
