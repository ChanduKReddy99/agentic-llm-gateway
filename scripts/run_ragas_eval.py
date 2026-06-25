"""
RAGAS Offline Evaluation Job
=============================
DEV / STAGING ONLY — not part of the production request pipeline.

How to use:
  1. Run your app and collect real queries + responses in dev/staging
  2. Run this script periodically (e.g. nightly CI job) to measure quality
  3. Results go to CSV + Prometheus metrics + Langfuse scores
  4. If scores drop below threshold → alert / block deployment

Run with:
  uv run python scripts/run_ragas_eval.py

In CI:
  make ragas   (exits non-zero if avg score < 0.6)
"""
import asyncio
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table

from app.ragas_eval.evaluator import RAGASEvaluator

console = Console()

# ── Sample dataset (in production: load from logged requests in DB/S3) ────────
EVAL_DATASET = [
    {
        "question": "What is LiteLLM and why is it useful?",
        "answer": "LiteLLM is an open-source LLM gateway providing a unified API over 100+ providers. It offers semantic caching, automatic fallbacks, rate limiting, and cost tracking.",
        "contexts": [
            "LiteLLM is an open-source proxy providing OpenAI-compatible APIs for 100+ LLM providers.",
            "Features include semantic caching in Redis, automatic model fallbacks, and spend tracking.",
        ],
    },
    {
        "question": "How does semantic caching reduce LLM costs?",
        "answer": "Semantic caching stores LLM responses indexed by embedding vectors. Similar queries hit the cache via cosine similarity, returning instantly without an API call.",
        "contexts": [
            "Semantic caching uses vector embeddings to store and retrieve LLM responses by query similarity.",
            "When similarity exceeds the threshold, the cached response is returned without an API call.",
            "Semantic caching can reduce LLM API costs by 30-70% on workloads with repetitive queries.",
        ],
    },
    {
        "question": "What are RAGAS evaluation metrics?",
        "answer": "RAGAS provides: Faithfulness (answer grounded in context?), Answer Relevancy (addresses the question?), Context Precision (retrieved contexts relevant?), Context Recall (all necessary context retrieved?).",
        "contexts": [
            "RAGAS evaluates RAG pipelines using Faithfulness, Answer Relevancy, Context Precision, Context Recall.",
            "Faithfulness measures whether the generated answer is grounded in the provided context.",
        ],
    },
    {
        "question": "What is the purpose of guardrails in agentic AI?",
        "answer": "Guardrails are safety checks at input and output stages. Input guardrails detect PII and prompt injection. Output guardrails check for toxicity and PII leakage.",
        "contexts": [
            "Guardrails are safety mechanisms that validate inputs before they reach LLM agents.",
            "Input guardrails detect PII, prompt injection, and malicious queries.",
            "Output guardrails check responses for toxicity, factual errors, and data leakage.",
        ],
    },
    {
        "question": "How does OpenTelemetry integrate with Tempo?",
        "answer": "OpenTelemetry instruments the app to create spans exported via OTLP to Tempo. Grafana then queries Tempo to display trace waterfalls.",
        "contexts": [
            "OpenTelemetry provides vendor-neutral APIs for distributed tracing.",
            "OTLP exports spans to backends like Tempo.",
            "Grafana integrates with Tempo to display distributed traces alongside logs and metrics.",
        ],
    },
]

QUALITY_THRESHOLD = 0.6   # CI fails below this average score


async def run_eval() -> float:
    console.print("\n[bold cyan]🔬 RAGAS Offline Evaluation[/bold cyan]")
    console.print("[dim]Dev/staging only — not part of the production pipeline[/dim]\n")

    evaluator = RAGASEvaluator()
    results = []

    with console.status("[green]Evaluating..."):
        for sample in EVAL_DATASET:
            scores = await evaluator.evaluate(
                question=sample["question"],
                answer=sample["answer"],
                contexts=sample["contexts"],
            )
            results.append({**sample, "scores": scores})

    # ── Print table ───────────────────────────────────────────────────────────
    table = Table(title="RAGAS Results", header_style="bold cyan", show_lines=True)
    table.add_column("#", width=3)
    table.add_column("Question", max_width=32)
    table.add_column("Faith.", justify="center", width=7)
    table.add_column("Ans.Rel.", justify="center", width=8)
    table.add_column("Ctx.P.", justify="center", width=7)
    table.add_column("Ctx.R.", justify="center", width=7)
    table.add_column("Avg", justify="center", width=6, style="bold")

    def fmt(v: float) -> str:
        color = "green" if v >= 0.8 else "yellow" if v >= 0.6 else "red"
        return f"[{color}]{v:.2f}[/{color}]"

    totals: dict[str, list] = {k: [] for k in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]}

    for i, r in enumerate(results, 1):
        s = r["scores"]
        avg = sum(s.values()) / len(s)
        for k in totals:
            totals[k].append(s.get(k, 0))
        table.add_row(
            str(i), r["question"][:32],
            fmt(s["faithfulness"]), fmt(s["answer_relevancy"]),
            fmt(s["context_precision"]), fmt(s["context_recall"]),
            fmt(avg),
        )

    avgs = {k: sum(v) / len(v) for k, v in totals.items()}
    overall = sum(avgs.values()) / len(avgs)

    table.add_row(
        "—", "[bold]AVERAGE[/bold]",
        fmt(avgs["faithfulness"]), fmt(avgs["answer_relevancy"]),
        fmt(avgs["context_precision"]), fmt(avgs["context_recall"]),
        fmt(overall),
    )
    console.print(table)

    # ── Save CSV ──────────────────────────────────────────────────────────────
    out = Path("ragas_eval_results.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["question", "faithfulness", "answer_relevancy", "context_precision", "context_recall", "average"])
        w.writeheader()
        for r in results:
            s = r["scores"]
            w.writerow({"question": r["question"], **s, "average": sum(s.values()) / len(s)})
    console.print(f"\n[green]Results saved → {out}[/green]")

    # ── CI gate ───────────────────────────────────────────────────────────────
    if overall >= QUALITY_THRESHOLD:
        console.print(f"\n[bold green]✅ Quality gate passed ({overall:.2f} ≥ {QUALITY_THRESHOLD})[/bold green]")
    else:
        console.print(f"\n[bold red]❌ Quality gate FAILED ({overall:.2f} < {QUALITY_THRESHOLD})[/bold red]")
        console.print("[red]Review low-scoring samples and improve retrieval/prompts.[/red]")
        return overall  # caller can sys.exit(1) if needed

    return overall


if __name__ == "__main__":
    score = asyncio.run(run_eval())
    sys.exit(0 if score >= QUALITY_THRESHOLD else 1)
