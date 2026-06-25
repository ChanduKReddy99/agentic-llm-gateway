"""
Langfuse First-Time Setup Script
==================================
Run this ONCE after `make up` to:
  1. Verify local Langfuse is reachable
  2. Create all prompt templates in Langfuse prompt registry
  3. Print confirmation with direct links

Run with: uv run python scripts/setup_langfuse.py

After this, agents will fetch prompts from Langfuse at runtime.
You can edit prompts in the UI without touching code.
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

# ── Prompt templates to register ─────────────────────────────────────────────
PROMPTS = {
    "research_system_prompt": {
        "text": """You are a specialized Research Agent. Your job is to:
1. Analyze the user's question to identify key information needs
2. Break it into 2-3 targeted search queries
3. Return a structured research brief

Output format:
## Research Brief
**Query Analysis**: [what the user is really asking]
**Key Findings**: [bullet points of most important facts]
**Context for Synthesis**: [paragraph of rich context]
**Sources**: [list the sources found]
**Confidence**: [High/Medium/Low with reason]""",
        "label": "production",
        "version": 1,
    },
    "research_query_prompt": {
        "text": "User Question: {question}\n\nIdentify the 2-3 most important sub-questions to research. List them briefly as a numbered list.",
        "label": "production",
        "version": 1,
    },
    "research_synthesis_prompt": {
        "text": """User Question: {question}

Search Results Collected:
{search_results}

Synthesize these results into a comprehensive Research Brief following your instructions.
Focus on facts that will help answer the original question.""",
        "label": "production",
        "version": 1,
    },
    "synthesis_system_prompt": {
        "text": """You are a Synthesis Agent specialized in producing clear, accurate,
and well-structured responses from research findings.

Guidelines:
- Directly answer the question first
- Support claims with evidence from the research
- Use markdown for readability
- Acknowledge gaps or uncertainties
- Include a Sources section at the end""",
        "label": "production",
        "version": 1,
    },
    "synthesis_draft_prompt": {
        "text": """Original Question: {question}

Research Brief:
{research_brief}

Additional context: {additional_context}

Synthesize into a comprehensive, well-structured response for the user.""",
        "label": "production",
        "version": 1,
    },
    "synthesis_critique_prompt": {
        "text": """Review this response for quality:

Question: {question}

Response:
{response}

Check:
1. Does it directly answer the question?
2. Are claims supported by the research?
3. Is it clear and well-structured?
4. Are there any factual errors?

If good, reply: "APPROVED: [brief reason]"
If revision needed, provide the improved response directly.""",
        "label": "production",
        "version": 1,
    },
}


def setup_prompts():
    console.print(Panel.fit(
        "[bold cyan]Langfuse First-Time Setup[/bold cyan]\n"
        "Creating prompt templates in local Langfuse...",
        border_style="cyan",
    ))

    from app.config.settings import get_settings
    settings = get_settings()

    console.print(f"\nConnecting to Langfuse at [cyan]{settings.langfuse_host}[/cyan]")
    console.print(f"Public key:  [dim]{settings.langfuse_public_key}[/dim]")
    console.print(f"Secret key:  [dim]{settings.langfuse_secret_key[:20]}...[/dim]\n")

    try:
        from langfuse import Langfuse
        lf = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
    except Exception as e:
        console.print(f"[red]❌ Cannot connect to Langfuse: {e}[/red]")
        console.print("\n[yellow]Make sure the Docker stack is running:[/yellow]")
        console.print("  make up")
        console.print("  # Wait ~30 seconds for Langfuse to start")
        console.print("  make setup-langfuse")
        return False

    # Create all prompts
    table = Table(title="Prompt Registration", header_style="bold cyan", show_lines=True)
    table.add_column("Prompt Name", style="white")
    table.add_column("Status", justify="center")
    table.add_column("Version")

    for name, config in PROMPTS.items():
        try:
            lf.create_prompt(
                name=name,
                prompt=config["text"],
                labels=[config["label"]],
                config={"temperature": 0.3},
            )
            table.add_row(name, "[green]✅ Created[/green]", str(config["version"]))
        except Exception as e:
            err = str(e)
            if "already exists" in err.lower() or "conflict" in err.lower():
                table.add_row(name, "[yellow]⚠ Already exists[/yellow]", "kept")
            else:
                table.add_row(name, f"[red]❌ {err[:40]}[/red]", "—")

    lf.flush()
    console.print(table)

    console.print(Panel(
        f"[bold green]✅ Setup complete![/bold green]\n\n"
        f"Open Langfuse UI:  [cyan]http://localhost:3001[/cyan]\n"
        f"Login:             [dim]admin@example.com / password[/dim]\n\n"
        f"Go to [bold]Prompts[/bold] to see and edit all {len(PROMPTS)} templates.\n"
        f"Changes take effect on the next request — no restart needed.\n\n"
        f"[dim]Agents fetch prompts by name at runtime.\n"
        f"Fallback to hardcoded strings if Langfuse is unreachable.[/dim]",
        border_style="green",
        title="Langfuse Ready",
    ))
    return True


if __name__ == "__main__":
    success = setup_prompts()
    sys.exit(0 if success else 1)
