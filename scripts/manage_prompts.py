"""
Prompt CI/CD Manager
=====================
Used by GitHub Actions to push/pull/promote prompts between Git and Langfuse.

Commands:
  push     → read prompts/ folder → push to Langfuse as new version
  pull     → fetch current production prompts from Langfuse → write to prompts/
  promote  → change prompt label from staging → production
  diff     → compare local prompts/ vs current Langfuse production versions
  status   → show all prompt versions and labels in Langfuse

Usage:
  uv run python scripts/manage_prompts.py push --label staging
  uv run python scripts/manage_prompts.py push --label production
  uv run python scripts/manage_prompts.py pull
  uv run python scripts/manage_prompts.py promote --from-label staging --to-label production
  uv run python scripts/manage_prompts.py diff
  uv run python scripts/manage_prompts.py status

Called by CI:
  .github/workflows/prompt-push.yml    → on merge to main: push --label staging
  .github/workflows/prompt-promote.yml → on tag v*: promote staging → production
  .github/workflows/ragas-eval.yml     → on PR: pull + eval + comment score
"""
import sys
import typer
import yaml
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table

console = Console()
app = typer.Typer(help="Prompt CI/CD manager — syncs prompts between Git and Langfuse")

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
REGISTRY    = PROMPTS_DIR / "prompts.yaml"


def _get_langfuse():
    from app.config.settings import get_settings
    settings = get_settings()
    from langfuse import Langfuse
    lf = Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host,
    )
    return lf


def _load_registry() -> dict:
    with open(REGISTRY) as f:
        return yaml.safe_load(f)


# ===== push ===========================================

@app.command()
def push(
    label: str = typer.Option("staging", help="Label to assign: staging | production"),
    dry_run: bool = typer.Option(False, help="Print what would be pushed without doing it"),
):
    """
    Push local prompts/ → Langfuse as a new version.

    Called by CI on merge to main:
      push --label staging

    Called manually to publish to production:
      push --label production
    """
    console.print(f"\n[bold cyan]Pushing prompts to Langfuse[/bold cyan] (label=[yellow]{label}[/yellow])")

    registry = _load_registry()
    prompts  = registry["prompts"]

    if dry_run:
        console.print("[yellow]DRY RUN — not actually pushing[/yellow]")

    table = Table(title=f"Push results (label={label})", header_style="bold cyan", show_lines=True)
    table.add_column("Prompt Name")
    table.add_column("File")
    table.add_column("Status", justify="center")
    table.add_column("New Version", justify="center")

    lf = None if dry_run else _get_langfuse()
    errors = []

    for p in prompts:
        name     = p["name"]
        filepath = PROMPTS_DIR / p["file"]
        config   = p.get("config", {})

        if not filepath.exists():
            table.add_row(name, p["file"], "[red]❌ file not found[/red]", "—")
            errors.append(name)
            continue

        text = filepath.read_text().strip()

        if dry_run:
            table.add_row(name, p["file"], "[dim]skipped (dry run)[/dim]", "—")
            continue

        try:
            result = lf.create_prompt(
                name=name,
                prompt=text,
                labels=[label, "latest"],
                config=config,
            )
            version = getattr(result, "version", "?")
            table.add_row(name, p["file"], "[green]✅ pushed[/green]", str(version))
        except Exception as e:
            table.add_row(name, p["file"], f"[red]❌ {str(e)[:40]}[/red]", "—")
            errors.append(name)

    if lf:
        lf.flush()

    console.print(table)

    if errors:
        console.print(f"\n[red]❌ {len(errors)} prompt(s) failed: {errors}[/red]")
        raise typer.Exit(1)

    console.print(f"\n[green]✅ {len(prompts)} prompts pushed with label=[bold]{label}[/bold][/green]")
    console.print(f"   View in Langfuse: http://localhost:3001 → Prompts")


# ==== pull =====================================================================

@app.command()
def pull(
    label: str = typer.Option("production", help="Which label to pull"),
):
    """
    Pull current prompts from Langfuse → write to prompts/ folder.

    Useful for:
      - Syncing local dev environment with production prompts
      - Reviewing what's currently live before making changes
    """
    console.print(f"\n[bold cyan]Pulling prompts from Langfuse[/bold cyan] (label=[yellow]{label}[/yellow])")

    registry = _load_registry()
    lf = _get_langfuse()

    table = Table(title=f"Pull results (label={label})", header_style="bold cyan", show_lines=True)
    table.add_column("Prompt Name")
    table.add_column("Version", justify="center")
    table.add_column("Status", justify="center")

    for p in registry["prompts"]:
        name     = p["name"]
        filepath = PROMPTS_DIR / p["file"]
        try:
            prompt_obj = lf.get_prompt(name, label=label)
            text       = prompt_obj.prompt  # raw template text
            version    = getattr(prompt_obj, "version", "?")
            filepath.write_text(text)
            table.add_row(name, str(version), "[green]✅ written[/green]")
        except Exception as e:
            table.add_row(name, "—", f"[red]❌ {str(e)[:50]}[/red]")

    console.print(table)
    console.print(f"\n[green]Prompts written to {PROMPTS_DIR}[/green]")


# ==== promote ====================================================================

@app.command()
def promote(
    from_label: str = typer.Option("staging",    help="Source label"),
    to_label:   str = typer.Option("production", help="Target label"),
):
    """
    Promote prompts from one label to another in Langfuse.

    Called by CI on Git tag push (e.g. v1.2.0):
      promote --from-label staging --to-label production

    This does NOT change the prompt text — it just moves the label.
    Agents fetching label=production will immediately use the new version.
    """
    console.print(
        f"\n[bold cyan]Promoting prompts[/bold cyan] "
        f"[yellow]{from_label}[/yellow] → [green]{to_label}[/green]"
    )

    registry = _load_registry()
    lf       = _get_langfuse()

    table = Table(title="Promotion results", header_style="bold cyan", show_lines=True)
    table.add_column("Prompt Name")
    table.add_column("Version", justify="center")
    table.add_column("Status", justify="center")

    errors = []
    for p in registry["prompts"]:
        name = p["name"]
        try:
            # Fetch the staging version, explicitly bypassing the 60s SDK cache
            staging = lf.get_prompt(name, label=from_label, cache_ttl_seconds=0)
            version = getattr(staging, "version", None)

            if version is None:
                raise ValueError(f"Could not resolve version for prompt '{name}' with label '{from_label}'")
            
            # update the prompt version
            # this assigns production (to_label) to this specific version number.
            # Langfuse will automatically remove the production label from any older version
            lf.update_prompt(
                name       = name,
                version    = version,
                new_labels = [to_label],
            )
            table.add_row(name, str(version), f"[green]✅ {from_label}→{to_label}[/green]")
        except Exception as e:
            error_msg = str(e)
            if len(error_msg) > 80:
                error_msg = error_msg[:80] + "..."
            table.add_row(name, "—", f"[red]❌ {error_msg}[/red]")
            console.print(f"[red]Full error for {name}:[/red] {str(e)}")
            errors.append(name)

    lf.flush()
    console.print(table)

    if errors:
        console.print(f"\n[red]❌ Promotion failed for: {errors}[/red]")
        raise typer.Exit(1)

    console.print(f"\n[green]✅ All prompts promoted to [bold]{to_label}[/bold][/green]")
    console.print("   Agents will use the new version on the next request.")


# ===== diff =====================================================================

@app.command()
def diff(
    label: str = typer.Option("production", help="Which Langfuse label to compare against"),
):
    """
    Compare local prompts/ vs current Langfuse prompts.

    Run before pushing to see what will change.
    """
    import difflib
    console.print(f"\n[bold cyan]Diffing local prompts vs Langfuse[/bold cyan] (label={label})\n")

    registry = _load_registry()
    lf       = _get_langfuse()
    has_diff = False

    for p in registry["prompts"]:
        name     = p["name"]
        filepath = PROMPTS_DIR / p["file"]
        local    = filepath.read_text().strip() if filepath.exists() else ""

        try:
            remote_obj = lf.get_prompt(name, label=label)
            remote     = remote_obj.prompt.strip()
            version    = getattr(remote_obj, "version", "?")
        except Exception:
            console.print(f"[yellow]{name}[/yellow]: not found in Langfuse ({label})")
            has_diff = True
            continue

        if local == remote:
            console.print(f"[green]✓ {name}[/green] (v{version}) — no changes")
        else:
            has_diff = True
            console.print(f"[yellow]~ {name}[/yellow] (v{version}) — CHANGED:")
            diff_lines = list(difflib.unified_diff(
                remote.splitlines(keepends=True),
                local.splitlines(keepends=True),
                fromfile=f"langfuse/{name} v{version}",
                tofile=f"local/{p['file']}",
                n=2,
            ))
            for line in diff_lines[:30]:
                if line.startswith("+"):
                    console.print(f"  [green]{line.rstrip()}[/green]")
                elif line.startswith("-"):
                    console.print(f"  [red]{line.rstrip()}[/red]")
                else:
                    console.print(f"  [dim]{line.rstrip()}[/dim]")

    if not has_diff:
        console.print("\n[green]✅ All prompts are in sync with Langfuse[/green]")
    else:
        console.print("\n[yellow]⚠ Differences found. Run 'push' to sync.[/yellow]")


# ===== status ====================================================================

@app.command()
def status():
    """Show all prompt versions and labels currently in Langfuse."""
    console.print("\n[bold cyan]Langfuse Prompt Status[/bold cyan]\n")

    registry = _load_registry()
    lf       = _get_langfuse()

    table = Table(title="Prompts in Langfuse", header_style="bold cyan", show_lines=True)
    table.add_column("Prompt Name")
    table.add_column("Labels", justify="center")
    table.add_column("Latest Version", justify="center")
    table.add_column("In Git", justify="center")

    for p in registry["prompts"]:
        name     = p["name"]
        in_git   = "✅" if (PROMPTS_DIR / p["file"]).exists() else "❌"
        try:
            prompt  = lf.get_prompt(name)
            labels  = ", ".join(getattr(prompt, "labels", []) or ["none"])
            version = str(getattr(prompt, "version", "?"))
            table.add_row(name, labels, version, in_git)
        except Exception:
            table.add_row(name, "[dim]not in Langfuse[/dim]", "—", in_git)

    console.print(table)
    # console.print(f"\nLangfuse UI: http://localhost:3001 → Prompts")
    # Get the host dynamically and clean up any trailing "/api"
    from app.config.settings import get_settings
    host_url = (get_settings().langfuse_host or "https://cloud.langfuse.com").split("/api")[0].rstrip("/")

    console.print(f"\nLangfuse UI: {host_url}/prompts")


if __name__ == "__main__":
    app()
