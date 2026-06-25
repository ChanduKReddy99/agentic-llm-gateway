"""
Tests for manage_prompts.py CLI — prompt CI/CD manager.
All Langfuse calls are mocked.
"""
import sys
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def mock_langfuse():
    lf = MagicMock()
    # Mock a prompt object returned by get_prompt
    prompt_obj = MagicMock()
    prompt_obj.prompt   = "You are a research agent."
    prompt_obj.version  = 3
    prompt_obj.labels   = ["production", "latest"]
    lf.get_prompt.return_value = prompt_obj
    return lf


@pytest.fixture
def prompts_dir(tmp_path):
    """Create a temporary prompts/ directory with sample files."""
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()

    # Write sample prompt files
    (prompts_dir / "research_system_prompt.txt").write_text("You are a research agent v1.")
    (prompts_dir / "research_query_prompt.txt").write_text("Question: {question}")
    (prompts_dir / "research_synthesis_prompt.txt").write_text("Synthesise: {search_results}")
    (prompts_dir / "synthesis_system_prompt.txt").write_text("You are a synthesis agent.")
    (prompts_dir / "synthesis_draft_prompt.txt").write_text("Draft: {question}")
    (prompts_dir / "synthesis_critique_prompt.txt").write_text("Critique: {response}")

    # Write registry yaml
    import yaml
    registry = {
        "prompts": [
            {"name": "research_system_prompt",    "file": "research_system_prompt.txt",    "label": "production", "config": {}},
            {"name": "research_query_prompt",     "file": "research_query_prompt.txt",     "label": "production", "config": {}},
            {"name": "research_synthesis_prompt", "file": "research_synthesis_prompt.txt", "label": "production", "config": {}},
            {"name": "synthesis_system_prompt",   "file": "synthesis_system_prompt.txt",   "label": "production", "config": {}},
            {"name": "synthesis_draft_prompt",    "file": "synthesis_draft_prompt.txt",    "label": "production", "config": {}},
            {"name": "synthesis_critique_prompt", "file": "synthesis_critique_prompt.txt", "label": "production", "config": {}},
        ],
        "quality_gate": {
            "ragas_avg_threshold": 0.60,
            "faithfulness_threshold": 0.55,
            "answer_relevancy_threshold": 0.60,
        }
    }
    (prompts_dir / "prompts.yaml").write_text(yaml.dump(registry))
    return prompts_dir


class TestPromptsDirectory:
    """Verify the prompts/ directory structure is correct."""

    def test_all_prompt_files_exist(self):
        prompts_dir = Path(__file__).parent.parent / "prompts"
        expected_files = [
            "research_system_prompt.txt",
            "research_query_prompt.txt",
            "research_synthesis_prompt.txt",
            "synthesis_system_prompt.txt",
            "synthesis_draft_prompt.txt",
            "synthesis_critique_prompt.txt",
            "prompts.yaml",
        ]
        for fname in expected_files:
            assert (prompts_dir / fname).exists(), f"Missing: prompts/{fname}"

    def test_prompts_yaml_valid(self):
        import yaml
        registry_path = Path(__file__).parent.parent / "prompts" / "prompts.yaml"
        with open(registry_path) as f:
            registry = yaml.safe_load(f)

        assert "prompts" in registry
        assert "quality_gate" in registry
        assert len(registry["prompts"]) == 6

        for p in registry["prompts"]:
            assert "name" in p
            assert "file" in p
            assert "label" in p

    def test_all_prompt_files_non_empty(self):
        prompts_dir = Path(__file__).parent.parent / "prompts"
        for txt_file in prompts_dir.glob("*.txt"):
            content = txt_file.read_text(encoding="utf-8").strip()
            assert len(content) > 20, f"Prompt file too short: {txt_file.name}"

    def test_prompt_files_referenced_in_yaml(self):
        import yaml
        prompts_dir = Path(__file__).parent.parent / "prompts"
        with open(prompts_dir / "prompts.yaml") as f:
            registry = yaml.safe_load(f)

        for p in registry["prompts"]:
            filepath = prompts_dir / p["file"]
            assert filepath.exists(), f"File in registry not found: {p['file']}"

    def test_quality_gate_thresholds_reasonable(self):
        import yaml
        with open(Path(__file__).parent.parent / "prompts" / "prompts.yaml") as f:
            registry = yaml.safe_load(f)

        gate = registry["quality_gate"]
        assert 0.0 < gate["ragas_avg_threshold"] < 1.0
        assert 0.0 < gate["faithfulness_threshold"] < 1.0
        assert 0.0 < gate["answer_relevancy_threshold"] < 1.0

    def test_research_system_prompt_has_required_sections(self):
        """Research system prompt must define the Brief output format."""
        content = (Path(__file__).parent.parent / "prompts" / "research_system_prompt.txt").read_text(encoding="utf-8")
        assert "Research Brief" in content
        assert "Key Findings"   in content
        assert "Sources"        in content

    def test_synthesis_critique_prompt_has_approved_keyword(self):
        """Critique prompt must contain APPROVED so agents can detect approval."""
        content = (Path(__file__).parent.parent / "prompts" / "synthesis_critique_prompt.txt").read_text(encoding="utf-8")
        assert "APPROVED" in content

    def test_prompts_have_correct_template_variables(self):
        """Each prompt must contain its expected {variable} placeholders."""
        prompts_dir = Path(__file__).parent.parent / "prompts"
        checks = {
            "research_query_prompt.txt":     ["{question}"],
            "research_synthesis_prompt.txt": ["{question}", "{search_results}"],
            "synthesis_draft_prompt.txt":    ["{question}", "{research_brief}"],
            "synthesis_critique_prompt.txt": ["{question}", "{response}"],
        }
        for filename, required_vars in checks.items():
            content = (prompts_dir / filename).read_text(encoding="utf-8")
            for var in required_vars:
                assert var in content, f"{filename} missing template var {var}"


class TestGitHubWorkflows:
    """Verify the GitHub Actions workflow files are present and valid."""

    def test_all_workflows_exist(self):
        workflows_dir = Path(__file__).parent.parent / ".github" / "workflows"
        expected = [
            "prompt-push.yml",
            "prompt-promote.yml",
            "ragas-eval.yml",
        ]
        for wf in expected:
            assert (workflows_dir / wf).exists(), f"Missing workflow: {wf}"

    def test_prompt_push_triggers_on_main(self):
        wf_path = Path(__file__).parent.parent / ".github" / "workflows" / "prompt-push.yml"
        content  = wf_path.read_text(encoding="utf-8")
        # Check raw text — 'on:' is a YAML boolean trap, easier to check source
        assert "branches:" in content
        assert "- main" in content

    def test_prompt_push_only_triggers_on_prompt_changes(self):
        wf_path  = Path(__file__).parent.parent / ".github" / "workflows" / "prompt-push.yml"
        content  = wf_path.read_text(encoding="utf-8")
        assert "prompts/" in content or "prompts/**" in content

    def test_ragas_eval_triggers_on_pr(self):
        wf_path = Path(__file__).parent.parent / ".github" / "workflows" / "ragas-eval.yml"
        content = wf_path.read_text(encoding="utf-8")
        assert "pull_request:" in content

    def test_promote_triggers_on_version_tag(self):
        wf_path = Path(__file__).parent.parent / ".github" / "workflows" / "prompt-promote.yml"
        content = wf_path.read_text(encoding="utf-8")
        assert "tags:" in content
        assert "- v*" in content or '"v*"' in content

    def test_all_workflows_use_secrets_for_langfuse(self):
        """All workflows must use GitHub secrets, not hardcoded keys."""
        workflows_dir = Path(__file__).parent.parent / ".github" / "workflows"
        for wf_file in workflows_dir.glob("*.yml"):
            content = wf_file.read_text(encoding="utf-8")
            # Must reference secrets, not hardcoded values
            assert "secrets.LANGFUSE" in content, f"{wf_file.name} missing Langfuse secrets"
            # Must NOT have hardcoded key values
            assert "pk-lf-agentic-local" not in content, f"{wf_file.name} has hardcoded key"
