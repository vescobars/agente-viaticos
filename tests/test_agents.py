import importlib
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("GEMINI_API_KEY", "test-key")
os.environ.setdefault("GEMINI_MODEL", "test-model")

import agents  # noqa: E402


@pytest.fixture
def fake_client(monkeypatch):
    client = MagicMock()
    client.interactions.create.return_value = MagicMock(output_text="Hola, ¡gusto en ayudarte!")
    monkeypatch.setattr(agents, "client", client)
    return client


def test_instructions_require_spanish_and_three_sentences():
    assert "español" in agents.INSTRUCTIONS
    assert "3 oraciones" in agents.INSTRUCTIONS


def test_ask_returns_output_text(fake_client):
    assert agents.ask("¿Cuál es la capital de Colombia?") == "Hola, ¡gusto en ayudarte!"


def test_ask_sends_model_input_and_instructions(fake_client):
    agents.ask("  hola  ")
    fake_client.interactions.create.assert_called_once_with(
        model=agents.MODEL,
        input="hola",
        system_instruction=agents.INSTRUCTIONS,
    )


@pytest.mark.parametrize("empty", ["", "   ", "\n"])
def test_ask_rejects_empty_question_without_calling_api(fake_client, empty):
    with pytest.raises(ValueError):
        agents.ask(empty)
    fake_client.interactions.create.assert_not_called()


def _run_agents(env_overrides, tmp_path, stdin_text=""):
    env = {k: v for k, v in os.environ.items() if not k.startswith("GEMINI_")}
    env.update(env_overrides)
    # Run a copy of agents.py in an empty dir: load_dotenv() searches upward from
    # the script, so running the original would pick up the real .env.
    script = tmp_path / "agents.py"
    script.write_text((ROOT / "agents.py").read_text(encoding="utf-8"), encoding="utf-8")
    return subprocess.run(
        [sys.executable, str(script)],
        input=stdin_text,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={**env, "PYTHONIOENCODING": "utf-8"},
        cwd=tmp_path,
        timeout=30,
    )


def test_missing_api_key_exits_with_clear_message(tmp_path):
    result = _run_agents({"GEMINI_MODEL": "m"}, tmp_path)
    assert result.returncode != 0
    assert "GEMINI_API_KEY" in result.stderr
    assert "Traceback" not in result.stderr


def test_missing_model_exits_with_clear_message(tmp_path):
    result = _run_agents({"GEMINI_API_KEY": "k"}, tmp_path)
    assert result.returncode != 0
    assert "GEMINI_MODEL" in result.stderr


def test_empty_input_in_terminal_gives_friendly_message(tmp_path):
    result = _run_agents({"GEMINI_API_KEY": "k", "GEMINI_MODEL": "m"}, tmp_path, stdin_text="\n")
    assert "Traceback" not in result.stderr
    assert "escribe una pregunta" in result.stdout


def test_api_key_is_never_printed_on_error(tmp_path):
    result = _run_agents({"GEMINI_MODEL": "m"}, tmp_path)
    assert "test-key" not in result.stdout + result.stderr


def test_env_files_are_wired_correctly():
    example = (ROOT / ".env.example").read_text(encoding="utf-8")
    assert "GEMINI_API_KEY=" in example
    assert "GEMINI_MODEL=" in example
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert ".env" in gitignore
    assert ".env.example" not in gitignore
