"""Tests for the terminal chat loop in chat.py."""

import json
from unittest.mock import MagicMock, patch

import pytest
from conftest import make_stream
from openai import OpenAIError

from missions.glass_cockpit.chat import chat, emit
from missions.glass_cockpit.telemetry import LLMMetrics


@pytest.mark.parametrize("exit_cmd", ["exit", "quit", "bye", "EXIT", " Quit ", "BYE"])
def test_chat_exit_keywords(
    exit_cmd: str, openai_create: MagicMock, capsys: pytest.CaptureFixture[str]
):
    """Typing any exit keyword returns 0 without hitting the LLM."""
    with patch("builtins.input", return_value=exit_cmd):
        result = chat()

    assert result == 0
    openai_create.assert_not_called()
    assert "Glass Cockpit — type a message." in capsys.readouterr().out


def test_chat_sends_messages_to_llm_then_exits(
    openai_create: MagicMock, capsys: pytest.CaptureFixture[str]
):
    """Each non-exit line is forwarded to the LLM and the streamed reply is printed."""
    openai_create.side_effect = [
        make_stream("hi", " ", "there", prompt_tokens=5, completion_tokens=2),
        make_stream("doing", " ", "well", prompt_tokens=7, completion_tokens=3),
    ]
    with patch("builtins.input", side_effect=["hello world", "how are you?", "exit"]):
        result = chat()

    assert result == 0
    prompts = [call.kwargs["messages"][-1]["content"] for call in openai_create.call_args_list]
    assert prompts == ["hello world", "how are you?"]

    captured = capsys.readouterr()
    assert "hi there" in captured.out
    assert "doing well" in captured.out
    # One telemetry line per call, on stdout...
    assert captured.out.count("[stats]") == 2
    # ...and one JSON object per call on stderr, each parseable for jq.
    metrics = [json.loads(line) for line in captured.err.splitlines() if line.strip()]
    assert [m["prompt_tokens"] for m in metrics] == [5, 7]


def test_chat_empty_and_whitespace_inputs(
    openai_create: MagicMock, capsys: pytest.CaptureFixture[str]
):
    """Empty or whitespace-only inputs are ignored and never reach the LLM."""
    with patch("builtins.input", side_effect=["", "   ", "\t", "exit"]):
        result = chat()

    assert result == 0
    openai_create.assert_not_called()


@pytest.mark.parametrize("exception_cls", [KeyboardInterrupt, EOFError])
def test_chat_interrupt_exceptions(
    exception_cls: type[BaseException],
    openai_create: MagicMock,
    capsys: pytest.CaptureFixture[str],
):
    """Graceful termination on KeyboardInterrupt or EOFError."""
    with patch("builtins.input", side_effect=exception_cls):
        result = chat()

    assert result == 0
    assert "Glass Cockpit — type a message." in capsys.readouterr().out


def test_chat_reports_request_errors_and_keeps_going(
    openai_create: MagicMock, capsys: pytest.CaptureFixture[str]
):
    """A failed request is reported without crashing the loop."""
    openai_create.side_effect = OpenAIError("rate limited")
    with patch("builtins.input", side_effect=["hello", "exit"]):
        result = chat()

    assert result == 0
    assert "error: request failed" in capsys.readouterr().out


def test_chat_returns_1_on_initialisation_error(capsys: pytest.CaptureFixture[str]):
    """A client that cannot be constructed (e.g. missing API key) exits with status 1."""
    with patch(
        "missions.glass_cockpit.llm_client.OpenAI",
        side_effect=OpenAIError("no api key configured"),
    ):
        result = chat()

    assert result == 1
    assert "Could not initialise the LLM client" in capsys.readouterr().out


@pytest.fixture
def metrics() -> LLMMetrics:
    return LLMMetrics(
        model_name="gpt-4o-mini",
        prompt_tokens=100,
        completion_tokens=50,
        latency_ms=1234,
    )


def test_emit_writes_human_line_to_stdout(metrics: LLMMetrics, capsys: pytest.CaptureFixture[str]):
    emit(metrics)

    out, _ = capsys.readouterr()
    assert out.strip() == str(metrics)
    assert "[stats]" in out


def test_emit_writes_single_json_line_to_stderr(
    metrics: LLMMetrics, capsys: pytest.CaptureFixture[str]
):
    emit(metrics)

    _, err = capsys.readouterr()
    assert err.count("\n") == 1  # newline-delimited: one JSON object per line, for jq

    parsed = json.loads(err)
    assert parsed["model_name"] == "gpt-4o-mini"
    assert parsed["prompt_tokens"] == 100
    assert parsed["completion_tokens"] == 50
    assert parsed["latency_ms"] == 1234
