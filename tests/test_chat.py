"""Tests for the terminal chat loop in chat.py."""

from unittest.mock import MagicMock, patch

import pytest
from conftest import make_response
from openai import OpenAIError

from missions.glass_cockpit.chat import chat


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
    """Each non-exit line is forwarded to the LLM and the reply is printed."""
    openai_create.side_effect = [make_response("hi there"), make_response("doing well")]
    with patch("builtins.input", side_effect=["hello world", "how are you?", "exit"]):
        result = chat()

    assert result == 0
    prompts = [call.kwargs["messages"][-1]["content"] for call in openai_create.call_args_list]
    assert prompts == ["hello world", "how are you?"]
    out = capsys.readouterr().out
    assert "hi there" in out
    assert "doing well" in out


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
