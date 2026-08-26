"""Tests for the terminal chat loop in chat.py."""

from unittest.mock import patch

import pytest

from missions.glass_cockpit.chat import chat


@pytest.mark.parametrize("exit_cmd", ["exit", "quit", "bye", "EXIT", " Quit ", "BYE"])
def test_chat_exit_keywords(exit_cmd, capsys):
    """Test that typing any exit keyword returns 0 and terminates the loop."""
    with patch("builtins.input", return_value=exit_cmd):
        result = chat()

    assert result == 0
    captured = capsys.readouterr()
    assert "Glass Cockpit — type a message." in captured.out
    assert "you said:" not in captured.out


def test_chat_normal_messages_then_exit(capsys):
    """Test standard message echo before exiting."""
    inputs = ["hello world", "how are you?", "exit"]
    with patch("builtins.input", side_effect=inputs):
        result = chat()

    assert result == 0
    captured = capsys.readouterr()
    assert "Glass Cockpit — type a message. Ctrl+C or 'exit' to quit." in captured.out
    assert "you said: hello world" in captured.out
    assert "you said: how are you?" in captured.out


def test_chat_empty_and_whitespace_inputs(capsys):
    """Test that empty or whitespace-only inputs are ignored and loop continues."""
    inputs = ["", "   ", "\t", "exit"]
    with patch("builtins.input", side_effect=inputs):
        result = chat()

    assert result == 0
    captured = capsys.readouterr()
    assert "you said:" not in captured.out


@pytest.mark.parametrize("exception_cls", [KeyboardInterrupt, EOFError])
def test_chat_interrupt_exceptions(exception_cls, capsys):
    """Test graceful termination on KeyboardInterrupt or EOFError."""
    with patch("builtins.input", side_effect=exception_cls):
        result = chat()

    assert result == 0
    captured = capsys.readouterr()
    assert "Glass Cockpit — type a message." in captured.out
