"""Tests for LLMClient — the OpenAI wrapper.

Everything is mocked at the edge (see conftest): the ``openai`` SDK's ``OpenAI``
class is patched, and ``OpenAI().chat.completions.create`` returns a canned
response.
"""

from unittest.mock import MagicMock, patch

import pytest
from openai import OpenAIError

from missions.glass_cockpit.llm_client import (
    SYSTEM_PROMPT,
    LLMClient,
    LLMInitialisationError,
    LLMRequestError,
)


def test_send_returns_reply_text(openai_create: MagicMock):
    assert LLMClient().send("hi") == "pong"


def test_send_passes_model_and_messages(openai_create: MagicMock, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MODEL_NAME", "gpt-test")

    LLMClient().send("hi there")

    openai_create.assert_called_once()
    kwargs = openai_create.call_args.kwargs
    assert kwargs["model"] == "gpt-test"
    assert kwargs["messages"] == [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "hi there"},
    ]


def test_each_send_is_independent(openai_create: MagicMock):
    client = LLMClient()
    client.send("first message")
    client.send("second message")

    # No history carried between calls: system + the current user turn only.
    messages = openai_create.call_args_list[1].kwargs["messages"]
    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[-1]["content"] == "second message"


def test_init_raises_initialisation_error_on_construction_failure():
    with (
        patch(
            "missions.glass_cockpit.llm_client.OpenAI",
            side_effect=OpenAIError("no api key configured"),
        ),
        pytest.raises(LLMInitialisationError, match="could not initialise OpenAI client"),
    ):
        LLMClient()


def test_send_raises_request_error_on_api_failure(openai_create: MagicMock):
    openai_create.side_effect = OpenAIError("rate limited")
    with pytest.raises(LLMRequestError, match="request failed"):
        LLMClient().send("hi")
