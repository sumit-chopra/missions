"""Tests for LLMClient — the OpenAI wrapper.

Everything is mocked at the edge (see conftest): the ``openai`` SDK's ``OpenAI``
class is patched, and ``OpenAI().chat.completions.create`` returns a canned
stream of chunks.
"""

from unittest.mock import MagicMock, patch

import pytest
from conftest import make_stream
from openai import OpenAIError

from missions.glass_cockpit.llm_client import (
    SYSTEM_PROMPT,
    LLMClient,
    LLMInitialisationError,
    LLMRequestError,
)


def test_send_streams_reply_text(openai_create: MagicMock):
    openai_create.return_value = make_stream("po", "ng")
    assert "".join(LLMClient().send("hi")) == "pong"


def test_send_skips_chunks_without_content(openai_create: MagicMock):
    openai_create.return_value = make_stream("hi", None, " there")
    assert "".join(LLMClient().send("hi")) == "hi there"


def test_send_passes_model_messages_and_stream_flag(
    openai_create: MagicMock, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("MODEL_NAME", "gpt-test")

    list(LLMClient().send("hi there"))

    openai_create.assert_called_once()
    kwargs = openai_create.call_args.kwargs
    assert kwargs["model"] == "gpt-test"
    assert kwargs["stream"] is True
    assert kwargs["stream_options"] == {"include_usage": True}
    assert kwargs["messages"] == [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "hi there"},
    ]


def test_each_send_is_independent(openai_create: MagicMock):
    client = LLMClient()
    list(client.send("first message"))
    list(client.send("second message"))

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
        list(LLMClient().send("hi"))


def test_send_records_metrics_from_usage_chunk(
    openai_create: MagicMock, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("MODEL_NAME", "gpt-4o-mini")
    openai_create.return_value = make_stream("hi", prompt_tokens=12, completion_tokens=8)

    client = LLMClient()
    list(client.send("hello"))

    assert client.last_metrics is not None
    assert client.last_metrics.model_name == "gpt-4o-mini"
    assert client.last_metrics.prompt_tokens == 12
    assert client.last_metrics.completion_tokens == 8
    assert client.last_metrics.latency_ms >= 0


def test_send_leaves_metrics_unset_on_failure(openai_create: MagicMock):
    openai_create.side_effect = OpenAIError("boom")

    client = LLMClient()
    with pytest.raises(LLMRequestError):
        list(client.send("hi"))

    assert client.last_metrics is None
