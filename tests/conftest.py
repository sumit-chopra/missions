"""Shared test fixtures."""

from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import pytest


def make_stream(
    *chunks: str | None, prompt_tokens: int = 0, completion_tokens: int = 0
) -> list[MagicMock]:
    """Build a stand-in for openai's streaming response: an iterable of chunks.

    Each text chunk mimics a ``ChatCompletionChunk`` with one choice whose
    ``delta.content`` is the given text (``None`` for a content-free chunk) and
    ``usage`` unset. A trailing usage-only chunk (empty ``choices``) mirrors
    ``stream_options={"include_usage": True}``.
    """
    text = [MagicMock(choices=[MagicMock(delta=MagicMock(content=c))], usage=None) for c in chunks]
    usage_chunk = MagicMock(
        choices=[],
        usage=MagicMock(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
    )
    return [*text, usage_chunk]


@pytest.fixture
def openai_create() -> Iterator[MagicMock]:
    """Patch ``OpenAI`` in llm_client; yield the mock ``chat.completions.create``.

    Defaults to a stream that yields the single chunk ``"pong"``. Override per
    test via ``openai_create.return_value`` or ``openai_create.side_effect``.
    """
    with patch("missions.glass_cockpit.llm_client.OpenAI") as openai_cls:
        create = openai_cls.return_value.chat.completions.create
        create.return_value = make_stream("pong")
        yield create
