"""Shared test fixtures."""

from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import pytest


def make_response(content: str | None) -> MagicMock:
    """Build a stand-in for an openai ChatCompletion response object."""
    return MagicMock(choices=[MagicMock(message=MagicMock(content=content))])


@pytest.fixture
def openai_create() -> Iterator[MagicMock]:
    """Patch ``OpenAI`` in llm_client; yield the mock ``chat.completions.create``.

    Defaults to returning a response whose text is ``"pong"``. Override per test
    via ``openai_create.return_value`` or ``openai_create.side_effect``.
    """
    with patch("missions.glass_cockpit.llm_client.OpenAI") as openai_cls:
        create = openai_cls.return_value.chat.completions.create
        create.return_value = make_response("pong")
        yield create
