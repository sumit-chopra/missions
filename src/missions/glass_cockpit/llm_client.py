"""Thin wrapper around the OpenAI chat completions API.

Each call is independent — no conversation history is kept.
"""

import os
from collections.abc import Iterator

from openai import OpenAI, OpenAIError

DEFAULT_MODEL = "gpt-4o-mini"
SYSTEM_PROMPT = "You are Glass Cockpit, a concise and helpful terminal assistant."
""" Max completion tokens to prevent runaway costs """
MAX_COMPLETION_TOKENS = 1024


class LLMError(Exception):
    """Base class for errors raised by this module."""


class LLMInitialisationError(LLMError):
    """The client could not be constructed — bad or missing credentials, bad base URL.

    Fatal: retrying without fixing the configuration will not help.
    """


class LLMRequestError(LLMError):
    """A single chat request failed — timeout, rate limit, server error, etc.

    Often transient: the caller may retry or carry on.
    """


class LLMClient:
    """Stateless OpenAI chat client.

    Credentials come from the environment (``OPENAI_API_KEY``, optionally
    ``OPENAI_BASE_URL``). The model comes from ``MODEL_NAME``, falling back to
    :data:`DEFAULT_MODEL`.
    """

    def __init__(self) -> None:
        self.model = os.environ.get("MODEL_NAME") or DEFAULT_MODEL
        try:
            self._client = OpenAI()
        except OpenAIError as exc:
            raise LLMInitialisationError(f"could not initialise OpenAI client: {exc}") from exc

    def send(self, message: str) -> Iterator[str]:
        """Send ``message`` as a one-shot prompt, yielding reply text as it streams in.

        Raises :class:`LLMRequestError` if the request fails — which, because the
        response is streamed, may happen after some text has already been yielded.
        """
        try:
            stream = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": message},
                ],
                max_completion_tokens=MAX_COMPLETION_TOKENS,
                stream=True,
            )
            for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    yield content
        except OpenAIError as exc:
            raise LLMRequestError(f"request failed: {exc}") from exc
