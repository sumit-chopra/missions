"""Thin wrapper around the OpenAI chat completions API.

Each call is independent — no conversation history is kept.
"""

import os

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

    def send(self, message: str) -> str:
        """Send ``message`` as a one-shot prompt and return the assistant reply.

        Raises :class:`LLMRequestError` if the request fails.
        """
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": message},
                ],
                max_completion_tokens=MAX_COMPLETION_TOKENS,
            )
        except OpenAIError as exc:
            raise LLMRequestError(f"request failed: {exc}") from exc

        reply = response.choices[0].message.content or ""
        return reply
