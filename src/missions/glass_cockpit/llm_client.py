"""Thin wrapper around the OpenAI chat completions API.

Each call is independent — no conversation history is kept.
"""

import os
import time
from collections.abc import Iterator

from openai import OpenAI, OpenAIError
from openai.types import CompletionUsage

from missions.glass_cockpit.telemetry import LLMMetrics

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
        self.last_metrics: LLMMetrics | None = None
        try:
            self._client = OpenAI()
        except OpenAIError as exc:
            raise LLMInitialisationError(f"could not initialise OpenAI client: {exc}") from exc

    def send(self, message: str) -> Iterator[str]:
        """Send ``message`` as a one-shot prompt, yielding reply text as it streams in.

        On success, :attr:`last_metrics` is set to the :class:`LLMMetrics` for the
        call. Raises :class:`LLMRequestError` if the request fails — which, because
        the response is streamed, may happen after some text has already been yielded.
        """
        self.last_metrics = None
        start = time.perf_counter()
        usage = None
        try:
            stream = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": message},
                ],
                max_completion_tokens=MAX_COMPLETION_TOKENS,
                stream=True,
                stream_options={"include_usage": True},
            )
            for chunk in stream:
                if chunk.usage is not None:
                    usage = chunk.usage
                if chunk.choices and (content := chunk.choices[0].delta.content):
                    yield content
        except OpenAIError as exc:
            raise LLMRequestError(f"request failed: {exc}") from exc

        if usage:
            self.last_metrics = self.construct_metrics(usage, start, time.perf_counter())

    def construct_metrics(self, usage: CompletionUsage, start_time, end_time):
        return LLMMetrics(
            model_name=self.model,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            latency_ms=round((end_time - start_time) * 1000),
        )
