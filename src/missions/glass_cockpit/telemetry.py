"""Telemetry data structures and utilities for LLM usage and cost tracking."""

from pydantic import BaseModel, ConfigDict, Field, computed_field

# Pricing per 1,000,000 tokens (USD), as (prompt_rate, completion_rate).
#
# https://developers.openai.com/api/docs/pricing
MODEL_PRICING: dict[str, tuple[float, float]] = {
    "gpt-5.4-nano": (0.20, 1.25),
    "gpt-5.4-mini": (0.75, 4.50),
    "gpt-5.4": (2.50, 15.00),
    "gpt-4o-mini": (0.15, 0.60),
}


class LLMMetrics(BaseModel):
    """Encapsulates latency, token usage, cost for an LLM call."""

    model_config = ConfigDict(extra="forbid")

    model_name: str
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    latency_ms: int = Field(default=0, ge=0)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def cost_usd(self) -> float:
        """Calculate estimated cost in USD based on token counts and model pricing."""
        prompt_rate, completion_rate = MODEL_PRICING.get(self.model_name, (0.0, 0.0))
        prompt_cost = (self.prompt_tokens / 1_000_000) * prompt_rate
        completion_cost = (self.completion_tokens / 1_000_000) * completion_rate
        return round(prompt_cost + completion_cost, 6)

    def __str__(self) -> str:
        return (
            f"[stats] prompt={self.prompt_tokens} completion={self.completion_tokens} "
            f"cost=${self.cost_usd:.6f} latency={self.latency_ms} ms model={self.model_name}"
        )
