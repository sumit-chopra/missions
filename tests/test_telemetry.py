"""Tests for LLMMetrics — the cost/latency gauges printed after every turn."""

import json

import pytest
from pydantic import ValidationError

from missions.glass_cockpit.telemetry import MODEL_PRICING, LLMMetrics


def test_cost_usd_is_zero_for_an_unknown_model():
    m = LLMMetrics(model_name="mystery-9000", prompt_tokens=1000, completion_tokens=1000)
    assert m.cost_usd == 0.0


def test_cost_usd_is_rounded_to_six_decimal_places():
    m = LLMMetrics(model_name="gpt-5.4", prompt_tokens=1, completion_tokens=1)
    assert m.cost_usd == round(m.cost_usd, 6)


@pytest.mark.parametrize("model_name", list(MODEL_PRICING))
def test_every_priced_model_produces_a_positive_cost(model_name: str):
    m = LLMMetrics(model_name=model_name, prompt_tokens=1000, completion_tokens=1000)
    assert m.cost_usd > 0.0


def test_str_matches_the_brief_stats_line_shape():
    m = LLMMetrics(model_name="gpt-5.4-nano", prompt_tokens=8, completion_tokens=23, latency_ms=623)
    line = str(m)

    # e.g. "[stats] prompt=8 completion=23 cost=$0.000030 latency=623 ms model=gpt-5.4-nano"
    assert "[stats]" in line
    assert "prompt=8" in line
    assert "completion=23" in line
    assert f"cost=${m.cost_usd:.6f}" in line
    assert "latency=623 ms" in line
    assert "model=gpt-5.4-nano" in line


def test_model_dump_json_carries_every_gauge_including_cost():
    m = LLMMetrics(model_name="gpt-5.4-nano", prompt_tokens=8, completion_tokens=23, latency_ms=623)
    assert json.loads(m.model_dump_json()) == {
        "model_name": "gpt-5.4-nano",
        "prompt_tokens": 8,
        "completion_tokens": 23,
        "latency_ms": 623,
        "cost_usd": m.cost_usd,
    }


def test_negative_token_counts_are_rejected():
    with pytest.raises(ValidationError):
        LLMMetrics(model_name="gpt-5.4-nano", prompt_tokens=-1)
