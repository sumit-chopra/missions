"""Tests for missions/co_pilot/agent.py — schema, trace hooks, and the run wrapper."""

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from agents.exceptions import AgentsException
from structlog.testing import capture_logs

from missions.co_pilot import agent as agent_module
from missions.co_pilot.agent import (
    AgentResult,
    OpsCopilotAgent,
    PlanStep,
)


def run(coro):
    return asyncio.run(coro)


@pytest.fixture
def runner(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    mock = MagicMock(name="Runner")
    monkeypatch.setattr(agent_module, "Runner", mock)
    return mock


def test_run_returns_the_validated_final_output(runner: MagicMock):
    final = AgentResult(outcome="refusal", summary="No application A-9999 exists.")
    runner.run_sync.return_value = SimpleNamespace(final_output=final)

    with capture_logs():
        result = OpsCopilotAgent().run("plan for #A-9999")

    assert result is final
    runner.run_sync.assert_called_once()
    _, kwargs = runner.run_sync.call_args
    assert kwargs["max_turns"] == agent_module.MAX_TURNS


def test_run_returns_none_and_logs_when_the_agent_raises(runner: MagicMock):
    runner.run_sync.side_effect = AgentsException("model exploded")

    with capture_logs() as logs:
        result = OpsCopilotAgent().run("plan for #A-1423")

    assert result is None
    assert any(e["event"] == "co_pilot.error" for e in logs)


def test_run_logs_start_and_final_events(runner: MagicMock):
    runner.run_sync.return_value = SimpleNamespace(
        final_output=AgentResult(
            outcome="plan", summary="s", steps=[PlanStep(order=1, action="a", rationale="r")]
        )
    )

    with capture_logs() as logs:
        OpsCopilotAgent().run("do the thing")

    events = [e["event"] for e in logs]
    assert events[0] == "co_pilot.start"
    assert "co_pilot.final" in events
    assert logs[0]["request"] == "do the thing"
