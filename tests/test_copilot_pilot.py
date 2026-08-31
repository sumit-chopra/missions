"""Tests for missions/co_pilot/pilot.py — the CLI entrypoint and its exit codes."""

import json

import pytest

from missions.co_pilot import pilot
from missions.co_pilot.agent import AgentResult, PlanStep


@pytest.fixture
def agent_cls(monkeypatch: pytest.MonkeyPatch):
    """Patch OpsCopilotAgent in pilot; configure `.run` per test."""
    from unittest.mock import MagicMock

    instance = MagicMock(name="agent")
    cls = MagicMock(name="OpsCopilotAgent", return_value=instance)
    monkeypatch.setattr(pilot, "OpsCopilotAgent", cls)
    monkeypatch.setattr(pilot, "load_dotenv", lambda *a, **k: None)
    return instance


def _plan() -> AgentResult:
    return AgentResult(
        outcome="plan",
        application_id="A-1423",
        summary="Nudge the customer before day-5 escalation.",
        steps=[PlanStep(order=1, action="SMS the customer", channel="sms", rationale="SLA §2")],
    )


def _refusal() -> AgentResult:
    return AgentResult(
        outcome="refusal", application_id="A-9999", summary="No application A-9999 exists."
    )


def test_a_produced_plan_prints_json_to_stdout_and_exits_0(agent_cls, capsys):
    agent_cls.run.return_value = _plan()

    code = pilot.main(["Draft a follow-up plan for application #A-1423"])

    assert code == pilot.EXIT_PLAN
    payload = json.loads(capsys.readouterr().out)
    assert payload["outcome"] == "plan"
    assert payload["steps"][0]["action"] == "SMS the customer"


def test_a_clean_refusal_prints_json_and_exits_3(agent_cls, capsys):
    agent_cls.run.return_value = _refusal()

    code = pilot.main(["Draft a follow-up plan for application #A-9999"])

    assert code == pilot.EXIT_REFUSAL
    assert json.loads(capsys.readouterr().out)["outcome"] == "refusal"


def test_empty_request_prints_usage_and_exits_1(agent_cls, capsys):
    code = pilot.main([])

    assert code == pilot.EXIT_ERROR
    assert "usage:" in capsys.readouterr().err
    agent_cls.run.assert_not_called()


def test_whitespace_only_request_is_treated_as_empty(agent_cls):
    assert pilot.main(["   "]) == pilot.EXIT_ERROR
    agent_cls.run.assert_not_called()


def test_argv_words_are_joined_into_one_request(agent_cls):
    agent_cls.run.return_value = _plan()

    pilot.main(["follow", "up", "on", "#A-1423"])

    agent_cls.run.assert_called_once_with("follow up on #A-1423")


def test_a_none_result_from_the_agent_exits_1(agent_cls, capsys):
    agent_cls.run.return_value = None

    code = pilot.main(["anything"])

    assert code == pilot.EXIT_ERROR
    assert capsys.readouterr().out == ""


def test_a_plan_outcome_with_no_steps_is_an_error_but_still_prints(agent_cls, capsys):
    agent_cls.run.return_value = AgentResult(outcome="plan", summary="s", steps=[])

    code = pilot.main(["anything"])

    assert code == pilot.EXIT_ERROR
    assert json.loads(capsys.readouterr().out)["outcome"] == "plan"


def test_an_exception_from_the_agent_is_caught_and_exits_1(agent_cls, capsys):
    agent_cls.run.side_effect = RuntimeError("missing OPENAI_API_KEY")

    code = pilot.main(["anything"])

    assert code == pilot.EXIT_ERROR
