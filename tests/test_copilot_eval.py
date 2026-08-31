"""Tests for the co-pilot scenario eval — the pure deterministic scoring.

The agent run and the LLM judge are not exercised here (they need the network);
``make eval-copilot`` covers the end-to-end path.
"""

from datetime import datetime

import pytest
from missions.co_pilot.eval.run import (
    RunTrace,
    ScenarioResult,
    check_scenario,
    load_scenarios,
)

from missions.co_pilot.agent import AgentResult, PlanStep
from missions.co_pilot.eval import run as eval_run

SLOT = datetime(2026, 4, 22, 9, 0)


def _plan(**over) -> AgentResult:
    base = dict(
        outcome="plan",
        application_id="A-1423",
        summary="Day 4 of a 3-business-day verification SLA; nudge before day-5 escalation.",
        steps=[
            PlanStep(order=1, action="SMS the customer", channel="sms", due=SLOT, rationale="r"),
        ],
    )
    base.update(over)
    return AgentResult(**base)


def _trace(**over) -> RunTrace:
    base = dict(
        tools_invoked=["crm_lookup", "policy_lookup", "calendar_next_slot"],
        calendar_slots={SLOT},
    )
    base.update(over)
    return RunTrace(**base)


# --- check_scenario ---------------------------------------------------------
def test_a_well_formed_plan_passes_every_check():
    expect = {
        "outcome": "plan",
        "tools_required": ["crm_lookup", "policy_lookup", "calendar_next_slot"],
        "min_steps": 1,
        "dated_steps_from_calendar": True,
    }
    assert check_scenario(expect, _plan(), _trace()) == []


def test_wrong_outcome_is_flagged():
    fails = check_scenario({"outcome": "refusal"}, _plan(), _trace())
    assert any("outcome" in f for f in fails)


def test_a_refusal_with_steps_is_flagged():
    result = _plan(outcome="refusal")
    fails = check_scenario({"outcome": "refusal"}, result, _trace())
    assert any("carries 1 steps" in f for f in fails)


def test_missing_required_tool_is_flagged():
    trace = _trace(tools_invoked=["crm_lookup"])
    fails = check_scenario({"outcome": "plan", "tools_required": ["policy_lookup"]}, _plan(), trace)
    assert any("policy_lookup" in f and "not invoked" in f for f in fails)


def test_forbidden_tool_that_ran_is_flagged():
    expect = {"outcome": "refusal", "tools_forbidden": ["calendar_next_slot"]}
    result = AgentResult(outcome="refusal", summary="No such application.")
    fails = check_scenario(expect, result, _trace())
    assert any("calendar_next_slot" in f and "should not" in f for f in fails)


def test_too_few_steps_is_flagged():
    fails = check_scenario({"outcome": "plan", "min_steps": 3}, _plan(), _trace())
    assert any(">= 3" in f for f in fails)


def test_forbidden_channel_is_flagged():
    expect = {"outcome": "plan", "forbidden_channels": ["sms", "call"]}
    fails = check_scenario(expect, _plan(), _trace())
    assert any("forbidden channel" in f and "sms" in f for f in fails)


def test_dated_step_not_from_calendar_is_flagged():
    stray = _plan(
        steps=[PlanStep(order=1, action="a", due=datetime(2026, 4, 30, 9, 0), rationale="r")]
    )
    fails = check_scenario({"outcome": "plan", "dated_steps_from_calendar": True}, stray, _trace())
    assert any("not a calendar_next_slot result" in f for f in fails)


def test_dated_steps_required_but_none_present_is_flagged():
    undated = _plan(steps=[PlanStep(order=1, action="a", rationale="r")])
    fails = check_scenario(
        {"outcome": "plan", "dated_steps_from_calendar": True}, undated, _trace()
    )
    assert any("no dated steps" in f for f in fails)


def test_a_none_result_is_a_single_failure():
    assert check_scenario({"outcome": "plan"}, None, _trace()) == ["agent produced no result"]


# --- ScenarioResult.passed ------------------------------------------------
@pytest.mark.parametrize(
    ("det", "judge", "expected"),
    [
        ([], True, True),
        ([], None, True),
        ([], False, False),
        (["boom"], True, False),
    ],
)
def test_passed_combines_deterministic_and_judge(det, judge, expected):
    r = ScenarioResult("x", "plan", "plan", 1, det, judge, "")
    assert r.passed is expected


# --- scenarios.json -------------------------------------------------------
def test_every_scenario_record_is_well_formed():
    scenarios = load_scenarios(eval_run.SCENARIOS_PATH)
    assert scenarios
    ids = [s["id"] for s in scenarios]
    assert len(ids) == len(set(ids)), "scenario ids must be unique"
    for s in scenarios:
        assert s["request"].strip()
        assert s["expect"]["outcome"] in {"plan", "refusal"}


def test_load_scenarios_honours_limit():
    assert len(load_scenarios(eval_run.SCENARIOS_PATH, limit=2)) == 2
