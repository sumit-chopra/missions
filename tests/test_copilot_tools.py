"""Tests for the pre-built Mission 3 mock tools wired into the Ops Co-pilot."""

from datetime import datetime

import pytest

from missions.co_pilot.tools import calendar_tool, policy
from missions.co_pilot.tools.crm import (
    ApplicationStatus,
    ContactChannel,
    CrmSnapshot,
    get_application,
    get_application_details,
)
from missions.the_vault.rag import Citation, CitedAnswer


def test_get_application_details_returns_a_snapshot_for_a_known_id():
    snap = get_application_details("A-1423")

    assert isinstance(snap, CrmSnapshot)
    assert snap.application_id == "A-1423"
    assert snap.status is ApplicationStatus.verification
    assert snap.days_in_current_status == 4
    assert snap.contact_preferences == [
        ContactChannel.sms,
        ContactChannel.email,
        ContactChannel.call,
    ]


def test_get_application_details_returns_none_for_the_unsolvable_id():
    assert get_application_details("A-9999") is None
    assert get_application("A-9999") is None


@pytest.mark.parametrize(
    ("business_days", "expected"),
    [
        (0, datetime(2026, 4, 21, 9, 0)),  # FIXED_NOW is a Tuesday
        (1, datetime(2026, 4, 22, 9, 0)),
        (3, datetime(2026, 4, 24, 9, 0)),  # Friday
        (4, datetime(2026, 4, 28, 9, 0)),  # skips Sat/Sun + Mon 27 (ANZAC observed)
    ],
)
def test_get_next_slot_counts_business_days_skipping_weekends_and_holidays(
    business_days: int, expected: datetime
):
    assert calendar_tool.get_next_slot(business_days) == expected


def test_get_next_slot_always_lands_at_0900():
    for n in range(0, 8):
        assert calendar_tool.get_next_slot(n).time() == datetime(2026, 1, 1, 9, 0).time()


def test_get_next_slot_rejects_a_negative_horizon():
    with pytest.raises(ValueError):
        calendar_tool.get_next_slot(-1)


@pytest.fixture
def fake_rag(monkeypatch: pytest.MonkeyPatch):
    from unittest.mock import MagicMock

    rag = MagicMock(name="rag")
    rag_cls = MagicMock(name="Rag")
    rag_cls.bootstrap.return_value = rag
    monkeypatch.setattr(policy, "Rag", rag_cls)
    policy._rag.cache_clear()
    yield rag
    policy._rag.cache_clear()


def test_lookup_policy_delegates_to_a_bootstrapped_rag(fake_rag):
    answer = CitedAnswer(
        answer="Escalate for manual review after 3 business days. [1]",
        citations=[Citation(source_file="acme_sla_handbook.md", section="2. SLAs", chunk=0)],
    )
    fake_rag.answer_question.return_value = answer

    result = policy.lookup_policy("What is the verification SLA?")

    assert result is answer
    fake_rag.answer_question.assert_called_once_with("What is the verification SLA?")


def test_lookup_policy_reuses_a_single_cached_rag_handle(fake_rag):
    fake_rag.answer_question.return_value = CitedAnswer(answer=None, citations=[])

    policy.lookup_policy("q1")
    policy.lookup_policy("q2")

    # bootstrap paid once; both questions hit the same handle
    assert policy.Rag.bootstrap.call_count == 1
    assert fake_rag.answer_question.call_count == 2
