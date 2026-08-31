"""Tests for missions/pii_mask.py — the PIIMasker structlog processor."""

import pytest

from missions.pii_mask import DEFAULT_SENSITIVE_KEYS, PIIMasker


@pytest.fixture
def mask() -> PIIMasker:
    return PIIMasker()


def process(masker: PIIMasker, event_dict: dict) -> dict:
    """Run the processor the way structlog would, on a fresh copy."""
    return masker(None, "info", dict(event_dict))


@pytest.mark.parametrize(
    ("prefix", "pii", "suffix"),
    [
        ("reach the customer at ", "john.doe+acme@example.com", " today"),
        ("card ", "4111 1111 1111 1111", " declined"),
        ("card ", "4111-1111-1111-1111", " declined"),
        ("dob ", "1988-05-03", " recorded"),
        ("dob ", "05/03/1988", " recorded"),
        ("account ", "12345678", " credited"),
        ("routing ", "021000021", " verified"),
    ],
)
def test_known_pii_patterns_are_redacted_but_surrounding_text_survives(
    mask: PIIMasker, prefix: str, pii: str, suffix: str
):
    out = process(mask, {"event": f"{prefix}{pii}{suffix}"})

    assert pii not in out["event"]
    assert out["event"] == f"{prefix}[REDACTED]{suffix}"


def test_multiple_matches_in_one_string_are_all_masked(mask: PIIMasker):
    out = process(mask, {"event": "email a@b.com or a2@b.com"})

    assert out["event"] == "email [REDACTED] or [REDACTED]"


def test_plain_text_without_pii_is_left_untouched(mask: PIIMasker):
    out = process(mask, {"event": "application A-1423 has been in verification 4 days"})

    assert out["event"] == "application A-1423 has been in verification 4 days"


@pytest.mark.parametrize("key", sorted(DEFAULT_SENSITIVE_KEYS))
def test_every_default_sensitive_key_has_its_value_replaced_wholesale(mask: PIIMasker, key: str):
    out = process(mask, {key: "whatever-the-value-is"})

    assert out[key] == "[REDACTED]"


def test_sensitive_key_match_is_case_insensitive(mask: PIIMasker):
    out = process(mask, {"Password": "hunter2", "API_KEY": "sk-123"})

    assert out == {"Password": "[REDACTED]", "API_KEY": "[REDACTED]"}


def test_sensitive_key_masks_non_string_values_too(mask: PIIMasker):
    out = process(mask, {"account_number": 12345678, "token": ["a", "b"]})

    assert out == {"account_number": "[REDACTED]", "token": "[REDACTED]"}


def test_non_sensitive_key_with_a_non_string_value_is_untouched(mask: PIIMasker):
    out = process(mask, {"latency_ms": 623, "count": 123456789})

    assert out == {"latency_ms": 623, "count": 123456789}


def test_sensitive_keys_are_masked_inside_nested_dicts(mask: PIIMasker):
    out = process(mask, {"payload": {"user": {"password": "p", "id": 7}}})

    assert out["payload"] == {"user": {"password": "[REDACTED]", "id": 7}}


def test_patterns_are_swept_inside_nested_dict_and_list_values(mask: PIIMasker):
    out = process(
        mask,
        {"ctx": {"emails": ["x@y.com", "z@y.com"], "note": "dob 1988-05-03"}},
    )

    assert out["ctx"] == {
        "emails": ["[REDACTED]", "[REDACTED]"],
        "note": "dob [REDACTED]",
    }


def test_list_of_dicts_under_a_safe_key_is_recursed(mask: PIIMasker):
    out = process(mask, {"steps": [{"secret": "s"}, {"action": "call a@b.com"}]})

    assert out["steps"] == [{"secret": "[REDACTED]"}, {"action": "call [REDACTED]"}]
