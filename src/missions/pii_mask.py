"""Shared structlog configuration for all missions with comprehensive PII masking."""

import re
from re import Pattern
from typing import Any

# Regex Patterns for Common PII Types
EMAIL_REGEX: Pattern[str] = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
CREDIT_CARD_REGEX: Pattern[str] = re.compile(r"\b(?:\d[ -]*?){13,16}\b")

# Matches dates formatted as YYYY-MM-DD, MM/DD/YYYY, or DD/MM/YYYY
DATE_OF_BIRTH_REGEX: Pattern[str] = re.compile(
    r"\b(?:19|20)\d{2}[-/](?:0[1-9]|1[0-2])[-/](?:0[1-9]|[12]\d|3[01])\b|"
    r"\b(?:0[1-9]|1[0-2])[-/](?:0[1-9]|[12]\d|3[01])[-/](?:19|20)\d{2}\b"
)

# Matches common international/US bank account numbers (8-17 digit sequences)
BANK_ACCOUNT_REGEX: Pattern[str] = re.compile(r"\b\d{8,17}\b")

REGEX_PATTERNS = [
    EMAIL_REGEX,
    CREDIT_CARD_REGEX,
    DATE_OF_BIRTH_REGEX,
    BANK_ACCOUNT_REGEX,
]

# Key names that should always have their values completely masked
DEFAULT_SENSITIVE_KEYS: set[str] = {
    # Auth & Secret Credentials
    "password",
    "token",
    "access_token",
    "secret",
    "api_key",
    "credential",
    "credentials",
    "apikey",
    # Financial Data
    "credit_card",
    "card_number",
    "pan",
    "cvv",
    "cvc",
    "bank_account",
    "account_number",
    "routing_number",
    "bsb",
    # Personal Identity Data
    "name",
    "firstname",
    "lastname",
    "first_name",
    "last_name",
    "fullname",
    "full_name",
    "dob",
    "date_of_birth",
    "birthdate",
    "medicare",
    "passport",
    "license",
}

MASK_STR = "[REDACTED]"


class PIIMasker:
    """Structlog processor that redacts PII patterns and sensitive dict keys."""

    def _mask_text_patterns(self, text: str) -> str:
        """Apply regex patterns against free-form text strings."""
        for sensitive_regex in REGEX_PATTERNS:
            text = sensitive_regex.sub(MASK_STR, text)
        return text

    def _mask_value(self, key: str, value: Any) -> Any:
        """Recursively scrub values based on key names, data types, and patterns."""
        # 1. Exact key match check
        if key.lower() in DEFAULT_SENSITIVE_KEYS:
            return MASK_STR

        # 2. String scrubbing
        if isinstance(value, str):
            return self._mask_text_patterns(value)

        # 3. Handle nested dictionaries
        if isinstance(value, dict):
            return {k: self._mask_value(k, v) for k, v in value.items()}

        # 4. Handle nested lists
        if isinstance(value, list):
            return [self._mask_value(key, item) for item in value]

        return value

    def __call__(self, logger: Any, method_name: str, event_dict: dict[str, Any]) -> dict[str, Any]:
        """Process event_dict and mask sensitive keys and text."""
        for key in list(event_dict.keys()):
            event_dict[key] = self._mask_value(key, event_dict[key])
        return event_dict
