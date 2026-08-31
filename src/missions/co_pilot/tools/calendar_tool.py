"""Mock calendar tool. Returns the next available follow-up slot,
respecting weekends and NSW public holidays.

Deterministic: 'now' is fixed to 2026-04-21 09:00 AEST so that agent
runs are reproducible across machines and timezones.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta


# Fixed reference "now" for deterministic tests.
FIXED_NOW = datetime(2026, 4, 21, 9, 0)  # Tuesday 21 April 2026, 09:00 AEST

# NSW public holidays for 2026 that fall within the likely test window.
# (Not exhaustive — enough for the assessment.)
_NSW_PUBLIC_HOLIDAYS_2026: set[date] = {
    date(2026, 1, 1),   # New Year's Day
    date(2026, 1, 26),  # Australia Day
    date(2026, 4, 3),   # Good Friday
    date(2026, 4, 4),   # Easter Saturday
    date(2026, 4, 5),   # Easter Sunday
    date(2026, 4, 6),   # Easter Monday
    date(2026, 4, 25),  # ANZAC Day (falls Sat in 2026; observed Mon 27 Apr)
    date(2026, 4, 27),  # ANZAC Day observed
    date(2026, 6, 8),   # King's Birthday
    date(2026, 10, 5),  # Labour Day
    date(2026, 12, 25), # Christmas Day
    date(2026, 12, 28), # Boxing Day observed
}


def _is_business_day(d: date) -> bool:
    return d.weekday() < 5 and d not in _NSW_PUBLIC_HOLIDAYS_2026


def get_next_slot(business_days_from_now: int) -> datetime:
    """Return the next available follow-up slot at least N business days from now.

    The slot is returned at 09:00 AEST on the target business day, which
    falls within Acme Operations' business hours (08:00–18:00 AEST,
    per SLA Handbook §1).

    Args:
        business_days_from_now: minimum number of business days forward.
            0 means "today if it's a business day, else the next one".
            Must be >= 0.

    Returns:
        A datetime at 09:00 on the target business day.

    Raises:
        ValueError: if business_days_from_now is negative.
    """
    if business_days_from_now < 0:
        raise ValueError("business_days_from_now must be >= 0")

    d = FIXED_NOW.date()
    remaining = business_days_from_now
    # Advance until we've consumed the required number of business days.
    while True:
        if _is_business_day(d):
            if remaining == 0:
                break
            remaining -= 1
        d = d + timedelta(days=1)

    return datetime.combine(d, time(9, 0))
