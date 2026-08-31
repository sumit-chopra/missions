"""Pre-built mock tools for Mission 3 of the Plenti AI Engineer Field Test.

These tools are intentionally simple. Do not modify them. Wire them into
your agent as-is.

    from tools import get_application, lookup_policy, get_next_slot

All three tools are deterministic: the same input always returns the same
output.
"""

from .crm import (
    Application,
    ApplicationStatus,
    ContactChannel,
    CrmSnapshot,
    get_application,
    get_application_details,
)
from .policy import lookup_policy
from .calendar_tool import get_next_slot

__all__ = [
    "Application",
    "ApplicationStatus",
    "ContactChannel",
    "CrmSnapshot",
    "get_application",
    "get_application_details",
    "lookup_policy",
    "get_next_slot",
]
