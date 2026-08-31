"""Mock CRM lookup. Returns application status + masked customer profile.

Returns None if the application does not exist — use this as the trigger
for the unsolvable-case test in Mission 3.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class ApplicationStatus(str, Enum):
    """Lifecycle status of a loan application."""

    submitted = "submitted"
    verification = "verification"
    approved = "approved"
    settled = "settled"
    hardship = "hardship"
    declined = "declined"


class ContactChannel(str, Enum):
    """A customer-contact channel, in the CRM's preference ordering."""

    sms = "sms"
    email = "email"
    call = "call"


@dataclass(frozen=True)
class Application:
    app_id: str
    status: str  # one of ApplicationStatus
    customer_id: str
    customer_name_masked: str
    customer_email_masked: str
    account_last4: str
    submitted_at: datetime
    last_contact_at: Optional[datetime]
    days_in_current_status: int
    contact_preferences: list[str]  # ordered, values from ContactChannel
    product: str
    notes: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["submitted_at"] = self.submitted_at.isoformat()
        d["last_contact_at"] = (
            self.last_contact_at.isoformat() if self.last_contact_at else None
        )
        return d


_APPLICATIONS: dict[str, Application] = {
    # Standard happy-path: verification stuck 4 days — this is the one from
    # the mission prompt.
    "A-1423": Application(
        app_id="A-1423",
        status="verification",
        customer_id="C-88291",
        customer_name_masked="John S****",
        customer_email_masked="j****@example.com",
        account_last4="1234",
        submitted_at=datetime(2026, 4, 15, 9, 22),
        last_contact_at=datetime(2026, 4, 17, 14, 0),
        days_in_current_status=4,
        contact_preferences=["sms", "email", "call"],
        product="personal_loan",
        notes="Awaiting payslip upload.",
    ),
    # Happy path: recently submitted, within SLA.
    "A-2001": Application(
        app_id="A-2001",
        status="verification",
        customer_id="C-90114",
        customer_name_masked="Priya K****",
        customer_email_masked="p****@example.com",
        account_last4="5521",
        submitted_at=datetime(2026, 4, 20, 11, 4),
        last_contact_at=datetime(2026, 4, 20, 11, 5),
        days_in_current_status=1,
        contact_preferences=["email", "sms"],
        product="personal_loan",
    ),
    # Hardship case: expects Hardship Policy escalation path.
    "A-3050": Application(
        app_id="A-3050",
        status="hardship",
        customer_id="C-71200",
        customer_name_masked="Marcus O****",
        customer_email_masked="m****@example.com",
        account_last4="8800",
        submitted_at=datetime(2025, 12, 10, 8, 30),
        last_contact_at=datetime(2026, 4, 14, 9, 15),
        days_in_current_status=7,
        contact_preferences=["sms", "call"],
        product="personal_loan",
        notes="Hardship notice received 2026-04-07; customer unresponsive since.",
    ),
    # Edge case: settled — nothing to action. Agent should say so clearly.
    "A-4100": Application(
        app_id="A-4100",
        status="settled",
        customer_id="C-55010",
        customer_name_masked="Aisha B****",
        customer_email_masked="a****@example.com",
        account_last4="3310",
        submitted_at=datetime(2025, 11, 1, 10, 0),
        last_contact_at=datetime(2025, 11, 4, 16, 22),
        days_in_current_status=170,
        contact_preferences=["email"],
        product="personal_loan",
        notes="Loan settled 2025-11-04. No action required.",
    ),
    # A-9999 is DELIBERATELY absent — this is the unsolvable case.
}


def get_application(app_id: str) -> Optional[Application]:
    """Return the Application record for app_id, or None if not found."""
    return _APPLICATIONS.get(app_id)


class CrmSnapshot(BaseModel):
    """What ``crm_lookup`` hands the model — only the fields needed to plan.

    Deliberately omits the customer's name, email, account number and internal
    customer id: none of them affect the follow-up plan, so they never enter the
    model context. Plain strings from :class:`Application` are coerced to the
    enums on construction.
    """

    model_config = ConfigDict(extra="forbid")

    application_id: str
    status: ApplicationStatus = Field(description="Lifecycle status of the application.")
    product: str
    days_in_current_status: int
    submitted_at: datetime
    last_contact_at: Optional[datetime]
    contact_preferences: list[ContactChannel] = Field(
        default_factory=list,
        description="Ordered channel preference, e.g. ['sms', 'email', 'call'].",
    )
    notes: str = Field(default="", description="Free-text operator notes on the file.")

def get_application_details(app_id: str) -> Optional[CrmSnapshot]:
    app = _APPLICATIONS.get(app_id)
    if app is None:
        return None

    return CrmSnapshot(
        application_id=app.app_id,
        status=app.status,
        product=app.product,
        days_in_current_status=app.days_in_current_status,
        submitted_at=app.submitted_at,
        last_contact_at=app.last_contact_at,
        contact_preferences=app.contact_preferences,
        notes=app.notes,
    )
    
        
