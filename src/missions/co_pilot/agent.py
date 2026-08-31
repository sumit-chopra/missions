"""The Ops Co-pilot agent"""

import json
from datetime import datetime
from typing import Any, Literal

import structlog
from agents import (
    Agent,
    AgentOutputSchema,
    ModelSettings,
    RunHooks,
    Runner,
    function_tool,
    set_tracing_disabled,
)
from agents.exceptions import AgentsException, MaxTurnsExceeded
from openai.types.shared.reasoning import Reasoning
from pydantic import BaseModel, ConfigDict, Field

from missions.co_pilot.tools import (
    CrmSnapshot,
    calendar_tool,
    get_application_details,
    lookup_policy,
)
from missions.the_vault.rag import CitedAnswer

log = structlog.get_logger("co_pilot")

DEFAULT_MODEL = "gpt-5.4"
MAX_TURNS = 24

SYSTEM_PROMPT = """\
You are the Ops Co-pilot for Acme, a personal-loan lender. Given one operator \
request about a loan application, you produce a short, policy-grounded follow-up \
action plan — or a clean refusal when no honest plan is possible.

How to work:
1. Always start with crm_lookup. If the application is not found, refuse.
   Otherwise set application_id to the id you looked up.
2. Call policy_lookup for the rules that govern the current status — the
   verification SLA and escalation timing, or the hardship communication /
   escalation ladder. Ground the plan in what it returns; do not rely on
   memorised numbers. Name the rule you relied on (its source_file + section)
   in the rationale of the step it justifies.
3. Honour explicit constraints in the request — budget, deadline, contact
   channel. Also honour implicit ones: use the customer's contact_preferences
   order from the CRM unless the request overrides it; never contact outside
   business hours. State the constraints you applied in the summary.
4. Schedule every dated step with calendar_next_slot and put the slot it
   returns in that step's due field.
5. Finish by producing the structured output (one AgentResult):
   - outcome "plan": a one/two-sentence summary covering the current status,
     where it sits against the SLA, and whether it needs escalation, plus an
     ordered, non-empty list of steps.
   - outcome "refusal": leave steps empty and give the plain-language reason,
     safe for an operator to read, in summary.
   Do not call a tool to finish, and never reply with prose.

Refuse (do not invent a plan) when:
- the application id is unknown;
- a follow-up deadline in the request has already passed, or leaves no business
  day to act;
- policy forbids the requested action;
- the application is already settled / declined and needs no action;
- you lack the information to proceed.
Say which of these applies, in plain language, in summary.
"""

Channel = Literal["sms", "email", "call", "internal_note", "escalation"]

Outcome = Literal["plan", "refusal"]


class PlanStep(BaseModel):
    """One concrete, ordered action in the follow-up plan."""

    model_config = ConfigDict(extra="forbid")

    order: int = Field(ge=1, description="1-based position of this step in the plan.")
    action: str = Field(description="What the operator should do, in the imperative.")
    channel: Channel | None = Field(
        default=None,
        description="Contact channel or internal action type, when applicable.",
    )
    due: datetime | None = Field(
        default=None,
        description="When this step is due — always a slot returned by the calendar tool.",
    )
    rationale: str = Field(description="Why this step, grounded in status + policy.")


class AgentResult(BaseModel):
    """The single structured shape the agent emits — a follow-up plan or a refusal."""

    model_config = ConfigDict(extra="forbid")

    outcome: Outcome = Field(description="'plan' for an action plan, 'refusal' to decline.")

    application_id: str | None = None
    summary: str = Field(
        description=(
            "Plan: the situation and the recommendation in one or two sentences. "
            "Refusal: the plain-language reason, safe for an operator to read."
        )
    )
    steps: list[PlanStep] = Field(
        default_factory=list, description="[plan] ordered actions; at least one."
    )


def _tool_failed(ctx: Any, error: Exception) -> str:
    """Surface a tool exception the Agents SDK would otherwise swallow.

    Without this, a raising tool is invisible: the SDK never fires
    ``on_tool_end``, feeds a generic "an error occurred" string to the model,
    and the run silently burns turns retrying until ``MAX_TURNS`` — which reads
    as the agent being "stuck". Log the real cause, then hand the model a short,
    honest message.
    """
    log.error("co_pilot.tool_error", error=str(error), error_type=type(error).__name__)
    return f"The tool failed and cannot be retried: {error}. Refuse and explain this in summary."


@function_tool(failure_error_function=_tool_failed)
def crm_lookup(application_id: str) -> CrmSnapshot | None:
    """Look up a loan application by id.

    Returns a CrmSnapshot if found, or None if the id does not exist in the CRM
    (refuse then, giving the reason in ``summary``). Customer name, email,
    account number and customer id are not returned — they don't affect the plan.
    """
    return get_application_details(application_id)


@function_tool(failure_error_function=_tool_failed)
def policy_lookup(query: str) -> CitedAnswer:
    """Answer a natural-language Acme policy / SLA question.

    Backed by retrieval service (RAG over the Acme policy corpus).
    Returns a ``CitedAnswer``: a grounded ``answer`` (null when the corpus has
    nothing on the query) plus ``citations`` (each ``source_file`` + ``section``
    + ``chunk``) — name those in the ``rationale`` of the step they justify.
    Covers the verification SLA, contact preferences, hardship escalation,
    business hours, cooling-off and fees. Ask a focused question, e.g. "What is
    the verification SLA and when is an application escalated for manual review?".
    """
    return lookup_policy(query)


@function_tool(failure_error_function=_tool_failed)
def calendar_next_slot(business_days_from_now: int) -> datetime:
    """Find the next available follow-up slot at least N business days from now.

    Weekends and NSW public holidays are automatically skipped. Every dated step
    in a plan MUST obtain its timestamp from this tool. ``business_days_from_now``
    must be >= 0; a negative value means the deadline has already passed — do not
    call the tool in that case, refuse and say so in ``summary``.
    """
    return calendar_tool.get_next_slot(business_days_from_now)


TOOLS = [crm_lookup, policy_lookup, calendar_next_slot]


# --- scratch-pad logging --------------------------------------------------
class _TraceHooks(RunHooks[Any]):
    """Emit the agent's reasoning trace as structured ``co_pilot.*`` log events."""

    async def on_agent_start(self, context: Any, agent: Any) -> None:
        log.info("co_pilot.agent_start", agent=agent.name)

    async def on_llm_end(self, context: Any, agent: Any, response: Any) -> None:
        for item in getattr(response, "output", []) or []:
            kind = getattr(item, "type", "")
            if kind == "function_call":
                log.info(
                    "co_pilot.tool_call",
                    tool=getattr(item, "name", "?"),
                    args=_parse_args(getattr(item, "arguments", "")),
                )
            elif kind == "reasoning":
                summary = "\n".join(
                    getattr(part, "text", "") for part in getattr(item, "summary", []) or []
                ).strip()
                if summary:
                    log.info("co_pilot.reasoning", text=summary)

    async def on_tool_end(self, context: Any, agent: Any, tool: Any, result: Any) -> None:
        log.info("co_pilot.tool_result", tool=tool.name, result=result)


def _parse_args(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


class OpsCopilotAgent:
    """Runs the plan/refuse workflow for a single request via the Agents SDK."""

    def __init__(self) -> None:
        set_tracing_disabled(True)
        self._agent: Agent[Any] = Agent(
            name="Ops Co-pilot",
            instructions=SYSTEM_PROMPT,
            model=DEFAULT_MODEL,
            model_settings=ModelSettings(reasoning=Reasoning(effort="medium", summary="detailed")),
            tools=TOOLS,
            output_type=AgentOutputSchema(AgentResult, strict_json_schema=True),
        )

    def run(self, request: str, *, hooks: RunHooks[Any] | None = None) -> AgentResult | None:
        """Drive the agent and return the validated result.

        ``hooks`` defaults to the scratch-pad tracer; pass a custom
        :class:`RunHooks` (e.g. the eval's recording hook) to observe the run.
        """
        log.info("co_pilot.start", request=request)

        hooks = hooks or _TraceHooks()
        try:
            result = Runner.run_sync(self._agent, request, hooks=hooks, max_turns=MAX_TURNS)
        except MaxTurnsExceeded as exc:
            # Usually the agent looping on a failing tool — see co_pilot.tool_error above.
            log.error("co_pilot.error", error=str(exc), hint="turn cap hit without finishing")
            return None
        except AgentsException as exc:
            log.error("co_pilot.error", error=str(exc))
            return None
        except Exception as exc:  # network / auth / config — keep it out of the caller's face
            log.error("co_pilot.error", error=str(exc), error_type=type(exc).__name__)
            return None

        final = result.final_output

        log.info("co_pilot.final", outcome=final.outcome)
        return final
