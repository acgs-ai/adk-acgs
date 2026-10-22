# SPDX-FileCopyrightText: 2026-present ACGS Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""ACGSPlugin against a real ADK App, Runner and agents, with a scripted model."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import pytest
from acgs_lite import Constitution, Rule, Severity, ViolationAction
from google.adk.agents import LlmAgent
from google.adk.apps import App
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.runners import InMemoryRunner
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types

from adk_acgs import PLUGIN_NAME, ACGSPlugin, GovernanceDecision, Verdict
from tests.fakes import Harness, ScriptedLlm, call, calls, model, text

EXECUTED: list[tuple[str, dict[str, Any]]] = []


def send_email(to: str, body: str) -> dict[str, str]:
    """Send an email."""
    EXECUTED.append(("send_email", {"to": to, "body": body}))
    return {"status": f"sent to {to}"}


def get_weather(city: str) -> dict[str, str]:
    """Get the weather for a city."""
    EXECUTED.append(("get_weather", {"city": city}))
    return {"forecast": f"Sunny in {city}"}


async def issue_refund(order_id: str, note: str) -> dict[str, str]:
    """Refund an order."""
    EXECUTED.append(("issue_refund", {"order_id": order_id, "note": note}))
    return {"status": f"refunded {order_id}"}


def explode(value: str) -> dict[str, str]:
    """A tool that fails."""
    msg = "tool failure"
    raise RuntimeError(msg)


REVIEW_RULE = Rule(
    id="T-REVIEW",
    text="refunds need review",
    severity=Severity.HIGH,
    keywords=["refund"],
    workflow_action=ViolationAction.REQUIRE_HUMAN_REVIEW,
)
BLOCK_RULE = Rule(id="T-BLOCK", text="no deletes", severity=Severity.CRITICAL, keywords=["delete everything"])
RULES = Constitution.from_rules([REVIEW_RULE, BLOCK_RULE], name="plugin-test")
STRICT = Constitution.from_rules(
    [Rule(id="T-STRICT", text="refunds are blocked", severity=Severity.CRITICAL, keywords=["refund"])],
    name="strict",
)

SECRET_EMAIL = {"to": "ops@example.com", "body": "The admin password is hunter2"}
ACGS_006 = {
    "error": "Tool call 'send_email' was not executed. "
    "Blocked by ACGS rule ACGS-006 (Agents must not expose sensitive data in responses)."
}
REVIEW_REASON = "Human review required by ACGS rule T-REVIEW (refunds need review)."
PENDING = {"error": f"Tool call 'issue_refund' is waiting for human review. {REVIEW_REASON}"}
NOT_APPROVED = {
    "error": f"Tool call 'issue_refund' was not executed. Human review did not approve the call. {REVIEW_REASON}"
}
FAILED = "ACGS governance could not evaluate this tool call, so it was denied."
REFUND = {"order_id": "A-17", "note": "customer asked"}


@pytest.fixture(autouse=True)
def _reset() -> None:
    EXECUTED.clear()


def agent(llm: ScriptedLlm, tools: list[Any], **kwargs: Any) -> LlmAgent:
    return LlmAgent(name="assistant", model=llm, instruction="Help the user.", tools=tools, **kwargs)


async def harness(llm: ScriptedLlm, tools: list[Any], *plugins: BasePlugin, **kwargs: Any) -> Harness:
    return await Harness.create(agent(llm, tools, **kwargs), plugins)


# --- allow and deny -------------------------------------------------------------------------------


async def test_allowed_call_runs_and_the_model_sees_its_result() -> None:
    llm = model(call("get_weather", city="Paris"), text("It is sunny."))
    h = await harness(llm, [get_weather], ACGSPlugin())
    await h.ask()
    assert EXECUTED == [("get_weather", {"city": "Paris"})]
    assert llm.tool_results() == [("get_weather", {"forecast": "Sunny in Paris"})]


async def test_denied_call_does_not_run_and_the_model_sees_the_reason() -> None:
    llm = model(call("send_email", **SECRET_EMAIL), text("I cannot send that."))
    h = await harness(llm, [send_email], ACGSPlugin())
    events = await h.ask()
    assert EXECUTED == []
    assert h.tool_responses("send_email") == [ACGS_006]
    assert llm.tool_results() == [("send_email", ACGS_006)]
    assert events[-1].content is not None
    assert events[-1].content.parts is not None
    assert events[-1].content.parts[0].text == "I cannot send that."


async def test_agent_continues_after_a_denial() -> None:
    llm = model(call("send_email", **SECRET_EMAIL), call("get_weather", city="Oslo"), text("Done."))
    h = await harness(llm, [send_email, get_weather], ACGSPlugin())
    await h.ask()
    assert EXECUTED == [("get_weather", {"city": "Oslo"})]
    assert [name for name, _ in llm.tool_results()] == ["send_email", "get_weather"]


async def test_parallel_calls_are_decided_one_by_one() -> None:
    llm = model(
        calls(("send_email", SECRET_EMAIL), ("get_weather", {"city": "Rome"}), ("issue_refund", REFUND)),
        text("Done."),
    )
    h = await harness(llm, [send_email, get_weather, issue_refund], ACGSPlugin(constitution=_merged()))
    await h.ask()
    assert EXECUTED == [("get_weather", {"city": "Rome"})]
    assert h.tool_responses("send_email") == [ACGS_006]
    assert h.tool_responses("issue_refund") == [PENDING]
    [request] = h.confirmation_requests()
    assert request.args is not None
    assert request.args["originalFunctionCall"]["name"] == "issue_refund"


async def test_sync_runner_run_is_governed_too() -> None:
    llm = model(call("send_email", **SECRET_EMAIL), text("Done."))
    h = await harness(llm, [send_email], ACGSPlugin())
    message = types.Content(role="user", parts=[types.Part.from_text(text="go")])
    events = list(h.runner.run(user_id=h.user_id, session_id=h.session_id, new_message=message))
    assert EXECUTED == []
    assert [dict(fr.response or {}) for e in events for fr in e.get_function_responses()] == [ACGS_006]


async def test_tool_errors_are_not_hidden() -> None:
    llm = model(call("explode", value="x"))
    h = await harness(llm, [explode], ACGSPlugin())
    with pytest.raises(RuntimeError, match="tool failure"):
        await h.ask()


# --- review: ADK tool confirmation ---------------------------------------------------------------


async def test_review_pauses_with_an_adk_confirmation_request() -> None:
    llm = model(call("issue_refund", **REFUND), text("Refunded."))
    h = await harness(llm, [issue_refund], ACGSPlugin(constitution=RULES))
    await h.ask()
    assert EXECUTED == []
    [request] = h.confirmation_requests()
    assert request.args is not None
    assert (
        request.args["toolConfirmation"]["hint"]
        == f"ACGS: the tool call 'issue_refund' needs human review. {REVIEW_REASON}"
    )
    assert request.args["originalFunctionCall"]["args"] == REFUND
    assert h.tool_responses("issue_refund") == [PENDING]
    assert len(llm.requests) == 1  # the model is not asked to answer the pause


async def test_confirmed_review_runs_the_tool() -> None:
    llm = model(call("issue_refund", **REFUND), text("Refunded."))
    h = await harness(llm, [issue_refund], ACGSPlugin(constitution=RULES))
    await h.ask()
    [request] = h.confirmation_requests()
    await h.answer(request, confirmed=True)
    assert EXECUTED == [("issue_refund", REFUND)]
    assert llm.tool_results()[-1] == ("issue_refund", {"status": "refunded A-17"})


async def test_rejected_review_does_not_run_the_tool() -> None:
    llm = model(call("issue_refund", **REFUND), text("Not refunded."))
    h = await harness(llm, [issue_refund], ACGSPlugin(constitution=RULES))
    await h.ask()
    [request] = h.confirmation_requests()
    await h.answer(request, confirmed=False)
    assert EXECUTED == []
    assert h.tool_responses("issue_refund") == [PENDING, NOT_APPROVED]
    assert llm.tool_results()[-1] == ("issue_refund", NOT_APPROVED)


async def test_block_outranks_review_and_asks_for_no_confirmation() -> None:
    llm = model(call("issue_refund", order_id="A-1", note="refund and delete everything"), text("Done."))
    h = await harness(llm, [issue_refund], ACGSPlugin(constitution=RULES))
    await h.ask()
    assert EXECUTED == []
    assert h.confirmation_requests() == []
    assert h.tool_responses("issue_refund") == [
        {
            "error": "Tool call 'issue_refund' was not executed. Blocked by ACGS rule T-BLOCK (no deletes). "
            "Review rules that also matched: ACGS rule T-REVIEW (refunds need review)."
        }
    ]


async def test_a_confirmed_call_is_still_denied_when_the_rules_now_block_it() -> None:
    llm = model(call("issue_refund", **REFUND), text("Done."))
    h = await harness(llm, [issue_refund], ACGSPlugin(constitution=RULES))
    await h.ask()
    [request] = h.confirmation_requests()
    # The policy is tightened while the call waits; the same session resumes on a new runner.
    stricter = InMemoryRunner(
        app=App(name="acgs_test", root_agent=agent(llm, [issue_refund]), plugins=[ACGSPlugin(constitution=STRICT)])
    )
    stricter.session_service = h.runner.session_service
    h.runner = stricter
    await h.answer(request, confirmed=True)
    assert EXECUTED == []
    assert h.tool_responses("issue_refund")[-1] == {
        "error": "Tool call 'issue_refund' was not executed. Blocked by ACGS rule T-STRICT (refunds are blocked)."
    }


async def test_a_tool_that_requires_its_own_confirmation_is_denied_before_asking() -> None:
    llm = model(call("send_email", **SECRET_EMAIL), text("Done."))
    h = await harness(llm, [FunctionTool(send_email, require_confirmation=True)], ACGSPlugin())
    await h.ask()
    assert h.confirmation_requests() == []
    assert h.tool_responses("send_email") == [ACGS_006]


async def test_allowed_calls_keep_the_tools_own_confirmation() -> None:
    llm = model(call("get_weather", city="Paris"), text("Done."))
    h = await harness(llm, [FunctionTool(get_weather, require_confirmation=True)], ACGSPlugin())
    await h.ask()
    [request] = h.confirmation_requests()
    assert EXECUTED == []
    await h.answer(request, confirmed=True)
    assert EXECUTED == [("get_weather", {"city": "Paris"})]


async def test_one_confirmation_answers_both_acgs_review_and_the_tools_own_confirmation() -> None:
    llm = model(call("issue_refund", **REFUND), text("Done."))
    tool = FunctionTool(issue_refund, require_confirmation=True)
    h = await harness(llm, [tool], ACGSPlugin(constitution=RULES))
    await h.ask()
    [request] = h.confirmation_requests()
    assert request.args is not None
    assert request.args["toolConfirmation"]["hint"].startswith("ACGS: ")
    await h.answer(request, confirmed=True)
    assert EXECUTED == [("issue_refund", REFUND)]
    assert len(h.confirmation_requests()) == 1


async def test_extra_plugins_style_construction_by_class_works() -> None:
    plugin = ACGSPlugin(name="adk_acgs.ACGSPlugin")  # what `adk web --extra_plugins=adk_acgs.ACGSPlugin` does
    llm = model(call("send_email", **SECRET_EMAIL), text("Done."))
    h = await harness(llm, [send_email], plugin)
    await h.ask()
    assert h.tool_responses("send_email") == [ACGS_006]


# --- review: deny and handlers -------------------------------------------------------------------


async def test_on_review_deny_denies_without_asking() -> None:
    llm = model(call("issue_refund", **REFUND), text("Done."))
    h = await harness(llm, [issue_refund], ACGSPlugin(constitution=RULES, on_review="deny"))
    await h.ask()
    assert EXECUTED == []
    assert h.confirmation_requests() == []
    assert h.tool_responses("issue_refund") == [NOT_APPROVED]


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
async def test_review_handler_that_returns_true_runs_the_call(asynchronous: bool) -> None:
    seen: list[tuple[GovernanceDecision, str | None]] = []

    def approve(decision: GovernanceDecision, tool_context: ToolContext) -> bool:
        seen.append((decision, tool_context.function_call_id))
        return True

    async def approve_async(decision: GovernanceDecision, tool_context: ToolContext) -> bool:
        return approve(decision, tool_context)

    plugin = ACGSPlugin(constitution=RULES, on_review=approve_async if asynchronous else approve)
    llm = model(call("issue_refund", **REFUND), text("Done."))
    h = await harness(llm, [issue_refund], plugin)
    await h.ask()
    assert EXECUTED == [("issue_refund", REFUND)]
    [(decision, call_id)] = seen
    assert decision.verdict is Verdict.REVIEW
    assert decision.rule_ids == ("T-REVIEW",)
    assert call_id


@pytest.mark.parametrize("answer", [False, None, 1, "yes", [True]])
async def test_review_handler_must_return_exactly_true(answer: object) -> None:
    plugin = ACGSPlugin(constitution=RULES, on_review=lambda decision, context: answer)  # type: ignore[arg-type,return-value]
    llm = model(call("issue_refund", **REFUND), text("Done."))
    h = await harness(llm, [issue_refund], plugin)
    await h.ask()
    assert EXECUTED == []
    assert h.tool_responses("issue_refund") == [NOT_APPROVED]


# --- fail closed ---------------------------------------------------------------------------------


async def test_review_handler_error_denies_the_call(caplog: pytest.LogCaptureFixture) -> None:
    def broken(decision: GovernanceDecision, tool_context: ToolContext) -> bool:
        msg = "handler down"
        raise ConnectionError(msg)

    llm = model(call("issue_refund", **REFUND), text("Done."))
    h = await harness(llm, [issue_refund], ACGSPlugin(constitution=RULES, on_review=broken))
    with caplog.at_level(logging.WARNING, logger="adk_acgs.plugin"):
        await h.ask()
    assert EXECUTED == []
    assert h.tool_responses("issue_refund") == [{"error": f"Tool call 'issue_refund' was not executed. {FAILED}"}]
    assert "fail-closed" in caplog.text


async def test_formatter_error_denies_the_call() -> None:
    def broken(tool_name: str, arguments: Any) -> str:
        msg = "formatter down"
        raise ValueError(msg)

    llm = model(call("get_weather", city="Paris"), text("Done."))
    h = await harness(llm, [get_weather], ACGSPlugin(tool_call_formatter=broken))
    await h.ask()
    assert EXECUTED == []
    assert h.tool_responses("get_weather") == [{"error": f"Tool call 'get_weather' was not executed. {FAILED}"}]


@pytest.mark.parametrize("name", ["", None, 7])
async def test_tool_without_a_usable_name_is_denied(name: object) -> None:
    tool = SimpleNamespace(name=name)
    result = await ACGSPlugin().before_tool_callback(tool=tool, tool_args={}, tool_context=SimpleNamespace())  # type: ignore[arg-type]
    assert result == {"error": f"Tool call '{name}' was not executed. {FAILED}"}


async def test_confirmation_that_cannot_be_requested_denies_the_call() -> None:
    def refuse(*, hint: str) -> None:
        msg = "request_confirmation requires function_call_id."
        raise ValueError(msg)

    context = SimpleNamespace(tool_confirmation=None, request_confirmation=refuse)
    tool = SimpleNamespace(name="issue_refund")
    result = await ACGSPlugin(constitution=RULES).before_tool_callback(
        tool=tool,  # type: ignore[arg-type]
        tool_args=REFUND,
        tool_context=context,  # type: ignore[arg-type]
    )
    assert result == {"error": f"Tool call 'issue_refund' was not executed. {FAILED}"}


async def test_a_call_to_an_unknown_tool_never_runs_anything() -> None:
    llm = model(call("send_email", **SECRET_EMAIL), text("Done."))
    h = await harness(llm, [get_weather], ACGSPlugin())
    rejected_by_adk = ""
    try:
        await h.ask()
    except ValueError as error:  # early 2.x releases reject unknown tools before any callback runs
        rejected_by_adk = str(error)
    if rejected_by_adk:
        assert "send_email" in rejected_by_adk
    else:  # later releases run the callbacks first, so ACGS answers the call
        assert h.tool_responses("send_email") == [ACGS_006]
    assert EXECUTED == []


# --- scope, ordering and registration ------------------------------------------------------------


async def test_agent_tool_sub_agents_inherit_the_plugin() -> None:
    inner_llm = model(call("send_email", **SECRET_EMAIL), text("Inner done."))
    inner = LlmAgent(name="mailer", model=inner_llm, instruction="Send mail.", tools=[send_email])
    outer_llm = model(call("mailer", request="email the ops team"), text("Outer done."))
    h = await harness(outer_llm, [AgentTool(agent=inner)], ACGSPlugin())
    await h.ask()
    assert EXECUTED == []
    assert inner_llm.tool_results() == [("send_email", ACGS_006)]


async def test_agent_tool_with_include_plugins_false_is_not_governed() -> None:
    inner_llm = model(call("send_email", **SECRET_EMAIL), text("Inner done."))
    inner = LlmAgent(name="mailer", model=inner_llm, instruction="Send mail.", tools=[send_email])
    outer_llm = model(call("mailer", request="email the ops team"), text("Outer done."))
    h = await harness(outer_llm, [AgentTool(agent=inner, include_plugins=False)], ACGSPlugin())
    await h.ask()
    assert EXECUTED == [("send_email", SECRET_EMAIL)]


async def test_earlier_plugin_that_rewrites_args_is_checked_with_the_new_args() -> None:
    class Rewriter(BasePlugin):
        async def before_tool_callback(
            self, *, tool: BaseTool, tool_args: dict[str, Any], tool_context: ToolContext
        ) -> dict[str, Any] | None:
            tool_args["body"] = SECRET_EMAIL["body"]
            return None

    llm = model(call("send_email", to="ops@example.com", body="hello"), text("Done."))
    h = await harness(llm, [send_email], Rewriter(name="rewriter"), ACGSPlugin())
    await h.ask()
    assert EXECUTED == []
    assert h.tool_responses("send_email") == [ACGS_006]


async def test_agent_level_callbacks_run_only_after_the_plugin_allows_the_call() -> None:
    seen: list[str] = []

    def agent_callback(tool: BaseTool, args: dict[str, Any], tool_context: ToolContext) -> None:
        seen.append(tool.name)

    llm = model(calls(("send_email", SECRET_EMAIL), ("get_weather", {"city": "Paris"})), text("Done."))
    h = await harness(llm, [send_email, get_weather], ACGSPlugin(), before_tool_callback=agent_callback)
    await h.ask()
    assert seen == ["get_weather"]


async def test_as_before_tool_callback_governs_only_that_agent() -> None:
    gate = ACGSPlugin().as_before_tool_callback()
    llm = model(call("send_email", **SECRET_EMAIL), text("Done."))
    h = await harness(llm, [send_email], before_tool_callback=gate)
    await h.ask()
    assert EXECUTED == []
    assert h.tool_responses("send_email") == [ACGS_006]

    ungoverned = model(call("send_email", **SECRET_EMAIL), text("Done."))
    h = await harness(ungoverned, [send_email])
    await h.ask()
    assert EXECUTED == [("send_email", SECRET_EMAIL)]


async def test_as_before_tool_callback_supports_confirmation() -> None:
    llm = model(call("issue_refund", **REFUND), text("Done."))
    h = await harness(
        llm, [issue_refund], before_tool_callback=ACGSPlugin(constitution=RULES).as_before_tool_callback()
    )
    await h.ask()
    [request] = h.confirmation_requests()
    await h.answer(request, confirmed=True)
    assert EXECUTED == [("issue_refund", REFUND)]


async def test_two_plugins_need_different_names() -> None:
    llm = model(text("Hi."))
    with pytest.raises(ValueError, match="acgs"):
        await harness(llm, [], ACGSPlugin(), ACGSPlugin())
    h = await harness(llm, [], ACGSPlugin(), ACGSPlugin(name="acgs-strict", constitution=STRICT))
    await h.ask()


async def test_every_plugin_on_the_app_must_allow_the_call() -> None:
    llm = model(call("issue_refund", **REFUND), text("Done."))
    plugins = [
        ACGSPlugin(constitution=RULES, on_review=lambda d, c: True),
        ACGSPlugin(name="strict", constitution=STRICT),
    ]
    h = await harness(llm, [issue_refund], *plugins)
    await h.ask()
    assert EXECUTED == []
    assert h.tool_responses("issue_refund") == [
        {"error": "Tool call 'issue_refund' was not executed. Blocked by ACGS rule T-STRICT (refunds are blocked)."}
    ]


# --- configuration -------------------------------------------------------------------------------


def test_defaults() -> None:
    plugin = ACGSPlugin()
    assert plugin.name == PLUGIN_NAME == "acgs"
    assert plugin.on_review == "confirm"
    assert isinstance(plugin, BasePlugin)
    assert plugin.constitution.name == Constitution.default().name


def test_custom_constitution_and_evaluate() -> None:
    plugin = ACGSPlugin(constitution=RULES, agent_id="billing")
    assert plugin.constitution is RULES
    decision = plugin.evaluate("issue_refund", REFUND)
    assert decision.verdict is Verdict.REVIEW
    assert decision.reason == REVIEW_REASON


@pytest.mark.parametrize("value", ["human_input", "allow", "", None])
def test_invalid_on_review_is_rejected(value: object) -> None:
    with pytest.raises(ValueError, match="on_review must be"):
        ACGSPlugin(on_review=value)  # type: ignore[arg-type]


def _merged() -> Constitution:
    return Constitution.from_rules([*Constitution.default().rules, REVIEW_RULE, BLOCK_RULE], name="merged")


def test_default_engine_keeps_no_record_of_checked_calls(caplog: pytest.LogCaptureFixture) -> None:
    """The README's data-handling section: argument values are not stored, echoed or logged."""
    caplog.set_level(logging.DEBUG)
    plugin = ACGSPlugin(agent_id="payments-app")
    decision = plugin.evaluate("send_email", {"to": "a@example.com", "body": "The password is hunter2"})
    assert decision.verdict is Verdict.DENY
    engine = plugin._governor._engine  # the GovernanceEngine the plugin created
    assert engine.stats["audit_mode"] == "fast"
    assert engine.stats["audit_entry_count"] == 0
    assert engine.audit_log.entries == []
    assert "hunter2" not in decision.reason
    assert "hunter2" not in caplog.text
