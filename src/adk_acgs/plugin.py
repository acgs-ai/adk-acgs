# SPDX-FileCopyrightText: 2026-present ACGS Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""ACGS governance for Google ADK tool calls, as an ADK plugin."""

from __future__ import annotations

import inspect
import logging
from collections.abc import Awaitable, Callable
from typing import Any, Literal

from acgs_lite import Constitution
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext

from adk_acgs._governance import GovernanceDecision, ToolCallFormatter, ToolCallGovernor, Verdict

logger = logging.getLogger(__name__)

ReviewHandler = Callable[[GovernanceDecision, ToolContext], bool | Awaitable[bool]]
"""A function, sync or async, that decides a review verdict. It must return ``True`` to run the tool call."""

BeforeToolCallback = Callable[[BaseTool, dict[str, Any], ToolContext], Awaitable[dict[str, Any] | None]]
"""The signature of ``LlmAgent.before_tool_callback``."""

PLUGIN_NAME = "acgs"
"""The default plugin name. ADK requires every plugin on an ``App`` to have a unique name."""

_EVALUATION_FAILED = "ACGS governance could not evaluate this tool call, so it was denied."


class ACGSPlugin(BasePlugin):
    """Check every Google ADK tool call against an ACGS constitution before the tool runs.

    Register the plugin on an ``App`` (``App(plugins=[ACGSPlugin()])``). Its ``before_tool_callback``
    runs for every tool call of every agent the runner manages, before agent-level callbacks:

    - Allowed calls return ``None`` and run unchanged.
    - Denied calls return ``{"error": "Tool call '<name>' was not executed. <reason>"}``. ADK uses
      that dict as the tool result, so the tool does not run and the model sees the reason.
    - Calls that only match review rules use ADK's tool confirmation by default
      (``on_review="confirm"``): the plugin calls ``tool_context.request_confirmation``, ADK pauses
      the call and emits an ``adk_request_confirmation`` function call, and the tool runs only when
      the client answers with ``confirmed: true``. ``on_review="deny"`` denies them instead, and a
      callable decides them in code.

    A blocking rule always wins: a call that matches one is denied even if it was confirmed. Any
    error while evaluating or reviewing a call denies it instead of failing the run.

    Example:
        >>> from adk_acgs import ACGSPlugin
        >>> plugin = ACGSPlugin()
        >>> plugin.name
        'acgs'
        >>> plugin.evaluate("send_email", {"to": "a@example.com", "body": "The password is hunter2"}).verdict
        <Verdict.DENY: 'deny'>
    """

    def __init__(
        self,
        *,
        constitution: Constitution | None = None,
        agent_id: str = "adk-agent",
        on_review: Literal["confirm", "deny"] | ReviewHandler = "confirm",
        tool_call_formatter: ToolCallFormatter | None = None,
        name: str = PLUGIN_NAME,
    ) -> None:
        """Create the plugin.

        Args:
            constitution: The ACGS constitution to enforce. Defaults to
                ``acgs_lite.Constitution.default()``.
            agent_id: Agent identifier recorded by ACGS for each validation.
            on_review: What to do when only review rules match. ``"confirm"`` asks for an ADK tool
                confirmation and runs the call only if it is confirmed. ``"deny"`` denies the call.
                A callable ``(decision, tool_context) -> bool`` (sync or async) runs the call only
                if it returns ``True``.
            tool_call_formatter: Optional function ``(tool_name, args) -> str`` that replaces
                ``format_tool_call``.
            name: The plugin name. Each plugin on an ``App`` needs a unique name.

        Raises:
            ValueError: If ``on_review`` is invalid.
        """
        if not callable(on_review) and on_review not in ("confirm", "deny"):
            msg = f"on_review must be 'confirm', 'deny' or a callable, got {on_review!r}"
            raise ValueError(msg)
        super().__init__(name=name)
        self.on_review = on_review
        self._governor = ToolCallGovernor(constitution, agent_id=agent_id, tool_call_formatter=tool_call_formatter)

    @property
    def constitution(self) -> Constitution:
        """The constitution this plugin enforces."""
        return self._governor.constitution

    def evaluate(self, tool_name: str, arguments: Any) -> GovernanceDecision:
        """Evaluate a tool call without a running agent. Never raises.

        Args:
            tool_name: Name of the tool.
            arguments: The tool arguments.

        Returns:
            The governance decision.
        """
        return self._governor.evaluate(tool_name, arguments)

    async def before_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
    ) -> dict[str, Any] | None:
        """ADK plugin hook: return ``None`` to run the tool call, or a dict that replaces its result.

        Args:
            tool: The tool the model called.
            tool_args: The call arguments.
            tool_context: The ADK tool context of the call.

        Returns:
            ``None`` if the call may run, otherwise the result the model receives instead.
        """
        tool_name = "unknown"
        try:
            tool_name = tool.name
            if not isinstance(tool_name, str) or not tool_name:
                msg = f"Tool name must be a non-empty string, got {tool_name!r}."
                raise TypeError(msg)
            decision = self.evaluate(tool_name, tool_args)
            if decision.verdict is Verdict.ALLOW:
                return None
            if decision.verdict is Verdict.DENY:
                return _denied(tool_name, decision.reason)
            return await self._review(decision, tool_context)
        except Exception:  # fail-closed: ADK would otherwise abort the whole run
            logger.warning("ACGS plugin failed for tool %r; denying the call (fail-closed).", tool_name, exc_info=True)
            return _denied(str(tool_name), _EVALUATION_FAILED)

    def as_before_tool_callback(self) -> BeforeToolCallback:
        """Return the same check as an agent-level ``before_tool_callback``.

        Use this to govern a single ``LlmAgent`` instead of every agent on the runner:
        ``Agent(..., before_tool_callback=plugin.as_before_tool_callback())``. Agent callbacks run
        after all plugin callbacks, and only for that agent.

        Returns:
            An async callable with the ``LlmAgent.before_tool_callback`` signature.
        """

        async def acgs_before_tool_callback(
            tool: BaseTool, args: dict[str, Any], tool_context: ToolContext
        ) -> dict[str, Any] | None:
            return await self.before_tool_callback(tool=tool, tool_args=args, tool_context=tool_context)

        return acgs_before_tool_callback

    async def _review(self, decision: GovernanceDecision, tool_context: ToolContext) -> dict[str, Any] | None:
        tool_name = decision.tool_name
        if self.on_review == "deny":
            return _not_approved(tool_name, decision.reason)
        if self.on_review == "confirm":
            confirmation = tool_context.tool_confirmation
            if confirmation is None:
                tool_context.request_confirmation(
                    hint=f"ACGS: the tool call '{tool_name}' needs human review. {decision.reason}"
                )
                # A pause is not a tool result for the model to answer; ADK's own confirmation
                # gate sets the same flag so the model is not called again until the reply arrives.
                tool_context.actions.skip_summarization = True
                return {"error": f"Tool call '{tool_name}' is waiting for human review. {decision.reason}"}
            if confirmation.confirmed is True:
                logger.info("ACGS review of tool '%s' was confirmed.", tool_name)
                return None
            return _not_approved(tool_name, decision.reason)
        approved = self.on_review(decision, tool_context)
        if inspect.isawaitable(approved):
            approved = await approved
        if approved is True:
            logger.info("ACGS review handler approved tool '%s'.", tool_name)
            return None
        return _not_approved(tool_name, decision.reason)


def _denied(tool_name: str, reason: str) -> dict[str, Any]:
    return {"error": f"Tool call '{tool_name}' was not executed. {reason}"}


def _not_approved(tool_name: str, reason: str) -> dict[str, Any]:
    return _denied(tool_name, f"Human review did not approve the call. {reason}")
