# SPDX-FileCopyrightText: 2026-present ACGS Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""A scripted ADK model and a small harness around a real ``App`` and ``InMemoryRunner``."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Sequence
from dataclasses import dataclass, field
from typing import Any

from google.adk.agents import LlmAgent
from google.adk.apps import App
from google.adk.events import Event
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.runners import InMemoryRunner
from google.genai import types
from pydantic import Field

CONFIRMATION_CALL = "adk_request_confirmation"


class ScriptedLlm(BaseLlm):
    """Replays scripted model turns and records every request it receives."""

    replies: list[types.Content] = Field(default_factory=list)
    requests: list[LlmRequest] = Field(default_factory=list)

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        self.requests.append(llm_request)
        reply = self.replies.pop(0) if self.replies else text("Done.")
        yield LlmResponse(content=reply)

    def tool_results(self) -> list[tuple[str, dict[str, Any]]]:
        """Every function response the model was shown, in order, without repeats."""
        seen: list[tuple[str, dict[str, Any]]] = []
        for request in self.requests:
            for content in request.contents:
                for part in content.parts or []:
                    response = part.function_response
                    if response is not None and (response.name, response.response) not in seen:
                        seen.append((str(response.name), dict(response.response or {})))
        return seen


def calls(*calls: tuple[str, dict[str, Any]]) -> types.Content:
    """A model turn with one or more function calls."""
    return types.Content(
        role="model",
        parts=[types.Part(function_call=types.FunctionCall(name=name, args=args)) for name, args in calls],
    )


def call(name: str, **args: Any) -> types.Content:
    """A model turn with a single function call."""
    return calls((name, args))


def text(value: str) -> types.Content:
    """A model turn with a text answer."""
    return types.Content(role="model", parts=[types.Part.from_text(text=value)])


def model(*replies: types.Content) -> ScriptedLlm:
    return ScriptedLlm(model="scripted", replies=list(replies))


@dataclass
class Harness:
    """A real ADK ``App`` + ``InMemoryRunner`` session for one test."""

    runner: InMemoryRunner
    session_id: str = ""
    user_id: str = "user"
    events: list[Event] = field(default_factory=list)

    @classmethod
    async def create(cls, agent: LlmAgent, plugins: Sequence[BasePlugin] = ()) -> Harness:
        runner = InMemoryRunner(app=App(name="acgs_test", root_agent=agent, plugins=list(plugins)))
        harness = cls(runner=runner)
        session = await runner.session_service.create_session(app_name="acgs_test", user_id=harness.user_id)
        harness.session_id = session.id
        return harness

    async def send(self, message: types.Content) -> list[Event]:
        events = [
            event
            async for event in self.runner.run_async(
                user_id=self.user_id, session_id=self.session_id, new_message=message
            )
        ]
        self.events.extend(events)
        return events

    async def ask(self, prompt: str = "Please help.") -> list[Event]:
        return await self.send(types.Content(role="user", parts=[types.Part.from_text(text=prompt)]))

    async def answer(self, confirmation: types.FunctionCall, *, confirmed: bool) -> list[Event]:
        """Reply to an ``adk_request_confirmation`` call, as a client or the ADK web UI would."""
        response = types.FunctionResponse(id=confirmation.id, name=CONFIRMATION_CALL, response={"confirmed": confirmed})
        return await self.send(types.Content(role="user", parts=[types.Part(function_response=response)]))

    def confirmation_requests(self) -> list[types.FunctionCall]:
        return [fc for event in self.events for fc in event.get_function_calls() if fc.name == CONFIRMATION_CALL]

    def tool_responses(self, name: str) -> list[dict[str, Any]]:
        return [
            dict(fr.response or {})
            for event in self.events
            for fr in event.get_function_responses()
            if fr.name == name and event.author != "user"
        ]
