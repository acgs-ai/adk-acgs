"""Run one allowed and one denied ADK tool call offline, then print a decision audit trail."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncGenerator
from dataclasses import asdict
from typing import Any

# Disable telemetry export before importing ADK; no model service is used.
os.environ["OTEL_SDK_DISABLED"] = "true"

from google.adk.agents import LlmAgent
from google.adk.apps import App
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import InMemoryRunner
from google.genai import types

from adk_acgs import ACGSPlugin, GovernanceDecision


class AuditedPlugin(ACGSPlugin):
    """Collect decisions through the public plugin API, without retaining argument values."""

    def __init__(self) -> None:
        super().__init__()
        self.decisions: list[GovernanceDecision] = []

    def evaluate(self, tool_name: str, arguments: Any) -> GovernanceDecision:
        decision = super().evaluate(tool_name, arguments)
        self.decisions.append(decision)
        return decision


class ScriptedModel(BaseLlm):
    """Return prewritten tool calls without an API key or network request."""

    replies: list[types.Content]

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        yield LlmResponse(content=self.replies.pop(0))


async def record_note(body: str) -> str:
    """Record a demo note by printing it locally."""
    print(f"Executed record_note: {body}")
    return "Note recorded."


async def main() -> None:
    replies = [
        types.Content(
            role="model",
            parts=[types.Part(function_call=types.FunctionCall(name="record_note", args={"body": body}))],
        )
        for body in ["The meeting starts at noon.", "The admin password is demo-secret."]
    ]
    replies.append(types.Content(role="model", parts=[types.Part.from_text(text="Done.")]))
    agent = LlmAgent(
        name="note_taker",
        model=ScriptedModel(model="scripted", replies=replies),
        instruction="Record the two notes safely.",
        tools=[record_note],
    )
    gate = AuditedPlugin()
    app = App(name="deny_and_audit", root_agent=agent, plugins=[gate])
    runner = InMemoryRunner(app=app)
    try:
        session = await runner.session_service.create_session(app_name=app.name, user_id="user")
        message = types.Content(role="user", parts=[types.Part.from_text(text="Record the notes.")])
        async for _event in runner.run_async(user_id="user", session_id=session.id, new_message=message):
            pass
    finally:
        # Untyped in the oldest supported google-adk, typed in newer ones.
        await runner.close()  # type: ignore[no-untyped-call,unused-ignore]
    print("Audit trail (application decisions):")
    for decision in gate.decisions:
        print(json.dumps(asdict(decision), sort_keys=True))


if __name__ == "__main__":
    asyncio.run(main())
