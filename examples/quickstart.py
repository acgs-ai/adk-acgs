"""Gate Google ADK tool calls with ACGS. Runs offline: a scripted model stands in for Gemini."""

import asyncio
from collections.abc import AsyncGenerator
from typing import Any

from acgs_lite import Constitution, Rule, Severity, ViolationAction
from google.adk.agents import LlmAgent
from google.adk.apps import App
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import InMemoryRunner
from google.genai import types

from adk_acgs import ACGSPlugin


class ScriptedModel(BaseLlm):
    """Replays scripted turns. Prints each tool result the model receives."""

    replies: list[types.Content]

    async def generate_content_async(
        self,
        llm_request: LlmRequest,
        stream: bool = False,
    ) -> AsyncGenerator[LlmResponse, None]:
        for part in llm_request.contents[-1].parts or []:
            if part.function_response:
                print(f"{part.function_response.name}: {part.function_response.response}")
        yield LlmResponse(content=self.replies.pop(0))


def call(name: str, **args: Any) -> types.Content:
    return types.Content(role="model", parts=[types.Part(function_call=types.FunctionCall(name=name, args=args))])


DONE = types.Content(role="model", parts=[types.Part.from_text(text="Done.")])


def send_email(to: str, body: str) -> dict[str, str]:
    """Send an email."""
    return {"status": f"sent to {to}"}


def get_weather(city: str) -> dict[str, str]:
    """Get the weather for a city."""
    return {"forecast": f"Sunny in {city}"}


def issue_refund(order_id: str, amount: float) -> dict[str, str]:
    """Refund an order."""
    return {"status": f"refunded {amount} for {order_id}"}


async def start(app: App) -> InMemoryRunner:
    runner = InMemoryRunner(app=app)
    await runner.session_service.create_session(app_name=app.name, user_id="user", session_id="demo")
    return runner


async def send(runner: InMemoryRunner, message: types.Content) -> list[types.FunctionCall]:
    """Run one turn and return the confirmation requests ADK emitted."""
    requests = []
    async for event in runner.run_async(user_id="user", session_id="demo", new_message=message):
        requests += [fc for fc in event.get_function_calls() if fc.name == "adk_request_confirmation"]
    return requests


def user(text: str) -> types.Content:
    return types.Content(role="user", parts=[types.Part.from_text(text=text)])


def confirm(request: types.FunctionCall, *, confirmed: bool) -> types.Content:
    """The reply a client (or the ADK web UI) sends to an ``adk_request_confirmation`` call."""
    response = types.FunctionResponse(id=request.id, name=request.name, response={"confirmed": confirmed})
    return types.Content(role="user", parts=[types.Part(function_response=response)])


async def main() -> None:
    # 1. The default constitution blocks the email that leaks a password; the weather lookup runs.
    model = ScriptedModel(
        model="scripted",
        replies=[
            call("send_email", to="ops@example.com", body="The admin password is hunter2"),
            call("get_weather", city="Paris"),
            DONE,
        ],
    )
    agent = LlmAgent(name="assistant", model=model, instruction="Help the user.", tools=[send_email, get_weather])
    runner = await start(App(name="default_rules", root_agent=agent, plugins=[ACGSPlugin()]))
    await send(runner, user("Hi"))

    # 2. A review rule: refunds pause for an ADK tool confirmation, then run once confirmed.
    refunds = Constitution.from_rules(
        [
            Rule(
                id="REFUND-REVIEW",
                text="Refunds need human approval",
                severity=Severity.HIGH,
                keywords=["refund"],
                workflow_action=ViolationAction.REQUIRE_HUMAN_REVIEW,
            )
        ],
        name="refunds",
    )
    model = ScriptedModel(model="scripted", replies=[call("issue_refund", order_id="A-17", amount=20.0), DONE])
    agent = LlmAgent(name="assistant", model=model, instruction="Help the user.", tools=[issue_refund])
    runner = await start(App(name="refunds", root_agent=agent, plugins=[ACGSPlugin(constitution=refunds)]))

    [request] = await send(runner, user("Refund order A-17"))
    print(f"paused: {request.args['toolConfirmation']['hint']}")  # type: ignore[index]
    await send(runner, confirm(request, confirmed=True))


asyncio.run(main())
