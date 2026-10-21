# adk-acgs

[![CI](https://github.com/acgs-ai/adk-acgs/actions/workflows/ci.yml/badge.svg)](https://github.com/acgs-ai/adk-acgs/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/adk-acgs.svg)](https://pypi.org/project/adk-acgs/)
[![Python](https://img.shields.io/pypi/pyversions/adk-acgs.svg)](https://pypi.org/project/adk-acgs/)

`adk-acgs` checks every [Google ADK](https://adk.dev) tool call against [ACGS](https://pypi.org/project/acgs-lite/)
constitution rules before the tool runs. It is an ADK [plugin](https://adk.dev/plugins/): you register it once on
your `App`, and its `before_tool_callback` runs for every tool call of every agent the runner manages. Agents and
tools stay unchanged.

- Calls that match no blocking rule run unchanged.
- Calls that match a blocking rule do not run. The plugin returns
  `{"error": "Tool call '<name>' was not executed. <reason>"}`, which ADK uses as the tool result, so the model sees
  the reason and can continue.
- Calls that only match review rules are paused with ADK's own
  [tool confirmation](https://adk.dev/tools-custom/confirmation/) and run only once a person (or your client)
  confirms them. They can also be denied outright or decided by your own function.
- Any error while evaluating or reviewing a call denies it (fail-closed).

The package is a thin adapter over the public `acgs-lite` API.

## Installation

```bash
pip install adk-acgs
```

Requires Python 3.10 or later, `google-adk>=2.0.0,<3` and `acgs-lite>=3.0.1`. (ADK 2.12 and later need Python 3.11.)

## Usage

```python
from google.adk.agents import LlmAgent
from google.adk.apps import App

from adk_acgs import ACGSPlugin

root_agent = LlmAgent(name="assistant", model="gemini-flash-latest", instruction="Help the user.")
app = App(name="my_app", root_agent=root_agent, plugins=[ACGSPlugin()])
# runner = InMemoryRunner(app=app)  # or any other Runner
```

ADK recommends plugins over agent callbacks for security guardrails
([Callbacks and Plugins for Security Guardrails](https://adk.dev/safety/#callbacks-and-plugins-for-security-guardrails)).
Plugin callbacks run before agent-level callbacks and apply to every agent on the runner, including agents wrapped
in an `AgentTool`.

`adk web` and `adk api_server` can also load the plugin without code changes:
`--extra_plugins=adk_acgs.ACGSPlugin` uses the default constitution, and the import path of a module-level
`ACGSPlugin(...)` instance uses your own settings. ADK only logs an error and continues without the plugin if that
import fails, so check the server log, or register the plugin on the `App` instead.

This complete example runs without an API key; a scripted model stands in for Gemini and prints each tool result
the model receives:

```python
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
```

Output of `python examples/quickstart.py` (ADK also logs warnings about experimental features to stderr):

```text
send_email: {'error': "Tool call 'send_email' was not executed. Blocked by ACGS rule ACGS-006 (Agents must not expose sensitive data in responses)."}
get_weather: {'forecast': 'Sunny in Paris'}
paused: ACGS: the tool call 'issue_refund' needs human review. Human review required by ACGS rule REFUND-REVIEW (Refunds need human approval).
issue_refund: {'status': 'refunded 20.0 for A-17'}
```

## How a tool call is evaluated

1. The tool name and arguments ADK passes to `before_tool_callback` are rendered as text by `format_tool_call`:

   ```python
   from adk_acgs import format_tool_call

   print(format_tool_call("update_config", {"service": "billing", "change": "disable\nlogging", "dryRun": False}))
   ```

   ```text
   tool: update_config
   arguments:
   change: disable logging
   dryRun: false
   service: billing
   normalized: update config | dry run
   ```

   Each argument leaf is one `path: value` line (nested values use `a.b[0].c` paths), so both argument names and
   values are visible to rules. Whitespace inside values is collapsed so phrases split across lines still match.
   Identifiers written in `snake_case`, `kebab-case` or `camelCase` (the tool name, argument names, and
   single-token string values) are repeated as plain words on the `normalized:` line, so the keyword
   `skip audit` also matches a tool called `skip_audit`. Nothing is truncated.
   Pass `tool_call_formatter=` to use a different representation.

2. The text is validated with `acgs_lite.GovernanceEngine.validate` against the constitution
   (`Constitution.default()` unless you pass one).

3. The result decides what happens to the call:

   | ACGS outcome | Result |
   |---|---|
   | No blocking rule matched (warn-only rules may match) | The tool runs |
   | Only `require_human_review` / `escalate_to_senior` rules matched | Review (see below) |
   | Any `block`, `block_and_notify` or `halt_and_alert` rule matched | Denied |
   | A rule ID not found in the constitution, or an invalid result without a rule | Denied |
   | Any exception while reading, evaluating or reviewing the tool call | Denied (fail-closed) |

   A matched blocking rule is never downgraded to review because a review rule matched as well.
   The denial reason then lists only the blocking rules after "Blocked by" and names the matched review
   rules separately.

A denied call is answered with `{"error": "Tool call '<name>' was not executed. <reason>"}`, the same `error` shape
ADK's own confirmation gate uses. ADK skips the tool, the `before_tool_callback`s of later plugins and of the agent,
and sends that result to the model (`after_tool_callback`s still see it). The reason names the matched rules but never repeats argument values. Exceptions raised by the tool itself
are reported by ADK as usual.

ADK turns an exception raised in a plugin callback into a `RuntimeError` that ends the run. The plugin therefore
catches every error itself and turns it into a denial, so one bad call does not stop the agent.

`ACGSPlugin.evaluate(tool_name, args)` returns the underlying `GovernanceDecision`
(`verdict`, `reason`, `rule_ids`, `warning_rule_ids`, `error`) without running anything.

## Review

| `on_review` | Behaviour |
|---|---|
| `"confirm"` (default) | Uses ADK tool confirmation. The plugin calls `tool_context.request_confirmation(hint=...)`, ADK emits an `adk_request_confirmation` function call and pauses the tool call, and the model is not asked to answer the pause. The client (the ADK web UI shows a dialog; API clients send a `FunctionResponse` to `/run` or `/run_sse`) replies with `{"confirmed": true}` to run the call or `{"confirmed": false}` to deny it. |
| `"deny"` | The call is denied with the review reason. Use it for unattended runs. |
| a callable `(decision, tool_context) -> bool`, sync or async | Your function decides, for example by calling an approval service. Only a return value of exactly `True` runs the call; other values and exceptions deny it. |

```python
from adk_acgs import ACGSPlugin

unattended = ACGSPlugin(on_review="deny")
own_review = ACGSPlugin(on_review=lambda decision, tool_context: decision.tool_name == "issue_refund")
```

When a confirmed call resumes, ADK runs the plugin again with `tool_context.tool_confirmation` set. The plugin
evaluates the call again, so a blocking rule still wins: if the rules changed while the call was waiting and now
block it, the call is denied even though it was confirmed. ADK resumes the call with the arguments recorded in the
session when confirmation was requested; the reply only carries the decision.

Tool confirmation is marked experimental in ADK. Its documented limitations apply: it does not work with
`DatabaseSessionService` or `VertexAiSessionService`, and in live (bidirectional streaming) runs a paused call is
never resumed, so a call that needs review is not run. Use `on_review="deny"` or a callable there.

A confirmation answers one function call ID. If a tool also requires its own confirmation
(`FunctionTool(..., require_confirmation=True)`) and ACGS asks for review on the same call, the user sees the ACGS
request first and one confirmation answers both.

## Scope and ordering

- Plugins run in the order of the `plugins` list, and ADK stops at the first plugin that returns a value. Put
  `ACGSPlugin` after any plugin that changes `tool_args`, so ACGS evaluates the arguments the tool receives.
  Agent-level `before_tool_callback`s always run after all plugins and only for calls the plugin allowed; if one
  of them changes the arguments, ACGS does not see the change.
- `AgentTool(agent=..., include_plugins=False)` runs the wrapped agent without the parent's plugins, so its tool
  calls are not checked.
- Each plugin on an `App` needs a unique name. To enforce two constitutions, give the second instance another
  name: `ACGSPlugin(name="acgs-payments", constitution=...)`. A call runs only if every instance allows it.

To govern a single agent instead of the whole runner, use the same check as an agent callback:

```python
from google.adk.agents import LlmAgent

from adk_acgs import ACGSPlugin

payments_agent = LlmAgent(
    name="payments",
    model="gemini-flash-latest",
    instruction="Handle payments.",
    before_tool_callback=ACGSPlugin().as_before_tool_callback(),
)
```

## Custom constitutions

```python
from acgs_lite import Constitution, Rule, Severity, ViolationAction

from adk_acgs import ACGSPlugin

constitution = Constitution.from_rules(
    [
        Rule(
            id="NO-DESTRUCTIVE-SQL",
            text="No destructive SQL",
            severity=Severity.CRITICAL,
            keywords=["drop table", "truncate table"],
        ),
        Rule(
            id="REFUND-REVIEW",
            text="Refunds need approval",
            severity=Severity.HIGH,
            keywords=["refund"],
            workflow_action=ViolationAction.REQUIRE_HUMAN_REVIEW,
        ),
    ],
    name="payments",
)
plugin = ACGSPlugin(constitution=constitution, agent_id="payments-app")
```

`Constitution.from_yaml("rules.yaml")` works as well.

## Options

| Argument | Default | Meaning |
|---|---|---|
| `constitution` | `Constitution.default()` | Rules to enforce |
| `agent_id` | `"adk-agent"` | Agent identifier recorded by ACGS |
| `on_review` | `"confirm"` | `"confirm"`, `"deny"`, or a callable `(decision, tool_context) -> bool` |
| `tool_call_formatter` | `format_tool_call` | `(tool_name, args) -> str` used as the evaluated text |
| `name` | `"acgs"` | Plugin name; must be unique on the `App` |

## Limitations

- Only tool calls are checked. User messages, model requests and responses, and tool results are not.
- Tested with `App` + `InMemoryRunner` (`run_async` and the synchronous `run`), parallel function calls in one
  model turn, `AgentTool` sub-agents, the confirmation pause and resume, and agent-level callbacks, against
  google-adk 2.0.0 and the latest release. Live (`run_live`) runs are not tested.
- `Constitution.default()` focuses on governance integrity (self-approval, skipping audits, privilege escalation)
  and on secrets and personal data (`password`, `api_key`, `sk-...` and `ghp_...` tokens, US SSN
  patterns). It does **not** cover destructive operations such as `rm -rf` or `DROP TABLE`. Write rules for the
  tools you expose. `tests/test_default_constitution.py` lists what the default rules do and do not catch.
- Keyword rules in acgs-lite 3.0.1 match literal, case-insensitive substrings. They do not understand negation
  ("never share the password" matches `password`) and do not treat `-`, `_` and spaces as equivalent
  (`api-key` does not match the keyword `api_key`).
- Because argument names are part of the evaluated text, a tool with a parameter named `password` is denied under
  the default constitution whenever it is called.

## Development

```bash
uv sync --group dev        # or use hatch: hatch run prepare
hatch run format           # ruff format + ruff check --fix
hatch run lint             # ruff check + ruff format --check
hatch run typecheck        # mypy --strict
hatch run cov              # pytest with a 100% branch-coverage gate, network disabled
hatch run min:cov          # the same tests against google-adk 2.0.0 and acgs-lite 3.0.1
pre-commit install         # format, lint, type check and tests on commit; commitizen on commit messages
```

Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/); `cz bump` updates the version
and the changelog.

## License

Apache-2.0. See [LICENSE](LICENSE).
