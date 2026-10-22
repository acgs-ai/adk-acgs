# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/). Entries for new releases are generated with `cz bump`.

## [Unreleased]

## 0.1.1 (2026-10-22)

### Changed

- Docs: `agent_id` is described as what it is, the identifier passed to `GovernanceEngine.validate`. The previous
  text said ACGS records it, but acgs-lite's default engine does not store it.
- Docs: a new README section, "Data handling", states what the plugin and acgs-lite keep, return and log about a
  tool call. A test pins the claims about the default engine and the logs.
- Tested against google-adk 2.12.0 (Python 3.11 to 3.14) and 2.11.0 (Python 3.10, which ADK 2.12 no longer
  supports), plus the google-adk 2.0.0 and acgs-lite 3.0.1 minimums. No code changes.

## 0.1.0 (2026-10-21)

### Added

- `ACGSPlugin`, a Google ADK plugin whose `before_tool_callback` checks each tool call against an ACGS
  constitution. Denied calls return `{"error": "Tool call '<name>' was not executed. <reason>"}` as the tool
  result, so the tool does not run and the model sees the reason.
- `on_review` (`"confirm"`, `"deny"` or a sync or async callable) for calls that only match review rules.
  `"confirm"` (the default) uses ADK tool confirmation (`ToolContext.request_confirmation`) and runs the call only
  once it is confirmed; a blocking rule still denies a confirmed call.
- `ACGSPlugin.as_before_tool_callback()`, the same check as an agent-level `before_tool_callback`.
- Fail-closed handling: any error while evaluating or reviewing a call denies it instead of ending the run.
- `format_tool_call`, the documented text representation of a tool call that ACGS rules are evaluated against.
