# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/). Entries for new releases are generated with `cz bump`.

## [Unreleased]

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
