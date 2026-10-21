# SPDX-FileCopyrightText: 2026-present ACGS Contributors
#
# SPDX-License-Identifier: Apache-2.0
import adk_acgs


def test_public_api() -> None:
    assert sorted(adk_acgs.__all__) == sorted(
        [
            "PLUGIN_NAME",
            "ACGSPlugin",
            "BeforeToolCallback",
            "GovernanceDecision",
            "ReviewHandler",
            "ToolCallFormatter",
            "Verdict",
            "format_tool_call",
        ]
    )
