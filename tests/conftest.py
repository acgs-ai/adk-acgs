# SPDX-FileCopyrightText: 2026-present ACGS Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Shared test setup: no telemetry export."""

from __future__ import annotations

import os

os.environ.setdefault("OTEL_SDK_DISABLED", "true")
