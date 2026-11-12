# SPDX-License-Identifier: Apache-2.0
"""Carry opt-in OPD events through v0.30's typed per-request metrics codec."""

from dataclasses import dataclass, fields
from functools import wraps
from typing import Any

from vllm.v1.metrics.stats import RequestSpecDecodeMetrics


@dataclass
class _OPDResponseFields:
    opd_trace: dict[str, Any] | None = None


def _apply_patch():
    metrics_type = RequestSpecDecodeMetrics
    if getattr(metrics_type, "_opd_trace_patch_applied", False):
        return

    # Keep the original class identity used by EngineCoreOutput's annotations.
    # msgspec inspects dataclass fields; a dynamic instance attribute is dropped.
    trace_field = fields(_OPDResponseFields)[0]
    metrics_type.__annotations__ = {**metrics_type.__annotations__, "opd_trace": trace_field.type}
    metrics_type.__dataclass_fields__ = {**metrics_type.__dataclass_fields__, "opd_trace": trace_field}
    metrics_type.opd_trace = None
    original_init = metrics_type.__init__
    original_to_dict = metrics_type.to_dict

    @wraps(original_init)
    def initialize(self, *args, opd_trace=None, **kwargs):
        original_init(self, *args, **kwargs)
        self.opd_trace = opd_trace

    @wraps(original_to_dict)
    def to_dict(self):
        result = original_to_dict(self)
        if self.opd_trace is not None:
            result["opd_trace"] = self.opd_trace
        return result

    metrics_type.__init__ = initialize
    metrics_type.to_dict = to_dict
    metrics_type._opd_trace_patch_applied = True


_apply_patch()
