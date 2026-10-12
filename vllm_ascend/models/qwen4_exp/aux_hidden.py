# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Auxiliary hidden-state readout for Qwen4Exp (Qwen3.8-Flash-Next).

An aux layer id ``k`` names *the input of layer k*, so aux id ``k`` reads the
gated residual readout that ``layers[k].attn_hyper_connection`` produces and
``layers[k]``'s attention then consumes. The SpecForge training side defines
the same state as tap ``k - 1`` (`specforge/offline_capture/qwen4_exp.py`,
``hyper_connection_taps``), which hooks ``layers[i + 1].attn_hyper_connection``
and keeps ``output[0]``. Aux id ``num_hidden_layers`` instead names the
``hyper_connection_mixer`` output, which is what ``lm_head`` consumes; this
model has no final norm, so that is the backbone's own return value.

**The tuple index differs between the two frameworks.** Transformers'
``Qwen4ExpTextGatedResidual.forward`` returns the single-stream readout at
``[0]``; vLLM's ``GatedResidual.mix``/``combine_and_mix`` return
``(hidden_states, block_input, injection)``, where ``[0]`` is the
``hc_count``-wide residual passed straight through and ``[1]`` is the readout.
Reading ``[0]`` here would capture the raw ``hc_count * hidden_size`` streams,
which the training side's ``_normalize_hidden_states`` then *silently averages
over the streams* - a quiet accuracy loss rather than an error. Always take
``[1]``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any

import torch
from vllm.logger import logger

# ``GatedResidual.mix`` and ``combine_and_mix`` both return
# ``(hidden_states, block_input, injection)``. The gated readout is at index 1;
# index 0 is the untouched multi-stream residual. See the module docstring.
_BLOCK_INPUT_INDEX = 1


class AuxHiddenStateReader:
    """Collect one gated readout per aux layer id during a forward pass.

    The readout is captured by wrapping the bound ``mix`` and ``combine_and_mix``
    methods of the selected layers' own ``attn_hyper_connection`` modules. The
    wrapper only reads the value the layer computes anyway, so it adds no
    arithmetic and cannot change what attention consumes.
    """

    def __init__(self, hidden_size: int, *, self_check: bool = False) -> None:
        self._hidden_size = hidden_size
        self._self_check = self_check
        self._sink: dict[int, torch.Tensor] = {}
        self._wrapped: list[tuple[Any, str, Callable[..., Any]]] = []
        self._layer_ids: tuple[int, ...] = ()
        self._last_final: torch.Tensor | None = None

    @property
    def layer_ids(self) -> tuple[int, ...]:
        """Aux ids served by a wrapped hyper-connection, in ascending order."""
        return self._layer_ids

    def reset(self) -> None:
        self._sink.clear()
        self._last_final = None

    def bind(self, layers: Sequence[Any], layer_ids: Iterable[int]) -> None:
        """Wrap ``layers[k].attn_hyper_connection`` for each requested id ``k``.

        Ids equal to ``len(layers)`` are not bound here: that id names the final
        mixer output, which the backbone returns directly.
        """
        self.unbind()
        bound: list[int] = []
        for layer_id in sorted(set(layer_ids)):
            if layer_id == len(layers):
                # The final mixer output is the backbone's return value.
                continue
            if not 0 <= layer_id < len(layers):
                raise ValueError(
                    f"aux layer id {layer_id} is out of range for a {len(layers)}-layer model"
                )
            layer = layers[layer_id]
            hyper_connection = getattr(layer, "attn_hyper_connection", None)
            if hyper_connection is None:
                raise ValueError(f"layer {layer_id} has no attn_hyper_connection to read")
            if getattr(layer, "ple", None) is not None:
                # PLE joins the streams before this read, so the readout would
                # carry layer k's own n-gram injection instead of layer k-1's
                # output. SpecForge refuses such taps for the same reason.
                raise ValueError(
                    f"aux layer id {layer_id} is a PLE layer; its readout is not "
                    "the preceding layer's output"
                )
            self._wrap(hyper_connection, layer_id)
            if self._self_check:
                self._wrap_attention_check(layer, layer_id)
            bound.append(layer_id)
        self._layer_ids = tuple(bound)
        logger.info(
            "Qwen4Exp aux hidden readout bound to layers %s (self_check=%s)",
            self._layer_ids,
            self._self_check,
        )

    def unbind(self) -> None:
        """Restore every wrapped method, newest first."""
        for owner, name, original in reversed(self._wrapped):
            setattr(owner, name, original)
        self._wrapped.clear()
        self._layer_ids = ()
        self.reset()

    def _wrap(self, hyper_connection: Any, layer_id: int) -> None:
        for name in ("mix", "combine_and_mix"):
            original = getattr(hyper_connection, name)
            self._wrapped.append((hyper_connection, name, original))
            setattr(hyper_connection, name, self._make_recorder(original, layer_id))

    def _make_recorder(self, original: Callable[..., Any], layer_id: int) -> Callable[..., Any]:
        def recorder(*args: Any, **kwargs: Any) -> Any:
            result = original(*args, **kwargs)
            readout = result[_BLOCK_INPUT_INDEX]
            if readout.shape[-1] != self._hidden_size:
                # A sharded readout would make the captured feature disagree
                # with the training side, which expects the full hidden size.
                raise RuntimeError(
                    f"aux layer id {layer_id} readout has width {readout.shape[-1]}, "
                    f"expected the full hidden size {self._hidden_size}"
                )
            self._sink[layer_id] = readout
            return result

        return recorder

    def _wrap_attention_check(self, layer: Any, layer_id: int) -> None:
        """Assert the captured readout is the tensor this layer's attention gets.

        ``is`` is a stronger statement than value equality: the captured object
        must be the very tensor handed to attention, so the check fails loudly
        if a future upstream layer body inserts anything between the gated read
        and the attention call. Only installed under self-check, because it
        overrides a module's bound ``forward``.
        """
        for name in ("linear_attn", "self_attn"):
            attention = getattr(layer, name, None)
            if attention is None:
                continue
            original = attention.forward
            self._wrapped.append((attention, "forward", original))
            setattr(attention, "forward", self._make_attention_check(original, layer_id))

    def _make_attention_check(self, original: Callable[..., Any], layer_id: int) -> Callable[..., Any]:
        def checked(*args: Any, **kwargs: Any) -> Any:
            consumed = kwargs.get("hidden_states")
            if consumed is None and args:
                consumed = args[0]
            captured = self._sink.get(layer_id)
            if captured is None:
                raise RuntimeError(f"aux layer id {layer_id} produced no readout before attention")
            if consumed is not captured:
                raise RuntimeError(
                    f"aux layer id {layer_id} readout is not the tensor its attention "
                    "consumes; the capture point no longer matches the model"
                )
            return original(*args, **kwargs)

        return checked

    def collect(self, final_hidden_states: torch.Tensor, aux_layer_ids: Sequence[int]) -> list[torch.Tensor]:
        """Return the readouts for ``aux_layer_ids`` in ascending id order.

        ``final_hidden_states`` serves the id equal to ``num_hidden_layers``.
        """
        self._last_final = final_hidden_states
        ordered: list[torch.Tensor] = []
        mixer_slots: list[int] = []
        missing: list[int] = []
        for layer_id in sorted(set(aux_layer_ids)):
            if layer_id in self._layer_ids:
                readout = self._sink.get(layer_id)
                if readout is None:
                    missing.append(layer_id)
                    continue
                ordered.append(readout)
            else:
                # The only id without a wrapped hyper-connection is the final
                # mixer output; bind() rejected every other out-of-range id.
                mixer_slots.append(len(ordered))
                ordered.append(final_hidden_states)
        if missing:
            raise RuntimeError(f"aux layer ids {missing} produced no readout during this forward")
        if self._self_check:
            # The identity to hold is between the mixer output and what we
            # return. The logits head is the wrong place to assert it: the
            # runner passes hidden_states[logits_indices], and that index
            # necessarily produces a new tensor.
            for slot in mixer_slots:
                if ordered[slot] is not final_hidden_states:
                    raise RuntimeError(
                        "the aux slot for the mixer output is not the backbone's "
                        "own return value, which lm_head consumes"
                    )
        return ordered


__all__ = ["AuxHiddenStateReader"]
