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

"""The Qwen4Exp auxiliary hidden-state readout against the training contract.

The contract comes from SpecForge ``specforge/offline_capture/qwen4_exp.py``:
tap ``i`` keeps ``layers[i + 1].attn_hyper_connection(streams)[0]``, the single
gated readout layer ``i + 1``'s attention consumes. vLLM numbers that state as
aux id ``i + 1`` and returns it from ``mix`` at index **1**, not 0.
"""

import pytest
import torch

from vllm_ascend.models.qwen4_exp.aux_hidden import AuxHiddenStateReader

HIDDEN_SIZE = 8
HC_COUNT = 4


class FakeGatedResidual:
    """Mimics ``GatedResidual``'s ``(streams, readout, injection)`` contract."""

    def __init__(self) -> None:
        self.readouts: list[torch.Tensor] = []

    def _readout(self, streams: torch.Tensor) -> torch.Tensor:
        # Stand-in for hc_norm -> low-rank mix -> hc_gate_mix: folds the
        # hc_count streams down to one hidden_size-wide vector.
        folded = streams.view(*streams.shape[:-1], HC_COUNT, HIDDEN_SIZE).mean(-2)
        self.readouts.append(folded)
        return folded

    def mix(self, streams: torch.Tensor):
        return streams, self._readout(streams), None

    def combine_and_mix(self, streams: torch.Tensor, prev_block_output, prev_injection):
        return streams, self._readout(streams), None


class FakeAttention:
    def __init__(self) -> None:
        self.seen: list[torch.Tensor] = []

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self.seen.append(hidden_states)
        return hidden_states


class FakeLayer:
    def __init__(self, *, ple: bool = False) -> None:
        self.attn_hyper_connection = FakeGatedResidual()
        self.linear_attn = FakeAttention()
        self.ple = object() if ple else None

    def run(self, streams: torch.Tensor, *, fused: bool) -> torch.Tensor:
        """Mirror the upstream layer body: mix, then feed attention."""
        if fused:
            _, block_input, _ = self.attn_hyper_connection.combine_and_mix(streams, None, None)
        else:
            _, block_input, _ = self.attn_hyper_connection.mix(streams)
        return self.linear_attn.forward(hidden_states=block_input)


def make_layers(count: int = 6) -> list[FakeLayer]:
    # Flash-Next puts PLE on layer 1, which is why SpecForge refuses the tap
    # whose reader is a PLE layer.
    return [FakeLayer(ple=(index == 1)) for index in range(count)]


def streams(tokens: int = 3) -> torch.Tensor:
    return torch.randn(tokens, HC_COUNT * HIDDEN_SIZE)


@pytest.mark.parametrize("fused", [False, True])
def test_readout_is_the_single_stream_not_the_residual(fused):
    """The captured value is the fold, never the hc_count-wide input.

    Capturing index 0 would hand the training side a 4x-wide tensor, which its
    loader silently averages over the streams instead of rejecting.
    """
    layers = make_layers()
    reader = AuxHiddenStateReader(HIDDEN_SIZE)
    reader.bind(layers, [2, 4])

    data = streams()
    layers[2].run(data, fused=fused)
    layers[4].run(data, fused=fused)

    captured = reader.collect(torch.zeros(3, HIDDEN_SIZE), [2, 4])
    assert [value.shape[-1] for value in captured] == [HIDDEN_SIZE, HIDDEN_SIZE]
    assert captured[0] is layers[2].attn_hyper_connection.readouts[-1]
    assert captured[1] is layers[4].attn_hyper_connection.readouts[-1]


def test_captured_tensor_is_what_attention_consumes():
    layers = make_layers()
    reader = AuxHiddenStateReader(HIDDEN_SIZE)
    reader.bind(layers, [3])

    layers[3].run(streams(), fused=False)

    (captured,) = reader.collect(torch.zeros(3, HIDDEN_SIZE), [3])
    assert captured is layers[3].linear_attn.seen[-1]


def test_ids_are_served_in_ascending_order_with_the_mixer_last():
    """SpecForge concatenates taps in ascending id order; the last id is the mixer."""
    layers = make_layers()
    reader = AuxHiddenStateReader(HIDDEN_SIZE)
    num_layers = len(layers)
    reader.bind(layers, [4, 2, num_layers])
    assert reader.layer_ids == (2, 4)

    layers[2].run(streams(), fused=False)
    layers[4].run(streams(), fused=False)
    final = torch.zeros(3, HIDDEN_SIZE)

    captured = reader.collect(final, [4, 2, num_layers])
    assert captured[0] is layers[2].attn_hyper_connection.readouts[-1]
    assert captured[1] is layers[4].attn_hyper_connection.readouts[-1]
    assert captured[2] is final


def test_ple_reader_is_refused():
    layers = make_layers()
    with pytest.raises(ValueError, match="PLE"):
        AuxHiddenStateReader(HIDDEN_SIZE).bind(layers, [1])


def test_out_of_range_id_is_refused():
    layers = make_layers()
    with pytest.raises(ValueError, match="out of range"):
        AuxHiddenStateReader(HIDDEN_SIZE).bind(layers, [len(layers) + 1])


def test_sharded_readout_is_refused():
    """A per-rank readout would disagree with the training feature width."""
    layers = make_layers()
    reader = AuxHiddenStateReader(HIDDEN_SIZE * 2)
    reader.bind(layers, [2])
    with pytest.raises(RuntimeError, match="expected the full hidden size"):
        layers[2].run(streams(), fused=False)


def test_missing_readout_is_refused():
    layers = make_layers()
    reader = AuxHiddenStateReader(HIDDEN_SIZE)
    reader.bind(layers, [2, 4])
    layers[2].run(streams(), fused=False)
    with pytest.raises(RuntimeError, match=r"aux layer ids \[4\]"):
        reader.collect(torch.zeros(3, HIDDEN_SIZE), [2, 4])


def test_self_check_catches_a_replaced_attention_input():
    layers = make_layers()
    reader = AuxHiddenStateReader(HIDDEN_SIZE, self_check=True)
    reader.bind(layers, [2])

    layer = layers[2]
    _, block_input, _ = layer.attn_hyper_connection.mix(streams())
    with pytest.raises(RuntimeError, match="not the tensor its attention"):
        layer.linear_attn.forward(hidden_states=block_input.clone())

    # The unmodified tensor still passes.
    assert layer.linear_attn.forward(hidden_states=block_input) is block_input


def test_mixer_slot_is_the_backbone_return_value():
    """The mixer id must carry the backbone's own output object.

    The identity belongs here, not at the logits head: the runner calls
    compute_logits with ``hidden_states[logits_indices]``, and that index
    always allocates a new tensor, so an ``is`` check there would reject
    every normal request.
    """
    layers = make_layers()
    reader = AuxHiddenStateReader(HIDDEN_SIZE, self_check=True)
    num_layers = len(layers)
    reader.bind(layers, [2, num_layers])

    layers[2].run(streams(), fused=False)
    final = torch.randn(3, HIDDEN_SIZE)
    captured = reader.collect(final, [2, num_layers])
    assert captured[-1] is final

    # A token-filtered view of the same values is a different object, and the
    # capture contract says nothing about it.
    filtered = final[torch.tensor([0, 2])]
    assert filtered is not final
    assert torch.equal(filtered, final[torch.tensor([0, 2])])


@pytest.mark.parametrize("fused", [False, True])
def test_self_check_passes_a_normal_step_including_token_filtering(fused):
    """Self-check must not fire on a healthy forward.

    Regression for an earlier version that asserted the logits input was the
    full backbone output. The runner filters tokens before the logits head, so
    that check rejected every normal request once self-check was on.
    """
    layers = make_layers()
    reader = AuxHiddenStateReader(HIDDEN_SIZE, self_check=True)
    num_layers = len(layers)
    reader.bind(layers, [2, 4, num_layers])

    for _ in range(2):  # two steps: stale readouts must not leak across them
        reader.reset()
        data = streams(4)
        layers[2].run(data, fused=fused)
        layers[4].run(data, fused=fused)
        final = torch.randn(4, HIDDEN_SIZE)

        captured = reader.collect(final, [2, 4, num_layers])
        assert len(captured) == 3
        assert captured[-1] is final

        # What the logits head actually receives, which the contract ignores.
        logits_input = final[torch.tensor([1, 3])]
        assert logits_input.shape == (2, HIDDEN_SIZE)


def test_unbind_restores_every_wrapped_method():
    layers = make_layers()
    original_mix = layers[2].attn_hyper_connection.mix
    original_forward = layers[2].linear_attn.forward

    reader = AuxHiddenStateReader(HIDDEN_SIZE, self_check=True)
    reader.bind(layers, [2])
    assert layers[2].attn_hyper_connection.mix is not original_mix

    reader.unbind()
    assert layers[2].attn_hyper_connection.mix == original_mix
    assert layers[2].linear_attn.forward == original_forward
    assert reader.layer_ids == ()
