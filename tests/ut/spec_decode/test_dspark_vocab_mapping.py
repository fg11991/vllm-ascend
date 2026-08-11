#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""Model runner v1: DSpark drafting against a reduced draft vocabulary.

The v1 proposer samples the block itself, so both halves of the token-space
contract live here: the Markov head works in *draft* space (its bias is
``draft_vocab_size`` wide), while every id written into the block buffer is a
*target* id -- ``markov_w1`` is indexed by the previous real token, so an
unmapped draft id is a valid row pointing at the wrong token.

Head derivation and mapping validation are shared with the v2 speculator and
tested in ``test_vocab_mapping.py``; what is pinned here is that v1 routes into
them, and that it reads the right ownership flag while doing so.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.spec_decode.llm_base_proposer import AscendSpecDecodeBaseProposer

_KEPT_TARGET_IDS = [1, 3, 6, 9]
_DRAFT_VOCAB_SIZE = len(_KEPT_TARGET_IDS)
_TARGET_VOCAB_SIZE = 10
_HIDDEN_SIZE = 3

_D2T = torch.tensor([kept - i for i, kept in enumerate(_KEPT_TARGET_IDS)], dtype=torch.long)


class _StubDSparkModel:
    """Minimal stand-in for ``Qwen3DSparkForCausalLM``'s drafting surface.

    ``markov_bias`` makes the next draft id a pure function of the id fed back
    in: ``draft_id = prev_token_id % draft_vocab_size``. Base logits are zero, so
    the argmax is decided entirely by the bias, which is what lets the test tell
    a target id from a draft id by looking only at the output.
    """

    def __init__(self, *, reduced_vocab: bool = True):
        self.vocab_size = _DRAFT_VOCAB_SIZE if reduced_vocab else _TARGET_VOCAB_SIZE
        self.draft_id_to_target_id = _D2T.clone() if reduced_vocab else None
        self.compute_logits_calls = 0
        self.compute_draft_logits_calls = 0
        self.markov_embed_inputs: list[torch.Tensor] = []

    def compute_draft_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self.compute_draft_logits_calls += 1
        return torch.zeros((hidden_states.shape[0], self.vocab_size))

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # The real one scatters a pruned draft's logits into the full target
        # vocabulary; the drafting path must not go through it. That scatter is
        # what made the Markov bias fail to broadcast (64000 vs 248320).
        self.compute_logits_calls += 1
        return torch.zeros((hidden_states.shape[0], _TARGET_VOCAB_SIZE))

    def markov_embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        self.markov_embed_inputs.append(token_ids.clone())
        return token_ids.to(torch.float32).unsqueeze(-1)

    def markov_bias(self, markov_embed: torch.Tensor) -> torch.Tensor:
        prev = markov_embed.squeeze(-1).to(torch.long)
        bias = torch.zeros((prev.shape[0], self.vocab_size))
        bias.scatter_(1, (prev % self.vocab_size).unsqueeze(-1), 1000.0)
        return bias

    def map_draft_to_target(self, draft_ids: torch.Tensor) -> torch.Tensor:
        if self.draft_id_to_target_id is None:
            return draft_ids
        return draft_ids + self.draft_id_to_target_id[draft_ids]


def _make_proposer(model, *, num_speculative_tokens: int, seed_token_id: int, num_blocks: int = 1):
    """Bypass ``__init__``; set only what the DSpark sampling path reads."""
    proposer = AscendSpecDecodeBaseProposer.__new__(AscendSpecDecodeBaseProposer)
    proposer.model = model
    proposer.num_speculative_tokens = num_speculative_tokens
    proposer._dspark_draft_buffer = torch.zeros((num_blocks, 1 + num_speculative_tokens), dtype=torch.int64)
    proposer._dspark_seed_buffer = torch.full((num_blocks,), seed_token_id, dtype=torch.int64)
    return proposer


@pytest.fixture(autouse=True)
def _stub_forward_context(monkeypatch):
    """``_disable_flash_comm_v1_context`` needs a forward context to toggle."""
    monkeypatch.setattr(
        "vllm_ascend.spec_decode.utils.get_forward_context",
        lambda: SimpleNamespace(flash_comm_v1_enabled=False),
    )


class TestDSparkSampleBlockRemapsIds:
    """The block buffer must carry target ids at every step, not draft ids."""

    def test_sampled_ids_are_target_ids_and_feed_back_as_target_ids(self):
        # seed 5 -> draft 5%4=1 -> target 3 -> draft 3%4=3 -> target 9
        #        -> draft 9%4=1 -> target 3
        num_spec = 3
        model = _StubDSparkModel()
        proposer = _make_proposer(model, num_speculative_tokens=num_spec, seed_token_id=5)

        block = proposer._dspark_sample_block(torch.zeros((num_spec, _HIDDEN_SIZE)))

        assert block[0].tolist() == [5, 3, 9, 3]
        # Every Markov step is fed the *target* id of the previous position.
        assert [int(t[0]) for t in model.markov_embed_inputs] == [5, 3, 9]

    def test_does_not_go_through_the_target_vocab_scatter(self):
        """The scatter is what the reported crash came from: a K-wide bias added
        onto V-wide logits cannot broadcast."""
        num_spec = 2
        model = _StubDSparkModel()
        proposer = _make_proposer(model, num_speculative_tokens=num_spec, seed_token_id=5)

        proposer._dspark_sample_block(torch.zeros((num_spec, _HIDDEN_SIZE)))

        assert model.compute_draft_logits_calls == 1
        assert model.compute_logits_calls == 0

    def test_full_vocabulary_draft_is_an_identity_remap(self):
        """No d2t -> the sampled draft id is already a target id."""
        num_spec = 3
        model = _StubDSparkModel(reduced_vocab=False)
        proposer = _make_proposer(model, num_speculative_tokens=num_spec, seed_token_id=5)

        block = proposer._dspark_sample_block(torch.zeros((num_spec, _HIDDEN_SIZE)))

        # bias picks prev % 10, and 5 % 10 == 5 is a fixed point.
        assert block[0].tolist() == [5, 5, 5, 5]

    def test_falls_back_to_compute_logits_without_compute_draft_logits(self):
        """DSV4 DSpark defines neither hook; behavior is unchanged for it."""

        class _FullVocabOnlyModel(_StubDSparkModel):
            def __getattribute__(self, name):
                if name in ("compute_draft_logits", "map_draft_to_target"):
                    raise AttributeError(name)
                return super().__getattribute__(name)

        num_spec = 2
        model = _FullVocabOnlyModel(reduced_vocab=False)
        proposer = _make_proposer(model, num_speculative_tokens=num_spec, seed_token_id=5)

        block = proposer._dspark_sample_block(torch.zeros((num_spec, _HIDDEN_SIZE)))

        assert model.compute_logits_calls == 1
        assert block[0].tolist() == [5, 5, 5]


class TestMaybeShareLMHeadRouting:
    """Which lm_head outcome each checkpoint shape lands on, in v1."""

    @staticmethod
    def _make_proposer(monkeypatch, *, draft_id_to_target_id, has_own_lm_head):
        proposer = AscendSpecDecodeBaseProposer.__new__(AscendSpecDecodeBaseProposer)
        proposer.method = "dspark"
        proposer.model = SimpleNamespace(
            lm_head="draft-head",
            draft_id_to_target_id=draft_id_to_target_id,
            has_own_lm_head=has_own_lm_head,
        )
        proposer.vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(is_deepseek_mla=False, get_vocab_size=lambda: _TARGET_VOCAB_SIZE),
            compilation_config=SimpleNamespace(cudagraph_mode=SimpleNamespace(has_full_cudagraphs=lambda: False)),
        )
        proposer.use_cuda_graph = False
        settled: list[tuple] = []
        monkeypatch.setattr(
            "vllm_ascend.spec_decode.llm_base_proposer.settle_reduced_vocab_lm_head",
            lambda draft, target, vocab: settled.append((draft, target, vocab)),
        )
        return proposer, settled

    def test_reduced_vocabulary_is_settled_and_never_shared(self, monkeypatch):
        """The model sets has_own_lm_head=True to refuse the share, so routing
        must key off d2t rather than that flag or it would fall through."""
        proposer, settled = self._make_proposer(
            monkeypatch, draft_id_to_target_id=_D2T.clone(), has_own_lm_head=True
        )
        target = SimpleNamespace(lm_head="target-head")

        proposer._maybe_share_lm_head(target)

        assert len(settled) == 1
        assert settled[0][1] is target
        assert proposer.model.lm_head == "draft-head"

    def test_full_vocabulary_checkpoint_still_shares_the_target_head(self, monkeypatch):
        """The unpruned path, unchanged: no d2t, no own head -> share."""
        proposer, settled = self._make_proposer(monkeypatch, draft_id_to_target_id=None, has_own_lm_head=False)
        target = SimpleNamespace(lm_head="target-head")

        proposer._maybe_share_lm_head(target)

        assert settled == []
        assert proposer.model.lm_head == "target-head"

    def test_full_vocabulary_checkpoint_with_its_own_head_keeps_it(self, monkeypatch):
        proposer, settled = self._make_proposer(monkeypatch, draft_id_to_target_id=None, has_own_lm_head=True)
        target = SimpleNamespace(lm_head="target-head")

        proposer._maybe_share_lm_head(target)

        assert settled == []
        assert proposer.model.lm_head == "draft-head"
