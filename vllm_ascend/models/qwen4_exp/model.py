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

"""Ascend wrappers for the upstream Qwen4Exp model implementation."""

import torch
from vllm.config import VllmConfig
from vllm.distributed.parallel_state import get_pp_group
from vllm.model_executor.models.interfaces import EagleModelMixin
from vllm.models.qwen4_exp.amd import (
    hyperconnection as upstream_hc,
)
from vllm.sequence import IntermediateTensors

from vllm_ascend import envs

from .aux_hidden import AuxHiddenStateReader
from .ops import (
    grouped_gemma_rmsnorm,
    hc_combine,
    hc_combine_norm,
    hc_gate_mix,
    hc_silu,
)
from .ple import patch_upstream_ple_short_conv

# Keep the upstream model and weight loader authoritative. HyperConnection's
# CUDA glue functions are module globals, so replace them before importing the
# model classes that instantiate GatedResidual.
upstream_hc.grouped_gemma_rmsnorm = grouped_gemma_rmsnorm
upstream_hc.hc_combine = hc_combine
upstream_hc.hc_combine_norm = hc_combine_norm
upstream_hc.hc_gate_mix = hc_gate_mix
upstream_hc.hc_silu = hc_silu

# PLE's depthwise F.conv1d otherwise lowers to ACLop Conv2D when torch-npu
# internal formats are enabled. Scope the graph-safe option override to the
# upstream PLE custom op instead of changing it globally for the worker.
patch_upstream_ple_short_conv()

from vllm.models.qwen4_exp.amd import model as upstream_model  # noqa: E402

from .qsa import AscendQwen4ExpQSAAttention  # noqa: E402

upstream_model.Qwen4ExpQSAAttention = AscendQwen4ExpQSAAttention


class AscendQwen4ExpModel(upstream_model.Qwen4ExpModel, EagleModelMixin):
    """Qwen4Exp backbone with Ascend-safe graph shape annotations.

    Also serves auxiliary hidden states. Upstream Qwen4Exp implements only MTP
    and exposes no aux readout, so ``EagleModelMixin`` is mixed in here to hold
    the requested ids and this class adds the readout itself.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        # With max_num_seqs=1 the PLE query-start buffer has a static length of
        # two. Marking that length dynamic makes TorchDynamo reject the static
        # specialization used by PLE during FULL_DECODE_ONLY graph capture.
        dynamic_arg_dims = getattr(self, "_dynamic_arg_dims", None)
        if dynamic_arg_dims is not None:
            self._dynamic_arg_dims = {
                name: dims for name, dims in dynamic_arg_dims.items() if name != "query_start_loc"
            }
        self._aux_reader = AuxHiddenStateReader(
            self.config.hidden_size,
            self_check=envs.VLLM_ASCEND_QWEN4EXP_AUX_HIDDEN_SELF_CHECK,
        )

    def _set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        if layers and get_pp_group().world_size > 1:
            # Two separate blockers. Layers outside this rank are PPMissingLayer
            # and own no attn_hyper_connection to read, and the Eagle3 PP aux
            # patch would swap this model's forward for one written against a
            # plain residual/norm backbone, discarding hyper-connection entirely.
            raise NotImplementedError(
                "Qwen4Exp auxiliary hidden states require pipeline_parallel_size=1; "
                f"got {get_pp_group().world_size}"
            )
        super()._set_aux_hidden_state_layers(layers)
        self._aux_reader.bind(self.layers, self.aux_hidden_state_layers)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        query_start_loc: torch.Tensor | None = None,
        ngram_context: torch.Tensor | None = None,
        deepstack_input_embeds: IntermediateTensors | None = None,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
        # The parameter names matter: upstream's _mark_dynamic_inputs locates
        # tensors such as input_ids and positions by name, so collapsing this
        # signature into *args/**kwargs would silently drop every dynamic-dim
        # mark and specialize the token dimension.
        if self.aux_hidden_state_layers:
            # Drop the previous step's readouts so a layer that fails to run
            # cannot serve a stale tensor.
            self._aux_reader.reset()
        hidden_states = super().forward(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
            query_start_loc,
            ngram_context,
            deepstack_input_embeds,
        )
        if not self.aux_hidden_state_layers:
            return hidden_states
        if isinstance(hidden_states, IntermediateTensors):
            # Only the last PP rank runs the final mixer. PP>1 is refused in
            # _set_aux_hidden_state_layers, so this is defensive only.
            return hidden_states
        aux_hidden_states = self._aux_reader.collect(hidden_states, self.aux_hidden_state_layers)
        return hidden_states, aux_hidden_states


# Qwen4ExpForCausalLM resolves this module global when it creates the backbone.
upstream_model.Qwen4ExpModel = AscendQwen4ExpModel


class _Qwen4ExpEagle3Mixin:
    """Aux hidden-state plumbing shared by the causal and multimodal wrappers.

    ``SupportsEagle3`` is a ``runtime_checkable`` Protocol and ``supports_eagle3()``
    tests it with ``isinstance``, so structural members are enough. Declaring
    them on a plain mixin keeps the Protocol out of the MRO, where its
    ``__init__`` would otherwise shadow ``nn.Module.__init__``, and lets this
    mixin precede the upstream class so its ``compute_logits`` override runs.
    """

    supports_eagle3 = True
    has_own_lm_head = False
    has_own_embed_tokens = False

    def get_eagle3_default_aux_hidden_state_layers(self) -> tuple[int, ...]:
        # Qwen4Exp has no published default tap set. Capture and drafting both
        # name their ids explicitly, so refuse to guess one here.
        raise NotImplementedError(
            "Qwen4Exp has no default aux hidden-state layers; set "
            "eagle_aux_hidden_state_layer_ids (capture) or target_layer_ids (draft)"
        )

    def _aux_backbone(self) -> "AscendQwen4ExpModel | None":
        """Resolve the backbone the way SupportsEagle3's default does."""
        parent = self
        if hasattr(self, "get_language_model"):
            parent = self.get_language_model()
        elif hasattr(self, "language_model"):
            parent = self.language_model
        backbone = getattr(parent, "model", parent)
        return backbone if isinstance(backbone, AscendQwen4ExpModel) else None

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        backbone = self._aux_backbone()
        if backbone is None:
            raise RuntimeError("Qwen4Exp aux hidden states need the Ascend backbone wrapper")
        backbone._set_aux_hidden_state_layers(layers)


class AscendQwen4ExpForCausalLM(_Qwen4ExpEagle3Mixin, upstream_model.Qwen4ExpForCausalLM):
    """Qwen3.8-Flash-Next text model using Ascend platform operators."""


class AscendQwen4ExpForConditionalGeneration(
    _Qwen4ExpEagle3Mixin, upstream_model.Qwen4ExpForConditionalGeneration
):
    """Qwen3.8-Flash-Next multimodal model using Ascend operators."""


__all__ = [
    "AscendQwen4ExpForCausalLM",
    "AscendQwen4ExpForConditionalGeneration",
    "AscendQwen4ExpModel",
]
