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

"""Hidden-state cache block sizing must ignore the QSA ring capacity.

A CircularBufferSpec's block_size is how many key rows the open compression
group holds, and its max_num_blocks_per_req() is 1, so it never indexes blocks
by token. Letting it into the token-indexed gcd collapses the hidden-state
cache to one block per 8 tokens and makes every other group pay that block
count out of the shared index space.
"""

import torch
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    HiddenStateCacheSpec,
    KVCacheGroupSpec,
    MambaSpec,
)

from vllm_ascend.patch.platform.patch_kv_cache_utils import _widen_hidden_state_cache_blocks

COMMON_PAGE = 786432
AUX_WIDTH = 6 * 2560  # six aux ids of hidden_size 2560


def hidden_group(block_size):
    spec = HiddenStateCacheSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=AUX_WIDTH,
        dtype=torch.bfloat16,
        page_size_padded=COMMON_PAGE,
    )
    return KVCacheGroupSpec(["cache_only_layers.48"], spec)


def qsa_main_group(block_size=768):
    spec = FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
    )
    return KVCacheGroupSpec(["layers.3.self_attn.attn"], spec)


def ring_group(capacity=8):
    spec = CircularBufferSpec(
        block_size=capacity,
        num_kv_heads=1,
        head_size=128,
        head_size_v=0,
        dtype=torch.bfloat16,
    )
    return KVCacheGroupSpec(["layers.3.self_attn.indexer.raw_key_cache"], spec)


def mamba_group(block_size=4608):
    spec = MambaSpec(
        shapes=((128,),),
        dtypes=(torch.bfloat16,),
        block_size=block_size,
    )
    return KVCacheGroupSpec(["layers.0.linear_attn"], spec)


def test_ring_capacity_does_not_cap_the_hidden_block_size():
    """gcd(768, 4608) = 768, and the page holds 25 tokens, so 24 is the answer."""
    groups = [qsa_main_group(), ring_group(8), mamba_group(), hidden_group(8)]
    widened = _widen_hidden_state_cache_blocks(groups)
    assert widened[-1].kv_cache_spec.block_size == 24
    # Everything else is untouched.
    assert widened[0].kv_cache_spec.block_size == 768
    assert widened[1].kv_cache_spec.block_size == 8
    assert widened[2].kv_cache_spec.block_size == 4608


def test_widening_is_bounded_by_what_the_page_holds():
    """24 tokens is the largest divisor of 768 that fits 786432 bytes."""
    spec = hidden_group(8).kv_cache_spec
    per_token = spec.num_kv_heads * spec.head_size * 2
    assert 24 * per_token <= COMMON_PAGE
    assert 32 * per_token > COMMON_PAGE  # the next divisor of 768 overflows


def test_already_wide_enough_is_left_alone():
    groups = [qsa_main_group(), ring_group(8), mamba_group(), hidden_group(24)]
    widened = _widen_hidden_state_cache_blocks(groups)
    assert widened[-1].kv_cache_spec.block_size == 24
    assert widened[-1] is groups[-1]


def test_no_hidden_cache_is_a_no_op():
    groups = [qsa_main_group(), ring_group(8), mamba_group()]
    assert _widen_hidden_state_cache_blocks(groups) is groups


def test_no_token_indexed_group_is_a_no_op():
    groups = [ring_group(8), hidden_group(8)]
    assert _widen_hidden_state_cache_blocks(groups) is groups
