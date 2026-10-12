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

"""Let Qwen4Exp accept the ``extract_hidden_states`` speculative method.

Upstream's ``Qwen4ExpForConditionalGenerationConfig.verify_and_update_config``
allows only ``{"mtp", "ngram", "ngram_gpu"}`` and rejects everything else with
"Qwen4Exp speculative decoding supports only its native MTP checkpoint and
linear n-gram proposers". ``extract_hidden_states`` is not a proposer at all:
it runs no draft model and predicts nothing, it only routes the target's
auxiliary hidden states into a cache-only layer for the KV connector to write
out. The Ascend side already carries the pieces it needs - a proposer in
``vllm_ascend/spec_decode/extract_hidden_states_proposer.py`` and the hybrid
mamba-cache-mode exemption in ``patch_mamba_config.py``.

The allow-list is a set literal inside the function body, so it cannot be
extended in place. Since neither the parent ``Qwen3_5ForConditionalGenerationConfig``
nor the rest of the Qwen4Exp body reads ``speculative_config`` (only that final
check does), the method runs with the field detached and restores it afterwards.
That keeps every other Qwen4Exp validation - hc_count, PLE/QSA versus DBO and
microbatching, PLE versus PP>1, M-RoPE stripping - in force.

Upstream plan: propose adding ``extract_hidden_states`` to the allow-list, since
the rejection reason does not apply to a non-proposing capture method.
"""

import vllm
from vllm.logger import logger

_CAPTURE_METHOD = "extract_hidden_states"
_UPSTREAM_CONFIG_CLS = vllm.model_executor.models.config.Qwen4ExpForConditionalGenerationConfig
_original_verify_and_update_config = _UPSTREAM_CONFIG_CLS.verify_and_update_config


def verify_and_update_config(vllm_config) -> None:
    spec_config = vllm_config.speculative_config
    if spec_config is None or getattr(spec_config, "method", None) != _CAPTURE_METHOD:
        _original_verify_and_update_config(vllm_config)
        return

    logger.info(
        "Allowing speculative method '%s' for Qwen4Exp: it runs no draft model "
        "and only captures auxiliary hidden states.",
        _CAPTURE_METHOD,
    )
    # Detach only for the upstream call; every other validation still runs.
    vllm_config.speculative_config = None
    try:
        _original_verify_and_update_config(vllm_config)
    finally:
        vllm_config.speculative_config = spec_config


_UPSTREAM_CONFIG_CLS.verify_and_update_config = staticmethod(verify_and_update_config)

# Patching the conditional-generation config is enough for the whole family.
# Qwen4ExpForCausalLMConfig and Qwen4ExpMTPConfig do override the method, but
# each one calls Qwen4ExpForConditionalGenerationConfig.verify_and_update_config
# explicitly, and that attribute lookup resolves to the function installed
# above. Rebinding them as well would wrap this patch twice.
