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

"""The shared-backing gate must keep six-region alive for hidden capture.

A KV transfer normally disqualifies a shared backing, because a connector may
own per-layer buffers. ExampleHiddenStatesConnector writes only the dedicated
hidden-state cache-only layer, so six-region must stay enabled - otherwise
every QSA state layer falls through to the generic reshape path and asks the
upstream backend for Ascend's own get_kv_cache_shape, which it lacks.
"""

from types import SimpleNamespace

import pytest


def shared_backing_is_supported(*, method, with_transfer):
    """Mirror the gate in NPUModelRunner.initialize_kv_cache_tensors.

    Kept as a standalone predicate because constructing a real NPUModelRunner
    needs a device; the production copy of this expression lives at the top of
    that method and the two must stay in sync.
    """
    spec_config = SimpleNamespace(method=method) if method is not None else None
    kv_transfer_config = SimpleNamespace() if with_transfer else None
    captures_hidden_states_only = (
        spec_config is not None and getattr(spec_config, "method", None) == "extract_hidden_states"
    )
    return kv_transfer_config is None or captures_hidden_states_only


def test_capture_keeps_the_shared_backing():
    assert shared_backing_is_supported(method="extract_hidden_states", with_transfer=True)


def test_plain_inference_keeps_the_shared_backing():
    assert shared_backing_is_supported(method=None, with_transfer=False)


@pytest.mark.parametrize("method", [None, "mtp", "dspark", "eagle3"])
def test_other_connectors_still_lose_the_shared_backing(method):
    """The exemption must stay narrow: a real KV-migrating connector keeps the
    conservative path."""
    assert not shared_backing_is_supported(method=method, with_transfer=True)


def test_capture_without_a_connector_is_unaffected():
    assert shared_backing_is_supported(method="extract_hidden_states", with_transfer=False)


def test_production_expression_matches_this_mirror():
    """Guard against the production gate drifting from the mirror above."""
    import pathlib
    import re

    source = pathlib.Path("vllm_ascend/worker/model_runner_v1.py").read_text()
    match = re.search(
        r"captures_hidden_states_only = \((.*?)\)\s*\n\s*supports_shared_backing_with_kv_transfer = \((.*?)\)",
        source,
        re.S,
    )
    assert match, "the shared-backing gate moved; update this test"
    detect, gate = match.group(1), match.group(2)
    assert "extract_hidden_states" in detect
    assert "kv_transfer_config is None" in gate
    assert "captures_hidden_states_only" in gate
