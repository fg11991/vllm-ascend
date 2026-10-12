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

"""The Qwen4Exp speculative-method allow-list exemption for hidden capture."""

from types import SimpleNamespace

import pytest

from vllm_ascend.patch.platform import patch_qwen4_exp_spec_methods as patch


@pytest.fixture
def recording_upstream(monkeypatch):
    """Replace the wrapped upstream call and record what it observed."""
    seen = []

    def fake_upstream(vllm_config):
        seen.append(getattr(vllm_config.speculative_config, "method", None))

    monkeypatch.setattr(patch, "_original_verify_and_update_config", fake_upstream)
    return seen


def make_config(method):
    spec = SimpleNamespace(method=method) if method is not None else None
    return SimpleNamespace(speculative_config=spec)


def test_capture_method_is_hidden_from_the_allow_list(recording_upstream):
    config = make_config("extract_hidden_states")
    original_spec = config.speculative_config

    patch.verify_and_update_config(config)

    # Upstream saw no speculative config, so its allow-list could not reject us.
    assert recording_upstream == [None]
    # And the caller still has its config afterwards.
    assert config.speculative_config is original_spec


@pytest.mark.parametrize("method", ["mtp", "ngram", "ngram_gpu", "dspark"])
def test_other_methods_reach_upstream_unchanged(recording_upstream, method):
    config = make_config(method)
    patch.verify_and_update_config(config)
    assert recording_upstream == [method]
    assert config.speculative_config.method == method


def test_no_speculative_config_reaches_upstream_unchanged(recording_upstream):
    config = make_config(None)
    patch.verify_and_update_config(config)
    assert recording_upstream == [None]
    assert config.speculative_config is None


def test_speculative_config_is_restored_even_when_upstream_raises(monkeypatch):
    def exploding_upstream(vllm_config):
        raise NotImplementedError("some other Qwen4Exp constraint")

    monkeypatch.setattr(patch, "_original_verify_and_update_config", exploding_upstream)
    config = make_config("extract_hidden_states")
    original_spec = config.speculative_config

    with pytest.raises(NotImplementedError, match="some other Qwen4Exp constraint"):
        patch.verify_and_update_config(config)
    assert config.speculative_config is original_spec


def test_patch_is_installed_on_the_upstream_class():
    installed = patch._UPSTREAM_CONFIG_CLS.verify_and_update_config
    assert installed is patch.verify_and_update_config
