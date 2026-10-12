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

"""The Qwen4Exp aux hidden-state wiring on the model classes.

Unlike ``test_aux_hidden.py``, which exercises the reader in isolation, these
need the real upstream model module.
"""

import inspect

import pytest
from vllm.model_executor.models.interfaces import supports_eagle3

from vllm_ascend.models.qwen4_exp import model as ascend_model
from vllm_ascend.models.qwen4_exp.model import (
    AscendQwen4ExpForCausalLM,
    AscendQwen4ExpForConditionalGeneration,
    AscendQwen4ExpModel,
)


def upstream_backbone_cls():
    """The upstream class, which this module replaces by module global."""
    return AscendQwen4ExpModel.__mro__[1]


def test_forward_keeps_the_upstream_parameter_names():
    """``_mark_dynamic_inputs`` finds tensors by parameter name.

    A ``*args, **kwargs`` signature would drop every dynamic-dim mark, which
    statically specializes the token dimension and would also degrade plain
    inference with no aux layers configured.
    """
    ours = inspect.signature(AscendQwen4ExpModel.forward)
    theirs = inspect.signature(upstream_backbone_cls().forward)
    assert list(ours.parameters) == list(theirs.parameters)
    for name, their_param in theirs.parameters.items():
        assert ours.parameters[name].kind == their_param.kind
        assert ours.parameters[name].default == their_param.default


def test_forward_has_no_var_positional_or_keyword():
    kinds = {param.kind for param in inspect.signature(AscendQwen4ExpModel.forward).parameters.values()}
    assert inspect.Parameter.VAR_POSITIONAL not in kinds
    assert inspect.Parameter.VAR_KEYWORD not in kinds


@pytest.mark.parametrize(
    "cls", [AscendQwen4ExpForCausalLM, AscendQwen4ExpForConditionalGeneration]
)
def test_models_satisfy_the_eagle3_protocol(cls):
    """``supports_eagle3`` is a runtime_checkable isinstance test."""
    fake = object.__new__(cls)
    assert supports_eagle3(fake)


@pytest.mark.parametrize(
    "cls", [AscendQwen4ExpForCausalLM, AscendQwen4ExpForConditionalGeneration]
)
def test_no_default_aux_layers_is_an_explicit_refusal(cls):
    fake = object.__new__(cls)
    with pytest.raises(NotImplementedError, match="no default aux hidden-state layers"):
        fake.get_eagle3_default_aux_hidden_state_layers()


class FakePPGroup:
    def __init__(self, world_size: int) -> None:
        self.world_size = world_size


def test_pipeline_parallel_is_refused(monkeypatch):
    """PP>1 has two independent blockers, so refuse it outright.

    Layers outside the rank are ``PPMissingLayer`` with no
    ``attn_hyper_connection``, and the Eagle3 PP aux patch keys off
    ``EagleModelMixin`` to replace this model's forward with one written for a
    plain residual/norm backbone.
    """
    monkeypatch.setattr(ascend_model, "get_pp_group", lambda: FakePPGroup(2))
    backbone = object.__new__(AscendQwen4ExpModel)
    with pytest.raises(NotImplementedError, match="pipeline_parallel_size=1"):
        AscendQwen4ExpModel._set_aux_hidden_state_layers(backbone, (4, 16))


def test_pipeline_parallel_refusal_ignores_an_empty_request(monkeypatch):
    """Clearing the aux layers must work on any topology."""
    monkeypatch.setattr(ascend_model, "get_pp_group", lambda: FakePPGroup(2))
    backbone = object.__new__(AscendQwen4ExpModel)
    recorded = {}

    class FakeReader:
        def bind(self, layers, layer_ids):
            recorded["ids"] = tuple(layer_ids)

    backbone._aux_reader = FakeReader()
    backbone.layers = []
    backbone.start_layer = 0
    AscendQwen4ExpModel._set_aux_hidden_state_layers(backbone, ())
    assert recorded["ids"] == ()
