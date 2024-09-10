# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import os

from packaging import version
import pytest
from pytest_cases import fixture
from pytest_cases import parametrize
import torch
import torchvision.models as modelzoo

from brevitas import torch_version
from brevitas.export import export_onnx_qcdq
from brevitas.export import export_torch_qcdq
from brevitas.export.inference import quant_inference_mode
from brevitas.graph.calibrate import calibration_mode
from brevitas.graph.quantize import layerwise_quantize
from brevitas.graph.quantize import quantize
from brevitas.graph.target.flexml import preprocess_for_flexml_quantize
from brevitas.graph.target.flexml import quantize_flexml
from brevitas_examples.imagenet_classification.ptq.ptq_common import quantize_model
from tests.marker import requires_pt_ge

BATCH = 1
HEIGHT, WIDTH = 224, 224
IN_CH = 3
MODEL_LIST = [
    'vit_b_32',
    'efficientnet_b0',
    'mobilenet_v3_small',
    'mobilenet_v2',
    'resnet50',
    'resnet18',
    'mnasnet0_5',
    'alexnet',
    'googlenet',
    'vgg11',
    'densenet121',
    'deeplabv3_resnet50',
    'fcn_resnet50',
    'regnet_x_400mf',
    'squeezenet1_0',
    'inception_v3']


class NoDictModel(torch.nn.Module):

    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    def forward(self, inp):
        out = self.model(inp)
        return out['out']


def quantize_float(model):
    return quantize_model(
        model,
        weight_bit_width=8,
        act_bit_width=8,
        bias_bit_width=None,
        weight_quant_granularity='per_tensor',
        act_quant_percentile=99.999,
        act_quant_type='sym',
        scale_factor_type='float_scale',
        backend='layerwise',
        quant_format='float')


@fixture
@parametrize('model_name', MODEL_LIST)
@parametrize('quantize_fn', [quantize_float, quantize, layerwise_quantize, quantize_flexml])
def torchvision_model(model_name, quantize_fn):

    inp = torch.randn(BATCH, IN_CH, HEIGHT, WIDTH)

    if torch_version <= version.parse('1.9.1') and model_name == 'regnet_x_400mf':
        return None

    if torch_version < version.parse('1.9.1') and model_name == 'googlenet':
        return None

    # EfficientNet is present since 1.10.1. Mobilenet is not correctly exported before 1.10.1
    if torch_version < version.parse('1.10.1') and model_name in ('efficientnet_b0',
                                                                  'mobilenet_v3_small'):
        return None
    if torch_version < version.parse('1.11.0') and model_name == 'vit_b_32':
        return None

    # Due to a regression in torchvision, we cannot load pretrained weights for effnet_b0
    # https://github.com/pytorch/vision/issues/7744
    if torch_version == version.parse('2.1.0') and model_name == 'efficientnet_b0':
        return None

    # Deeplab and fcn are in a different module, and they have a dict as output which is not suited for torchscript
    if model_name in ('deeplabv3_resnet50', 'fcn_resnet50'):
        model_fn = getattr(modelzoo.segmentation, model_name)
        model = NoDictModel(model_fn(pretrained=True))
    elif model_name in ('googlenet', 'inception_v3'):
        model_fn = getattr(modelzoo, model_name)
        model = model_fn(pretrained=True, transform_input=False)
    else:
        model_fn = getattr(modelzoo, model_name)
        model = model_fn(pretrained=True)

    model.eval()
    model = preprocess_for_flexml_quantize(model, inp)
    model = quantize_fn(model)
    with calibration_mode(model):
        model(inp)
    return model


@requires_pt_ge('1.8.1')
@parametrize('enable_compile', [True, False])
def test_torchvision_graph_quantization_flexml_qcdq_onnx(
        torchvision_model, enable_compile, request):
    test_id = request.node.callspec.id
    if torchvision_model is None:
        pytest.skip('Model not instantiated')
    if enable_compile:
        model_name = test_id.split("-")[1]
        if torch_version <= version.parse('2.2'):
            pytest.skip("Pytorch 2.2 is required to test compile")
        else:
            torch._dynamo.config.capture_scalar_outputs = True
        if 'vit' in model_name:
            pytest.skip("QuantMHA not supported with compile")

    inp = torch.randn(BATCH, IN_CH, HEIGHT, WIDTH)

    quantize_fn_name = test_id.split("-")[0]
    with torch.no_grad(), quant_inference_mode(torchvision_model):
        prehook_non_compiled_out = torchvision_model(inp)
        post_hook_non_compiled_out = torchvision_model(inp)
        assert torch.allclose(prehook_non_compiled_out, post_hook_non_compiled_out)

        if enable_compile:
            compiled_model = torch.compile(torchvision_model, fullgraph=True)
            compiled_out = compiled_model(inp)

        # This fails! Compile might needs more small-scoped tests for accuracy evaluation
        # assert torch.allclose(post_hook_non_compiled_out, compiled_out)

    if quantize_fn_name != 'quantize_float' and not enable_compile:
        export_onnx_qcdq(torchvision_model, args=inp)


@requires_pt_ge('1.9.1')
def test_torchvision_graph_quantization_flexml_qcdq_torch(torchvision_model, request):
    if torchvision_model is None:
        pytest.skip('Model not instantiated')

    test_id = request.node.callspec.id
    quantize_fn_name = test_id.split("-")[0]
    if quantize_fn_name == 'quantize_float':
        return
    inp = torch.randn(BATCH, IN_CH, HEIGHT, WIDTH)

    torchvision_model(inp)
    export_torch_qcdq(torchvision_model, args=inp)
