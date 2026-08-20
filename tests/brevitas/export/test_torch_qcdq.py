# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import torch

from brevitas.export import export_torch_qcdq
from brevitas.export.torch.qcdq.custom_ops import dequantize_fp8
from brevitas.export.torch.qcdq.custom_ops import dequantize_mx_fp4
from brevitas.export.torch.qcdq.custom_ops import dequantize_mx_fp8
from brevitas.export.torch.qcdq.custom_ops import quantize_fp8
from brevitas.export.torch.qcdq.custom_ops import quantize_mx_fp4
from brevitas.export.torch.qcdq.custom_ops import quantize_mx_fp8
from export_quant_linear import build_models
from export_quant_linear import capture_model
from export_quant_linear import IN_FEATURES
from tests.marker import jit_disabled_for_compile
from tests.marker import jit_disabled_for_export
from tests.marker import requires_pt_ge

from .quant_module_fixture import *


@jit_disabled_for_export()
@torch.no_grad()
def test_torch_qcdq_wbiol_export(
        quant_module,
        quant_module_impl,
        weight_act_quantizers,
        input_bit_width,
        weight_bit_width,
        output_bit_width,
        bias_bit_width,
        bias_quantizer):

    weight_act_quantizers_name, _ = weight_act_quantizers
    bias_quantizer_name, _ = bias_quantizer

    if 'asymmetric' in weight_act_quantizers_name and (input_bit_width > 8 or output_bit_width > 8
                                                       or weight_bit_width > 8):
        pytest.skip("Unsigned zero point supported on 8b or less.")
    if 'internal_scale' in bias_quantizer_name and bias_bit_width == 32:
        pytest.skip("This combination is prone to numerical errors as the scale gets too small.")

    if quant_module_impl == QuantLinear:
        in_size = (1, IN_CH)
    elif quant_module_impl == QuantConv1d or quant_module_impl == QuantConvTranspose1d:
        in_size = (1, IN_CH, FEATURES)
    elif quant_module_impl == QuantConv2d or quant_module_impl == QuantConvTranspose2d:
        in_size = (1, IN_CH, FEATURES, FEATURES)
    else:
        in_size = (1, IN_CH, FEATURES, FEATURES, FEATURES)

    inp = torch.randn(in_size)
    quant_module(inp)  # Collect scale factors
    quant_module.eval()
    inp = torch.randn(in_size) * IN_SCALE + IN_MEAN  # redefine inp for testing
    out = quant_module(inp)
    pytorch_qcdq_model = export_torch_qcdq(quant_module, args=inp)
    torchscript_out = pytorch_qcdq_model(inp)
    torchscript_out_value = torchscript_out[0]
    tolerance = TOLERANCE * out.scale
    del pytorch_qcdq_model
    assert torch.allclose(out, torchscript_out_value, atol=tolerance)


@jit_disabled_for_export()
@parametrize('input_signed', [True, False])
@torch.no_grad()
def test_torch_qcdq_avgpool_export(input_signed, output_bit_width):
    in_size = (1, IN_CH, FEATURES, FEATURES)
    inp = torch.randn(in_size)
    quant_module = nn.Sequential(
        QuantIdentity(signed=input_signed, return_quant_tensor=True),
        TruncAvgPool2d(kernel_size=3, stride=2, float_to_int_impl_type='round'))
    quant_module(inp)  # Collect scale factors
    quant_module.eval()
    inp = torch.randn(in_size) * IN_SCALE + IN_MEAN  # redefine inp for testing
    out = quant_module(inp)
    pytorch_qcdq_model = export_torch_qcdq(quant_module, args=inp)
    torchscript_out = pytorch_qcdq_model(inp)
    torchscript_out_value = torchscript_out[0]
    tolerance = TOLERANCE * out.scale
    del pytorch_qcdq_model
    assert torch.allclose(out, torchscript_out_value, atol=tolerance)


@requires_pt_ge('2.12')
@jit_disabled_for_export()
@pytest.mark.parametrize(
    'quant_format,expected_q,expected_dq',
    [(
        'int8',
        'quantized_decomposed.quantize_per_tensor',
        'quantized_decomposed.dequantize_per_tensor'),
     ('fp8', 'brevitas.quantize_fp8', 'brevitas.dequantize_fp8'),
     ('mxfp8', 'brevitas.quantize_mx_fp8', 'brevitas.dequantize_mx_fp8'),
     ('mxfp4', 'brevitas.quantize_mx_fp4', 'brevitas.dequantize_mx_fp4')])
@torch.no_grad()
def test_torch_export_quant_linear_qdq(quant_format, expected_q, expected_dq):
    inp = torch.randn(2, IN_FEATURES)
    model = build_models()[quant_format]
    model(inp)
    model.eval()
    expected = model(inp)

    exported_program = export_torch_qcdq(model, args=inp, dynamo=True)
    actual = exported_program.module()(inp)
    new_inp = torch.randn_like(inp)
    expected_new = model(new_inp)
    actual_new = exported_program.module()(new_inp)
    targets = [str(node.target) for node in exported_program.graph.nodes]

    assert sum(expected_q in target for target in targets) == 2
    assert sum(expected_dq in target for target in targets) == 2
    assert not any('_scaled_mm' in target for target in targets)
    assert torch.allclose(expected, actual, rtol=1e-4, atol=1e-4)
    assert torch.allclose(expected_new, actual_new, rtol=1e-4, atol=1e-4)


@requires_pt_ge('2.4')
@jit_disabled_for_compile()
@pytest.mark.parametrize('quant_format', ['int8', 'fp8', 'mxfp8', 'mxfp4'])
@torch.no_grad()
def test_torch_compile_quant_linear_qdq(quant_format):
    inp = torch.randn(2, IN_FEATURES)
    fresh_inp = torch.randn_like(inp)
    model = build_models()[quant_format]

    graph_module = capture_model(quant_format, model, inp, fresh_inp)

    assert graph_module is not None


@requires_pt_ge('2.4')
@jit_disabled_for_compile()
@torch.no_grad()
def test_torch_compile_mxfp4_non_divisible_group():
    inp = torch.randn(2, IN_FEATURES + 1)
    fresh_inp = torch.randn_like(inp)
    model = build_models(in_features=IN_FEATURES + 1)['mxfp4']

    graph_module = capture_model('mxfp4', model, inp, fresh_inp)

    assert graph_module is not None


@requires_pt_ge('2.12')
@jit_disabled_for_export()
@torch.no_grad()
def test_torch_export_mxfp4_non_divisible_group():
    inp = torch.randn(2, IN_FEATURES + 1)
    model = build_models(in_features=IN_FEATURES + 1)['mxfp4']
    model(inp)
    model.eval()
    expected = model(inp)

    exported_program = export_torch_qcdq(model, args=inp, dynamo=True)
    actual = exported_program.module()(inp)

    assert torch.allclose(expected, actual, rtol=1e-4, atol=1e-4)


@requires_pt_ge('2.12')
@jit_disabled_for_export()
@torch.no_grad()
def test_torch_export_restores_quant_linear_state():
    inp = torch.randn(2, IN_FEATURES)
    model = build_models()['mxfp4']
    model(inp)
    model.train()
    model.return_quant_tensor = True
    return_quant_tensor = model.return_quant_tensor
    skip_create_quant_tensor = {
        module: module.skip_create_quant_tensor
        for module in model.modules()
        if hasattr(module, 'skip_create_quant_tensor')}

    export_torch_qcdq(model, args=inp, dynamo=True)

    assert model.training is True
    assert model.return_quant_tensor == return_quant_tensor
    assert all(
        module.skip_create_quant_tensor == state for module,
        state in skip_create_quant_tensor.items())


@requires_pt_ge('2.4')
@torch.no_grad()
def test_mxfp4_custom_qdq_codes():
    positive = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.])
    values = torch.cat((positive, -positive[1:]))
    grouped_values = torch.zeros(1, 32)
    grouped_values[0, :values.numel()] = values
    scale = torch.ones(1, 1)

    quantized = quantize_mx_fp4(grouped_values, scale, 32, -1)
    dequantized = dequantize_mx_fp4(quantized, scale, 32, -1, torch.float32)

    expected_codes = torch.tensor(list(range(8)) + list(range(9, 16)), dtype=torch.uint8)
    assert torch.equal(quantized[0, :values.numel()], expected_codes)
    assert torch.equal(dequantized[0, :values.numel()], values)


@requires_pt_ge('2.4')
@torch.no_grad()
def test_mxfp8_custom_qdq():
    values = torch.linspace(-8., 8., 32).reshape(1, 1, 32)
    scale = torch.full((1, 1, 1), .5)

    quantized = quantize_mx_fp8(values, scale, 32, 1, torch.float8_e4m3fn, -448., 448.)
    dequantized = dequantize_mx_fp8(quantized, scale, 32, 1, torch.float32)
    expected = (values / scale).to(torch.float8_e4m3fn).to(torch.float32) * scale

    assert quantized.dtype == torch.float8_e4m3fn
    assert torch.equal(dequantized, expected)


@requires_pt_ge('2.4')
@torch.no_grad()
def test_fp8_custom_qdq_per_channel():
    values = torch.linspace(-8., 8., 32).reshape(2, 16)
    scale = torch.linspace(.25, 1., 16)

    quantized = quantize_fp8(values, scale, torch.float8_e4m3fn, 1, -448., 448.)
    dequantized = dequantize_fp8(quantized, scale, 1, torch.float32)
    expected = (values / scale).to(torch.float8_e4m3fn).to(torch.float32) * scale

    assert quantized.dtype == torch.float8_e4m3fn
    assert torch.equal(dequantized, expected)


@requires_pt_ge('2.4')
@torch.no_grad()
def test_mx_custom_qdq_negative_group_dim():
    values = torch.linspace(-8., 8., 192).reshape(2, 1, 32, 3)
    scale = torch.full((2, 1, 1, 3), .5)

    quantized_fp8 = quantize_mx_fp8(values, scale, 32, -2, torch.float8_e4m3fn, -448., 448.)
    dequantized_fp8 = dequantize_mx_fp8(quantized_fp8, scale, 32, -2, torch.float32)
    quantized_fp4 = quantize_mx_fp4(values, scale, 32, -2)
    dequantized_fp4 = dequantize_mx_fp4(quantized_fp4, scale, 32, -2, torch.float32)

    assert quantized_fp8.shape == values.shape
    assert dequantized_fp8.shape == values.shape
    assert quantized_fp4.shape == values.shape
    assert dequantized_fp4.shape == values.shape


@requires_pt_ge('2.12')
@jit_disabled_for_export()
@torch.no_grad()
def test_torch_export_restores_state_on_error(monkeypatch):
    inp = torch.randn(2, IN_FEATURES)
    model = build_models()['int8']
    model(inp)
    model.train()
    model.return_quant_tensor = True
    return_quant_tensor = model.return_quant_tensor
    skip_create_quant_tensor = {
        module: module.skip_create_quant_tensor
        for module in model.modules()
        if hasattr(module, 'skip_create_quant_tensor')}

    def export_error(*args, **kwargs):
        raise RuntimeError('export failed')

    monkeypatch.setattr(torch.export, 'export', export_error)
    with pytest.raises(RuntimeError, match='export failed'):
        export_torch_qcdq(model, args=inp, dynamo=True)

    assert model.training is True
    assert model.return_quant_tensor == return_quant_tensor
    assert all(
        module.skip_create_quant_tensor == state for module,
        state in skip_create_quant_tensor.items())
    assert all(
        not module.export_mode for module in model.modules() if hasattr(module, 'export_mode'))
