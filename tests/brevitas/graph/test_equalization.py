# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import copy

import torch
from torchvision import models

from brevitas.fx import symbolic_trace
from brevitas.graph.equalize import _extract_regions
from brevitas.graph.equalize import _is_supported_module
from brevitas.graph.utils import get_module

from .equalization_fixtures import *


def test_load_state_dict_equalizedbn():
    model = models.resnet18(pretrained=True)

    torch.manual_seed(SEED)
    inp = torch.randn(IN_SIZE_CONV)
    model.eval()
    model = symbolic_trace(model)
    expected_out = model(inp)

    model_orig = copy.deepcopy(model)
    regions = _extract_regions(model)
    _ = equalize_test(
        model, regions, merge_bias=False, bias_shrinkage='vaiq', scale_computation_type='maxabs')
    out = model(inp)

    # Check that equalization is not introducing FP variations
    assert torch.allclose(expected_out, out, atol=ATOL)

    regions = sorted(regions, key=lambda region: region[0][0])
    resnet_18_regions = sorted(RESNET_18_REGIONS, key=lambda region: region[0][0])
    equalized_layers = set()
    for region in resnet_18_regions:
        equalized_layers.update(region[0])
        equalized_layers.update(region[1])

    # Check that we found all the expected regions
    for region, expected_region in zip(regions, resnet_18_regions):
        sources_check = set(region[0]) == set(expected_region[0])
        sinks_check = set(region[1]) == set(expected_region[1])
        assert sources_check
        assert sinks_check

    # Check that all BatchNorm were equalized
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            assert hasattr(module, 'orig_bias')

    # Check that all layers were equalized and weights changed
    for layer in equalized_layers:
        eq_module = get_module(model, layer)
        orig_module = get_module(model_orig, layer)
        assert not torch.allclose(eq_module.weight, orig_module.weight)

    # We re-load the state dict of the equalized model
    model.load_state_dict(model.state_dict())
    for module in model.modules():
        if hasattr(module, 'orig_weight') and hasattr(module, 'orig_bias'):
            # For an equalized BN, the equalized weight and the original weights should be different
            assert not torch.allclose(module.weight, module.orig_weight)
    out = model(inp)
    assert torch.allclose(expected_out, out, atol=ATOL)
    # We re-load the original non-equalized state dict
    model.load_state_dict(model_orig.state_dict())
    for module in model.modules():
        if hasattr(module, 'orig_weight') and hasattr(module, 'orig_bias'):
            # In this case, all equalization parameters should be set to 1 (i.e., no-op)
            assert torch.allclose(module.scaling_factors, torch.tensor(1.))
            assert torch.allclose(module.inverse_scaling_factors, torch.tensor(1.))
            # The weight and the original weight must be the same
            assert torch.allclose(module.weight, module.orig_weight)


@pytest_cases.parametrize("merge_bias", [True, False])
def test_equalization_torchvision_models(model_coverage: tuple, merge_bias: bool):
    model, coverage = model_coverage

    torch.manual_seed(SEED)
    inp = torch.randn(IN_SIZE_CONV)
    model.eval()
    # The isistance does not work after symbolic trace
    is_alexnet = isinstance(model, models.AlexNet)
    model = symbolic_trace(model)

    expected_out = model(inp)

    regions = _extract_regions(model)
    scale_factor_regions = equalize_test(
        model,
        regions,
        merge_bias=merge_bias,
        bias_shrinkage='vaiq',
        scale_computation_type='maxabs')
    shape_scale_regions = [scale.shape for scale in scale_factor_regions]

    out = model(inp)
    srcs = set()
    sinks = set()
    count = 0
    for r in regions:
        srcs.update(list(r[0]))
        sinks.update(list(r[1]))

    for n in model.graph.nodes:
        if _is_supported_module(model, n):
            count += 1
    src_coverage = len(srcs) / count
    sink_coverage = len(sinks) / count
    assert src_coverage >= coverage[0]
    assert sink_coverage >= coverage[1]
    assert torch.allclose(expected_out, out, atol=ATOL)
    # Graph equalization can exit in case of shape mismatches or other error without performing any
    # equalization and returning a scalar value. We check that the equalized regions are as many as
    # expected
    if is_alexnet:
        # In AlexNet, we cannot equalize only through one region
        assert sum([shape == () for shape in shape_scale_regions]) == 1
    else:
        assert all([shape != () for shape in shape_scale_regions])


# Test that if we change BN stats, the bias value (which is dependant on them) also changes.
@pytest_cases.parametrize("merge_bias", [True, False])
def test_bn_stats_torchvision_models(model_coverage: tuple, merge_bias: bool):
    model, _ = model_coverage
    torch.manual_seed(SEED)
    model.eval()
    model = symbolic_trace(model)

    regions = _extract_regions(model)
    _ = equalize_test(
        model,
        regions,
        merge_bias=merge_bias,
        bias_shrinkage='vaiq',
        scale_computation_type='maxabs')

    pre_bn_stats = []
    post_bn_stats = []
    for _, module in model.named_modules():
        # Check for Equalized BN where scaling factors did not converge to one.
        # If they are all one, then the bias is no longer a function of running mean and running_var
        if hasattr(module, 'orig_bias') and not torch.allclose(
                module.scaling_factors, torch.ones_like(module.scaling_factors)):
            module.train()
            pre_bn_stats.append(module.bias.data.clone())
            # Simulate big changes in statistics otherwise we might get false negative in the tests
            module.running_mean.fill_(100)
            module.running_var.fill_(500)
            post_bn_stats.append(module.bias.data.clone())

    for pre_val, post_val in zip(pre_bn_stats, post_bn_stats):
        assert not torch.allclose(pre_val, post_val, atol=ATOL)


@pytest_cases.parametrize("merge_bias", [True, False])
def test_models(toy_model, merge_bias, request):
    test_id = request.node.callspec.id

    if 'mha' in test_id:
        in_shape = IN_SIZE_LINEAR
    else:
        in_shape = IN_SIZE_CONV

    model_class = toy_model
    model = model_class()
    inp = torch.randn(in_shape)

    model.eval()
    expected_out = model(inp)
    model = symbolic_trace(model)
    regions = _extract_regions(model)
    scale_factor_regions = equalize_test(
        model,
        regions,
        merge_bias=merge_bias,
        bias_shrinkage='vaiq',
        scale_computation_type='maxabs')
    shape_scale_regions = [scale.shape for scale in scale_factor_regions]

    out = model(inp)
    assert len(regions) > 0
    assert torch.allclose(expected_out, out, atol=ATOL)
    # Check that at least one region performs "true" equalization
    # If all shapes are scalar, no equalization has been performed
    assert all([shape != () for shape in shape_scale_regions])
