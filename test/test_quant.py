import torch
import torch.nn as nn
import brevitas.core.quant as q
import brevitas.function.ops as ops
from brevitas.core.scaling import StandaloneScaling, RuntimeStatsScaling, ParameterStatsScaling
from brevitas.core.scaling import SCALING_SCALAR_SHAPE
from brevitas.core.restrict_val import RestrictValueType, RestrictValue, FloatToIntImplType
from brevitas.core.stats import StatsInputViewShapeImpl
from brevitas.proxy.quant_proxy import ZERO_HW_SENTINEL_VALUE
from brevitas.core.stats import StatsOp
from brevitas.core.function_wrapper import TensorClamp
import numpy as np
import random
import itertools
import pytest
from unittest import mock
from hypothesis import given, assume, note
import hypothesis.strategies as st

SEED = 123456
random.seed(SEED)
np.random.seed(SEED)
torch.random.manual_seed(SEED)

# Size Constants
IN = 10
OUT = 100

MIN_INT = 0
MAX_INT = 10
RTOL = 1e-05
ATOL = 1e-03

float_st = st.floats(allow_nan=False, allow_infinity=False, width=32)

@given(input_values=st.lists(float_st), scale_factor=float_st)
def test_BinaryQuantWithStandaloneScaling(input_values, scale_factor):

    input_values = torch.tensor(input_values, requires_grad=True)
    input_values = 1 * input_values
    assume(scale_factor)
    with mock.patch('brevitas.core.scaling.StandaloneScaling') as mockStandaloneScaling:
        mockStandaloneScaling.return_value = torch.tensor(scale_factor)
        obj = q.BinaryQuant(scaling_impl=mockStandaloneScaling)
        output, scale, bit = obj(input_values, torch.tensor(ZERO_HW_SENTINEL_VALUE))
        output = output/scale
        expected_output = torch.sign(input_values)
        expected_output[expected_output == 0] = 1
        note(output)
        note(expected_output)
        assert(torch.allclose(output, expected_output, RTOL, ATOL))

    # def test_BinaryQuantWithParameterStatsScaling(self):
    #     scalingObj = ParameterStatsScalingClass()
    #     iterations = scalingObj.length()
    #     for i in range(iterations):
    #
    #         weight = nn.Parameter(torch.rand(IN, OUT), True)
    #         try:
    #             scaling_impl = scalingObj(i, weight)
    #         except:
    #             continue
    #         obj = q.BinaryQuant(scaling_impl=scaling_impl)
    #         output, scale, bit = obj(weight, torch.tensor(ZERO_HW_SENTINEL_VALUE))
    #
    #         scaling = scaling_impl(torch.tensor(ZERO_HW_SENTINEL_VALUE))
    #         output_gt = torch.sign(weight) * scaling
    #         assert (torch.allclose(output, output_gt, RTOL, ATOL))
    #
    # def test_ClampedBinaryQuantWithStandaloneScaling(self):
    #     scalingObj = StandaloneScalingClass()
    #     iterations = scalingObj.length()
    #     for i in range(iterations):
    #
    #         input = nn.Parameter(torch.rand(IN, OUT), True) * 2 - 1
    #         try:
    #             scaling_impl = scalingObj(i)
    #         except:
    #             continue
    #         obj = q.BinaryQuant(scaling_impl=scaling_impl)
    #         output, scale, bit = obj(input, torch.tensor(ZERO_HW_SENTINEL_VALUE))
    #
    #         scaling = scaling_impl(torch.tensor(ZERO_HW_SENTINEL_VALUE))
    #         output_gt = torch.where(input > scaling, scaling, input)
    #         output_gt = torch.where(output_gt < -scaling, -scaling, output_gt)
    #         output_gt = torch.sign(output_gt) * scaling
    #         assert (torch.allclose(output, output_gt, RTOL, ATOL))
    #
    # def test_ClampedBinaryQuantWithRuntimeStatsScaling(self):
    #     scalingObj = RuntimeStatsScalingClass()
    #     iterations = scalingObj.length()
    #     for i in range(iterations):
    #
    #         input = nn.Parameter(torch.rand(IN, OUT), True) * 2 - 1
    #         try:
    #             scaling_impl = scalingObj(i)
    #         except:
    #             continue
    #
    #         scaling_impl.eval()
    #         obj = q.ClampedBinaryQuant(scaling_impl=scaling_impl)
    #         output, scale, bit = obj(input, torch.tensor(ZERO_HW_SENTINEL_VALUE))
    #
    #         scaling = scaling_impl(torch.tensor(ZERO_HW_SENTINEL_VALUE))
    #         output_gt = torch.where(input > scaling, scaling, input)
    #         output_gt = torch.where(output_gt < -scaling, -scaling, output_gt)
    #         output_gt = torch.sign(output_gt) * scaling
    #
    #         assert (torch.allclose(output, output_gt, RTOL, ATOL))
    #
    # def test_TernaryQuantWithStandaloneScaling(self):
    #     scalingObj = StandaloneScalingClass()
    #     iterations = scalingObj.length()
    #     for i in range(iterations):
    #         weight = nn.Parameter(torch.rand(IN, OUT), True)
    #         try:
    #             scaling_impl = scalingObj(i)
    #         except:
    #             continue
    #         threshold = 0.7
    #
    #         obj = q.TernaryQuant(scaling_impl=scaling_impl, threshold=threshold)
    #         output, scale, bit = obj(weight, torch.tensor(ZERO_HW_SENTINEL_VALUE))
    #
    #         scaling = scaling_impl(torch.tensor(ZERO_HW_SENTINEL_VALUE))
    #         output_gt = (weight.abs().ge(threshold * scaling)).float() * torch.sign(weight) * scaling
    #         assert (torch.allclose(output, output_gt, RTOL, ATOL))
    #
    # def test_TernaryQuantWithParameterStatsScaling(self):
    #     scalingObj = ParameterStatsScalingClass()
    #     iterations = scalingObj.length()
    #     for i in range(iterations):
    #         weight = nn.Parameter(torch.rand(IN, OUT), True)
    #         try:
    #             scaling_impl = scalingObj(i)
    #         except:
    #             continue
    #         threshold = 0.7
    #
    #         obj = q.TernaryQuant(scaling_impl=scaling_impl, threshold=threshold)
    #         output, scale, bit = obj(weight, torch.tensor(ZERO_HW_SENTINEL_VALUE))
    #
    #         scaling = scaling_impl(torch.tensor(ZERO_HW_SENTINEL_VALUE))
    #         output_gt = (weight.abs().ge(threshold * scaling)).float() * torch.sign(weight) * scaling
    #         assert (torch.allclose(output, output_gt, RTOL, ATOL))
    #
    # def test_IntQuant(self):
    #     tensor_clamp_impl = TensorClamp()
    #     narrow_range_poss = [True, False]
    #     signed_poss = [True, False]
    #     float_to_int_poss = [FloatToIntImplType.ROUND, FloatToIntImplType.FLOOR, FloatToIntImplType.CEIL]
    #     restrict_value_type = RestrictValueType.INT
    #     combinations = list(itertools.product(*[narrow_range_poss, signed_poss, float_to_int_poss]))
    #     for narrow_range, signed, float_to_int in combinations:
    #         scaling = torch.rand(1)[0]
    #         int_scale = torch.tensor(1.0)
    #         bit_width = torch.tensor(8.0)
    #         float_to_int_impl = RestrictValue(restrict_value_type=restrict_value_type,
    #                                           float_to_int_impl_type=float_to_int,
    #                                           min_val=None)
    #         input = torch.rand(IN, OUT)
    #         obj = q.IntQuant(narrow_range=True, signed=True, float_to_int_impl=float_to_int_impl,
    #                          tensor_clamp_impl=tensor_clamp_impl)
    #         output = obj(scaling, int_scale, bit_width, input)
    #
    #         min = ops.min_int(signed, narrow_range, bit_width)
    #         max = ops.max_int(signed, bit_width)
    #         input = (input / scaling) * int_scale
    #         input = tensor_clamp_impl(input, min_val=min, max_val=max)
    #         output_gt = float_to_int_impl(input) / int_scale * scaling
    #
    #         assert (torch.allclose(output, output_gt, RTOL, ATOL))
