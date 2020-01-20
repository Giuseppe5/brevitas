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

SEED = 123456
random.seed(SEED)
np.random.seed(SEED)
torch.random.manual_seed(SEED)

# Weight Constants
IN = 10
OUT = 100

MIN_INT = 0
MAX_INT = 10
RTOL = 1e-05
ATOL = 1e-03

class StandaloneScalingClass(nn.Module):
    def __init__(self):
        super(StandaloneScalingClass).__init__()
        is_parameter_poss = [True, False]
        restrict_scaling_type_poss = [RestrictValueType.LOG_FP, RestrictValueType.FP, RestrictValueType.POWER_OF_TWO]
        shape = [SCALING_SCALAR_SHAPE, OUT]
        self.combinations = list(itertools.product(*[is_parameter_poss, restrict_scaling_type_poss, shape]))

    def forward(self, input):
        is_parameter, restrict_scaling_type, shape = self.combinations[input]
        if shape == SCALING_SCALAR_SHAPE:
            scaling = torch.rand(1)[0]
        else:
            scaling = torch.rand(shape)
        scaling_impl = StandaloneScaling(scaling_init=scaling, is_parameter=is_parameter,
                                         parameter_shape=scaling.shape,
                                         scaling_min_val=None,
                                         restrict_scaling_type=restrict_scaling_type)
        return scaling_impl, scaling

    def length(self):
        return len(self.combinations)

class ParameterStatsScalingClass(nn.Module):
    def __init__(self):
        super(ParameterStatsScalingClass).__init__()
        stats_op_poss = [StatsOp.MAX, StatsOp.AVE, StatsOp.MAX_AVE, StatsOp.MEAN_SIGMA_STD,
                         StatsOp.MEAN_LEARN_SIGMA_STD]
        restrict_scaling_type_poss = [RestrictValueType.LOG_FP, RestrictValueType.FP, RestrictValueType.POWER_OF_TWO,
                                      RestrictValueType.INT]

        weight_scaling_per_output_channel_poss = [True, False]
        affine_poss = [True, False]
        self.combinations = list(itertools.product(*[stats_op_poss, restrict_scaling_type_poss,
                                                     affine_poss,
                                                     weight_scaling_per_output_channel_poss]))

    def forward(self, input, weight):
        stats_op, restrict_scaling_type, affine, weight_scaling_per_output_channel = self.combinations[input]
        scaling_stats_input_concat_dim = 1
        scaling_min_val = 0
        sigma = 3.0

        if weight_scaling_per_output_channel:
            stats_input_view_shape_impl = StatsInputViewShapeImpl.OVER_OUTPUT_CHANNELS
            scaling_shape = (OUT, 1)  # Supposed a Linear Weight with 100 as output dim
            scaling_stats_reduce_dim = 1
        else:
            stats_input_view_shape_impl = StatsInputViewShapeImpl.OVER_TENSOR
            scaling_shape = SCALING_SCALAR_SHAPE
            scaling_stats_reduce_dim = None

        if stats_op == StatsOp.MAX_AVE:
            stats_input_view_shape_impl = StatsInputViewShapeImpl.OVER_OUTPUT_CHANNELS
            scaling_stats_reduce_dim = 1

        scaling_impl = ParameterStatsScaling(stats_op=stats_op, restrict_scaling_type=restrict_scaling_type,
                                             tracked_parameter_list=[weight],
                                             stats_input_view_shape_impl=stats_input_view_shape_impl,
                                             stats_input_concat_dim=scaling_stats_input_concat_dim,
                                             scaling_min_val=scaling_min_val,
                                             sigma=sigma,
                                             stats_reduce_dim=scaling_stats_reduce_dim,
                                             stats_output_shape=scaling_shape,
                                             affine=affine)
        return scaling_impl

    def length(self):
        return len(self.combinations)

class RuntimeStatsScalingClass():
    def __init__(self):
        super(RuntimeStatsScalingClass).__init__()
        stats_op_poss = [StatsOp.MAX, StatsOp.AVE, StatsOp.MAX_AVE, StatsOp.MEAN_SIGMA_STD,
                         StatsOp.MEAN_LEARN_SIGMA_STD]
        restrict_scaling_type_poss = [RestrictValueType.LOG_FP, RestrictValueType.FP,
                                      RestrictValueType.POWER_OF_TWO,
                                      RestrictValueType.INT]
        scaling_stats_input_view_shape_impl_poss = [StatsInputViewShapeImpl.OVER_TENSOR,
                                                    StatsInputViewShapeImpl.OVER_BATCH_OVER_OUTPUT_CHANNELS,
                                                    StatsInputViewShapeImpl.OVER_BATCH_OVER_TENSOR,
                                                    StatsInputViewShapeImpl.OVER_OUTPUT_CHANNELS]
        scaling_per_channel_poss = [True, False]
        affine_poss = [True, False]

        self.combinations = list(itertools.product(*[stats_op_poss, restrict_scaling_type_poss,
                                                scaling_stats_input_view_shape_impl_poss,
                                                scaling_per_channel_poss, affine_poss]))
    def forward(self, input):
        stats_op, restrict_scaling_type, scaling_stats_input_view_shape_impl, scaling_per_channel, affine = self.combinations[input]
        sigma = 3.0
        scaling_min_val = 0
        stats_buffer_momentum = 0.1
        stats_buffer_init = 1.0
        scaling_stats_permute_dims = (1, 0, 2, 3)
        if scaling_per_channel:
            per_channel_broadcastable_shape = OUT
        if scaling_per_channel and not stats_op == StatsOp.MAX_AVE:
            scaling_shape = per_channel_broadcastable_shape
            scaling_stats_reduce_dim = 1
        elif scaling_per_channel and stats_op == StatsOp.MAX_AVE:
            raise Exception("Not Supported")
        elif not scaling_per_channel and stats_op == StatsOp.MAX_AVE:
            raise Exception("Not supported")
        else:  # not scaling_per_channel
            scaling_shape = SCALING_SCALAR_SHAPE
            scaling_stats_input_view_shape_impl = StatsInputViewShapeImpl.OVER_TENSOR
            scaling_stats_reduce_dim = None
            scaling_stats_permute_dims = None
        scaling_impl = RuntimeStatsScaling(stats_op=stats_op,
                            restrict_scaling_type=restrict_scaling_type,
                            stats_input_view_shape_impl=scaling_stats_input_view_shape_impl,
                            stats_output_shape=scaling_shape,
                            sigma=sigma,
                            scaling_min_val=scaling_min_val,
                            stats_reduce_dim=scaling_stats_reduce_dim,
                            stats_buffer_momentum=stats_buffer_momentum,
                            stats_buffer_init=stats_buffer_init,
                            stats_permute_dims=scaling_stats_permute_dims,
                            affine=affine)
        return scaling_impl

    def length(self):
        return len(self.combinations)


class TestQuant:
    def test_BinaryQuantWithStandaloneScaling(self):
        scalingObj = StandaloneScalingClass()
        iterations = scalingObj.length()
        for i in range(iterations):

            weight = nn.Parameter(torch.rand(IN, OUT), True)
            try:
                scaling_impl = scalingObj(i)
            except:
                continue
            obj = q.BinaryQuant(scaling_impl=scaling_impl)
            output, scale, bit = obj(weight, torch.tensor(ZERO_HW_SENTINEL_VALUE))

            scaling = scaling_impl(torch.tensor(ZERO_HW_SENTINEL_VALUE))
            output_gt = torch.sign(weight) * scaling
            assert (torch.allclose(output, output_gt, RTOL, ATOL))

    def test_BinaryQuantWithParameterStatsScaling(self):
        scalingObj = ParameterStatsScalingClass()
        iterations = scalingObj.length()
        for i in range(iterations):

            weight = nn.Parameter(torch.rand(IN, OUT), True)
            try:
                scaling_impl = scalingObj(i, weight)
            except:
                continue
            obj = q.BinaryQuant(scaling_impl=scaling_impl)
            output, scale, bit = obj(weight, torch.tensor(ZERO_HW_SENTINEL_VALUE))

            scaling = scaling_impl(torch.tensor(ZERO_HW_SENTINEL_VALUE))
            output_gt = torch.sign(weight) * scaling
            assert (torch.allclose(output, output_gt, RTOL, ATOL))

    def test_ClampedBinaryQuantWithStandaloneScaling(self):
        scalingObj = StandaloneScalingClass()
        iterations = scalingObj.length()
        for i in range(iterations):

            input = nn.Parameter(torch.rand(IN, OUT), True) * 2 - 1
            try:
                scaling_impl = scalingObj(i)
            except:
                continue
            obj = q.BinaryQuant(scaling_impl=scaling_impl)
            output, scale, bit = obj(input, torch.tensor(ZERO_HW_SENTINEL_VALUE))

            scaling = scaling_impl(torch.tensor(ZERO_HW_SENTINEL_VALUE))
            output_gt = torch.where(input > scaling, scaling, input)
            output_gt = torch.where(output_gt < -scaling, -scaling, output_gt)
            output_gt = torch.sign(output_gt) * scaling
            assert (torch.allclose(output, output_gt, RTOL, ATOL))

    def test_ClampedBinaryQuantWithRuntimeStatsScaling(self):
        scalingObj = RuntimeStatsScalingClass()
        iterations = scalingObj.length()
        for i in range(iterations):

            input = nn.Parameter(torch.rand(IN, OUT), True) * 2 - 1
            try:
                scaling_impl = scalingObj(i)
            except:
                continue

            scaling_impl.eval()
            obj = q.ClampedBinaryQuant(scaling_impl=scaling_impl)
            output, scale, bit = obj(input, torch.tensor(ZERO_HW_SENTINEL_VALUE))

            scaling = scaling_impl(torch.tensor(ZERO_HW_SENTINEL_VALUE))
            output_gt = torch.where(input > scaling, scaling, input)
            output_gt = torch.where(output_gt < -scaling, -scaling, output_gt)
            output_gt = torch.sign(output_gt) * scaling

            assert (torch.allclose(output, output_gt, RTOL, ATOL))

    def test_TernaryQuantWithStandaloneScaling(self):
        scalingObj = StandaloneScalingClass()
        iterations = scalingObj.length()
        for i in range(iterations):
            weight = nn.Parameter(torch.rand(IN, OUT), True)
            try:
                scaling_impl = scalingObj(i)
            except:
                continue
            threshold = 0.7

            obj = q.TernaryQuant(scaling_impl=scaling_impl, threshold=threshold)
            output, scale, bit = obj(weight, torch.tensor(ZERO_HW_SENTINEL_VALUE))

            scaling = scaling_impl(torch.tensor(ZERO_HW_SENTINEL_VALUE))
            output_gt = (weight.abs().ge(threshold * scaling)).float() * torch.sign(weight) * scaling
            assert (torch.allclose(output, output_gt, RTOL, ATOL))

    def test_TernaryQuantWithParameterStatsScaling(self):
        scalingObj = ParameterStatsScalingClass()
        iterations = scalingObj.length()
        for i in range(iterations):
            weight = nn.Parameter(torch.rand(IN, OUT), True)
            try:
                scaling_impl = scalingObj(i)
            except:
                continue
            threshold = 0.7

            obj = q.TernaryQuant(scaling_impl=scaling_impl, threshold=threshold)
            output, scale, bit = obj(weight, torch.tensor(ZERO_HW_SENTINEL_VALUE))

            scaling = scaling_impl(torch.tensor(ZERO_HW_SENTINEL_VALUE))
            output_gt = (weight.abs().ge(threshold * scaling)).float() * torch.sign(weight) * scaling
            assert (torch.allclose(output, output_gt, RTOL, ATOL))

    def test_IntQuant(self):
        tensor_clamp_impl = TensorClamp()
        narrow_range_poss = [True, False]
        signed_poss = [True, False]
        float_to_int_poss = [FloatToIntImplType.ROUND, FloatToIntImplType.FLOOR, FloatToIntImplType.CEIL]
        restrict_value_type = RestrictValueType.INT
        combinations = list(itertools.product(*[narrow_range_poss, signed_poss, float_to_int_poss]))
        for narrow_range, signed, float_to_int in combinations:

            scaling = torch.rand(1)[0]
            int_scale = torch.tensor(1.0)
            bit_width = torch.tensor(8.0)
            float_to_int_impl = RestrictValue(restrict_value_type=restrict_value_type,
                                            float_to_int_impl_type=float_to_int,
                                            min_val=None)
            input = torch.rand(IN, OUT)
            obj = q.IntQuant(narrow_range=True, signed=True, float_to_int_impl=float_to_int_impl,
                             tensor_clamp_impl=tensor_clamp_impl)
            output = obj(scaling, int_scale, bit_width, input)

            min = ops.min_int(signed, narrow_range, bit_width)
            max = ops.max_int(signed, bit_width)
            input = (input/scaling) * int_scale
            input = tensor_clamp_impl(input, min_val=min, max_val=max)
            output_gt = float_to_int_impl(input)/int_scale * scaling

            assert (torch.allclose(output, output_gt, RTOL, ATOL))



