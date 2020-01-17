import torch
import brevitas.core.quant as q
from brevitas.core.scaling import StandaloneScaling, RuntimeStatsScaling, SCALING_SCALAR_SHAPE
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

INPUT_SIZE = 20
MIN_INT = 0
MAX_INT = 10
RTOL = 1e-05
ATOL = 1e-03


class TestQuant:
    def test_BinaryQuant(self):
        CHANNEL_NUMBER = np.random.randint(MIN_INT, MAX_INT, 1)[0]

        is_parameter_poss = [True, False]
        restrict_scaling_type_poss = [RestrictValueType.LOG_FP, RestrictValueType.FP, RestrictValueType.POWER_OF_TWO]
        shape = [SCALING_SCALAR_SHAPE, CHANNEL_NUMBER]

        combinations = list(itertools.product(*[is_parameter_poss, restrict_scaling_type_poss, shape]))

        for is_parameter, restrict_scaling_type, shape in combinations:
            if not is_parameter and shape == CHANNEL_NUMBER:
                continue  # Combination not supported by StandaloneScaling

            if shape == SCALING_SCALAR_SHAPE:
                input = torch.rand(INPUT_SIZE)
                scaling = torch.rand(1)[0]
            else:
                input = torch.rand(INPUT_SIZE, shape)
                scaling = torch.rand(shape)


            scaling_impl = StandaloneScaling(scaling_init=scaling, is_parameter=is_parameter,
                                             parameter_shape=scaling.shape,
                                             scaling_min_val=None,
                                             restrict_scaling_type=restrict_scaling_type)
            obj = q.BinaryQuant(scaling_impl=scaling_impl)
            output, scale, bit = obj(input, torch.tensor(ZERO_HW_SENTINEL_VALUE))

            scaling = scaling_impl(torch.tensor(ZERO_HW_SENTINEL_VALUE))
            output_gt = torch.sign(input) * scaling
            assert(torch.allclose(output, output_gt, RTOL, ATOL))


    def test_ClampedBinaryQuantWithStandaloneScaling(self):
        CHANNEL_NUMBER = np.random.randint(MIN_INT, MAX_INT, 1)[0]

        is_parameter_poss = [True, False]
        restrict_scaling_type_poss = [RestrictValueType.LOG_FP, RestrictValueType.FP, RestrictValueType.POWER_OF_TWO]
        shape = [SCALING_SCALAR_SHAPE, CHANNEL_NUMBER]

        combinations = list(itertools.product(*[is_parameter_poss, restrict_scaling_type_poss, shape]))
        for is_parameter, restrict_scaling_type, shape in combinations:
            if not is_parameter and shape == CHANNEL_NUMBER:
                continue  # Combination not supported by StandaloneScaling

            if shape == SCALING_SCALAR_SHAPE:
                input = torch.rand(INPUT_SIZE)
                scaling = torch.rand(1)[0]
            else:
                input = torch.rand(INPUT_SIZE, shape)
                scaling = torch.rand(shape)

            scaling_impl = StandaloneScaling(scaling_init=scaling, is_parameter=is_parameter,
                                             parameter_shape=scaling.shape,
                                             scaling_min_val=None,
                                             restrict_scaling_type=restrict_scaling_type)
            obj = q.ClampedBinaryQuant(scaling_impl=scaling_impl)
            output, scale, bit = obj(input, torch.tensor(ZERO_HW_SENTINEL_VALUE))

            scaling = scaling_impl(torch.tensor(ZERO_HW_SENTINEL_VALUE))
            output_gt = torch.where(input > scaling, scaling, input)
            output_gt = torch.where(output_gt < -scaling, -scaling, output_gt)
            output_gt = torch.sign(output_gt) * scaling
            assert(torch.allclose(output, output_gt, RTOL, ATOL))


        def test_ClampedBinaryQuantWithRuntimStatsScaling(self):
            stats_op_poss =[StatsOp.MAX, StatsOp.AVE, StatsOp.MAX_AVE, StatsOp.MEAN_SIGMA_STD,
                            StatsOp.MEAN_LEARN_SIGMA_STD]
            restrict_scaling_type_poss = [RestrictValueType.LOG_FP, RestrictValueType.FP, RestrictValueType.POWER_OF_TWO,
                                          RestrictValueType.INT]
            scaling_stats_input_view_shape_impl_poss = [StatsInputViewShapeImpl.OVER_TENSOR,
                                                        StatsInputViewShapeImpl.OVER_BATCH_OVER_OUTPUT_CHANNELS,
                                                        StatsInputViewShapeImpl.OVER_BATCH_OVER_TENSOR,
                                                        StatsInputViewShapeImpl.OVER_OUTPUT_CHANNELS]
            # scaling_shape TBD according to combinations
            # sigma is random float
            # scaling_min_val is 0
            # scaling_stats_reduce_dim TBD according to combinations
            # stats_buffer_momentum is random float
            # stats_buffer_init is initialization
            scaling_stats_permute_dims = (1, 0, 2, 3)  # To be set to NONE according to combinations
            affine_poss = [True, False]

#
    #
    # def test_TernaryQuant(self):
    #     scaling = torch.rand(1)[0]
    #     threshold = 0.7
    #     scaling_impl = StandaloneScaling(scaling_init=scaling, is_parameter=False, parameter_shape=SCALING_SCALAR_SHAPE,
    #                                      scaling_min_val=None, restrict_scaling_type=RestrictValueType.LOG_FP)
    #     obj = q.TernaryQuant(scaling_impl=scaling_impl, threshold=threshold)
    #     input = torch.rand(20)
    #     output, scale, bit = obj(input, torch.tensor(ZERO_HW_SENTINEL_VALUE))
    #     output_gt = (input.abs().ge(threshold * scaling)).float() * torch.sign(input) * scaling
    #     assert(all(output == output_gt))
    #
    #
    # def test_IntQuant(self):
    #     scaling = torch.rand(1)[0]
    #     int_scale = torch.tensor(1)
    #     bit_width = 8
    #     float_to_int_impl = RestrictValue(restrict_value_type=RestrictValueType.INT,
    #                                       float_to_int_impl_type=FloatToIntImplType.ROUND,
    #                                       min_val=None)
    #     tensor_clamp_impl = TensorClamp()
    #     input = torch.rand(20)
    #     obj = q.IntQuant(narrow_range=True, signed=True, float_to_int_impl=float_to_int_impl,
    #                      tensor_clamp_impl=tensor_clamp_impl)
    #     output = obj(scaling, int_scale, )