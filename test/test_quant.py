import torch
import brevitas.core.quant as q
from brevitas.core.scaling import StandaloneScaling, SCALING_SCALAR_SHAPE
from brevitas.core.restrict_val import RestrictValueType
from brevitas.proxy.quant_proxy import ZERO_HW_SENTINEL_VALUE
import numpy as np
import random
SEED = 123456
random.seed(SEED)
np.random.seed(SEED)
torch.random.manual_seed(SEED)


class TestQuant:
    def test_BinaryQuant(self):
        scaling = torch.rand(1)[0]
        scaling_impl = StandaloneScaling(scaling_init=scaling, is_parameter=False, parameter_shape=SCALING_SCALAR_SHAPE,
                                         scaling_min_val=None, restrict_scaling_type=RestrictValueType.LOG_FP)
        obj = q.BinaryQuant(scaling_impl=scaling_impl)
        input = torch.rand(20)
        output, scale, bit = obj(input, torch.tensor(ZERO_HW_SENTINEL_VALUE))
        output_gt = torch.sign(input) * scale
        assert(all(output == output_gt))

    def test_ClampedBinaryQuant(self):
        scaling = torch.rand(1)[0]
        scaling_impl = StandaloneScaling(scaling_init=scaling, is_parameter=False, parameter_shape=SCALING_SCALAR_SHAPE,
                                         scaling_min_val=None, restrict_scaling_type=RestrictValueType.LOG_FP)
        obj = q.ClampedBinaryQuant(scaling_impl=scaling_impl)
        input = torch.rand(20)
        output, scale, bit = obj(input, torch.tensor(ZERO_HW_SENTINEL_VALUE))
        output_gt = torch.sign(torch.clamp(input, min=-scale, max=scale)) * scale
        assert(all(output == output_gt))


    def test_TernaryQuant(self):
        scaling = torch.rand(1)[0]
        threshold = 0.7
        scaling_impl = StandaloneScaling(scaling_init=scaling, is_parameter=False, parameter_shape=SCALING_SCALAR_SHAPE,
                                         scaling_min_val=None, restrict_scaling_type=RestrictValueType.LOG_FP)
        obj = q.TernaryQuant(scaling_impl=scaling_impl, threshold=threshold)
        input = torch.rand(20)
        output, scale, bit = obj(input, torch.tensor(ZERO_HW_SENTINEL_VALUE))
        output_gt = (input.abs().ge(threshold * scaling)).float() * torch.sign(input) * scaling
        assert(all(output == output_gt))


    def test_IntQuant(self):
        float_to_int_impl = RestrictValue(restrict_value_type=RestrictValueType.INT,
                                          float_to_int_impl_type=FloatToIntImplType.ROUND,
                                          min_val=None)
        tensor_clamp_impl = TensorClamp()
        obj = q.IntQuant(narrow_range=True, signed=True)