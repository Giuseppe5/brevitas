# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABC
from abc import abstractmethod
from typing import Tuple

import torch
from torch import Tensor
import torch.nn as nn

from brevitas.function import compute_max_mantissa
from brevitas.function.ops import max_float
from brevitas.function.ops import max_int
from brevitas.function.ops import min_int
from brevitas.proxy.float_parameter_quant import WeightFloatQuantProxyFromInjector
from brevitas.proxy.float_runtime_quant import ActFloatQuantProxyFromInjector
from brevitas.proxy.float_runtime_quant import DynamicActFloatQuantProxyFromInjector
from brevitas.proxy.groupwise_float_parameter_quant import \
    GroupwiseWeightFloatQuantProxyFromInjector
from brevitas.proxy.groupwise_float_runtime_quant import GroupwiseActFloatQuantProxyFromInjector
from brevitas.proxy.groupwise_int_parameter_quant import GroupwiseWeightQuantProxyFromInjector
from brevitas.proxy.parameter_quant import BiasQuantProxyFromInjector
from brevitas.proxy.parameter_quant import WeightQuantProxyFromInjector
from brevitas.proxy.runtime_quant import ActQuantProxyFromInjector
from brevitas.proxy.runtime_quant import DynamicActQuantProxyFromInjector
from brevitas.quant.experimental.mx_quant_ocp import GroupwiseActQuantProxyFromInjector
from brevitas.utils.quant_utils import groupwise_dequant_expand
from brevitas.utils.torch_utils import float_internal_scale


class InferenceHandler(torch.nn.Module, ABC):

    def attach_debug_info(self, module: nn.Module):
        pass

    @abstractmethod
    def prepare_for_export(self, module: nn.Module):
        pass

    @abstractmethod
    def quantize(self, x: Tensor):
        pass

    @abstractmethod
    def dequantize(self, x: Tensor):
        pass


class IntInferencetHandler(InferenceHandler):
    handled_layer = (ActQuantProxyFromInjector, BiasQuantProxyFromInjector)

    def __init__(self):
        super().__init__()
        self.register_buffer('scale', torch.ones(1))
        self.register_buffer('zero_point', torch.ones(0))

    def prepare_for_export(self, module: nn.Module):
        if module.is_quant_enabled:
            scale = module.scale_() if hasattr(module, 'scale_') else module.scale()
            zero_point = module.zero_point_() if hasattr(module,
                                                         'zero_point_') else module.zero_point()
            # Continguous is used to be extra-safe with torch.compile
            self.scale = scale.contiguous()
            self.zero_point = zero_point.contiguous()

            self.zero_point = self.zero_point.to(self.scale.device)
            self.bit_width = module.bit_width()
            self.min_clamp = min_int(module.is_signed, module.is_narrow_range, self.bit_width)
            self.max_clamp = max_int(module.is_signed, module.is_narrow_range, self.bit_width)
            if hasattr(module.tensor_quant, 'int_quant'):
                self.float_to_int_impl = module.tensor_quant.int_quant.float_to_int_impl
            elif hasattr(module, 'fused_activation_quant_proxy'):
                self.float_to_int_impl = module.fused_activation_quant_proxy.tensor_quant.int_quant.float_to_int_impl

    def quantize(self, x: Tensor, scale: Tensor, zero_point: Tensor) -> Tuple[Tensor]:
        return torch.clamp(
            self.float_to_int_impl(x / scale + zero_point), self.min_clamp, self.max_clamp)

    def dequantize(self, x: Tensor, scale: Tensor, zero_point: Tensor) -> Tensor:
        return (x - zero_point) * scale

    def forward(self, x: Tensor, unused_scale: Tensor = None) -> Tuple[Tensor]:
        return self.dequantize(self.quantize(x, self.scale, self.zero_point), self.scale, self.zero_point), self.scale, self.zero_point, self.bit_width


class IntWeightInferencetHandler(IntInferencetHandler):
    handled_layer = WeightQuantProxyFromInjector

    def __init__(self):
        super().__init__()
        self.register_buffer('cached_weight', torch.ones(1))

    def prepare_for_export(self, module: nn.Module):
        super().prepare_for_export(module)
        if module.is_quant_enabled:
            if module._cached_weight is not None and not module.cache_inference_quant_weight_metadata_only:
                self.cached_weight = module._cached_weight.value
            else:
                self.cached_weight = None

    def inner_forward(self, x: Tensor, scale: Tensor, zero_point: Tensor) -> Tensor:
        return self.dequantize(self.quantize(x, scale, zero_point), scale, zero_point)

    def forward(self, x: Tensor) -> Tuple[Tensor]:
        if self.cached_weight is not None:
            x = self.cached_weight
        else:
            x = self.inner_forward(x, self.scale, self.zero_point)

        return x, self.scale, self.zero_point, self.bit_width


class DynamicIntInferenceHandler(IntInferencetHandler):
    handled_layer = DynamicActQuantProxyFromInjector

    def prepare_for_export(self, module: nn.Module):
        if module.is_quant_enabled:
            self.module_forward = module.fused_activation_quant_proxy.tensor_quant

    def forward(self, x: Tensor, unused_scale: Tensor = None) -> Tuple[Tensor]:
        return self.module_forward(x)


class GroupwiseIntInferenceHandler(IntInferencetHandler):
    handled_layer = GroupwiseActQuantProxyFromInjector

    def __init__(self):
        super().__init__()
        self.skip_create_quant_tensor = True

    def prepare_for_export(self, module):
        if module.is_quant_enabled:
            self.module_forward = module.fused_activation_quant_proxy.tensor_quant
            self.group_dim = module.group_dim

    def forward(self, x: Tensor, unused_scale: Tensor = None) -> Tuple[Tensor]:
        # In inference mode, we never return quant tensors
        assert self.skip_create_quant_tensor
        inp_shape = x.shape
        x, scale, zero_point, *other = self.module_forward(x)

        # If we skip quant tensor, we return the flattened version of the groupwise tensor
        if self.skip_create_quant_tensor:
            x = groupwise_dequant_expand(x, scale, zero_point, self.group_dim, inp_shape)[0]
        output_args = tuple([x, scale, zero_point] + list(other))
        return output_args


class GroupwiseIntWeightInferenceHandler(IntWeightInferencetHandler):
    handled_layer = GroupwiseWeightQuantProxyFromInjector

    def __init__(self):
        super().__init__()
        self.skip_create_quant_tensor = True

    def prepare_for_export(self, module):
        super().prepare_for_export(module)
        if module.is_quant_enabled:
            self.group_dim = module.group_dim
            self.input_view = module.input_view_impl

    def inner_forward(self, x: Tensor, scale: Tensor, zero_point: Tensor) -> Tensor:
        return self.dequantize(self.quantize(x, scale, zero_point), scale, zero_point)

    def forward(self, x: Tensor) -> Tuple[Tensor]:
        # In inference mode, we never return quant tensors
        assert self.skip_create_quant_tensor
        scale = self.scale
        if scale.shape != ():
            scale = self.input_view(scale)
        zero_point = self.zero_point
        if zero_point.shape != ():
            zero_point = self.input_view(zero_point)

        if self.cached_weight is not None:
            out = self.cached_weight
        else:
            inp_shape = x.shape
            x = self.input_view(x)
            out = self.inner_forward(x, scale, zero_point)

            # If we skip quant tensor, we return the flattened version of the groupwise tensor
            out = groupwise_dequant_expand(out, scale, zero_point, self.group_dim, inp_shape)[0]
        return out, scale, zero_point, self.bit_width


class FloatInferencetHandler(InferenceHandler):
    handled_layer = (ActFloatQuantProxyFromInjector, BiasQuantProxyFromInjector)

    def __init__(self):
        super().__init__()
        self.register_buffer('scale', torch.ones(1))
        self.register_buffer('zero_point', torch.ones(0))

    def prepare_for_export(self, module):
        if module.is_quant_enabled:
            self.scale = module.scale_() if hasattr(module, 'scale_') else module.scale()
            self.zero_point = module.zero_point_() if hasattr(
                module, 'zero_point_') else module.zero_point()
            # Continguous is used to be extra-safe with torch.compile
            self.zero_point = self.zero_point.contiguous()
            self.scale = self.scale.contiguous()
            self.zero_point = self.zero_point.to(self.scale.device)
            self.exponent_bit_width = module.exponent_bit_width()
            self.mantissa_bit_width = module.mantissa_bit_width()
            self.exponent_bias = module.exponent_bias()
            self.saturating = module.is_saturating()
            self.inf_values = module.inf_values()
            self.nan_values = module.nan_values()
            self.eps = torch.finfo(self.scale.dtype).tiny
            if hasattr(module.tensor_quant, 'float_to_int_impl'):
                self.float_to_int_impl = module.tensor_quant.float_to_int_impl
                self.float_clamp_impl = module.tensor_quant.float_clamp_impl
                self.max_available_float = module.tensor_quant.float_clamp_impl.max_available_float
            elif hasattr(module, 'fused_activation_quant_proxy'):
                self.float_to_int_impl = module.fused_activation_quant_proxy.tensor_quant.float_to_int_impl
                self.float_clamp_impl = module.fused_activation_quant_proxy.tensor_quant.float_clamp_impl
                self.max_available_float = module.fused_activation_quant_proxy.tensor_quant.float_clamp_impl.max_available_float

            self.pre_compute_max_mantissa = compute_max_mantissa(self.mantissa_bit_width)
            self.max_clamp = max_float(
                self.exponent_bit_width, self.pre_compute_max_mantissa, self.exponent_bias)
            self.min_clamp = -self.max_clamp
            self.fp_internal_scale_min = 1. - self.exponent_bias - self.mantissa_bit_width
            self.max_value = max_float(
                self.exponent_bit_width, self.pre_compute_max_mantissa, self.exponent_bias)
            self.max_value = self.max_value if self.max_available_float is None else torch.min(
                self.max_value, self.max_available_float())
            self.min_value = torch.tensor(0.) if not module.is_signed else -self.max_value

    def quantize(self, x: Tensor, scale: Tensor, zero_point: Tensor) -> Tuple[Tensor]:
        # Quantize
        x = x / scale
        internal_scale = float_internal_scale(
            x, self.mantissa_bit_width, self.fp_internal_scale_min, self.eps)
        x = internal_scale * self.float_to_int_impl(x / internal_scale)

        # Compute masks
        if not self.saturating:
            inf_mask = x.isinf()
            p_max_val_mask = x > self.max_value
            n_max_val_mask = -x > self.max_value

        # Clamp
        x = self.float_clamp_impl.saturating_clamp(x, self.max_value, self.min_value)
        if not self.saturating:
            x = self.float_clamp_impl.inf_nan_clamp(x, inf_mask, p_max_val_mask, n_max_val_mask)

        return x

    def dequantize(self, x: Tensor, scale: Tensor, zero_point: Tensor) -> Tensor:
        return (x - zero_point) * scale

    def forward(self, x: Tensor) -> Tuple[Tensor]:
        return self.dequantize(self.quantize(x, self.scale, self.zero_point), self.scale, self.zero_point), self.scale, self.zero_point, self.exponent_bit_width, self.mantissa_bit_width, self.exponent_bias, self.saturating, self.inf_values, self.nan_values


class FloatWeightInferencetHandler(FloatInferencetHandler):
    handled_layer = WeightFloatQuantProxyFromInjector

    def __init__(self):
        super().__init__()
        self.register_buffer('cached_weight', torch.ones(1))

    def prepare_for_export(self, module):
        super().prepare_for_export(module)
        if module.is_quant_enabled:
            if module._cached_weight is not None and not module.cache_inference_quant_weight_metadata_only:
                self.cached_weight = module._cached_weight.value
            else:
                self.cached_weight = None

    def inner_forward(self, x: Tensor, scale: Tensor, zero_point: Tensor) -> Tensor:
        return self.dequantize(self.quantize(x, scale, zero_point), scale, zero_point)

    def forward(self, x: Tensor) -> Tuple[Tensor]:
        if self.cached_weight is not None:
            x = self.cached_weight
        else:
            x = self.inner_forward(x, self.scale, self.zero_point)
        return x, self.scale, self.zero_point, self.exponent_bit_width, self.mantissa_bit_width, self.exponent_bias, self.saturating, self.inf_values, self.nan_values


class GroupwiseFloatInferenceHandler(FloatInferencetHandler):
    handled_layer = GroupwiseActFloatQuantProxyFromInjector

    def __init__(self):
        super().__init__()
        self.skip_create_quant_tensor = True

    def prepare_for_export(self, module: nn.Module):
        if module.is_quant_enabled:
            self.module_forward = module.fused_activation_quant_proxy.tensor_quant
            self.group_dim = module.group_dim

    def forward(self, x: Tensor) -> Tuple[Tensor]:
        # In inference mode, we never return quant tensors
        assert self.skip_create_quant_tensor
        inp_shape = x.shape
        x, scale, zero_point, *other = self.module_forward(x)
        # If we skip quant tensor, we return the flattened version of the groupwise tensor
        x = groupwise_dequant_expand(x, scale, zero_point, self.group_dim, inp_shape)[0]
        output_args = tuple([x, scale, zero_point] + list(other))
        return output_args


class GroupwiseFloatWeightInferenceHandler(FloatWeightInferencetHandler):
    handled_layer = GroupwiseWeightFloatQuantProxyFromInjector

    def __init__(self):
        super().__init__()
        self.skip_create_quant_tensor = True

    def prepare_for_export(self, module: nn.Module):
        super().prepare_for_export(module)
        if module.is_quant_enabled:
            self.input_view = module.input_view_impl
            self.group_dim = module.group_dim

    def inner_forward(self, x: Tensor, scale: Tensor, zero_point: Tensor) -> Tuple[Tensor]:
        out = self.dequantize(self.quantize(x, scale, zero_point), scale, zero_point)
        return out

    def quantize_no_expand(self, x):
        scale = self.scale
        if scale.shape != ():
            scale = self.input_view(scale)

        zero_point = self.zero_point
        if zero_point.shape != ():
            zero_point = self.input_view(zero_point)

        out = self.inner_forward(x, scale, zero_point)
        return out

    def forward(self, x: Tensor) -> Tuple[Tensor]:
        # In inference mode, we never return quant tensors
        assert self.skip_create_quant_tensor
        if self.cached_weight is not None:
            out = self.cached_weight
        else:
            inp_shape = x.shape
            scale = self.scale
            if scale.shape != ():
                scale = self.input_view(scale)

            zero_point = self.zero_point
            if zero_point.shape != ():
                zero_point = self.input_view(zero_point)

            x = self.input_view(x)
            out = self.quantize_no_expand(x)
            # out = self.inner_forward(x, scale, zero_point)
            out, scale, zp = groupwise_dequant_expand(out, scale, zero_point, self.group_dim, inp_shape)


        return out, scale, zero_point, self.exponent_bit_width, self.mantissa_bit_width, self.exponent_bias, self.saturating, self.inf_values, self.nan_values


class DynamicFloatInferenceHandler(FloatInferencetHandler):
    handled_layer = DynamicActFloatQuantProxyFromInjector

    def prepare_for_export(self, module: nn.Module):
        if module.is_quant_enabled:
            self.module_forward = module.fused_activation_quant_proxy.tensor_quant

    def forward(self, x: Tensor, unused_scale: Tensor = None) -> Tuple[Tensor]:
        return self.module_forward(x)

import brevitas.nn as qnn
from aiter.utility import dtypes, fp4_utils
from aiter.ops.triton.gemm_afp4wfp4 import gemm_afp4wfp4
from aiter.ops.quant import per_1x32_f4_quant_triton, per_1x32_f4_quant
import math

# def triton_to_torch(x: torch.Tensor):
#     # Ensure float32 input
#     x = x.to(dtype=torch.float32)

#     # amax = tl.max(tl.abs(x), axis=1, keep_dims=True)
#     amax = x.abs().amax(dim=1, keepdim=True)

#     # amax = amax.to(tl.int32, bitcast=True)
#     amax_int = amax.view(torch.int32)

#     # amax = (amax + 0x200000).to(tl.uint32, bitcast=True) & 0xFF800000
#     amax_int = amax_int + 0x200000
#     amax_int = amax_int & 0xFF800000

#     # amax = amax.to(tl.float32, bitcast=True)
#     amax = amax_int.view(torch.float32)

#     # scale_e8m0_unbiased = tl.log2(amax).floor() - 2
#     scale_e8m0_unbiased = torch.floor(torch.log2(amax)) - 2.0

#     # scale_e8m0_unbiased = tl.clamp(scale_e8m0_unbiased, min=-127, max=127)
#     scale_e8m0_unbiased = torch.clamp(scale_e8m0_unbiased, -127, 127)

#     # quant_scale = tl.exp2(-scale_e8m0_unbiased)
#     quant_scale = torch.exp2(-scale_e8m0_unbiased)

#     return amax, scale_e8m0_unbiased, quant_scale


import wave_lang.kernel.lang as tkl
import wave_lang.kernel.wave as tkw
from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile
from wave_lang.kernel.wave.scheduling.schedule import SchedulingType
from wave_lang.kernel.wave.utils.run_utils import (
    set_default_run_config,
)
from wave_lang.kernel.lang.global_symbols import *
from wave_lang.kernel.wave.utils.general_utils import (
    get_default_scheduling_params,
    torch_dtype_to_wave,
)
from wave_lang.kernel.wave.constraints import (
    ScaledMMAType,
)

def get_mxfp4_gemm(shape, c_dtype, use_async=False):
    mfma_variant = ScaledMMAType.F32_16x16x128_F8F6F4
    c_wave_dtype = torch_dtype_to_wave(c_dtype)
    # Input sizes
    M = tkl.sym.M
    N = tkl.sym.N
    K = tkl.sym.K
    K_SCALE = tkl.sym.K_SCALE
    # Workgroup tile sizes
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K = tkl.sym.BLOCK_K
    # Address space (for GPU, shared(1) or global(0))
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    # Expose user-constraints
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 4)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64, waves_per_block=(4, 2, 1), mma_type=mfma_variant
        )
    ]

    @tkw.wave(constraints)
    def gemm_afp4_wfp4_wave(
        a: tkl.Memory[M, K / 2, ADDRESS_SPACE, tkl.i8],
        a_scale: tkl.Memory[M, K / 32, ADDRESS_SPACE, tkl.i8],
        b: tkl.Memory[N, K / 2, ADDRESS_SPACE, tkl.i8],
        b_scale: tkl.Memory[N, K / 32, ADDRESS_SPACE, tkl.i8],
        c: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.bf16],
    ):
        c_reg = tkl.Register[M, N, tkl.f32](0.0)

        @tkw.iterate(K, init_args=[c_reg])
        def repeat(acc: tkl.Register[M, N, tkl.f32]) -> tkl.Register[M, N, tkl.f32]:
            a_reg = tkw.read(a)
            a_reg = tkw.bitcast(a_reg, tkl.f4e2m1fn)
            a_scale_reg = tkw.read(a_scale)
            a_scale_reg = tkw.bitcast(a_scale_reg, tkl.f8e8m0fnu)
            b_reg = tkw.read(b)
            b_reg = tkw.bitcast(b_reg, tkl.f4e2m1fn)
            b_scale_reg = tkw.read(b_scale)
            b_scale_reg = tkw.bitcast(b_scale_reg, tkl.f8e8m0fnu)
            acc = tkw.scaled_mma(a_reg, a_scale_reg, b_reg, b_scale_reg, acc)
            return acc

        casted = tkw.cast(repeat, c_wave_dtype)
        tkw.write(casted, c)

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        BLOCK_M: 256,
        BLOCK_N: 256,
        BLOCK_K: 256,
        # M: shape[0],
        N: shape[1],
        K: shape[2],
        K_SCALE: shape[2] // 32,
    }
    hyperparams.update(get_default_scheduling_params())
    dynamic_symbols = [M]

    schedule = SchedulingType.PREFETCH
    if use_async:
        # TODO: Add scheduling async support
        schedule = SchedulingType.PREFETCH
    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        schedule=schedule,
        wave_runtime=True,
        dump_intermediates="./inter",
        use_buffer_ops=True,
        waves_per_eu=1,
        use_global_to_shared=use_async,
        minimize_shared_allocs=False,
        dynamic_symbols=dynamic_symbols
    )
    options = set_default_run_config(options)
    gemm = wave_compile(options, gemm_afp4_wfp4_wave)
    return gemm


class MXFp4LinearBase(torch.nn.Module):
    handled_layer = qnn.QuantLinear

    def attach_debug_info(self, module: nn.Module):
        pass

    def __init__(self):
        super().__init__()
        self.input_quant = GroupwiseFloatInferenceHandler()
        self.weight_quant = GroupwiseFloatWeightInferenceHandler()

    def prepare_for_export(self, module: nn.Module):
        if module.input_quant.is_quant_enabled:
            self.input_module_forward = module.input_quant.fused_activation_quant_proxy.tensor_quant
            self.group_dim = module.input_quant.group_dim
        if module.weight_quant.is_quant_enabled:
            self.weight_quant.prepare_for_export(module.weight_quant)
            self.weight = self.pre_quantize_weight(module.weight)
            self.scale = self.pre_quantize_weight_scale(module.weight_quant.scale)

    @abstractmethod
    def pre_quantize_weight_scale(self, scale):
        pass

    @abstractmethod
    def pre_quantize_weight(self, weight):
        pass

    @abstractmethod
    def quantize_inp(self, x: Tensor):
        pass    


class MXFp4Linear(torch.nn.Module):
    handled_layer = qnn.QuantLinear

    def attach_debug_info(self, module: nn.Module):
        pass

    def __init__(self):
        super().__init__()
        self.input_quant = GroupwiseFloatInferenceHandler()
        self.weight_quant = GroupwiseFloatWeightInferenceHandler()

    def prepare_for_export(self, module: nn.Module):
        if module.input_quant.is_quant_enabled:
            self.input_module_forward = module.input_quant.fused_activation_quant_proxy.tensor_quant
            self.group_dim = module.input_quant.group_dim
        if module.weight_quant.is_quant_enabled:
            self.weight_quant.prepare_for_export(module.weight_quant)
            self.weight = self.pre_quantize_weight(module.weight)
            self.scale = self.pre_quantize_scale(self.weight_quant.scale)

    def pre_quantize_scale(self, scale):
        return scale

    def pre_quantize_weight(self, weight):
        weight = self.weight_quant(weight)[0]
        return weight

    def quantize_inp(self, x: Tensor):
        orig_inp_shape = x.shape
        x = x.reshape(-1, orig_inp_shape[-1])
        quantizable_inp_shape = x.shape

        x, xs, zero_point, *other = self.input_module_forward(x)
        x, xs_expanded = groupwise_dequant_expand(x, xs, zero_point, self.group_dim, quantizable_inp_shape)[0:2]

        return x, xs_expanded
            
    def forward(self, x: Tensor) -> Tuple[Tensor]:
        # In inference mode, we never return quant tensors

        orig_inp_shape = x.shape
        x = x.reshape(-1, orig_inp_shape[-1])

        x, expanded_x_scale_0 = self.quantize_inp(x)

        weight = self.weight

        out = torch.nn.functional.linear(x, weight)
        out = out.reshape(*orig_inp_shape[:-1], out.shape[-1])
        return out



class MXFp4LinearTriton(torch.nn.Module):
    handled_layer = qnn.QuantLinear

    def attach_debug_info(self, module: nn.Module):
        pass

    def __init__(self):
        super().__init__()
        self.input_quant = GroupwiseFloatInferenceHandler()
        self.weight_quant = GroupwiseFloatWeightInferenceHandler()

    def prepare_for_export(self, module: nn.Module):
        if module.input_quant.is_quant_enabled:
            self.input_module_forward = module.input_quant.fused_activation_quant_proxy.tensor_quant
            self.group_dim = module.input_quant.group_dim
        if module.weight_quant.is_quant_enabled:
            self.weight_quant.prepare_for_export(module.weight_quant)
            self.weight = self.pre_quantize_weight(module.weight)
            self.scale = self.pre_quantize_scale(self.weight_quant.scale)
        N,K = module.weight.shape
        wave_shape = (1, N, K)
        self.gemm = get_mxfp4_gemm(wave_shape, module.weight.dtype)

    def pre_quantize_scale(self, scale):
        weight_shape = self.weight.shape
        return self.post_process_quant_triton(scale, weight_shape)

    def pre_quantize_weight(self, weight):
        weight, scale = self.weight_quant(weight)[0:2]
        weight = fp4_utils.f32_to_mxfp4((weight/scale).float())
        return weight

    def post_process_quant_triton(self, scale, inp_shape):
        # Scale post-process
        first_shape = inp_shape[0]
        reshaped_scale = scale.view(first_shape, -1) 
        expanded_shape = math.ceil(reshaped_scale.shape[0]/256) * 256, math.ceil(reshaped_scale.shape[1]/8) * 8
        padded_scale = 127 * torch.ones(expanded_shape[0], expanded_shape[1], dtype=torch.uint8, device=reshaped_scale.device)
        padded_scale[:reshaped_scale.shape[0], :reshaped_scale.shape[1]] = torch.log2(reshaped_scale) + 127
        padded_scale = padded_scale.view(torch.float8_e8m0fnu)
        return padded_scale

    def quantize_inp(self, x: Tensor):
        return per_1x32_f4_quant_triton(x, shuffle=False)
            
    def forward(self, x: Tensor) -> Tuple[Tensor]:
        # In inference mode, we never return quant tensors

        orig_inp_shape = x.shape
        orig_dtype = x.dtype
        x = x.reshape(-1, orig_inp_shape[-1])
        M,K = x.shape
        N,K = self.weight.shape
        wave_shape = (M, N, K)

        x, expanded_x_scale_0 = self.quantize_inp(x)

        weight = self.weight
        expanded_w_scale_0 = self.scale
        # self.gemm = get_mxfp4_gemm(wave_shape, orig_dtype)

        wave_out = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
        triton_out = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
        self.gemm(x.view(torch.uint8), expanded_x_scale_0.view(torch.uint8), weight.view(torch.uint8), expanded_w_scale_0.view(torch.uint8), torch.bfloat16, wave_out)
        gemm_afp4wfp4(x.view(torch.uint8), weight.view(torch.uint8), expanded_x_scale_0.view(torch.uint8), expanded_w_scale_0.view(torch.uint8), torch.bfloat16, triton_out)
        triton_out = triton_out.reshape(*orig_inp_shape[:-1], triton_out.shape[-1])
        return triton_out
