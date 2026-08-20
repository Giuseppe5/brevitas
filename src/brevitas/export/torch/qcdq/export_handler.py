# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABC

import torch
import torch.ao.quantization.fx._decomposed  # noqa: F401

from brevitas.export.common.handler.qcdq import FloatQCDQCastActQuantProxyHandlerMixin
from brevitas.export.common.handler.qcdq import FloatQCDQCastWeightQuantProxyHandlerMixin
from brevitas.export.common.handler.qcdq import QCDQCastActQuantProxyHandlerMixin
from brevitas.export.common.handler.qcdq import QCDQCastWeightQuantProxyHandlerMixin
from brevitas.export.common.handler.qcdq import QMixin
from brevitas.export.inference.handler import GroupwiseFloatInferenceHandler
from brevitas.export.inference.handler import GroupwiseFloatWeightInferenceHandler
from brevitas.function.ops import max_int
from brevitas.function.ops import min_int
from brevitas.proxy.groupwise_float_parameter_quant import \
    GroupwiseWeightFloatQuantProxyFromInjector
from brevitas.proxy.groupwise_float_runtime_quant import GroupwiseActFloatQuantProxyFromInjector
from brevitas.utils.quant_utils import groupwise_dequant_expand

from .custom_ops import dequantize_fp8
from .custom_ops import dequantize_mx_fp4
from .custom_ops import dequantize_mx_fp8
from .custom_ops import quantize_fp8
from .custom_ops import quantize_mx_fp4
from .custom_ops import quantize_mx_fp8
from .handler import TorchCDQCastMixin
from .handler import TorchQCDQHandler


class TorchExportQCDQCastMixin(QMixin, TorchCDQCastMixin, ABC):

    @classmethod
    def int_clip_symbolic_kwargs(cls, narrow, signed, bit_width):
        # quantized_decomposed Q/DQ operators already enforce quant_min/quant_max.
        return None

    @classmethod
    def int8_dtype(cls):
        return torch.int8

    @classmethod
    def uint8_dtype(cls):
        return torch.uint8

    @classmethod
    def int32_dtype(cls):
        return torch.int32

    def validate(self, module):
        super().validate(module)
        assert not module.is_groupwise, 'Integer groupwise export is not supported'
        assert module.rounding_mode.upper() == 'ROUND', 'Only round to nearest even supported'
        self.quant_min = int(
            min_int(module.is_signed, module.is_narrow_range, module.bit_width()).item())
        self.quant_max = int(
            max_int(module.is_signed, module.is_narrow_range, module.bit_width()).item())
        self.storage_dtype = self.signed_dtype(module.bit_width(), module.is_signed)

    def quantize_fn(self, x, scale, zero_point, dtype, axis):
        if axis is None:
            return torch.ops.quantized_decomposed.quantize_per_tensor(
                x, scale, zero_point, self.quant_min, self.quant_max, dtype)
        return torch.ops.quantized_decomposed.quantize_per_channel(
            x, scale, zero_point, axis, self.quant_min, self.quant_max, dtype)

    def dequantize_fn(self, x, scale, zero_point, axis):
        if axis is None:
            return torch.ops.quantized_decomposed.dequantize_per_tensor.tensor(
                x,
                scale,
                zero_point,
                self.quant_min,
                self.quant_max,
                self.storage_dtype,
                out_dtype=torch.float32)
        return torch.ops.quantized_decomposed.dequantize_per_channel(
            x,
            scale,
            zero_point,
            axis,
            self.quant_min,
            self.quant_max,
            self.storage_dtype,
            out_dtype=torch.float32)


class TorchExportFloatQCDQCastMixin(TorchCDQCastMixin, ABC):

    @classmethod
    def signed_dtype(cls, exponent_bit_width, mantissa_bit_width, is_ocp, is_fnuz):
        assert is_ocp and not is_fnuz, 'Only OCP FP8 export is supported'
        if exponent_bit_width == 4 and mantissa_bit_width == 3:
            return torch.float8_e4m3fn
        if exponent_bit_width == 5 and mantissa_bit_width == 2:
            return torch.float8_e5m2
        raise RuntimeError('Only FP8 E4M3 and E5M2 export is supported')

    def validate(self, module):
        assert module.is_ocp and not module.is_groupwise, \
            'Only non-groupwise OCP FP8 is supported'
        assert module.is_saturating(), 'Only saturating OCP FP8 export is supported'
        assert module.rounding_mode.upper() == 'ROUND', 'Only round to nearest even supported'
        exponent_bit_width = int(module.exponent_bit_width().item())
        mantissa_bit_width = int(module.mantissa_bit_width().item())
        self.fp8_dtype = self.signed_dtype(
            exponent_bit_width, mantissa_bit_width, module.is_ocp, module.is_fnuz)
        self.min_val = float(torch.finfo(self.fp8_dtype).min)
        self.max_val = float(torch.finfo(self.fp8_dtype).max)

    def quantize_fn(self, x, scale, zero_point, dtype, axis):
        return quantize_fp8(x, scale, dtype, axis, self.min_val, self.max_val)

    def dequantize_fn(self, x, scale, zero_point, axis):
        return dequantize_fp8(x, scale, axis, torch.float32)

    @property
    def flatten_dequantize_params(self):
        return False

    @property
    def itemize_quantize_scalar_params(self):
        return False

    def clip_fn(self, x, min_val, max_val):
        return x


class TorchExportQCDQWeightQuantProxyHandler(TorchExportQCDQCastMixin,
                                             QCDQCastWeightQuantProxyHandlerMixin,
                                             TorchQCDQHandler):
    _export_q_node = True


class TorchExportQCDQActQuantProxyHandler(TorchExportQCDQCastMixin,
                                          QCDQCastActQuantProxyHandlerMixin,
                                          TorchQCDQHandler):
    pass


class TorchExportFloatQCDQWeightQuantProxyHandler(TorchExportFloatQCDQCastMixin,
                                                  FloatQCDQCastWeightQuantProxyHandlerMixin,
                                                  TorchQCDQHandler):
    _export_q_node = True


class TorchExportFloatQCDQActQuantProxyHandler(TorchExportFloatQCDQCastMixin,
                                               FloatQCDQCastActQuantProxyHandlerMixin,
                                               TorchQCDQHandler):
    pass


class TorchExportMXFloatMixin:

    def prepare_for_export(self, module):
        super().prepare_for_export(module)
        if module.is_quant_enabled:
            if not module.is_ocp or module.is_fnuz:
                raise RuntimeError('MX export supports only OCP formats')
            if module.rounding_mode.upper() != 'ROUND':
                raise RuntimeError('MX export supports only round to nearest even')
            if not module.is_saturating():
                raise RuntimeError('MX export supports only saturating quantizers')
            if isinstance(module, GroupwiseActFloatQuantProxyFromInjector):
                zero_point = module._cached_act.zero_point_
            elif isinstance(module, GroupwiseWeightFloatQuantProxyFromInjector):
                zero_point = module._cached_weight.zero_point_
            else:
                zero_point = module.zero_point()
            if torch.any(zero_point != 0).item():
                raise RuntimeError('MX export requires zero point to be zero')
            exponent_bit_width = int(module.exponent_bit_width().item())
            mantissa_bit_width = int(module.mantissa_bit_width().item())
            self.group_size_value = int(module.group_size)
            self.group_dim_value = int(module.group_dim)
            if self.group_size_value != 32:
                raise RuntimeError('MX export requires group_size=32')
            if exponent_bit_width == 4 and mantissa_bit_width == 3:
                self.mx_format = 'fp8'
                self.fp8_dtype = torch.float8_e4m3fn
                self.min_val = float(torch.finfo(self.fp8_dtype).min)
                self.max_val = float(torch.finfo(self.fp8_dtype).max)
            elif exponent_bit_width == 2 and mantissa_bit_width == 1:
                self.mx_format = 'fp4'
            else:
                raise RuntimeError('MX export supports only E4M3 FP8 and E2M1 FP4')
            self.cached_weight = None

    def quantize(self, x, scale, zero_point):
        if self.mx_format == 'fp8':
            return quantize_mx_fp8(
                x,
                scale,
                self.group_size_value,
                self.group_dim_value,
                self.fp8_dtype,
                self.min_val,
                self.max_val)
        return quantize_mx_fp4(x, scale, self.group_size_value, self.group_dim_value)

    def dequantize(self, x, scale, output_dtype):
        if self.mx_format == 'fp8':
            return dequantize_mx_fp8(
                x, scale, self.group_size_value, self.group_dim_value, output_dtype)
        return dequantize_mx_fp4(
            x, scale, self.group_size_value, self.group_dim_value, output_dtype)

    def inner_forward(self, x, scale, zero_point):
        return self.dequantize(self.quantize(x, scale, zero_point), scale, x.dtype)


class TorchExportMXFloatWeightQuantProxyHandler(TorchExportMXFloatMixin,
                                                GroupwiseFloatWeightInferenceHandler):
    handled_layer = GroupwiseWeightFloatQuantProxyFromInjector

    def quantize_forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.scale
        zero_point = self.zero_point
        if self.cached_weight is not None:
            out = self.cached_weight
        else:
            inp_shape = x.shape
            grouped_x = self.reshape_input(x)
            out = self.quantize(grouped_x, scale, zero_point)
            out = groupwise_dequant_expand(out, scale, zero_point, self.group_dim_value,
                                           inp_shape)[0]
        return out, None

    def forward(self, x: torch.Tensor):
        assert self.skip_create_quant_tensor
        scale = self.scale
        zero_point = self.zero_point
        if self.cached_weight is not None:
            out = self.cached_weight
        else:
            inp_shape = x.shape
            grouped_x = self.reshape_input(x)
            out = self.inner_forward(grouped_x, scale, zero_point)
            out = groupwise_dequant_expand(out, scale, zero_point, self.group_dim_value,
                                           inp_shape)[0]

        return (
            out,
            scale,
            zero_point,
            self.exponent_bit_width,
            self.mantissa_bit_width,
            self.exponent_bias,
            self.saturating,
            self.inf_values,
            self.nan_values)

    def reshape_input(self, x: torch.Tensor):
        from brevitas.core.function_wrapper.shape import dynamic_over_sub_channel_block_view
        return dynamic_over_sub_channel_block_view(x, self.group_size_value, self.group_dim_value)


class TorchExportMXFloatActQuantProxyHandler(TorchExportMXFloatMixin,
                                             GroupwiseFloatInferenceHandler):
    handled_layer = GroupwiseActFloatQuantProxyFromInjector

    def forward(self, x):
        assert self.skip_create_quant_tensor
        inp_shape = x.shape
        _, scale, zero_point, *other = self.module_forward(x)
        grouped_x = self.reshape_input(x)
        out = self.inner_forward(grouped_x, scale, zero_point)
        out = groupwise_dequant_expand(out, scale, zero_point, self.group_dim_value, inp_shape)[0]
        return tuple([out, scale, zero_point] + list(other))

    def reshape_input(self, x):
        from brevitas.core.function_wrapper.shape import dynamic_over_sub_channel_block_view
        return dynamic_over_sub_channel_block_view(x, self.group_size_value, self.group_dim_value)


__all__ = [
    'TorchExportFloatQCDQActQuantProxyHandler',
    'TorchExportFloatQCDQWeightQuantProxyHandler',
    'TorchExportMXFloatActQuantProxyHandler',
    'TorchExportMXFloatWeightQuantProxyHandler',
    'TorchExportQCDQActQuantProxyHandler',
    'TorchExportQCDQWeightQuantProxyHandler']
