# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABC

import torch

from brevitas.export.common.handler.base import BaseHandler
from brevitas.export.common.handler.qcdq import CDQCastBiasQuantProxyHandlerMixin
from brevitas.export.common.handler.qcdq import CDQCastMixin
from brevitas.export.common.handler.qcdq import DQCastMixin
from brevitas.export.common.handler.qcdq import FloatQCDQCastActQuantProxyHandlerMixin
from brevitas.export.common.handler.qcdq import FloatQCDQCastWeightQuantProxyHandlerMixin
from brevitas.export.common.handler.qcdq import FloatQMixin
from brevitas.export.common.handler.qcdq import QCDQCastActQuantProxyHandlerMixin
from brevitas.export.common.handler.qcdq import QCDQCastDecoupledWeightQuantProxyHandlerMixin
from brevitas.export.common.handler.qcdq import \
    QCDQCastDecoupledWeightQuantWithInputProxyHandlerMixin
from brevitas.export.common.handler.qcdq import QCDQCastTruncQuantProxyHandlerMixin
from brevitas.export.common.handler.qcdq import QCDQCastWeightQuantProxyHandlerMixin
from brevitas.export.common.handler.qcdq import QMixin


def _itemize_clip_bounds(clip_args):
    if clip_args is not None:
        clip_args['min_val'] = clip_args['min_val'].item()
        clip_args['max_val'] = clip_args['max_val'].item()
    return clip_args


LIBRARY = torch.library.Library("brevitas", 'DEF')
LIBRARY.define("quantize(Tensor x, Tensor scale, Tensor zero_point,  int axis) -> Tensor")
LIBRARY.define("dequantize(Tensor x, Tensor scale, Tensor zero_point, int axis) -> Tensor")


def quantize_fn(x, scale, zero_point, axis):
    return torch.empty_like(x).type(zero_point.dtype)


def dequantize_fn(x, scale, zero_point, axis):
    return torch.empty_like(x).type(scale.dtype)


LIBRARY.impl("quantize", quantize_fn, "Meta")
LIBRARY.impl("dequantize", dequantize_fn, "Meta")


class TorchDQCastMixin(DQCastMixin, ABC):

    def __init__(self) -> None:
        super().__init__()
        self.symbolic_kwargs = {}

    def dequantize_fn(self, x, scale, zero_point, axis):
        if axis is None:
            axis = -1
        # cast zero_point to float, otherwise if both x
        # and zero_point are uint (as in asym quant)
        # uint - uint can lead to errors. Don't cast x to float
        # as the main float datatype might not be float32 (e.g float16)
        return torch.ops.brevitas.dequantize(x, scale, zero_point, axis)
        # if isinstance(zero_point, torch.Tensor):
        #     zero_point = zero_point.to(torch.float)
        # else:
        #     zero_point = float(zero_point)
        # return (x - zero_point) * scale

    def cast_fn(self, x, dtype):
        return x.type(dtype)

    @property
    def flatten_dequantize_params(self):
        return False

    @property
    def itemize_quantize_scalar_params(self):
        return False

    def validate(self, module):
        assert module.bit_width() > 1., 'Binary quant not supported'


class TorchCDQCastMixin(CDQCastMixin, TorchDQCastMixin, ABC):

    def clip_fn(self, x, min_val, max_val):
        return torch.clamp(x, min_val, max_val)


class TorchQCDQCastMixin(QMixin, TorchCDQCastMixin, ABC):

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
        if getattr(self, '_export_q_node', True):
            assert module.rounding_mode.upper() == 'ROUND', 'Only round to nearest even supported'
        assert not module.is_groupwise, "Export with Per Group quantization not supported"

    def quantize_fn(self, x, scale, zero_point, dtype, axis):
        if axis is None:
            axis = -1
        return torch.ops.brevitas.quantize(x, scale, zero_point, axis)

        # if axis is None:
        #     y = torch.quantize_per_tensor(x, scale, zero_point, dtype)
        # else:
        #     y = torch.quantize_per_channel(x, scale, zero_point, axis, dtype)
        # return y.int_repr()


class StdFloatDQCastONNXMixin(TorchDQCastMixin, ABC):

    def is_ocp(self, module):
        is_e4m3 = module.mantissa_bit_width() == 3 and module.exponent_bit_width() == 4

        is_ocp_e4m3 = is_e4m3 and module.inf_values() is None and module.nan_values() == (('111',))

        is_e5m2 = module.mantissa_bit_width() == 5 and module.exponent_bit_width() == 2

        is_ocp_e5m2 = is_e5m2 and module.inf_values() == (
            ('00',)) and module.nan_values() == ('01', '11', '10')

        return is_ocp_e4m3 or is_ocp_e5m2

    def validate(self, module):
        assert self.is_ocp(module), 'Only OCP Standard is supported for FP8 export'


class StdFloatCDQCastONNXMixin(CDQCastMixin, StdFloatDQCastONNXMixin, ABC):

    def clip_fn(self, x, min_val, max_val):
        raise NotImplementedError


class StdFloatQCDQCastONNXMixin(FloatQMixin, StdFloatCDQCastONNXMixin, ABC):

    def validate(self, module):
        if getattr(self, '_export_q_node', True):
            assert module.rounding_mode.upper() == 'ROUND', 'Only round to nearest even supported'
        super().validate(module)

    def quantize_fn(self, x, scale, zero_point, dtype, axis):
        if axis is None:
            axis = -1
        return torch.ops.brevitas.quantize(x, scale, zero_point, axis)


class TorchQCDQHandler(BaseHandler):

    def forward(self, *args, **kwargs):
        return self.symbolic_execution(*args, **kwargs)


class TorchFloatQCDQCastWeightQuantProxyHandler(StdFloatQCDQCastONNXMixin,
                                                FloatQCDQCastWeightQuantProxyHandlerMixin,
                                                TorchQCDQHandler):
    _export_q_node = False


class TorchFloatQCDQCastActQuantProxyHandler(StdFloatQCDQCastONNXMixin,
                                             FloatQCDQCastActQuantProxyHandlerMixin,
                                             TorchQCDQHandler):
    pass


class TorchQCDQCastWeightQuantProxyHandler(TorchQCDQCastMixin,
                                           QCDQCastWeightQuantProxyHandlerMixin,
                                           TorchQCDQHandler):
    _export_q_node = False

    @classmethod
    def int_clip_symbolic_kwargs(cls, narrow, signed, bit_width):
        clip_args = super().int_clip_symbolic_kwargs(narrow, signed, bit_width)
        return _itemize_clip_bounds(clip_args)


class TorchQCDQCastDecoupledWeightQuantProxyHandler(TorchQCDQCastMixin,
                                                    QCDQCastDecoupledWeightQuantProxyHandlerMixin,
                                                    TorchQCDQHandler):
    _export_q_node = False

    @classmethod
    def int_clip_symbolic_kwargs(cls, narrow, signed, bit_width):
        clip_args = super().int_clip_symbolic_kwargs(narrow, signed, bit_width)
        return _itemize_clip_bounds(clip_args)


class TorchQCDQCastDecoupledWeightQuantWithInputProxyHandler(
        TorchQCDQCastMixin, QCDQCastDecoupledWeightQuantWithInputProxyHandlerMixin,
        TorchQCDQHandler):
    _export_q_node = False

    @classmethod
    def int_clip_symbolic_kwargs(cls, narrow, signed, bit_width):
        clip_args = super().int_clip_symbolic_kwargs(narrow, signed, bit_width)
        return _itemize_clip_bounds(clip_args)


class TorchQCDQCastActQuantProxyHandler(TorchQCDQCastMixin,
                                        QCDQCastActQuantProxyHandlerMixin,
                                        TorchQCDQHandler):

    @classmethod
    def int_clip_symbolic_kwargs(cls, narrow, signed, bit_width):
        clip_args = super().int_clip_symbolic_kwargs(narrow, signed, bit_width)
        return _itemize_clip_bounds(clip_args)


class TorchCDQCastBiasQuantProxyHandler(TorchDQCastMixin,
                                        CDQCastBiasQuantProxyHandlerMixin,
                                        TorchQCDQHandler):
    pass


class TorchQCDQCastTruncQuantProxyHandler(TorchQCDQCastMixin,
                                          QCDQCastTruncQuantProxyHandlerMixin,
                                          TorchQCDQHandler):

    @classmethod
    def int_clip_symbolic_kwargs(cls, narrow, signed, bit_width):
        clip_args = super().int_clip_symbolic_kwargs(narrow, signed, bit_width)
        return _itemize_clip_bounds(clip_args)
