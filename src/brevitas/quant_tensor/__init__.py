# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABC
from typing import NamedTuple, Optional
import warnings

import torch
from torch import Tensor

import brevitas.config as config
from brevitas.function.ops import max_int
from brevitas.function.ops import min_int
from brevitas.function.ops_ste import ceil_ste
from brevitas.function.ops_ste import round_ste

from .torch_handler import QUANT_TENSOR_FN_HANDLER

IS_VALID_ATOL = 2e-1
BFLOAT16_IS_VALID_ATOL = 0.5


def _get_dequantize_tensor(input):
    return input.value if isinstance(input, QuantTensor) else input


class QuantTensorBase(NamedTuple):
    qt_value: Optional[Tensor]
    scale: Optional[Tensor]
    zero_point: Optional[Tensor]
    bit_width: Optional[Tensor]
    signed_t: Optional[Tensor]
    training_t: Optional[Tensor]


def _unpack_quant_tensor(input_data):
    if isinstance(input_data, QuantTensor):
        return input_data.value
    elif isinstance(input_data, tuple):
        return tuple([_unpack_quant_tensor(v) for v in input_data])
    elif isinstance(input_data, list):
        return [_unpack_quant_tensor(v) for v in input_data]
    elif isinstance(input_data, dict):
        return {k: _unpack_quant_tensor(v) for k, v in input_data.items()}
    else:
        return input_data


def _is_all_nested_not_none(input_data):
    if isinstance(input_data, QuantTensor):
        return input_data.is_not_none
    elif isinstance(input_data, (tuple, list)):
        return all([_is_all_nested_not_none(v) for v in input_data])
    elif isinstance(input_data, dict):
        return all([_is_all_nested_not_none(v) for v in input_data.values()])
    else:
        return True


class QuantTensor(QuantTensorBase):

    def __new__(
            cls,
            qt_value=None,
            scale=None,
            zero_point=None,
            bit_width=None,
            signed=None,
            training=None,
            _allow_empty=False):

        if scale is not None and not isinstance(scale, torch.Tensor):
            scale = torch.tensor(scale, dtype=torch.float)
        if zero_point is not None and not isinstance(zero_point, torch.Tensor):
            zero_point = torch.tensor(zero_point, dtype=torch.float)
        if bit_width is not None and not isinstance(bit_width, torch.Tensor):
            bit_width = torch.tensor(bit_width, dtype=torch.float)
        if signed is not None and not isinstance(signed, torch.Tensor):
            signed = torch.tensor(signed, dtype=torch.bool)
        if training is not None and not isinstance(training, torch.Tensor):
            training = torch.tensor(training, dtype=torch.bool)

        if _allow_empty:
            warnings.warn(
                "Empty QuantTensor are deprecated and will be removed in a future version")
        # elif value is not None and scale is not None and zero_point is not None:
        #     is_int = torch.allclose(torch.round(int_value), int_value)
        #     if not is_int:
        #         quant_tensor = quant_tensor.set(int_value = torch.round(int_value / scale + zero_point))
        # elif int_value is None and value is not None:
        #     pass
        elif not _allow_empty and (scale is None or bit_width is None or zero_point is None):
            raise RuntimeError("To create an emtpy QuantTensor, set _allow_empty=True")

        quant_tensor = super().__new__(
            cls, qt_value, scale, zero_point, bit_width, signed, training)
        return quant_tensor

    @classmethod
    def from_fake_quantized(cls, fake_quant_value, scale, zero_point, bit_width, signed, training):
        if config._ONGOING_EXPORT:
            quant_tensor = fake_quant_value
        else:
            quant_tensor = torch.round(fake_quant_value / scale + zero_point)
        return cls(quant_tensor, scale, zero_point, bit_width, signed, training)

    @property
    def signed(self):
        if self.signed_t is not None:
            return self.signed_t.item()
        else:
            return None

    @property
    def training(self):
        if self.training_t is not None:
            return self.training_t.item()
        else:
            return None

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if (func not in QUANT_TENSOR_FN_HANDLER or
                not all(issubclass(t, QuantTensor) for t in types) or
                not (_is_all_nested_not_none(args) and _is_all_nested_not_none(kwargs))):
            args = _unpack_quant_tensor(args)
            kwargs = _unpack_quant_tensor(kwargs)
            return func(*args, **kwargs)
        return QUANT_TENSOR_FN_HANDLER[func](*args, **kwargs)

    @property
    def tensor(self):
        return self.value

    @property
    def value(self):
        if self.is_valid and not config._ONGOING_EXPORT:
            if self.zero_point is None or self.scale is None:
                return self.qt_value
            return (self.qt_value - self.zero_point) * self.scale
        else:
            return self.qt_value

    @property
    def is_not_none(self):
        return (
            self.value is not None and self.scale is not None and self.zero_point is not None and
            self.bit_width is not None and self.signed is not None)

    @property
    def _pre_round_int_value(self):
        value = self.value
        scale = self.scale
        zero_point = self.zero_point
        if self.scale.dtype == torch.bfloat16:
            value = self.value.type(torch.float32)
            scale = self.scale.type(torch.float32)
            zero_point = self.zero_point.type(torch.float32)
        int_value = value / scale
        int_value = int_value + zero_point
        return int_value

    @property
    def is_valid(self):
        if torch.allclose(self.qt_value.to(torch.int).to(self.qt_value.dtype),
                          self.qt_value,
                          rtol=0,
                          atol=0):
            return True
        else:
            return False

    #     return True
    #     # if self.is_not_none:
    #     #     with torch.no_grad():
    #     #         pre_round_int_value = self._pre_round_int_value
    #     #         rounded_int_value = torch.round(pre_round_int_value)
    #     #         is_int = torch.isclose(
    #     #             pre_round_int_value, rounded_int_value, atol=IS_VALID_ATOL).all()
    #     #         if self.bit_width >= 2:
    #     #             if self.signed:
    #     #                 is_upper_b = (2.0 ** (self.bit_width - 1) - 1 >= rounded_int_value).all()
    #     #                 is_lower_b = (-2.0 ** (self.bit_width - 1) <= rounded_int_value).all()
    #     #             else:
    #     #                 is_upper_b = (2.0 ** self.bit_width - 1 >= rounded_int_value).all()
    #     #                 is_lower_b = (0. <= rounded_int_value).all()
    #     #             return (is_int & is_upper_b & is_lower_b).item()
    #     #         else:  # binary case
    #     #             unique_vals = rounded_int_value.unique(
    #     #                 sorted=False, return_counts=False, return_inverse=False)
    #     #             is_binary = unique_vals.view(-1).size()[0] == 2
    #     #             is_signed = (unique_vals < 0.).any().item()
    #     #             sign_match = is_signed == self.signed
    #     #             return is_int.item() and is_binary and sign_match
    #     # else:
    #     #     return False

    @property
    def device(self):
        value_device = self.qt_value.device
        is_same_device = True
        for t in [self.scale, self.zero_point, self.bit_width]:
            if t is not None:
                is_same_device &= value_device == t.device
        if not is_same_device:
            raise RuntimeError("Value and metadata are on different devices")
        return value_device

    def set(self, **kwargs):
        return self._replace(**kwargs)

    def detach_(self):
        self.qt_value.detach_()
        self.scale.detach_()
        self.zero_point.detach_()
        self.bit_width.detach_()

    def detach(self):
        return QuantTensor(
            self.qt_value.detach(),
            self.scale.detach() if self.scale is not None else None,
            self.zero_point.detach() if self.zero_point is not None else None,
            self.bit_width.detach() if self.bit_width is not None else None,
            self.signed,
            self.training)

    def contiguous(self):
        return QuantTensor(
            self.qt_value.contiguous(),
            self.scale.contiguous() if self.scale is not None else None,
            self.zero_point.contiguous() if self.zero_point is not None else None,
            self.bit_width.contiguous() if self.bit_width is not None else None,
            self.signed,
            self.training)

    def int(self, float_datatype=False):
        if self.is_valid:
            int_value = self.qt_value
            if float_datatype:
                return int_value
            else:
                if self.bit_width <= 8. and self.signed_t.item():
                    return int_value.to(torch.int8)
                elif self.bit_width <= 8. and not self.signed_t.item():
                    return int_value.to(torch.uint8)
                else:
                    return int_value.to(torch.int32)
        else:
            raise RuntimeError(f"QuantTensor not valid.")

    @staticmethod
    def check_input_type(tensor):
        if not isinstance(tensor, QuantTensor):
            raise RuntimeError("Tensor is not a QuantTensor")

    @staticmethod
    def is_zero_zero_point(tensor):
        QuantTensor.check_input_type(tensor)
        if tensor.zero_point is not None:
            return (tensor.zero_point == 0.).all()
        else:
            return None

    def check_scaling_factors_same(self, other):
        if not torch.allclose(self.scale, other.scale):
            return False
        return True

    def check_zero_points_same(self, other):
        if self.training is not None and self.training:
            return True
        if not torch.allclose(self.zero_point, other.zero_point):
            raise RuntimeError("Zero points are different")

    def check_bit_width_same(self, other):
        if not torch.allclose(self.bit_width, other.bit_width):
            raise RuntimeError("Bit widths are different")

    def check_sign_same(self, other):
        if not self.signed == other.signed:
            raise RuntimeError("Signs are different")

    def view(self, *args, **kwargs):
        return self.set(int_value=self.qt_value.view(*args, **kwargs))

    def reshape(self, *args, **kwargs):
        return self.set(int_value=self.qt_value.reshape(*args, **kwargs))

    def flatten(self, *args, **kwargs):
        return self.set(int_value=self.qt_value.flatten(*args, **kwargs))

    def transpose(self, *args, **kwargs):
        qt_value = self.qt_value.transpose(*args, **kwargs)
        tensor_meta = {
            'scale': self.scale, 'zero_point': self.zero_point, 'bit_width': self.bit_width}
        for k, tm in tensor_meta.items():
            if tm is not None and len(qt_value.shape) == len(tm.shape):
                tensor_meta[k] = tm.transpose(*args, **kwargs)
        return self.set(qt_value=qt_value, **tensor_meta)

    def permute(self, *args, **kwargs):
        int_value = self.qt_value.permute(*args, **kwargs)
        tensor_meta = {
            'scale': self.scale, 'zero_point': self.zero_point, 'bit_width': self.bit_width}
        for k, tm in tensor_meta.items():
            if tm is not None and len(int_value.shape) == len(tm.shape):
                tensor_meta[k] = tm.permute(*args, **kwargs)
        return self.set(int_value=int_value, **tensor_meta)

    def size(self, *args, **kwargs):
        return self.qt_value.size(*args, **kwargs)

    @property
    def shape(self):
        return self.qt_value.shape

    def dim(self):
        return self.qt_value.dim()

    def add(self, other):
        return self + other

    @staticmethod
    def cat(tensors, dim, out=None):
        if out is not None:
            raise RuntimeError("Out not supported.")
        if len(tensors) < 2:
            return tensors[0]
        else:
            first_qt = tensors[0]
            if all([isinstance(qt, QuantTensor) and qt.is_not_none for qt in tensors]):
                for qt in tensors[1:]:
                    is_output_qt_valid = True
                    if first_qt.training is not None and first_qt.training:
                        if not (first_qt.check_scaling_factors_same(qt) and
                                first_qt.check_zero_points_same(qt) and
                                first_qt.check_bit_width_same(qt) and first_qt.check_sign_same(qt)):
                            is_output_qt_valid = False
                    else:
                        first_qt.check_scaling_factors_same(qt)
                        first_qt.check_zero_points_same(qt)
                        first_qt.check_bit_width_same(qt)
                        first_qt.check_sign_same(qt)

                output_training = any([qt.training for qt in tensors])
                if output_training:
                    output_scale = sum([qt.scale for qt in tensors]) / len(tensors)
                    output_zero_point = sum([qt.zero_point for qt in tensors]) / len(tensors)
                    output_bit_width = sum([qt.bit_width for qt in tensors]) / len(tensors)
                else:  # at eval time, they are the same
                    output_scale = first_qt.scale
                    output_zero_point = first_qt.zero_point
                    output_bit_width = first_qt.bit_width
                output_signed = first_qt.signed  # they are the same
                output_value = torch.cat([qt.int() for qt in tensors],
                                         dim=dim) if is_output_qt_valid else torch.cat(
                                             [qt.value for qt in tensors], dim=dim)

                output = QuantTensor(
                    output_value,
                    scale=output_scale,
                    zero_point=output_zero_point,
                    bit_width=output_bit_width,
                    signed=output_signed,
                    training=output_training)

                return output
            else:
                tensors = [qt.value if isinstance(qt, QuantTensor) else qt for qt in tensors]
                output_value = torch.cat(tensors, dim=dim)
                return output_value

    # Reference: https://docs.python.org/3/reference/datamodel.html#emulating-numeric-types

    def __neg__(self):
        neg_value = (-self.int(float_datatype=True) - self.zero_point) * self.scale
        # In case the dtype of self.int is different from the one of the scale
        neg_value = neg_value.type(self.scale.dtype)
        if self.signed:
            return QuantTensor(
                int_value=neg_value,
                scale=self.scale,
                zero_point=self.zero_point,
                bit_width=self.bit_width,
                signed=self.signed,
                training=self.training)
        else:
            return QuantTensor(
                int_value=neg_value,
                scale=self.scale,
                zero_point=self.zero_point,
                bit_width=self.bit_width + 1,
                signed=True,
                training=self.training)

    def to(self, *args, **kwargs):
        return QuantTensor(
            self.qt_value.to(*args, **kwargs),
            self.scale.to(*args, **kwargs) if self.scale is not None else None,
            self.zero_point.to(*args, **kwargs) if self.zero_point is not None else None,
            self.bit_width.to(*args, **kwargs) if self.bit_width is not None else None,
            self.signed,
            self.training)

    def cuda(self, *args, **kwargs):
        return QuantTensor(
            self.qt_value.cuda(*args, **kwargs),
            self.scale.cuda(*args, **kwargs) if self.scale is not None else None,
            self.zero_point.cuda(*args, **kwargs) if self.zero_point is not None else None,
            self.bit_width.cuda(*args, **kwargs) if self.bit_width is not None else None,
            self.signed,
            self.training)

    def cpu(self, *args, **kwargs):
        return QuantTensor(
            self.qt_value.cpu(*args, **kwargs),
            self.scale.cpu(*args, **kwargs) if self.scale is not None else None,
            self.zero_point.cpu(*args, **kwargs) if self.zero_point is not None else None,
            self.bit_width.cpu(*args, **kwargs) if self.bit_width is not None else None,
            self.signed,
            self.training)

    def __add__(self, other):
        if isinstance(other, QuantTensor) and self.is_not_none and other.is_not_none:
            is_output_qt_valid = True
            if self.training is not None and self.training:
                if not self.check_scaling_factors_same(other):
                    is_output_qt_valid = False
            elif not self.check_scaling_factors_same(other):
                raise RuntimeError("Scaling factors are different")

            output_scale = (self.scale + other.scale) / 2
            output_zero_point = self.zero_point + other.zero_point
            max_val = max_int(signed=self.signed, narrow_range=False, bit_width=self.bit_width)
            max_val += max_int(signed=other.signed, narrow_range=False, bit_width=other.bit_width)
            min_val = min_int(signed=self.signed, narrow_range=False, bit_width=self.bit_width)
            min_val += min_int(signed=other.signed, narrow_range=False, bit_width=other.bit_width)
            output_bit_width = ceil_ste(torch.log2(max_val - min_val))
            output_signed = self.signed or other.signed
            output_training = self.training or other.training
            output_value = self.int() + other.int(
            ) if is_output_qt_valid else self.value + other.value

            output = QuantTensor(
                output_value,
                scale=output_scale,
                zero_point=output_zero_point,
                bit_width=output_bit_width,
                signed=output_signed,
                training=output_training)
        else:
            output = self.value + other
        return output

    def __radd__(self, other):
        return self.__add__(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __mul__(self, other):
        if isinstance(other, QuantTensor) and self.is_not_none and other.is_not_none:
            output_value = self.value * other.value
            output_scale = self.scale * other.scale
            output_bit_width = self.bit_width + other.bit_width
            output_signed = self.signed or other.signed
            output_training = self.training or other.training
            if self.is_zero_zero_point(self) and self.is_zero_zero_point(other):
                output_zero_point = self.zero_point * other.zero_point
            else:
                raise RuntimeError("Zero-points of mul operands are non-zero, not supported.")
            output = QuantTensor.from_fake_quantized(
                output_value,
                scale=output_scale,
                zero_point=output_zero_point,
                bit_width=output_bit_width,
                signed=output_signed,
                training=output_training)
        elif isinstance(other, QuantTensor):
            output = self.value * other.value
        else:
            output = self.value * other
        return output

    def __sub__(self, other):
        return self.__add__(-other)

    def __str__(self):
        return f"QuantTensor(value={self.value}, scale={self.scale}, zero_point={self.zero_point}, bit_width={self.bit_width}, signed_t={self.signed_t}, training_t={self.training_t})"

    def __truediv__(self, other):
        if isinstance(other, QuantTensor) and self.is_not_none and other.is_not_none:
            output_tensor = self.value / other.value  # Note, output tensor not guaranteed to pass self.is_valid()
            max_int_denominator = 2 ** (other.bit_width - int(other.signed))
            output_scale = self.scale / (other.scale * max_int_denominator)
            output_bit_width = self.bit_width + other.bit_width
            output_signed = self.signed or other.signed
            output_training = self.training or other.training
            if self.is_zero_zero_point(self) and self.is_zero_zero_point(other):
                output_zero_point = self.zero_point * other.zero_point  # Output zero_point is a new, zero-valued tensor
            else:
                raise RuntimeError("Zero-points of div operands are non-zero, not supported.")
            output = QuantTensor(
                int_value=output_tensor,
                scale=output_scale,
                zero_point=output_zero_point,
                bit_width=output_bit_width,
                signed=output_signed,
                training=output_training)
        elif isinstance(other, QuantTensor):
            output = self.value / other.value
        else:
            output = self.value / other
        return output

    def __abs__(self):
        if self.signed:
            abs_value = (torch.abs(self.int(float_datatype=True)) - self.zero_point) * self.scale
            # In case the dtype of self.int is different from the one of the scale
            abs_value = abs_value.type(self.scale.dtype)
            return QuantTensor(
                int_value=abs_value,
                scale=self.scale,
                zero_point=self.zero_point,
                bit_width=self.bit_width - 1,
                signed=False,
                training=self.training)
        else:
            return self

    def __pos__(self):
        return self
