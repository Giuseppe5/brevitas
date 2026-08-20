# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Optional

import torch
from torch import Tensor


def _reshape_qparam(x: Tensor, qparam: Tensor, axis: Optional[int]) -> Tensor:
    if axis is None or qparam.numel() == 1:
        return qparam
    shape = [1] * x.dim()
    shape[axis] = qparam.numel()
    return qparam.reshape(shape)


def _quantize_fp8_ref(
        x: Tensor,
        scale: Tensor,
        dtype: torch.dtype,
        axis: Optional[int],
        min_val: float,
        max_val: float) -> Tensor:
    scale = _reshape_qparam(x, scale, axis)
    return torch.clamp(x.to(torch.float32) / scale, min_val, max_val).to(dtype)


def _dequantize_fp8_ref(
        x: Tensor, scale: Tensor, axis: Optional[int], output_dtype: torch.dtype) -> Tensor:
    scale = _reshape_qparam(x, scale, axis)
    return (x.to(output_dtype) * scale).to(output_dtype)


if hasattr(torch.library, 'custom_op'):
    quantize_fp8 = torch.library.custom_op(
        'brevitas::quantize_fp8', mutates_args=())(_quantize_fp8_ref)
    dequantize_fp8 = torch.library.custom_op(
        'brevitas::dequantize_fp8', mutates_args=())(_dequantize_fp8_ref)

    @quantize_fp8.register_fake
    def _quantize_fp8_fake(x, scale, dtype, axis, min_val, max_val):
        return torch.empty_like(x, dtype=dtype)

    @dequantize_fp8.register_fake
    def _dequantize_fp8_fake(x, scale, axis, output_dtype):
        return torch.empty_like(x, dtype=output_dtype)
else:
    quantize_fp8 = _quantize_fp8_ref
    dequantize_fp8 = _dequantize_fp8_ref

__all__ = ['dequantize_fp8', 'quantize_fp8']
