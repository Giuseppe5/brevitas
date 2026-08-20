# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch
from torch import Tensor


def _validate_grouped_input(x: Tensor, group_size: int, group_dim: int, format_name: str):
    original_dim = x.dim() - 1
    if not -original_dim <= group_dim < original_dim:
        raise RuntimeError(f'{format_name} group_dim is out of range')
    group_size_dim = group_dim + 1 if group_dim >= 0 else group_dim
    grouped_size = x.shape[group_size_dim]
    if isinstance(grouped_size, int) and grouped_size != group_size:
        raise RuntimeError(
            f'{format_name} input is not grouped according to group_size and group_dim')


def _quantize_mx_fp8_ref(
        x: Tensor,
        scale: Tensor,
        group_size: int,
        group_dim: int,
        dtype: torch.dtype,
        min_val: float,
        max_val: float) -> Tensor:
    _validate_grouped_input(x, group_size, group_dim, 'MXFP8')
    return torch.clamp(x.to(torch.float32) / scale, min_val, max_val).to(dtype)


def _dequantize_mx_fp8_ref(
        x: Tensor, scale: Tensor, group_size: int, group_dim: int,
        output_dtype: torch.dtype) -> Tensor:
    _validate_grouped_input(x, group_size, group_dim, 'MXFP8')
    return (x.to(output_dtype) * scale).to(output_dtype)


def _encode_e2m1(x: Tensor) -> Tensor:
    levels = x.new_tensor([0., .5, 1., 1.5, 2., 3., 4., 6.])
    abs_x = x.abs().clamp_max(6.)
    distance = (abs_x.unsqueeze(-1) - levels).abs()
    min_distance = distance.amin(dim=-1, keepdim=True)
    tied = distance == min_distance
    has_tie = tied.sum(dim=-1, keepdim=True) > 1
    odd_code = torch.arange(8, device=x.device) % 2 == 1
    distance = torch.where(tied & has_tie & odd_code, torch.inf, distance)
    magnitude = distance.argmin(dim=-1).to(torch.uint8)
    sign = torch.where(x < 0, 8, 0).to(torch.uint8)
    return magnitude | sign


def _decode_e2m1(x: Tensor, output_dtype: torch.dtype) -> Tensor:
    levels = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.], device=x.device)
    magnitude = levels[x.to(torch.long) & 7]
    sign = torch.where((x & 8) != 0, -1., 1.)
    return (magnitude * sign).to(output_dtype)


def _quantize_mx_fp4_ref(x: Tensor, scale: Tensor, group_size: int, group_dim: int) -> Tensor:
    _validate_grouped_input(x, group_size, group_dim, 'MXFP4')
    if torch.isnan(x).any():
        raise RuntimeError('MXFP4 quantization does not support NaN values')
    return _encode_e2m1(x.to(torch.float32) / scale)


def _dequantize_mx_fp4_ref(
        x: Tensor, scale: Tensor, group_size: int, group_dim: int,
        output_dtype: torch.dtype) -> Tensor:
    _validate_grouped_input(x, group_size, group_dim, 'MXFP4')
    return (_decode_e2m1(x, output_dtype) * scale).to(output_dtype)


if hasattr(torch.library, 'custom_op'):
    quantize_mx_fp8 = torch.library.custom_op(
        'brevitas::quantize_mx_fp8', mutates_args=())(_quantize_mx_fp8_ref)
    dequantize_mx_fp8 = torch.library.custom_op(
        'brevitas::dequantize_mx_fp8', mutates_args=())(_dequantize_mx_fp8_ref)
    quantize_mx_fp4 = torch.library.custom_op(
        'brevitas::quantize_mx_fp4', mutates_args=())(_quantize_mx_fp4_ref)
    dequantize_mx_fp4 = torch.library.custom_op(
        'brevitas::dequantize_mx_fp4', mutates_args=())(_dequantize_mx_fp4_ref)

    @quantize_mx_fp8.register_fake
    def _quantize_mx_fp8_fake(x, scale, group_size, group_dim, dtype, min_val, max_val):
        _validate_grouped_input(x, group_size, group_dim, 'MXFP8')
        return torch.empty_like(x, dtype=dtype)

    @dequantize_mx_fp8.register_fake
    def _dequantize_mx_fp8_fake(x, scale, group_size, group_dim, output_dtype):
        _validate_grouped_input(x, group_size, group_dim, 'MXFP8')
        return torch.empty_like(x, dtype=output_dtype)

    @quantize_mx_fp4.register_fake
    def _quantize_mx_fp4_fake(x, scale, group_size, group_dim):
        _validate_grouped_input(x, group_size, group_dim, 'MXFP4')
        return torch.empty_like(x, dtype=torch.uint8)

    @dequantize_mx_fp4.register_fake
    def _dequantize_mx_fp4_fake(x, scale, group_size, group_dim, output_dtype):
        _validate_grouped_input(x, group_size, group_dim, 'MXFP4')
        return torch.empty_like(x, dtype=output_dtype)
else:
    quantize_mx_fp8 = _quantize_mx_fp8_ref
    dequantize_mx_fp8 = _dequantize_mx_fp8_ref
    quantize_mx_fp4 = _quantize_mx_fp4_ref
    dequantize_mx_fp4 = _dequantize_mx_fp4_ref

__all__ = ['dequantize_mx_fp4', 'dequantize_mx_fp8', 'quantize_mx_fp4', 'quantize_mx_fp8']
