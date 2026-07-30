# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Register backend-oriented QDQ operators before loading an exported program.
from .custom_ops import dequantize_fp8
from .custom_ops import dequantize_mx_fp4
from .custom_ops import quantize_fp8
from .custom_ops import quantize_mx_fp4

__all__ = ['dequantize_fp8', 'dequantize_mx_fp4', 'quantize_fp8', 'quantize_mx_fp4']
