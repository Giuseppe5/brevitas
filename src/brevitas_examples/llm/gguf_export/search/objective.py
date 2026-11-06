# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# This code is adapted from https://github.com/ggml-org/llama.cpp, under the following LICENSE:
# Copyright (c) 2023-2026 The ggml authors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""What a fit is weighed by: the importance of each input channel times a
magnitude term computed from the weights themselves.

llama.cpp's usual magnitude, `rms_quad`, behaves like |x| for large weights,
making the fit relative rather than absolute, and saturates for near-zero
weights so they keep some say.
"""

import torch

__all__ = ["MAGNITUDE_FORMS", "SCOPES", "objective_weights"]

# Each takes the weights and the mean square over the configured scope.
_MAGNITUDES = {
    'one': lambda x,
           mean_sq,
           factor: torch.ones_like(x),
    'square': lambda x,
              mean_sq,
              factor: x * x,
    'abs': lambda x,
           mean_sq,
           factor: x.abs(),
    'sqrt_abs': lambda x,
                mean_sq,
                factor: torch.sqrt(x.abs()),
    'rms_quad': lambda x,
                mean_sq,
                factor: torch.sqrt(factor * mean_sq + x * x),
    'rms_plus_abs': lambda x,
                    mean_sq,
                    factor: torch.sqrt(mean_sq) + x.abs(),}
MAGNITUDE_FORMS = tuple(_MAGNITUDES)

# What the mean square of a magnitude term is taken over.
SCOPES = ('block', 'super_block', 'row')


def objective_weights(
        x: torch.Tensor,
        importance: torch.Tensor,
        magnitude: str = 'rms_quad',
        factor: float = 1.0,
        scope: str = 'row') -> torch.Tensor:
    """The per-element weight of the fit.

    x is [rows, blocks, block], or [rows, super_blocks, blocks, block] for a
    layer whose blocks are grouped into super-blocks sharing a scale step.
    importance has one value per input channel, shaped as x without its rows.
    """
    if scope == 'super_block' and x.dim() < 4:
        raise ValueError(
            "a 'super_block' scope needs a quantizer whose scales are quantized on a grid "
            "of their own")
    dims = {'block': (-1,), 'super_block': (-2, -1), 'row': tuple(range(1, x.dim()))}[scope]
    mean_sq = (x * x).mean(dim=dims, keepdim=True)
    return importance * _MAGNITUDES[magnitude](x, mean_sq, factor)
