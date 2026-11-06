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

"""llama.cpp's scale searches as grid optimization solvers.

Each solver runs one of llama.cpp's block searches on whatever layer grid
optimization hands it, and every layer the same way. What a layer contributes is
read from its quantizer:

  - the code range comes from the unscaled grid: a signed grid [-nmax, nmax - 1]
    is fitted symmetrically, an unsigned one [0, nmax] with an offset, which is
    written to the quantizer's zero-point;
  - the block is the quantizer's group;
  - a quantizer that quantizes its scales (and zero-points) in turn on a nested
    grid, a step shared by a run of groups times an integer per group, as the
    GGUF K formats do, has them placed on that grid as llama.cpp places them;
  - a symmetric fit whose quantizer only holds positive scales is seeded with a
    positive scale, instead of llama.cpp's signed one.

A solver only accepts the layers its fit can be written into (see check_layer);
the others are skipped, as grid optimization skips any unsupported layer.

The importance of each input channel is the diagonal of the covariance matrix H
that grid optimization accumulates, H = 2/N X^T X, halved to the mean square
activation llama.cpp calls an imatrix. With covariance_mode='identity' it is
constant, so the fit ignores the data.
"""

from typing import Optional
from typing import Tuple

import torch
import torch.nn as nn

from brevitas.graph.grid_optimization.solvers import SCALE_SOLVER_REGISTRY
from brevitas.graph.grid_optimization.solvers import ScaleSolution
from brevitas.graph.grid_optimization.solvers import ScaleSolverFamily
from brevitas.graph.grid_optimization.utils import _get_scale_quant
from brevitas.graph.grid_optimization.utils import _get_scale_quant_grid
from brevitas.graph.grid_optimization.utils import _get_zero_point_quant
from brevitas.graph.grid_optimization.utils import _get_zero_point_quant_grid
from brevitas.graph.grid_optimization.utils import _has_base_zero_point
from brevitas.graph.grid_optimization.utils import _has_zero_zero_point
from brevitas.graph.grid_optimization.utils import _is_int_weight_quant
from brevitas.graph.grid_optimization.utils import NestedGrid
from brevitas_examples.llm.gguf_export.search.kernels import affine_scale_search
from brevitas_examples.llm.gguf_export.search.kernels import affine_seed
from brevitas_examples.llm.gguf_export.search.kernels import positive_absmax_grid
from brevitas_examples.llm.gguf_export.search.kernels import positive_grid_search
from brevitas_examples.llm.gguf_export.search.kernels import signed_absmax_grid
from brevitas_examples.llm.gguf_export.search.kernels import symmetric_descent_search
from brevitas_examples.llm.gguf_export.search.kernels import symmetric_scale_search
from brevitas_examples.llm.gguf_export.search.kernels import symmetric_seed
from brevitas_examples.llm.gguf_export.search.objective import objective_weights

__all__ = ["GGUF_SOLVERS", "SCALE_PLACEMENTS"]

# How the scales of a layer with a nested scale grid are placed on it: by their
# largest magnitude alone, or by a weighted sweep around it.
SCALE_PLACEMENTS = ('seed', 'sweep')


def _code_range(unscaled_grid: torch.Tensor) -> Tuple[int, bool]:
    """(nmax, signed) of an integer grid [-nmax, nmax - 1] or [0, nmax]."""
    signed = bool(unscaled_grid[0] < 0)
    nmax = int(-unscaled_grid[0]) if signed else int(unscaled_grid[-1])
    return nmax, signed


def _place(values: torch.Tensor, weights: torch.Tensor, grid: NestedGrid,
           placement: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Place per-group values [..., groups_per_step] on a nested grid.

    Returns (step, codes): one step per run of groups, one integer per group.
    """
    if grid.min_int < 0:
        nmax = -grid.min_int
        if placement == 'seed':
            return signed_absmax_grid(values, nmax)
        return symmetric_scale_search(values, weights, nmax)
    if placement == 'seed':
        return positive_absmax_grid(values, grid.max_int)
    return positive_grid_search(values, weights, grid.max_int)


class _GgufSolverFamily(ScaleSolverFamily):
    """A llama.cpp block search, run on whatever per-group layer it is given.

    Only the per-group diagonal slot is implemented: the searches weigh each
    weight by the importance of its input channel, which is the diagonal of H,
    and fit one scale per block. Subclasses say which search they run.
    """

    # 'symmetric' or 'affine' for a fit that only exists in one of them; None for
    # one that follows the layer's grid.
    space: Optional[str] = None

    @classmethod
    def _fit(
            cls,
            x: torch.Tensor,
            w: torch.Tensor,
            nmax: int,
            signed: bool,
            positive_scale: bool,
            sweep_steps: Optional[int],
            sweep_step: Optional[float]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Fit every block of x, weighed by w. Returns (scale, offset or None)."""
        raise NotImplementedError

    @classmethod
    def check_layer(cls, layer: nn.Module) -> Optional[str]:
        weight_quant = layer.weight_quant
        if not _is_int_weight_quant(weight_quant):
            return "the GGUF solvers need an integer weight quantizer"
        if not weight_quant.is_groupwise:
            return "the GGUF solvers need per-group weight quantization"
        if weight_quant.is_narrow_range:
            return "the GGUF solvers need a full-range integer grid"
        signed = weight_quant.is_signed
        if cls.space == 'symmetric' and not signed:
            return "a symmetric fit needs a signed weight quantizer"
        if cls.space == 'affine' and signed:
            return "an affine fit needs an unsigned weight quantizer"
        scale_quant_grid = _get_scale_quant_grid(layer)
        zero_point_quant_grid = _get_zero_point_quant_grid(layer)
        if scale_quant_grid is None and _get_scale_quant(layer) is not None:
            return "the layer quantizes its scales on a grid the GGUF solvers cannot write into"
        if signed:
            if not _has_zero_zero_point(layer):
                return "a symmetric fit needs a zero zero-point"
        elif zero_point_quant_grid is None:
            if _get_zero_point_quant(layer) is not None:
                return (
                    "the layer quantizes its zero-points on a grid the GGUF solvers cannot "
                    "write into")
            if not _has_base_zero_point(layer):
                return (
                    "an affine fit needs a stored zero-point that is not rounded to an integer "
                    "(a ParameterFromStatsFromParameterZeroPoint with quantize_zero_point=False)")
        if scale_quant_grid is None:
            return None
        if not signed and zero_point_quant_grid is None:
            return "an affine fit with quantized scales needs quantized zero-points too"
        for grid in (scale_quant_grid, zero_point_quant_grid):
            if grid is not None and grid.min_int not in (0, -grid.max_int - 1):
                return "the GGUF solvers need nested grids of the form [-n, n - 1] or [0, n]"
        signed_scales = signed and not weight_quant.restrict_scale_positive
        if (scale_quant_grid.min_int < 0) != signed_scales:
            kind = 'signed' if signed_scales else 'non-negative'
            return f"the fit produces {kind} scales, which the layer's scale grid cannot hold"
        if zero_point_quant_grid is not None:
            if zero_point_quant_grid.min_int < 0:
                return "the zero-points are non-negative, so they need a non-negative grid"
            if zero_point_quant_grid.groups_per_step != scale_quant_grid.groups_per_step:
                return "the scales and the zero-points need to share their steps in the same runs"
        return None

    @classmethod
    def _per_group_diag(
            cls,
            w_b: torch.Tensor,
            H_diag: torch.Tensor,
            unscaled_grid: torch.Tensor,
            G_diag: Optional[torch.Tensor] = None,
            eps: float = 1e-6,
            allow_negative_s: bool = False,
            scale_quant_grid: Optional[NestedGrid] = None,
            zero_point_quant_grid: Optional[NestedGrid] = None,
            magnitude: str = 'rms_quad',
            magnitude_factor: float = 1.0,
            magnitude_scope: str = 'row',
            sweep_steps: Optional[int] = None,
            sweep_step: Optional[float] = None,
            scale_placement: str = 'sweep') -> ScaleSolution:
        """Fit the blocks of a batch of rows, w_b [rows, groups, block].

        H_diag is the diagonal of H over the whole row. eps belongs to the PiSO
        interface and is not read. A symmetric fit whose scales cannot be
        negative (allow_negative_s False) seeds them positive. A layer whose
        scales are quantized on a nested grid has them, and its offsets, placed
        on it, and the solution carries the step of every run of groups.
        """
        if G_diag is not None:
            raise ValueError(
                "the GGUF solvers fit one set of activations, so they do not support the "
                "cross-activation objective")
        if scale_placement not in SCALE_PLACEMENTS:
            raise ValueError(
                f"unknown scale placement {scale_placement!r}; expected one of "
                f"{list(SCALE_PLACEMENTS)}")
        n_rows, n_groups, block = w_b.shape
        nmax, signed = _code_range(unscaled_grid)
        # llama.cpp's imatrix is the mean square activation, half of diag(H).
        importance = (H_diag / 2).reshape(n_groups, block)
        x = w_b
        if scale_quant_grid is not None:
            x = w_b.reshape(n_rows, -1, scale_quant_grid.groups_per_step, block)
            importance = importance.reshape(x.shape[1:])
        w = objective_weights(x, importance, magnitude, magnitude_factor, magnitude_scope)
        scale, offset = cls._fit(
            x, w, nmax, signed, signed and not allow_negative_s, sweep_steps, sweep_step)
        if scale_quant_grid is None:
            return ScaleSolution(scale=scale, offset=offset)

        # llama.cpp weighs the placement of a group's scale by the total weight of its fit.
        group_weights = w.sum(-1)
        scale_step, scale_codes = _place(scale, group_weights, scale_quant_grid, scale_placement)
        scale = scale_step.unsqueeze(-1) * scale_codes
        offset_step = None
        if offset is not None:
            # The offsets are non-positive and placed by their magnitude.
            offset_step, offset_codes = _place(
                -offset, group_weights, zero_point_quant_grid, scale_placement)
            offset = -offset_step.unsqueeze(-1) * offset_codes
        return ScaleSolution(
            scale=scale, offset=offset, scale_step=scale_step, offset_step=offset_step)


@SCALE_SOLVER_REGISTRY.register(names='gguf-seed')
class GgufSeedSolverFamily(_GgufSolverFamily):
    """No search: the largest magnitude of a block (symmetric) or its range
    (affine), as llama.cpp's reference paths store them."""

    @classmethod
    def _fit(cls, x, w, nmax, signed, positive_scale, sweep_steps, sweep_step):
        if signed:
            return symmetric_seed(x, nmax, positive_scale), None
        return affine_seed(x, nmax)


@SCALE_SOLVER_REGISTRY.register(names='gguf-symmetric-sweep')
class GgufSymmetricSweepSolverFamily(_GgufSolverFamily):
    """llama.cpp's `make_qx_quants`: refit at the seed, then trial scales on both
    sides of it, each scored by the quality of its own refit."""
    space = 'symmetric'

    @classmethod
    def _fit(cls, x, w, nmax, signed, positive_scale, sweep_steps, sweep_step):
        scale, _ = symmetric_scale_search(
            x,
            w,
            nmax,
            n_trials=9 if sweep_steps is None else sweep_steps,
            trial_step=0.1 if sweep_step is None else sweep_step,
            positive_scale=positive_scale)
        return scale, None


@SCALE_SOLVER_REGISTRY.register(names='gguf-symmetric-descent')
class GgufSymmetricDescentSolverFamily(_GgufSolverFamily):
    """llama.cpp's `make_q3_quants`: refit at the seed, then move one code at a
    time while the fit improves. It weighs each weight by its own square and
    has no trial scales, so the importance and the sweep options are not read,
    except to place the scales of a layer with a nested scale grid."""
    space = 'symmetric'

    @classmethod
    def _fit(cls, x, w, nmax, signed, positive_scale, sweep_steps, sweep_step):
        return symmetric_descent_search(x, nmax, positive_scale=positive_scale), None


@SCALE_SOLVER_REGISTRY.register(names='gguf-affine-sweep')
class GgufAffineSweepSolverFamily(_GgufSolverFamily):
    """llama.cpp's `make_qkx3_quants`: trial scales over the block's range, each
    with its own best offset."""
    space = 'affine'

    @classmethod
    def _fit(cls, x, w, nmax, signed, positive_scale, sweep_steps, sweep_step):
        return affine_scale_search(
            x,
            w,
            nmax,
            n_steps=36 if sweep_steps is None else sweep_steps,
            trial_step=0.05 if sweep_step is None else sweep_step)


GGUF_SOLVERS = ('gguf-seed', 'gguf-symmetric-sweep', 'gguf-symmetric-descent', 'gguf-affine-sweep')
