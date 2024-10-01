# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from typing import List, Optional, Tuple, Union

import torch
from torch.nn import Module
from torch.nn import Parameter

import brevitas
import brevitas.config as config
from brevitas.core.function_wrapper import Identity
from brevitas.core.restrict_val import _RestrictClampValue
from brevitas.core.stats import _ParameterListStats
from brevitas.core.stats import _RuntimeStats
from brevitas.core.stats import DEFAULT_MOMENTUM
from brevitas.core.utils import ParameterWrapper
from brevitas.core.utils import StatelessBuffer
from brevitas.function.ops_ste import abs_binary_sign_grad


class StatsFromParameterScaling(brevitas.jit.ScriptModule):

    def __init__(
            self,
            scaling_stats_impl: Module,
            scaling_stats_input_view_shape_impl: Module,
            scaling_stats_input_concat_dim: int,
            tracked_parameter_list: List[torch.nn.Parameter],
            restrict_scaling_impl: Module,
            scaling_shape: Tuple[int, ...],
            affine_rescaling: bool = False,
            affine_shift_scale: bool = False,
            scaling_min_val: Optional[float] = None,
            dtype: Optional[torch.dtype] = None,
            device: Optional[torch.device] = None) -> None:
        super(StatsFromParameterScaling, self).__init__()
        self.parameter_list_stats = _ParameterListStats(
            scaling_stats_impl,
            scaling_shape,
            scaling_stats_input_view_shape_impl,
            scaling_stats_input_concat_dim,
            tracked_parameter_list)
        self.stats_scaling_impl = _StatsScaling(
            restrict_scaling_impl,
            scaling_shape,
            scaling_min_val,
            affine_rescaling,
            affine_shift_scale,
            dtype,
            device)

    @brevitas.jit.script_method
    def forward(
            self, ignored: torch.Tensor, threshold: Optional[torch.Tensor] = None) -> torch.Tensor:
        stats = self.parameter_list_stats()
        if threshold is None:
            threshold = torch.ones(1).type_as(stats)
        return self.stats_scaling_impl(stats, threshold)


class _StatsScaling(brevitas.jit.ScriptModule):

    def __init__(
            self,
            restrict_scaling_impl: Module,
            scaling_shape: Tuple[int, ...],
            scaling_min_val: Optional[float],
            affine_rescaling: bool,
            affine_shift_scale: bool,
            dtype: Optional[torch.dtype],
            device: Optional[torch.device]) -> None:
        super(_StatsScaling, self).__init__()
        if affine_shift_scale and not affine_rescaling:
            raise RuntimeError(
                "Disabling shifting of the scale requires to enable affine rescaling first.")
        if affine_rescaling:
            self.affine_rescaling = _AffineRescaling(
                scaling_shape, affine_shift_scale, dtype, device)
        else:
            self.affine_rescaling = Identity()
        self.restrict_clamp_scaling = _RestrictClampValue(scaling_min_val, restrict_scaling_impl)
        self.restrict_scaling_pre = restrict_scaling_impl.restrict_init_module()

    @brevitas.jit.script_method
    def forward(
            self, stats: torch.Tensor, threshold: Optional[torch.Tensor] = None) -> torch.Tensor:
        if threshold is None:
            threshold = torch.ones(1).type_as(stats)
        stats = self.restrict_scaling_pre(stats / threshold)
        stats = self.affine_rescaling(stats)
        stats = self.restrict_clamp_scaling(stats)
        return stats


class RuntimeStatsScaling(brevitas.jit.ScriptModule):

    def __init__(
            self,
            scaling_stats_impl: Module,
            scaling_stats_input_view_shape_impl: Module,
            restrict_scaling_impl: Module,
            scaling_shape: Tuple[int, ...],
            affine_rescaling: bool = False,
            affine_shift_scale: bool = False,
            scaling_stats_momentum: float = DEFAULT_MOMENTUM,
            scaling_min_val: Optional[float] = None,
            dtype: Optional[torch.dtype] = None,
            device: Optional[torch.device] = None) -> None:
        super(RuntimeStatsScaling, self).__init__()

        self.runtime_stats = _RuntimeStats(
            scaling_stats_impl,
            scaling_shape,
            scaling_stats_input_view_shape_impl,
            scaling_stats_momentum,
            dtype,
            device)
        self.stats_scaling_impl = _StatsScaling(
            restrict_scaling_impl,
            scaling_shape,
            scaling_min_val,
            affine_rescaling,
            affine_shift_scale,
            dtype,
            device)

    @brevitas.jit.script_method
    def forward(self, x: torch.Tensor, threshold: Optional[torch.Tensor] = None) -> torch.Tensor:
        stats = self.runtime_stats(x)
        return self.stats_scaling_impl(stats, threshold)


class _AffineRescaling(brevitas.jit.ScriptModule):

    def __init__(
            self,
            scaling_shape,
            shift_scale,
            dtype: Optional[torch.dtype],
            device: Optional[torch.device]):
        super(_AffineRescaling, self).__init__()
        self.affine_weight = Parameter(torch.ones(scaling_shape, dtype=dtype, device=device))
        if shift_scale:
            self.affine_bias = ParameterWrapper(
                torch.zeros(scaling_shape, dtype=dtype, device=device))
        else:
            self.affine_bias = StatelessBuffer(torch.tensor(0., dtype=dtype, device=device))

    @brevitas.jit.script_method
    def forward(self, x):
        out = x * self.affine_weight + self.affine_bias()
        out = abs_binary_sign_grad(out)
        return out

    def _load_from_state_dict(
            self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
            error_msgs):
        super(_AffineRescaling, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        affine_weight_key = prefix + 'affine_weight'
        affine_bias_key = prefix + 'affine_bias'
        if config.IGNORE_MISSING_KEYS and affine_weight_key in missing_keys:
            missing_keys.remove(affine_weight_key)
        if config.IGNORE_MISSING_KEYS and affine_bias_key in missing_keys:
            missing_keys.remove(affine_bias_key)


class RuntimeDynamicGroupStatsScaling(brevitas.jit.ScriptModule):

    def __init__(
            self,
            group_size: int,
            group_dim: int,
            input_view_impl: torch.nn.Module,
            scaling_stats_impl: torch.nn.Module,
            scaling_min_val: Optional[float],
            restrict_scaling_impl: Optional[torch.nn.Module]) -> None:
        super(RuntimeDynamicGroupStatsScaling, self).__init__()
        self.group_size = group_size
        self.group_dim = group_dim
        self.scaling_stats_impl = scaling_stats_impl
        self.scaling_min_val = scaling_min_val
        self.input_view_impl = input_view_impl
        self.restrict_clamp_scaling = _RestrictClampValue(scaling_min_val, restrict_scaling_impl)

    @brevitas.jit.script_method
    def forward(
            self,
            stats_input: torch.Tensor,
            threshold: Optional[torch.Tensor] = None) -> torch.Tensor:
        if threshold is None:
            threshold = torch.ones(1).type_as(stats_input)
        stats_input_reshaped = self.input_view_impl(stats_input)
        out = self.scaling_stats_impl(stats_input_reshaped) / threshold
        # Scaling min val
        out = self.restrict_clamp_scaling(out)
        return out
