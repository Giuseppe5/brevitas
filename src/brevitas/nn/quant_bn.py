# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause


from abc import ABC
from typing import Optional

import torch

import brevitas.config as config
from brevitas.inject.defaults import Int8WeightPerTensorFloat

from .quant_layer import ActQuantType
from .quant_layer import BiasQuantType
from .quant_layer import WeightQuantType
from .quant_scale_bias import QuantScaleBias
from .utils import mul_add_from_bn


class _BatchNormToQuantScaleBias(QuantScaleBias, ABC):

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        weight_key = prefix + 'weight'
        bias_key = prefix + 'bias'
        running_mean_key = prefix + 'running_mean'
        running_var_key = prefix + 'running_var'
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if running_mean_key in state_dict and running_var_key in state_dict:
            weight_init, bias_init = mul_add_from_bn(
                bn_bias=state_dict[bias_key],
                bn_weight=state_dict[weight_key],
                bn_mean=state_dict[running_mean_key],
                bn_var=state_dict[running_var_key],
                bn_eps=self.eps)
            self.weight.data = weight_init
            self.bias.data = bias_init
            del state_dict[bias_key]
            del state_dict[weight_key]
            del state_dict[running_mean_key]
            del state_dict[running_var_key]
            del state_dict[num_batches_tracked_key]
        super(_BatchNormToQuantScaleBias, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        if config.IGNORE_MISSING_KEYS and bias_key in missing_keys:
            missing_keys.remove(bias_key)
        if config.IGNORE_MISSING_KEYS and weight_key in missing_keys:
            missing_keys.remove(weight_key)
        if num_batches_tracked_key in unexpected_keys:
            unexpected_keys.remove(num_batches_tracked_key)


class BatchNorm1dToQuantScaleBias(_BatchNormToQuantScaleBias):

    def __init__(
            self,
            num_features,
            eps: float = 1e-5,
            weight_quant: Optional[WeightQuantType] = Int8WeightPerTensorFloat,
            bias_quant: Optional[BiasQuantType] = None,
            input_quant: Optional[ActQuantType] = None,
            output_quant: Optional[ActQuantType] = None,
            return_quant_tensor: bool = False,
            **kwargs):
        super(BatchNorm1dToQuantScaleBias, self).__init__(
            num_features,
            bias=True,
            runtime_shape=(1, -1, 1),
            weight_quant=weight_quant,
            bias_quant=bias_quant,
            input_quant=input_quant,
            output_quant=output_quant,
            return_quant_tensor=return_quant_tensor,
            **kwargs)
        self.eps = eps


class BatchNorm2dToQuantScaleBias(_BatchNormToQuantScaleBias):

    def __init__(
            self,
            num_features,
            eps: float = 1e-5,
            weight_quant: Optional[WeightQuantType] = Int8WeightPerTensorFloat,
            bias_quant: Optional[BiasQuantType] = None,
            input_quant: Optional[ActQuantType] = None,
            output_quant: Optional[ActQuantType] = None,
            return_quant_tensor: bool = False,
            **kwargs):
        super(BatchNorm2dToQuantScaleBias, self).__init__(
            num_features,
            bias=True,
            runtime_shape=(1, -1, 1, 1),
            weight_quant=weight_quant,
            bias_quant=bias_quant,
            input_quant=input_quant,
            output_quant=output_quant,
            return_quant_tensor=return_quant_tensor,
            **kwargs)
        self.eps = eps


def _equalize_bn(bn_module: torch.nn.Module, scaling_factors: torch.Tensor):
    class_name = bn_module.__class__.__name__ + 'Equalized'
    bn_module.register_parameter('orig_bias', torch.nn.Parameter(bn_module.bias.clone().detach()))
    bn_module.register_parameter('orig_weight', torch.nn.Parameter(bn_module.weight.clone().detach()))
    bn_module.register_buffer('scaling_factors',  scaling_factors.clone().detach())
    bn_module.register_buffer('inverse_scaling_factors', torch.ones_like(bn_module.orig_bias))


    del bn_module.bias
    def new_bias(self):
        return self.inverse_scaling_factors * \
        (self.running_mean.data * self.orig_weight / torch.sqrt(self.running_var + self.eps) \
        * (self.scaling_factors - 1) + self.orig_bias)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        output_dict = super(self.__class__, self).state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars)
        output_dict[prefix + 'bias'] = self.bias

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):


        equalized_bn_key = prefix + 'orig_weight'
        is_equalized_bn = equalized_bn_key in state_dict
        if not is_equalized_bn:
            self.scaling_factors.fill_(1.)
            self.inverse_scaling_factors.fill_(1.)
            state_dict[prefix + 'orig_bias'] = state_dict[prefix + 'bias']
            state_dict[prefix + 'orig_weight'] = state_dict[prefix + 'weight']
        super(self.__class__, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        if not is_equalized_bn:
            missing_keys.remove(prefix + 'scaling_factors')
            missing_keys.remove(prefix + 'inverse_scaling_factors')
        unexpected_keys.remove(prefix + 'bias')
    var = {'bias': property(new_bias),
           'state_dict': state_dict,
           '_load_from_state_dict': _load_from_state_dict}
    child_class = type(class_name, (bn_module.__class__,), var)
    bn_module.__class__ = child_class
