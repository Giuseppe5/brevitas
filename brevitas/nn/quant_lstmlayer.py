# From PyTorch:
#
# Copyright (c) 2019-     Xilinx, Inc              (Giuseppe Franco)
# Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
# Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
# Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
# Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
# Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
# Copyright (c) 2011-2013 NYU                      (Clement Farabet)
# Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
# Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
# Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)
#
# From Caffe2:
#
# Copyright (c) 2016-present, Facebook Inc. All rights reserved.
#
# All contributions by Facebook:
# Copyright (c) 2016 Facebook Inc.
#
# All contributions by Google:
# Copyright (c) 2015 Google Inc.
# All rights reserved.
#
# All contributions by Yangqing Jia:
# Copyright (c) 2015 Yangqing Jia
# All rights reserved.
#
# All contributions from Caffe:
# Copyright(c) 2013, 2014, 2015, the respective contributors
# All rights reserved.
#
# All other contributions:
# Copyright(c) 2015, 2016 the respective contributors
# All rights reserved.
#
# Caffe2 uses a copyright model similar to Caffe: each contributor holds
# copyright over their contributions to Caffe2. The project versioning records
# all such contribution and copyright details. If a contributor wants to further
# mark their specific copyright on a particular contribution, they should
# indicate their copyright solely in the commit message of the change when it is
# committed.
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.
#
# 3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories America
# and IDIAP Research Institute nor the names of its contributors may be
# used to endorse or promote products derived from this software without
# specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

from brevitas.core.bit_width import BitWidthImplType
from brevitas.core.quant import QuantType
from brevitas.nn import QuantSigmoid, QuantTanh, QuantHardTanh
import torch.nn as nn
from brevitas.core.restrict_val import RestrictValueType, FloatToIntImplType
from brevitas.core.scaling import ScalingImplType, SCALING_SCALAR_SHAPE
from brevitas.core.stats import StatsInputViewShapeImpl, StatsOp
from brevitas.proxy.parameter_quant import WeightQuantProxy, BiasQuantProxy, WeightReg, _weight_quant_init_impl
from brevitas.proxy.runtime_quant import _activation_quant_init_impl
from brevitas.core.quant import RescalingIntQuant, IdentityQuant
from brevitas.core.restrict_val import RestrictValueType, RestrictValue, FloatToIntImplType, RestrictValueOpImplType

from brevitas.nn.quant_layer import QuantLayer, SCALING_MIN_VAL
import torch

from typing import Union, Optional, Tuple, List, Callable
from torch import Tensor
from brevitas.quant_tensor import QuantTensor
from brevitas.core import ZERO_HW_SENTINEL_NAME, ZERO_HW_SENTINEL_VALUE
import numbers
from collections import namedtuple

OVER_BATCH_OVER_CHANNELS_SHAPE = (1, -1, 1, 1)
LSTMState = namedtuple('LSTMState', ['hx', 'cx'])

__all__ = ['QuantLSTMLayer', 'BidirLSTMLayer']

brevitas_QuantType = {
    'QuantType.INT' : QuantType.INT,
    'QuantType.FP' : QuantType.FP,
    'QuantType.BINARY': QuantType.BINARY,
    'QuantType.TERNARY': QuantType.TERNARY
}

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        # XXX: This is true for our LSTM / NLP use case and helps simplify code
        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    # @jit.script_method
    def compute_layernorm_stats(self, input):
        mu = input.mean(-1, keepdim=True)
        sigma = input.std(-1, keepdim=True, unbiased=False)
        return mu, sigma

    # @jit.script_method
    def forward(self, input):
        mu, sigma = self.compute_layernorm_stats(input)
        return (input - mu) / sigma * self.weight + self.bias


class IdentityBias(nn.Module):
    def __init__(self, normalized_shape):
        super(IdentityBias, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        # XXX: This is true for our LSTM / NLP use case and helps simplify code
        assert len(normalized_shape) == 1

        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    # @jit.script_method
    def forward(self, input):
        return input + self.bias


class IdentityZeroSentinel(nn.Module):
    def __init__(self):
        super(IdentityBias, self).__init__()

    def forward(self, input, zero_hw_sentinel):
        return input

def reverse(lst):
    # type: (List[Tensor]) -> List[Tensor]
    out = torch.jit.annotate(List[Tensor], [])
    start = len(lst)-1
    end = -1
    step = -1
    for i in range(start, end, step):
        out += [lst[i]]
    return out


class QuantLSTMLayer(nn.Module):
    def __init__(self, input_size, hidden_size, weight_config, activation_config,
                 reverse_input=False, layer_norm='identity', compute_output_scale=False,
                 compute_output_bit_width=False, return_quant_tensor=False,
                 recurrent_quant=None, output_quant=None):

        super(QuantLSTMLayer, self).__init__()
        self.register_buffer(ZERO_HW_SENTINEL_NAME, torch.tensor(ZERO_HW_SENTINEL_VALUE))
        self.return_quant_tensor = return_quant_tensor
        self.weight_config = weight_config
        self.activation_config = activation_config

        self.weight_ih = nn.Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.randn(4 * hidden_size, hidden_size))

        self.reverse_input = reverse_input

        self.layer_norm = layer_norm
        if self.layer_norm == 'identity':
            self.layernorm_i = torch.jit.script(IdentityBias(4 * hidden_size))
            self.layernorm_h = torch.jit.script(IdentityBias(4 * hidden_size))
            self.layernorm_c = torch.jit.script(nn.Identity())
        elif self.layer_norm == 'decompose':
            self.layernorm_i = torch.jit.script(LayerNorm(4 * hidden_size))
            self.layernorm_h = torch.jit.script(LayerNorm(4 * hidden_size))
            self.layernorm_c = torch.jit.script(LayerNorm(hidden_size))
        else:
            self.layernorm_i = nn.LayerNorm(4 * hidden_size)
            self.layernorm_h = nn.LayerNorm(4 * hidden_size)
            self.layernorm_c = nn.LayerNorm(hidden_size)

        self.weight_config['weight_quant_type'] = brevitas_QuantType[weight_config.get('weight_quant_type', 'QuantType.FP')]
        self.weight_config['bias_quant_type'] = brevitas_QuantType[weight_config.get('bias_quant_type', 'QuantType.FP')]
        self.activation_config['quant_type'] = brevitas_QuantType[activation_config.get('quant_type', 'QuantType.FP')]

        self.weight_config['weight_scaling_shape'] = SCALING_SCALAR_SHAPE
        self.weight_config['weight_stats_input_view_shape_impl'] = StatsInputViewShapeImpl.OVER_TENSOR
        self.weight_config['weight_scaling_stats_input_concat_dim'] = 0
        self.weight_config['weight_scaling_stats_reduce_dim'] = None

        tracked_param_list = [self.weight_ih, self.weight_hh]
        self.weight_proxy, self.bias_proxy = self.configure_weight(tracked_param_list, self.weight_config)

        self.quant_sigmoid = self.configure_activation(self.activation_config, QuantSigmoid)
        self.quant_tanh = self.configure_activation(self.activation_config, QuantTanh)

        if output_quant is not None:
            self.out_quant = self.configure_activation(output_quant, QuantHardTanh)
            if recurrent_quant is None:
                self.rec_quant = self.configure_activation(output_quant, QuantHardTanh)
            else:
                self.out_quant = self.configure_activation(recurrent_quant, QuantHardTanh)
        else:
            self.rec_quant = IdentityQuant()
            self.out_quant = IdentityQuant()

        if self.weight_config.get('weight_quant_type', QuantType.FP) == QuantType.FP and compute_output_bit_width:
            raise Exception("Computing output bit width requires enabling quantization")
        if self.weight_config.get('bias_quant_type', QuantType.FP) != QuantType.FP and not (
                compute_output_scale and compute_output_bit_width):
            raise Exception("Quantizing bias requires to compute output scale and output bit width")

    def forward(self, inputs, state):
        # type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]

        inputs, input_scale, input_bit_width = self.unpack_input(inputs)

        zero_hw_sentinel = getattr(self, 'zero_hw_sentinel')
        out, scale, bit_width = self.weight_proxy(self.weight_ih, zero_hw_sentinel)
        quant_weight_ih, quant_weight_ih_scale, quant_weight_ih_bit_width = out, scale, bit_width

        out, scale, bit_width = self.weight_proxy(self.weight_hh, zero_hw_sentinel)
        quant_weight_hh, quant_weight_hh_scale, quant_weight_hh_bit_width = out, scale, bit_width

        zero_hw_sentinel = getattr(self, 'zero_hw_sentinel')
        inputs = inputs.unbind(0)

        start = 0
        end = len(inputs)
        step = 1
        if self.reverse_input:
            start = end-1
            end = -1
            step = -1
            # inputs = reverse(inputs)

        outputs = torch.jit.annotate(List[Tensor], [])
        for i in range(start, end, step):
            hx, cx = state
            igates = self.layernorm_i(torch.mm(inputs[i], quant_weight_ih.t()))
            hgates = self.layernorm_h(torch.mm(hx, quant_weight_hh.t()))
            gates = igates + hgates
            ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

            ingate, _, _ = self.quant_sigmoid(ingate, zero_hw_sentinel)
            forgetgate, _, _ = self.quant_sigmoid(forgetgate, zero_hw_sentinel)
            cellgate, _, _ = self.quant_tanh(cellgate, zero_hw_sentinel)
            outgate, _, _ = self.quant_sigmoid(outgate, zero_hw_sentinel)

            cy = self.layernorm_c((forgetgate * cx) + (ingate * cellgate))

            hy = outgate * self.quant_tanh(cy, zero_hw_sentinel)[0]
            hy1, _, _ = self.out_quant(hy, zero_hw_sentinel)
            hy2, _, _ = self.rec_quant(hy, zero_hw_sentinel)
            outputs += [hy1]
            state = (hy2, cy)

        if self.reverse_input:
            return torch.stack(reverse(outputs)), state
        else:
            return torch.stack(outputs), state

    def max_output_bit_width(self, input_bit_width, weight_bit_width):
        pass
        #
        # max_uint_input = max_uint(bit_width=input_bit_width, narrow_range=False)
        # max_kernel_val = self.weight_quant.tensor_quant.int_quant.max_uint(weight_bit_width)
        # group_size = self.out_channels // self.groups
        # max_uint_output = max_uint_input * max_kernel_val * self.kernel_size[0] * group_size
        # max_output_bit_width = ceil_ste(torch.log2(max_uint_output))
        # return max_output_bit_width

    def unpack_input(self, input):
        if isinstance(input, QuantTensor):
            return input
        else:
            return input, None, None

    def pack_output(self,
                    output,
                    output_scale,
                    output_bit_width):
        if self.return_quant_tensor:
            return QuantTensor(tensor=output, scale=output_scale, bit_width=output_bit_width)
        else:
            return output

    def configure_weight(self, weight, weight_config):
        zero_hw_sentinel = getattr(self, 'zero_hw_sentinel')
        wqp: IdentityQuant = _weight_quant_init_impl(bit_width=weight_config.get('weight_bit_width', 8),
                                                     quant_type=weight_config.get('weight_quant_type'),
                                                     narrow_range=weight_config.get('weight_narrow_range', True),
                                                     scaling_override=weight_config.get('weight_scaling_override',
                                                                                        None),
                                                     restrict_scaling_type=weight_config.get(
                                                         'weight_restrict_scaling_type', RestrictValueType.LOG_FP),
                                                     scaling_const=weight_config.get('weight_scaling_const', None),
                                                     scaling_stats_op=weight_config.get('weight_scaling_stats_op',
                                                                                        StatsOp.MAX),
                                                     scaling_impl_type=weight_config.get('weight_scaling_impl_type',
                                                                                         ScalingImplType.STATS),
                                                     scaling_stats_reduce_dim=weight_config.get(
                                                         'weight_scaling_stats_reduce_dim', None),
                                                     scaling_shape=weight_config.get('weight_scaling_shape',
                                                                                     SCALING_SCALAR_SHAPE),
                                                     bit_width_impl_type=weight_config.get('weight_bit_width_impl_type',
                                                                                           BitWidthImplType.CONST),
                                                     bit_width_impl_override=weight_config.get(
                                                         'weight_bit_width_impl_override', None),
                                                     restrict_bit_width_type=weight_config.get(
                                                         'weight_restrict_bit_width_type', RestrictValueType.INT),
                                                     min_overall_bit_width=weight_config.get(
                                                         'weight_min_overall_bit_width', 2),
                                                     max_overall_bit_width=weight_config.get(
                                                         'weight_max_overall_bit_width', None),
                                                     ternary_threshold=weight_config.get('weight_ternary_threshold',
                                                                                         0.5),
                                                     scaling_stats_input_view_shape_impl=weight_config.get(
                                                         'weight_stats_input_view_shape_impl',
                                                         StatsInputViewShapeImpl.OVER_TENSOR),
                                                     scaling_stats_input_concat_dim=weight_config.get(
                                                         'weight_scaling_stats_input_concat_dim', 0),
                                                     scaling_stats_sigma=weight_config.get('weight_scaling_stats_sigma',
                                                                                           3.0),
                                                     scaling_min_val=weight_config.get('weight_scaling_min_val',
                                                                                       SCALING_MIN_VAL),
                                                     override_pretrained_bit_width=weight_config.get(
                                                         'weight_override_pretrained_bit_width', False),
                                                     tracked_parameter_list=weight,
                                                     zero_hw_sentinel=zero_hw_sentinel)
        bqp = BiasQuantProxy(quant_type=weight_config.get('bias_quant_type'),
                             bit_width=weight_config.get('bias_bit_width', 8),
                             narrow_range=weight_config.get('bias_narrow_range', True))
        return wqp, bqp

    def configure_activation(self, activation_config, activation_func=QuantSigmoid):
        signed = True
        min_val = -1
        max_val = 1
        if activation_func == QuantTanh:
            activation_impl = nn.Tanh()
            min_val = -1
            signed = True
        elif activation_func == QuantSigmoid:
            activation_impl = nn.Sigmoid()
            min_val = 0
            signed = False

        activation_object = _activation_quant_init_impl(activation_impl=activation_impl,
                                                        bit_width=activation_config.get('bit_width', 8),
                                                        narrow_range=activation_config.get('narrow_range', True),
                                                        quant_type=activation_config.get('quant_type'),
                                                        float_to_int_impl_type=activation_config.get(
                                                            'float_to_int_impl_type', FloatToIntImplType.ROUND),
                                                        min_overall_bit_width=activation_config.get(
                                                            'min_overall_bit_width', 2),
                                                        max_overall_bit_width=activation_config.get(
                                                            'max_overall_bit_width', None),
                                                        bit_width_impl_override=activation_config.get(
                                                            'bit_width_impl_override', None),
                                                        bit_width_impl_type=activation_config.get('bit_width_impl_type',
                                                                                                  BitWidthImplType.CONST),
                                                        restrict_bit_width_type=activation_config.get(
                                                            'restrict_bit_width_type', RestrictValueType.INT),
                                                        restrict_scaling_type=activation_config.get(
                                                            'restrict_scaling_type', RestrictValueType.LOG_FP),
                                                        scaling_min_val=activation_config.get('scaling_min_val',
                                                                                              SCALING_MIN_VAL),
                                                        override_pretrained_bit_width=activation_config.get(
                                                            'override_pretrained_bit_width', False),
                                                        min_val=activation_config.get('min_val', min_val),
                                                        max_val=activation_config.get('max_val', max_val),
                                                        signed=activation_config.get('signed', signed),
                                                        per_channel_broadcastable_shape=None,
                                                        scaling_per_channel=False,
                                                        scaling_override=activation_config.get('scaling_override',
                                                                                               None),
                                                        scaling_impl_type=ScalingImplType.CONST,
                                                        scaling_stats_sigma=None,
                                                        scaling_stats_input_view_shape_impl=None,
                                                        scaling_stats_op=None,
                                                        scaling_stats_buffer_momentum=None,
                                                        scaling_stats_permute_dims=None)

        # if activation_config.get('bit_width_impl_type', BitWidthImplType.CONST) == BitWidthImplType.PARAMETER:
        return activation_object
        # else:
        #     return torch.jit.script(activation_object)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):

        super(QuantLSTMCELL, self)._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                                         missing_keys, unexpected_keys, error_msgs)
        zero_hw_sentinel_key = prefix + ZERO_HW_SENTINEL_NAME
        if zero_hw_sentinel_key in missing_keys:
            missing_keys.remove(zero_hw_sentinel_key)
        if zero_hw_sentinel_key in unexpected_keys:  # for retrocompatibility with when it wasn't removed
            unexpected_keys.remove(zero_hw_sentinel_key)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        output_dict = super(QuantLSTMCELL, self).state_dict(destination, prefix, keep_vars)
        del output_dict[prefix + ZERO_HW_SENTINEL_NAME]
        return output_dict


class BidirLSTMLayer(nn.Module):
    __constants__ = ['directions']

    def __init__(self, input_size, hidden_size, weight_config, activation_config, layer_norm = 'identity',
                 compute_output_scale=False, compute_output_bit_width=False,
                 return_quant_tensor=False):
        super(BidirLSTMLayer, self).__init__()
        self.directions = nn.ModuleList([
            torch.jit.script(QuantLSTMLayer(input_size, hidden_size, weight_config, activation_config,
                                            False, layer_norm, compute_output_scale, compute_output_bit_width,
                                            return_quant_tensor)),
            torch.jit.script(QuantLSTMLayer(input_size, hidden_size, weight_config, activation_config,
                                            True, layer_norm, compute_output_scale, compute_output_bit_width,
                                            return_quant_tensor)),
        ])

    # @jit.script_method
    def forward(self, input, states):
        # type: (Tensor, List[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, List[Tuple[Tensor, Tensor]]]
        # List[LSTMState]: [forward LSTMState, backward LSTMState]
        outputs = torch.jit.annotate(List[Tensor], [])
        output_states = torch.jit.annotate(List[Tuple[Tensor, Tensor]], [])
        # XXX: enumerate https://github.com/pytorch/pytorch/issues/14471
        i = 0
        for direction in self.directions:
            state = states[i]
            out, out_state = direction(input, state)
            outputs += [out]
            output_states += [out_state]
            i += 1

        return torch.cat(outputs, -1), output_states
