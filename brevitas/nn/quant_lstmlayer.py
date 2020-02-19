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
from collections import namedtuple, OrderedDict
import itertools

OVER_BATCH_OVER_CHANNELS_SHAPE = (1, -1, 1, 1)
LSTMState = namedtuple('LSTMState', ['hx', 'cx'])

__all__ = ['QuantLSTMLayer', 'BidirLSTMLayer']

brevitas_QuantType = {
    'QuantType.INT': QuantType.INT,
    'QuantType.FP': QuantType.FP,
    'QuantType.BINARY': QuantType.BINARY,
    'QuantType.TERNARY': QuantType.TERNARY
}


class _IncompatibleKeys(namedtuple('IncompatibleKeys', ['missing_keys', 'unexpected_keys'])):
    def __repr__(self):
        if not self.missing_keys and not self.unexpected_keys:
            return '<All keys matched successfully>'
        return super(_IncompatibleKeys, self).__repr__()

    __str__ = __repr__

def reverse(lst):
    # type: (List[Tensor]) -> List[Tensor]
    out = torch.jit.annotate(List[Tensor], [])
    start = len(lst) - 1
    end = -1
    step = -1
    for i in range(start, end, step):
        out += [lst[i]]
    return out

class TensorBatchNorm(nn.Module):
    def __init__(self, eps=1e-5, momentum=0.1):
        super(TensorBatchNorm, self).__init__()
        self.register_buffer('running_mean', torch.zeros(1))
        self.register_buffer('running_var', torch.ones(1))
        self.eps = eps
        self.momentum = momentum
        self.weight = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.running_mean.zero_()
        self.running_var.fill_(1)

    def forward(self, input, first):
        if self.training and not first:
            mean = input.mean()
            unbias_var = input.var(unbiased=True)
            self.running_mean = (1-self.momentum) * self.running_mean + mean.detach() * self.momentum
            self.running_var = (1-self.momentum) * self.running_var + unbias_var.detach() * self.momentum
            biased_var = input.var(unbiased=False)
            inv_std = 1/(biased_var + self.eps).pow(0.5)
            output = (input-mean) * inv_std * self.weight
            return output
        else:
            return (input-self.running_mean) * (1/(self.running_var+self.eps).pow(0.5)) * self.weight

class QuantLSTMLayer(nn.Module):
    def __init__(self, input_size, hidden_size, weight_config, activation_config, batch=None,
                 reverse_input=False, layer_norm='identity', compute_output_scale=False,
                 compute_output_bit_width=False, return_quant_tensor=False,
                 recurrent_quant=None, output_quant=None):

        super(QuantLSTMLayer, self).__init__()
        self.register_buffer(ZERO_HW_SENTINEL_NAME, torch.tensor(ZERO_HW_SENTINEL_VALUE))
        self.return_quant_tensor = return_quant_tensor
        self.weight_config = weight_config
        self.activation_config = activation_config

        self.weight_ii = nn.Parameter(torch.randn(hidden_size, input_size), requires_grad=True)
        self.weight_fi = nn.Parameter(torch.randn(hidden_size, input_size), requires_grad=True)
        self.weight_ai = nn.Parameter(torch.randn(hidden_size, input_size), requires_grad=True)
        self.weight_oi = nn.Parameter(torch.randn(hidden_size, input_size), requires_grad=True)

        self.weight_ih = nn.Parameter(torch.randn(hidden_size, hidden_size), requires_grad=True)
        self.weight_fh = nn.Parameter(torch.randn(hidden_size, hidden_size), requires_grad=True)
        self.weight_ah = nn.Parameter(torch.randn(hidden_size, hidden_size), requires_grad=True)
        self.weight_oh = nn.Parameter(torch.randn(hidden_size, hidden_size), requires_grad=True)

        self.bias_i = nn.Parameter(torch.rand(hidden_size), requires_grad=True)
        self.bias_f = nn.Parameter(torch.rand(hidden_size), requires_grad=True)
        self.bias_a = nn.Parameter(torch.rand(hidden_size), requires_grad=True)
        self.bias_o = nn.Parameter(torch.randn(hidden_size), requires_grad=True)
        self.reverse_input = reverse_input

        self.hidden_init = nn.Parameter(torch.zeros(hidden_size), requires_grad=True), \
                           nn.Parameter(torch.zeros(hidden_size), requires_grad=True)

        if layer_norm == 'decompose':
            self.layernorm_i, self.layernorm_f, self.layernorm_a, self.layernorm_o = \
            torch.jit.script(TensorBatchNorm()), torch.jit.script(TensorBatchNorm()), \
            torch.jit.script(TensorBatchNorm()), torch.jit.script(TensorBatchNorm()),
        else:
            self.layernorm_i, self.layernorm_f, self.layernorm_a, self.layernorm_o = \
                torch.jit.script(nn.Identity()), torch.jit.script(nn.Identity()), \
                torch.jit.script(nn.Identity()), torch.jit.script(nn.Identity())

        self.weight_config['weight_scaling_shape'] = SCALING_SCALAR_SHAPE
        self.weight_config['weight_stats_input_view_shape_impl'] = StatsInputViewShapeImpl.OVER_TENSOR
        self.weight_config['weight_scaling_stats_input_concat_dim'] = 0
        self.weight_config['weight_scaling_stats_reduce_dim'] = None

        self.weight_proxy_i = self.configure_weight([self.weight_ii, self.weight_ih], self.weight_config)
        self.weight_proxy_f = self.configure_weight([self.weight_fi, self.weight_fh], self.weight_config)
        self.weight_proxy_a = self.configure_weight([self.weight_ai, self.weight_ah], self.weight_config)
        self.weight_proxy_o = self.configure_weight([self.weight_oi, self.weight_oh], self.weight_config)


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

        if self.weight_config.get('weight_quant_type', 'QuantType.FP') == 'QuantType.FP' and compute_output_bit_width:
            raise Exception("Computing output bit width requires enabling quantization")
        if self.weight_config.get('bias_quant_type', 'QuantType.FP') != 'QuantType.FP' and not (
                compute_output_scale and compute_output_bit_width):
            raise Exception("Quantizing bias requires to compute output scale and output bit width")


    def forward_iteration(self, input, state, first,
                          quant_weight_ii, quant_weight_fi, quant_weight_ai, quant_weight_oi,
                          quant_weight_ih, quant_weight_fh, quant_weight_ah, quant_weight_oh):
        hx, cx = state
        zero_hw_sentinel = getattr(self, 'zero_hw_sentinel')

        igates_ii = torch.mm(input, quant_weight_ii.t())
        hgates_ih = torch.mm(hx, quant_weight_ih.t())

        igates_fi = torch.mm(input, quant_weight_fi.t())
        hgates_fh = torch.mm(hx, quant_weight_fh.t())

        igates_ai = torch.mm(input, quant_weight_ai.t())
        hgates_ah = torch.mm(hx, quant_weight_ah.t())

        igates_oi = torch.mm(input, quant_weight_oi.t())
        hgates_oh = torch.mm(hx, quant_weight_oh.t())

        ingate = self.layernorm_i(igates_ii + hgates_ih, first) + self.bias_i
        forgetgate = self.layernorm_f(igates_fi + hgates_fh, first) + self.bias_f
        cellgate = self.layernorm_a(igates_ai + hgates_ah, first) + self.bias_a
        outgate = self.layernorm_o(igates_oi + hgates_oh, first) + self.bias_o

        ingate, _, _ = self.quant_sigmoid(ingate, zero_hw_sentinel)
        forgetgate, _, _ = self.quant_sigmoid(forgetgate, zero_hw_sentinel)
        cellgate, _, _ = self.quant_tanh(cellgate, zero_hw_sentinel)
        outgate, _, _ = self.quant_sigmoid(outgate, zero_hw_sentinel)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * self.quant_tanh(cy, zero_hw_sentinel)[0]
        hy1, _, _ = self.out_quant(hy, zero_hw_sentinel)
        hy2, _, _ = self.rec_quant(hy, zero_hw_sentinel)

        return hy1, (hy2, cy)




    def forward(self, inputs, state):
        # type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]

        # TODO unpack_input is broken. Worst case scenario, do it in here
        # inputs, input_scale, input_bit_width = self.unpack_input(inputs)

        zero_hw_sentinel = getattr(self, 'zero_hw_sentinel')

        quant_weight_ii, quant_weight_ii_scale, quant_weight_ii_bit_width = self.weight_proxy_i(self.weight_ii,
                                                                                                zero_hw_sentinel)
        quant_weight_fi, quant_weight_fi_scale, quant_weight_fi_bit_width = self.weight_proxy_f(self.weight_fi,
                                                                                                zero_hw_sentinel)
        quant_weight_ai, quant_weight_ai_scale, quant_weight_ai_bit_width = self.weight_proxy_a(self.weight_ai,
                                                                                                zero_hw_sentinel)
        quant_weight_oi, quant_weight_oi_scale, quant_weight_oi_bit_width = self.weight_proxy_o(self.weight_oi,
                                                                                                zero_hw_sentinel)
        quant_weight_ih, quant_weight_ih_scale, quant_weight_ih_bit_width = self.weight_proxy_i(self.weight_ih,
                                                                                                zero_hw_sentinel)
        quant_weight_fh, quant_weight_fh_scale, quant_weight_fh_bit_width = self.weight_proxy_f(self.weight_fh,
                                                                                                zero_hw_sentinel)
        quant_weight_ah, quant_weight_ah_scale, quant_weight_ah_bit_width = self.weight_proxy_a(self.weight_ah,
                                                                                                zero_hw_sentinel)
        quant_weight_oh, quant_weight_oh_scale, quant_weight_oh_bit_width = self.weight_proxy_o(self.weight_oh,
                                                                                                zero_hw_sentinel)

        inputs = inputs.unbind(0)

        start = 1
        end = len(inputs)
        step = 1
        if self.reverse_input:
            start = end - 2
            end = -1
            step = -1
            # inputs = reverse(inputs)
        outputs = torch.jit.annotate(List[Tensor], [])
        output, state = self.forward_iteration(inputs[0], state, True,
              quant_weight_ii, quant_weight_fi, quant_weight_ai, quant_weight_oi,
              quant_weight_ih, quant_weight_fh, quant_weight_ah, quant_weight_oh)
        outputs += [output]
        for i in range(start, end, step):
            output, state = self.forward_iteration(inputs[i], state, False,
                          quant_weight_ii, quant_weight_fi, quant_weight_ai, quant_weight_oi,
                          quant_weight_ih, quant_weight_fh, quant_weight_ah, quant_weight_oh)
            outputs += [output]

        if self.reverse_input:
            return torch.stack(reverse(outputs)), state
        else:
            return torch.stack(outputs), state




    def max_output_bit_width(self, input_bit_width, weight_bit_width):
        pass


    def configure_weight(self, weight, weight_config):
        zero_hw_sentinel = getattr(self, 'zero_hw_sentinel')
        wqp = _weight_quant_init_impl(bit_width=weight_config.get('weight_bit_width', 8),
                                      quant_type=brevitas_QuantType[
                                          weight_config.get('weight_quant_type', 'QuantType.FP')],
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
        # bqp = BiasQuantProxy(quant_type=brevitas_QuantType[weight_config.get('bias_quant_type', 'QuantType.FP')],
        #                      bit_width=weight_config.get('bias_bit_width', 8),
        #                      narrow_range=weight_config.get('bias_narrow_range', True))
        return wqp  # , bqp

    def configure_activation(self, activation_config, activation_func=QuantSigmoid):
        signed = True
        min_val = -1
        max_val = 1
        if activation_func == QuantTanh:
            activation_impl = nn.Tanh()
        elif activation_func == QuantSigmoid:
            activation_impl = nn.Sigmoid()
            min_val = 0
            signed = False

        activation_object = _activation_quant_init_impl(activation_impl=activation_impl,
                                                        bit_width=activation_config.get('bit_width', 8),
                                                        narrow_range=activation_config.get('narrow_range', True),
                                                        quant_type=brevitas_QuantType[
                                                            activation_config.get('quant_type', 'QuantType.FP')],
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
        r"""Copies parameters and buffers from :attr:`state_dict` into only
        this module, but not its descendants. This is called on every submodule
        in :meth:`~torch.nn.Module.load_state_dict`. Metadata saved for this
        module in input :attr:`state_dict` is provided as :attr:`local_metadata`.
        For state dicts without metadata, :attr:`local_metadata` is empty.
        Subclasses can achieve class-specific backward compatible loading using
        the version number at `local_metadata.get("version", None)`.

        .. note::
            :attr:`state_dict` is not the same object as the input
            :attr:`state_dict` to :meth:`~torch.nn.Module.load_state_dict`. So
            it can be modified.

        Arguments:
            state_dict (dict): a dict containing parameters and
                persistent buffers.
            prefix (str): the prefix for parameters and buffers used in this
                module
            local_metadata (dict): a dict containing the metadata for this module.
                See
            strict (bool): whether to strictly enforce that the keys in
                :attr:`state_dict` with :attr:`prefix` match the names of
                parameters and buffers in this module
            missing_keys (list of str): if ``strict=True``, add missing keys to
                this list
            unexpected_keys (list of str): if ``strict=True``, add unexpected
                keys to this list
            error_msgs (list of str): error messages should be added to this
                list, and will be reported together in
                :meth:`~torch.nn.Module.load_state_dict`
        """
        for hook in self._load_state_dict_pre_hooks.values():
            hook(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

        local_name_params = itertools.chain(self._parameters.items(), self._buffers.items())
        local_state = {k: v.data for k, v in local_name_params if v is not None}

        for name, param in local_state.items():
            key = prefix + name
            if key in state_dict:
                input_param = state_dict[key]

                # Backward compatibility: loading 1-dim tensor from 0.3.* to version 0.4+
                if len(param.shape) == 0 and len(input_param.shape) == 1:
                    input_param = input_param[0]

                if input_param.shape != param.shape:
                    # local shape should match the one in checkpoint
                    error_msgs.append('size mismatch for {}: copying a param with shape {} from checkpoint, '
                                      'the shape in current model is {}.'
                                      .format(key, input_param.shape, param.shape))
                    continue

                if isinstance(input_param, nn.Parameter):
                    # backwards compatibility for serialized parameters
                    input_param = input_param.data
                try:
                    param.copy_(input_param)
                except Exception:
                    error_msgs.append('While copying the parameter named "{}", '
                                      'whose dimensions in the model are {} and '
                                      'whose dimensions in the checkpoint are {}.'
                                      .format(key, param.size(), input_param.size()))
            elif strict:
                missing_keys.append(key)

        if strict:
            for key in state_dict.keys():
                if key.startswith(prefix):
                    input_name = key[len(prefix):]
                    input_name = input_name.split('.', 1)[0]  # get the name of param/buffer/child
                    if input_name not in self._modules and input_name not in local_state:
                        unexpected_keys.append(key)

        zero_hw_sentinel_key = prefix + "zero_hw_sentinel"

        if zero_hw_sentinel_key in missing_keys:
            missing_keys.remove(zero_hw_sentinel_key)
        if zero_hw_sentinel_key in unexpected_keys:  # for retrocompatibility with when it wasn't removed
            unexpected_keys.remove(zero_hw_sentinel_key)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        output_dict = super(QuantLSTMLayer, self).state_dict(destination, prefix, keep_vars)
        del output_dict[prefix + ZERO_HW_SENTINEL_NAME]
        return output_dict

    def load_state_dict_new(self, state_dict, strict=True):
        r"""Copies parameters and buffers from :attr:`state_dict` into
        this module and its descendants. If :attr:`strict` is ``True``, then
        the keys of :attr:`state_dict` must exactly match the keys returned
        by this module's :meth:`~torch.nn.Module.state_dict` function.

        Arguments:
            state_dict (dict): a dict containing parameters and
                persistent buffers.
            strict (bool, optional): whether to strictly enforce that the keys
                in :attr:`state_dict` match the keys returned by this module's
                :meth:`~torch.nn.Module.state_dict` function. Default: ``True``

        Returns:
            ``NamedTuple`` with ``missing_keys`` and ``unexpected_keys`` fields:
                * **missing_keys** is a list of str containing the missing keys
                * **unexpected_keys** is a list of str containing the unexpected keys
        """
        state_dict = self.fix_state_dict(state_dict)
        missing_keys = []
        unexpected_keys = []
        error_msgs = []

        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')

        load(self)
        load = None  # break load->load reference cycle

        if strict:
            if len(unexpected_keys) > 0:
                error_msgs.insert(
                    0, 'Unexpected key(s) in state_dict: {}. '.format(
                        ', '.join('"{}"'.format(k) for k in unexpected_keys)))
            if len(missing_keys) > 0:
                error_msgs.insert(
                    0, 'Missing key(s) in state_dict: {}. '.format(
                        ', '.join('"{}"'.format(k) for k in missing_keys)))

        if len(error_msgs) > 0:
            raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                self.__class__.__name__, "\n\t".join(error_msgs)))
        return _IncompatibleKeys(missing_keys, unexpected_keys)

    def fix_state_dict(self, state_dict):
        newstate = OrderedDict()
        hidden = self.weight_fh.shape[0]
        bias_i = torch.zeros(hidden)
        bias_f = torch.zeros(hidden)
        bias_a = torch.zeros(hidden)
        bias_o = torch.zeros(hidden)
        for name, value in state_dict.items():
            if name[:7] == 'bias_ih':
                bias_i = bias_i + value[:hidden]
                bias_f = bias_f + value[hidden:hidden * 2]
                bias_a = bias_a + value[2 * hidden:hidden * 3]
                bias_o = bias_o + value[3 * hidden:]
                # newstate['layernorm_ii.bias'] = value[:hidden]
                # newstate['layernorm_fi.bias'] = value[hidden:hidden * 2]
                # newstate['layernorm_ai.bias'] = value[2 * hidden:hidden * 3]
                # newstate['layernorm_oi.bias'] = value[3 * hidden:]
            elif name[:7] == 'bias_hh':
                bias_i = bias_i + value[:hidden]
                bias_f = bias_f + value[hidden:hidden * 2]
                bias_a = bias_a + value[2 * hidden:hidden * 3]
                bias_o = bias_o + value[3 * hidden:]
                # newstate['layernorm_ih.bias'] = value[:hidden]
                # newstate['layernorm_fh.bias'] = value[hidden:hidden * 2]
                # newstate['layernorm_ah.bias'] = value[2 * hidden:hidden * 3]
                # newstate['layernorm_oh.bias'] = value[3 * hidden:]
            elif name.split('_')[1] == 'ih':
                newstate['weight_ii'] = value[:hidden, :]
                newstate['weight_fi'] = value[hidden:hidden * 2, :]
                newstate['weight_ai'] = value[2 * hidden:hidden * 3, :]
                newstate['weight_oi'] = value[3 * hidden:, :]
            elif name.split('_')[1] == 'hh':
                newstate['weight_ih'] = value[:hidden, :]
                newstate['weight_fh'] = value[hidden:hidden * 2, :]
                newstate['weight_ah'] = value[2 * hidden:hidden * 3, :]
                newstate['weight_oh'] = value[3 * hidden:, :]
            else:
                newstate[name] = value
        if 'layernorm_i.weight' not in state_dict.items():
            newstate['layernorm_i.weight'] = torch.ones(1)
            newstate['layernorm_f.weight'] = torch.ones(1)
            newstate['layernorm_a.weight'] = torch.ones(1)
            newstate['layernorm_o.weight'] = torch.ones(1)
            newstate['layernorm_c.weight'] = torch.ones(1)

            newstate['layernorm_i.running_mean'] = torch.zeros(1)
            newstate['layernorm_f.running_mean'] = torch.zeros(1)
            newstate['layernorm_a.running_mean'] = torch.zeros(1)
            newstate['layernorm_o.running_mean'] = torch.zeros(1)
            newstate['layernorm_c.running_mean'] = torch.zeros(1)

            newstate['layernorm_i.running_var'] = torch.ones(1)
            newstate['layernorm_f.running_var'] = torch.ones(1)
            newstate['layernorm_a.running_var'] = torch.ones(1)
            newstate['layernorm_o.running_var'] = torch.ones(1)
            newstate['layernorm_c.running_var'] = torch.ones(1)

        newstate['bias_i'] = bias_i
        newstate['bias_f'] = bias_f
        newstate['bias_a'] = bias_a
        newstate['bias_o'] = bias_o

        return newstate


class BidirLSTMLayer(nn.Module):
    __constants__ = ['directions']

    def __init__(self, input_size, hidden_size, weight_config, activation_config, layer_norm='identity',
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
