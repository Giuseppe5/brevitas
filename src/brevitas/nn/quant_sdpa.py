"""
Copyright (C) 2024,     Advanced Micro Devices, Inc.
Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
Copyright (c) 2011-2013 NYU                      (Clement Farabet)
Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

3. Neither the names of AMD, Facebook, Deepmind Technologies, NYU,
   NEC Laboratories America and IDIAP Research Institute nor the names
   of its contributors may be used to endorse or promote products derived
   from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""

import math
from typing import Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn import Module
from torch.nn import Parameter
import torch.nn.functional as F

from brevitas.core.function_wrapper.misc import Identity
from brevitas.function import identity
from brevitas.quant.scaled_int import Int8ActPerTensorFloat
from brevitas.quant.scaled_int import Uint8ActPerTensorFloat

from .quant_activation import QuantIdentity


class ScaledDotProductAttention(Module):

    def __init__(self, pre_process_q=identity, pre_process_k=identity, pre_process_v=identity):
        super().__init__()
        self.pre_process_q = pre_process_q
        self.pre_process_k = pre_process_k
        self.pre_process_v = pre_process_v

    def forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            attn_mask: Optional[Tensor] = None,
            dropout_p: float = 0.0,
            is_causal: bool = False,
            scale: Optional[float] = None,
            enable_gqa: bool = False):
        r"""
        Args:
            query (Tensor): Query tensor; shape :math:`(N, ..., Hq, L, E)`.
            key (Tensor): Key tensor; shape :math:`(N, ..., H, S, E)`.
            value (Tensor): Value tensor; shape :math:`(N, ..., H, S, Ev)`.
            attn_mask (optional Tensor): Attention mask; shape must be broadcastable to the shape of attention weights,
                which is :math:`(N,..., L, S)`. Two types of masks are supported.
                A boolean mask where a value of True indicates that the element *should* take part in attention.
                A float mask of the same type as query, key, value that is added to the attention score.
            dropout_p (float): Dropout probability; if greater than 0.0, dropout is applied
            is_causal (bool): If set to true, the attention masking is a lower triangular matrix when the mask is a
                square matrix. The attention masking has the form of the upper left causal bias due to the alignment
                (see :class:`torch.nn.attention.bias.CausalBias`) when the mask is a non-square matrix.
                An error is thrown if both attn_mask and is_causal are set.
            scale (optional float, keyword-only): Scaling factor applied prior to softmax. If None, the default value is set
                to :math:`\frac{1}{\sqrt{E}}`.
            enable_gqa (bool): Ignored to make calling interface compatible with PyTorch >v2.5. Always set to False.

        Returns:
            output (Tensor): Attention output; shape :math:`(N, ..., Hq, L, Ev)`.

        Shape legend:
            - :math:`N: \text{Batch size} ... : \text{Any number of other batch dimensions (optional)}`
            - :math:`S: \text{Source sequence length}`
            - :math:`L: \text{Target sequence length}`
            - :math:`E: \text{Embedding dimension of the query and key}`
            - :math:`Ev: \text{Embedding dimension of the value}`
            - :math:`Hq: \text{Number of heads of query}`
            - :math:`H: \text{Number of heads of key and value}`
        """
        kwargs = {}
        if scale is not None:
            kwargs["scale"] = scale
        if enable_gqa:
            kwargs["enable_gqa"] = enable_gqa
        return F.scaled_dot_product_attention(
            query=self.pre_process_q(query),
            key=self.pre_process_k(key),
            value=self.pre_process_v(value),
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            **kwargs)

from iree.turbine.kernel.wave.utils.general_utils import (
    get_default_scheduling_params,
)
from iree.turbine.kernel.wave.utils.run_utils import (
    set_default_run_config,
)
from iree.turbine.kernel.wave.utils.torch_utils import (
    device_zeros,
)
from iree.turbine.kernel.wave.compile import WaveCompileOptions, wave_compile
from iree.turbine.kernel.wave.constraints import MMAType
from iree.turbine.kernel.wave.templates.quantized_attention import (
    get_brevitas_pertensor_fp8_attention_kernel,
)
from iree.turbine.kernel.wave.templates.attention_common import AttentionShape
from iree.turbine.kernel.wave.scheduling.schedule import SchedulingType

# base_attention = None
# hyperparams = None
def functional_wave_kernel(q, k, v, q_scale, k_scale, v_scale):
    # Order of shapes: (B, M, N, K1, K2)
    # global base_attention
    # global hyperparams
    q_shape = q.shape
    k_shape = k.shape
    v_shape = v.shape
    original_dtype = q.dtype
    shape = AttentionShape(
        num_query_heads=q_shape[-3],
        num_kv_heads=k_shape[-3],
        query_seq_len=q_shape[-2],
        head_size_kv=v_shape[-1],
        head_size=q_shape[-1],
        kv_seq_len=v_shape[-2],
    )
    mfma_variant = (MMAType.F32_16x16x32_F8, MMAType.F32_16x16x32_K4_F8)#(MMAType.F32_32x32x16_F8, MMAType.F32_32x32x16_K4_F8)

    (
        base_attention,
        hyperparams,
        dynamic_symbols,
        dynamic_symbols_map,
    ) = get_brevitas_pertensor_fp8_attention_kernel(shape, mfma_variant, q_scale=q_scale.item(), k_scale=k_scale.item(), v_scale=v_scale.item())
    hyperparams.update(get_default_scheduling_params())
    dynamic_symbols = []
    dynamic_symbols_map = {}
    o_shape = (shape.num_query_heads, shape.query_seq_len, shape.head_size_kv)
    options = WaveCompileOptions(
        subs=hyperparams,
        schedule=SchedulingType.NONE,
        use_scheduling_barriers=False,
        dynamic_symbols=dynamic_symbols,
        dynamic_symbols_map=dynamic_symbols_map,
        waves_per_eu=2,
        denorm_fp_math_f32="preserve-sign",
        wave_runtime=True
    )

    options = set_default_run_config(options)
    base_attention_compiled = wave_compile(options, base_attention)

    output = device_zeros(o_shape, dtype=torch.float32)
    base_attention_compiled(q/q_scale, k/k_scale, v/v_scale, output)
    output = output.to(dtype=original_dtype)
    return output

class QuantScaledDotProductAttention(Module):

    def __init__(
            self,
            pre_process_q=identity,
            pre_process_k=identity,
            pre_process_v=identity,
            softmax_input_quant=None,
            attn_output_weights_quant=Uint8ActPerTensorFloat,
            q_scaled_quant=Int8ActPerTensorFloat,
            k_transposed_quant=Int8ActPerTensorFloat,
            v_quant=Int8ActPerTensorFloat,
            sdpa_output_quant=None,
            **kwargs) -> None:
        super(QuantScaledDotProductAttention, self).__init__()

        self.pre_process_q = pre_process_q
        self.pre_process_k = pre_process_k
        self.pre_process_v = pre_process_v
        self.use_wave = False

        def filter_kwargs(prefix):
            return {k[len(prefix):]: v for k, v in kwargs.items() if k.startswith(prefix)}

        self.q_scaled_quant = QuantIdentity(act_quant=q_scaled_quant, **filter_kwargs('q_scaled_'))
        self.k_transposed_quant = QuantIdentity(
            act_quant=k_transposed_quant, **filter_kwargs('k_transposed_'))
        self.v_quant = QuantIdentity(act_quant=v_quant, **filter_kwargs('v_'))
        self.softmax_input_quant = QuantIdentity(
            act_quant=softmax_input_quant, **filter_kwargs('softmax_input_'))
        self.attn_output_weights_quant = QuantIdentity(
            act_quant=attn_output_weights_quant, **filter_kwargs('attn_output_weights_'))
        self.sdpa_output_quant = QuantIdentity(
            act_quant=sdpa_output_quant, **filter_kwargs('sdpa_output_'))

    def forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            attn_mask: Optional[Tensor] = None,
            dropout_p: float = 0.0,
            is_causal: bool = False,
            scale: Optional[float] = None,
            enable_gqa: bool = False):
        r"""
        Args:
            query (Tensor): Query tensor; shape :math:`(N, ..., Hq, L, E)`.
            key (Tensor): Key tensor; shape :math:`(N, ..., H, S, E)`.
            value (Tensor): Value tensor; shape :math:`(N, ..., H, S, Ev)`.
            attn_mask (optional Tensor): Attention mask; shape must be broadcastable to the shape of attention weights,
                which is :math:`(N,..., L, S)`. Two types of masks are supported.
                A boolean mask where a value of True indicates that the element *should* take part in attention.
                A float mask of the same type as query, key, value that is added to the attention score.
            dropout_p (float): Dropout probability; if greater than 0.0, dropout is applied
            is_causal (bool): If set to true, the attention masking is a lower triangular matrix when the mask is a
                square matrix. The attention masking has the form of the upper left causal bias due to the alignment
                (see :class:`torch.nn.attention.bias.CausalBias`) when the mask is a non-square matrix.
                An error is thrown if both attn_mask and is_causal are set.
            scale (optional float, keyword-only): Scaling factor applied prior to softmax. If None, the default value is set
                to :math:`\frac{1}{\sqrt{E}}`.
            enable_gqa (bool): Ignored to make calling interface compatible with PyTorch >v2.5. Always set to False.

        Returns:
            output (Tensor): Attention output; shape :math:`(N, ..., Hq, L, Ev)`.

        Shape legend:
            - :math:`N: \text{Batch size} ... : \text{Any number of other batch dimensions (optional)}`
            - :math:`S: \text{Source sequence length}`
            - :math:`L: \text{Target sequence length}`
            - :math:`E: \text{Embedding dimension of the query and key}`
            - :math:`Ev: \text{Embedding dimension of the value}`
            - :math:`Hq: \text{Number of heads of query}`
            - :math:`H: \text{Number of heads of key and value}`
        """
        L, S = query.size(-2), key.size(-2)
        scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
        if attn_mask is None:
            attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
        else:
            attn_bias = torch.zeros(size=attn_mask.shape, dtype=query.dtype, device=query.device)
        if is_causal:
            assert attn_mask is None
            temp_mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril(diagonal=0)
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
            attn_bias.to(dtype=query.dtype, device=query.device)

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
            else:
                attn_bias += attn_mask
        query, key, value = self.pre_process_q(query), self.pre_process_k(key), self.pre_process_v(value)

        if not self.use_wave:

            q_scaled = self.q_scaled_quant(query )* scale_factor
            k_transpose = self.k_transposed_quant(key.transpose(-2, -1))
            attn_weight = q_scaled @ k_transpose
            attn_weight += attn_bias
            # attn_weight = self.softmax_input_quant(attn_weight)
            attn_weight = torch.softmax(attn_weight, dim=-1)
            attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
            # attn_weight = self.attn_output_weights_quant(attn_weight)
            attn_output = attn_weight @ self.v_quant(value)
            # attn_output = self.sdpa_output_quant(attn_output)

            return attn_output
        else:
            bq = None
            if len(query.shape) > 3:
                bq, sq = query.shape[0], query.shape[1]
                bk, sk = key.shape[0], key.shape[1]
                bv, sv = value.shape[0], value.shape[1]
                query = query.view(bq * sq, *query.shape[2:])
                key = key.view(bk * sk, *key.shape[2:])
                value = value.view(bv * sv, *value.shape[2:])
            # torch.save(query, 'query.pt')
            # torch.save(key, 'key.pt')
            # torch.save(value, 'value.pt')
            # raise
            q_scaled = self.q_scaled_quant(query)
            k_transpose = self.k_transposed_quant(key)
            value = self.v_quant(value)
            q_scale = self.q_scaled_quant.act_quant.fused_activation_quant_proxy(q_scaled)[1]
            k_scale = self.k_transposed_quant.act_quant.fused_activation_quant_proxy(k_transpose)[1]
            v_scale = self.v_quant.act_quant.fused_activation_quant_proxy(value)[1]
            oo = functional_wave_kernel(q_scaled, k_transpose, value, q_scale, k_scale, v_scale).unsqueeze(0)

            if bq:
                oo = oo.view(bq, -1, *oo.shape[2:])

            return oo

