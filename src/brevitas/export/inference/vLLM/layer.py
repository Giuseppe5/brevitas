# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import math
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import torch
from vllm.model_executor.layers.linear import LinearMethodBase
from vllm.model_executor.parameter import ModelWeightParameter

from brevitas.graph.hadamard import get_hadK
from brevitas.nn.equalized_layer import RotatedModule

from ..handler import FloatInferencetHandler
from ..handler import FloatWeightInferencetHandler
from ..handler import GroupwiseFloatWeightInferenceHandler
from ..handler import GroupwiseIntWeightInferenceHandler
from ..handler import IntInferenceHandler
from ..handler import IntWeightInferencetHandler
from .handler import vLLMDynamicPerRowFloatInferenceHandler
from .handler import vLLMGroupwiseFloatInferenceHandler
from .handler import vLLMGroupwiseIntInferenceHandler

class_mapping = {
    'vLLMGroupwiseFloatInferenceHandler': vLLMGroupwiseFloatInferenceHandler,
    'vLLMGroupwiseIntInferenceHandler': vLLMGroupwiseIntInferenceHandler,
    'GroupwiseIntWeightInferenceHandler': GroupwiseIntWeightInferenceHandler,
    'GroupwiseFloatWeightInferenceHandler': GroupwiseFloatWeightInferenceHandler,
    'FloatInferencetHandler': FloatInferencetHandler,
    'FloatWeightInferencetHandler': FloatWeightInferencetHandler,
    'IntWeightInferencetHandler': IntWeightInferencetHandler,
    'IntInferenceHandler': IntInferenceHandler,
    'vLLMDynamicPerRowFloatInferenceHandler': vLLMDynamicPerRowFloatInferenceHandler}


class QuantLinear(LinearMethodBase):

    def __init__(self, quant_configs: Optional[Dict[str, Any]] = None) -> None:

        self.input_quant = self.configure_proxy(quant_configs["input_config"])
        weight_config = quant_configs["weight_config"]
        if isinstance(weight_config, list):
            self.weight_quant = {
                i: self.configure_proxy(config) for i, config in enumerate(weight_config)}
        else:
            self.weight_quant = self.configure_proxy(weight_config)
        self.bias_quant = self.configure_proxy(quant_configs["bias_config"])
        self.output_quant = self.configure_proxy(quant_configs["output_config"])
        self.rotation = self.configure_rotation(quant_configs["rotation_config"])

    def configure_rotation(self, rotation_config: Optional[Dict[str,
                                                                Any]]) -> Optional[RotatedModule]:
        if rotation_config is None:
            return None
        rot_mat_shape = rotation_config['rot_mat_shape']
        k = rotation_config['k']
        if rot_mat_shape is None:
            had_mat = None
        else:
            had_mat, _ = get_hadK(rot_mat_shape)
        return RotatedModule(self, had_mat, k)

    def configure_proxy(self, quant_config: Optional[Dict[str, Any]]) -> torch.nn.Module:
        # No config, no quantizer
        if quant_config is None:
            return torch.nn.Identity()

        # Extract element that are not part of the state dict
        quant_class_name = quant_config['class_type']
        float_to_int_impl_type = quant_config['float_to_int_impl_type']
        scaling_restriction = quant_config['scaling_restriction']
        threshold_restriction = quant_config['threshold_restriction']
        del quant_config['class_type']
        del quant_config['float_to_int_impl_type']
        del quant_config['scaling_restriction']
        del quant_config['threshold_restriction']

        # Scale and zero-point are the only float elements in the state dict
        for k, v in quant_config.items():
            if not isinstance(v, torch.Tensor):
                if k == 'scale' or k == 'zero_point':
                    quant_config[k] = torch.tensor(v)
                else:
                    quant_config[k] = torch.tensor(v, dtype=torch.int)

        # Shapes must be set otherwise the state dict loading will fail
        scale = quant_config.get('scale', None)
        zero_point = quant_config.get('zero_point', None)
        quant_class = class_mapping[quant_class_name]
        if scale is None and zero_point is None:
            quantizer = quant_class()
        else:
            scale_shape = scale.shape
            zero_point_shape = zero_point.shape
            quantizer = quant_class(scale_shape=scale_shape, zero_point_shape=zero_point_shape)

        # Set the remaining attributes
        quantizer.float_to_int_impl_type = float_to_int_impl_type
        if scaling_restriction is not None:
            quantizer.scaling_restriction = scaling_restriction
        if threshold_restriction is not None:
            quantizer.threshold_restriction = threshold_restriction
        quantizer.float_to_int_impl_type = float_to_int_impl_type
        quantizer.load_state_dict(quant_config)
        return quantizer

    def create_weights(
            self,
            layer: torch.nn.Module,
            input_size_per_partition: int,
            output_partition_sizes: List[int],
            input_size: int,
            output_size: int,
            params_dtype: torch.dtype,
            **extra_weight_attrs) -> None:
        weight_loader = extra_weight_attrs.get("weight_loader")
        self.input_size_per_partition = input_size_per_partition
        self.output_partition_sizes = output_partition_sizes

        weight = ModelWeightParameter(
            data=torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

    def process_weights_after_loading(self, module: torch.nn.Module) -> None:
        weight = module.weight.data
        for i in range(len(self.output_partition_sizes)):
            logical_widths = list(self.output_partition_sizes)
            start_idx = sum(logical_widths[:i])
            end_idx = start_idx + logical_widths[i]
            if isinstance(self.weight_quant, dict):
                weight_quant = self.weight_quant[i]
            else:
                weight_quant = self.weight_quant

            weight[start_idx:end_idx] = weight_quant(weight[start_idx:end_idx])[0]

    def apply(
            self,
            layer: torch.nn.Module,
            x: torch.Tensor,
            bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.rotation is not None:
            x = self.rotation.rotation_forward(x)
        x = self.input_quant(x)[0]
        bias = self.bias_quant(bias) if bias is not None else None
        y = torch.nn.functional.linear(x, layer.weight, bias)
        y = self.output_quant(y)
        return y


from aiter.ops.quant import per_1x32_f8_scale_f8_quant
from aiter.ops.triton.gemm.basic.gemm_a8w8_blockscale import gemm_a8w8_blockscale
from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
from aiter.ops.triton.quant.quant import dynamic_mxfp4_quant
from aiter.ops.triton.quant.sage_attention_quant_wrappers import create_hadamard_matrix
from aiter.utility import dtypes
from aiter.utility import fp4_utils


class QOPQuantLinearBase(QuantLinear):
    """Base class for aiter-accelerated QOP linear layers.

    Provides shared weight quantization setup, blockwise Hadamard rotation via
    aiter's ``create_hadamard_matrix``, and weight scale post-processing.
    Subclasses implement precision-specific activation quantization and GEMM.
    """

    def __init__(self, quant_configs: Optional[Dict[str, Any]] = None):
        weight_config = quant_configs["weight_config"]
        if isinstance(weight_config, list):
            self.weight_quant = {
                i: self.configure_proxy(config) for i, config in enumerate(weight_config)}
            self.weight_scale = torch.cat([x.scale for x in self.weight_quant.values()])
        else:
            self.weight_quant = self.configure_proxy(weight_config)
            self.weight_scale = self.weight_quant.scale
        self.init_weight = False
        self.had_mat = None
        self.had_k = None
        rotation_config = quant_configs.get("rotation_config", None)
        if rotation_config is not None:
            self._configure_aiter_rotation(rotation_config)

    def _configure_aiter_rotation(self, rotation_config):
        """Set up blockwise Hadamard rotation using aiter's create_hadamard_matrix."""
        if rotation_config is None:
            return
        rot_mat_shape = rotation_config.get('rot_mat_shape', None)
        k = rotation_config.get('k', None)
        if rot_mat_shape is not None and k is not None:
            self.had_k = k
            if k == 1:
                # Pure power-of-2 Hadamard: use aiter's recursive construction
                self.had_mat = None
            else:
                # Non-power-of-2: load pre-computed Hadamard from brevitas
                had_mat, _ = get_hadK(rot_mat_shape)
                self.had_mat = had_mat

    def _apply_blockwise_hadamard(self, x):
        """Apply blockwise Hadamard rotation using aiter kernels.

        For a 2D input (M, K), applies H @ x blockwise along the last dimension.
        Uses aiter's create_hadamard_matrix for power-of-2 block sizes, and
        the pre-computed Hadamard tensor from brevitas for non-power-of-2 cases.
        """
        if self.had_k is None:
            return x
        init_shape = x.shape
        n = x.shape[-1]
        if self.had_k == 1:
            # Pure power-of-2: apply full Hadamard via blockwise matmul
            had = create_hadamard_matrix(n, device=x.device, dtype=x.dtype) / math.sqrt(n)
            x = x.reshape(-1, n)
            x = x @ had.t()
        else:
            # Block-structured Hadamard: H_K (x) H_{n/K}
            K = self.had_k
            had_K = self.had_mat.to(device=x.device, dtype=x.dtype)
            block_size = n // K
            # Apply power-of-2 Hadamard on inner blocks
            had_block = create_hadamard_matrix(
                block_size, device=x.device, dtype=x.dtype) / math.sqrt(n)
            x = x.reshape(-1, K, block_size)
            x = x @ had_block.t()
            # Apply the K-sized Hadamard
            x = had_K.to(x.dtype) @ x
        return x.reshape(init_shape)


class QOPQuantLinearMXFP4(QOPQuantLinearBase):
    """MXFP4 linear layer using aiter kernels for fast quantized execution.

    Uses aiter's ``dynamic_mxfp4_quant`` for on-the-fly activation quantization,
    ``gemm_afp4wfp4`` for FP4×FP4 GEMM, and blockwise Hadamard rotation via
    aiter's ``create_hadamard_matrix``.
    """

    def __init__(self, quant_configs: Optional[Dict[str, Any]] = None):
        super().__init__(quant_configs)
        self.input_quant = self.quantize_inp

    def quantize_inp(self, x):
        return dynamic_mxfp4_quant(x)

    def post_process_quant_triton(self, scale, inp_shape):
        """Post-process weight scales to padded float8_e8m0fnu format for Triton GEMM."""
        first_shape = inp_shape[0]
        reshaped_scale = scale.view(first_shape, -1)
        expanded_shape = (
            math.ceil(reshaped_scale.shape[0] / 256) * 256,
            math.ceil(reshaped_scale.shape[1] / 8) * 8)
        padded_scale = 127 * torch.ones(
            expanded_shape[0], expanded_shape[1], dtype=torch.uint8, device=reshaped_scale.device)
        padded_scale[:reshaped_scale.shape[0], :reshaped_scale
                     .shape[1]] = torch.log2(reshaped_scale) + 127
        padded_scale = padded_scale.view(torch.float8_e8m0fnu)
        return padded_scale

    def process_weights_after_loading(self, module: torch.nn.Module) -> None:
        weight = module.weight.data
        for i in range(len(self.output_partition_sizes)):
            logical_widths = list(self.output_partition_sizes)
            start_idx = sum(logical_widths[:i])
            end_idx = start_idx + logical_widths[i]
            if isinstance(self.weight_quant, dict):
                weight_quant = self.weight_quant[i]
            else:
                weight_quant = self.weight_quant

            weight[start_idx:end_idx] = weight_quant.quantize_forward(weight[start_idx:end_idx])[0]
        weight_shape = module.weight.shape
        self.weight_scale = self.post_process_quant_triton(self.weight_scale, weight_shape)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        orig_inp_shape = x.shape
        x = x.reshape(-1, orig_inp_shape[-1])

        # Blockwise Hadamard rotation via aiter
        x = self._apply_blockwise_hadamard(x)

        M, K = x.shape
        N, _ = layer.weight.shape

        weight = fp4_utils.f32_to_mxfp4(layer.weight.data.float())

        weight_scale = self.weight_scale
        x, x_scale = self.input_quant(x)

        y = torch.empty(M, N, device=x.device, dtype=torch.bfloat16)
        gemm_afp4wfp4(
            x.view(torch.uint8),
            weight.view(torch.uint8),
            x_scale.view(torch.uint8),
            weight_scale.view(torch.uint8),
            torch.bfloat16,
            y)
        y = y.reshape(*orig_inp_shape[:-1], y.shape[-1])
        if bias is not None:
            y = y + bias
        return y


class QOPQuantLinearMXFP8(QOPQuantLinearBase):
    """MXFP8 linear layer using aiter kernels for fast quantized execution.

    Follows the MXFP spec: group_size=32, power-of-two (E8M0) scale factors.
    Uses aiter's ``per_1x32_f8_scale_f8_quant`` for on-the-fly FP8 activation
    quantization with E8M0 scales, ``gemm_a8w8_blockscale`` for FP8 blockscale
    GEMM, and blockwise Hadamard rotation via aiter's ``create_hadamard_matrix``.
    """

    # MXFP spec: group size of 32 with E8M0 (power-of-two) scale factors
    GROUP_SIZE = 32

    # GEMM config with BLOCK_SIZE_K=32 to match the MXFP group size
    _GEMM_CONFIG = {
        'BLOCK_SIZE_M': 128,
        'BLOCK_SIZE_N': 128,
        'BLOCK_SIZE_K': 32,
        'GROUP_SIZE_M': 1,
        'NUM_KSPLIT': 1,
        'num_warps': 4,
        'num_stages': 2,
        'waves_per_eu': 2,}

    def __init__(self, quant_configs: Optional[Dict[str, Any]] = None):
        super().__init__(quant_configs)
        self.input_quant = self.quantize_inp

    def quantize_inp(self, x):
        """Dynamic MXFP8 quantization: group_size=32, E8M0 power-of-two scales.

        Uses aiter's per_1x32_f8_scale_f8_quant with E8M0 scale type to produce
        FP8 quantized activations with power-of-two block scales per the MX spec.
        The E8M0 scales are converted to float32 for the GEMM kernel.
        Returns (x_fp8, x_scale_f32).
        """
        x_fp8, x_scale_e8m0 = per_1x32_f8_scale_f8_quant(
            x, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0)
        x_scale_f32 = fp4_utils.e8m0_to_f32(x_scale_e8m0)
        return x_fp8, x_scale_f32

    def _post_process_weight_scale(self, scale, weight_shape):
        """Post-process weight scales for FP8 blockscale GEMM.

        Reshapes the per-group weight scale to (N, scale_k) where
        scale_k = ceil(K / GROUP_SIZE). Keeps float32 format as required
        by gemm_a8w8_blockscale.
        """
        N, K = weight_shape
        scale_k = math.ceil(K / self.GROUP_SIZE)
        return scale.view(N, scale_k).float()

    def process_weights_after_loading(self, module: torch.nn.Module) -> None:
        weight = module.weight.data
        for i in range(len(self.output_partition_sizes)):
            logical_widths = list(self.output_partition_sizes)
            start_idx = sum(logical_widths[:i])
            end_idx = start_idx + logical_widths[i]
            if isinstance(self.weight_quant, dict):
                weight_quant = self.weight_quant[i]
            else:
                weight_quant = self.weight_quant

            weight[start_idx:end_idx] = weight_quant.quantize_forward(weight[start_idx:end_idx])[0]
        weight_shape = module.weight.shape
        self.weight_scale = self._post_process_weight_scale(self.weight_scale, weight_shape)
        # Quantize weights to FP8 format
        module.weight.data = module.weight.data.to(dtypes.fp8)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        orig_inp_shape = x.shape
        x = x.reshape(-1, orig_inp_shape[-1])

        # Blockwise Hadamard rotation via aiter
        x = self._apply_blockwise_hadamard(x)

        M, K = x.shape
        N, _ = layer.weight.shape

        weight_scale = self.weight_scale

        # Dynamic MXFP8 activation quantization (group_size=32, E8M0 scales)
        x_fp8, x_scale = self.input_quant(x)

        y = gemm_a8w8_blockscale(
            x_fp8,
            layer.weight.data,
            x_scale,
            weight_scale,
            dtype=torch.bfloat16,
            config=dict(self._GEMM_CONFIG))
        y = y.reshape(*orig_inp_shape[:-1], y.shape[-1])
        if bias is not None:
            y = y + bias
        return y
