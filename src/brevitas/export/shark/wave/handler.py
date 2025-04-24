# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABC
from abc import abstractmethod
from typing import Tuple

import torch
import torch.nn as nn

from brevitas.export.inference.handler import FloatInferencetHandler
from brevitas.export.inference.handler import FloatWeightInferencetHandler
from brevitas.export.inference.handler import IntInferencetHandler
from brevitas.export.inference.handler import IntWeightInferencetHandler
from brevitas.nn import QuantLinear


class InferenceHandler(torch.nn.Module, ABC):

    def attach_debug_info(self, module: nn.Module):
        pass

    @abstractmethod
    def prepare_for_export(self, module: nn.Module):
        pass


class QuantLinearHandler(InferenceHandler):
    handled_layer = QuantLinear

    def __init__(self):
        super().__init__()

        self.weight_quant = None  #FloatWeightInferencetHandler()
        self.input_quant = None  #FloatInferencetHandler()
        self.wave_linear = None

    def validate(self, module):
        # TODO: Check that we are quantizing to the correct fp8 type, etc. etc.
        pass

    def prepare_for_export(self, module):
        if True:  #check if is fp8 quantization
            self.weight_quant = FloatWeightInferencetHandler()
            self.input_quant = FloatInferencetHandler()
            self.inference_dtype = torch.float8_e4m3fnuz
        else:
            self.weight_quant = IntWeightInferencetHandler()
            self.input_quant = IntInferencetHandler()
            self.inference_dtype = torch.int8

        ## Weight export
        out_feat, input_feat = module.weight.shape[0], module.weight.shape[1]
        device = module.weight.device
        self.dtype = module.weight.dtype
        if module.weight_quant.is_quant_enabled:
            weight_quant = module.weight_quant
            self.weight_quant.prepare_for_export(weight_quant)
            self.weight_scale = self.weight_quant.scale.to(device).to(torch.float32)

        if module.input_quant.is_quant_enabled:
            input_quant = module.input_quant
            self.input_quant.prepare_for_export(input_quant)
            self.input_scale = self.input_quant.scale.to(device).to(torch.float32)

        self.bias = module.bias
        self.weight = module.weight
        del module.weight
        del module.bias

    def forward(self, input):
        input_q = self.input_quant.quantize(input, self.input_scale, None)
        weight_q = self.weight_quant.quantize(self.weight, self.weight_scale, None)

        if len(input_q.shape) == 3:
            B = input_q.shape[0]
            out = torch.stack([
                torch._scaled_mm(
                    input_q[i].to(self.inference_dtype),
                    weight_q.t().to(self.inference_dtype),
                    scale_a=self.input_scale,
                    scale_b=self.weight_scale,
                    bias=self.bias,
                    out_dtype=self.dtype) for i in range(B)],
                              dim=0)
        elif len(input_q.shape) == 2:
            out = torch._scaled_mm(
                input_q.to(self.inference_dtype),
                weight_q.t().to(self.inference_dtype),
                scale_a=self.input_scale,
                scale_b=self.weight_scale,
                bias=self.bias,
                out_dtype=self.dtype)

        return out
