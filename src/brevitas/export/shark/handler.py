# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Optional

from brevitas.proxy.groupwise_float_parameter_quant import GroupwiseWeightFloatQuantProxyFromInjector
<<<<<<< HEAD
from brevitas_examples.stable_diffusion.sd_quant.nn import QuantAttention
=======
>>>>>>> 03653239c02f2bef1da03223b6b5f452c98f8652
from sharktank.types import DefaultPrimitiveTensor
from sharktank.types import DynamicFp4BlockQuantizer
from sharktank.types import StaticFp4BlockQuantizer
from sharktank.types import StaticScaledQuantizer
import torch
from torch import Tensor
import torch.nn as nn

from brevitas.export.inference.handler import GroupwiseActFloatQuantProxyFromInjector
from brevitas.export.inference.handler import GroupwiseWeightQuantProxyFromInjector
from brevitas.function.ops import max_int
from brevitas.function.ops import min_int
import brevitas.nn as qnn
from brevitas.nn.equalized_layer import EqualizedModule
from brevitas.proxy import ActFloatQuantProxyFromInjector
from brevitas.proxy.float_parameter_quant import WeightFloatQuantProxyFromInjector
from brevitas.proxy.parameter_quant import WeightQuantProxyFromInjector
from brevitas.proxy.runtime_quant import ActQuantProxyFromInjector


class SharkActEqualization(nn.Module):
    handled_layer = EqualizedModule

    def __init__(self):
        super().__init__()
        self.layer_name = None
        self.shared_dict = None

    def attach_debug_info(self, module: nn.Module):
        pass

    def prepare_for_export(self, module: nn.Module):
        if hasattr(module, 'allocate_params'):
            module.allocate_params(module)
        # We need to flatten out the structure of EqualizedLayer + Wrapped layer
        # To do this, we change the name (i.e., state dict prefix) of the wrapped layer to match
        # the name of the EqualizedLayer.
        # Similarly, we lift the activation equalization weight from the scale layer
        # The result is:
        # from: name.scale.weight, name.layer.weight, name.bias
        # to : name.premul_input, name.weight, name.bias
        if hasattr(module.layer, 'export_handler') and module.layer.export_handler is not None:
            module.layer.export_handler.layer_name = self.layer_name
        else:
            for n, p in module.layer.named_parameters():
                param_name = f"{self.layer_name}.{n}"
                param = DefaultPrimitiveTensor(
                    name=param_name,
                    data=p)
                self.shared_dict[param_name] = param
        self.premul_input = module.scale.weight.contiguous()
        self.premul_module = module.scale
        if hasattr(module, 'offload_params'):
            module.offload_params(module)

    def forward(self, x):
        assert self.layer_name is not None
        assert self.shared_dict is not None

        premul_input = DefaultPrimitiveTensor(
            name=f"{self.layer_name}.premul_input",
            data=self.premul_input,
        )
        self.shared_dict[premul_input.name] = premul_input
        return self.premul_module(x)


class SharkWeightQuantMixin:

    def __init__(self):
        self.module = None

    @staticmethod
    def prepare_weight_for_export(module: nn.Module):
        if module.is_quant_enabled:
            # Continguous is used to be extra-safe with torch.compile
            qw = module(module.tracked_parameter_list[0])
            scale = qw.scale_ if hasattr(qw, 'scale_') else qw.scale
            scale = scale.contiguous().cpu()
            zero_point = module.zero_point().contiguous().to(torch.float32).cpu()
            zero_point = None if torch.count_nonzero(zero_point) == 0 else (zero_point -
                                                                            128.).to(scale.device)
            quant_metadata = {'scale': scale, 'zero_point': zero_point}
            quant_metadata['quant'] = StaticScaledQuantizer
            quant_metadata['groupwise'] = False

            if isinstance(module, WeightQuantProxyFromInjector):
                assert module.bit_width() == 8., "Only Int8 is supported for export"
                quant_metadata['dtype'] = torch.int8
            elif isinstance(module, WeightFloatQuantProxyFromInjector):
                if module.is_ocp_e5m2:
                    quant_metadata['dtype'] = torch.float8_e5m2
                elif module.is_ocp_e4m3:
                    quant_metadata['dtype'] = torch.float8_e4m3fn
                elif module.is_fnuz_e5m2:
                    quant_metadata['dtype'] = torch.float8_e5m2fnuz
                elif module.is_fnuz_e4m3:
                    quant_metadata['dtype'] = torch.float8_e4m3fnuz
                else:
                    raise ValueError("Dtype not supported for export")
            elif isinstance(module, GroupwiseWeightFloatQuantProxyFromInjector):
                # TODO assert for all the other properties of MXFP4
                assert module.group_size == 32
                assert module.is_ocp_e2m1
                quant_metadata['quant'] = StaticFp4BlockQuantizer
                quant_metadata['groupwise'] = True
                # Not sure if this is correct?
                quant_metadata['dtype'] = scale.dtype
            else:
                raise

            return quant_metadata
        else:
            return None

    def weight_quant(self, weight, quant_metadata, *args):
        QuantClass = quant_metadata['quant']
        scale = quant_metadata['scale']
        zero_point = quant_metadata['zero_point']
        layer_name = quant_metadata['layer_name']
        shared_dict = quant_metadata['shared_dict']
        dtype = quant_metadata['dtype']
        scale = scale.to(weight.device)
        if zero_point is not None:
            zero_point = zero_point.to(weight.device)

        is_groupwise = quant_metadata['groupwise']

        if not is_groupwise:
            kwargs = {
                'scale': torch.reciprocal(scale),
                'reciprocal_scale': scale,
                'offset': zero_point,
                'dtype': dtype}
        else:
            scale = scale.squeeze()
            kwargs = {'scales': torch.reciprocal(scale), 'dtype': dtype}

        weight_quant = QuantClass(name=layer_name, **kwargs)
        quant_weight = weight_quant.quantize(weight, name=layer_name)
        shared_dict[layer_name] = quant_weight
        return args


class SharkActQuantMixin:

    @staticmethod
    def prepare_act_for_export(module: nn.Module):
        if module.is_quant_enabled:
            # Continguous is used to be extra-safe with torch.compile
            quant_metadata = dict()
            quant_metadata['is_dynamic'] = False
            if isinstance(module, ActQuantProxyFromInjector):
                scale = module.scale().contiguous().cpu()
                zero_point = module.zero_point().contiguous().to(torch.float32).cpu()
                zero_point = None if torch.count_nonzero(zero_point) == 0 else (zero_point - 128.)
                quant_metadata['scale'] = scale
                quant_metadata['zero_point'] = zero_point
                assert module.bit_width() == 8., "Only Int8 is supported for export"
                quant_metadata['dtype'] = torch.int8
            elif isinstance(module, ActFloatQuantProxyFromInjector):
                scale = module.scale().contiguous().cpu()
                zero_point = module.zero_point().contiguous().to(torch.float32).cpu()
                zero_point = None if torch.count_nonzero(zero_point) == 0 else (zero_point - 128.)
                quant_metadata['scale'] = scale
                quant_metadata['zero_point'] = zero_point
                if module.is_ocp_e5m2:
                    quant_metadata['dtype'] = torch.float8_e5m2
                elif module.is_ocp_e4m3:
                    quant_metadata['dtype'] = torch.float8_e4m3fn
                elif module.is_fnuz_e5m2:
                    quant_metadata['dtype'] = torch.float8_e5m2fnuz
                elif module.is_fnuz_e4m3:
                    quant_metadata['dtype'] = torch.float8_e4m3fnuz
                else:
                    raise ValueError("Dtype not supported for export")
            elif isinstance(module, GroupwiseActFloatQuantProxyFromInjector):
                # TODO assert for all the other properties of MXFP4
                assert module.group_size == 32
                assert module.is_ocp_e2m1
                quant_metadata['is_dynamic'] = True
                # Not sure if this is correct?

            return quant_metadata
        else:
            return None

    def static_act_quant(self, quant_metadata, *args):
        scale = quant_metadata['scale']
        zero_point = quant_metadata['zero_point']
        layer_name = quant_metadata['layer_name']
        shared_dict = quant_metadata['shared_dict']
        dtype = quant_metadata['dtype']

        input_quant = StaticScaledQuantizer(
            name=layer_name,
            scale=torch.reciprocal(scale),
            reciprocal_scale=scale,
            offset=zero_point,
            dtype=dtype)
        shared_dict[layer_name] = input_quant

    def dynamic_act_quant(self, quant_metadata, *args):
        inp = args[0]
        dtype = inp.dtype
        layer_name = quant_metadata['layer_name']
        shared_dict = quant_metadata['shared_dict']
        input_quant = DynamicFp4BlockQuantizer(name=layer_name, dtype=dtype)
        shared_dict[layer_name] = input_quant

    def act_quant(self, quant_metadata, *args):
        if quant_metadata is None:
            return args
        if quant_metadata['is_dynamic']:
            self.dynamic_act_quant(quant_metadata, *args)
        else:
            self.static_act_quant(quant_metadata, *args)
        return args


class SharkLinearQuant(nn.Module, SharkWeightQuantMixin, SharkActQuantMixin):
    handled_layer = qnn.QuantLinear

    def __init__(self):
        super().__init__()
        self.layer_name = None
        self.shared_dict = None
        self.init_done = False

    def attach_debug_info(self, module: nn.Module):
        pass

    def prepare_for_export(self, module: nn.Module):
        if hasattr(module, 'allocate_params'):
            module.allocate_params(module)
        self.quant_weight_metadata = self.prepare_weight_for_export(module.weight_quant)
        if self.quant_weight_metadata is not None:
            self.quant_weight_metadata['layer_name'] = self.layer_name + '.weight'
            self.quant_weight_metadata['shared_dict'] = self.shared_dict
        self.quant_input_metadata = self.prepare_act_for_export(module.input_quant)
        if self.quant_input_metadata is not None:
            self.quant_input_metadata['layer_name'] = self.layer_name + '.q_input'
            self.quant_input_metadata['shared_dict'] = self.shared_dict
        self.quant_output_metadata = self.prepare_act_for_export(module.output_quant)
        if self.quant_output_metadata is not None:
            self.quant_output_metadata['layer_name'] = self.layer_name + '.q_output'
            self.quant_output_metadata['shared_dict'] = self.shared_dict
        if hasattr(module, 'offload_params'):
            module.offload_params(module)
        self.module_weight = module.weight
        self.module_bias = module.bias

    def forward(self, x):
        assert self.layer_name is not None
        assert self.shared_dict is not None

        quant_weight = self.weight_quant(self.module_weight, self.quant_weight_metadata, self.module_weight)[0]
        quant_input = self.act_quant(self.quant_input_metadata, x)[0]
        out = torch.nn.functional.linear(quant_input, quant_weight, self.module_bias)
        quant_out = self.act_quant(self.quant_output_metadata, out)[0]
        return quant_out


class SharkConvQuant(nn.Module, SharkWeightQuantMixin, SharkActQuantMixin):
    handled_layer = qnn.QuantConv2d

    def __init__(self):
        super().__init__()
        self.layer_name = None
        self.shared_dict = None
        self.init_done = False

    def attach_debug_info(self, module: nn.Module):
        pass

    def prepare_for_export(self, module: nn.Module):
        if hasattr(module, 'allocate_params'):
            module.allocate_params(module)
        self.quant_weight_metadata = self.prepare_weight_for_export(module.weight_quant)
        if self.quant_weight_metadata is not None:
            self.quant_weight_metadata['layer_name'] = self.layer_name + '.weight'
            self.quant_weight_metadata['shared_dict'] = self.shared_dict
        self.quant_input_metadata = self.prepare_act_for_export(module.input_quant)
        if self.quant_input_metadata is not None:
            self.quant_input_metadata['layer_name'] = self.layer_name + '.q_input'
            self.quant_input_metadata['shared_dict'] = self.shared_dict
        self.quant_output_metadata = self.prepare_act_for_export(module.output_quant)
        if self.quant_output_metadata is not None:
            self.quant_output_metadata['layer_name'] = self.layer_name + '.q_output'
            self.quant_output_metadata['shared_dict'] = self.shared_dict
        if hasattr(module, 'offload_params'):
            module.offload_params(module)
        self.module_weight = module.weight
        self.module_bias = module.bias
        self.stride = module.stride
        self.padding = module.padding
        self.dilation = module.dilation
        self.groups = module.groups

    def forward(self, x):
        assert self.layer_name is not None
        assert self.shared_dict is not None

        quant_weight = self.weight_quant(self.module_weight, self.quant_weight_metadata, self.module_weight)[0]
        quant_input = self.act_quant(self.quant_input_metadata, x)[0]
        out = torch.nn.functional.conv2d(quant_input, quant_weight, self.module_bias, self.stride, self.padding, self.dilation, self.groups)
        quant_out = self.act_quant(self.quant_output_metadata, out)[0]
        return quant_out


class SharkQuantIdentity(nn.Module, SharkWeightQuantMixin, SharkActQuantMixin):
    handled_layer = qnn.QuantIdentity

    def __init__(self):
        super().__init__()
        self.layer_name = None
        self.shared_dict = None
        self.init_done = False

    def attach_debug_info(self, module: nn.Module):
        pass

    def prepare_for_export(self, module: nn.Module):
        if hasattr(module, 'allocate_params'):
            module.allocate_params(module)
        # self.quant_weight_metadata = self.prepare_weight_for_export(module.weight_quant)
        # if self.quant_weight_metadata is not None:
        #     self.quant_weight_metadata['layer_name'] = self.layer_name + '.weight'
        #     self.quant_weight_metadata['shared_dict'] = self.shared_dict
        # self.quant_input_metadata = self.prepare_act_for_export(module.input_quant)
        # if self.quant_input_metadata is not None:
        #     self.quant_input_metadata['layer_name'] = self.layer_name + '.q_input'
        #     self.quant_input_metadata['shared_dict'] = self.shared_dict
        self.quant_act = self.prepare_act_for_export(module.act_quant)
        if self.quant_act is not None:
            self.quant_act['layer_name'] = self.layer_name
            self.quant_act['shared_dict'] = self.shared_dict
        if hasattr(module, 'offload_params'):
            module.offload_params(module)


    def forward(self, x):
        assert self.layer_name is not None
        assert self.shared_dict is not None

        quant_input = self.act_quant(self.quant_act, x)[0]

        return x


class SharkQuantSDPA(nn.Module, SharkWeightQuantMixin, SharkActQuantMixin):
    handled_layer = qnn.QuantScaledDotProductAttention

    def __init__(self):
        super().__init__()
        self.layer_name = None
        self.shared_dict = None
        self.init_done = False

    def attach_debug_info(self, module: nn.Module):
        pass

    def prepare_for_export(self, module: nn.Module):
        self.q_scaled_quant = self.prepare_act_for_export(module.q_scaled_quant.act_quant)
        if self.q_scaled_quant is not None:
            self.q_scaled_quant['layer_name'] = self.layer_name + '_q_output'
            self.q_scaled_quant['shared_dict'] = self.shared_dict
        self.k_transposed_quant = self.prepare_act_for_export(module.k_transposed_quant.act_quant)
        if self.k_transposed_quant is not None:
            self.k_transposed_quant['layer_name'] = self.layer_name + '_k_output'
            self.k_transposed_quant['shared_dict'] = self.shared_dict
        self.v_quant = self.prepare_act_for_export(module.v_quant.act_quant)
        if self.v_quant is not None:
            self.v_quant['layer_name'] = self.layer_name + '_v_output'
            self.v_quant['shared_dict'] = self.shared_dict
        self.pre_forward = module.pre_forward

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
        assert self.layer_name is not None
        assert self.shared_dict is not None

        query = self.act_quant(self.q_scaled_quant, query)[0]
        key = self.act_quant(self.k_transposed_quant, key)[0]
        value = self.act_quant(self.v_quant, value)[0]

        kwargs = {}
        if scale is not None:
            kwargs["scale"] = scale
        if enable_gqa is not None:
            kwargs["enable_gqa"] = enable_gqa

        return torch.nn.functional.scaled_dot_product_attention(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            **kwargs)
