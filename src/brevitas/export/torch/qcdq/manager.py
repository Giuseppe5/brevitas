# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Optional

from packaging.version import parse
import torch
from torch import Tensor
from torch.nn import Module

from brevitas import torch_version
from brevitas.export.manager import _set_proxy_export_handler
from brevitas.export.manager import _set_proxy_export_mode
from brevitas.export.manager import BaseManager
from brevitas.export.manager import ExportContext
from brevitas.graph.calibrate import QuantizationStatusManager
from brevitas.proxy.quant_proxy import QuantProxyProtocol

from .export_handler import TorchExportFloatQCDQActQuantProxyHandler
from .export_handler import TorchExportFloatQCDQWeightQuantProxyHandler
from .export_handler import TorchExportMXFP4ActQuantProxyHandler
from .export_handler import TorchExportMXFP4WeightQuantProxyHandler
from .export_handler import TorchExportQCDQActQuantProxyHandler
from .export_handler import TorchExportQCDQWeightQuantProxyHandler
from .handler import TorchCDQCastBiasQuantProxyHandler
from .handler import TorchQCDQCastActQuantProxyHandler
from .handler import TorchQCDQCastDecoupledWeightQuantProxyHandler
from .handler import TorchQCDQCastDecoupledWeightQuantWithInputProxyHandler
from .handler import TorchQCDQCastTruncQuantProxyHandler
from .handler import TorchQCDQCastWeightQuantProxyHandler


def _set_skip_create_quant_tensor(module: Module, enabled: bool):
    state = {}
    for submodule in module.modules():
        if hasattr(submodule, 'skip_create_quant_tensor'):
            state[submodule] = submodule.skip_create_quant_tensor
            submodule.skip_create_quant_tensor = enabled
    return state


def _restore_skip_create_quant_tensor(state):
    for module, enabled in state.items():
        module.skip_create_quant_tensor = enabled


class TorchQCDQManager(BaseManager):
    target_name = 'torch'

    handlers = [
        TorchQCDQCastWeightQuantProxyHandler,
        TorchQCDQCastDecoupledWeightQuantProxyHandler,
        TorchQCDQCastDecoupledWeightQuantWithInputProxyHandler,
        TorchQCDQCastActQuantProxyHandler,
        TorchCDQCastBiasQuantProxyHandler,
        TorchQCDQCastTruncQuantProxyHandler]

    @classmethod
    def set_export_mode(cls, model: Module, enabled: bool):
        _set_proxy_export_mode(model, enabled)

    @classmethod
    def set_export_handler(cls, module: Module):
        _set_proxy_export_handler(cls, module)

    @classmethod
    def change_weight_export(cls, export_weight_q_node: bool = False):
        for handler in cls.handlers:
            if hasattr(handler, '_export_q_node'):
                handler._export_weight_q_node = export_weight_q_node

    @classmethod
    def export(
            cls,
            module: Module,
            args,
            export_path: Optional[str] = None,
            export_weight_q_node: bool = False):
        cls.change_weight_export(export_weight_q_node=export_weight_q_node)
        with ExportContext(cls):
            traced_module = cls.jit_inference_trace(module, args, export_path)
        return traced_module


class TorchQCDQExportManager(BaseManager):
    target_name = 'torch_export'

    handlers = [
        TorchExportMXFP4WeightQuantProxyHandler,
        TorchExportMXFP4ActQuantProxyHandler,
        TorchExportFloatQCDQWeightQuantProxyHandler,
        TorchExportFloatQCDQActQuantProxyHandler,
        TorchExportQCDQWeightQuantProxyHandler,
        TorchExportQCDQActQuantProxyHandler]

    @classmethod
    def set_export_mode(cls, model: Module, enabled: bool):
        _set_proxy_export_mode(model, enabled)

    @classmethod
    def set_export_handler(cls, module: Module):
        _set_proxy_export_handler(cls, module)

    @classmethod
    def validate_exportable(cls, module: Module):
        for submodule in module.modules():
            if not isinstance(submodule, QuantProxyProtocol):
                continue
            if submodule.export_mode or submodule.export_handler is not None:
                raise RuntimeError('Torch QDQ export requires proxies outside export mode')
            if (submodule.requires_export_handler and
                    cls.handler_from_module(submodule, no_inheritance=True) is None):
                raise RuntimeError(
                    f'Torch QDQ export does not support quantizer proxy {type(submodule).__name__}')

    @classmethod
    def export(
        cls,
        module: Module,
        args,
        export_path: Optional[str] = None,
        kwargs=None,
        dynamic_shapes=None,
        strict: bool = True,
        preserve_module_call_signature=()):
        if torch_version < parse('2.12'):
            raise RuntimeError('Torch QDQ export requires PyTorch >= 2.12')
        if isinstance(args, Tensor):
            args = (args,)
        elif not isinstance(args, tuple):
            raise TypeError('args must be a Tensor or tuple of example inputs')
        if kwargs is None:
            kwargs = {}

        training_state = {submodule: submodule.training for submodule in module.modules()}
        return_quant_tensor_state = {}
        skip_create_quant_tensor_state = {}
        handlers_attached = False
        with torch.no_grad(), ExportContext(cls):
            try:
                cls.validate_exportable(module)
                module.eval()
                module.apply(cls.set_export_handler)
                handlers_attached = True
                cls._cache_inp_out(module, *args, **kwargs)
                cls.set_export_mode(module, enabled=True)
                return_quant_tensor_state = QuantizationStatusManager.disable_return_quant_tensor(
                    module)
                skip_create_quant_tensor_state = _set_skip_create_quant_tensor(module, True)
                exported_program = torch.export.export(
                    module,
                    args,
                    kwargs=kwargs,
                    dynamic_shapes=dynamic_shapes,
                    strict=strict,
                    preserve_module_call_signature=preserve_module_call_signature)
                if export_path is not None:
                    torch.export.save(exported_program, export_path)
                return exported_program
            finally:
                if handlers_attached:
                    cls.set_export_mode(module, enabled=False)
                _restore_skip_create_quant_tensor(skip_create_quant_tensor_state)
                QuantizationStatusManager.restore_return_quant_tensor(
                    module, return_quant_tensor_state)
                for submodule, training in training_state.items():
                    submodule.training = training
