# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from brevitas.nn.equalized_layer import EqualizedModule
from brevitas.nn.mixin.base import QuantLayerMixin
from sharktank.types import Dataset
from sharktank.types import DefaultPrimitiveTensor
from sharktank.types import Theta
import torch
from torch.nn import Module

from brevitas.export.manager import _set_layer_export_handler
from brevitas.export.manager import _set_layer_export_mode
from brevitas.export.manager import BaseManager
from brevitas.export.shark.handler import SharkActEqualization, SharkConvQuant, SharkQuantIdentity
from brevitas.export.shark.handler import SharkLinearQuant
from brevitas.export.shark.handler import SharkQuantSDPA


# Inheritance from BaseManager is not techincally needed
class SharkManager(BaseManager):
    handlers = [SharkActEqualization, SharkLinearQuant, SharkQuantSDPA, SharkConvQuant, SharkQuantIdentity]

    def __init__(self, model = None, config=None, output = None):
        super().__init__()
        if config == None:
            config = dict()
        self.config = config
        self.model = model
        self.output = output
        self.shared_dict = dict()

    @classmethod
    def set_export_mode(cls, model: Module, enabled: bool):
        _set_layer_export_mode(model, enabled)

    @classmethod
    def set_export_handler(cls, module: Module, name: str, shared_dict: dict):
        _set_layer_export_handler(cls, module)
        if hasattr(module, 'export_handler') and module.export_handler is not None:
            module.export_handler.layer_name = name
            module.export_handler.shared_dict = shared_dict
    
    def __enter__(self):
        

        for name, module in self.model.named_modules():
            self.set_export_handler(module, name, self.shared_dict)
        self.set_export_mode(self.model, enabled=True)

    def __exit__(self, *args, **kwargs):
        tensor_ids = []
        for v in self.shared_dict.values():
            if hasattr(v, '_data'):
                tensor_ids.append(id(v._data))
        
        def named_children(model, prefix = ''):
            for n,m in model.named_children():
                full_name = prefix + '.' + n if prefix != '' else n
                if isinstance(m, QuantLayerMixin):
                    continue
                elif isinstance(m, EqualizedModule):
                    continue
                elif len(list(m.children())) == 0:
                    for n_p, p in m.named_parameters():
                        param_name = full_name + '.' + n_p
                        self.shared_dict[param_name] = DefaultPrimitiveTensor(name=param_name, data=p)
                else:
                    named_children(m, prefix = full_name)
        named_children(self.model)
        # for n, m in self.model.named_modules():
        #     if len(list(m.children())) == 0:
        #         for n_p, p in m.named_parameters():
        #             param_name = n + '.' + n_p
        #             if id(p) in tensor_ids or param_name in self.shared_dict:
        #                 continue
        #             self.shared_dict[param_name] = DefaultPrimitiveTensor(name=param_name, data=p)

        self.set_export_mode(self.model, enabled=False)
        theta = Theta(self.shared_dict)
        print(self.shared_dict.keys())
        ds = Dataset(self.config, theta)
        self.output.append(ds)

    def export(self, model, *model_args, **model_kwargs):

        shared_dict = {}

        for name, module in model.named_modules():
            self.set_export_handler(module, name, shared_dict)
        self.set_export_mode(model, enabled=True)

        with torch.no_grad():
            model(*model_args, **model_kwargs)

        for n, m in model.named_modules():
            if isinstance(m, torch.nn.Module) and len(list(m.children())) == 0:
                for n_p, p in m.named_parameters():
                    param_name = n + '.' + n_p
                    param_eq_name = n + '.layer.' + n_p
                    if param_name in shared_dict or param_eq_name in shared_dict:
                        print(param_name, param_eq_name)
                        continue
                    shared_dict[param_name] = DefaultPrimitiveTensor(name=param_name, data=p)

        self.set_export_mode(model, enabled=False)

        theta = Theta(shared_dict)
        print(shared_dict.keys())
        ds = Dataset(self.config.to_dict(), theta)
        return ds
