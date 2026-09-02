# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import inspect

import torch
import torch.nn.functional as F
from transformers.integrations.executorch import TorchExportableModuleForDecoderOnlyLM
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from brevitas.graph import TorchFunctionalToModule
from brevitas.nn import ScaledDotProductAttention
from brevitas.utils.logging import setup_logger

logging = setup_logger(__name__)


def replace_sdpa_with_quantizable_layers(model, is_fx=True, eager_quant_sdpa_class=None):
    if is_fx:
        fn_to_module_map = ((F.scaled_dot_product_attention, ScaledDotProductAttention),)
        model = TorchFunctionalToModule(fn_to_module_map=fn_to_module_map).apply(model)
    else:
        # We rely on the following:
        # - Attention functions accepts the current module as input
        # - We can add a new entry in the dict of supported attention functions
        # - Attention Modules' name end with `Attention`. The user can also override this

        from brevitas_examples.llm.llm_quant.mha_layers import quant_sdpa_attention_forward
        ALL_ATTENTION_FUNCTIONS['quant_sdpa'] = quant_sdpa_attention_forward
        model.config._attn_implementation = 'quant_sdpa'
        for n, m in model.named_modules():
            if eager_quant_sdpa_class == 'auto':
                if type(m).__name__.lower().endswith('attention'):
                    quant_block_type = type(m)
                    break
            else:
                if type(m).__name__.lower() == eager_quant_sdpa_class.lower():
                    quant_block_type = type(m)
                    break
        logging.info(f"Attention module is {quant_block_type}")
        for m in model.modules():
            if isinstance(m, quant_block_type):
                m.attn = ScaledDotProductAttention()

    return model


@torch.no_grad()
def add_zero_bias_to_linear(model: torch.nn.Module) -> torch.nn.Module:
    for name, module in model.named_modules():
        if type(module) == torch.nn.Linear:
            if module.bias is None:
                module.register_parameter(
                    "bias",
                    torch.nn.Parameter(
                        torch.zeros((module.weight.shape[0],),
                                    device=module.weight.device,
                                    dtype=module.weight.dtype)),
                )
    return model


class make_dynamo_compatible:

    def __init__(self, model):
        self.model = model
        self.model_config = model.config
        self.text_config = (
            model.config.get_text_config()
            if hasattr(model.config, 'get_text_config') else model.config)
        self.model_use_cache = getattr(self.text_config, 'use_cache', None)
        self.generation_config = model.generation_config
        self.model_cache_implementation = getattr(
            self.generation_config, 'cache_implementation', None)
        self.model_generation_use_cache = getattr(self.generation_config, 'use_cache', None)

    def _restore_cache_config(self):
        self.text_config.use_cache = self.model_use_cache
        self.generation_config.cache_implementation = self.model_cache_implementation
        self.generation_config.use_cache = self.model_generation_use_cache

    def __enter__(self):
        # The ExecuTorch wrapper requires caching while it patches the model. Caching is
        # disabled again on the unwrapped model before Dynamo tracing.
        self.text_config.use_cache = True
        self.generation_config.cache_implementation = "static"
        self.generation_config.use_cache = True
        # Because getattr does not fall back to default with `config` class, we need to manually fill
        # `head_dim` if it is None
        # https://github.com/huggingface/transformers/blob/47b0e478f324b54f177ea7998a0791870fdd0324/src/transformers/integrations/executorch.py#L538
        if not hasattr(self.model.config, 'head_dim') or self.model.config.head_dim is None:
            self.model.config.head_dim = self.model.config.hidden_size // self.model.config.num_attention_heads
        parameters = inspect.signature(TorchExportableModuleForDecoderOnlyLM.__init__).parameters
        has_batch_size = 'batch_size' in parameters
        has_max_batch_size = 'max_batch_size' in parameters
        if has_batch_size and has_max_batch_size:
            self._restore_cache_config()
            raise RuntimeError(
                "Unsupported Transformers ExecuTorch API: expected exactly one of "
                "'batch_size' or 'max_batch_size'.")

        wrapper_kwargs = {'max_cache_len': 1}
        if has_batch_size:
            wrapper_kwargs['batch_size'] = 1
        elif has_max_batch_size:
            wrapper_kwargs['max_batch_size'] = 1
        else:
            self._restore_cache_config()
            raise RuntimeError(
                "Unsupported Transformers ExecuTorch API: missing both 'batch_size' "
                "and 'max_batch_size'.")

        # Wrapping applies the Dynamo compatibility patches; the cache itself is not
        # used after the model is immediately unwrapped.
        try:
            self.model = TorchExportableModuleForDecoderOnlyLM(
                self.model, **wrapper_kwargs).model.model
        except Exception:
            self._restore_cache_config()
            raise
        # Caching should be disabled to make it work with dynamo
        # The other alternative is to use static_cache
        self.model.config.use_cache = False
        return self

    def __exit__(self, *args, **kwargs):
        # Restore configuration. Always restore cache_implementation: HF's default is None,
        # so guarding on `is not None` would leak "static" into downstream generation
        # (which then triggers torch.compile + StaticCache recompiles in lighteval).
        self.model.config = self.model_config
        self._restore_cache_config()
