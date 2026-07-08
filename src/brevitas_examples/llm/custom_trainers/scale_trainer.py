# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Custom trainer plugin for scale-factor optimization.

This plugin registers a custom Trainer and TrainingArguments into
Brevitas's ``TRAINER_REGISTRY`` so that it can be used via the LLM
entrypoint with ``--custom-trainer``.

Usage
-----
::

    python -m brevitas_examples.llm.main \\
        --model <model_name> \\
        --custom-trainer path/to/custom_trainers/scale_trainer.py:scale \\
        -- \\
        --max_steps 500 \\
        --per_device_train_batch_size 4 \\
        --gradient_accumulation_steps 4 \\
        --bf16 True \\
        --scale_learning_rate 1e-3

Design
------
* **Scale-factor optimization** — this plugin optimizes only the
  quantizers' scale factors, i.e. every model parameter whose name ends
  in ``value`` (the naming convention used by Brevitas learnable
  quantizer parameters such as ``ParameterScaling``). All other model
  parameters are left frozen.
* The scale factors are optimized with AdamW through the standard
  ``optimizer_scheduler_args`` mechanism.
* **Distillation loss** is available through ``GeneralizedTrainer`` and
  can be enabled with ``--use_distillation_loss True``.
* Specifying ``--custom-trainer`` automatically implies ``--fine-tune``.
"""

from dataclasses import dataclass
from typing import List
from dataclasses import field

from typing import Optional


import torch
import transformers

from brevitas_examples.llm.llm_quant.rotation_optimization import GeneralizedTrainer
from brevitas_examples.llm.llm_quant.trainer_utils import TRAINER_REGISTRY
from brevitas_examples.llm.llm_quant.trainer_utils import TrainingArguments


# ---------------------------------------------------------------------------
# Parameter selector for the optimizer config
# ---------------------------------------------------------------------------
def _select_scale_params(model: torch.nn.Module,
                         training_args: transformers.TrainingArguments) -> List[torch.nn.Parameter]:
    """Return the quantizers' scale factors (one parameter group).

    Scale factors are identified as all model parameters whose name ends
    in ``value`` — the convention used by Brevitas learnable quantizer
    parameters. Duplicates (shared parameters) are removed by identity.
    """
    seen: set = set()
    params: List[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if name.endswith("value") and id(param) not in seen:
            seen.add(id(param))
            params.append(param)
    return params


# ---------------------------------------------------------------------------
# Training arguments
# ---------------------------------------------------------------------------
@dataclass
class ScaleTrainingArguments(TrainingArguments):
    """Training arguments for the scale-only optimization flow.

    Expresses AdamW-on-scale-factors through the standard
    ``optimizer_scheduler_args`` mechanism: a single optimizer whose
    single parameter group holds the quantizers' scale factors.
    """
    learning_rate: Optional[float] = field(
        default=None,
        metadata={
            "help":
                "Learning rate for AdamW on the scale factors. "
                "Defaults to --learning_rate when unset."})

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.optimizer_scheduler_args is None:
            self.optimizer_scheduler_args = [{
                "optimizer_cls":
                    "AdamW",
                "param_setup": [{
                    "get_param_fn": _select_scale_params,
                    "optimizer_kwargs": {
                        "lr": self.learning_rate,},}],}]


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class ScaleTrainer(GeneralizedTrainer):
    """Trainer for scale-factor optimization.

    Uses :class:`ScaleTrainingArguments`, whose ``optimizer_scheduler_args``
    expresses AdamW on the quantizers' scale factors (selected via
    ``param_setup``). Inherits distillation-loss support from
    :class:`GeneralizedTrainer`.
    """
    training_args_cls = ScaleTrainingArguments


# ---------------------------------------------------------------------------
# Register the trainer under the name "scale". The optimizer setup (including
# the scale-factor selector) lives in ScaleTrainingArguments.
# ---------------------------------------------------------------------------
TRAINER_REGISTRY.register("scale")(ScaleTrainer)
