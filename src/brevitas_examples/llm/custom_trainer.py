# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Custom trainer plugin for full fine-tuning of a quantized LLM.

This plugin registers a custom Trainer, TrainingArguments, and optimizer
configuration into Brevitas's registries so that it can be used via the
LLM entrypoint with ``--custom-trainer``.

Usage
-----
::

    python -m brevitas_examples.llm.main \
        --model <model_name> \
        --custom-trainer path/to/full_finetuning_trainer.py:full_finetune \
        --nsamples-rot-calibration 1000 \
        -- \
        --max_steps 500 \
        --per_device_train_batch_size 4 \
        --gradient_accumulation_steps 4 \
        --bf16 True \
        --use_distillation_loss True \
        --gamma 0.5 \
        --temperature 2.0 \
        --adamw_lr 2e-5

The ``--`` separator passes all subsequent arguments to the HuggingFace
``TrainingArguments`` parser (extended with fields from
``FullFinetuningTrainingArguments``).

Design
------
* **No rotations required** — this plugin is designed for full fine-tuning
  of a quantized model without rotation optimization.  Specifying
  ``--custom-trainer`` automatically implies ``--fine-tune``.
* **Distillation loss** is enabled by default (``use_distillation_loss=True``)
  because it consistently outperforms plain cross-entropy when fine-tuning
  a quantized model.  The teacher signal comes from the same model with
  quantization disabled (handled by ``GeneralizedTrainer``).
* A single **AdamW** optimizer is configured for all trainable model
  weights.
* Training defaults to **bf16** mixed precision when available.
"""

from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import torch

from brevitas.utils.python_utils import Registry
from brevitas_examples.llm.llm_quant.rotation_optimization import GeneralizedTrainer
from brevitas_examples.llm.llm_quant.rotation_optimization import OPTIMIZER_CONFIG_REGISTRY
from brevitas_examples.llm.llm_quant.rotation_optimization import TRAINER_REGISTRY
from brevitas_examples.llm.llm_quant.rotation_optimization import TRAINING_ARGS_REGISTRY
from brevitas_examples.llm.llm_quant.rotation_optimization import TrainingArguments


# ---------------------------------------------------------------------------
# Training arguments
# ---------------------------------------------------------------------------
@dataclass
@Registry.register(TRAINING_ARGS_REGISTRY, "full_finetune")
class FullFinetuningTrainingArguments(TrainingArguments):
    """Extended training arguments for full fine-tuning.

    Inherits all fields from ``TrainingArguments`` (which itself extends
    ``transformers.TrainingArguments``).  Adds fine-tuning-specific
    defaults and new knobs.
    """

    # Override defaults from parent to sensible fine-tuning values.
    # Users can still override these via the CLI.
    bf16: bool = field(default=True, metadata={"help": "Train in bf16 mixed precision."})
    use_distillation_loss: bool = field(
        default=True,
        metadata={"help": "Use distillation loss (KL divergence against the unquantized teacher)."})
    gamma: float = field(
        default=0.,
        metadata={
            "help":
                "Balance between CE loss (gamma) and distillation loss (1 - gamma). "
                "0.5 gives equal weight to both."})
    temperature: float = field(
        default=1.0, metadata={"help": "Temperature for softmax in distillation loss."})
    topk: int = field(
        default=-1,
        metadata={
            "help":
                "If > 0, only use the top-k teacher logits for distillation. "
                "Helps reduce memory for large-vocabulary models."})

    # AdamW hyper-parameters
    adamw_lr: float = field(
        default=4e-5, metadata={"help": "Learning rate for AdamW."})
    adamw_weight_decay: float = field(
        default=0.01, metadata={"help": "Weight decay for AdamW."})
    adamw_betas: str = field(
        default="0.9,0.999",
        metadata={
            "help":
                "Comma-separated beta1,beta2 for AdamW. "
                "Parsed into a tuple at runtime."})

    def __post_init__(self):
        super().__post_init__()
        # Auto-populate optimizer_scheduler_args from the fine-tuning-specific
        # fields so that _build_optimizers_from_configs receives the right
        # per-optimizer kwargs without the user having to set them manually.
        if self.optimizer_scheduler_args is None:
            betas = tuple(float(x) for x in self.adamw_betas.split(","))
            self.optimizer_scheduler_args = [
                # Single group: AdamW for all trainable weights
                {
                    "optimizer_kwargs": {
                        "lr": self.adamw_lr,
                        "weight_decay": self.adamw_weight_decay,
                        "betas": betas,
                    },
                },
            ]


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
@Registry.register(TRAINER_REGISTRY, "full_finetune")
class FullFinetuningTrainer(GeneralizedTrainer):
    """Trainer for full fine-tuning of a quantized model.

    Inherits distillation loss support from ``GeneralizedTrainer``.
    No additional logic is required; the custom optimizer configuration
    below handles the optimizer setup.
    """
    pass


# ---------------------------------------------------------------------------
# Optimizer configuration
# ---------------------------------------------------------------------------
def _get_all_params(model: torch.nn.Module, training_args: FullFinetuningTrainingArguments):
    """Return all model parameters for full fine-tuning."""
    return list(model.parameters())


def _build_optimizer_configs():
    """Build and return the list of optimizer configuration dicts.

    A single optimizer group with AdamW for all model weights.
    """
    return [
        {
            "params": _get_all_params,
            "optimizer_class": torch.optim.AdamW,
            "scheduler_class": None,
        },
    ]


Registry.register(OPTIMIZER_CONFIG_REGISTRY, "full_finetune")(_build_optimizer_configs)
