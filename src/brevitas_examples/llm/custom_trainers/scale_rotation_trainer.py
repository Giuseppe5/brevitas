# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Custom trainer plugin for joint scale-factor and rotation optimization.

This plugin registers a custom Trainer and TrainingArguments into
Brevitas's ``TRAINER_REGISTRY`` so that it can be used via the LLM
entrypoint with ``--custom-trainer``.

Usage
-----
::

    python -m brevitas_examples.llm.main \\
        --model <model_name> \\
        --rotation fx --optimize-rotations \\
        --custom-trainer path/to/custom_trainers/scale_rotation_trainer.py:scale_rotation \\
        -- \\
        --max_steps 500 \\
        --per_device_train_batch_size 4 \\
        --gradient_accumulation_steps 4 \\
        --bf16 True \\
        --rotation_learning_rate 1e-4 \\
        --scale_learning_rate 1e-3

Design
------
* **Joint scale + rotation optimization** — this plugin optimizes both
  the quantizers' scale factors and the trainable rotation matrices:

  * **Rotation matrices** (selected via
    :func:`extract_trainable_rotation_matrices`) are optimized with
    ``CaileySGD`` on the Stiefel manifold, mirroring the default
    :class:`RotationTrainer`.
  * **Scale factors** — every model parameter whose name ends in
    ``value`` (the naming convention used by Brevitas learnable
    quantizer parameters) — are optimized with AdamW.

  The two optimizers are combined into a ``MultiOptimizer`` /
  ``MultiScheduler`` automatically by the optimizer-building helpers.
* **Distillation loss** is available through ``GeneralizedTrainer`` and
  can be enabled with ``--use_distillation_loss True``.
* Specifying ``--custom-trainer`` automatically implies ``--fine-tune``.
"""

from dataclasses import dataclass
from dataclasses import field
from typing import List
from typing import Optional

import torch
import transformers

from brevitas.utils.parametrization_utils import extract_trainable_rotation_matrices
from brevitas_examples.llm.llm_quant.rotation_optimization import GeneralizedTrainer
from brevitas_examples.llm.llm_quant.trainer_utils import TRAINER_REGISTRY
from brevitas_examples.llm.llm_quant.trainer_utils import TrainingArguments


# ---------------------------------------------------------------------------
# Parameter selectors for the optimizer configs
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


def _select_rotation_params(
        model: torch.nn.Module,
        training_args: transformers.TrainingArguments) -> List[torch.nn.Parameter]:
    """Return the model's trainable rotation matrices (one parameter group)."""
    return extract_trainable_rotation_matrices(model)


# ---------------------------------------------------------------------------
# Training arguments
# ---------------------------------------------------------------------------
@dataclass
class ScaleRotationTrainingArguments(TrainingArguments):
    """Training arguments for the joint scale + rotation optimization flow.

    Builds two optimizers through the standard ``optimizer_scheduler_args``
    mechanism: ``CaileySGD`` on the trainable rotation matrices and AdamW
    on the quantizers' scale factors.
    """

    rotation_learning_rate: Optional[float] = field(
        default=None,
        metadata={
            "help":
                "Learning rate for CaileySGD on the rotation matrices. "
                "Defaults to --learning_rate when unset."})
    scale_learning_rate: Optional[float] = field(
        default=None,
        metadata={
            "help":
                "Learning rate for AdamW on the scale factors. "
                "Defaults to --learning_rate when unset."})

    def __post_init__(self) -> None:
        super().__post_init__()
        rotation_lr = (
            self.rotation_learning_rate
            if self.rotation_learning_rate is not None else self.learning_rate)
        scale_lr = (
            self.scale_learning_rate
            if self.scale_learning_rate is not None else self.learning_rate)
        if self.optimizer_scheduler_args is None:
            self.optimizer_scheduler_args = [
                # Optimizer 0: CaileySGD for rotation matrices
                {
                    "optimizer_cls":
                        "CaileySGD",
                    "param_setup": [{
                        "get_param_fn": _select_rotation_params,
                        "optimizer_kwargs": {
                            "lr": rotation_lr,
                            "stiefel": True,
                            "dtype": self.optimizer_dtype,},}],},
                # Optimizer 1: AdamW for scale factors
                {
                    "optimizer_cls":
                        "AdamW",
                    "param_setup": [{
                        "get_param_fn": _select_scale_params,
                        "optimizer_kwargs": {
                            "lr": scale_lr,},}],},]


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class ScaleRotationTrainer(GeneralizedTrainer):
    """Trainer for joint scale-factor and rotation optimization.

    Uses :class:`ScaleRotationTrainingArguments`, whose
    ``optimizer_scheduler_args`` expresses CaileySGD on the rotation
    matrices and AdamW on the scale factors (both selected via
    ``param_setup``). Inherits distillation-loss support from
    :class:`GeneralizedTrainer`.
    """
    training_args_cls = ScaleRotationTrainingArguments


# ---------------------------------------------------------------------------
# Register the trainer under the name "scale_rotation". The optimizer setup
# (including the per-optimizer param selectors) lives in
# ScaleRotationTrainingArguments.
# ---------------------------------------------------------------------------
TRAINER_REGISTRY.register("scale_rotation")(ScaleRotationTrainer)
