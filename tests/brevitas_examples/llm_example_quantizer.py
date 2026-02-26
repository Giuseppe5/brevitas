from dataclasses import dataclass

from brevitas.quant.scaled_int import Int8WeightPerTensorFloat
from brevitas.utils.python_utils import Registry
from brevitas_examples.common.generative.quantizers import BaseQuantizer
from brevitas_examples.common.generative.quantizers import QUANTIZERS_REGISTRY
from brevitas_examples.llm.llm_quant.rotation_optimization import GeneralizedTrainer
from brevitas_examples.llm.llm_quant.rotation_optimization import TRAINER_REGISTRY
from brevitas_examples.llm.llm_quant.rotation_optimization import TRAINING_ARGS_REGISTRY
from brevitas_examples.llm.llm_quant.rotation_optimization import TrainingArguments


@Registry.register(QUANTIZERS_REGISTRY, "example_int4_weight_quant")
class ExampleInt8WeightQuantizer(BaseQuantizer):
    weight_quant = Int8WeightPerTensorFloat.let(bit_width=4)


@dataclass
class ExampleTrainingArguments(TrainingArguments):
    pass


class ExampleTrainer(GeneralizedTrainer):
    pass


@Registry.register(TRAINING_ARGS_REGISTRY, "minimal_trainer")
class RegisteredExampleTrainingArguments(ExampleTrainingArguments):
    pass


@Registry.register(TRAINER_REGISTRY, "minimal_trainer")
class RegisteredExampleTrainer(ExampleTrainer):
    pass
