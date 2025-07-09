import argparse
import array
from contextlib import nullcontext
from copy import deepcopy
from datetime import timedelta
import functools
import pprint
import sys
from typing import Dict
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers.generation.configuration_utils import GenerationConfig
import yaml

from brevitas.core.function_wrapper import FloorSte
from brevitas.export.inference.manager import quant_inference_mode
from brevitas.graph.quantize import layerwise_quantize
import brevitas.nn as qnn
from brevitas.quant.experimental.float_quant_fnuz import Fp8e4m3FNUZActPerTensorFloat
from brevitas.quant.experimental.float_quant_fnuz import Fp8e4m3FNUZWeightPerTensorFloat
from brevitas.quant.experimental.mx_quant_ocp import MXFloat8e4m3Act
from brevitas.quant.experimental.mx_quant_ocp import MXFloat8e4m3WeightMSE
from brevitas.utils.python_utils import hooked_on_a_function
from brevitas_examples.common.accelerate_utils.accelerate import offload_model
from brevitas_examples.common.accelerate_utils.accelerate import remove_hooks
from brevitas_examples.common.accelerate_utils.accelerate import update_internal_dict
from brevitas_examples.common.generative.quantize import generate_quant_maps
from brevitas_examples.common.generative.quantize import generate_quantizers
from brevitas_examples.llm.gguf_export.export import save_quantized_as_gguf
from brevitas_examples.llm.llm_args import create_llm_args_parser
from brevitas_examples.llm.llm_args import validate
from brevitas_examples.llm.llm_quant.awq.pre_quant import apply_awq
from brevitas_examples.llm.llm_quant.bias_corr import apply_bias_correction
from brevitas_examples.llm.llm_quant.calibrate import apply_calibration
from brevitas_examples.llm.llm_quant.data_utils import get_dataset_for_model
from brevitas_examples.llm.llm_quant.equalize import apply_act_equalization
from brevitas_examples.llm.llm_quant.equalize import apply_weight_equalization
from brevitas_examples.llm.llm_quant.eval import compute_perplexity
from brevitas_examples.llm.llm_quant.export import BlockQuantProxyLevelManager
from brevitas_examples.llm.llm_quant.export import brevitas_proxy_export_mode
from brevitas_examples.llm.llm_quant.gpxq import apply_gpfq
from brevitas_examples.llm.llm_quant.gpxq import apply_gptq
from brevitas_examples.llm.llm_quant.gpxq import apply_magr
from brevitas_examples.llm.llm_quant.gpxq import apply_qronos
from brevitas_examples.llm.llm_quant.learned_round_utils import apply_learned_round
from brevitas_examples.llm.llm_quant.ln_affine_merge import apply_layernorm_affine_merge
from brevitas_examples.llm.llm_quant.ln_affine_merge import apply_layernorm_to_rmsnorm
from brevitas_examples.llm.llm_quant.ln_affine_merge import replace_rmsnorm_with_torch
from brevitas_examples.llm.llm_quant.prepare_for_quantize import replace_mlperf_attn
from brevitas_examples.llm.llm_quant.prepare_for_quantize import \
    replace_sdpa_with_quantizable_layers
from brevitas_examples.llm.llm_quant.rotation_optimization import apply_rotation_optimization
from brevitas_examples.llm.llm_quant.rotation_optimization import parse_rotation_optimization_args
from brevitas_examples.llm.llm_quant.run_utils import fix_rewriter
from brevitas_examples.llm.llm_quant.svd_quant import apply_svd_quant

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str)
parser.add_argument('--gptq', action='store_true')
parser.add_argument('--magr', action='store_true')
parser.add_argument('--magr_alpha', type=float)
torch.set_float32_matmul_precision('high')

calib_path = '/data/mlperf_llama/data/mlperf_llama3.1_405b_calibration_dataset_512_processed_fp16_eval.pkl'
eval_data = '/data/mlperf_llama/data/mlperf_llama3.1_405b_dataset_8313_processed_fp16_eval.pkl'


class Dataset:

    def __init__(
            self,
            model_name=None,
            total_sample_count=8313,
            perf_count_override=None,
            dataset_path=None,
            dtype="bfloat16"):
        self.model_name = model_name or f"Meta-Llama-3.1-405B-Instruct{'-FP8' if dtype == 'float8' else ''}"
        self.dataset_path = dataset_path

        # self.total_sample_count = total_sample_count
        self.load_processed_dataset()

        self.total_sample_count = min(len(self.input_ids), total_sample_count)
        self.perf_count = perf_count_override or self.total_sample_count

    def load_processed_dataset(self):
        import pandas as pd

        self.processed_data = pd.read_pickle(self.dataset_path)

        self.input = self.processed_data.input.tolist()
        self.input_ids = self.processed_data.tok_input.tolist()
        self.input_lens = self.processed_data.tok_input_len.tolist()

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return torch.tensor(self.input_ids[idx])

    @staticmethod
    def postProcess(
        out_tokens,
        query_id_list=None,
        sample_index_list=None,
    ):
        """Postprocesses output prediction"""

        # TODO: Create response object in postProcess(?)

        """
        preds = []
        for i in range(out_tokens.shape[0]):
            #pred = out_tokens[i].reshape(-1).cpu().numpy() # Slice up to original input length as below?

            input_len = input_seq_lens[i] if input_seq_lens else 0
            pred = out_tokens[i, input_len:].reshape(-1).cpu().numpy()
            preds.append(pred)
        """
        # Everything is padded to max_len (1024), so prune the input and parse
        # to numpy
        output_seq = out_tokens

        return [np.asarray(out, dtype=np.int32) for out in output_seq]


class CustomDataset(torch.utils.data.Dataset):

    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return len(next(iter(self.encodings.values())))


def get_mlperf_data(
        data_path: str,
        tokenizer: AutoTokenizer = None,
        batch_size: int = 1,
        num_calib_data: int = 128,
        seqlen: int = 1024,
        device: str = 'cpu') -> DataLoader[torch.Tensor]:

    import pickle

    print("mlperf calibration data path: ", data_path)

    with open(data_path, 'rb') as fh:
        mlperf_df = pickle.load(fh)

    system_prompt_instruction = mlperf_df['input'].tolist()[:num_calib_data]

    batch_encoded = tokenizer.batch_encode_plus(
        system_prompt_instruction,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=seqlen,
    )
    if device:
        batch_encoded = batch_encoded.to(device)

    tokenized_dataset = CustomDataset({"input_ids": batch_encoded["input_ids"]})

    calib_dataloader: DataLoader[List[Dict[str, torch.Tensor]]] = DataLoader(
        tokenized_dataset, batch_size=batch_size, shuffle=False, drop_last=True)  # type: ignore

    return calib_dataloader


class MXFP4Weight(MXFloat8e4m3WeightMSE):
    restrict_value_float_to_int_impl = FloorSte


class MXFP4Act(MXFloat8e4m3Act):
    restrict_value_float_to_int_impl = FloorSte


class FP8Weight(Fp8e4m3FNUZWeightPerTensorFloat):
    pass


class FP8Act(Fp8e4m3FNUZActPerTensorFloat):
    pass


def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def main(args):
    set_seed(0)

    kwargs = {"torch_dtype": torch.float16}

    print("Model loading...")
    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
    dtype = next(model.parameters()).dtype
    print("Model loaded.")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    if tokenizer.pad_token != "<unk>":
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    calibration_loader = get_mlperf_data(calib_path, tokenizer, 1, 128, 1024, 'cuda')

    validation_loader = get_dataset_for_model(
        args.model,
        bos_preprocessing=None,
        dataset_name='wikitext2',
        tokenizer=tokenizer,
        nsamples=128,
        seqlen=2048,
        split="validation",
        seed=0,
        require_fx=False,
        device=None)

    validation_dataset = Dataset(dataset_path=eval_data)
    validation_dataloader = DataLoader(
        validation_dataset, shuffle=False, batch_size=1, drop_last=True)

    device = next(iter(model.parameters())).device
    model = offload_model(model)
    # print("Data loaded.")
    # # quant_ppl = compute_perplexity(
    # #     model, validation_loader, context_length=2048// 2, tokenizer=tokenizer)
    remove_hooks(model)
    # print(f"Quantized perplexity: {quant_ppl:.3f}")
    if args.magr:
        print("Applying MagR...")
        model = offload_model(model)
        apply_magr(model, calibration_loader, create_weight_orig=False, alpha=args.magr_alpha)
        remove_hooks(model)
        print(f"MagR applied.")

    # model = replace_mlperf_attn(model)

    # Attn Quantization
    q_scaled_quant = FP8Act
    k_transposed_quant = FP8Act
    q_scaled_quant = q_scaled_quant.let(**{'group_dim': -1, 'group_size': 32})
    k_transposed_quant = k_transposed_quant.let(**{'group_dim': -2, 'group_size': 32})
    v_quant = k_transposed_quant

    quant_sdpa_kwargs = {
        'softmax_input_quant': None,
        'attn_output_weights_quant': None,
        'q_scaled_quant': q_scaled_quant,
        'k_transposed_quant': k_transposed_quant,
        'v_quant': v_quant,
        'attn_output_quant': None,
        'dtype': dtype,
        'device': device}

    mxfp4_layer_types = []

    quant_linear_kwargs = {
        'input_quant':
            lambda name,
            module: MXFP4Act if any([pattern in name for pattern in mxfp4_layer_types]) else FP8Act,
        'weight_quant':
            lambda name,
            module: MXFP4Weight
            if any([pattern in name for pattern in mxfp4_layer_types]) else FP8Weight,}

    layer_map = {
        nn.Linear: (qnn.QuantLinear, quant_linear_kwargs),
        qnn.ScaledDotProductAttention: (qnn.QuantScaledDotProductAttention, quant_sdpa_kwargs)}
    name_blacklist = []

    model = layerwise_quantize(
        model=model, compute_layer_map=layer_map, name_blacklist=name_blacklist)
    # Just to be sure
    model.eval()
    model = model.to(dtype)

    # model = offload_model(model)
    model.cuda()

    # We initialize weights scale factor
    with torch.no_grad():
        first_elem = next(iter(calibration_loader))
        model(**first_elem)

    print("Apply act calibration...")
    apply_calibration(model, calibration_loader)
    print("Act calibration applied.")

    if args.gptq and not args.load_checkpoint:
        print("Applying GPTQ...")
        apply_gptq(
            model,
            calibration_loader,
            act_order=True,
            use_quant_activations=True,
            create_weight_orig=False,
            block_name='model.layers')
        print("GPTQ applied.")

    # quant_ppl = compute_perplexity(
    #     model, validation_loader, context_length=2048// 2, tokenizer=tokenizer)

    # print(f"Quantized perplexity: {quant_ppl:.3f}")
    validation_data = []
    generate_configs = {
        'temperature': 1,
        'top_k': 1,
        'top_p': 1,
        'seed': 42,
        'max_new_tokens': 100,
        'min_new_tokens': 2,
        'do_sample': True,
        'bos_token_id': tokenizer.bos_token_id,
        'eos_token_id': tokenizer.eos_token_id,  # 'decoder_start_token_id': None,
        'pad_token_id': tokenizer.eos_token_id,}
    config = GenerationConfig(**generate_configs)
    import time

    with torch.no_grad():
        # for m in model.modules():
        #     if isinstance(m, torch.nn.Linear):
        #         m = torch.compile(m, dynamic=True, fullgraph=True)
        #         print(m.forward)

        # for m in model.modules():
        #     if isinstance(m, torch.nn.Linear):
        #         print(m.forward)
        #         break

        with quant_inference_mode(model, compile=True, enabled=True):

            for ii, data in enumerate(validation_dataloader):
                tic = time.time()
                print(data.shape)
                outputs = model.generate(
                    data.to('cuda'), generation_config=config, cache_implementation="static")
                torch.cuda.synchronize()
                print(time.time() - tic)
                pred_output_tokens = []
                for i, output in enumerate(outputs):
                    input_token_length = data[i].shape[0]
                    pred_output_tokens.append(output[input_token_length:].tolist())
                processed_output = Dataset.postProcess(pred_output_tokens)
                for el in processed_output:
                    n_tokens = el.shape[0]
                    response_array = array.array("B", el.tobytes())
                    bi = response_array.buffer_info()
                    print(n_tokens)
                    # print(response_array)
                    # print(response_array.tobytes().hex())
                    # print(bi)
                if ii == 5:
                    break


if __name__ == '__main__':
    # from aiter.ops.gemm_op_a8w8 import gemm_a8w8
    # import aiter.utility.dtypes as dtype
    # torch.cuda.set_device("cuda:0")
    # a = torch.randn(4096, 4096, device='cuda')
    # b = torch.randn(4096, 4096, device='cuda').t()
    # a_fp8 = a.to(torch.float8_e4m3fnuz)
    # b_fp8 = b.to(torch.float8_e4m3fnuz)
    # out = torch.empty(4096, 4096, device='cuda', dtype=torch.bfloat16)
    # scale = torch.tensor(1., device='cuda', dtype=torch.float32)

    # @torch.compile
    # def mm(a,b,out,scale):
    #     gemm_a8w8(a, b, scale, scale, out, None)
    #     return out
    # mm(a_fp8,b_fp8, out, scale)

    # import time
    # s = time.time()
    # for _ in range(1000):
    #     mm(a_fp8,b_fp8, out, scale)
    # torch.cuda.synchronize()
    # print(time.time() - s)
    # @torch.compile
    # def mm(a,b):
    #     return torch.matmul(a,b)
    # mm(a,b)
    # import time
    # s = time.time()
    # for _ in range(1000):
    #     mm(a,b)
    # torch.cuda.synchronize()
    # print(time.time() - s)

    # @torch.compile
    # def mm(a,b,scale):
    #     return torch._scaled_mm(a,b, scale, scale)
    # mm(a_fp8,b_fp8, scale)

    # s = time.time()
    # for _ in range(1000):
    #     mm(a_fp8,b_fp8, scale)
    # torch.cuda.synchronize()
    # print(time.time() - s)

    args, _ = parser.parse_known_args()
    main(args)
