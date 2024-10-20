from inspect import signature

import torch

from brevitas.nn.quant_mha import QuantMultiheadAttention
import fast_hadamard_transform

INPUT_NAMES = ['input', 'inp', 'query', 'x', 'hidden_states']


class EqualizedModule(torch.nn.Module):

    def __init__(self, scale_module, layer) -> None:
        super().__init__()
        self.scale = scale_module
        self.layer = layer

    def forward(self, *args, **kwargs):
        # Convert args + kwargs + defaults into kwargs
        bound_arguments = signature(self.layer.forward).bind(*args, **kwargs)
        bound_arguments.apply_defaults()
        kwargs = bound_arguments.arguments

        possible_input_kwargs = INPUT_NAMES
        input_kwarg = [x for x in kwargs.keys() if x in possible_input_kwargs][0]
        x = kwargs[input_kwarg]
        out = x
        if 'key' in kwargs:
            if kwargs['key'].data_ptr() != out.data_ptr():
                raise ValueError(
                    "Cross MHA is not supported for activation equalization."
                    "Replace kwargs with positional args to avoid this exception.")
        out = self.scale(out)

        kwargs[input_kwarg] = out
        # QuantMultiheadAttention is not a subclass of MultiheadAttention
        # We need to preserve the correctness of the forward even after
        # quantization has been applied
        if isinstance(self.layer, (torch.nn.MultiheadAttention, QuantMultiheadAttention)):
            kwargs['key'] = out
            kwargs['value'] = out
        # We convert everything to args so that hooks can work correctly
        out = self.layer(*kwargs.values())
        return out

class RotatedModule(torch.nn.Module):

    def __init__(self, had_mat, k, layer) -> None:
        super().__init__()
        self.had_mat = torch.nn.Parameter(had_mat).cpu()
        self.layer = layer
        self.k = k

    def forward(self, inp, **kwargs):

        shape = inp.shape
        n = inp.shape[-1]
        if self.k == 1:
            inp = fast_hadamard_transform.hadamard_transform(inp.contiguous(), 1.0/torch.tensor(n).sqrt()) 
            o = self.layer(inp)

        # if transpose:
        #     hadK = hadK.T.contiguous()
        inp = inp.view(*inp.shape[:-1], self.k, n // self.k)
        inp = fast_hadamard_transform.hadamard_transform(inp.contiguous(), 1.0/torch.tensor(n).sqrt())
        inp = self.had_mat.to(inp.device).to(inp.dtype) @ inp
        inp = inp.reshape(shape)
        o = self.layer(inp)

        return o

