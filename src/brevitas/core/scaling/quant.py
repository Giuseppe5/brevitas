import torch.nn as nn

class QuantScaling(nn.Module):
    def __init__(self, scale_rescaling_int_quant) -> None:
        super().__init__()
        self.scale_rescaling_int_quant = scale_rescaling_int_quant

    def forward(self, x):
        _,scale, *_ = self.scale_rescaling_int_quant(x)
        # print(x.shape, scale.shape)
        print(x, scale)
        return x * scale