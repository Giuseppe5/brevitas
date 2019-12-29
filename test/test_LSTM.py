from brevitas.nn import QuantLSTMLayer, LSTMLayer
import torch.nn as nn
from LSTMcell import *
import torch
from brevitas.core.quant import QuantType
import time
if __name__ == '__main__':
    weight_config = {
        'weight_quant_type' : QuantType.INT
    }
    # weight_config['quant_type'] = 'QuantType.INT'
    activation_config = {
        'quant_type': QuantType.INT
    }
    state1 = torch.rand(2,500)
    state2 = torch.rand(2,500)
    state = (state1,state2)
    # B = LSTMCell(100,500)



    B = (LSTMLayer(LSTMCell, 100, 500))
    A = torch.jit.script(QuantLSTMLayer(100,500, activation_config=activation_config, weight_config=weight_config))

    input = torch.rand(1000,2, 100)


    quant = time.time()
    C = A(input,state)
    quant = time.time() -quant

    original = time.time()
    D = B(input, state)
    original = time.time() - original

    print(quant/1000)
    print(original/1000)
    print('END')
    # print(A.graph_for(input,state))
    # print(B.graph_for(input,state))