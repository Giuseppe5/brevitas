from brevitas.nn import QuantLSTMLayer, BidirLSTMLayer
import torch.nn as nn
# from LSTMcell import *
import torch
from brevitas.core.quant import QuantType
import time
from collections import namedtuple, OrderedDict

SEQ = 1000
INPUT_SIZE = 5
BATCH = 5
HIDDEN = 100
SEED = 123456
LSTMState = namedtuple('LSTMState', ['hx', 'cx'])
torch.manual_seed(SEED)


class TestLSTMQuant:
    def test_naiveLSTM(self):

        weight_config = {
            'weight_quant_type' : 'QuantType.FP'
        }

        activation_config = {
            'quant_type': 'QuantType.FP'
        }

        input = torch.randn(SEQ, BATCH, INPUT_SIZE)
        states = LSTMState(torch.randn(BATCH, HIDDEN),
                           torch.randn(BATCH, HIDDEN))

        q_lstm = torch.jit.script(QuantLSTMLayer(INPUT_SIZE, HIDDEN, activation_config=activation_config,
                                                 weight_config=weight_config, layer_norm='decompose'))
        q_lstm.eval()

        # Control
        lstm = torch.nn.LSTM(INPUT_SIZE, HIDDEN, 1)
        lstm_state = LSTMState(states.hx.unsqueeze(0), states.cx.unsqueeze(0))
        q_lstm.load_state_dict_new(lstm.state_dict())
        lstm_out, lstm_out_state = lstm(input, lstm_state)
        start = time.time()
        out, custom_state = q_lstm(input, states)
        end = time.time()-start
        print(end)

        assert (out - lstm_out).abs().max() < 1e-5
        assert (custom_state[0] - lstm_out_state[0]).abs().max() < 1e-5
        assert (custom_state[1] - lstm_out_state[1]).abs().max() < 1e-5
#
#
#
#     # def test_BILSTM(self):
#     #
#     #     def double_flatten_states(states):
#     #         # XXX: Can probably write this in a nicer way
#     #         states = flatten_states([flatten_states(inner) for inner in states])
#     #         return [hidden.view([-1] + list(hidden.shape[2:])) for hidden in states]
#     #
#     #     def flatten_states(states):
#     #         states = list(zip(*states))
#     #         assert len(states) == 2
#     #         return [torch.stack(state) for state in states]
#     #
#     #
#     #     weight_config = {
#     #         'weight_quant_type' : 'QuantType.FP'
#     #     }
#     #
#     #     activation_config = {
#     #         'quant_type': 'QuantType.FP'
#     #     }
#     #
#     #     input = torch.randn(SEQ, BATCH, INPUT_SIZE)
#     #     states = [LSTMState(torch.randn(BATCH, HIDDEN),
#     #                        torch.randn(BATCH, HIDDEN))
#     #               for _ in range(2)]
#     #     q_lstm = torch.jit.script(BidirLSTMLayer(INPUT_SIZE, HIDDEN,
#     #                                              activation_config=activation_config, weight_config=weight_config))
#     #
#     #     # Control
#     #     lstm = torch.nn.LSTM(INPUT_SIZE, HIDDEN, 1, bidirectional=True)
#     #     lstm_state = flatten_states(states)
#     #
#     #     for name_c, custom_param in q_lstm.named_parameters():
#     #         splitted_name = name_c.split('.')
#     #         name_c = ''.join(splitted_name[2:])
#     #
#     #
#     #         bias_name = splitted_name[2]
#     #         if bias_name == 'layernorm_h':
#     #             name_c = 'bias_ih'
#     #         elif bias_name == 'layernorm_i':
#     #             name_c = 'bias_hh'
#     #         name_c = name_c + '_l0'
#     #
#     #         if True: # check for bidirectionality here
#     #             if splitted_name[1] == '1':
#     #                 name_c = name_c + '_reverse'
#     #
#     #         for name_l, lstm_param in lstm.named_parameters():
#     #             if name_l == name_c:
#     #                 with torch.no_grad():
#     #                     lstm_param.copy_(custom_param)
#     #                     break
#     #     lstm_out, lstm_out_state = lstm(input, lstm_state)
#     #     out, custom_state = q_lstm(input, states)
#     #
#     #     states = (torch.stack([custom_state[0][0]] + [custom_state[1][0]]),
#     #               torch.stack([custom_state[0][1]] + [custom_state[1][1]]))
#     #
#     #     assert (out - lstm_out).abs().max() < 1e-5
#     #     assert (states[0] - lstm_out_state[0]).abs().max() < 1e-5
#     #     assert (states[1] - lstm_out_state[1]).abs().max() < 1e-5