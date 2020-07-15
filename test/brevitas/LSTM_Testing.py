import torch.nn as nn
import time
import brevitas.nn as qnn
import torch
from highlevel_torch import script_lstm
from collections import namedtuple
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--jit", action="store_true")
parser.add_argument("--type", type=str, default="basic")

LSTMState = namedtuple('LSTMState', ['hx', 'cx'])

BATCH_SIZE = 128
FEAT_IN = 64
SEQ_LENGTH = 512

HIDDEN_SIZE = 512


def flatten_states(states):
    states = list(zip(*states))
    assert len(states) == 2
    return [torch.stack(state) for state in states]


def test_basic():
    lstm = nn.LSTM(input_size=FEAT_IN, hidden_size=HIDDEN_SIZE)
    lstm.cuda()
    inp = torch.rand(SEQ_LENGTH, BATCH_SIZE, FEAT_IN, device=next(lstm.parameters()).device)
    state = [LSTMState(torch.randn(BATCH_SIZE, HIDDEN_SIZE, device=next(lstm.parameters()).device),
                       torch.randn(BATCH_SIZE, HIDDEN_SIZE, device=next(lstm.parameters()).device))]
    lstm_state = flatten_states(state)
    start = time.process_time()
    for _ in range(1000):
        _ = lstm(inp, lstm_state)
    end = time.process_time() - start
    end = end/1000.0
    print("Default LSTM took {} seconds".format(end))


def test_lstm_highlevel(jit):
    lstm = script_lstm(FEAT_IN, HIDDEN_SIZE, num_layers=1)
    lstm.cuda()
    inp = torch.rand(SEQ_LENGTH, BATCH_SIZE, FEAT_IN, device=next(lstm.parameters()).device)
    state = [LSTMState(torch.randn(BATCH_SIZE, HIDDEN_SIZE, device=next(lstm.parameters()).device),
                       torch.randn(BATCH_SIZE, HIDDEN_SIZE, device=next(lstm.parameters()).device))]
    start = time.process_time()
    for _ in range(1000):
        _ = lstm(inp, state)
    end = time.process_time() - start
    end = end/1000.0
    print("High Level LSTM, JIT {}, took {} seconds".format(jit, end))



def test_lstm_brevitas(jit):
    weight_config = {
        'weight_quant_type': 'INT'
    }
    activation_config = {
        'quant_type': 'INT'
    }
    hidden_activation_config = {
        'quant_type': 'INT',
        'min_val': -1e32,
        'max_val': 1e32
    }
    lstm = qnn.QuantLSTMLayer(input_size=FEAT_IN, hidden_size=HIDDEN_SIZE, weight_config=weight_config,
                              activation_config=activation_config,
                              norm_scale_hidden_config=hidden_activation_config,
                              norm_scale_out_config=hidden_activation_config)
    lstm.cuda()
    inp = torch.rand(SEQ_LENGTH, BATCH_SIZE, FEAT_IN, device=next(lstm.parameters()).device)
    state = LSTMState(torch.randn(BATCH_SIZE, HIDDEN_SIZE, device=next(lstm.parameters()).device),
                       torch.randn(BATCH_SIZE, HIDDEN_SIZE, device=next(lstm.parameters()).device))
    start = time.process_time()
    for _ in range(1000):
        _ = lstm(inp, state)
    end = time.process_time() - start
    end = end/1000.0
    print("Brevitas LSTM, JIT {}, took {} seconds".format(jit, end))


def test_lstm_weight_brevitas(jit):
    weight_config = {
        'weight_quant_type': 'INT'
    }
    activation_config = {
        'quant_type': 'FP'
    }
    hidden_activation_config = {
        'quant_type': 'FP',
        'min_val': -1e32,
        'max_val': 1e32
    }
    lstm = qnn.QuantLSTMLayer(input_size=FEAT_IN, hidden_size=HIDDEN_SIZE, weight_config=weight_config,
                              activation_config=activation_config,
                              norm_scale_hidden_config=hidden_activation_config,
                              norm_scale_out_config=hidden_activation_config)
    lstm.cuda()
    inp = torch.rand(SEQ_LENGTH, BATCH_SIZE, FEAT_IN, device=next(lstm.parameters()).device)
    state = LSTMState(torch.randn(BATCH_SIZE, HIDDEN_SIZE, device=next(lstm.parameters()).device),
                       torch.randn(BATCH_SIZE, HIDDEN_SIZE, device=next(lstm.parameters()).device))
    start = time.process_time()
    for _ in range(1000):
        _ = lstm(inp, state)
    end = time.process_time() - start
    end = end/1000.0
    print("Brevitas Weight-Only LSTM, JIT {}, took {} seconds".format(jit, end))

if __name__ == '__main__':
    args = parser.parse_args()
    os.environ["PYTORCH_JIT"] = "1" if args.jit else "0"
    if args.type == "basic":
        test_basic()
    elif args.type == "high_level":
        test_lstm_highlevel(args.jit)
    else:
        test_lstm_brevitas(args.jit)

