# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause


from hypothesis import given
import pytest
import warnings

import torch
from brevitas.nn import QuantRNN, QuantLSTM

from tests.brevitas.hyp_helper import float_tensor_random_size_st
import hypothesis.strategies as st


ATOL=1e-6


class TestRecurrent:

    @given(
        inp=float_tensor_random_size_st(dims=3, max_size=3),
        hidden_size=st.integers(min_value=1, max_value=3))
    @pytest.mark.parametrize("batch_first", [True, False])
    @pytest.mark.parametrize("bidirectional", [True, False])
    @pytest.mark.parametrize("num_layers", [1, 2])
    @pytest.mark.parametrize("bias", [True, False])
    def test_rnn_quant_disabled_fwd(
            self, inp, hidden_size, batch_first, bidirectional, num_layers, bias):
        inp_size = inp.size(-1)
        m = torch.nn.RNN(
            inp_size,
            hidden_size,
            batch_first=batch_first,
            bidirectional=bidirectional,
            num_layers=num_layers,
            bias=bias)
        qm = QuantRNN(
            inp_size,
            hidden_size,
            weight_quant=None,
            bias_quant=None,
            io_quant=None,
            gate_acc_quant=None, 
            batch_first=batch_first,
            bidirectional=bidirectional,
            num_layers=num_layers,
            bias=bias)
        qm.load_state_dict(m.state_dict())
        ref_out = m(inp)
        out = qm(inp)
        assert torch.isclose(out[0], ref_out[0], atol=ATOL).all()
        assert torch.isclose(out[1], ref_out[1], atol=ATOL).all()

    @given(
        inp=float_tensor_random_size_st(dims=3, max_size=3),
        hidden_size=st.integers(min_value=1, max_value=3))
    @pytest.mark.parametrize("batch_first", [True, False])
    @pytest.mark.parametrize("bidirectional", [True, False])
    @pytest.mark.parametrize("num_layers", [1, 2])
    @pytest.mark.parametrize("bias", [True, False])
    def test_rnn_quant_disabled_fwd_state_dict(
            self, inp, hidden_size, batch_first, bidirectional, num_layers, bias):
        inp_size = inp.size(-1)
        qm = QuantRNN(
            inp_size,
            hidden_size,
            weight_quant=None,
            bias_quant=None,
            io_quant=None,
            gate_acc_quant=None,
            batch_first=batch_first,
            bidirectional=bidirectional,
            num_layers=num_layers,
            bias=bias)
        # Test that the brevitas model can be saved/loaded without warning
        with warnings.catch_warnings(record=True) as wlist:
            qm.load_state_dict(qm.state_dict())
            for w in wlist:
                assert "Positional args are being deprecated" not in str(w.message)

    @given(
         inp=float_tensor_random_size_st(dims=3, max_size=3),
         hidden_size=st.integers(min_value=1, max_value=3))
    @pytest.mark.parametrize("batch_first", [True, False])
    @pytest.mark.parametrize("bidirectional", [True, False])
    @pytest.mark.parametrize("num_layers", [1, 2])
    @pytest.mark.parametrize("bias", [True, False])
    def test_lstm_quant_disabled_fwd(
             self, inp, hidden_size, batch_first, bidirectional, num_layers, bias):
        inp_size = inp.size(-1)
        m = torch.nn.LSTM(
            inp_size,
            hidden_size,
            batch_first=batch_first,
            bidirectional=bidirectional,
            num_layers=num_layers,
            bias=bias)
        qm = QuantLSTM(
            inp_size,
            hidden_size,
            weight_quant=None,
            io_quant=None,
            gate_acc_quant=None,
            cell_state_quant=None,
            bias_quant=None,
            sigmoid_quant=None,
            tanh_quant=None,
            cat_output_cell_states=True,
            batch_first=batch_first,
            bidirectional=bidirectional,
            num_layers=num_layers,
            bias=bias)
        qm.load_state_dict(m.state_dict())
        ref_out = m(inp)
        out = qm(inp)
        # output values
        assert torch.isclose(out[0], ref_out[0], atol=ATOL).all()
        # hidden states
        assert torch.isclose(out[1][0], ref_out[1][0], atol=ATOL).all()
        # cell states
        assert torch.isclose(out[1][1], ref_out[1][1], atol=ATOL).all()

    @given(
        inp=float_tensor_random_size_st(dims=3, max_size=3),
        hidden_size=st.integers(min_value=1, max_value=3))
    @pytest.mark.parametrize("batch_first", [True, False])
    @pytest.mark.parametrize("bidirectional", [True, False])
    @pytest.mark.parametrize("num_layers", [1, 2])
    @pytest.mark.parametrize("bias", [True, False])
    def test_lstm_quant_disabled_fwd_state_dict(
            self, inp, hidden_size, batch_first, bidirectional, num_layers, bias):
        inp_size = inp.size(-1)
        qm = QuantLSTM(
            inp_size,
            hidden_size,
            weight_quant=None,
            io_quant=None,
            gate_acc_quant=None,
            cell_state_quant=None,
            bias_quant=None,
            sigmoid_quant=None,
            tanh_quant=None,
            cat_output_cell_states=True,
            batch_first=batch_first,
            bidirectional=bidirectional,
            num_layers=num_layers,
            bias=bias)
        # Test that the brevitas model can be saved/loaded without warning
        with warnings.catch_warnings(record=True) as wlist:
            qm.load_state_dict(qm.state_dict())
            for w in wlist:
                assert "Positional args are being deprecated" not in str(w.message)


    @pytest.mark.parametrize("batch_first", [True, False])
    @pytest.mark.parametrize("bidirectional", [True, False, 'shared_input_hidden_weights'])
    @pytest.mark.parametrize("num_layers", [1, 2])
    @pytest.mark.parametrize("bias", [True, False])
    @pytest.mark.parametrize("return_quant_tensor", [True, False])
    def test_quant_rnn_fwd_call(
            self,
            batch_first,
            bidirectional,
            num_layers,
            bias,
            return_quant_tensor):
        inp_size = 4
        hidden_size = 5
        inp = torch.randn(2, 3, inp_size)
        m = QuantRNN(
            inp_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=batch_first,
            bidirectional=bidirectional,
            bias=bias,
            shared_input_hidden_weights=bidirectional=='shared_input_hidden_weights',
            return_quant_tensor=return_quant_tensor)
        assert m(inp) is not None

    @pytest.mark.parametrize("batch_first", [True, False])
    @pytest.mark.parametrize("bidirectional", [True, False, 'shared_input_hidden_weights'])
    @pytest.mark.parametrize("num_layers", [1, 2])
    @pytest.mark.parametrize("bias", [True, False])
    @pytest.mark.parametrize("return_quant_tensor", [True, False])
    @pytest.mark.skip("FIXME: many warnings due to __torch_function__")
    def test_quant_rnn_state_dict(
            self,
            batch_first,
            bidirectional,
            num_layers,
            bias,
            return_quant_tensor):
        inp_size = 4
        hidden_size = 5
        m = QuantRNN(
            inp_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=batch_first,
            bidirectional=bidirectional,
            bias=bias,
            shared_input_hidden_weights=bidirectional=='shared_input_hidden_weights',
            return_quant_tensor=return_quant_tensor)
        with warnings.catch_warnings(record=True) as wlist:
            m.load_state_dict(m.state_dict())
            for w in wlist:
                assert "Positional args are being deprecated" not in str(w.message)

    @pytest.mark.parametrize("batch_first", [True, False])
    @pytest.mark.parametrize("bidirectional", [True, False, 'shared_input_hidden_weights'])
    @pytest.mark.parametrize("num_layers", [1, 2])
    @pytest.mark.parametrize("bias", [True, False])
    @pytest.mark.parametrize("coupled_input_forget_gates", [True, False])
    @pytest.mark.parametrize("return_quant_tensor", [True, False])
    @pytest.mark.parametrize("shared_intra_layer_weight_quant", [True, False])
    @pytest.mark.parametrize("shared_intra_layer_gate_acc_quant", [True, False])
    @pytest.mark.parametrize("shared_cell_state_quant", [True, False])
    def test_quant_lstm_fwd_call(
            self,
            batch_first,
            bidirectional,
            num_layers,
            bias,
            coupled_input_forget_gates,
            return_quant_tensor,
            shared_intra_layer_weight_quant,
            shared_intra_layer_gate_acc_quant,
            shared_cell_state_quant):
        inp_size = 4
        hidden_size = 5
        inp = torch.randn(2, 3, inp_size)
        m = QuantLSTM(
            inp_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=batch_first,
            bidirectional=bidirectional,
            bias=bias,
            coupled_input_forget_gates=coupled_input_forget_gates,
            shared_input_hidden_weights=bidirectional=='shared_input_hidden_weights',
            shared_intra_layer_weight_quant=shared_intra_layer_weight_quant,
            shared_intra_layer_gate_acc_quant=shared_intra_layer_gate_acc_quant,
            shared_cell_state_quant=shared_cell_state_quant,
            cat_output_cell_states=shared_cell_state_quant,
            return_quant_tensor=return_quant_tensor)
        assert m(inp) is not None

    @pytest.mark.parametrize("batch_first", [True, False])
    @pytest.mark.parametrize("bidirectional", [True, False, 'shared_input_hidden_weights'])
    @pytest.mark.parametrize("num_layers", [1, 2])
    @pytest.mark.parametrize("bias", [True, False])
    @pytest.mark.parametrize("coupled_input_forget_gates", [True, False])
    @pytest.mark.parametrize("return_quant_tensor", [True, False])
    @pytest.mark.parametrize("shared_intra_layer_weight_quant", [True, False])
    @pytest.mark.parametrize("shared_intra_layer_gate_acc_quant", [True, False])
    @pytest.mark.parametrize("shared_cell_state_quant", [True, False])
    @pytest.mark.skip("FIXME: many warnings due to __torch_function__")
    def test_quant_lstm_fwd_state_dict(
            self,
            batch_first,
            bidirectional,
            num_layers,
            bias,
            coupled_input_forget_gates,
            return_quant_tensor,
            shared_intra_layer_weight_quant,
            shared_intra_layer_gate_acc_quant,
            shared_cell_state_quant):
        inp_size = 4
        hidden_size = 5
        qm = QuantLSTM(
            inp_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=batch_first,
            bidirectional=bidirectional,
            bias=bias,
            coupled_input_forget_gates=coupled_input_forget_gates,
            shared_input_hidden_weights=bidirectional=='shared_input_hidden_weights',
            shared_intra_layer_weight_quant=shared_intra_layer_weight_quant,
            shared_intra_layer_gate_acc_quant=shared_intra_layer_gate_acc_quant,
            shared_cell_state_quant=shared_cell_state_quant,
            cat_output_cell_states=shared_cell_state_quant,
            return_quant_tensor=return_quant_tensor)
        # Test that the brevitas model can be saved/loaded without warning
        with warnings.catch_warnings(record=True) as wlist:
            qm.load_state_dict(qm.state_dict())
            for w in wlist:
                assert "Positional args are being deprecated" not in str(w.message)
