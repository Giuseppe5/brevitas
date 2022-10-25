from operator import mul
from functools import reduce
import pytest_cases

import torch
import pytest

from brevitas.quant.scaled_int import Int32Bias
from .common import *
from pytest_cases import parametrize_with_cases
from .quant_module_cases import QuantWBIOLCases
from pytest_cases import parametrize_with_cases, get_case_id

@parametrize_with_cases('model', cases=QuantWBIOLCases)
@pytest.mark.parametrize('export_type', ['qop', 'qcdq'])
def test_ort(model, export_type, current_cases):
    cases_generator_func = current_cases['model'][1]
    case_id = get_case_id(cases_generator_func)
    impl = case_id.split('-')[-2] # Inverse list of definition, 'export_type' is -1, 'impl' is -2, etc.

    if impl in ('QuantConvTranspose1d', 'QuantConvTranspose2d'):
        pytest.skip('Export of ConvTranspose is not supported')

    if impl in ('QuantLinear'):
        IN_SIZE = (1, IN_CH)
    elif impl in ('QuantConv1d'):
        IN_SIZE = (1, IN_CH, FEATURES)
    else:
        IN_SIZE = (1, IN_CH, FEATURES, FEATURES)
    
    inp = gen_linspaced_data(reduce(mul, IN_SIZE), -1, 1).reshape(IN_SIZE)

    model(torch.from_numpy(inp))  # accumulate scale factors
    model.eval()
    export_name='qcdq_qop_export.onnx'
    assert is_brevitas_ort_close(model, inp, export_name, export_type, tolerance_flag=TOLERANCE)