# Copyright (c) 2019-     Xilinx, Inc              (Giuseppe Franco)
# Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
# Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
# Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
# Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
# Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
# Copyright (c) 2011-2013 NYU                      (Clement Farabet)
# Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
# Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
# Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)

# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.

# 3. Neither the names of Xilinx, Facebook, Deepmind Technologies, NYU,
#    NEC Laboratories America and IDIAP Research Institute nor the names
#    of its contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import torch
import math
import brevitas.function.ops as ops
import pytest
import hypothesis.strategies as st
from hypothesis import assume, given, note
from hypothesis import seed as set_seed
from hypothesis import settings

SEED = 123456
RTOL = 1e-03
ATOL = 1e-03

torch.random.manual_seed(SEED)
set_seed(SEED)

#  Define custom type of floating point generator
float_st = st.floats(allow_nan=False, allow_infinity=False, width=32)
float_st_nz = st.floats(allow_nan=False, allow_infinity=False, width=32).filter(lambda x: x != 0.0)


# Create custom strategy for generating two lists of floats with equal size
@st.composite
def two_lists_equal_size(draw):
    list_one = draw(st.lists(float_st, min_size=1))
    size = len(list_one)
    list_two = draw(st.lists(float_st_nz, min_size=size, max_size=size))
    return list_one, list_two


# Create custom strategy for generating three floating point numbers such that minimum < value < maximum
@st.composite
def two_ordered_numbers(draw):
    minimum = draw(float_st)
    maximum = draw(
        st.floats(allow_infinity=False, allow_nan=False, width=32, min_value=minimum).filter(lambda x: x != 0.0))
    return minimum, maximum


@st.composite
def two_ordered_lists(draw):
    size = draw(st.integers().filter(lambda x: not math.isinf(x) or not math.isnan(x)))
    minimum_list = [0] * size
    maximum_list = [0] * size
    for i in range(size):
        minimum, maximum = draw(two_ordered_numbers)
        minimum_list[i] = minimum
        maximum_list[i] = maximum
    return minimum_list, maximum_list


@given(lists=two_lists_equal_size())
@settings(deadline=None)
def test_ste_of_round_ste(lists):
    x = lists[0]
    grad = lists[1]
    x = torch.tensor(x, requires_grad=True)
    grad = torch.tensor(grad)

    output = ops.round_ste(x)
    output.backward(grad, retain_graph=True)

    assert (torch.allclose(x.grad, grad, RTOL, ATOL))


@given(x=st.lists(float_st, min_size=1))
def test_result_of_round_ste(x):
    x = torch.tensor(x)

    output = ops.round_ste(x)
    expected_output = torch.round(x)

    assert (torch.allclose(expected_output, output, RTOL, ATOL))


@given(minmax=two_ordered_numbers(), x=st.lists(float_st, min_size=1))
def test_result_of_tensor_clamp(minmax, x):
    minimum = torch.tensor(minmax[0])
    value = torch.tensor(x)
    maximum = torch.tensor(minmax[1])

    output = ops.tensor_clamp(value, minimum, maximum)
    expected_output = torch.clamp(value, minmax[0], minmax[1])  # torch.clamp requires float and not tensor as arguments
    assert ((output >= minimum).all() and (output <= maximum).all())
    assert (torch.allclose(expected_output, output, RTOL, ATOL))


@given(narrow_range=st.booleans(), bit_width=st.integers(min_value=0, max_value=8))
def test_result_of_max_uint(narrow_range, bit_width):
    bit_width = torch.tensor(bit_width, dtype=torch.float)
    output = ops.max_uint(narrow_range, bit_width)

    if narrow_range:
        expected_output = (2 ** bit_width) - 2
    else:
        expected_output = (2 ** bit_width) - 1
    expected_output = torch.round(expected_output)

    assert (output % 1 == 0)  # Check that number is an integer, not most elegant solution
    assert (torch.allclose(expected_output, output, RTOL, ATOL))


@given(signed=st.booleans(), bit_width=st.integers(min_value=0, max_value=8))
def test_result_of_max_int(signed, bit_width):
    bit_width = torch.tensor(bit_width, dtype=torch.float)
    output = ops.max_int(signed, bit_width)

    if signed:
        expected_output = (2 ** (bit_width - 1)) - 1
    else:
        expected_output = (2 ** bit_width) - 1
    expected_output = torch.round(expected_output)

    assert (output % 1 == 0)  # Check that number is an integer, not most elegant solution
    assert (torch.allclose(expected_output, output, RTOL, ATOL))


@given(narrow_range=st.booleans(), signed=st.booleans(), bit_width=st.integers(min_value=0, max_value=8))
def test_result_of_max_int(narrow_range, signed, bit_width):
    bit_width = torch.tensor(bit_width, dtype=torch.float)
    output = ops.min_int(signed, narrow_range, bit_width)

    if signed and narrow_range:
        expected_output = -(2 ** (bit_width - 1)) + 1
    elif signed and not narrow_range:
        expected_output = -(2 ** (bit_width - 1))
    else:
        expected_output = torch.tensor(0.0)

    expected_output = torch.round(expected_output)

    assert (output % 1 == 0)  # Check that number is an integer, not most elegant solution
    assert (torch.allclose(expected_output, output, RTOL, ATOL))


@given(minmax=two_ordered_numbers(), lists=two_lists_equal_size())
@settings(deadline=None)
def test_ste_of_round_ste(minmax, lists):
    minimum = torch.tensor(minmax[0])
    value = torch.tensor(lists[0], requires_grad=True)
    grad = lists[1]
    grad = torch.tensor(grad)
    maximum = torch.tensor(minmax[1])

    output = ops.tensor_clamp_ste(value, minimum, maximum)

    output.backward(grad, retain_graph=True)

    assert (torch.allclose(value.grad, grad, RTOL, ATOL))
