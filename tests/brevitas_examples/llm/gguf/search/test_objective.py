# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""What a fit is weighed by."""

import pytest
import torch

from brevitas_examples.llm.gguf_export.search.objective import MAGNITUDE_FORMS
from brevitas_examples.llm.gguf_export.search.objective import objective_weights

N_OUT, N_SUPER, N_SUB, BLOCK = 4, 2, 8, 32


def _weights(seed=0):
    torch.manual_seed(seed)
    x = torch.randn(N_OUT, N_SUPER, N_SUB, BLOCK)
    importance = torch.rand(N_SUPER, N_SUB, BLOCK) + 0.1
    return x, importance


@pytest.mark.parametrize("factor", [1.0, 2.0])
def test_rms_quad_over_a_super_block(factor):
    x, importance = _weights()
    mean_sq = (x * x).reshape(N_OUT, N_SUPER, -1).mean(-1).reshape(N_OUT, N_SUPER, 1, 1)
    expected = importance * torch.sqrt(factor * mean_sq + x * x)
    got = objective_weights(x, importance, 'rms_quad', factor, 'super_block')
    assert torch.allclose(got, expected)


def test_rms_quad_over_a_row():
    x, importance = _weights()
    flat = x.reshape(N_OUT, N_SUPER * N_SUB, BLOCK)
    flat_importance = importance.reshape(N_SUPER * N_SUB, BLOCK)
    mean_sq = (x * x).reshape(N_OUT, -1).mean(-1).reshape(N_OUT, 1, 1)
    expected = flat_importance * torch.sqrt(mean_sq + flat * flat)
    got = objective_weights(flat, flat_importance, 'rms_quad', 1.0, 'row')
    assert torch.allclose(got, expected)


def test_rms_plus_abs_over_a_block():
    x, importance = _weights()
    expected = importance * (torch.sqrt((x * x).mean(-1, keepdim=True)) + x.abs())
    assert torch.allclose(objective_weights(x, importance, 'rms_plus_abs', scope='block'), expected)


def test_one_leaves_the_importance_alone():
    x, importance = _weights()
    assert torch.equal(objective_weights(x, importance, 'one'), importance.expand_as(x))


def test_the_scopes_disagree():
    x, importance = _weights()
    seen = [objective_weights(x, importance, scope=s) for s in ('block', 'super_block', 'row')]
    for a, b in ((0, 1), (0, 2), (1, 2)):
        assert not torch.allclose(seen[a], seen[b])


def test_a_super_block_scope_needs_super_blocks():
    x, importance = _weights()
    flat = x.reshape(N_OUT, N_SUPER * N_SUB, BLOCK)
    with pytest.raises(ValueError, match="super_block"):
        objective_weights(flat, importance.reshape(-1, BLOCK), scope='super_block')


@pytest.mark.parametrize("form", MAGNITUDE_FORMS)
def test_every_form_is_positive_on_nonzero_weights(form):
    x, importance = _weights()
    assert (objective_weights(x, importance, form) > 0).all()
