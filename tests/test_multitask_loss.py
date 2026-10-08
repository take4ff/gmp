"""A4: MultiTaskLoss（Kendall不確実性重み付け）。重みログ追加（2026-10-05）の前提を固定。"""
import math

import pytest
import torch

from transformer.model import MultiTaskLoss


def test_initial_weights_are_one():
    m = MultiTaskLoss(num_tasks=6)
    assert m.get_weights() == pytest.approx([1.0] * 6)


def test_initial_loss_is_plain_sum():
    m = MultiTaskLoss(num_tasks=3)
    losses = [torch.tensor(1.5), torch.tensor(2.0), torch.tensor(0.5)]
    assert float(m(*losses)) == pytest.approx(4.0)


def test_get_weights_matches_exp_neg_log_vars():
    m = MultiTaskLoss(num_tasks=3)
    with torch.no_grad():
        m.log_vars.copy_(torch.tensor([0.0, 1.0, -1.0]))
    assert m.get_weights() == pytest.approx([1.0, math.exp(-1.0), math.exp(1.0)])


def test_loss_formula_precision_times_loss_plus_log_var():
    m = MultiTaskLoss(num_tasks=2)
    with torch.no_grad():
        m.log_vars.copy_(torch.tensor([0.5, -0.5]))
    out = m(torch.tensor(2.0), torch.tensor(3.0))
    expected = math.exp(-0.5) * 2.0 + 0.5 + math.exp(0.5) * 3.0 - 0.5
    assert float(out) == pytest.approx(expected)


def test_gradient_flows_to_log_vars():
    m = MultiTaskLoss(num_tasks=2)
    m(torch.tensor(2.0), torch.tensor(3.0)).backward()
    assert m.log_vars.grad is not None
    assert torch.all(m.log_vars.grad != 0)
