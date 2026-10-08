"""A3: _build_soft_target（分岐K×共起Mの確率分配）。"""
import pytest
import torch

from transformer_261008 import config
from transformer_261008.db.dataset import _build_soft_target

TASKS = ['region', 'position', 'aa_pos', 'codon_pos', 'synonymous']


def t(region, pos, aa, codon, syn):
    return (region, pos, aa, codon, syn)


def test_empty_group_returns_zero_vectors():
    out = _build_soft_target([])
    assert set(out) == set(TASKS)
    assert all(float(v.sum()) == 0.0 for v in out.values())


def test_each_task_sums_to_one():
    group = [[t(1, 100, 10, 1, 0)], [t(2, 200, 20, 2, 1), t(3, 300, 30, 3, 0)]]
    out = _build_soft_target(group)
    for task in TASKS:
        assert float(out[task].sum()) == pytest.approx(1.0)


def test_probability_is_one_over_K_times_one_over_M():
    # K=2 ルート。ルート0: M=1、ルート1: M=2  → 0.5 / 0.25 / 0.25
    group = [[t(1, 100, 10, 1, 0)], [t(2, 200, 20, 2, 1), t(3, 300, 30, 3, 0)]]
    pos = _build_soft_target(group)['position']
    assert float(pos[100]) == pytest.approx(0.5)
    assert float(pos[200]) == pytest.approx(0.25)
    assert float(pos[300]) == pytest.approx(0.25)


def test_same_variant_in_multiple_routes_accumulates():
    group = [[t(1, 100, 10, 1, 0)], [t(1, 100, 10, 1, 0)], [t(2, 200, 20, 2, 1)]]
    pos = _build_soft_target(group)['position']
    assert float(pos[100]) == pytest.approx(2 / 3)
    assert float(pos[200]) == pytest.approx(1 / 3)


def test_out_of_range_ids_are_ignored():
    bad = config.VOCAB_SIZE_POSITION          # 範囲外
    group = [[t(1, bad, 10, 1, 0)], [t(1, 100, 10, 1, 0)]]
    pos = _build_soft_target(group)['position']
    assert float(pos[100]) == pytest.approx(0.5)
    assert float(pos.sum()) == pytest.approx(0.5)   # 範囲外分は捨てられ和は1未満


def test_temperature_sharpens_and_renormalizes():
    group = [[t(1, 100, 10, 1, 0)], [t(1, 100, 10, 1, 0)], [t(2, 200, 20, 2, 1)]]
    base = _build_soft_target(group, temperature=1.0)['position']
    sharp = _build_soft_target(group, temperature=0.5)['position']
    flat = _build_soft_target(group, temperature=2.0)['position']
    for v in (sharp, flat):
        assert float(v.sum()) == pytest.approx(1.0)
    assert float(sharp[100]) > float(base[100]) > float(flat[100])


def test_temperature_one_is_exact_noop():
    group = [[t(1, 100, 10, 1, 0), t(2, 200, 20, 2, 1)]]
    a = _build_soft_target(group, temperature=1.0)
    b = _build_soft_target(group)    # configの既定(1.0)
    for task in TASKS:
        assert torch.equal(a[task], b[task])
