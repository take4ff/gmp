"""B5: 数値回帰（golden）。固定seedの順伝播と1エポック学習の出力が、保存済みの基準値から変わっていないこと。

リファクタ・バージョンコピー・一括置換で、意図せず計算が変わる（形は同じでも値が違う）ことを検知する。
意図した変更（モデル構造・損失・初期化など）の場合のみ、基準値を更新する:
    UPDATE_GOLDEN=1 python -m pytest -q tests/test_golden_numerics.py
更新したgoldenは差分をレビューしてからコミットすること。基準値はCPU・torch 2.8系で生成（許容 rel 1e-3）。
"""
import json
import os

import pytest
import torch

from transformer import config
from transformer.db.dataset import create_db_dataloader
from transformer.model import HierarchicalTransformer, MultiTaskLoss
from transformer.train import train_one_epoch
from transformer.utils.losses import build_loss_fn

GOLDEN = os.path.join(os.path.dirname(__file__), 'golden', 'numerics.json')
HEADS = ['region', 'position', 'aa_pos', 'strength', 'codon_pos', 'synonymous']


def _forward_stats():
    torch.manual_seed(0)
    model = HierarchicalTransformer().eval()
    T, C, F, N = config.TRAIN_MAX, config.MAX_CO_OCCURRENCE, config.NUM_FEATURE_STRING, config.NUM_CHEM_FEATURES
    g = torch.Generator().manual_seed(7)
    x_cat = torch.zeros(2, T, C, F, dtype=torch.long)
    x_num = torch.zeros(2, T, C, N)
    mask = torch.ones(2, T, dtype=torch.bool)
    for b, L in enumerate((6, 9)):
        mask[b, T - L:] = False
        for t in range(T - L, T):
            for c in range(1 + (t % 3)):
                v = torch.randint(0, 2, (F,), generator=g)
                v[0] = 1 + (t + c) % 4
                v[1] = 1 + int(torch.randint(0, 29000, (1,), generator=g))
                x_cat[b, t, c] = v
                x_num[b, t, c] = torch.rand(N, generator=g)
    with torch.no_grad():
        out = model(x_cat, x_num, src_key_padding_mask=mask)
    return {h: {'sum': float(o.sum()), 'abs_mean': float(o.abs().mean()), 'first': float(o.flatten()[0])}
            for h, o in zip(HEADS, out[:6])}


def _train_stats(synthetic_db):
    torch.manual_seed(0)
    model = HierarchicalTransformer()
    lw = MultiTaskLoss(6)
    opt = torch.optim.AdamW(list(model.parameters()) + list(lw.parameters()), lr=1e-3)
    before = [p.detach().clone() for p in model.parameters()]
    loader = create_db_dataloader(synthetic_db, 0, batch_size=3, shuffle=False, max_cooccurrence=20,
                                  num_workers_override=0)
    loss = train_one_epoch(model, loader, opt, build_loss_fn(None), lw)
    # 更新量(L2): パラメータ総和では勾配クリッピング等の小さな差が埋もれるため、更新そのものを見る
    delta = float(torch.sqrt(sum(((p.detach() - b) ** 2).sum() for p, b in zip(model.parameters(), before))))
    return {'loss': float(loss), 'task_weights': lw.get_weights(), 'param_update_l2': delta,
            'param_abs_sum': float(sum(p.detach().abs().sum() for p in model.parameters()))}


def _compare(actual, expected, path=''):
    if isinstance(expected, dict):
        assert set(actual) == set(expected), path
        for k in expected:
            _compare(actual[k], expected[k], f'{path}/{k}')
    elif isinstance(expected, list):
        assert len(actual) == len(expected), path
        for i, (a, e) in enumerate(zip(actual, expected)):
            _compare(a, e, f'{path}[{i}]')
    else:
        assert actual == pytest.approx(expected, rel=1e-3, abs=1e-5), f'{path}: {actual} != {expected}'


@pytest.fixture
def _golden_cfg(cfg):
    cfg(DEVICE='cpu', USE_SUBSTITUTION_HEAD=False, USE_MULTITASK_LOSS=True, MAX_GRAD_NORM=1.0,
        USE_REGION_CONDITIONED_POSITION=True, USE_BROADCAST_BACK_ATTENTION=False, USE_HOMOPLASY_PRIOR=False,
        USE_COATTN_FREQUENCY_PENALTY=False, DROPOUT=0.1)


def _load_or_update(key, actual):
    store = json.load(open(GOLDEN)) if os.path.exists(GOLDEN) else {}
    if os.environ.get('UPDATE_GOLDEN') == '1':
        store[key] = actual
        json.dump(store, open(GOLDEN, 'w'), indent=2, sort_keys=True)
        pytest.skip(f'golden[{key}] を更新しました')
    assert key in store, f"golden[{key}] が無い。UPDATE_GOLDEN=1 で生成し、差分をレビューすること"
    return store[key]


def test_forward_matches_golden(_golden_cfg):
    actual = _forward_stats()
    _compare(actual, _load_or_update('forward', actual))


def test_one_epoch_training_matches_golden(_golden_cfg, synthetic_db):
    actual = _train_stats(synthetic_db)
    _compare(actual, _load_or_update('train_one_epoch', actual))


def test_forward_is_deterministic(_golden_cfg):
    assert _forward_stats() == _forward_stats()
