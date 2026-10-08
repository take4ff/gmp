"""B4: 合成DBで1エポック学習するスモーク。損失が有限・勾配が全パラメータに流れる・パラメータが更新される。
新フラグON時の追加パラメータ（例: homoplasy_scale）にも勾配が流れることを確認する。"""
import math

import pytest
import torch

from transformer_260817 import config
from transformer_260817.db.dataset import create_db_dataloader
from transformer_260817.model import HierarchicalTransformer, MultiTaskLoss
from transformer_260817.train import train_one_epoch
from transformer_260817.utils.losses import build_loss_fn


def _train_one(synthetic_db, cfg, **flags):
    cfg(USE_MULTITASK_LOSS=True, USE_SUBSTITUTION_HEAD=False, MAX_GRAD_NORM=1.0, **flags)
    torch.manual_seed(0)
    model = HierarchicalTransformer()
    loss_wrapper = MultiTaskLoss(num_tasks=6)
    params = list(model.parameters()) + list(loss_wrapper.parameters())
    opt = torch.optim.AdamW(params, lr=1e-3)
    before = [p.detach().clone() for p in params]
    loader = create_db_dataloader(synthetic_db, 0, batch_size=3, shuffle=False,
                                  max_cooccurrence=20, num_workers_override=0)
    loss = train_one_epoch(model, loader, opt, build_loss_fn(None), loss_wrapper)
    return model, loss_wrapper, params, before, loss


def test_one_epoch_loss_finite_params_update_and_no_nan(synthetic_db, cfg):
    model, lw, params, before, loss = _train_one(synthetic_db, cfg)
    assert math.isfinite(loss) and loss > 0
    assert all(torch.isfinite(p).all() for p in params)
    assert any(not torch.equal(b, p.detach()) for b, p in zip(before, params))   # 少なくとも一部は更新


# 既定設定で使われないのが仕様のパラメータ（常に定義されるが、特定フラグ時のみforwardで使う）。
# 新しい「使われないパラメータ」が増えたらこのテストが失敗して気付ける（配線漏れの検知）。
_UNUSED_BY_DEFAULT = {'cls_token'}      # TEMPORAL_POOLING='cls' のときのみ使用


def test_gradients_reach_every_parameter_by_default(synthetic_db, cfg):
    model, lw, params, before, loss = _train_one(synthetic_db, cfg)
    names = {id(p): n for n, p in list(model.named_parameters()) + list(lw.named_parameters())}
    missing = [names[id(p)] for p in params if p.requires_grad and p.grad is None]
    assert set(missing) == _UNUSED_BY_DEFAULT, f"勾配が流れていないパラメータ: {missing}"


def test_cls_token_receives_gradient_with_cls_pooling(synthetic_db, cfg):
    model, *_ = _train_one(synthetic_db, cfg, TEMPORAL_POOLING='cls')
    assert model.cls_token.grad is not None


def test_region_conditioned_scale_receives_gradient_when_enabled(synthetic_db, cfg):
    model, *_ = _train_one(synthetic_db, cfg, USE_REGION_CONDITIONED_POSITION=True)
    assert model.region_hier_scale.grad is not None


def test_homoplasy_scale_receives_gradient_when_enabled(synthetic_db, cfg, tmp_path):
    csv = tmp_path / 'h.csv'
    csv.write_text('position_id,recurrence_count\n100,5\n200,50\n300,500\n')
    model, *_ = _train_one(synthetic_db, cfg, USE_HOMOPLASY_PRIOR=True, HOMOPLASY_CSV=str(csv))
    assert model.homoplasy_scale is not None and model.homoplasy_scale.grad is not None


def test_multitask_weights_move_after_training(synthetic_db, cfg):
    _, lw, *_ = _train_one(synthetic_db, cfg)
    assert any(abs(w - 1.0) > 1e-6 for w in lw.get_weights())   # log_vars が更新されている
