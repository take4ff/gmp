"""B5: 学習→checkpoint保存→config_snapshot経由の読み戻し（_xai_common.load_model_and_loader）。

読み戻したモデルが同じ入力で同じ出力を返し、スナップショット経由で学習時のフラグが復元されること。
"""
import os

import torch

from transformer_260817 import config
from transformer_260817.db.dataset import create_db_dataloader
from transformer_260817.model import HierarchicalTransformer
from transformer_260817.scripts.analysis.xai import _xai_common as X
from transformer_260817.utils.io import save_config_copy


def _save(tmp_path, model):
    out = tmp_path / 'run'
    save_config_copy(str(out))                      # → run/models/config_snapshot.py
    ck = out / 'models' / 'best_model.pth'
    torch.save({'model_state_dict': model.state_dict(), 'epoch': 0}, ck)
    return str(ck)


def _first_batch(synthetic_db, split):
    loader = create_db_dataloader(synthetic_db, split, batch_size=3, shuffle=False,
                                  max_cooccurrence=20, num_workers_override=0)
    return next(iter(loader))


def test_roundtrip_reproduces_outputs_and_flags(synthetic_db, cfg, tmp_path):
    cfg(USE_REGION_CONDITIONED_POSITION=True, USE_HIERARCHICAL_PREDICTION=False)
    torch.manual_seed(0)
    model = HierarchicalTransformer().eval()
    ck = _save(tmp_path, model)
    batch = _first_batch(synthetic_db, 2)
    x_cat, x_num, mask = batch.inputs
    with torch.no_grad():
        ref = model(x_cat, x_num, src_key_padding_mask=mask)

    # 読み戻し側の config を意図的に別の値へ（スナップショットが学習時の値へ戻すことを確認）
    cfg(USE_REGION_CONDITIONED_POSITION=False, USE_HIERARCHICAL_PREDICTION=True)
    loaded, loader, device = X.load_model_and_loader(ck, split='test')
    assert config.USE_REGION_CONDITIONED_POSITION is True
    assert config.USE_HIERARCHICAL_PREDICTION is False        # 学習時の値（スナップショットが上書き）
    with torch.no_grad():
        out = loaded(x_cat, x_num, src_key_padding_mask=mask)
    assert len(ref) == len(out)
    for a, b in zip(ref, out):
        if a is None or b is None:                      # 無効なヘッド（USE_SUBSTITUTION_HEAD=False等）はNone
            assert a is None and b is None
        else:
            assert torch.allclose(a, b, atol=1e-5), "読み戻し後の出力が一致しない"


def test_loader_built_from_snapshot_reads_the_requested_split(synthetic_db, cfg, tmp_path):
    torch.manual_seed(0)
    ck = _save(tmp_path, HierarchicalTransformer().eval())
    _, loader, _ = X.load_model_and_loader(ck, split='test')
    paths = [p for b in loader for p in b.full_paths]
    assert sorted(paths) == sorted(['G1T>A2C>C3G', 'T9C>A8G>C7T'])
