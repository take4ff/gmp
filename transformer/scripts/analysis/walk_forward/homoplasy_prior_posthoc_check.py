# --- transformer/scripts/analysis/walk_forward/homoplasy_prior_posthoc_check.py ---
"""USE_HOMOPLASY_PRIOR（position headへのlog(1+再発回数)バイアス）を、このフラグOFFで
学習済みのwalk_forward checkpointに後付けで加え、評価時のみ（再学習なしに）効果の
方向性を見る簡易チェック。

注意: homoplasy_scale は本来学習可能パラメータだが、後付けのため固定値
（--scale、既定1.0=学習時初期値）で動く。したがって「学習時から有効にした場合」の
効果ではなく、あくまで方向性の確認（公平な評価には学習時からのONが必要）。

hierarchical_prediction_check.py と同様、assign_fold_test_window() でfold窓を
明示的に揃えてから評価する。比較基準は --scale 0（バイアス無効）を同一スクリプトで
走らせた値、または既に得ている同設定の値を使うこと。

Usage:
  python -m transformer.scripts.analysis.walk_forward.homoplasy_prior_posthoc_check \\
      --walk_forward_dir outputs/transformer/results/walk_forward/<timestamp> \\
      --folds 3 [--scale 1.0]
"""
import argparse

import torch
import torch.nn as nn

from transformer import config
from transformer.db.connection import get_db_path
from transformer.scripts.analysis.xai import _xai_common as X
from transformer.evaluate import evaluate
from transformer.utils.losses import build_loss_fn
from transformer.utils.logging import force_print


def _weighted(metrics, key):
    total_n = sum(m['num_samples'] for m in metrics.values())
    if total_n == 0:
        return 0.0, 0
    return sum(m[key] * m['num_samples'] for m in metrics.values()) / total_n, total_n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--walk_forward_dir', required=True)
    ap.add_argument('--folds', type=int, nargs='+', default=None)
    ap.add_argument('--scale', type=float, default=1.0,
                    help='homoplasy_scale の固定値（1.0=学習時の初期値）')
    args = ap.parse_args()

    fold_windows = X.get_fold_windows()
    fold_ckpts = X.discover_fold_checkpoints(args.walk_forward_dir, args.folds)
    if not fold_ckpts:
        raise SystemExit(f"[ERROR] fold チェックポイントが見つかりません: {args.walk_forward_dir}")

    db_path = get_db_path()
    loss_fn = build_loss_fn(None)

    for fold_id in sorted(fold_ckpts):
        train_start, split_date, split_end = fold_windows[fold_id]
        X.assign_fold_test_window(db_path, train_start, split_date, split_end)
        model, loader, device = X.load_model_and_loader(fold_ckpts[fold_id], split='test')
        X.assign_fold_test_window(db_path, train_start, split_date, split_end)

        # config_snapshot が USE_HIERARCHICAL_PREDICTION を学習時の値(False)に上書きするため、
        # 現行既定（True）に明示的に揃える（比較基準: hier=True の fold_3 = 5.4709%）。
        config.USE_HIERARCHICAL_PREDICTION = True
        config.HIERARCHICAL_TOPK_REGIONS = 3

        bias = model._load_homoplasy_bias('USE_HOMOPLASY_PRIOR')
        if bias is None:
            raise SystemExit("[ERROR] homoplasy bias を読み込めませんでした")
        model.homoplasy_bias = bias.to(device)
        model.homoplasy_scale = nn.Parameter(torch.tensor(float(args.scale), device=device))

        force_print(f"\n===== Fold {fold_id}: homoplasy_scale={args.scale} "
                    f"(USE_HIERARCHICAL_PREDICTION={config.USE_HIERARCHICAL_PREDICTION}) =====")
        _, metrics, _, _, _, _ = evaluate(model, loader, loss_fn, (6.0, 10.0))
        for key in ('position_hit_rate', 'region_hit_rate', 'aa_pos_hit_rate'):
            if all(key in m for m in metrics.values()):
                v, n = _weighted(metrics, key)
                force_print(f"  {key}={v:.4f}  n={n}")
        del model


if __name__ == '__main__':
    main()
