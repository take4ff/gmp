# --- transformer_260817/scripts/analysis/walk_forward/hierarchical_prediction_check.py ---
"""USE_HIERARCHICAL_PREDICTION（予測Region上位K個に属さないposition候補をマスク）の
効果を、学習済みwalk_forward checkpointに対して評価時マスキングのみで（再学習なしに）
全fold横断で検証する。

_xai_common.load_model_and_loader() はcheckpoint同梱のconfig_snapshot.pyを読み込む
だけでDB側のsplit_type_wf（どのサンプルがtest扱いか）は再割り当てしないため、直近の
walk_forward実行が別foldで終わっているとそのfoldのtestデータを誤って読んでしまう
（2026-08-08 fold_3の検証中に発覚：fold_7のtestデータで評価していた）。本スクリプトは
assign_fold_test_window()で明示的にfold窓を揃えてから評価する。

Usage:
  python -m transformer_260817.scripts.analysis.walk_forward.hierarchical_prediction_check \\
      --walk_forward_dir outputs/transformer_260817/results/walk_forward/<timestamp> \\
      [--folds 3 4] [--topk_regions 3]
"""
import argparse
import os

import pandas as pd

from transformer_260817 import config
from transformer_260817.db.connection import get_db_path
from transformer_260817.scripts.analysis.xai import _xai_common as X
from transformer_260817.evaluate import evaluate
from transformer_260817.utils.losses import build_loss_fn
from transformer_260817.utils.logging import force_print


def _weighted(metrics, key):
    total_n = sum(m['num_samples'] for m in metrics.values())
    if total_n == 0:
        return 0.0, 0
    val = sum(m[key] * m['num_samples'] for m in metrics.values()) / total_n
    return val, total_n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--walk_forward_dir', required=True)
    ap.add_argument('--folds', type=int, nargs='+', default=None)
    ap.add_argument('--topk_regions', type=int, default=3)
    ap.add_argument('--output_dir', default=None)
    args = ap.parse_args()

    fold_windows = X.get_fold_windows()
    from transformer_260817.scripts.eval.walk_forward import FOLDS
    fold_desc = {f[0]: f[4] for f in FOLDS}

    fold_ckpts = X.discover_fold_checkpoints(args.walk_forward_dir, args.folds)
    if not fold_ckpts:
        raise SystemExit(f"[ERROR] fold チェックポイントが見つかりません: {args.walk_forward_dir}")
    force_print(f"[INFO] 発見したフォールド: {sorted(fold_ckpts)}")

    db_path = get_db_path()
    loss_fn = build_loss_fn(None)
    rows = []

    for fold_id in sorted(fold_ckpts):
        if fold_id not in fold_windows:
            force_print(f"[WARN] fold_{fold_id} は FOLDS 定義に無いためスキップ")
            continue
        train_start, split_date, split_end = fold_windows[fold_id]
        checkpoint = fold_ckpts[fold_id]

        # loader構築前にDBのsplit_type_wf割り当てをこのfoldのtest windowへ揃える
        X.assign_fold_test_window(db_path, train_start, split_date, split_end)
        model, loader, device = X.load_model_and_loader(checkpoint, split='test')
        # load_model_and_loader内のload_config_snapshotがTEMPORAL_SPLIT_DATE等を
        # 上書きするため、loader構築後にもう一度このfoldの割り当てへ揃え直す
        # （DB書き込み自体は冪等な再適用）
        X.assign_fold_test_window(db_path, train_start, split_date, split_end)

        force_print(f"\n===== Fold {fold_id}: {fold_desc.get(fold_id, '')} =====")
        fold_result = {'fold': fold_id, 'era': fold_desc.get(fold_id, '')}
        for hier in (False, True):
            config.USE_HIERARCHICAL_PREDICTION = hier
            config.HIERARCHICAL_TOPK_REGIONS = args.topk_regions
            _, metrics, _, _, _, _ = evaluate(model, loader, loss_fn, (6.0, 10.0))
            pos, n = _weighted(metrics, 'position_hit_rate')
            reg, _ = _weighted(metrics, 'region_hit_rate')
            suffix = 'hier' if hier else 'base'
            fold_result[f'position_hit_{suffix}'] = pos
            fold_result[f'region_hit_{suffix}'] = reg
            fold_result['n'] = n
            force_print(f"  USE_HIERARCHICAL_PREDICTION={hier}: position_hit_rate={pos:.4f}  region_hit_rate={reg:.4f}  n={n}")
        fold_result['position_diff'] = fold_result['position_hit_hier'] - fold_result['position_hit_base']
        rows.append(fold_result)
        del model

    df = pd.DataFrame(rows)
    out_dir = X.make_output_dir('hierarchical_prediction_check', args.output_dir)
    csv_path = X.save_csv(df, out_dir, 'hierarchical_prediction_check.csv')

    overall_base = (df['position_hit_base'] * df['n']).sum() / df['n'].sum()
    overall_hier = (df['position_hit_hier'] * df['n']).sum() / df['n'].sum()
    force_print(f"\n[SUMMARY] Overall position_hit_rate: base={overall_base:.4f}  "
                f"hier(topk_regions={args.topk_regions})={overall_hier:.4f}  "
                f"diff={overall_hier - overall_base:+.4f}pt")
    force_print(f"[INFO] Saved: {csv_path}")


if __name__ == '__main__':
    main()
