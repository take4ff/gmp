# --- transformer_260707/scripts/analysis/position_tolerance_monthly.py ---
"""変異位置予測の許容誤差(tolerance)付きTop-1正解率を月別・全fold stitchで集計する。

完全一致（tolerance=0、既存の position top1 hit_rate と同じ判定基準）に加え、
|予測位置-正解位置|<=tolerance を正解とみなす複数の tolerance を同時に計算する
（共起正解が複数ある場合は最も近い正解との距離で判定）。

petra.plot_monthly_position_tolerance と対になる月別粒度のCSVを出力し、
plot_position_tolerance_comparison.py で両モデルを重ねてプロットする。

Usage:
  python -m transformer_260707.scripts.analysis.position_tolerance_monthly \
      --walk_forward_dir outputs/transformer_260702/results/walk_forward/<timestamp> \
      --tolerances 0 5 10 50
"""
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

from transformer_260707 import config
from transformer_260707.utils.logging import force_print
from transformer_260707.scripts.analysis import _xai_common as X


@torch.no_grad()
def eval_position_tolerance_by_month(model, loader, tolerances, device):
    model.eval()
    month_hits = {tol: defaultdict(list) for tol in tolerances}
    for (x_cat, x_num, mask), _y, _, _, _, _, _, _, raw_y_batch, *extra in loader:
        x_cat = x_cat.to(device)
        x_num = x_num.to(device)
        mask  = mask.to(device)
        collection_dates = extra[0] if extra else [None] * len(raw_y_batch)

        out_tuple = model(x_cat, x_num, src_key_padding_mask=mask)
        pred_position = out_tuple[1]
        top1 = pred_position.argmax(dim=1).cpu().tolist()

        for i, targets_tuples in enumerate(raw_y_batch):
            if not targets_tuples:
                continue
            t_set = set(t[1] for t in targets_tuples)
            if not t_set:
                continue
            date = collection_dates[i]
            if not date or len(date) < 7:
                continue
            ym = date[:7]
            pred_pos = top1[i]
            min_dist = min(abs(pred_pos - tp) for tp in t_set)
            for tol in tolerances:
                month_hits[tol][ym].append(1.0 if min_dist <= tol else 0.0)
    return month_hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--walk_forward_dir', required=True)
    ap.add_argument('--folds', type=int, nargs='+', default=None)
    ap.add_argument('--tolerances', type=int, nargs='+', default=[0, 5, 10, 50])
    ap.add_argument('--force_cpu', action='store_true')
    ap.add_argument('--output_dir', default=None)
    args = ap.parse_args()

    from transformer_260707.db.connection import get_db_path
    from transformer_260707.db.dataset import create_db_dataloader

    device = 'cpu' if args.force_cpu else config.DEVICE
    db_path = get_db_path()
    fold_windows = X.get_fold_windows()
    fold_ckpts = X.discover_fold_checkpoints(args.walk_forward_dir, args.folds)
    if not fold_ckpts:
        raise SystemExit(f"[ERROR] fold チェックポイントが見つかりません: {args.walk_forward_dir}")
    force_print(f"[INFO] 発見したフォールド: {sorted(fold_ckpts)}  tolerances={args.tolerances}")

    all_month_hits = {tol: defaultdict(list) for tol in args.tolerances}
    per_fold_rows = []

    for fold_id in sorted(fold_ckpts):
        if fold_id not in fold_windows:
            continue
        train_start, split_date, split_end = fold_windows[fold_id]
        force_print(f"\n===== Fold {fold_id}: test window [{split_date}, {split_end}) =====")

        model = X.load_fold_model(fold_ckpts[fold_id], device)
        if getattr(model, 'use_flat_coattn', False):
            force_print(f"[WARN] fold {fold_id} は USE_FLAT_COATTN のためスキップ")
            continue

        X.assign_fold_test_window(db_path, train_start, split_date, split_end)
        loader = create_db_dataloader(
            db_path=db_path, split_type=2,
            batch_size=config.BATCH_SIZE, shuffle=False,
            max_cooccurrence=config.MAX_CO_OCCURRENCE,
        )
        month_hits = eval_position_tolerance_by_month(model, loader, args.tolerances, device)

        base_tol = args.tolerances[0]
        for ym, hits in sorted(month_hits[base_tol].items()):
            for tol in args.tolerances:
                all_month_hits[tol][ym].extend(month_hits[tol][ym])
            row = {'fold': fold_id, 'month': ym, 'n_samples': len(hits)}
            for tol in args.tolerances:
                row[f'hit_rate_tol{tol}'] = float(np.mean(month_hits[tol][ym]) * 100)
            per_fold_rows.append(row)
            force_print(f"  {ym}: n={len(hits):,}  " +
                        "  ".join(f"tol{tol}={row[f'hit_rate_tol{tol}']:.2f}%" for tol in args.tolerances))
        del model

    if not any(all_month_hits[args.tolerances[0]]):
        raise SystemExit("[ERROR] 集計できたフォールドがありません。")

    months = sorted(all_month_hits[args.tolerances[0]].keys())
    rows = []
    for ym in months:
        row = {'month': ym, 'n_samples': len(all_month_hits[args.tolerances[0]][ym])}
        for tol in args.tolerances:
            row[f'hit_rate_tol{tol}'] = float(np.mean(all_month_hits[tol][ym]) * 100)
        rows.append(row)
    df = pd.DataFrame(rows)

    out_dir = args.output_dir or X.make_output_dir('position_tolerance_monthly')
    X.save_csv(df, out_dir, 'position_tolerance_monthly.csv')
    X.save_csv(pd.DataFrame(per_fold_rows), out_dir, 'position_tolerance_monthly_by_fold.csv')
    force_print(f"\n{df.to_string(index=False)}")


if __name__ == '__main__':
    main()
