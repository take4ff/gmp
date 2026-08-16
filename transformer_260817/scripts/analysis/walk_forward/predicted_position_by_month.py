# --- transformer_260817/scripts/analysis/walk_forward/predicted_position_by_month.py ---
# モデルのTop-1予測位置（NucPos）の分布を月別に集計し、ヒートマップとして可視化する。
# scripts/visualization/plot_target_position_by_month{,_normalized}.py の「正解データ」版に対する
# 「モデル予測」版。両者を並べることで、モデルが月ごとの実際のホットスポット変化を追えているか、
# それとも常に同じ位置ばかり予測する偏り（mode collapse）があるかを視覚的に比較できる。
#
# walk_forward の各foldチェックポイントを読み込み、そのfoldのtest windowで推論するため
# position_tolerance_monthly.py と同じ構造（要checkpoint再読込・DB使用）。
# 同一walk_forward_dirに対しDBの split_type_wf を書き換えるため、他のwalk_forward関連
# プロセスと同時実行しないこと。
#
# 出力（2枚 + CSV）:
#   predicted_position_by_month.png            ... 全期間通算Top100予測位置（生カウント）
#   predicted_position_by_month_normalized.png ... 各月Top10予測位置の和集合、月の総サンプル数に対する%
#
# Usage:
#   python -m transformer_260817.scripts.analysis.walk_forward.predicted_position_by_month \
#       --walk_forward_dir outputs/transformer_260817/results/walk_forward/<timestamp>
import argparse
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from transformer_260817 import config
from transformer_260817.utils.logging import force_print
from transformer_260817.scripts.analysis.xai import _xai_common as X

TOP_N_GLOBAL = 100
TOP_N_PER_MONTH = 10


@torch.no_grad()
def collect_month_pred_counts(model, loader, device):
    """月ごとに Top-1 予測位置の Counter と評価サンプル数を集める。"""
    month_counts: dict[str, Counter] = defaultdict(Counter)
    month_n: dict[str, int] = defaultdict(int)
    for (x_cat, x_num, mask), _y, _, _, _, _, _, _, raw_y_batch, *extra in loader:
        x_cat = x_cat.to(device)
        x_num = x_num.to(device)
        mask = mask.to(device)
        collection_dates = extra[0] if extra else [None] * len(raw_y_batch)

        out_tuple = model(x_cat, x_num, src_key_padding_mask=mask)
        pred_position = out_tuple[1]
        top1 = pred_position.argmax(dim=1).cpu().tolist()

        for i, date in enumerate(collection_dates):
            if not date or len(date) < 7:
                continue
            ym = date[:7]
            month_counts[ym][top1[i]] += 1
            month_n[ym] += 1
    return month_counts, month_n


def _plot_heatmap(matrix, row_labels, months, title, cbar_label, tick_fmt, out_path):
    n_pos, n_months = matrix.shape
    fig_w = max(18, n_months * 0.38)
    fig_h = max(14, n_pos * 0.19)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    log_matrix = np.log1p(matrix)
    im = ax.imshow(log_matrix, aspect='auto', cmap='YlGnBu', interpolation='nearest')

    cbar = plt.colorbar(im, ax=ax, pad=0.01, fraction=0.02)
    cbar.set_label(cbar_label, fontsize=9)
    max_val = float(matrix.max())
    raw_ticks = [v for v in tick_fmt if v <= max_val]
    if max_val not in raw_ticks:
        raw_ticks.append(round(max_val, 2))
    cbar.set_ticks([np.log1p(v) for v in raw_ticks])
    cbar.set_ticklabels([f'{v:g}' for v in raw_ticks], fontsize=7)

    ax.set_xticks(range(n_months))
    ax.set_xticklabels(months, rotation=45, ha='right', fontsize=7)
    ax.set_yticks(range(n_pos))
    ax.set_yticklabels([str(p) for p in row_labels], fontsize=6)
    ax.set_xlabel('Month')
    ax.set_ylabel('Predicted NucPos (top-1)')
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    force_print(f"[INFO] Saved: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--walk_forward_dir', required=True)
    ap.add_argument('--folds', type=int, nargs='+', default=None)
    ap.add_argument('--force_cpu', action='store_true')
    ap.add_argument('--output_dir', default=None)
    args = ap.parse_args()

    from transformer_260817.db.connection import get_db_path
    from transformer_260817.db.dataset import create_db_dataloader

    device = 'cpu' if args.force_cpu else config.DEVICE
    db_path = get_db_path()
    fold_windows = X.get_fold_windows()
    fold_ckpts = X.discover_fold_checkpoints(args.walk_forward_dir, args.folds)
    if not fold_ckpts:
        raise SystemExit(f"[ERROR] fold チェックポイントが見つかりません: {args.walk_forward_dir}")
    force_print(f"[INFO] 発見したフォールド: {sorted(fold_ckpts)}")

    all_month_counts: dict[str, Counter] = defaultdict(Counter)
    all_month_n: dict[str, int] = defaultdict(int)

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
        month_counts, month_n = collect_month_pred_counts(model, loader, device)
        for ym, cnt in month_counts.items():
            all_month_counts[ym].update(cnt)
            all_month_n[ym] += month_n[ym]
            force_print(f"  {ym}: n={month_n[ym]:,}")
        del model

    if not all_month_counts:
        raise SystemExit("[ERROR] 集計できたフォールドがありません。")

    months = sorted(all_month_counts.keys())
    out_dir = args.output_dir or X.make_output_dir('predicted_position_by_month')

    # CSV（生データ: month, position, count）
    rows = [{'month': m, 'position': p, 'count': c, 'n_samples': all_month_n[m]}
            for m in months for p, c in all_month_counts[m].items()]
    X.save_csv(pd.DataFrame(rows), out_dir, 'predicted_position_by_month.csv')

    # --- (a) 全期間通算Top100（生カウント） ---
    pos_total: Counter = Counter()
    for cnt in all_month_counts.values():
        pos_total.update(cnt)
    top_global = sorted(pos_total, key=lambda p: pos_total[p], reverse=True)[:TOP_N_GLOBAL]
    top_global_sorted = sorted(top_global)

    month_idx = {m: i for i, m in enumerate(months)}
    pos_idx = {p: i for i, p in enumerate(top_global_sorted)}
    matrix_raw = np.zeros((len(top_global_sorted), len(months)), dtype=np.float64)
    for m, cnt in all_month_counts.items():
        mi = month_idx[m]
        for p, c in cnt.items():
            pi = pos_idx.get(p)
            if pi is not None:
                matrix_raw[pi, mi] = c
    _plot_heatmap(
        matrix_raw, top_global_sorted, months,
        title=f'Predicted position (top-1) sample count by month (top {TOP_N_GLOBAL} positions)',
        cbar_label='Predicted count (log scale)',
        tick_fmt=[0, 10, 100, 1000, 10000, 100000],
        out_path=f'{out_dir}/predicted_position_by_month.png',
    )

    # --- (b) 各月Top10予測位置の和集合、月の総サンプル数に対する% ---
    selected: set[int] = set()
    for m, cnt in all_month_counts.items():
        selected.update(p for p, _ in cnt.most_common(TOP_N_PER_MONTH))
    sel_sorted = sorted(selected)
    pos_idx2 = {p: i for i, p in enumerate(sel_sorted)}
    matrix_pct = np.zeros((len(sel_sorted), len(months)), dtype=np.float64)
    for m, cnt in all_month_counts.items():
        mi = month_idx[m]
        total = all_month_n[m]
        if total <= 0:
            continue
        for p, c in cnt.items():
            pi = pos_idx2.get(p)
            if pi is not None:
                matrix_pct[pi, mi] = c / total * 100.0
    force_print(f"[INFO] {len(sel_sorted)} unique positions selected "
                f"(union of top {TOP_N_PER_MONTH} predicted per month)")
    _plot_heatmap(
        matrix_pct, sel_sorted, months,
        title=f'% of month\'s samples by predicted NucPos (top-1)\n'
              f'(rows = union of each month\'s top {TOP_N_PER_MONTH} predicted positions, '
              f'{len(sel_sorted)} positions total)',
        cbar_label='% of month\'s samples (log scale)',
        tick_fmt=[0, 0.1, 1, 5, 10, 25, 50, 75, 100],
        out_path=f'{out_dir}/predicted_position_by_month_normalized.png',
    )

    force_print(f"\n[INFO] 完了。出力先: {out_dir}")


if __name__ == '__main__':
    main()
