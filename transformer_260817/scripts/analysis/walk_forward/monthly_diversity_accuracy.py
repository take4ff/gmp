# --- transformer_260817/scripts/analysis/walk_forward/monthly_diversity_accuracy.py ---
"""fold横断で月別の位置予測精度（tolerance=0, top1完全一致）を集計し、
その月のターゲット位置多様度（entropy_norm / unique_ratio / simpson）で
各点を色分けした折れ線グラフを出力する。

position_tolerance_monthly.py の月別集計ロジックを土台に、
utils/plotting.py の _lineage_diversity_rows（系統粒度）と同じ多様度計算式を
月粒度に適用する。fold間で test window が重複しないため、月はそのまま
時系列順に1本の折れ線として fold を跨いで接続できる。

出力（同一ディレクトリに4枚）:
  monthly_accuracy.png                       ... 色分けなし（基本パターン）
  monthly_accuracy_by_entropy.png            ... entropy_norm で色分け
  monthly_accuracy_by_unique_ratio.png       ... unique_ratio で色分け
  monthly_accuracy_by_simpson.png            ... simpson で色分け

Usage:
  python -m transformer_260817.scripts.analysis.walk_forward.monthly_diversity_accuracy \
      --walk_forward_dir outputs/transformer_260817/results/walk_forward/<timestamp>
"""
import argparse
import math
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from transformer_260817 import config
from transformer_260817.utils.logging import force_print
from transformer_260817.scripts.analysis.xai import _xai_common as X

plt.rcParams['font.family'] = 'Noto Sans CJK JP'

DIVERSITY_SPECS = [
    ('entropy_norm',  'Normalized Target-Position Entropy', 'viridis'),
    ('unique_ratio',  'Unique Target-Position Ratio',       'plasma'),
    ('simpson',       'Gini-Simpson Diversity',              'cividis'),
]


@torch.no_grad()
def collect_month_stats(model, loader, device):
    """月ごとに (hit数リスト, ターゲット位置リスト) を集める。"""
    month_hits = defaultdict(list)
    month_positions = defaultdict(list)
    for (x_cat, x_num, mask), _y, _, _, _, _, _, _, raw_y_batch, *extra in loader:
        x_cat = x_cat.to(device)
        x_num = x_num.to(device)
        mask = mask.to(device)
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
            month_hits[ym].append(1.0 if pred_pos in t_set else 0.0)
            month_positions[ym].extend(t_set)
    return month_hits, month_positions


def _diversity_from_positions(pos_list):
    """_lineage_diversity_rows と同じ式で entropy_norm / unique_ratio / simpson を計算する。"""
    if not pos_list:
        return 0.0, 0.0, 0.0
    counts = Counter(pos_list)
    total = len(pos_list)
    probs = [c / total for c in counts.values()]
    entr = -sum(p * math.log2(p) for p in probs)
    max_e = math.log2(len(counts)) if len(counts) > 1 else 1.0
    entropy_norm = entr / max_e
    unique_ratio = len(counts) / total
    simpson = 1.0 - sum(p * p for p in probs)
    return entropy_norm, unique_ratio, simpson


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--walk_forward_dir', required=True)
    ap.add_argument('--folds', type=int, nargs='+', default=None)
    ap.add_argument('--force_cpu', action='store_true')
    ap.add_argument('--output_dir', default=None)
    ap.add_argument('--min_samples', type=int, default=None,
                     help='未指定時は config.DIVERSITY_PLOT_MIN_SAMPLES（既定10）')
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

    all_month_hits = defaultdict(list)
    all_month_positions = defaultdict(list)

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
        month_hits, month_positions = collect_month_stats(model, loader, device)
        for ym, hits in sorted(month_hits.items()):
            all_month_hits[ym].extend(hits)
            all_month_positions[ym].extend(month_positions[ym])
            force_print(f"  {ym}: n={len(hits):,}  hit_rate={np.mean(hits) * 100:.2f}%")
        del model

    if not all_month_hits:
        raise SystemExit("[ERROR] 集計できたフォールドがありません。")

    min_samples = args.min_samples if args.min_samples is not None else \
        getattr(config, 'DIVERSITY_PLOT_MIN_SAMPLES', 10)

    months = sorted(all_month_hits.keys())
    rows = []
    for ym in months:
        hits = all_month_hits[ym]
        n = len(hits)
        if n < min_samples:
            continue
        entropy_norm, unique_ratio, simpson = _diversity_from_positions(all_month_positions[ym])
        rows.append({
            'month': ym, 'n_samples': n,
            'hit_rate_pct': float(np.mean(hits) * 100),
            'entropy_norm': entropy_norm, 'unique_ratio': unique_ratio, 'simpson': simpson,
        })
    df = pd.DataFrame(rows)

    out_dir = X.make_output_dir('monthly_diversity_accuracy', args.output_dir)
    X.save_csv(df, out_dir, 'monthly_diversity_accuracy.csv')
    force_print(f"\n{df.to_string(index=False)}")

    months_plot = df['month'].tolist()
    hit_rates = df['hit_rate_pct'].to_numpy()

    # --- (a) 色分けなし基本パターン ---
    fig, ax = plt.subplots(figsize=(max(14, len(months_plot) * 0.3), 6))
    ax.plot(months_plot, hit_rates, marker='o', color='#2980b9')
    ax.set_ylabel('Position Hit Rate (%, tolerance=0)')
    ax.set_title('Monthly Position Accuracy (folds stitched)')
    ax.grid(alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    X.save_fig(fig, out_dir, 'monthly_accuracy.png')

    # --- (b)(c)(d) 多様度3パターンで色分け ---
    x_idx = np.arange(len(months_plot))
    for col, label, cmap in DIVERSITY_SPECS:
        fig, ax = plt.subplots(figsize=(max(14, len(months_plot) * 0.3), 6))
        ax.plot(x_idx, hit_rates, color='#cccccc', linewidth=1.2, zorder=1)
        sc = ax.scatter(x_idx, hit_rates, c=df[col], cmap=cmap, s=60,
                         edgecolors='black', linewidths=0.4, zorder=2)
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label(label)
        ax.set_xticks(x_idx)
        ax.set_xticklabels(months_plot, rotation=45, ha='right')
        ax.set_ylabel('Position Hit Rate (%, tolerance=0)')
        ax.set_title(f'Monthly Position Accuracy colored by {label} (folds stitched)')
        ax.grid(alpha=0.3)
        plt.tight_layout()
        X.save_fig(fig, out_dir, f'monthly_accuracy_by_{col}.png')

    force_print(f"\n[INFO] 完了。4枚のプロットを {out_dir} に保存しました。")


if __name__ == '__main__':
    main()
