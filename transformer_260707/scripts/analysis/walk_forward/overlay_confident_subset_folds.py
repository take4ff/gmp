# --- transformer_260707/scripts/analysis/walk_forward/overlay_confident_subset_folds.py ---
"""各foldの {split}_confident_subset.csv（確信度上位サブセットの hit-rate）を集め、
fold（時代）ごとの coverage-vs-hit_rate 曲線をタスク別の小図に重ね描きする。

確信度に基づく絞り込み（coverage=1.00→0.10）が時代を跨いで同じように効くかを見る。
再学習・再推論不要でDBにも触れないため walk_forward 実行中でも並行実行できる。

出力（1枚 + CSV）:
  confident_subset_trend_by_fold.png  ... タスク別（5小図）coverage-vs-hit_rate、fold別に色分け
  confident_subset_trend_pooled.csv   ... 全fold分の生データ

Usage:
  python -m transformer_260707.scripts.analysis.walk_forward.overlay_confident_subset_folds \
      --walk_forward_dir outputs/transformer_260707/results/walk_forward/<timestamp> \
      --split test
"""
import argparse

import pandas as pd
import matplotlib.pyplot as plt

from transformer_260707.scripts.analysis.xai import _xai_common as X

plt.rcParams['font.family'] = 'Noto Sans CJK JP'

TASK_ORDER = ['region', 'position', 'aa_pos', 'codon_pos', 'synonymous']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--walk_forward_dir', required=True)
    ap.add_argument('--split', default='test', choices=['train', 'valid', 'test'])
    ap.add_argument('--folds', type=int, nargs='+', default=None)
    ap.add_argument('--output_dir', default=None)
    args = ap.parse_args()

    from transformer_260707.scripts.eval.walk_forward import FOLDS
    fold_desc = {f[0]: f[4] for f in FOLDS}

    fold_csvs = X.discover_fold_files(args.walk_forward_dir, f'{args.split}_confident_subset.csv', args.folds)
    if not fold_csvs:
        raise SystemExit(f"[ERROR] {args.split}_confident_subset.csv が見つかりません: {args.walk_forward_dir}")
    print(f"[INFO] 発見したフォールド: {sorted(fold_csvs)}")

    rows = []
    for fold_id, path in fold_csvs.items():
        df = pd.read_csv(path)
        df['fold_id'] = fold_id
        df['era'] = fold_desc.get(fold_id, 'unknown')
        rows.append(df)
    pooled = pd.concat(rows, ignore_index=True)

    out_dir = X.make_output_dir('overlay_confident_subset_folds', args.output_dir)
    X.save_csv(pooled, out_dir, 'confident_subset_trend_pooled.csv')

    folds_sorted = sorted(pooled['fold_id'].unique())
    cmap = plt.get_cmap('viridis', max(len(folds_sorted), 2))

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True)
    axes = axes.flatten()
    for ax, task in zip(axes, TASK_ORDER):
        for i, fid in enumerate(folds_sorted):
            sub = pooled[(pooled['task'] == task) & (pooled['fold_id'] == fid)].sort_values('coverage')
            ax.plot(sub['coverage'], sub['hit_rate_pct'], marker='o', color=cmap(i),
                     label=f"fold{fid}: {fold_desc.get(fid, '')}")
        ax.invert_xaxis()
        ax.set_title(task)
        ax.set_xlabel('Coverage (1.0 = all samples, 0.1 = top 10% confident)')
        ax.set_ylabel('Hit Rate (%)')
        ax.grid(alpha=0.3)
    axes[-1].axis('off')
    handles, labels = axes[0].get_legend_handles_labels()
    axes[-1].legend(handles, labels, title='Fold (era)', fontsize=7, loc='center')
    fig.suptitle(f'Confidence-based coverage curve across folds ({args.split} split)')
    plt.tight_layout()
    X.save_fig(fig, out_dir, 'confident_subset_trend_by_fold.png')

    print(f"\n[INFO] 完了。プロットを {out_dir} に保存しました。")


if __name__ == '__main__':
    main()
