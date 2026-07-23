# --- transformer_260723/scripts/analysis/walk_forward/overlay_diversity_folds.py ---
"""各foldの test_lineage_diversity.csv（utils/plotting.py の3種散布図の裏づけCSV）を集め、
系統粒度の「多様度 vs 位置正解率」散布図を全fold重ね合わせで1枚にまとめる。

fold単体では utils/plotting.plot_entropy_vs_accuracy / plot_labeldiversity_vs_accuracy /
plot_simpson_vs_accuracy が {split}_entropy_vs_accuracy.png 等を系統ファミリー色分けで
出力済み（例: fold_1/.../plots/test_entropy_vs_accuracy.png）。
本スクリプトはそれらの元データ（*_lineage_diversity.csv）を読み直し、色分けをfold単位に
差し替えて全fold重畳版を作る（再学習・再推論不要、DBアクセスもしないため
walk_forward実行中でも並行実行できる）。

出力（3枚）:
  overlay_entropy_norm_by_fold.png
  overlay_unique_ratio_by_fold.png
  overlay_simpson_by_fold.png

Usage:
  python -m transformer_260723.scripts.analysis.walk_forward.overlay_diversity_folds \
      --walk_forward_dir outputs/transformer_260723/results/walk_forward/<timestamp> \
      --split test
"""
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from transformer_260723.scripts.analysis.xai import _xai_common as X

plt.rcParams['font.family'] = 'Noto Sans CJK JP'

METRIC_SPECS = [
    ('entropy_norm', 'Normalized Target-Position Entropy'),
    ('unique_ratio', 'Unique Target-Position Ratio'),
    ('simpson',      'Gini-Simpson Diversity'),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--walk_forward_dir', required=True)
    ap.add_argument('--split', default='test', choices=['train', 'valid', 'test'])
    ap.add_argument('--folds', type=int, nargs='+', default=None)
    ap.add_argument('--output_dir', default=None)
    args = ap.parse_args()

    from transformer_260723.scripts.eval.walk_forward import FOLDS
    fold_desc = {f[0]: f[4] for f in FOLDS}

    fold_csvs = X.discover_fold_files(args.walk_forward_dir, f'{args.split}_lineage_diversity.csv', args.folds)
    if not fold_csvs:
        raise SystemExit(f"[ERROR] {args.split}_lineage_diversity.csv が見つかりません: {args.walk_forward_dir}")
    print(f"[INFO] 発見したフォールド: {sorted(fold_csvs)}")

    rows = []
    for fold_id, path in fold_csvs.items():
        df = pd.read_csv(path)
        df['fold_id'] = fold_id
        df['era'] = fold_desc.get(fold_id, 'unknown')
        rows.append(df)
    pooled = pd.concat(rows, ignore_index=True)

    out_dir = X.make_output_dir('overlay_diversity_folds', args.output_dir)
    X.save_csv(pooled, out_dir, 'overlay_diversity_pooled.csv')

    folds_sorted = sorted(pooled['fold_id'].unique())
    cmap = plt.get_cmap('viridis', max(len(folds_sorted), 2))

    for col, label in METRIC_SPECS:
        fig, ax = plt.subplots(figsize=(9, 6))
        for i, fid in enumerate(folds_sorted):
            sub = pooled[pooled['fold_id'] == fid]
            log_n = np.log1p(sub['n'])
            sizes = 20 + (log_n - np.log1p(pooled['n']).min()) / \
                (np.log1p(pooled['n']).max() - np.log1p(pooled['n']).min() + 1e-9) * 400
            ax.scatter(sub[col], sub['hit_rate'], s=sizes, alpha=0.65, color=cmap(i),
                       edgecolors='gray', linewidths=0.3,
                       label=f"fold{fid}: {fold_desc.get(fid, '')}")
        ax.set_xlabel(label)
        ax.set_ylabel('Position Hit Rate (%)')
        ax.set_title(f'{label} vs Position Hit Rate — all folds overlaid\n'
                     f'(per-lineage, {args.split} split, point size ∝ log(n_samples))')
        ax.legend(fontsize=7, loc='best')
        ax.grid(linestyle='--', alpha=0.35)
        plt.tight_layout()
        X.save_fig(fig, out_dir, f'overlay_{col}_by_fold.png')

    print(f"\n[INFO] 完了。3枚のプロットを {out_dir} に保存しました。")


if __name__ == '__main__':
    main()
