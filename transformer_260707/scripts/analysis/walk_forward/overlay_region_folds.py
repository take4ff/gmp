# --- transformer_260707/scripts/analysis/walk_forward/overlay_region_folds.py ---
"""各foldの {split}_region_metrics.csv（遺伝子/region別 precision/recall/F1）を集め、
サンプル数上位の遺伝子について fold（時代）を跨いだ recall トレンドを1枚にまとめる。

再学習・再推論不要でDBにも触れないため walk_forward 実行中でも並行実行できる。
region数が多い（~37）ため、全fold合計サンプル数の上位 --top_n 遺伝子のみを線で表示し、
残りは「その他」として集計に含めず除外する（凡例が煩雑になるのを防ぐ。全遺伝子の生データは
CSVに保持）。

出力（1枚 + CSV）:
  region_recall_trend_by_fold.png  ... 上位遺伝子の recall_pct fold横断トレンド
  region_trend_pooled.csv          ... 全fold・全region分の生データ

Usage:
  python -m transformer_260707.scripts.analysis.walk_forward.overlay_region_folds \
      --walk_forward_dir outputs/transformer_260707/results/walk_forward/<timestamp> \
      --split test --top_n 12
"""
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from transformer_260707.scripts.analysis.xai import _xai_common as X

plt.rcParams['font.family'] = 'Noto Sans CJK JP'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--walk_forward_dir', required=True)
    ap.add_argument('--split', default='test', choices=['train', 'valid', 'test'])
    ap.add_argument('--folds', type=int, nargs='+', default=None)
    ap.add_argument('--top_n', type=int, default=12, help='凡例に表示する上位遺伝子数')
    ap.add_argument('--output_dir', default=None)
    args = ap.parse_args()

    from transformer_260707.scripts.eval.walk_forward import FOLDS
    fold_desc = {f[0]: f[4] for f in FOLDS}

    fold_csvs = X.discover_fold_files(args.walk_forward_dir, f'{args.split}_region_metrics.csv', args.folds)
    if not fold_csvs:
        raise SystemExit(f"[ERROR] {args.split}_region_metrics.csv が見つかりません: {args.walk_forward_dir}")
    print(f"[INFO] 発見したフォールド: {sorted(fold_csvs)}")

    rows = []
    for fold_id, path in fold_csvs.items():
        df = pd.read_csv(path)
        df['fold_id'] = fold_id
        df['era'] = fold_desc.get(fold_id, 'unknown')
        rows.append(df)
    pooled = pd.concat(rows, ignore_index=True)

    out_dir = X.make_output_dir('overlay_region_folds', args.output_dir)
    X.save_csv(pooled, out_dir, 'region_trend_pooled.csv')

    folds_sorted = sorted(pooled['fold_id'].unique())
    xtick_labels = [f"fold{f}\n{fold_desc.get(f, '')}" for f in folds_sorted]

    top_regions = (pooled.groupby('region_name')['sample_count'].sum()
                   .sort_values(ascending=False).head(args.top_n).index.tolist())

    cmap = plt.get_cmap('tab20', max(len(top_regions), 2))
    fig, ax = plt.subplots(figsize=(max(12, len(folds_sorted) * 1.8), 7))
    for i, region in enumerate(top_regions):
        sub = pooled[pooled['region_name'] == region].set_index('fold_id').reindex(folds_sorted)
        ax.plot(range(len(folds_sorted)), sub['recall_pct'], marker='o', color=cmap(i), label=region)

    ax.set_xticks(range(len(folds_sorted)))
    ax.set_xticklabels(xtick_labels, fontsize=8, rotation=45, ha='right')
    ax.set_ylabel('Recall (%)')
    ax.set_title(f'Region-level recall trend across folds ({args.split} split)\n'
                 f'Top {len(top_regions)} genes by total sample count')
    ax.legend(fontsize=7, ncol=2, loc='best')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    X.save_fig(fig, out_dir, 'region_recall_trend_by_fold.png')

    print(f"\n[INFO] 完了。プロットを {out_dir} に保存しました。")


if __name__ == '__main__':
    main()
