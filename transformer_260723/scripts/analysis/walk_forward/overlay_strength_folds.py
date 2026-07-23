# --- transformer_260723/scripts/analysis/walk_forward/overlay_strength_folds.py ---
"""各foldの {split}_strength_fine.csv（流行度カテゴリ別精度）を集め、
カテゴリごとに fold（時代）を跨いだ精度トレンドをタスク別の小図にまとめる。

流行度カテゴリは NCBI 出現数ベースの9段階ビン（~100 〜 >500K、evaluate.py 側で決定）。
再学習・再推論不要でDBにも触れないため walk_forward 実行中でも並行実行できる。

出力（1枚 + CSV）:
  strength_category_trend_by_fold.png  ... タスク別（5小図）カテゴリ別 hit-rate の fold横断トレンド
  strength_trend_pooled.csv            ... 全fold・全カテゴリ分の生データ

Usage:
  python -m transformer_260723.scripts.analysis.walk_forward.overlay_strength_folds \
      --walk_forward_dir outputs/transformer_260723/results/walk_forward/<timestamp> \
      --split test
"""
import argparse

import pandas as pd
import matplotlib.pyplot as plt

from transformer_260723.scripts.analysis.xai import _xai_common as X

plt.rcParams['font.family'] = 'Noto Sans CJK JP'

TASK_COLS = ['region_hit_rate_pct', 'position_hit_rate_pct', 'aa_pos_hit_rate_pct',
             'codon_pos_hit_rate_pct', 'synonymous_hit_rate_pct']
# evaluate.py が生成する固定の流行度ビン順（低頻度→高頻度）
CATEGORY_ORDER = ['~100', '~500', '~1K', '~5K', '~10K', '~50K', '~100K', '~500K', '>500K']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--walk_forward_dir', required=True)
    ap.add_argument('--split', default='test', choices=['train', 'valid', 'test'])
    ap.add_argument('--folds', type=int, nargs='+', default=None)
    ap.add_argument('--output_dir', default=None)
    args = ap.parse_args()

    from transformer_260723.scripts.eval.walk_forward import FOLDS
    fold_desc = {f[0]: f[4] for f in FOLDS}

    fold_csvs = X.discover_fold_files(args.walk_forward_dir, f'{args.split}_strength_fine.csv', args.folds)
    if not fold_csvs:
        raise SystemExit(f"[ERROR] {args.split}_strength_fine.csv が見つかりません: {args.walk_forward_dir}")
    print(f"[INFO] 発見したフォールド: {sorted(fold_csvs)}")

    rows = []
    for fold_id, path in fold_csvs.items():
        df = pd.read_csv(path)
        df['fold_id'] = fold_id
        df['era'] = fold_desc.get(fold_id, 'unknown')
        rows.append(df)
    pooled = pd.concat(rows, ignore_index=True)

    out_dir = X.make_output_dir('overlay_strength_folds', args.output_dir)
    X.save_csv(pooled, out_dir, 'strength_trend_pooled.csv')

    folds_sorted = sorted(pooled['fold_id'].unique())
    xtick_labels = [f"fold{f}\n{fold_desc.get(f, '')}" for f in folds_sorted]
    categories = [c for c in CATEGORY_ORDER if c in set(pooled['strength_category'])]
    cmap = plt.get_cmap('plasma', max(len(categories), 2))

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True)
    axes = axes.flatten()
    for ax, col in zip(axes, TASK_COLS):
        for i, cat in enumerate(categories):
            sub = pooled[pooled['strength_category'] == cat].set_index('fold_id').reindex(folds_sorted)
            ax.plot(range(len(folds_sorted)), sub[col], marker='o', color=cmap(i), label=cat)
        ax.set_title(col.replace('_hit_rate_pct', ''))
        ax.set_xticks(range(len(folds_sorted)))
        ax.set_xticklabels(xtick_labels, fontsize=7, rotation=45, ha='right')
        ax.set_ylabel('Hit Rate (%)')
        ax.grid(alpha=0.3)
    axes[-1].axis('off')
    handles, labels = axes[0].get_legend_handles_labels()
    axes[-1].legend(handles, labels, title='Strength category\n(prevalence, low→high)',
                     fontsize=8, loc='center')
    fig.suptitle(f'Strength-category accuracy trend across folds ({args.split} split)')
    plt.tight_layout()
    X.save_fig(fig, out_dir, 'strength_category_trend_by_fold.png')

    print(f"\n[INFO] 完了。プロットを {out_dir} に保存しました。")


if __name__ == '__main__':
    main()
