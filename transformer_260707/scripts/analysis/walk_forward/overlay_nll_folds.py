# --- transformer_260707/scripts/analysis/walk_forward/overlay_nll_folds.py ---
"""各foldの {split}_nll.csv（タスク別 held-out NLL）を集め、時代（fold）を跨いだ
予測難易度トレンドを1枚の折れ線グラフにまとめる。

各foldは学習・評価の副産物として {split}_nll.csv（task, n_samples, nll_nats, nll_bits, perplexity）
を出力済みのため、再学習・再推論不要でDBにも触れず、walk_forward実行中でも並行実行できる。

出力（1枚 + CSV）:
  nll_trend_by_fold.png    ... タスク別（5小図）NLL[nats] の fold 横断トレンド
  nll_trend_pooled.csv     ... 全fold分の生データ

Usage:
  python -m transformer_260707.scripts.analysis.walk_forward.overlay_nll_folds \
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

    fold_csvs = X.discover_fold_files(args.walk_forward_dir, f'{args.split}_nll.csv', args.folds)
    if not fold_csvs:
        raise SystemExit(f"[ERROR] {args.split}_nll.csv が見つかりません: {args.walk_forward_dir}")
    print(f"[INFO] 発見したフォールド: {sorted(fold_csvs)}")

    rows = []
    for fold_id, path in fold_csvs.items():
        df = pd.read_csv(path)
        df['fold_id'] = fold_id
        df['era'] = fold_desc.get(fold_id, 'unknown')
        rows.append(df)
    pooled = pd.concat(rows, ignore_index=True)

    out_dir = X.make_output_dir('overlay_nll_folds', args.output_dir)
    X.save_csv(pooled, out_dir, 'nll_trend_pooled.csv')

    folds_sorted = sorted(pooled['fold_id'].unique())
    xtick_labels = [f"fold{f}\n{fold_desc.get(f, '')}" for f in folds_sorted]

    fig, axes = plt.subplots(1, len(TASK_ORDER), figsize=(4 * len(TASK_ORDER), 5), sharex=True)
    for ax, task in zip(axes, TASK_ORDER):
        sub = pooled[pooled['task'] == task].set_index('fold_id').reindex(folds_sorted)
        ax.plot(range(len(folds_sorted)), sub['nll_nats'], marker='o', color='#c0392b')
        ax.set_title(task)
        ax.set_xticks(range(len(folds_sorted)))
        ax.set_xticklabels(xtick_labels, fontsize=7, rotation=45, ha='right')
        ax.set_ylabel('NLL (nats)')
        ax.grid(alpha=0.3)
    fig.suptitle(f'Held-out NLL trend across folds ({args.split} split)')
    plt.tight_layout()
    X.save_fig(fig, out_dir, 'nll_trend_by_fold.png')

    print(f"\n[INFO] 完了。プロットを {out_dir} に保存しました。")


if __name__ == '__main__':
    main()
