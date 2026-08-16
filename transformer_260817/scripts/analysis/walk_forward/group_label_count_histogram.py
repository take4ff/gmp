# --- transformer_260817/scripts/analysis/walk_forward/group_label_count_histogram.py ---
"""共起グループ内のユニークラベル数（any-of-set評価でグループ全メンバーの
ターゲットを合算した際のユニーク数）のヒストグラムをfold別に色分けして描く。

evaluate_topk()のR-Precision（動的K=len(target_set)）はバッチ内の最大値を
そのバッチ全サンプル共通のneed_kとして使うため、巨大な共起グループが1つ
バッチに混入するだけでtorch.topk(k=need_k)がバッチ全体に対して走り、
OOMの引き金になった（walk_forward fold_3、Omicron早期BA.1/BA.2で実測、
2026-07-30。config.MAX_R_PRECISION_Kで対策済み）。本スクリプトはこの
グループサイズ分布がfold間でどう異なるかを可視化する診断ツール。

_xai_common.assign_fold_test_window()（軽量版: split_type_wfの割り当てのみ、
USE_UNIQUE_FILTER等は無効化）を使うため、実際の学習/評価パイプラインとは
サンプル集合が厳密には一致しない（他のwalk-forward分析スクリプト群と同じ
制約）。相対比較・傾向把握が目的。

Usage:
  python -m transformer_260817.scripts.analysis.walk_forward.group_label_count_histogram \
      [--folds 2 3] [--task position]
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from transformer_260817 import config
from transformer_260817.db.connection import get_db_path
from transformer_260817.db.dataset import DBIterableDataset
from transformer_260817.scripts.analysis.xai import _xai_common as X

plt.rcParams['font.family'] = 'Noto Sans CJK JP'

# dataviz skill 検証済みカテゴリカルパレット（8色中7色、fold1〜7に固定順で割当）
_FOLD_COLORS = {
    1: '#2a78d6',  # blue
    2: '#eb6834',  # orange
    3: '#1baf7a',  # aqua
    4: '#eda100',  # yellow
    5: '#e87ba4',  # magenta
    6: '#008300',  # green
    7: '#4a3aa7',  # violet
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', type=int, nargs='+', default=None,
                     help='対象フォールド番号（省略時は全7fold）')
    ap.add_argument('--task', default='position',
                     choices=['region', 'position', 'aa_pos', 'codon_pos', 'synonymous'],
                     help='ラベルをカウントするタスク（既定: position、R-Precision OOMの直接原因）')
    ap.add_argument('--output_dir', default=None)
    args = ap.parse_args()

    from transformer_260817.scripts.eval.walk_forward import FOLDS
    fold_windows = X.get_fold_windows()
    fold_desc = {f[0]: f[4] for f in FOLDS}
    target_idx = {'region': 0, 'position': 1, 'aa_pos': 2, 'codon_pos': 3, 'synonymous': 4}
    tidx = target_idx[args.task]

    db_path = get_db_path()
    folds = args.folds or sorted(fold_windows.keys())

    per_fold_counts = {}
    for fold_id in folds:
        if fold_id not in fold_windows:
            print(f"[WARN] fold_{fold_id} は FOLDS 定義に無いためスキップ")
            continue
        train_start, split_date, split_end = fold_windows[fold_id]
        print(f"[INFO] Fold {fold_id}: assigning test window [{split_date}, {split_end})...")
        X.assign_fold_test_window(db_path, train_start, split_date, split_end)

        ds = DBIterableDataset(db_path=db_path, split_type=2)
        counts = []
        for _repr_id, (_y_repr, group_targets_list) in ds._group_label_cache.items():
            labels = set()
            for member_targets in group_targets_list:
                for t in member_targets:
                    labels.add(t[tidx])
            counts.append(len(labels))
        per_fold_counts[fold_id] = counts
        if counts:
            print(f"[INFO] Fold {fold_id} ({fold_desc.get(fold_id, '')}): {len(counts):,} groups, "
                  f"max={max(counts):,}, mean={np.mean(counts):.2f}")
        del ds

    output_dir = os.path.join(config.OUTPUT_DIR, 'scripts', 'group_label_count_histogram') \
        if args.output_dir is None else args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # CSV保存（生データ。fold横断の再集計・別ビン幅での再描画用）
    rows = [{'fold_id': fold_id, 'label_count': c}
            for fold_id, counts in per_fold_counts.items() for c in counts]
    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, f'group_label_count_{args.task}.csv')
    df.to_csv(csv_path, index=False)
    print(f"[INFO] Saved: {csv_path}")

    # --- プロット: 対数ビンのヒストグラム（step、foldごとに色分け） ---
    all_counts = [c for counts in per_fold_counts.values() for c in counts]
    if not all_counts:
        print("[WARN] No groups found across the selected folds.")
        return
    max_count = max(all_counts)
    bins = np.logspace(0, np.log10(max_count + 1), 50)

    fig, ax = plt.subplots(figsize=(12, 7))
    for fold_id in folds:
        counts = per_fold_counts.get(fold_id)
        if not counts:
            continue
        ax.hist(counts, bins=bins, histtype='step', linewidth=2,
                color=_FOLD_COLORS.get(fold_id, '#888888'),
                label=f'Fold {fold_id}: {fold_desc.get(fold_id, "")}')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(f'グループ内ユニークラベル数（task={args.task}）')
    ax.set_ylabel('グループ数（log scale）')
    ax.set_title('共起グループ内ラベル数の分布（fold別）')
    ax.legend(loc='upper right', fontsize=8, frameon=False)
    ax.grid(axis='both', alpha=0.3, linewidth=0.5)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    png_path = os.path.join(output_dir, f'group_label_count_histogram_{args.task}.png')
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved: {png_path}")


if __name__ == '__main__':
    main()
