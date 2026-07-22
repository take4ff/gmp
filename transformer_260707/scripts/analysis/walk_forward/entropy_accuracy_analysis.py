# --- transformer_260707/scripts/analysis/walk_forward/entropy_accuracy_analysis.py ---
"""系統別ターゲットエントロピーと精度の関係を分析し、Fano不等式で理論的達成可能精度を導く。

背景: abstractの「オミクロン株は変異が多様なため塩基配列位置精度が10%に低下」という記述を
定量的に裏付けるため、以下2点を計算する。
  (1) 系統別の正規化ターゲット位置エントロピー(db_target_entropy_norm) と 各タスク精度の相関
      （評価のたびに utils/io.py の save_lineage_metrics_csv が自動出力する既存データを利用、
      新規学習・推論は不要）。
  (2) Fano不等式による理論的達成可能精度の上限:
        H(X|Y) <= H_b(Pe) + Pe*log2(M-1)
      をPeについて解き、観測エントロピー H(X|Y)（系統別 db_target_entropy を近似として使用）から
      理論的に達成可能な最小誤り率（=最大精度）を導く。
      注意: db_target_entropy は「系統が分かっている」という条件のみでのエントロピーであり、
      モデルの実際の入力（変異履歴全体）で条件づけた真の H(X|Y) の上界（緩い側）に留まる。
      したがってここで導く精度上限は「これより下がることはない」という安全側（緩め）の値である。

Usage:
    python -m transformer_260707.scripts.analysis.walk_forward.entropy_accuracy_analysis \\
        --walk_forward_dir outputs/transformer_260707/results/walk_forward/<timestamp>
"""
import argparse
import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from scipy.optimize import brentq

plt.rcParams['font.family'] = 'Noto Sans CJK JP'

from transformer_260707 import config
from transformer_260707.utils.logging import force_print
from transformer_260707.scripts.analysis.xai import _xai_common as X

TASK_COLS = ['region_hit_rate_pct', 'position_hit_rate_pct', 'aa_pos_hit_rate_pct',
             'codon_pos_hit_rate_pct', 'synonymous_hit_rate_pct']


def fano_min_error_rate(h_bits: float, vocab_size: int) -> float:
    """Fano不等式 H(X|Y) <= H_b(Pe) + Pe*log2(M-1) を満たす最小のPeを二分探索で求める。

    右辺 f(pe) = H_b(pe) + pe*log2(M-1) は pe*=1-1/M で最大値log2(M)を取り、
    そこから先は減少に転じる（M=2だとlog2(M-1)=0のため特に顕著）。求めたいのは
    「達成可能な最小のPe」＝f(0)<h_bits<f(pe*)となる区間内の小さい方の解なので、
    探索区間は [0, pe*] に限定する（[0, 1]全体だと、M=2等でf(0)とf(1)が両方とも
    h_bits未満になり得て符号が反転せずbrentqが失敗し、常にpe=1-1/Mへフォール
    バックしてしまうバグがあった。Synonymous(M=2)タスクでFano上限が実測エントロピー
    に関わらず常に50%固定になっていたのはこれが原因）。
    """
    M = vocab_size
    if M <= 1 or h_bits <= 0:
        return 0.0
    max_h = np.log2(M)
    if h_bits >= max_h:
        return 1.0 - 1.0 / M

    def gap(pe):
        if pe <= 0.0 or pe >= 1.0:
            hb = 0.0
        else:
            hb = -pe * np.log2(pe) - (1 - pe) * np.log2(1 - pe)
        return hb + pe * np.log2(max(M - 1, 1)) - h_bits

    pe_star = 1.0 - 1.0 / M
    try:
        return brentq(gap, 1e-12, pe_star - 1e-12)
    except ValueError:
        return 1.0 - 1.0 / M


def _discover_fold_csvs(walk_forward_dir, split):
    """fold_N/*/csv/{split}_lineage_metrics.csv を探索する。"""
    found = {}
    fname = f'{split}_lineage_metrics.csv'
    for root, _dirs, files in os.walk(walk_forward_dir):
        if fname in files:
            m = re.search(r'fold_(\d+)', root)
            if m:
                found[int(m.group(1))] = os.path.join(root, fname)
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--walk_forward_dir', required=True,
                    help='fold_N/*/csv/{split}_lineage_metrics.csv を含む walk_forward 実行ディレクトリ')
    ap.add_argument('--split', default='test', choices=['train', 'valid', 'test'])
    ap.add_argument('--output_dir', default=None)
    args = ap.parse_args()

    from transformer_260707.scripts.eval.walk_forward import FOLDS
    fold_desc = {f[0]: f[4] for f in FOLDS}

    fold_csvs = _discover_fold_csvs(args.walk_forward_dir, args.split)
    if not fold_csvs:
        raise SystemExit(f"[ERROR] {args.split}_lineage_metrics.csv が見つかりません: {args.walk_forward_dir}")
    force_print(f"[INFO] 発見したフォールド: {sorted(fold_csvs)}")

    pooled_rows = []
    fold_summaries = []
    for fold_id in sorted(fold_csvs):
        df = pd.read_csv(fold_csvs[fold_id])
        df['fold_id'] = fold_id
        df['era'] = fold_desc.get(fold_id, 'unknown')

        # Fano: 位置予測の理論的達成可能精度上限（系統別 db_target_entropy[bits] を使用）
        df['fano_max_position_accuracy_pct'] = df['db_target_entropy'].apply(
            lambda h: (1 - fano_min_error_rate(h, config.VOCAB_SIZE_POSITION)) * 100
        )
        pooled_rows.append(df)

        w = df['num_samples']
        summary = {
            'fold_id': fold_id, 'era': fold_desc.get(fold_id, 'unknown'),
            'n_lineages': len(df), 'n_samples': int(w.sum()),
            'weighted_mean_entropy_norm': float(np.average(df['db_target_entropy_norm'], weights=w)),
            'weighted_mean_fano_max_position_accuracy_pct':
                float(np.average(df['fano_max_position_accuracy_pct'], weights=w)),
        }
        for col in TASK_COLS:
            summary[f'weighted_mean_{col}'] = float(np.average(df[col], weights=w))
        fold_summaries.append(summary)
        force_print(f"[INFO] fold{fold_id} ({fold_desc.get(fold_id,'')}): "
                    f"entropy_norm={summary['weighted_mean_entropy_norm']:.3f}, "
                    f"position_hit={summary['weighted_mean_position_hit_rate_pct']:.2f}%, "
                    f"fano_ceiling={summary['weighted_mean_fano_max_position_accuracy_pct']:.2f}%")

    pooled = pd.concat(pooled_rows, ignore_index=True)

    # 相関: エントロピー(正規化) vs 各タスク精度（系統粒度でプール）
    correlations = {}
    for col in TASK_COLS:
        valid = pooled[['db_target_entropy_norm', col]].dropna()
        rho, pval = spearmanr(valid['db_target_entropy_norm'], valid[col])
        correlations[col] = {'spearman_r': float(rho), 'spearman_p': float(pval), 'n': len(valid)}
        force_print(f"[INFO] entropy_norm vs {col}: Spearman r={rho:.3f}, p={pval:.2e} (n={len(valid)})")

    pooled['fano_gap_pct'] = pooled['fano_max_position_accuracy_pct'] - pooled['position_hit_rate_pct']

    out_dir = X.make_output_dir('entropy_accuracy', args.output_dir)
    X.save_csv(pd.DataFrame(fold_summaries), out_dir, 'entropy_accuracy_by_fold.csv')
    X.save_csv(pooled, out_dir, 'entropy_accuracy_pooled_lineages.csv')

    # --- プロット1: entropy_norm vs position_hit_rate_pct（foldごとに色分け） ---
    fig, ax = plt.subplots(figsize=(8, 6))
    folds_sorted = sorted(pooled['fold_id'].unique())
    cmap = plt.get_cmap('viridis', max(len(folds_sorted), 2))
    for i, fid in enumerate(folds_sorted):
        sub = pooled[pooled['fold_id'] == fid]
        sizes = np.clip(sub['num_samples'] / max(sub['num_samples'].max(), 1) * 200, 5, 200)
        ax.scatter(sub['db_target_entropy_norm'], sub['position_hit_rate_pct'],
                   s=sizes, alpha=0.6, color=cmap(i), label=f"fold{fid}: {fold_desc.get(fid, '')}")
    ax.set_xlabel('Normalized target-position entropy (per lineage)')
    ax.set_ylabel('Position hit-rate (%)')
    ax.set_title('Entropy vs position accuracy (bubble size = #samples)')
    ax.legend(fontsize=7, loc='best')
    plt.tight_layout()
    X.save_fig(fig, out_dir, 'entropy_vs_position_accuracy.png')

    # --- プロット2: Fano理論上限 vs 実測位置精度 ---
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.scatter(pooled['fano_max_position_accuracy_pct'], pooled['position_hit_rate_pct'],
                s=8, alpha=0.4, color='#2980b9')
    lim_max = max(pooled['fano_max_position_accuracy_pct'].max(),
                  pooled['position_hit_rate_pct'].max()) * 1.05
    ax2.plot([0, lim_max], [0, lim_max], 'k--', linewidth=1, label='y=x（理論上限に到達）')
    ax2.set_xlabel('Fano理論上限: 達成可能な最大位置精度 (%)')
    ax2.set_ylabel('実測位置精度 (%)')
    ax2.set_title('Fano不等式による理論上限 vs 実測精度（系統別）')
    ax2.legend(fontsize=8)
    plt.tight_layout()
    X.save_fig(fig2, out_dir, 'fano_ceiling_vs_actual.png')

    meta = {
        'walk_forward_dir': args.walk_forward_dir,
        'split': args.split,
        'correlations_entropy_vs_hitrate': correlations,
        'fano_vocab_size_position': config.VOCAB_SIZE_POSITION,
        'fold_summaries': fold_summaries,
        'mean_fano_gap_pct': float(pooled['fano_gap_pct'].mean()),
    }
    X.save_json(meta, out_dir, 'entropy_accuracy_meta.json')
    force_print(f"[INFO] 完了。平均 Fano gap（理論上限-実測, pct pt）: {pooled['fano_gap_pct'].mean():.2f}")


if __name__ == '__main__':
    main()
