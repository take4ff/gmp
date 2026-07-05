# --- transformer_260702/scripts/analysis/genome_track.py ---
"""① ゲノムトラック型アトリビューション。

学習済みモデルが「次変異と予測する確率質量」をゲノム位置ごとに集計し、
遺伝子アノテーション（codon_mutation4.csv の protein）に重ねて可視化する。
生物学的問い: モデルは Spike / RdRp(nsp12) 等の意味のある領域に注目しているか？

出力:
  {out}/position_importance.csv   … position, gene, pred_mass, target_count, pred_minus_target
  {out}/gene_importance.csv        … 遺伝子別の集計
  {out}/genome_track.png           … 位置 vs 予測質量（遺伝子を帯で注釈）
  {out}/gene_importance.png        … 遺伝子別バー

Usage:
  python -m transformer_260702.scripts.analysis.genome_track \\
      --checkpoint outputs/.../models/best_model.pth --split test --n_batches 100
"""

import argparse
import os
from collections import defaultdict

import numpy as np
import pandas as pd

from transformer_260702.utils.logging import force_print
from transformer_260702.scripts.analysis import _xai_common as X


def main():
    parser = argparse.ArgumentParser(description='Genome-track attribution (position/gene importance)')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--n_batches', type=int, default=100)
    parser.add_argument('--force_cpu', action='store_true', help='GPU が埋まっている場合に CPU で実行')
    parser.add_argument('--output_dir', type=str, default=None)
    args = parser.parse_args()

    out_dir = X.make_output_dir('genome_track', args.output_dir)
    force_print(f"Output dir: {out_dir}")

    model, loader, device = X.load_model_and_loader(
        args.checkpoint, split=args.split, force_cpu=args.force_cpu)
    pos2gene = X.load_position_gene_map()

    force_print(f"[INFO] Aggregating position importance over {args.n_batches} batches ({args.split})...")
    pred_mass, target_count, n_samples = X.compute_position_importance(
        model, loader, args.n_batches, device)
    force_print(f"[INFO] n_samples={n_samples}")

    # 正規化（1サンプルあたりの平均予測質量 / 出現率）
    if n_samples > 0:
        pred_mass = pred_mass / n_samples
        target_rate = target_count / n_samples
    else:
        target_rate = target_count

    # --- 位置別テーブル（予測質量 or 実出現がある位置のみ）---
    rows = []
    V = len(pred_mass)
    for p in range(V):
        if pred_mass[p] == 0 and target_count[p] == 0:
            continue
        gene, gpos = pos2gene.get(p, ('unknown', '0'))
        rows.append({
            'position': p, 'gene': gene, 'protein_pos': gpos,
            'pred_mass': pred_mass[p], 'target_rate': target_rate[p],
            'pred_minus_target': pred_mass[p] - target_rate[p],
        })
    df = pd.DataFrame(rows).sort_values('pred_mass', ascending=False).reset_index(drop=True)
    X.save_csv(df, out_dir, 'position_importance.csv')
    force_print(f"[INFO] {len(df)} positions")

    # --- 遺伝子別集計 ---
    gene_agg = defaultdict(lambda: {'pred_mass': 0.0, 'target_rate': 0.0, 'n_pos': 0})
    for r in rows:
        g = gene_agg[r['gene']]
        g['pred_mass'] += r['pred_mass']
        g['target_rate'] += r['target_rate']
        g['n_pos'] += 1
    gdf = pd.DataFrame([
        {'gene': k, 'pred_mass': v['pred_mass'], 'target_rate': v['target_rate'], 'n_pos': v['n_pos']}
        for k, v in gene_agg.items()
    ]).sort_values('pred_mass', ascending=False).reset_index(drop=True)
    X.save_csv(gdf, out_dir, 'gene_importance.csv')

    _plot_genome_track(df, out_dir)
    _plot_gene_bar(gdf, out_dir)

    force_print("\nTop-10 予測質量の高い遺伝子:")
    for _, r in gdf.head(10).iterrows():
        force_print(f"    {r['gene']:14s} pred_mass={r['pred_mass']:.4f}  (target_rate={r['target_rate']:.4f}, {int(r['n_pos'])} pos)")


def _plot_genome_track(df, out_dir):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as e:
        force_print(f"[WARNING] plot skipped: {e}")
        return
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(df['position'], df['pred_mass'], width=8, color='#4C72B0')
    ax.set_xlabel('Genome position (nt)')
    ax.set_ylabel('Mean predicted mass / sample')
    ax.set_title('Genome-track: where the model predicts the next mutation')
    # 主要遺伝子境界に注釈
    for gene in ('S', 'nsp12', 'N', 'ORF3a'):
        sub = df[df['gene'] == gene]
        if not sub.empty:
            x = int(sub['position'].median())
            ax.axvline(x, color='#c44', lw=0.6, alpha=0.5)
            ax.text(x, ax.get_ylim()[1] * 0.9, gene, fontsize=8, color='#c44', ha='center')
    fig.tight_layout()
    X.save_fig(fig, out_dir, 'genome_track.png')


def _plot_gene_bar(gdf, out_dir):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as e:
        force_print(f"[WARNING] plot skipped: {e}")
        return
    d = gdf.sort_values('pred_mass', ascending=True)
    fig, ax = plt.subplots(figsize=(8, max(4, 0.3 * len(d))))
    ax.barh(d['gene'], d['pred_mass'], color='#55a868')
    ax.set_xlabel('Total predicted mass / sample')
    ax.set_title('Predicted next-mutation mass by gene')
    fig.tight_layout()
    X.save_fig(fig, out_dir, 'gene_importance.png')


if __name__ == '__main__':
    main()
