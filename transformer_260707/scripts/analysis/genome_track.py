# --- transformer_260707/scripts/analysis/genome_track.py ---
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
  python -m transformer_260707.scripts.analysis.genome_track \\
      --checkpoint outputs/.../models/best_model.pth --split test --n_batches 100

  # walk-forward: 各フォールドを自分の test 期間で評価し、全期間分を積算する
  # （単一のtimestep/全履歴モデルを別途学習する必要がない。split_type_wf は
  #  他の walk_forward プロセスと共有のため同時実行しないこと）
  python -m transformer_260707.scripts.analysis.genome_track \\
      --walk_forward_dir outputs/transformer_260707/results/walk_forward/<timestamp> --n_batches 100
"""

import argparse
import os
from collections import defaultdict

import numpy as np
import pandas as pd

from transformer_260707 import config
from transformer_260707.utils.logging import force_print
from transformer_260707.scripts.analysis import _xai_common as X


def _aggregate_and_report(pred_mass, target_count, n_samples, out_dir, pos2gene):
    if n_samples > 0:
        pred_mass = pred_mass / n_samples
        target_rate = target_count / n_samples
    else:
        target_rate = target_count

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


def run_walk_forward(args):
    """各フォールドを自分の test ウィンドウで評価し、全期間分の位置重要度を積算する。"""
    from transformer_260707.db.connection import get_db_path
    from transformer_260707.db.dataset import create_db_dataloader

    device = 'cpu' if args.force_cpu else config.DEVICE
    db_path = get_db_path()
    fold_windows = X.get_fold_windows()

    fold_ckpts = X.discover_fold_checkpoints(args.walk_forward_dir, args.folds)
    if not fold_ckpts:
        raise SystemExit(f"[ERROR] fold チェックポイントが見つかりません: {args.walk_forward_dir}")
    force_print(f"[INFO] 発見したフォールド: {sorted(fold_ckpts)}")

    out_dir = X.make_output_dir('genome_track_walk_forward', args.output_dir)
    force_print(f"Output dir: {out_dir}")
    pos2gene = X.load_position_gene_map()

    V = config.VOCAB_SIZE_POSITION
    pred_mass_total = np.zeros(V, dtype=np.float64)
    target_count_total = np.zeros(V, dtype=np.float64)
    n_samples_total = 0
    used_folds = []

    for fold_id in sorted(fold_ckpts):
        if fold_id not in fold_windows:
            force_print(f"[WARN] fold {fold_id} は FOLDS 定義に無いためスキップ")
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
        pred_mass, target_count, n = X.compute_position_importance(model, loader, args.n_batches, device)
        pred_mass_total += pred_mass
        target_count_total += target_count
        n_samples_total += n
        used_folds.append(fold_id)
        del model

    force_print(f"[INFO] n_samples_total={n_samples_total} folds={used_folds}")
    _aggregate_and_report(pred_mass_total, target_count_total, n_samples_total, out_dir, pos2gene)
    X.save_json({'mode': 'walk_forward', 'folds': used_folds, 'n_samples': n_samples_total,
                 'n_batches_per_fold': args.n_batches, 'walk_forward_dir': args.walk_forward_dir},
                out_dir, 'genome_track_meta.json')


def main():
    parser = argparse.ArgumentParser(description='Genome-track attribution (position/gene importance)')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--walk_forward_dir', type=str, default=None,
                         help='walk_forward 結果ディレクトリ（指定時は --checkpoint の代わりに全フォールドを積算）')
    parser.add_argument('--folds', type=int, nargs='+', default=None, help='walk_forward_dir 使用時に対象を絞る')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--n_batches', type=int, default=100)
    parser.add_argument('--force_cpu', action='store_true', help='GPU が埋まっている場合に CPU で実行')
    parser.add_argument('--output_dir', type=str, default=None)
    args = parser.parse_args()

    if args.walk_forward_dir:
        run_walk_forward(args)
        return
    if not args.checkpoint:
        raise SystemExit("--checkpoint または --walk_forward_dir のいずれかが必要です")

    out_dir = X.make_output_dir('genome_track', args.output_dir)
    force_print(f"Output dir: {out_dir}")

    model, loader, device = X.load_model_and_loader(
        args.checkpoint, split=args.split, force_cpu=args.force_cpu)
    pos2gene = X.load_position_gene_map()

    force_print(f"[INFO] Aggregating position importance over {args.n_batches} batches ({args.split})...")
    pred_mass, target_count, n_samples = X.compute_position_importance(
        model, loader, args.n_batches, device)
    force_print(f"[INFO] n_samples={n_samples}")
    _aggregate_and_report(pred_mass, target_count, n_samples, out_dir, pos2gene)


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
