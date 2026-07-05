# --- transformer_260702/scripts/analysis/homoplasy_alignment.py ---
"""② ホモプラシー（収斂進化）との整合検証。

モデルの位置別「予測質量」を、系統樹上で繰り返し出現した部位（正の選択下）の
再発回数 reference/homoplasy/site_recurrence.csv と相関させ、
「モデルが進化的に重要な部位を学習したか」を定量化する。

負のコントロール: reference/genome/problematic_sites_sarsCov2.vcf（配列アーチファクト部位）
での予測質量が背景より低いこと＝ノイズ部位に依存していないことを確認する。

出力:
  {out}/homoplasy_alignment.json   … Spearman/Pearson 相関、負コントロールの統計
  {out}/homoplasy_scatter.png      … 再発回数 vs 予測質量の散布図

Usage:
  python -m transformer_260702.scripts.analysis.homoplasy_alignment \\
      --checkpoint outputs/.../models/best_model.pth --split test --n_batches 100
"""

import argparse
import json
import os

import numpy as np

from transformer_260702 import config
from transformer_260702.utils.logging import force_print
from transformer_260702.scripts.analysis import _xai_common as X


def _load_recurrence(path):
    import csv as _csv
    rec = {}
    with open(path, newline='', encoding='utf-8-sig') as f:
        for row in _csv.DictReader(f):
            try:
                rec[int(row['position_id'])] = float(row['recurrence_count'])
            except (KeyError, ValueError, TypeError):
                continue
    return rec


def _load_problematic_positions(vcf_path):
    masked = set()
    with open(vcf_path) as f:
        for line in f:
            if line.startswith('#'):
                continue
            cols = line.rstrip('\n').split('\t')
            if len(cols) >= 7 and 'mask' in cols[6].lower():
                try:
                    masked.add(int(cols[1]))
                except ValueError:
                    continue
    return masked


def _corr(x, y):
    """Pearson / Spearman を返す（scipy があれば使用、無ければ numpy でフォールバック）。"""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    try:
        from scipy.stats import pearsonr, spearmanr
        pr = float(pearsonr(x, y)[0])
        sr = float(spearmanr(x, y)[0])
        return pr, sr
    except Exception:
        def _pear(a, b):
            if a.std() == 0 or b.std() == 0:
                return float('nan')
            return float(np.corrcoef(a, b)[0, 1])
        def _rank(a):
            order = a.argsort()
            r = np.empty_like(order, dtype=float)
            r[order] = np.arange(len(a))
            return r
        return _pear(x, y), _pear(_rank(x), _rank(y))


def main():
    parser = argparse.ArgumentParser(description='Homoplasy alignment of position importance')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--n_batches', type=int, default=100)
    parser.add_argument('--force_cpu', action='store_true')
    parser.add_argument('--output_dir', type=str, default=None)
    args = parser.parse_args()

    out_dir = X.make_output_dir('homoplasy_alignment', args.output_dir)
    force_print(f"Output dir: {out_dir}")

    model, loader, device = X.load_model_and_loader(
        args.checkpoint, split=args.split, force_cpu=args.force_cpu)

    force_print(f"[INFO] Aggregating position importance over {args.n_batches} batches...")
    pred_mass, target_count, n_samples = X.compute_position_importance(
        model, loader, args.n_batches, device)
    if n_samples > 0:
        pred_mass = pred_mass / n_samples
        target_rate = target_count / n_samples
    else:
        target_rate = target_count

    rec = _load_recurrence(os.path.join('reference', 'homoplasy', 'site_recurrence.csv'))
    masked = _load_problematic_positions(
        os.path.join('reference', 'genome', 'problematic_sites_sarsCov2.vcf'))
    force_print(f"[INFO] homoplasy positions={len(rec)}, problematic(masked)={len(masked)}")

    # --- 相関: site_recurrence にある位置で pred_mass / target_rate と再発回数 ---
    positions = sorted(rec.keys())
    V = len(pred_mass)
    positions = [p for p in positions if 0 <= p < V]
    recur = np.array([rec[p] for p in positions])
    pm = np.array([pred_mass[p] for p in positions])
    tr = np.array([target_rate[p] for p in positions])

    pr_model, sr_model = _corr(recur, pm)
    pr_data, sr_data = _corr(recur, tr)

    # --- 負コントロール: problematic vs non-problematic の予測質量 ---
    all_pos = np.arange(V)
    is_masked = np.array([p in masked for p in all_pos])
    pm_masked = pred_mass[is_masked]
    pm_unmasked = pred_mass[~is_masked]
    neg_control = {
        'mean_pred_mass_problematic': float(pm_masked.mean()) if pm_masked.size else 0.0,
        'mean_pred_mass_normal': float(pm_unmasked.mean()) if pm_unmasked.size else 0.0,
        'n_problematic': int(is_masked.sum()),
    }
    ratio = (neg_control['mean_pred_mass_problematic'] /
             neg_control['mean_pred_mass_normal']) if neg_control['mean_pred_mass_normal'] > 0 else float('nan')
    neg_control['problematic_over_normal_ratio'] = ratio

    result = {
        'n_samples': n_samples,
        'n_homoplasy_positions': len(positions),
        'correlation_model_vs_homoplasy': {'pearson': pr_model, 'spearman': sr_model},
        'correlation_data_vs_homoplasy': {'pearson': pr_data, 'spearman': sr_data},
        'negative_control_problematic_sites': neg_control,
        'note': 'model が data(実出現) 以上に homoplasy と相関するなら、頻度以上の進化的シグナルを捉えている。'
                'ratio<1 なら problematic 部位を回避できている。',
    }
    X.save_json(result, out_dir, 'homoplasy_alignment.json')

    _plot_scatter(recur, pm, sr_model, out_dir)

    force_print("\n===== 結果 =====")
    force_print(f"  homoplasy 相関  model : Spearman={sr_model:.3f}, Pearson={pr_model:.3f}")
    force_print(f"  homoplasy 相関  data  : Spearman={sr_data:.3f}, Pearson={pr_data:.3f}")
    force_print(f"  負コントロール problematic/normal 予測質量比 = {ratio:.3f} (1未満が望ましい)")


def _plot_scatter(recur, pm, spearman, out_dir):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as e:
        force_print(f"[WARNING] plot skipped: {e}")
        return
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(recur + 1, pm + 1e-9, s=8, alpha=0.4, color='#4C72B0')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Homoplasy recurrence count (+1, log)')
    ax.set_ylabel('Model predicted mass / sample (log)')
    ax.set_title(f'Position importance vs homoplasy  (Spearman={spearman:.3f})')
    fig.tight_layout()
    X.save_fig(fig, out_dir, 'homoplasy_scatter.png')


if __name__ == '__main__':
    main()
