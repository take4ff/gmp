# --- transformer_260707/scripts/analysis/permutation_importance.py ---
"""Permutation feature importance（数値特徴量）。

各数値特徴量を batch 内でシャッフルして評価指標（hit-rate）の低下量を測る。
再学習不要・モデル非依存。低下量が大きい特徴ほどモデルが依存している＝重要。

勾配法 (feature_importance.py) が「感度」を測るのに対し、こちらは「精度への寄与」を
直接測るため相補的。ABLATION_MASKS で 0 マスク済みの特徴は置換しても変化せず
importance≈0 になる（正しく「使われていない」と示される）。

決定性のため num_workers=0・shuffle=False を強制し、baseline と各特徴で
同一サンプル集合を評価する。

Usage:
    python -m transformer_260707.scripts.analysis.permutation_importance \\
        --checkpoint outputs/transformer_260707/results/<ts>/models/best_model.pth \\
        --split test --metric position_hit_rate --n_batches 50 --n_repeats 2

指標 (--metric): region_hit_rate | position_hit_rate | aa_pos_hit_rate |
                 codon_pos_hit_rate | synonymous_hit_rate
"""

import argparse
import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch

from transformer_260707 import config
from transformer_260707.evaluate import evaluate
from transformer_260707.utils.losses import build_loss_fn
from transformer_260707.utils.logging import force_print
from transformer_260707.scripts.analysis import _xai_common as X
from transformer_260707.scripts.analysis.feature_importance import _NUM_FEATURE_NAMES


def _weighted_metric(val_metrics, key):
    """タイムステップ長別 hit-rate を num_samples 重み付き平均する。"""
    if not val_metrics:
        return float('nan')
    total = sum(m['num_samples'] for m in val_metrics.values())
    if total == 0:
        return float('nan')
    return sum(m.get(key, 0.0) * m['num_samples'] for m in val_metrics.values()) / total


class _PermutingLoader:
    """base_loader を包み、指定した数値特徴量 (x_num[..., feat_idx]) を
    batch 内でサンプル方向にシャッフルして yield する。

    feat_idx=None のとき無置換（baseline）。先頭 n_batches のみ処理する。
    num_workers=0・shuffle=False 前提で、baseline/各特徴で同一サンプルを保証する。
    """

    def __init__(self, base_loader, feat_idx, n_batches, seed):
        self.base_loader = base_loader
        self.feat_idx = feat_idx
        self.n_batches = n_batches
        self.seed = seed

    def __len__(self):
        try:
            return min(len(self.base_loader), self.n_batches)
        except TypeError:
            return self.n_batches

    def __iter__(self):
        gen = torch.Generator().manual_seed(self.seed)
        for i, batch in enumerate(self.base_loader):
            if i >= self.n_batches:
                break
            if self.feat_idx is None:
                yield batch
                continue
            (x_cat, x_num, mask), *rest = batch
            x_num = x_num.clone()
            B = x_num.shape[0]
            perm = torch.randperm(B, generator=gen)
            # サンプル方向に feature 列を入れ替え（他の特徴・構造は保持）
            x_num[..., self.feat_idx] = x_num[perm][..., self.feat_idx]
            yield ((x_cat, x_num, mask), *rest)


def main():
    parser = argparse.ArgumentParser(description='Permutation feature importance (numeric features)')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--metric', type=str, default='position_hit_rate',
                        help='低下量を測る指標キー。<task>_<指標> 形式。'
                             'task={region,position,aa_pos,codon_pos,synonymous}, '
                             '指標={hit_rate,macro_recall,weighted_recall,f1,precision,recall}')
    parser.add_argument('--n_batches', type=int, default=50,
                        help='評価に使う先頭バッチ数（多いほど安定・遅い）')
    parser.add_argument('--n_repeats', type=int, default=2,
                        help='各特徴のシャッフル反復回数（平均して分散を減らす）')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--force_cpu', action='store_true', help='GPU が埋まっている場合に CPU で実行')
    args = parser.parse_args()

    # 共通ローダで config_snapshot 反映・モデル・DataLoader を用意（_xai_common に集約）。
    # load_model_and_loader が num_workers=0・shuffle=False に固定するため決定性が保たれる。
    out_dir = X.make_output_dir('permutation_importance', args.output_dir)
    force_print(f"Output dir: {out_dir}")
    model, base_loader, device = X.load_model_and_loader(
        args.checkpoint, split=args.split, force_cpu=args.force_cpu)

    loss_fn = build_loss_fn()
    metric_key = args.metric

    # ── baseline（無置換）─────────────────────────────────────
    force_print(f"\n[baseline] evaluating {args.n_batches} batches on {args.split} "
                f"(metric={metric_key})...")
    _, base_metrics, *_ = evaluate(
        model, _PermutingLoader(base_loader, None, args.n_batches, config.SEED),
        loss_fn, None)
    baseline = _weighted_metric(base_metrics, metric_key)
    if np.isnan(baseline):
        raise ValueError(f"metric '{metric_key}' が val_metrics に見つかりません。"
                         f"利用可能: {sorted(k for m in base_metrics.values() for k in m if k != 'num_samples')}")
    force_print(f"[baseline] {metric_key} = {baseline:.4f}")

    # ── 各特徴を置換して低下量を測る ─────────────────────────
    n_feats = config.NUM_CHEM_FEATURES
    names = _NUM_FEATURE_NAMES[:n_feats]
    rows = []
    for j in range(n_feats):
        drops = []
        for r in range(args.n_repeats):
            _, pm, *_ = evaluate(
                model, _PermutingLoader(base_loader, j, args.n_batches, config.SEED + 1 + r),
                loss_fn, None)
            drops.append(baseline - _weighted_metric(pm, metric_key))
        drop_mean = float(np.mean(drops))
        drop_std = float(np.std(drops)) if len(drops) > 1 else 0.0
        rows.append({'feature': names[j], 'index': j,
                     'importance_drop': drop_mean, 'std': drop_std})
        force_print(f"  [{j+1:2d}/{n_feats}] {names[j]:24s} Δ{metric_key}={drop_mean:+.4f} (±{drop_std:.4f})")

    df = pd.DataFrame(rows).sort_values('importance_drop', ascending=False).reset_index(drop=True)
    csv_path = os.path.join(out_dir, f'permutation_importance_{metric_key}.csv')
    df.to_csv(csv_path, index=False)
    force_print(f"\n[INFO] Saved CSV: {csv_path}")

    _plot(df, metric_key, baseline, out_dir)

    force_print(f"\nTop-10 important features (larger drop = more important):")
    for _, row in df.head(10).iterrows():
        force_print(f"    {row['feature']:24s}  Δ={row['importance_drop']:+.4f} (±{row['std']:.4f})")
    force_print(f"\n[DONE] baseline {metric_key}={baseline:.4f}, "
                f"{args.n_batches} batches × {args.n_repeats} repeats")


def _plot(df, metric_key, baseline, out_dir):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as e:
        force_print(f"[WARNING] plot skipped: {e}")
        return
    d = df.sort_values('importance_drop', ascending=True)
    fig, ax = plt.subplots(figsize=(8, max(4, 0.32 * len(d))))
    ax.barh(d['feature'], d['importance_drop'], xerr=d['std'],
            color='#4C72B0', ecolor='#999999')
    ax.axvline(0, color='k', lw=0.8)
    ax.set_xlabel(f'Δ {metric_key} (baseline − permuted)')
    ax.set_title(f'Permutation importance (baseline {metric_key}={baseline:.3f})')
    fig.tight_layout()
    p = os.path.join(out_dir, f'permutation_importance_{metric_key}.png')
    fig.savefig(p, dpi=130)
    plt.close(fig)
    force_print(f"[INFO] Saved plot: {p}")


if __name__ == '__main__':
    main()
