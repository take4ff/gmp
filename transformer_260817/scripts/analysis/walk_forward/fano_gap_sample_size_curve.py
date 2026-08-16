# --- transformer_260817/scripts/analysis/walk_forward/fano_gap_sample_size_curve.py ---
"""系統別のFano理論上限に対する「達成率」がサンプル数(n)に応じてどう飽和していくかを
フィットし、理論上限に近づくために必要なサンプル数を推定する。

entropy_accuracy_analysis.py が出力する entropy_accuracy_pooled_lineages.csv
（系統×fold粒度、n・実測position精度・系統別条件付きFano理論上限を含む）を入力とする。

達成率 = position_hit_rate_pct / fano_max_position_accuracy_pct （0〜1、理論上限のうち
実際どれだけ達成できているか）を n に対して飽和曲線でフィットする（2モデル）:
  saturating exponential: ratio(n) = 1 - exp(-n / n0)
  hyperbolic (Michaelis-Menten型): ratio(n) = n / (n + n0)
n0（特性サンプル数）から、達成率が任意の閾値（既定 0.5/0.75/0.90/0.95）に到達するのに
必要なnを逆算する。

理論上限自体は「真のラベル分布から情報理論的に導かれる上限」であり、それに到達するのに
何サンプル必要かは統計的学習理論の別問題のため、本スクリプトはあくまで経験的フィットに
よる推定値を与える（因果的な保証ではない）。

Usage:
  python -m transformer_260817.scripts.analysis.walk_forward.fano_gap_sample_size_curve \\
      --pooled_csv outputs/transformer_260817/scripts/entropy_accuracy/<ts>/entropy_accuracy_pooled_lineages.csv
"""
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

plt.rcParams['font.family'] = 'Noto Sans CJK JP'

from transformer_260817.utils.logging import force_print
from transformer_260817.scripts.analysis.xai import _xai_common as X

THRESHOLDS = [0.5, 0.75, 0.90, 0.95]


def _model_exp(n, n0):
    return 1 - np.exp(-n / n0)


def _model_hyperbolic(n, n0):
    return n / (n + n0)


def _r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pooled_csv', required=True,
                    help='entropy_accuracy_analysis.py の entropy_accuracy_pooled_lineages.csv')
    ap.add_argument('--min_samples', type=int, default=10)
    ap.add_argument('--output_dir', default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.pooled_csv)
    df = df[df['num_samples'] >= args.min_samples].copy()
    df = df[df['fano_max_position_accuracy_pct'] > 0]

    df['achievement_ratio'] = (df['position_hit_rate_pct'] / df['fano_max_position_accuracy_pct'])
    n_exceeds = (df['achievement_ratio'] > 1.0).sum()
    if n_exceeds:
        force_print(f"[WARNING] {n_exceeds}行で実測が理論上限を超過（達成率>1）。"
                    f"飽和曲線フィットのため1.0にクリップする。")
    df['achievement_ratio'] = df['achievement_ratio'].clip(upper=1.0)

    n = df['num_samples'].to_numpy(dtype=float)
    ratio = df['achievement_ratio'].to_numpy(dtype=float)
    force_print(f"[INFO] フィット対象: {len(df):,}行（系統×fold、n>={args.min_samples}）")

    fits = {}
    for name, model in [('exp', _model_exp), ('hyperbolic', _model_hyperbolic)]:
        try:
            popt, _pcov = curve_fit(model, n, ratio, p0=[np.median(n)], maxfev=10000)
            n0 = float(popt[0])
            pred = model(n, n0)
            r2 = _r_squared(ratio, pred)
            fits[name] = {'n0': n0, 'r_squared': float(r2)}
            force_print(f"[INFO] モデル={name}: n0={n0:,.0f}, R^2={r2:.4f}")
        except Exception as e:
            force_print(f"[WARNING] モデル={name} のフィットに失敗: {e}")

    if not fits:
        raise SystemExit("[ERROR] どちらのモデルもフィットできませんでした。")

    best_name = max(fits, key=lambda k: fits[k]['r_squared'])
    best_n0 = fits[best_name]['n0']
    force_print(f"[INFO] 採用モデル: {best_name} (R^2={fits[best_name]['r_squared']:.4f})")

    model_fn = {'exp': _model_exp, 'hyperbolic': _model_hyperbolic}[best_name]
    n_at_threshold = {}
    for th in THRESHOLDS:
        if best_name == 'exp':
            n_needed = -best_n0 * np.log(1 - th)
        else:  # hyperbolic
            n_needed = best_n0 * th / (1 - th)
        n_at_threshold[th] = float(n_needed)
        force_print(f"[INFO]   達成率{th*100:.0f}%に必要な推定サンプル数: n≈{n_needed:,.0f}")

    out_dir = X.make_output_dir('fano_gap_sample_size_curve', args.output_dir)
    X.save_csv(df[['lineage', 'fold_id', 'era', 'num_samples', 'position_hit_rate_pct',
                   'fano_max_position_accuracy_pct', 'achievement_ratio']],
              out_dir, 'achievement_ratio_by_lineage.csv')
    X.save_json({'fits': fits, 'best_model': best_name,
                'n_at_threshold': n_at_threshold,
                'n_rows': len(df), 'n_exceeds_clipped': int(n_exceeds)},
               out_dir, 'fano_gap_sample_size_fit.json')

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(n, ratio, s=10, alpha=0.35, color='#2980b9', edgecolors='none',
              label='系統×fold（実測値）')
    n_line = np.logspace(np.log10(max(n.min(), 1)), np.log10(n.max()), 200)
    model_fns = {'exp': _model_exp, 'hyperbolic': _model_hyperbolic}
    for name, style in [('exp', '-'), ('hyperbolic', '--')]:
        if name in fits:
            ax.plot(n_line, model_fns[name](n_line, fits[name]['n0']), style, lw=2,
                    label=f"{name} フィット (n0={fits[name]['n0']:,.0f}, R^2={fits[name]['r_squared']:.3f})")
    colors = plt.get_cmap('viridis', len(THRESHOLDS))
    for i, th in enumerate(THRESHOLDS):
        ax.axhline(th, color=colors(i), ls=':', lw=1, alpha=0.6)
        ax.axvline(n_at_threshold[th], color=colors(i), ls=':', lw=1, alpha=0.6,
                  label=f'達成率{th*100:.0f}% → n≈{n_at_threshold[th]:,.0f}')
    ax.set_xscale('log')
    ax.set_xlabel('サンプル数 n（系統×fold、log scale）')
    ax.set_ylabel('達成率 = 実測位置精度 / Fano理論上限')
    ax.set_title('Fano理論上限への達成率とサンプル数の関係')
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    X.save_fig(fig, out_dir, 'fano_gap_sample_size_curve.png')

    force_print(f"\n[INFO] 完了。出力先: {out_dir}")


if __name__ == '__main__':
    main()
