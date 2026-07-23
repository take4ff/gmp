# --- transformer_260723/scripts/analysis/walk_forward/fano_gap_multivariate.py ---
"""Fano理論上限への「達成率」を、サンプル数(n)だけでなく多様度・phylo距離・recency・
founder効果(分岐)を統制した多変量回帰で説明し、他条件を揃えた「典型的な系統」について
理論上限に到達するのに必要なサンプル数を逆算する。

fano_gap_sample_size_curve.py の単純な n vs 達成率フィットは R^2 が負（nだけでは
ほとんど説明できない）という結果になった。これは達成率が n 以外の要因（特に多様度）と
強く交絡しているためと考えられる。本スクリプトは、lineage_accuracy_drivers.py が
出力する多様度・phylo_dist・recency・founder(K/M)の系統別特徴量と、
entropy_accuracy_analysis.py が出力するFano理論上限を (lineage, fold) で結合し、

  logit(達成率) ~ z(entropy_norm) + z(log1p(n)) + z(norm_phylo_dist)
                  + z(recency_months) + z(frac_branching)

を推定する（標準化係数、達成率は[eps, 1-eps]にクリップしてロジット変換）。
n以外の係数を0（＝平均的な値）に固定した上で、log1p(n)の係数から
「達成率が閾値に届くのに必要なn」を逆算する。

Usage:
  python -m transformer_260723.scripts.analysis.walk_forward.fano_gap_multivariate \\
      --fano_csv outputs/transformer_260723/scripts/entropy_accuracy/<ts>/entropy_accuracy_pooled_lineages.csv \\
      --drivers_csv outputs/transformer_260723/scripts/lineage_accuracy_drivers/<ts>/lineage_accuracy_drivers_pooled.csv
"""
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

plt.rcParams['font.family'] = 'Noto Sans CJK JP'

from transformer_260723.utils.logging import force_print
from transformer_260723.scripts.analysis.xai import _xai_common as X

FACTOR_COLS = ['entropy_norm', 'log1p_n', 'norm_phylo_dist', 'recency_months', 'frac_branching']
FACTOR_LABELS = {
    'entropy_norm':    '多様度 (entropy_norm)',
    'log1p_n':         'log1p(サンプル数)',
    'norm_phylo_dist': '系統距離 (norm_phylo_dist)',
    'recency_months':  'recency (経過月数)',
    'frac_branching':  'founder効果 (frac_branching)',
}
THRESHOLDS = [0.5, 0.75, 0.90, 0.95]
EPS = 1e-3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fano_csv', required=True,
                    help='entropy_accuracy_analysis.py の entropy_accuracy_pooled_lineages.csv')
    ap.add_argument('--drivers_csv', required=True,
                    help='lineage_accuracy_drivers.py の lineage_accuracy_drivers_pooled.csv')
    ap.add_argument('--min_samples', type=int, default=10)
    ap.add_argument('--output_dir', default=None)
    args = ap.parse_args()

    fano_df = pd.read_csv(args.fano_csv).rename(columns={'fold_id': 'fold'})
    drivers_df = pd.read_csv(args.drivers_csv)

    merged = drivers_df.merge(
        fano_df[['lineage', 'fold', 'fano_max_position_accuracy_pct']],
        on=['lineage', 'fold'], how='inner')
    merged = merged[merged['n'] >= args.min_samples].copy()
    merged = merged[merged['fano_max_position_accuracy_pct'] > 0]
    force_print(f"[INFO] 結合後: {len(merged):,}行（系統×fold、n>={args.min_samples}）")

    ratio = (merged['hit_rate'] / merged['fano_max_position_accuracy_pct']).clip(EPS, 1 - EPS)
    merged['achievement_ratio'] = ratio
    merged['logit_ratio'] = np.log(ratio / (1 - ratio))

    sub = merged[['logit_ratio', 'n'] + FACTOR_COLS].dropna().copy()
    z_cols = []
    z_stats = {}
    for col in FACTOR_COLS:
        z_col = f'z_{col}'
        mean, std = sub[col].mean(), sub[col].std()
        sub[z_col] = (sub[col] - mean) / std if std > 0 else 0.0
        z_stats[col] = {'mean': float(mean), 'std': float(std)}
        z_cols.append(z_col)

    X_mat = sm.add_constant(sub[z_cols])
    y = sub['logit_ratio']
    ols = sm.OLS(y, X_mat).fit()
    wls = sm.WLS(y, X_mat, weights=sub['n']).fit()

    out_dir = X.make_output_dir('fano_gap_multivariate', args.output_dir)
    summary_text = f"{'='*70}\nOLS\n{'='*70}\n{ols.summary()}\n\n{'='*70}\nWLS (n重み付き)\n{'='*70}\n{wls.summary()}\n"
    with open(f'{out_dir}/regression_summary.txt', 'w') as f:
        f.write(summary_text)
    force_print(f"[INFO] Saved {out_dir}/regression_summary.txt")

    force_print(f"\n[解釈サマリ] (OLS, R^2={ols.rsquared:.3f}, n={int(ols.nobs)})")
    for col in FACTOR_COLS:
        term = f'z_{col}'
        coef, p = ols.params[term], ols.pvalues[term]
        sig = '有意' if p < 0.05 else '非有意'
        force_print(f"  {FACTOR_LABELS[col]:<28} 標準化係数={coef:+.3f}  p={p:.3e}  [{sig}]")

    # --- n以外を平均的な値(z=0)に固定し、log1p(n)の係数から必要サンプル数を逆算 ---
    b0 = ols.params['const']
    b_n = ols.params['z_log1p_n']
    n_mean, n_std = z_stats['log1p_n']['mean'], z_stats['log1p_n']['std']

    n_at_threshold = {}
    force_print(f"\n[必要サンプル数の推定]（多様度・phylo距離・recency・founder効果は平均的な値に固定）")
    for th in THRESHOLDS:
        logit_th = np.log(th / (1 - th))
        z_needed = (logit_th - b0) / b_n if b_n != 0 else float('nan')
        log1p_n_needed = z_needed * n_std + n_mean
        n_needed = max(np.expm1(log1p_n_needed), 0)
        n_at_threshold[th] = float(n_needed)
        force_print(f"  達成率{th*100:.0f}%に必要な推定サンプル数: n≈{n_needed:,.0f}")

    if b_n <= 0:
        force_print("[WARNING] log1p(n)の係数が非正のため、閾値到達に必要なnが定義できない/発散する"
                    "区間があります（上記値は参考程度）。")

    X.save_csv(merged[['fold', 'era', 'lineage', 'n', 'hit_rate', 'fano_max_position_accuracy_pct',
                       'achievement_ratio'] + FACTOR_COLS],
              out_dir, 'achievement_ratio_merged.csv')
    X.save_json({
        'ols': {'r_squared': float(ols.rsquared), 'n_obs': int(ols.nobs),
               'coefficients': {t: {'coef': float(ols.params[t]), 'p_value': float(ols.pvalues[t])}
                                for t in ols.params.index}},
        'wls': {'r_squared': float(wls.rsquared), 'n_obs': int(wls.nobs),
               'coefficients': {t: {'coef': float(wls.params[t]), 'p_value': float(wls.pvalues[t])}
                                for t in wls.params.index}},
        'z_stats': z_stats,
        'n_at_threshold_holding_others_at_mean': n_at_threshold,
    }, out_dir, 'fano_gap_multivariate_fit.json')

    # --- 可視化: 実測 vs モデル予測（logit_ratio）の散布図 ---
    fig, ax = plt.subplots(figsize=(7, 6))
    pred = ols.predict(X_mat)
    ax.scatter(pred, y, s=8, alpha=0.3, color='#2980b9', edgecolors='none')
    lim = [min(pred.min(), y.min()), max(pred.max(), y.max())]
    ax.plot(lim, lim, 'k--', lw=1, label='y=x')
    ax.set_xlabel('モデル予測 logit(達成率)')
    ax.set_ylabel('実測 logit(達成率)')
    ax.set_title(f'多変量回帰の当てはまり (OLS R^2={ols.rsquared:.3f})')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    X.save_fig(fig, out_dir, 'fano_gap_multivariate_fit.png')

    force_print(f"\n[INFO] 完了。出力先: {out_dir}")


if __name__ == '__main__':
    main()
