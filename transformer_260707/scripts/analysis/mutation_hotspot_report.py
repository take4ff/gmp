# --- transformer_260707/scripts/analysis/mutation_hotspot_report.py ---
"""⑦ 遺伝子別・位置別の変異蓄積傾向レポート。

`genome_track.py` が出力する position_importance.csv / gene_importance.csv
(position, gene, protein_pos, pred_mass, target_rate, pred_minus_target /
 gene, pred_mass, target_rate, n_pos) を読み込み、
「どの遺伝子・位置が変異蓄積しやすいか」「モデルの予測質量(pred_mass)が
実際の出現率(target_rate)とどれだけ一致するか」を定量化して保存する。
genome_track.py 自体は再実行しない（モデルforward不要・軽量な後処理）。

生物学的問い:
  - 変異はどの遺伝子に集中するか（実データ）、モデルはそれを再現できているか
  - モデルはどの遺伝子/位置を過大・過小評価しているか
  - 実出現率トップの位置は既知VOC変異位置（KNOWN_VOC_POSITIONS, local_explain.py）と一致するか

出力:
  {out}/gene_ranking.csv       … 遺伝子別 target_rate 降順ランキング + pred/target比率
  {out}/gene_ranking.png       … 遺伝子別 target_rate vs pred_mass バー比較（上位N、正規化済み）
  {out}/position_hotspots.csv  … target_rate 上位N位置 + 既知VOC変異との一致フラグ
  {out}/position_hotspots.png  … pred_mass vs target_rate 散布図（log軸、VOC位置を強調）
  {out}/summary.json           … Spearman相関・VOC一致件数などのサマリ統計

Usage:
  python -m transformer_260707.scripts.analysis.mutation_hotspot_report \\
      --position_csv outputs/.../genome_track*/position_importance.csv \\
      --gene_csv outputs/.../genome_track*/gene_importance.csv
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from scipy.stats import pearsonr

from transformer_260707.utils.logging import force_print
from transformer_260707.scripts.analysis import _xai_common as X
from transformer_260707.scripts.analysis.local_explain import KNOWN_VOC_POSITIONS


def calibration_stats(gene_ranking):
    """順位相関(Spearman)だけでは捉えられない「大きさのズレ」を定量化する。

    pred_norm / target_norm は合計1の分布(単体上の点)なので、Spearmanは
    順位さえ保たれていれば個々の遺伝子の配分過剰/過小を検出できない。
    Total Variation Distance (TVD) と Pearson相関(線形・大きさに敏感)、
    および TVD に対する各遺伝子の寄与率を併記して較正誤差を可視化する。
    """
    p = gene_ranking['pred_norm'].to_numpy()
    t = gene_ranking['target_norm'].to_numpy()
    pear_r, pear_p = pearsonr(p, t)
    abs_diff = np.abs(p - t)
    tvd = 0.5 * abs_diff.sum()
    kl = float(np.sum(t * np.log((t + 1e-12) / (p + 1e-12))))
    contrib = gene_ranking[['gene']].copy()
    contrib['tvd_contribution'] = abs_diff / (2 * tvd) if tvd > 0 else 0.0
    top_contrib = contrib.sort_values('tvd_contribution', ascending=False).head(3)
    return {
        'pearson_r': float(pear_r),
        'pearson_p': float(pear_p),
        'total_variation_distance': float(tvd),
        'kl_target_given_pred': kl,
        'top_tvd_contributors': [
            {'gene': row['gene'], 'tvd_contribution_pct': float(row['tvd_contribution'] * 100)}
            for _, row in top_contrib.iterrows()
        ],
    }

plt.rcParams['font.family'] = 'Noto Sans CJK JP'

BLUE = '#2b5c8f'
ORANGE = '#d95f02'
RED = '#c0392b'
GRAY = '#888888'


def build_gene_ranking(gene_df):
    df = gene_df[gene_df['n_pos'] > 0].copy()
    df['target_per_pos'] = df['target_rate'] / df['n_pos']
    df['pred_norm'] = df['pred_mass'] / df['pred_mass'].sum()
    df['target_norm'] = df['target_rate'] / df['target_rate'].sum()
    df['ratio'] = df['pred_norm'] / df['target_norm'].replace(0, np.nan)
    df = df.sort_values('target_rate', ascending=False).reset_index(drop=True)
    return df


def plot_gene_ranking(df, out_dir, top_n=15):
    top = df.head(top_n).iloc[::-1]  # 横棒グラフ用に昇順反転
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(top) + 1.5))
    y = np.arange(len(top))
    h = 0.38
    ax.barh(y + h / 2, top['target_norm'], height=h, color=BLUE, label='実出現率 (target)')
    ax.barh(y - h / 2, top['pred_norm'], height=h, color=ORANGE, label='モデル予測質量 (pred)')
    ax.set_yticks(y)
    ax.set_yticklabels(top['gene'])
    ax.set_xlabel('正規化された比率（全遺伝子中の割合）')
    ax.set_title(f'遺伝子別 変異蓄積率 vs モデル予測質量（上位{top_n}遺伝子, target_rate降順）')
    ax.legend(fontsize=9)
    ax.grid(axis='x', alpha=0.3)
    X.save_fig(fig, out_dir, 'gene_ranking.png')


def build_position_hotspots(pos_df, top_n=30):
    df = pos_df.sort_values('target_rate', ascending=False).head(top_n).copy()
    df['known_voc'] = df['position'].map(KNOWN_VOC_POSITIONS)
    df['is_known_voc'] = df['known_voc'].notna()
    return df.reset_index(drop=True)


def plot_position_hotspots(pos_df, hotspots_df, out_dir):
    df = pos_df[(pos_df['pred_mass'] > 0) | (pos_df['target_rate'] > 0)].copy()
    voc_positions = set(KNOWN_VOC_POSITIONS.keys())
    is_voc = df['position'].isin(voc_positions)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(df.loc[~is_voc, 'target_rate'] + 1e-6, df.loc[~is_voc, 'pred_mass'] + 1e-6,
               s=10, alpha=0.35, color=GRAY, label='その他の位置')
    ax.scatter(df.loc[is_voc, 'target_rate'] + 1e-6, df.loc[is_voc, 'pred_mass'] + 1e-6,
               s=60, color=RED, edgecolor='black', linewidth=0.5, zorder=5,
               label='既知VOC変異位置')
    for _, row in df[is_voc].iterrows():
        label = KNOWN_VOC_POSITIONS.get(row['position'], '').split(' ')[0]
        ax.annotate(label, (row['target_rate'] + 1e-6, row['pred_mass'] + 1e-6),
                    fontsize=8, xytext=(4, 4), textcoords='offset points')

    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, color='black', linestyle='--', linewidth=1, alpha=0.5, label='pred = target')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('実出現率 target_rate (log)')
    ax.set_ylabel('モデル予測質量 pred_mass (log)')
    ax.set_title('位置別 予測質量 vs 実出現率（既知VOC変異位置を強調）')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which='both')
    X.save_fig(fig, out_dir, 'position_hotspots.png')


def main():
    parser = argparse.ArgumentParser(description='遺伝子別・位置別の変異蓄積傾向レポート')
    parser.add_argument('--position_csv', type=str, required=True,
                        help='genome_track.py が出力した position_importance.csv')
    parser.add_argument('--gene_csv', type=str, required=True,
                        help='genome_track.py が出力した gene_importance.csv')
    parser.add_argument('--top_n_genes', type=int, default=15)
    parser.add_argument('--top_n_positions', type=int, default=30)
    parser.add_argument('--output_dir', type=str, default=None)
    args = parser.parse_args()

    pos_df = pd.read_csv(args.position_csv)
    gene_df = pd.read_csv(args.gene_csv)

    out_dir = X.make_output_dir('mutation_hotspot_report', args.output_dir)
    force_print(f"Output dir: {out_dir}")

    # --- 遺伝子ランキング ---
    gene_ranking = build_gene_ranking(gene_df)
    X.save_csv(gene_ranking, out_dir, 'gene_ranking.csv')
    plot_gene_ranking(gene_ranking, out_dir, top_n=args.top_n_genes)

    # --- 位置ホットスポット ---
    hotspots = build_position_hotspots(pos_df, top_n=args.top_n_positions)
    X.save_csv(hotspots, out_dir, 'position_hotspots.csv')
    plot_position_hotspots(pos_df, hotspots, out_dir)

    # --- サマリ統計 ---
    gene_rho, gene_p = spearmanr(gene_ranking['pred_mass'], gene_ranking['target_rate'])
    pos_nonzero = pos_df[(pos_df['pred_mass'] > 0) | (pos_df['target_rate'] > 0)]
    pos_rho, pos_p = spearmanr(pos_nonzero['pred_mass'], pos_nonzero['target_rate'])
    n_voc_in_top = int(hotspots['is_known_voc'].sum())
    calib = calibration_stats(gene_ranking)

    summary = {
        'source': {
            'position_csv': os.path.abspath(args.position_csv),
            'gene_csv': os.path.abspath(args.gene_csv),
        },
        'gene_level': {
            'n_genes': int(len(gene_ranking)),
            'spearman_rho': float(gene_rho),
            'spearman_p': float(gene_p),
            'note_on_spearman': 'Spearmanは順位一致のみを見る。大きさのズレ(較正誤差)は calibration を参照。',
            'calibration': calib,
            'top_gene_by_target_rate': gene_ranking.iloc[0]['gene'],
            'most_overpredicted_gene': gene_ranking.sort_values('ratio', ascending=False).iloc[0]['gene'],
            'most_underpredicted_gene': gene_ranking.sort_values('ratio', ascending=True).iloc[0]['gene'],
        },
        'position_level': {
            'n_positions_nonzero': int(len(pos_nonzero)),
            'spearman_rho': float(pos_rho),
            'spearman_p': float(pos_p),
            'n_known_voc_positions_total': len(KNOWN_VOC_POSITIONS),
            'n_known_voc_in_top_hotspots': n_voc_in_top,
            'top_n_positions_checked': args.top_n_positions,
        },
    }
    X.save_json(summary, out_dir, 'summary.json')

    force_print(f"\n[SUMMARY] 遺伝子レベル Spearman rho={gene_rho:.4f} (p={gene_p:.2e}) "
                f"／ Pearson r={calib['pearson_r']:.4f}（大きさに敏感）")
    force_print(f"[SUMMARY] 較正誤差 Total Variation Distance={calib['total_variation_distance']:.4f} "
                f"(0=完全一致)　主な寄与: " +
                ", ".join(f"{c['gene']}({c['tvd_contribution_pct']:.1f}%)"
                          for c in calib['top_tvd_contributors']))
    force_print(f"[SUMMARY] 位置レベル   Spearman rho={pos_rho:.4f} (p={pos_p:.2e})")
    force_print(f"[SUMMARY] 上位{args.top_n_positions}ホットスポット中の既知VOC変異位置: "
                f"{n_voc_in_top}/{len(KNOWN_VOC_POSITIONS)}")
    force_print(f"[SUMMARY] 最も過大評価: {summary['gene_level']['most_overpredicted_gene']} / "
                f"最も過小評価: {summary['gene_level']['most_underpredicted_gene']}")


if __name__ == '__main__':
    main()
