# --- transformer_260707/scripts/analysis/plot_position_tolerance_comparison.py ---
"""提案モデルとPETRA型ベースラインの、位置予測の許容誤差(tolerance)別・月別Hit Rateを比較する。

position_tolerance_monthly.py（提案モデル側）と
petra.plot_monthly_position_tolerance（PETRA側）の出力CSVを読み込み、
tolerance ごとに別ファイル（1ファイル=1tolerance、両モデルの月別折れ線を重ねる）で保存する。
PETRA側はnon_leaked（train窓に未出現＝真の汎化性能）の数値を使う（本体は構造的にリークしない
ため素の数値をそのまま使う。詳細は petra_comparison.md 参照）。

Usage:
  python -m transformer_260707.scripts.analysis.plot_position_tolerance_comparison \
      --main_csv outputs/transformer_260707/scripts/position_tolerance_monthly/<ts>/position_tolerance_monthly.csv \
      --petra_csv outputs/petra/walk_forward/<ts>/monthly_plot/petra_position_tolerance_monthly.csv \
      --tolerances 0 5 10 50
"""
import argparse
import os

import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Noto Sans CJK JP'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--main_csv', required=True)
    ap.add_argument('--petra_csv', required=True)
    ap.add_argument('--tolerances', type=int, nargs='+', default=[0, 5, 10, 50])
    ap.add_argument('--output_dir', default=None)
    args = ap.parse_args()

    main_df = pd.read_csv(args.main_csv)
    petra_df = pd.read_csv(args.petra_csv)

    months = sorted(set(main_df['month']) | set(petra_df['month']))
    main_df = main_df.set_index('month').reindex(months)
    petra_df = petra_df.set_index('month').reindex(months)

    out_dir = args.output_dir or os.path.dirname(args.main_csv)
    os.makedirs(out_dir, exist_ok=True)

    n_main_col = 'n_samples' if 'n_samples' in main_df.columns else 'n'
    n_petra_col = ('n_sequences_nonleaked' if 'n_sequences_nonleaked' in petra_df.columns
                   else 'n_sequences')

    saved = []
    for tol in args.tolerances:
        main_col = f'hit_rate_tol{tol}'
        petra_col = f'hit_rate_tol{tol}_nonleaked'
        if petra_col not in petra_df.columns:
            petra_col = f'hit_rate_tol{tol}'  # non_leaked列が無ければ全体にフォールバック

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(max(14, len(months) * 0.35), 8),
            gridspec_kw={'height_ratios': [3, 1]}, sharex=True)

        label = '完全一致 (tolerance=0)' if tol == 0 else f'許容誤差 ±{tol}塩基以内'
        ax1.plot(months, main_df[main_col], marker='o', color='#2980b9',
                 label=f'提案モデル (position top1, {label})')
        ax1.plot(months, petra_df[petra_col], marker='s', color='#c0392b', linestyle='--',
                 label=f'PETRA型ベースライン (tail-only, non_leaked, {label})')
        ax1.set_ylabel('Hit Rate (%)')
        ax1.set_title(f'Monthly Position Hit Rate Comparison (tolerance = ±{tol} nt)')
        ax1.legend()
        ax1.grid(alpha=0.3)

        ax2.bar(months, main_df[n_main_col], color='#2980b9', alpha=0.5, label='n (proposed)')
        ax2.bar(months, petra_df[n_petra_col], color='#c0392b', alpha=0.4, label='n (PETRA non_leaked)')
        ax2.set_yscale('log')
        ax2.set_ylabel('n\n(log)')
        ax2.legend(fontsize=8)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        fig_path = os.path.join(out_dir, f'position_tolerance_comparison_tol{tol}.png')
        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        saved.append(fig_path)
        print(f"[INFO] Saved: {fig_path}")

    print(f"\n[INFO] {len(saved)}枚のtolerance別プロットを {out_dir} に保存しました。")


if __name__ == '__main__':
    main()
