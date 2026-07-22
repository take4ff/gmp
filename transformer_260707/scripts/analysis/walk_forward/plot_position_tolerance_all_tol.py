# --- transformer_260707/scripts/analysis/walk_forward/plot_position_tolerance_all_tol.py ---
"""提案モデル単体、またはPETRA単体（比較なし）で、位置予測の許容誤差(tolerance)別Hit Rateを
月別に1枚のグラフへ重ねて表示する。

--main_csv: position_tolerance_monthly.py が出力する position_tolerance_monthly.csv
            （hit_rate_tol{0,5,10,50}列）
--petra_csv_grouped: petra.eval.plot_monthly_position_tolerance が出力する
            petra_position_tolerance_monthly_grouped.csv
            （hit_rate_tol{0,5,10,50}[_nonleaked]列、既定でnon_leaked列を使う）
いずれか一方のみ指定する。どちらもDB・モデル読み込み不要（既存CSVを読むだけ）。

Usage:
  python -m transformer_260707.scripts.analysis.walk_forward.plot_position_tolerance_all_tol \\
      --main_csv outputs/transformer_260707/scripts/position_tolerance_monthly/<ts>/position_tolerance_monthly.csv \\
      --tolerances 0 5 10 50

  python -m transformer_260707.scripts.analysis.walk_forward.plot_position_tolerance_all_tol \\
      --petra_csv_grouped outputs/petra/walk_forward/<ts>/monthly_plot/petra_position_tolerance_monthly_grouped.csv \\
      --tolerances 0 5 10 50 [--leak_tolerant]
"""
import argparse
import os
import re

import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Noto Sans CJK JP'

_YEARMONTH_RE = re.compile(r'^\d{4}-\d{2}$')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--main_csv', default=None)
    ap.add_argument('--petra_csv_grouped', default=None)
    ap.add_argument('--leak_tolerant', action='store_true',
                    help='PETRA側でleaked/non_leakedを区別せず全件プールした列を使う')
    ap.add_argument('--tolerances', type=int, nargs='+', default=[0, 5, 10, 50])
    ap.add_argument('--output_dir', default=None)
    ap.add_argument('--output_name', default=None)
    args = ap.parse_args()

    if bool(args.main_csv) == bool(args.petra_csv_grouped):
        raise SystemExit("[ERROR] --main_csv か --petra_csv_grouped のどちらか一方だけを指定してください。")

    is_petra = args.petra_csv_grouped is not None
    src_csv = args.petra_csv_grouped if is_petra else args.main_csv

    df = pd.read_csv(src_csv)
    df = df[df['month'].astype(str).str.match(_YEARMONTH_RE)]
    months = df['month'].tolist()

    if is_petra:
        n_col = 'n_groups' if args.leak_tolerant else (
            'n_groups_nonleaked' if 'n_groups_nonleaked' in df.columns else 'n_groups')
        title_subject = 'PETRA型ベースライン（親グループ・any-of-set）'
    else:
        n_col = 'n_samples' if 'n_samples' in df.columns else 'n'
        title_subject = '提案モデル'

    out_dir = args.output_dir or os.path.dirname(src_csv)
    os.makedirs(out_dir, exist_ok=True)
    output_name = args.output_name or (
        'petra_position_tolerance_all_tol.png' if is_petra else 'position_tolerance_all_tol.png')

    cmap = plt.get_cmap('viridis', max(len(args.tolerances), 2))
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(max(14, len(months) * 0.35), 8),
        gridspec_kw={'height_ratios': [3, 1]}, sharex=True)

    for i, tol in enumerate(args.tolerances):
        col = f'hit_rate_tol{tol}'
        if is_petra and not args.leak_tolerant:
            col = f'hit_rate_tol{tol}_nonleaked'
            if col not in df.columns:
                col = f'hit_rate_tol{tol}'
        label = '完全一致 (tolerance=0)' if tol == 0 else f'許容誤差 ±{tol}塩基以内'
        ax1.plot(months, df[col], marker='o', ms=4, lw=1.5, color=cmap(i), label=label)

    ax1.set_ylabel('Hit Rate (%)')
    ax1.set_title(f'{title_subject}: 月別位置予測Hit Rate（tolerance別重ね描き）')
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.bar(months, df[n_col], color='#2980b9', alpha=0.5)
    ax2.set_yscale('log')
    ax2.set_ylabel('n\n(log)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    fig_path = os.path.join(out_dir, output_name)
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[INFO] Saved: {fig_path}")


if __name__ == '__main__':
    main()
