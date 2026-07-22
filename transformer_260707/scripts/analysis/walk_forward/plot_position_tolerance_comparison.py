# --- transformer_260707/scripts/analysis/walk_forward/plot_position_tolerance_comparison.py ---
"""提案モデルとPETRA型ベースライン（親グループ・any-of-set版）の、位置予測の許容誤差(tolerance)
別・月別Hit Rateを比較する。

position_tolerance_monthly.py（提案モデル側）と
petra.eval.plot_monthly_position_tolerance が出力する「親グループ単位・any-of-set」版
（本体と同じ分岐+共起グルーピング・hit判定に揃えたPETRA評価、--petra_csv_grouped）を読み込み、
tolerance ごとに別ファイル（1ファイル=1tolerance）で両モデルの月別折れ線を重ねる。

PETRAの「枝ごと・単一ターゲット」版（tail-only、本体とは分母・正解集合の広さが異なり直接比較
不可）はプロットしない（2026-07-18、ユーザー判断）。

--leak_tolerant （2026-07-19追加）:
  同一raw_pathが採取日の異なる複数サンプル間で完全一致する「リーク」は、train/testが時系列分割
  である以上、本体・PETRA双方に等しく存在する構造的な性質であり、本体側はこれを層別せず全件
  プールして評価している。この事実を踏まえ、PETRA側も同じ基準（全件プール、leaked/non_leaked
  を区別しない）に揃えた比較図を出す場合はこのフラグを付ける（出力ファイル名に `_pooled`
  サフィックスが付き、既定の出力とは別ファイルになる）。leaked/non_leaked を区別する既定の
  出力はそのまま残る（リーク層別自体の分析価値はあるため）。層別評価の本格導入は
  docs/roadmap.md の今後の展望（項目6）を参照。

--fano_csv （2026-07-21追加）:
  entropy_fano_by_month.py が出力する entropy_fano_by_month.csv を渡すと、tolerance=0の図に
  限り Fano理論上限（fano_max_position_accuracy_pct、月別ターゲットエントロピーから導く達成
  可能な最大位置精度）の折れ線も重ねる。Fano上限は「完全一致」の理論限界であり、tolerance>0の
  緩和判定には対応しないため、tol=0の図にのみ追加する。

Usage:
  python -m transformer_260707.scripts.analysis.walk_forward.plot_position_tolerance_comparison \
      --main_csv outputs/transformer_260707/scripts/position_tolerance_monthly/<ts>/position_tolerance_monthly.csv \
      --petra_csv_grouped outputs/petra/walk_forward/<ts>/monthly_plot/petra_position_tolerance_monthly_grouped.csv \
      --tolerances 0 5 10 50 [--leak_tolerant] \
      [--fano_csv outputs/transformer_260707/scripts/entropy_accuracy/by_month/entropy_fano_by_month.csv]
"""
import argparse
import os
import re

import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Noto Sans CJK JP'

# 既知issue: reference/sequences-241017_2.csv の一部レコード（Zimbabwe/Harare 217件）が
# Collection_Date='2022/2024' という不正値を持ち、月ラベルが '2022/20' になる
# （data_quality_collection_date_bug 参照）。上流のどのCSVに残っていても、最終表示段で除外する。
_YEARMONTH_RE = re.compile(r'^\d{4}-\d{2}$')


def _filter_valid_months(df):
    return df[df['month'].astype(str).str.match(_YEARMONTH_RE)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--main_csv', required=True)
    ap.add_argument('--petra_csv_grouped', required=True,
                    help='本体と同じ分岐+共起グルーピング・any-of-set判定に揃えたPETRA評価CSV')
    ap.add_argument('--tolerances', type=int, nargs='+', default=[0, 5, 10, 50])
    ap.add_argument('--output_dir', default=None)
    ap.add_argument('--leak_tolerant', action='store_true',
                    help='PETRA側もleaked/non_leakedを区別せず全件プールした数値を使う'
                         '（本体と同じ「リーク許容」評価に揃える）。出力ファイル名に_pooledが付く。')
    ap.add_argument('--fano_csv', default=None,
                    help='entropy_fano_by_month.csv。指定するとtolerance=0の図にFano理論上限を重ねる')
    args = ap.parse_args()

    main_df = _filter_valid_months(pd.read_csv(args.main_csv))
    petra_df = _filter_valid_months(pd.read_csv(args.petra_csv_grouped))

    fano_df = None
    if args.fano_csv:
        fano_df = pd.read_csv(args.fano_csv).rename(columns={'yearmonth': 'month'})
        fano_df = _filter_valid_months(fano_df)

    months = sorted(set(main_df['month']) | set(petra_df['month']))
    main_df = main_df.set_index('month').reindex(months)
    petra_df = petra_df.set_index('month').reindex(months)
    if fano_df is not None:
        fano_df = fano_df.set_index('month').reindex(months)

    out_dir = args.output_dir or os.path.dirname(args.main_csv)
    os.makedirs(out_dir, exist_ok=True)

    n_main_col = 'n_samples' if 'n_samples' in main_df.columns else 'n'

    if args.leak_tolerant:
        # 本体と同じ「リーク許容（全件プール、leaked/non_leakedを区別しない）」評価に揃える。
        n_petra_col = 'n_groups'
        petra_label_suffix = '全件プール, リーク許容'
        n_petra_legend = 'n (PETRA grouped, 全件)'
        file_suffix = '_pooled'
    else:
        n_petra_col = ('n_groups_nonleaked' if 'n_groups_nonleaked' in petra_df.columns
                       else 'n_groups')
        petra_label_suffix = 'train窓非出現(non_leaked)'
        n_petra_legend = 'n (PETRA grouped non_leaked)'
        file_suffix = ''

    saved = []
    for tol in args.tolerances:
        main_col = f'hit_rate_tol{tol}'
        if args.leak_tolerant:
            petra_col = f'hit_rate_tol{tol}'
        else:
            petra_col = f'hit_rate_tol{tol}_nonleaked'
            if petra_col not in petra_df.columns:
                petra_col = f'hit_rate_tol{tol}'  # non_leaked列が無ければ全体にフォールバック

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(max(14, len(months) * 0.35), 8),
            gridspec_kw={'height_ratios': [3, 1]}, sharex=True)

        label = '完全一致 (tolerance=0)' if tol == 0 else f'許容誤差 ±{tol}塩基以内'
        ax1.plot(months, main_df[main_col], marker='o', color='#2980b9',
                 label=f'提案モデル (position top1, {label})')
        ax1.plot(months, petra_df[petra_col], marker='^', color='#27ae60', linestyle='--',
                 label=f'PETRA型ベースライン (親グループ・any-of-set, {petra_label_suffix}, {label})')
        if tol == 0 and fano_df is not None and 'fano_max_position_accuracy_pct' in fano_df.columns:
            ax1.plot(months, fano_df['fano_max_position_accuracy_pct'], marker='s', ms=4,
                     color='#c0392b', linestyle=':',
                     label='Fano理論上限（月別ターゲットエントロピーから導出、緩い上界）')
        ax1.set_ylabel('Hit Rate (%)')
        ax1.set_title(f'Monthly Position Hit Rate Comparison (tolerance = ±{tol} nt)')
        ax1.legend()
        ax1.grid(alpha=0.3)

        ax2.bar(months, main_df[n_main_col], color='#2980b9', alpha=0.5, label='n (proposed)')
        ax2.bar(months, petra_df[n_petra_col], color='#27ae60', alpha=0.4, label=n_petra_legend)
        ax2.set_yscale('log')
        ax2.set_ylabel('n\n(log)')
        ax2.legend(fontsize=8)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        fig_path = os.path.join(out_dir, f'position_tolerance_comparison_tol{tol}{file_suffix}.png')
        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        saved.append(fig_path)
        print(f"[INFO] Saved: {fig_path}")

    print(f"\n[INFO] {len(saved)}枚のtolerance別プロットを {out_dir} に保存しました。")


if __name__ == '__main__':
    main()
