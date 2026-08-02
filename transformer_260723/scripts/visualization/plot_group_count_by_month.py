# 月別に、共起グルーピング前のサンプル数（raw_count）とグルーピング後のグループ数
# （group_count = 同一input_path_str = raw_pathの最終ステップを除いた履歴、を持つ
# サンプルの集合数。db/dataset.py:DBIterableDataset._build_groups と同一定義）を集計・
# プロットする。walk_forward fold_3（Omicron早期 BA.1/BA.2）で共起グループが
# 他foldより顕著に大きい現象の時系列的な裏付けを取るための診断スクリプト。
# Usage: python -m transformer_260723.scripts.visualization.plot_group_count_by_month
import os

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from transformer_260723 import config
from transformer_260723.db.connection import get_db_path, connect_db

plt.rcParams['font.family'] = 'Noto Sans CJK JP'

# dataviz skill 検証済みカテゴリカルパレット（slot1=blue, slot2=orange）
_COLOR_RAW = '#2a78d6'
_COLOR_GROUP = '#eb6834'


def main():
    db_path = get_db_path()
    print(f"[INFO] Connecting to: {db_path}")
    con = connect_db(db_path, read_only=True)

    print("[INFO] Aggregating raw_count / group_count by month...")
    # group_count: 同一input_path_str（raw_pathの最終ステップを除いた履歴）を持つ
    # サンプルの集合数。_build_groups()と同一定義（'>'.join(parts[:-1])）。
    # raw_pathに'>'を含まない（最初の1手のみの）サンプルは input_path_str='' とする。
    df = con.execute("""
        SELECT
            substr(collection_date, 1, 7) AS month,
            COUNT(*) AS raw_count,
            COUNT(DISTINCT CASE WHEN strpos(raw_path, '>') = 0 THEN ''
                                 ELSE regexp_replace(raw_path, '>[^>]*$', '')
                            END) AS group_count
        FROM samples
        WHERE length(collection_date) >= 7 AND substr(collection_date, 5, 1) = '-'
        GROUP BY month
        ORDER BY month
    """).fetchdf()
    con.close()

    if df.empty:
        print("[WARN] No samples with valid collection_date found.")
        return

    df['group_ratio'] = df['group_count'] / df['raw_count']
    print(f"[INFO] {len(df)} months aggregated (total raw={df['raw_count'].sum():,}, "
          f"total groups={df['group_count'].sum():,})")

    output_dir = os.path.join(config.OUTPUT_DIR, 'scripts', 'group_count_by_month')
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, 'group_count_by_month.csv')
    df.to_csv(csv_path, index=False)
    print(f"[INFO] Saved: {csv_path}")

    # --- プロット（上段: raw_count/group_count の推移、下段: グループ化率） ---
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(max(14, len(df) * 0.28), 8), sharex=True,
        gridspec_kw={'height_ratios': [3, 1]},
    )

    x = range(len(df))
    ax1.plot(x, df['raw_count'], marker='o', markersize=3, linewidth=2,
              color=_COLOR_RAW, label='raw_count（グルーピング前）')
    ax1.plot(x, df['group_count'], marker='o', markersize=3, linewidth=2,
              color=_COLOR_GROUP, label='group_count（グルーピング後）')
    ax1.set_yscale('log')
    ax1.set_ylabel('Sample / Group count (log scale)')
    ax1.set_title('月別 共起グルーピング前後のサンプル数・グループ数')
    ax1.legend(loc='upper left', frameon=False)
    ax1.grid(axis='y', alpha=0.3, linewidth=0.5)

    ax2.plot(x, df['group_ratio'] * 100, marker='o', markersize=3, linewidth=2,
              color=_COLOR_GROUP)
    ax2.set_ylabel('group_count / raw_count (%)')
    ax2.set_xlabel('Month')
    ax2.grid(axis='y', alpha=0.3, linewidth=0.5)

    tick_step = max(1, len(df) // 40)
    ax1.set_xticks(list(x)[::tick_step])
    ax2.set_xticks(list(x)[::tick_step])
    ax2.set_xticklabels(df['month'].iloc[::tick_step], rotation=45, ha='right', fontsize=7)

    for spine in ('top', 'right'):
        ax1.spines[spine].set_visible(False)
        ax2.spines[spine].set_visible(False)

    plt.tight_layout()
    png_path = os.path.join(output_dir, 'group_count_by_month.png')
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved: {png_path}")


if __name__ == '__main__':
    main()
