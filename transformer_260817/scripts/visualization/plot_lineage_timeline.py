# --- transformer_260817/scripts/plot_lineage_timeline.py ---
# Pango系統の系譜と拡大時期を3種類の図で可視化する。
#   図1: 積み上げ面グラフ（月別割合）
#   図2: Ganttチャート（拡大期間）
#   図3: バブルチャート（月別×系統、バブルサイズ=サンプル数）
# Usage: python -m transformer_260817.scripts.plot_lineage_timeline
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba

from transformer_260817 import config
from transformer_260817.db.connection import get_db_path, connect_db

# ------------------------------------------------------------------ #
# 系統グループ定義（Pango命名規則から主要変異株にマッピング）
# ------------------------------------------------------------------ #
GROUP_ORDER = [
    'Early (A/B)',
    'Alpha',
    'Beta/Gamma',
    'Delta (AY)',
    'BA.1',
    'BA.2',
    'BA.4/BA.5',
    'BQ.1',
    'XBB/EG/HK',
    'BA.2.86/JN.1',
    'Other',
]

GROUP_COLORS = {
    'Early (A/B)':     '#aaaaaa',
    'Alpha':           '#d62728',
    'Beta/Gamma':      '#ff7f0e',
    'Delta (AY)':      '#1f77b4',
    'BA.1':            '#2ca02c',
    'BA.2':            '#98df8a',
    'BA.4/BA.5':       '#ffbb78',
    'BQ.1':            '#e377c2',
    'XBB/EG/HK':       '#9467bd',
    'BA.2.86/JN.1':    '#17becf',
    'Other':           '#dddddd',
}


def get_major_group(name: str) -> str:
    if not name or name == 'unclassifiable':
        return 'Other'
    n = name.upper()
    if n in ('A', 'B') or n.startswith('B.1.') or n.startswith('B.2') or n.startswith('C.'):
        # Alpha / Beta を先に除外
        if n.startswith('B.1.1.7'):
            return 'Alpha'
        if n.startswith('B.1.351') or n.startswith('B.1.1.28') or n.startswith('P.'):
            return 'Beta/Gamma'
        if n.startswith('B.1.617') or n.startswith('B.1.1.529'):
            pass  # 下のBA.*チェックへ
        return 'Early (A/B)'
    if name.startswith('B.1.1.7'):
        return 'Alpha'
    if name.startswith('B.1.351') or name.startswith('P.1') or name.startswith('P.2'):
        return 'Beta/Gamma'
    if name.startswith('AY.') or name == 'B.1.617.2':
        return 'Delta (AY)'
    if name.startswith('BA.1'):
        return 'BA.1'
    if (name.startswith('BA.2.86') or name.startswith('JN.') or
            name.startswith('KP.') or name.startswith('LB.') or
            name.startswith('MC.') or name.startswith('NB.')):
        return 'BA.2.86/JN.1'
    if name.startswith('BA.2'):
        return 'BA.2'
    if (name.startswith('BA.4') or name.startswith('BA.5') or
            name.startswith('BF.') or name.startswith('BE.')):
        return 'BA.4/BA.5'
    if name.startswith('BQ.'):
        return 'BQ.1'
    if (name.startswith('XBB') or name.startswith('EG.') or
            name.startswith('HK.') or name.startswith('XD') or
            name.startswith('GE.') or name.startswith('FL.')):
        return 'XBB/EG/HK'
    return 'Other'


def load_data() -> pd.DataFrame:
    con = connect_db(get_db_path(), read_only=True)
    rows = con.execute("""
        SELECT
            substr(s.collection_date, 1, 7) AS month,
            st.strain_name_usher            AS lineage,
            COUNT(*)                        AS n,
            AVG(s.path_length)              AS avg_path
        FROM samples s JOIN strains st ON s.strain_id = st.strain_id
        WHERE length(s.collection_date) >= 7
          AND substr(s.collection_date, 5, 1) = '-'
          AND st.strain_name_usher IS NOT NULL
        GROUP BY month, lineage
        ORDER BY month, n DESC
    """).fetchall()
    con.close()

    df = pd.DataFrame(rows, columns=['month', 'lineage', 'n', 'avg_path'])
    df['group'] = df['lineage'].apply(get_major_group)
    return df


# ------------------------------------------------------------------ #
# 図1: 積み上げ面グラフ
# ------------------------------------------------------------------ #
def plot_stacked_area(df: pd.DataFrame, output_dir: str):
    monthly = df.groupby(['month', 'group'])['n'].sum().reset_index()
    pivot = monthly.pivot(index='month', columns='group', values='n').fillna(0)
    pivot = pivot.reindex(columns=GROUP_ORDER, fill_value=0)
    prop = pivot.div(pivot.sum(axis=1), axis=0) * 100

    months = prop.index.tolist()
    x = range(len(months))

    fig, ax = plt.subplots(figsize=(18, 6))
    bottom = np.zeros(len(months))
    for group in GROUP_ORDER:
        vals = prop[group].values
        ax.fill_between(x, bottom, bottom + vals,
                        color=GROUP_COLORS[group], alpha=0.85, label=group)
        bottom += vals

    ax.set_xlim(0, len(months) - 1)
    ax.set_ylim(0, 100)
    tick_step = 3
    ax.set_xticks([i for i in x if i % tick_step == 0])
    ax.set_xticklabels([months[i] for i in x if i % tick_step == 0],
                       rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Proportion (%)')
    ax.set_title('Monthly lineage proportion by major Pango group')
    ax.legend(loc='upper left', fontsize=7, ncol=2, framealpha=0.7)
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'lineage_timeline_1_stacked.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[INFO] Saved: {path}")


# ------------------------------------------------------------------ #
# 図2: Ganttチャート
# ------------------------------------------------------------------ #
def plot_gantt(df: pd.DataFrame, output_dir: str):
    monthly = df.groupby(['month', 'group'])['n'].sum().reset_index()
    all_months = sorted(df['month'].unique())
    month_idx = {m: i for i, m in enumerate(all_months)}

    # グループごとに first/last/peak を計算
    records = []
    for group in GROUP_ORDER:
        sub = monthly[monthly['group'] == group]
        if sub.empty:
            continue
        first = sub['month'].min()
        last = sub['month'].max()
        peak_row = sub.loc[sub['n'].idxmax()]
        records.append({
            'group': group,
            'first': first,
            'last': last,
            'peak_month': peak_row['month'],
            'peak_n': peak_row['n'],
            'total_n': sub['n'].sum(),
        })
    rec_df = pd.DataFrame(records)

    # GROUP_ORDERの並び順（上から古い順）
    rec_df['order'] = rec_df['group'].apply(lambda g: GROUP_ORDER.index(g))
    rec_df = rec_df.sort_values('order', ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(18, 6))
    max_total = rec_df['total_n'].max()

    for i, row in rec_df.iterrows():
        x0 = month_idx[row['first']]
        x1 = month_idx[row['last']]
        color = GROUP_COLORS[row['group']]
        alpha = 0.4 + 0.5 * math.log1p(row['total_n']) / math.log1p(max_total)
        ax.barh(i, x1 - x0 + 1, left=x0, height=0.6,
                color=color, alpha=alpha, edgecolor='white', linewidth=0.5)
        # ピーク月マーカー
        xp = month_idx[row['peak_month']]
        ax.plot(xp + 0.5, i, 'v', color='black', markersize=5, zorder=5)
        # サンプル数ラベル
        ax.text(x1 + 0.3, i, f"{row['total_n'] / 1e6:.1f}M" if row['total_n'] >= 1e6
                else f"{row['total_n'] / 1e3:.0f}K",
                va='center', fontsize=7, color='dimgray')

    ax.set_yticks(range(len(rec_df)))
    ax.set_yticklabels(rec_df['group'].tolist(), fontsize=9)
    tick_step = 3
    ax.set_xticks([i for i in range(len(all_months)) if i % tick_step == 0])
    ax.set_xticklabels([all_months[i] for i in range(len(all_months)) if i % tick_step == 0],
                       rotation=45, ha='right', fontsize=8)
    ax.set_xlim(-0.5, len(all_months))
    ax.set_title('Lineage expansion timeline  (▼ = peak month,  bar opacity ∝ total samples)')
    ax.grid(axis='x', linestyle='--', alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, 'lineage_timeline_2_gantt.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[INFO] Saved: {path}")


# ------------------------------------------------------------------ #
# 図3: バブルチャート
# ------------------------------------------------------------------ #
def plot_bubble(df: pd.DataFrame, output_dir: str):
    monthly = df.groupby(['month', 'group'])['n'].sum().reset_index()
    all_months = sorted(df['month'].unique())
    month_idx = {m: i for i, m in enumerate(all_months)}

    # Y軸: GROUP_ORDERの並び順（上が新しい）
    group_y = {g: i for i, g in enumerate(reversed(GROUP_ORDER))}

    max_n = monthly['n'].max()
    scale = 2000  # バブル面積スケール

    fig, ax = plt.subplots(figsize=(18, 7))
    for _, row in monthly.iterrows():
        if row['n'] == 0:
            continue
        x = month_idx[row['month']]
        y = group_y[row['group']]
        size = (row['n'] / max_n) * scale
        ax.scatter(x, y, s=size, color=GROUP_COLORS[row['group']],
                   alpha=0.6, edgecolors='none')

    ax.set_yticks(range(len(GROUP_ORDER)))
    ax.set_yticklabels(list(reversed(GROUP_ORDER)), fontsize=9)
    tick_step = 3
    ax.set_xticks([i for i in range(len(all_months)) if i % tick_step == 0])
    ax.set_xticklabels([all_months[i] for i in range(len(all_months)) if i % tick_step == 0],
                       rotation=45, ha='right', fontsize=8)
    ax.set_xlim(-1, len(all_months))
    ax.set_title('Monthly sample count by lineage group  (bubble size ∝ sample count)')
    ax.grid(linestyle='--', alpha=0.3)

    # 凡例（バブルサイズ）
    for ref_n, label in [(100_000, '100K'), (500_000, '500K'), (1_000_000, '1M')]:
        ax.scatter([], [], s=(ref_n / max_n) * scale, color='gray',
                   alpha=0.6, label=label, edgecolors='none')
    ax.legend(title='Samples', loc='upper left', fontsize=7, framealpha=0.7)

    plt.tight_layout()
    path = os.path.join(output_dir, 'lineage_timeline_3_bubble.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[INFO] Saved: {path}")


# ------------------------------------------------------------------ #
# main
# ------------------------------------------------------------------ #
def main():
    output_dir = os.path.join(config.OUTPUT_DIR, 'scripts', 'lineage')
    os.makedirs(output_dir, exist_ok=True)

    print("[INFO] Loading data...")
    df = load_data()
    print(f"[INFO] Rows: {len(df):,}, Months: {df['month'].nunique()}, Lineages: {df['lineage'].nunique()}")

    print("[INFO] Plotting figure 1: stacked area...")
    plot_stacked_area(df, output_dir)

    print("[INFO] Plotting figure 2: Gantt...")
    plot_gantt(df, output_dir)

    print("[INFO] Plotting figure 3: bubble...")
    plot_bubble(df, output_dir)

    print("[INFO] Done.")


if __name__ == '__main__':
    main()
