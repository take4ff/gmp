# --- transformer_260723/scripts/plot_timestep_by_month.py ---
# 月別のサンプルのタイムステップ数（path_length）分布を箱ひげ図でプロットする。
# ひげ: IQR×1.5 (外れ値は非表示), 各箱の上にサンプル数を表示。
# Usage: python -m transformer_260723.scripts.plot_timestep_by_month
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from transformer_260723 import config
from transformer_260723.db.connection import get_db_path, connect_db


def main():
    db_path = get_db_path()
    print(f"[INFO] Connecting to: {db_path}")
    con = connect_db(db_path, read_only=True)

    rows = con.execute("""
        SELECT
            substr(collection_date, 1, 7)                                    AS month,
            COUNT(*)                                                          AS n,
            MIN(path_length)                                                  AS p_min,
            percentile_cont(0.25) WITHIN GROUP (ORDER BY path_length)        AS q1,
            percentile_cont(0.50) WITHIN GROUP (ORDER BY path_length)        AS q2,
            percentile_cont(0.75) WITHIN GROUP (ORDER BY path_length)        AS q3,
            MAX(path_length)                                                  AS p_max
        FROM samples
        WHERE length(collection_date) >= 7
          AND substr(collection_date, 5, 1) = '-'
        GROUP BY month
        ORDER BY month
    """).fetchall()
    con.close()

    months = [r[0] for r in rows]
    stats = []
    for _, n, p_min, q1, q2, q3, p_max in rows:
        iqr = q3 - q1
        stats.append({
            'med':    q2,
            'q1':     q1,
            'q3':     q3,
            'whislo': max(p_min, q1 - 1.5 * iqr),
            'whishi': min(p_max, q3 + 1.5 * iqr),
            'fliers': [],
        })

    n_samples = [r[1] for r in rows]

    fig, ax = plt.subplots(figsize=(20, 6))

    # サンプル数の棒グラフ（背面・薄色）
    ax2 = ax.twinx()
    ax2.bar(range(len(months)), n_samples, color='steelblue', alpha=0.15, zorder=0)
    ax2.set_ylabel('Sample count', color='steelblue', fontsize=9)
    ax2.tick_params(axis='y', labelcolor='steelblue', labelsize=8)
    ax2.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f'{x/1000:.0f}K' if x >= 1000 else str(int(x)))
    )
    ax2.set_zorder(0)
    ax.set_zorder(1)
    ax.patch.set_visible(False)

    # 箱ひげ図（前面）
    ax.bxp(
        stats,
        positions=range(len(months)),
        showfliers=False,
        patch_artist=True,
        boxprops=dict(facecolor='steelblue', alpha=0.6),
        medianprops=dict(color='black', linewidth=1.5),
        whiskerprops=dict(linewidth=1.0),
        capprops=dict(linewidth=1.0),
    )

    # サンプル数を各箱の上に表示
    for i, r in enumerate(rows):
        n = r[1]
        label = f'{n / 1000:.0f}K' if n >= 1000 else str(n)
        ax.text(i, stats[i]['whishi'] + 0.3, label,
                ha='center', va='bottom', fontsize=6, color='dimgray')

    ax.set_xticks(range(len(months)))
    ax.set_xticklabels(months, rotation=45, ha='right', fontsize=8)
    ax.set_xlabel('Month')
    ax.set_ylabel('Timestep count (path_length)')
    ax.set_title('Distribution of timestep count per month')
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()

    output_dir = os.path.join(config.OUTPUT_DIR, 'scripts', 'timestep')
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, 'timestep_by_month.png')
    plt.savefig(out_path, dpi=150)
    print(f"[INFO] Saved: {out_path}")

    from transformer_260723.scripts.visualization._fold_annotate import add_fold_annotations
    add_fold_annotations(ax, months)
    ax.set_title(ax.get_title(), pad=30)
    out_path_fold = os.path.join(output_dir, 'timestep_by_month_by_fold.png')
    plt.savefig(out_path_fold, dpi=150, bbox_inches='tight')
    print(f"[INFO] Saved: {out_path_fold}")

    plt.close()
    print(f"[INFO] Months: {len(months)}, Total samples: {sum(r[1] for r in rows):,}")


if __name__ == '__main__':
    main()
