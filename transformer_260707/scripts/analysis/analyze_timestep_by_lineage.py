# --- transformer_260707/scripts/analyze_timestep_by_lineage.py ---
# 指定月のタイムステップ数（path_length）分布を系統別に分析する。
# 分布が広い月（例: 2021-12, 2023-12）の原因系統を特定するために使用。
# Usage: python -m transformer_260707.scripts.analyze_timestep_by_lineage
import os
import pandas as pd
from transformer_260707 import config
from transformer_260707.db.connection import get_db_path, connect_db

TARGET_MONTHS = ['2021-12', '2023-12']
TOP_N = 10
MIN_SAMPLES = 10


def analyze_month(con, month: str) -> dict:
    stats = con.execute(f"""
        SELECT
            COUNT(*)                                                          AS n,
            MIN(path_length)                                                  AS p_min,
            percentile_cont(0.10) WITHIN GROUP (ORDER BY path_length)        AS p10,
            percentile_cont(0.25) WITHIN GROUP (ORDER BY path_length)        AS q1,
            percentile_cont(0.50) WITHIN GROUP (ORDER BY path_length)        AS q2,
            percentile_cont(0.75) WITHIN GROUP (ORDER BY path_length)        AS q3,
            percentile_cont(0.90) WITHIN GROUP (ORDER BY path_length)        AS p90,
            MAX(path_length)                                                  AS p_max
        FROM samples
        WHERE substr(collection_date, 1, 7) = '{month}'
    """).fetchone()

    long_lineages = con.execute(f"""
        SELECT st.strain_name_usher, AVG(s.path_length) AS avg_pl, COUNT(*) AS n
        FROM samples s JOIN strains st ON s.strain_id = st.strain_id
        WHERE substr(s.collection_date, 1, 7) = '{month}'
        GROUP BY st.strain_name_usher
        HAVING n >= {MIN_SAMPLES}
        ORDER BY avg_pl DESC
        LIMIT {TOP_N}
    """).fetchall()

    short_lineages = con.execute(f"""
        SELECT st.strain_name_usher, AVG(s.path_length) AS avg_pl, COUNT(*) AS n
        FROM samples s JOIN strains st ON s.strain_id = st.strain_id
        WHERE substr(s.collection_date, 1, 7) = '{month}'
        GROUP BY st.strain_name_usher
        HAVING n >= {MIN_SAMPLES}
        ORDER BY avg_pl ASC
        LIMIT {TOP_N}
    """).fetchall()

    return {
        'stats': stats,
        'long': long_lineages,
        'short': short_lineages,
    }


def main():
    db_path = get_db_path()
    print(f"[INFO] Connecting to: {db_path}\n")
    con = connect_db(db_path, read_only=True)

    output_dir = os.path.join(config.OUTPUT_DIR, 'scripts', 'timestep')
    os.makedirs(output_dir, exist_ok=True)

    all_rows = []

    for month in TARGET_MONTHS:
        result = analyze_month(con, month)
        n, p_min, p10, q1, q2, q3, p90, p_max = result['stats']

        print(f"{'=' * 60}")
        print(f"  {month}  (N={n:,})")
        print(f"{'=' * 60}")
        print(f"  min={p_min}  P10={p10:.0f}  Q1={q1:.0f}  Med={q2:.0f}  "
              f"Q3={q3:.0f}  P90={p90:.0f}  max={p_max}")

        print(f"\n  [長パス系統 Top{TOP_N}]")
        for lineage, avg_pl, cnt in result['long']:
            print(f"    {lineage:<35} avg_path={avg_pl:.1f}  n={cnt:,}")
            all_rows.append({'month': month, 'group': 'long', 'lineage': lineage,
                             'avg_path_length': round(avg_pl, 2), 'n': cnt})

        print(f"\n  [短パス系統 Top{TOP_N}]")
        for lineage, avg_pl, cnt in result['short']:
            print(f"    {lineage:<35} avg_path={avg_pl:.1f}  n={cnt:,}")
            all_rows.append({'month': month, 'group': 'short', 'lineage': lineage,
                             'avg_path_length': round(avg_pl, 2), 'n': cnt})
        print()

    con.close()

    out_csv = os.path.join(output_dir, 'timestep_by_lineage.csv')
    pd.DataFrame(all_rows).to_csv(out_csv, index=False)
    print(f"[INFO] Saved: {out_csv}")


if __name__ == '__main__':
    main()
