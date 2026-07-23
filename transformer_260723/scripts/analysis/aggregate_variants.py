# --- transformer_260723/scripts/aggregate_variants.py ---
# 月別・変異株別の出現数を集計する。
# Usage: python -m transformer_260723.scripts.aggregate_variants
import os
import re
import pandas as pd
from transformer_260723 import config


def main():
    csv_path = os.path.abspath(config.SEQUENCES_CSV)
    output_dir = os.path.join(config.OUTPUT_DIR, 'scripts', 'variants')
    os.makedirs(output_dir, exist_ok=True)

    print(f"Reading CSV from {csv_path}...")
    chunksize = 1_000_000
    counts = {}
    date_pat = re.compile(r'^(\d{4}-\d{2})')
    total_rows = 0

    for chunk in pd.read_csv(csv_path, usecols=['Pangolin', 'Collection_Date'], chunksize=chunksize):
        total_rows += len(chunk)
        print(f"Processed {total_rows} rows...")

        chunk['Pangolin'] = chunk['Pangolin'].fillna('Unknown')
        chunk['Collection_Date'] = chunk['Collection_Date'].fillna('Unknown')

        def get_month(x):
            m = date_pat.match(str(x))
            return m.group(1) if m else 'Unknown'

        chunk['Month'] = chunk['Collection_Date'].apply(get_month)

        grp = chunk.groupby(['Month', 'Pangolin']).size().reset_index(name='count')
        for _, row in grp.iterrows():
            m, p, c = row['Month'], row['Pangolin'], row['count']
            counts.setdefault(m, {})[p] = counts.get(m, {}).get(p, 0) + c

    print("Aggregation finished. Formatting data...")

    flat_data = [
        {'Month': month, 'Pangolin': p, 'Count': count}
        for month, p_counts in counts.items()
        for p, count in p_counts.items()
    ]
    df = pd.DataFrame(flat_data)
    df = df[df['Month'].str.match(r'^(201[9]|202[0-6])-\d{2}$')]

    df.to_csv(os.path.join(output_dir, 'monthly_variants_raw.csv'), index=False)
    print("Saved monthly_variants_raw.csv")

    # 各月の上位5系統
    monthly_top = []
    for month in sorted(df['Month'].unique()):
        m_df = df[df['Month'] == month].sort_values(by='Count', ascending=False)
        total_month = m_df['Count'].sum()
        top5 = m_df.head(5)
        top5_list = [f"{r['Pangolin']} ({r['Count']:,} / {r['Count']/total_month*100:.1f}%)"
                     for _, r in top5.iterrows()]
        while len(top5_list) < 5:
            top5_list.append('-')
        monthly_top.append({'Month': month, 'Total': total_month,
                            'Top 1': top5_list[0], 'Top 2': top5_list[1],
                            'Top 3': top5_list[2], 'Top 4': top5_list[3], 'Top 5': top5_list[4]})

    df_top = pd.DataFrame(monthly_top)
    df_top.to_csv(os.path.join(output_dir, 'monthly_top_variants.csv'), index=False)
    print("Saved monthly_top_variants.csv")

    # 上位15系統のピボット
    overall_top = df.groupby('Pangolin')['Count'].sum().reset_index()
    overall_top = overall_top[overall_top['Pangolin'] != 'Unknown']
    top15 = overall_top.sort_values(by='Count', ascending=False).head(15)['Pangolin'].tolist()
    df_pivot = (df[df['Pangolin'].isin(top15)]
                .pivot(index='Month', columns='Pangolin', values='Count')
                .fillna(0).astype(int).sort_index())
    df_pivot.to_csv(os.path.join(output_dir, 'monthly_top15_pivot.csv'))
    print("Saved monthly_top15_pivot.csv")

    print("\n--- MONTHLY TOP 3 VARIANTS PREVIEW ---")
    print(df_top[['Month', 'Total', 'Top 1', 'Top 2', 'Top 3']].to_string(index=False))


if __name__ == '__main__':
    main()
