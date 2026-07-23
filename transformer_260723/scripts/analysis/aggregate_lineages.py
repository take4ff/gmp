# --- transformer_260723/scripts/aggregate_lineages.py ---
# 全体での系統別サンプル数・採取日（中央値・平均値）を集計する。
# Usage: python -m transformer_260723.scripts.analysis.aggregate_lineages
import os
from collections import defaultdict

import numpy as np
import pandas as pd

from transformer_260723 import config


def _to_ordinals(series: pd.Series) -> pd.Series:
    """採取日列（'YYYY-MM-DD' / 'YYYY-MM' / 'YYYY'）を日付序数に変換。不正値は NaN。"""
    s = series.str.strip()
    # 桁数で補完
    s = s.where(s.str.len() != 4,  s + '-01-01')
    s = s.where(s.str.len() != 7,  s + '-01')
    dates = pd.to_datetime(s, errors='coerce')
    return dates.map(lambda d: d.toordinal() if pd.notna(d) else np.nan)


def main():
    csv_path = os.path.abspath(config.SEQUENCES_CSV)
    output_dir = os.path.join(config.OUTPUT_DIR, 'scripts', 'lineage')
    os.makedirs(output_dir, exist_ok=True)

    print(f"Reading CSV: {csv_path}")
    chunksize = 1_000_000

    counts: dict[str, int] = defaultdict(int)
    # 系統ごとの採取日序数リスト
    date_ordinals: dict[str, list] = defaultdict(list)

    total_rows = 0
    for chunk in pd.read_csv(csv_path, usecols=['Pangolin', 'Collection_Date'],
                              chunksize=chunksize):
        total_rows += len(chunk)
        print(f"  processed {total_rows:,} rows...")

        chunk['Pangolin'] = chunk['Pangolin'].fillna('Unknown')
        chunk['ordinal'] = _to_ordinals(chunk['Collection_Date'].fillna(''))

        # カウント集計
        for pangolin, cnt in chunk['Pangolin'].value_counts().items():
            counts[pangolin] += cnt

        # 有効な日付のみ系統別に蓄積
        valid = chunk.dropna(subset=['ordinal'])
        for pangolin, grp in valid.groupby('Pangolin'):
            date_ordinals[pangolin].extend(grp['ordinal'].tolist())

    print("Computing date statistics...")
    records = []
    for pangolin, count in counts.items():
        ords = date_ordinals.get(pangolin, [])
        if ords:
            arr = np.array(ords)
            median_date = pd.Timestamp.fromordinal(int(np.median(arr))).date().isoformat()
            mean_date   = pd.Timestamp.fromordinal(int(np.mean(arr))).date().isoformat()
        else:
            median_date = ''
            mean_date   = ''
        records.append({
            'Pangolin':    pangolin,
            'Count':       count,
            'Median_Date': median_date,
            'Mean_Date':   mean_date,
        })

    df = pd.DataFrame(records)
    df['Percentage'] = df['Count'] / df['Count'].sum() * 100
    df = df.sort_values('Count', ascending=False).reset_index(drop=True)
    df = df[['Pangolin', 'Count', 'Percentage', 'Median_Date', 'Mean_Date']]

    output_path = os.path.join(output_dir, 'lineage_counts.csv')
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")

    print("\n--- TOP 30 LINEAGES ---")
    print(df.head(30).to_string(index=False))


if __name__ == '__main__':
    main()
