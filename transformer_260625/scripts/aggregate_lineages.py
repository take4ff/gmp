# --- transformer_260625/scripts/aggregate_lineages.py ---
# 全体での系統別サンプル数を集計する。
# Usage: python -m transformer_260625.scripts.aggregate_lineages
import os
import pandas as pd
from transformer_260625 import config


def main():
    csv_path = os.path.abspath(config.SEQUENCES_CSV)
    output_dir = os.path.join(config.OUTPUT_DIR, 'scripts')
    os.makedirs(output_dir, exist_ok=True)

    print(f"Reading CSV from {csv_path} and counting lineages...")
    chunksize = 1_000_000
    counts = {}
    total_rows = 0

    for chunk in pd.read_csv(csv_path, usecols=['Pangolin'], chunksize=chunksize):
        total_rows += len(chunk)
        print(f"Processed {total_rows} rows...")
        chunk['Pangolin'] = chunk['Pangolin'].fillna('Unknown')
        for p, count in chunk['Pangolin'].value_counts().items():
            counts[p] = counts.get(p, 0) + count

    print("Formatting counts...")
    df = pd.DataFrame(list(counts.items()), columns=['Pangolin', 'Count'])
    df = df.sort_values(by='Count', ascending=False).reset_index(drop=True)
    df['Percentage'] = df['Count'] / df['Count'].sum() * 100

    output_path = os.path.join(output_dir, 'lineage_counts.csv')
    df.to_csv(output_path, index=False)
    print(f"Saved all lineage counts to {output_path}")

    print("\n--- TOP 30 LINEAGES PREVIEW ---")
    print(df.head(30).to_string(index=False))


if __name__ == '__main__':
    main()
