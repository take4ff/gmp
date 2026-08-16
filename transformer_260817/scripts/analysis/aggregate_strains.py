# --- transformer_260817/scripts/aggregate_strains.py ---
# DuckDB の strains / samples テーブルから株別サンプル数を集計する。
# Usage: python -m transformer_260817.scripts.aggregate_strains
import os
from transformer_260817 import config
from transformer_260817.db.connection import get_db_path, connect_db


def main():
    db_path = get_db_path()
    output_dir = os.path.join(config.OUTPUT_DIR, 'scripts', 'lineage')
    os.makedirs(output_dir, exist_ok=True)

    print(f"Connecting to {db_path}...")
    con = connect_db(db_path, read_only=True)

    # strains テーブルの事前集計値
    print("Querying strains table (pre-calculated sample_count)...")
    df_strains = con.execute("""
        SELECT strain_name, sample_count
        FROM strains
        ORDER BY sample_count DESC
    """).fetchdf()
    total_strains_count = df_strains['sample_count'].sum()
    df_strains['Percentage'] = df_strains['sample_count'] / total_strains_count * 100

    print("\n--- TOP 30 FROM strains TABLE ---")
    print(df_strains.head(30).to_string(index=False))

    # samples テーブルから実件数を JOIN 集計
    print("\nQuerying samples table and joining with strains...")
    df_samples_joined = con.execute("""
        SELECT s.strain_name, COUNT(*) as sample_count
        FROM samples sa
        JOIN strains s ON sa.strain_id = s.strain_id
        GROUP BY s.strain_name
        ORDER BY sample_count DESC
    """).fetchdf()
    total_samples_count = df_samples_joined['sample_count'].sum()
    df_samples_joined['Percentage'] = df_samples_joined['sample_count'] / total_samples_count * 100

    print("\n--- TOP 30 FROM samples TABLE (JOINED) ---")
    print(df_samples_joined.head(30).to_string(index=False))

    output_path = os.path.join(output_dir, 'db_lineage_counts.csv')
    df_samples_joined.to_csv(output_path, index=False)
    print(f"\nSaved all DB lineage counts to {output_path}")

    con.close()


if __name__ == '__main__':
    main()
