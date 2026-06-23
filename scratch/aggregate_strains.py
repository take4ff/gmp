import duckdb
import pandas as pd
import os

def main():
    db_path = "/mnt/ssd1/home3/aiba/gmp/db/features_7dba6a01.duckdb"
    output_dir = "/mnt/ssd1/home3/aiba/gmp/scratch/outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Connecting to {db_path}...")
    con = duckdb.connect(db_path)
    
    # 1. strainsテーブルからの集計 (sample_countカラム)
    print("Querying strains table (pre-calculated sample_count)...")
    df_strains = con.execute("""
        SELECT strain_name, sample_count 
        FROM strains 
        ORDER BY sample_count DESC
    """).fetchdf()
    
    total_strains_count = df_strains['sample_count'].sum()
    df_strains['Percentage'] = (df_strains['sample_count'] / total_strains_count) * 100
    
    print("\n--- TOP 30 FROM strains TABLE ---")
    print(df_strains.head(30).to_string(index=False))
    
    # 2. samplesテーブルをベースにJOINして実際の件数をカウント
    print("\nQuerying samples table and joining with strains...")
    df_samples_joined = con.execute("""
        SELECT s.strain_name, COUNT(*) as sample_count
        FROM samples sa
        JOIN strains s ON sa.strain_id = s.strain_id
        GROUP BY s.strain_name
        ORDER BY sample_count DESC
    """).fetchdf()
    
    total_samples_count = df_samples_joined['sample_count'].sum()
    df_samples_joined['Percentage'] = (df_samples_joined['sample_count'] / total_samples_count) * 100
    
    print("\n--- TOP 30 FROM samples TABLE (JOINED) ---")
    print(df_samples_joined.head(30).to_string(index=False))
    
    # CSVに書き出し
    output_path = os.path.join(output_dir, "db_lineage_counts.csv")
    df_samples_joined.to_csv(output_path, index=False)
    print(f"\nSaved all DB lineage counts to {output_path}")
    
    con.close()

if __name__ == "__main__":
    main()
