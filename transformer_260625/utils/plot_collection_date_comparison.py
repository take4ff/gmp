# --- transformer_260625.utils.plot_collection_date_comparison.py ---
# python -m transformer_260625.utils.plot_collection_date_comparison --db_path db/features_724e4a32.duckdb
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

# プロジェクトルートを sys.path に追加
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from transformer_260625 import config
from transformer_260625.db.connection import get_db_path, connect_db


def plot_comparison(db_path=None, output_dir=None):
    """CSVとDBそれぞれの採取日（年月単位）別のサンプル件数を比較したグラフを生成して保存する。"""
    csv_path = os.path.join(project_root, config.SEQUENCES_CSV)
    
    if db_path is None:
        db_path = get_db_path()
        
    db_basename = os.path.splitext(os.path.basename(db_path))[0]

    if output_dir is None:
        output_dir = os.path.join(project_root, "outputs/collection-date")
        
    os.makedirs(output_dir, exist_ok=True)
    output_png = os.path.join(output_dir, f"collection_date_comparison_{db_basename}.png")

    print(f"[INFO] Loading CSV from: {csv_path}")
    if not os.path.exists(csv_path):
        print(f"[ERROR] CSV file not found: {csv_path}")
        return

    # CSV の読み込みと集計 (Collection_Date カラムのみ)
    df_csv = pd.read_csv(csv_path, usecols=['Collection_Date'])
    # 文字列スライスにより YYYY-MM を抽出
    df_csv['year_month'] = df_csv['Collection_Date'].astype(str).str[:7]
    # YYYY-MM 形式（長さ7で5文字目が'-'）のみにフィルタリング
    df_csv = df_csv[(df_csv['year_month'].str.len() == 7) & (df_csv['year_month'].str.slice(4, 5) == '-')]
    csv_counts = df_csv['year_month'].value_counts().sort_index()

    print(f"[INFO] Loading DB from: {db_path}")
    if not os.path.exists(db_path):
        print(f"[ERROR] DB file not found: {db_path}")
        return

    # DB の読み込みと集計
    con = connect_db(db_path, read_only=True)
    db_data = con.execute("""
        SELECT SUBSTR(collection_date, 1, 7) as year_month, COUNT(*) as count
        FROM samples
        WHERE collection_date IS NOT NULL AND collection_date != ''
        GROUP BY year_month
        ORDER BY year_month
    """).fetchall()
    con.close()

    db_counts = pd.Series({row[0]: row[1] for row in db_data if row[0] is not None and len(row[0]) == 7}).sort_index()

    # データフレームに集約してアラインメント
    all_months = sorted(list(set(csv_counts.index) | set(db_counts.index)))
    df_compare = pd.DataFrame(index=all_months)
    df_compare['CSV (sequences-241017.csv)'] = csv_counts
    df_compare['DB (samples table)'] = db_counts
    df_compare = df_compare.fillna(0)

    # 期間フィルタ（COVID-19流行期間の 2019-12 から 2025-12 に制限）
    valid_months = [m for m in df_compare.index if '2019-12' <= m <= '2025-12']
    df_compare = df_compare.loc[valid_months]

    print("[INFO] Creating plot...")
    plt.figure(figsize=(14, 7))
    plt.plot(df_compare.index, df_compare['CSV (sequences-241017.csv)'], marker='o', label='CSV (sequences-241017.csv)', linewidth=2)
    plt.plot(df_compare.index, df_compare['DB (samples table)'], marker='s', label='DB (samples table)', linewidth=2)

    plt.title('Comparison of Sample Counts by Collection Date (Year-Month)', fontsize=16)
    plt.xlabel('Collection Date (Year-Month)', fontsize=12)
    plt.ylabel('Sample Count', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)
    plt.tight_layout()

    plt.savefig(output_png, dpi=300)
    print(f"[INFO] Graph successfully saved to {output_png}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--db_path', type=str, default=None, help='Path to the DuckDB file')
    parser.add_argument('--output_dir', type=str, default=None, help='Directory to save output comparison plot')
    args = parser.parse_args()
    plot_comparison(db_path=args.db_path, output_dir=args.output_dir)
