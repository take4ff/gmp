# --- transformer_260625/scripts/inspect_duckdb.py ---
# DuckDB のテーブル一覧・スキーマ・先頭データを表示する。
# Usage: python -m transformer_260625.scripts.inspect_duckdb
from transformer_260625.db.connection import get_db_path, connect_db


def main():
    db_path = get_db_path()
    print(f"Connecting to {db_path}...")
    con = connect_db(db_path, read_only=True)

    tables = con.execute("SHOW TABLES").fetchall()
    print("Tables:", tables)

    for (table_name,) in tables:
        print(f"\n--- Schema for table: {table_name} ---")
        for col in con.execute(f"DESCRIBE {table_name}").fetchall():
            print(col)
        print(f"\nSample data from {table_name}:")
        print(con.execute(f"SELECT * FROM {table_name} LIMIT 3").fetchdf())

    con.close()


if __name__ == '__main__':
    main()
