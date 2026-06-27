# --- transformer_260625/scripts/inspect_duckdb.py ---
# DuckDB のテーブル一覧・スキーマ・先頭データを表示する。
# Usage: python -m transformer_260625.scripts.inspect_duckdb
from transformer_260625.db.connection import get_db_path, connect_db

_SPLIT_LABEL = {-1: 'excluded', 0: 'train', 1: 'valid', 2: 'test'}


def _print_split_dist(con, col, title):
    print(f"\n--- {title} ---")
    rows = con.execute(f"""
        SELECT {col}, COUNT(*) AS n
        FROM samples
        GROUP BY {col}
        ORDER BY {col}
    """).fetchall()
    total = sum(r[1] for r in rows)
    for val, n in rows:
        label = _SPLIT_LABEL.get(val, str(val))
        print(f"  {label:>10} ({val:>2}): {n:>10,}  ({100*n/total:.1f}%)")
    print(f"  {'total':>10}     : {total:>10,}")


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

    _print_split_dist(con, 'split_type',      'Split distribution (timestep mode)')
    _print_split_dist(con, 'split_type_date', 'Split distribution (date mode)')
    _print_split_dist(con, 'split_type_wf',   'Split distribution (walk_forward mode — last fold)')

    con.close()


if __name__ == '__main__':
    main()
