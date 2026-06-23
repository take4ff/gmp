import duckdb

def main():
    db_path = "/mnt/ssd1/home3/aiba/gmp/db/features_7dba6a01.duckdb"
    print(f"Connecting to {db_path}...")
    con = duckdb.connect(db_path)
    
    # テーブル一覧の取得
    tables = con.execute("SHOW TABLES").fetchall()
    print("Tables:", tables)
    
    for table_tuple in tables:
        table_name = table_tuple[0]
        print(f"\n--- Schema for table: {table_name} ---")
        schema = con.execute(f"DESCRIBE {table_name}").fetchall()
        for col in schema:
            print(col)
            
        # 先頭数行の表示
        print(f"\nSample data from {table_name}:")
        sample = con.execute(f"SELECT * FROM {table_name} LIMIT 3").fetchdf()
        print(sample)
        
    con.close()

if __name__ == "__main__":
    main()
