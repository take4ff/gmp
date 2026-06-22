# --- scratch/test_dataloader_filter.py ---
import os
import sys
import pickle

project_root = "/mnt/ssd1/home3/aiba/gmp"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from transformer_260528 import config
from transformer_260528.db.connection import get_db_path, connect_db

def test_filter():
    db_path = get_db_path()
    print(f"[INFO] Using DB: {db_path}")

    con = connect_db(db_path, read_only=True)

    # Validスプリット(1)とTestスプリット(2)からそれぞれ2000件を取得
    for split_type, split_name in [(1, 'Valid'), (2, 'Test')]:
        query = f"""
            SELECT s.sample_id, l.targets
            FROM samples s
            JOIN labels l ON s.sample_id = l.sample_id
            WHERE s.split_type = ?
            LIMIT 5000
        """
        rows = con.execute(query, [split_type]).fetchall()
        
        total = len(rows)
        if total == 0:
            print(f"[WARNING] No samples found in {split_name} split")
            continue
            
        ex_count = 0
        for row in rows:
            targets = pickle.loads(row[1])
            # ターゲット（Y）の共起数が EVAL_MAX_Y_CO_OCCURRENCE (5) より大きいものをカウント
            if len(targets) > getattr(config, 'EVAL_MAX_Y_CO_OCCURRENCE', 5):
                ex_count += 1
                
        print(f"\n[{split_name} Split (Sampled {total:,} rows)]")
        print(f"  Target co-occurrence > 5 (To be excluded) : {ex_count:,} ({ex_count / total * 100:.2f}%)")
        print(f"  Target co-occurrence <= 5 (To be kept)    : {total - ex_count:,} ({(total - ex_count) / total * 100:.2f}%)")

    con.close()

if __name__ == "__main__":
    test_filter()
