# --- transformer_260707/scripts/test_dataloader_filter.py ---
# Valid / Test スプリットのターゲット共起数フィルタ効果を検証する。
# Usage: python -m transformer_260707.scripts.test_dataloader_filter
import pickle
from transformer_260707 import config
from transformer_260707.db.connection import get_db_path, connect_db


def main():
    db_path = get_db_path()
    print(f"[INFO] Using DB: {db_path}")
    con = connect_db(db_path, read_only=True)

    for split_type, split_name in [(1, 'Valid'), (2, 'Test')]:
        rows = con.execute("""
            SELECT s.sample_id, l.targets
            FROM samples s
            JOIN labels l ON s.sample_id = l.sample_id
            WHERE s.split_type = ?
            LIMIT 5000
        """, [split_type]).fetchall()

        total = len(rows)
        if total == 0:
            print(f"[WARNING] No samples found in {split_name} split")
            continue

        threshold = getattr(config, 'EVAL_MAX_Y_CO_OCCURRENCE', 5)
        ex_count = sum(1 for row in rows if len(pickle.loads(row[1])) > threshold)

        print(f"\n[{split_name} Split (Sampled {total:,} rows)]")
        print(f"  Target co-occurrence > {threshold} (excluded) : "
              f"{ex_count:,} ({ex_count / total * 100:.2f}%)")
        print(f"  Target co-occurrence <= {threshold} (kept)    : "
              f"{total - ex_count:,} ({(total - ex_count) / total * 100:.2f}%)")

    con.close()


if __name__ == '__main__':
    main()
