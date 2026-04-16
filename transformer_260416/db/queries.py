# --- db/queries.py ---
# SQL関連クエリ（サンプル取得・統計・分割割り当て）

import random
import pickle
from .connection import connect_db, get_db_path
from .. import config


def get_db_data_info(db_path=None, split_type=None, max_cooccurrence=None):
    """DBからデータ情報を取得（data_info相当）。

    Returns:
        dict: train/val/test件数・長さ範囲・流行度閾値を含む辞書
    """
    if db_path is None:
        db_path = get_db_path()

    con = connect_db(db_path, read_only=True)

    query_base = "SELECT MIN(path_length), MAX(path_length), COUNT(*) FROM samples WHERE 1=1"

    def get_stats(st):
        q = query_base + f" AND split_type = {st}"
        if max_cooccurrence:
            q += f" AND max_cooccurrence <= {max_cooccurrence}"
        return con.execute(q).fetchone()

    train_stats = get_stats(0)
    valid_stats = get_stats(1)
    test_stats = get_stats(2)

    # 流行度閾値
    strength_stats = con.execute("SELECT MIN(strength_score), MAX(strength_score) FROM samples").fetchone()
    min_s, max_s = strength_stats
    score_range = (max_s - min_s) if max_s and min_s else 10
    low_max = int(min_s + score_range / 3) + 1 if min_s else 6
    med_max = int(min_s + 2 * score_range / 3) + 1 if min_s else 10

    con.close()

    return {
        'train_min_len': train_stats[0] or 0,
        'train_max_len': train_stats[1] or 0,
        'train_count': train_stats[2] or 0,
        'val_min_len': valid_stats[0] or 0,
        'val_max_len': valid_stats[1] or 0,
        'val_count': valid_stats[2] or 0,
        'test_min_len': test_stats[0] or 0,
        'test_max_len': test_stats[1] or 0,
        'test_count': test_stats[2] or 0,
        'strength_low_max': low_max,
        'strength_med_max': med_max,
    }


def load_samples_from_db(db_path=None, split_type=None, max_cooccurrence=None,
                         min_length=None, max_length=None, limit=None):
    """DBからサンプルを読み込み、学習用データ形式に変換する。

    Args:
        db_path: DuckDBファイルパス
        split_type: 0=train, 1=valid, 2=test
        max_cooccurrence: 最大共起数フィルタ
        min_length, max_length: パス長フィルタ
        limit: 取得件数制限

    Returns:
        list of (x, y_targets, path_length, raw_path, strain_name, strength_score)
    """
    if db_path is None:
        db_path = get_db_path()

    con = connect_db(db_path, read_only=True)

    query = """
        SELECT s.sample_id, s.raw_path, s.path_length, s.strength_score,
               st.strain_name
        FROM samples s
        JOIN strains st ON s.strain_id = st.strain_id
        WHERE 1=1
    """
    params = []

    if split_type is not None:
        query += " AND s.split_type = ?"
        params.append(split_type)

    if max_cooccurrence is not None:
        query += " AND s.max_cooccurrence <= ?"
        params.append(max_cooccurrence)

    if min_length is not None:
        query += " AND s.path_length >= ?"
        params.append(min_length)

    if max_length is not None:
        query += " AND s.path_length <= ?"
        params.append(max_length)

    query += " ORDER BY s.sample_id"

    if limit is not None:
        query += f" LIMIT {limit}"

    samples = con.execute(query, params).fetchall()

    result = []
    for sample_id, raw_path, path_length, strength_score, strain_name in samples:
        features = con.execute("""
            SELECT timestep, cat_features, num_features
            FROM features
            WHERE sample_id = ?
            ORDER BY timestep
        """, [sample_id]).fetchall()

        x = []
        for ts, cat_blob, num_blob in features:
            cat_feat = pickle.loads(cat_blob)
            num_feat = pickle.loads(num_blob)
            x.append([(cat_feat, num_feat)])

        label_row = con.execute("""
            SELECT targets FROM labels WHERE sample_id = ?
        """, [sample_id]).fetchone()

        y_targets = pickle.loads(label_row[0]) if label_row else []
        result.append((x, y_targets, path_length, raw_path, strain_name, strength_score))

    con.close()
    return result


def assign_splits(con, train_ratio=0.8, valid_ratio=0.1):
    """パス長に基づいてtrain/valid/testに分割し、DBを更新する。"""
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [INFO] Assigning train/valid/test splits based on path length...")

    train_max = config.TRAIN_MAX
    valid_num = config.VALID_NUM
    overlap_start = train_max - valid_num + 1

    # Train (オーバーラップ外)
    con.execute(f"""
        UPDATE samples SET split_type = 0
        WHERE path_length < {overlap_start}
    """)

    # Test
    con.execute(f"""
        UPDATE samples SET split_type = 2
        WHERE path_length > {train_max}
    """)

    # オーバーラップ領域 (valid_ratio で分割)
    overlap_samples = con.execute(f"""
        SELECT sample_id FROM samples
        WHERE path_length >= {overlap_start} AND path_length <= {train_max}
        ORDER BY sample_id
    """).fetchall()

    if overlap_samples:
        sample_ids = [s[0] for s in overlap_samples]
        random.seed(config.SEED)
        random.shuffle(sample_ids)

        n_valid = int(len(sample_ids) * valid_ratio)
        valid_ids = sample_ids[:n_valid]
        train_ids = sample_ids[n_valid:]

        if valid_ids:
            con.execute(f"UPDATE samples SET split_type = 1 WHERE sample_id IN ({','.join(map(str, valid_ids))})")
        if train_ids:
            con.execute(f"UPDATE samples SET split_type = 0 WHERE sample_id IN ({','.join(map(str, train_ids))})")

    # 統計を表示
    stats = con.execute("SELECT * FROM split_stats").fetchall()
    for split_type, split_name, count in stats:
        print(f"[{timestamp}]   {split_name}: {count:,} samples")


def get_processed_strains(con):
    """DBから処理済みの株一覧を取得する（再開機能用）。"""
    try:
        result = con.execute("""
            SELECT DISTINCT st.strain_name
            FROM samples s
            JOIN strains st ON s.strain_id = st.strain_id
        """).fetchall()
        return set(r[0] for r in result)
    except Exception:
        return set()


def get_next_ids(con):
    """DBから次のID値を取得する（再開機能用）。"""
    try:
        sample_id = con.execute("SELECT COALESCE(MAX(sample_id), -1) + 1 FROM samples").fetchone()[0]
        feature_id = con.execute("SELECT COALESCE(MAX(feature_id), -1) + 1 FROM features").fetchone()[0]
        label_id = con.execute("SELECT COALESCE(MAX(label_id), -1) + 1 FROM labels").fetchone()[0]
        return sample_id, feature_id, label_id
    except Exception:
        return 0, 0, 0
