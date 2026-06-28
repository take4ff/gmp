# --- db/connection.py ---
# DuckDB接続・初期化処理

import os
import duckdb
import hashlib
from .. import config


def get_feature_config_hash():
    """特徴量設定に基づくハッシュを生成（DBファイル名のバージョン管理用）"""
    relevant_configs = [
        str(config.NUM_FEATURE_STRING),
        str(config.NUM_CHEM_FEATURES),
        str(config.MAX_SEQ_LEN),
        str(config.TARGET_LEN),
        str(config.DATA_BASE_DIR),
        str(config.MAX_CO_OCCURRENCE),  # 最大共起数上限
        "v2_strength_clade", # スキーマバージョン情報追加（DB再構築を強制）
    ]
    config_string = "_".join(relevant_configs)
    return hashlib.md5(config_string.encode('utf-8')).hexdigest()


def get_db_path():
    """設定からDBパスを取得（設定ハッシュをファイル名に含める）"""
    db_file = getattr(config, 'DB_FILE', None)
    if db_file is not None:
        db_dir = os.path.dirname(db_file)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        return db_file

    db_dir = getattr(config, 'DB_DIR', 'cache/db')
    os.makedirs(db_dir, exist_ok=True)
    config_hash = get_feature_config_hash()[:8]
    return os.path.join(db_dir, f'features_{config_hash}.duckdb')


def connect_db(db_path=None, read_only=False):
    """DuckDB接続を取得する。

    Args:
        db_path: DuckDBファイルパス。Noneの場合はget_db_path()を使用
        read_only: 読み取り専用で接続するか

    Returns:
        duckdb.DuckDBPyConnection
    """
    if db_path is None:
        db_path = get_db_path()
    return duckdb.connect(db_path, read_only=read_only)


def init_db(db_path=None):
    """データベースを初期化（テーブル・インデックス・ビュー作成）。

    Args:
        db_path: DuckDBファイルパス。Noneの場合はget_db_path()を使用

    Returns:
        str: 初期化されたDBのパス
    """
    if db_path is None:
        db_path = get_db_path()

    con = connect_db(db_path)

    # 株マスタ
    con.execute("""
        CREATE TABLE IF NOT EXISTS strains (
            strain_id INTEGER PRIMARY KEY,
            strain_name VARCHAR UNIQUE NOT NULL,
            strain_name_ncbi VARCHAR,
            strain_name_usher VARCHAR,
            sample_count INTEGER DEFAULT 0,
            strength_score REAL DEFAULT 0.0,
            strength_score_ncbi REAL DEFAULT 0.0,
            strength_score_usher REAL DEFAULT 0.0,
            clade VARCHAR,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # サンプル
    con.execute("""
        CREATE TABLE IF NOT EXISTS samples (
            sample_id INTEGER PRIMARY KEY,
            strain_id INTEGER,
            raw_path VARCHAR,
            path_length INTEGER,
            max_cooccurrence INTEGER,
            split_type INTEGER DEFAULT 0,
            split_type_date INTEGER DEFAULT 0,
            split_type_wf INTEGER DEFAULT 0,
            strength_score REAL DEFAULT 0.0,
            collection_date VARCHAR,
            country VARCHAR
        )
    """)

    # 特徴量（X: 入力系列）
    con.execute("""
        CREATE TABLE IF NOT EXISTS features (
            feature_id INTEGER PRIMARY KEY,
            sample_id INTEGER,
            timestep INTEGER,
            cooccurrence INTEGER,
            cat_features BLOB,
            num_features BLOB
        )
    """)

    # ラベル（Y: 予測対象）
    con.execute("""
        CREATE TABLE IF NOT EXISTS labels (
            label_id INTEGER PRIMARY KEY,
            sample_id INTEGER UNIQUE,
            targets BLOB
        )
    """)

    # スキーマ情報
    con.execute("""
        CREATE TABLE IF NOT EXISTS schema_info (
            id INTEGER PRIMARY KEY,
            version INTEGER,
            num_cat_features INTEGER,
            num_num_features INTEGER,
            feature_config_hash VARCHAR,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)



    # 統計ビュー
    con.execute("""
        CREATE OR REPLACE VIEW split_stats AS
        SELECT
            split_type,
            CASE split_type WHEN 0 THEN 'train' WHEN 1 THEN 'valid' WHEN 2 THEN 'test' END as split_name,
            COUNT(*) as sample_count
        FROM samples
        GROUP BY split_type
    """)

    # スキーマ情報を登録
    config_hash = get_feature_config_hash()
    con.execute("DELETE FROM schema_info WHERE id = 1")
    con.execute("""
        INSERT INTO schema_info (id, version, num_cat_features, num_num_features, feature_config_hash, created_at)
        VALUES (1, 1, ?, ?, ?, CURRENT_TIMESTAMP)
    """, [config.NUM_FEATURE_STRING, config.NUM_CHEM_FEATURES, config_hash])

    con.close()
    print(f"[INFO] Database initialized: {db_path}")
    return db_path


def check_db_exists(db_path=None):
    """DBが存在し、スキーマ（設定ハッシュ）が一致するか確認する。

    Returns:
        Tuple[bool, str]: (存在・一致するか, メッセージ)
    """
    if db_path is None:
        db_path = get_db_path()

    if not os.path.exists(db_path):
        return False, "DB does not exist"

    try:
        con = connect_db(db_path, read_only=True)
        result = con.execute("SELECT feature_config_hash FROM schema_info LIMIT 1").fetchone()
        con.close()

        if result is None:
            return False, "No schema info found"

        db_hash = result[0]
        current_hash = get_feature_config_hash()

        if db_hash != current_hash:
            print(f"[WARNING] Config mismatch bypassed: DB={db_hash[:8]}, Current={current_hash[:8]}")

        return True, "OK"
    except Exception as e:
        return False, str(e)


def get_db_stats(db_path=None):
    """DBの統計情報を辞書で返す。"""
    if db_path is None:
        db_path = get_db_path()

    con = connect_db(db_path, read_only=True)

    stats = {}
    stats['strain_count'] = con.execute("SELECT COUNT(*) FROM strains").fetchone()[0]
    stats['sample_count'] = con.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
    stats['feature_count'] = con.execute("SELECT COUNT(*) FROM features").fetchone()[0]
    stats['label_count'] = con.execute("SELECT COUNT(*) FROM labels").fetchone()[0]

    split_stats = con.execute("SELECT * FROM split_stats").fetchall()
    stats['split_stats'] = {row[1]: row[2] for row in split_stats}

    con.close()
    return stats


def print_db_stats(db_path=None):
    """DBの統計情報を標準出力に表示する。"""
    stats = get_db_stats(db_path)
    print("\n[INFO] === Database Statistics ===")
    print(f"  Strains:  {stats['strain_count']:,}")
    print(f"  Samples:  {stats['sample_count']:,}")
    print(f"  Features: {stats['feature_count']:,}")
    print(f"  Labels:   {stats['label_count']:,}")
    print(f"  Split: {stats['split_stats']}")


def create_db_indexes(con):
    """データベースのインデックスを一括作成する（データインサート完了後に呼ぶことで高速化）"""
    print("[INFO] Creating database indexes...")
    con.execute("CREATE INDEX IF NOT EXISTS idx_samples_strain ON samples(strain_id)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_samples_split ON samples(split_type)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_samples_split_date ON samples(split_type_date)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_samples_split_wf ON samples(split_type_wf)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_samples_cooccur ON samples(max_cooccurrence)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_samples_length ON samples(path_length)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_features_sample ON features(sample_id)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_labels_sample ON labels(sample_id)")
    print("[INFO] Indexes created successfully")


def run_db_migrations(db_path=None):
    """データベースのスキーマ拡張マイグレーションを実行する。"""
    if db_path is None:
        db_path = get_db_path()

    con = connect_db(db_path)
    try:
        # カラム情報の取得
        cols = [r[1] for r in con.execute("PRAGMA table_info(samples)").fetchall()]
        if 'split_type_date' not in cols:
            print("[INFO] Migration: Adding 'split_type_date' column to samples table...")
            con.execute("ALTER TABLE samples ADD COLUMN split_type_date INTEGER DEFAULT 0")
            con.execute("CREATE INDEX IF NOT EXISTS idx_samples_split_date ON samples(split_type_date)")
            from .queries import assign_date_splits
            assign_date_splits(con)
            print("[INFO] Migration: 'split_type_date' column added and initialized successfully.")
        if 'split_type_wf' not in cols:
            print("[INFO] Migration: Adding 'split_type_wf' column to samples table...")
            con.execute("ALTER TABLE samples ADD COLUMN split_type_wf INTEGER DEFAULT 0")
            con.execute("CREATE INDEX IF NOT EXISTS idx_samples_split_wf ON samples(split_type_wf)")
            print("[INFO] Migration: 'split_type_wf' column added (will be populated at walk_forward runtime).")
    except Exception as e:
        print(f"[ERROR] Migration failed: {e}")
        raise e
    finally:
        con.close()
