# db_utils.py
# DuckDB データベース操作ユーティリティ

import os
import pickle
import duckdb
import hashlib
from datetime import datetime
from . import config


def get_db_path():
    """設定からDBパスを取得"""
    db_dir = getattr(config, 'DB_DIR', 'cache/db')
    os.makedirs(db_dir, exist_ok=True)
    
    # 設定ハッシュをファイル名に含める
    config_hash = get_feature_config_hash()[:8]
    return os.path.join(db_dir, f'features_{config_hash}.duckdb')


def get_feature_config_hash():
    """特徴量設定に基づくハッシュを生成"""
    relevant_configs = [
        str(config.NUM_FEATURE_STRING),
        str(config.NUM_CHEM_FEATURES),
        str(config.MAX_SEQ_LEN),
        str(config.TARGET_LEN),
        str(config.DATA_BASE_DIR),
    ]
    config_string = "_".join(relevant_configs)
    return hashlib.md5(config_string.encode('utf-8')).hexdigest()


def connect_db(db_path=None, read_only=False):
    """DuckDB接続を取得"""
    if db_path is None:
        db_path = get_db_path()
    return duckdb.connect(db_path, read_only=read_only)


def init_db(db_path=None):
    """データベースを初期化（テーブル作成）"""
    if db_path is None:
        db_path = get_db_path()
    
    con = connect_db(db_path)
    
    # 株マスタ
    con.execute("""
        CREATE TABLE IF NOT EXISTS strains (
            strain_id INTEGER PRIMARY KEY,
            strain_name VARCHAR UNIQUE NOT NULL,
            sample_count INTEGER DEFAULT 0,
            strength_score REAL DEFAULT 0.0,
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
            strength_score REAL DEFAULT 0.0
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
    
    # インデックス作成
    con.execute("CREATE INDEX IF NOT EXISTS idx_samples_strain ON samples(strain_id)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_samples_split ON samples(split_type)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_samples_cooccur ON samples(max_cooccurrence)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_samples_length ON samples(path_length)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_features_sample ON features(sample_id)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_labels_sample ON labels(sample_id)")
    
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
    """DBが存在し、スキーマが一致するか確認"""
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
            return False, f"Config mismatch: DB={db_hash[:8]}, Current={current_hash[:8]}"
        
        return True, "OK"
    except Exception as e:
        return False, str(e)


def get_db_stats(db_path=None):
    """DBの統計情報を取得"""
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
    """DBの統計情報を表示"""
    stats = get_db_stats(db_path)
    print("\n[INFO] === Database Statistics ===")
    print(f"  Strains:  {stats['strain_count']:,}")
    print(f"  Samples:  {stats['sample_count']:,}")
    print(f"  Features: {stats['feature_count']:,}")
    print(f"  Labels:   {stats['label_count']:,}")
    print(f"  Split: {stats['split_stats']}")


def load_samples_from_db(db_path=None, split_type=None, max_cooccurrence=None, 
                         min_length=None, max_length=None, limit=None):
    """
    DBからサンプルを読み込み、学習用データ形式に変換
    
    Args:
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
    
    # サンプル取得クエリを構築
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
        # 特徴量を取得
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
        
        # ラベルを取得
        label_row = con.execute("""
            SELECT targets FROM labels WHERE sample_id = ?
        """, [sample_id]).fetchone()
        
        if label_row:
            y_targets = pickle.loads(label_row[0])
        else:
            y_targets = []
        
        result.append((x, y_targets, path_length, raw_path, strain_name, strength_score))
    
    con.close()
    return result


# ==========================================
# メモリ効率的なDB DataLoader
# ==========================================

import torch
from torch.utils.data import IterableDataset, DataLoader
import numpy as np


class DBIterableDataset(IterableDataset):
    """
    DuckDBからバッチ単位でデータを読み込むIterableDataset
    全データをメモリに保持せず、必要な分だけ読み込む
    """
    
    def __init__(self, db_path, split_type=None, max_cooccurrence=None,
                 min_length=None, max_length=None, shuffle=False, chunk_size=1000):
        """
        Args:
            db_path: DuckDBファイルパス
            split_type: 0=train, 1=valid, 2=test
            max_cooccurrence: 最大共起数フィルタ
            min_length, max_length: パス長フィルタ
            shuffle: シャッフルするか
            chunk_size: 一度に読み込むサンプル数
        """
        self.db_path = db_path
        self.split_type = split_type
        self.max_cooccurrence = max_cooccurrence
        self.min_length = min_length
        self.max_length = max_length
        self.shuffle = shuffle
        self.chunk_size = chunk_size
        
        # サンプルIDリストを取得
        self.sample_ids = self._get_sample_ids()
        self._length = len(self.sample_ids)
    
    def _get_sample_ids(self):
        """条件に合うサンプルIDのリストを取得"""
        con = connect_db(self.db_path, read_only=True)
        
        query = "SELECT sample_id FROM samples WHERE 1=1"
        params = []
        
        if self.split_type is not None:
            query += " AND split_type = ?"
            params.append(self.split_type)
        
        if self.max_cooccurrence is not None:
            query += " AND max_cooccurrence <= ?"
            params.append(self.max_cooccurrence)
        
        if self.min_length is not None:
            query += " AND path_length >= ?"
            params.append(self.min_length)
        
        if self.max_length is not None:
            query += " AND path_length <= ?"
            params.append(self.max_length)
        
        # raw_pathも取得してPython側でユニーク化判定を行う
        query = query.replace("SELECT sample_id", "SELECT sample_id, raw_path")
        query += " ORDER BY sample_id"
        
        result = con.execute(query, params).fetchall()
        con.close()
        
        total_count = len(result)
        split_name = {0: 'Train', 1: 'Valid', 2: 'Test'}.get(self.split_type, 'Unknown')
        print(f"[INFO] {split_name} data - Initial sample count: {total_count:,}")
        
        if getattr(config, 'USE_UNIQUE_FILTER', False):
            seen_paths = set()
            unique_ids = []
            for sample_id, raw_path in result:
                if raw_path not in seen_paths:
                    seen_paths.add(raw_path)
                    unique_ids.append(sample_id)
            
            unique_count = len(unique_ids)
            diff = total_count - unique_count
            print(f"[INFO] {split_name} data - After unique filter: {unique_count:,} (Removed {diff:,} duplicates)")
            return unique_ids
        else:
            return [r[0] for r in result]
    
    def __len__(self):
        return self._length
    
    def __iter__(self):
        import random
        
        sample_ids = self.sample_ids.copy()
        if self.shuffle:
            random.shuffle(sample_ids)
        
        # チャンク単位で処理
        for i in range(0, len(sample_ids), self.chunk_size):
            chunk_ids = sample_ids[i:i + self.chunk_size]
            samples = self._load_chunk(chunk_ids)
            
            for sample in samples:
                yield self._process_sample(sample)
    
    def _load_chunk(self, sample_ids):
        """指定されたサンプルIDのデータをDBからバッチ読み込み（IN句使用で高速化）"""
        if not sample_ids:
            return []
        
        con = connect_db(self.db_path, read_only=True)
        
        # IN句用のプレースホルダ生成
        placeholders = ','.join(['?' for _ in sample_ids])
        
        # サンプル情報を一括取得
        sample_rows = con.execute(f"""
            SELECT s.sample_id, s.raw_path, s.path_length, s.strength_score, st.strain_name
            FROM samples s
            JOIN strains st ON s.strain_id = st.strain_id
            WHERE s.sample_id IN ({placeholders})
            ORDER BY s.sample_id
        """, sample_ids).fetchall()
        
        # 特徴量を一括取得
        features_rows = con.execute(f"""
            SELECT sample_id, timestep, cat_features, num_features
            FROM features
            WHERE sample_id IN ({placeholders})
            ORDER BY sample_id, timestep
        """, sample_ids).fetchall()
        
        # ラベルを一括取得
        label_rows = con.execute(f"""
            SELECT sample_id, targets
            FROM labels
            WHERE sample_id IN ({placeholders})
        """, sample_ids).fetchall()
        
        con.close()
        
        # データを辞書に整理
        sample_dict = {row[0]: row for row in sample_rows}
        
        # 特徴量をsample_id別にグループ化
        features_dict = {}
        for sample_id, timestep, cat_blob, num_blob in features_rows:
            if sample_id not in features_dict:
                features_dict[sample_id] = []
            cat_feat = pickle.loads(cat_blob)
            num_feat = pickle.loads(num_blob)
            features_dict[sample_id].append((timestep, cat_feat, num_feat))
        
        # ラベルを辞書化
        label_dict = {row[0]: pickle.loads(row[1]) for row in label_rows}
        
        # 結果を構築
        results = []
        for sample_id in sample_ids:
            if sample_id not in sample_dict:
                continue
            
            _, raw_path, path_length, strength_score, strain_name = sample_dict[sample_id]
            
            # 特徴量を時系列順に整理
            x = []
            if sample_id in features_dict:
                for ts, cat_feat, num_feat in sorted(features_dict[sample_id], key=lambda x: x[0]):
                    x.append([(cat_feat, num_feat)])
            
            y_targets = label_dict.get(sample_id, [])
            
            results.append((x, y_targets, path_length, raw_path, strain_name, strength_score))
        
        return results
    
    def _process_sample(self, sample):
        """サンプルをモデル入力形式に変換"""
        sequence, targets, original_len, raw_path, strain, strength_score = sample
        
        target_len = config.TARGET_LEN
        
        if len(raw_path) > target_len:
            input_path_str = raw_path[:-target_len]
            target_path_str = raw_path[-target_len:]
        else:
            input_path_str = []
            target_path_str = raw_path
        
        if len(input_path_str) > config.MAX_SEQ_LEN:
            start_index_str = len(input_path_str) - config.MAX_SEQ_LEN
            input_path_str = input_path_str[start_index_str:]
        
        if len(sequence) > config.TRAIN_MAX:
            start_index = len(sequence) - config.TRAIN_MAX
            sequence_to_pad = sequence[start_index:]
            active_seq_len = config.TRAIN_MAX
        else:
            sequence_to_pad = sequence
            active_seq_len = len(sequence)
        
        padded_cat = np.zeros(
            (config.TRAIN_MAX, config.MAX_CO_OCCURRENCE, config.NUM_FEATURE_STRING), dtype=np.int64
        )
        padded_num = np.zeros(
            (config.TRAIN_MAX, config.MAX_CO_OCCURRENCE, config.NUM_CHEM_FEATURES), dtype=np.float32
        )
        
        padding_mask = np.ones(config.TRAIN_MAX, dtype=bool)
        pad_len = config.TRAIN_MAX - active_seq_len
        padding_mask[pad_len:] = False
        
        for t, timestep_events in enumerate(sequence_to_pad):
            target_idx = pad_len + t
            for c, (cat_origin, num_origin) in enumerate(timestep_events):
                if c >= config.MAX_CO_OCCURRENCE:
                    break
                
                cat = list(cat_origin)
                num = list(num_origin)
                
                # 数値特徴量の個別マスク
                # num構造: [freq(0), hydro(1), charge(2), size(3), blsm(4), pam250(5)]
                mask_keys = ['FREQ', 'HYDRO', 'CHARGE', 'SIZE', 'BLSM', 'PAM250']
                for i, key in enumerate(mask_keys):
                    if config.ABLATION_MASKS[key]:
                        num[i] = 0.0
                
                padded_cat[target_idx, c] = cat
                padded_num[target_idx, c] = num
        
        return {
            "x_cat": torch.tensor(padded_cat, dtype=torch.long),
            "x_num": torch.tensor(padded_num, dtype=torch.float),
            "mask": torch.tensor(padding_mask, dtype=torch.bool),
            "y": targets,
            "original_len": original_len,
            "raw_path": raw_path,
            "strain": strain,
            "strength_score": strength_score,
            "input_path_str": input_path_str,
            "target_path_str": target_path_str,
            "full_raw_path": raw_path
        }


def create_db_dataloader(db_path, split_type, batch_size, shuffle=False,
                         max_cooccurrence=None, min_length=None, max_length=None,
                         chunk_size=1000):
    """
    DBからデータを読み込むDataLoaderを作成
    
    Args:
        db_path: DuckDBファイルパス
        split_type: 0=train, 1=valid, 2=test
        batch_size: バッチサイズ
        shuffle: シャッフルするか
        max_cooccurrence: 最大共起数フィルタ
        min_length, max_length: パス長フィルタ
        chunk_size: DBから一度に読み込むサンプル数
    
    Returns:
        DataLoader
    """
    dataset = DBIterableDataset(
        db_path=db_path,
        split_type=split_type,
        max_cooccurrence=max_cooccurrence,
        min_length=min_length,
        max_length=max_length,
        shuffle=shuffle,
        chunk_size=chunk_size
    )
    
    def collate_fn(batch):
        batch_x_cat = torch.stack([item['x_cat'] for item in batch])
        batch_x_num = torch.stack([item['x_num'] for item in batch])
        batch_mask = torch.stack([item['mask'] for item in batch])
        batch_y = [item['y'] for item in batch]
        batch_lens = [item['original_len'] for item in batch]
        batch_strains = [item['strain'] for item in batch]
        batch_strength_scores = [item['strength_score'] for item in batch]
        batch_input_strs = [item['input_path_str'] for item in batch]
        batch_target_strs = [item['target_path_str'] for item in batch]
        batch_full_paths = [item['full_raw_path'] for item in batch]
        
        return (batch_x_cat, batch_x_num, batch_mask), batch_y, batch_lens, batch_strains, batch_strength_scores, batch_input_strs, batch_target_strs, batch_full_paths
    
    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)


def get_db_data_info(db_path=None, split_type=None, max_cooccurrence=None):
    """
    DBからデータ情報を取得（data_info相当）
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
