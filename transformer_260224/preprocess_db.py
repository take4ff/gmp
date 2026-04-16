# preprocess_db.py
# Step 2: キャッシュ → DB変換
# Usage: python -m transformer_260224.preprocess_db
#
# Step 1で生成した株ごとのキャッシュを読み込み、DuckDBに書き込む。
# ストリーミング処理でメモリ効率を保つ。

import os
import pickle
import gc
import time
import pandas as pd
from tqdm import tqdm
from datetime import datetime

from . import config
from .db_utils import init_db, connect_db, get_db_path, print_db_stats
from .utils import get_config_hash


def force_print(msg):
    """即時出力"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)


def compute_strain_strength_from_csv():
    """CSVから株の流行度を計算"""
    csv_path = 'meta_data/sequences-241017.csv'
    
    if not os.path.exists(csv_path):
        force_print(f"[WARNING] Sequences CSV not found: {csv_path}")
        return {}
    
    import math
    df = pd.read_csv(csv_path)
    strain_counts = df.groupby('Pangolin').size()
    
    strain_to_strength = {
        strain: float(f"{math.log(count + 1):.2f}")
        for strain, count in strain_counts.items()
    }
    
    return strain_to_strength


def get_strain_cache_path(strain_name, config_hash):
    """株ごとのキャッシュパスを取得"""
    cache_dir = os.path.join(config.INCREMENTAL_CACHE_DIR, 'strains')
    return os.path.join(cache_dir, f'strain_{config_hash}_{strain_name}.pkl')


def load_strain_cache(cache_path):
    """株キャッシュをロード"""
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception:
            return None
    return None


def get_processed_strains(con):
    """DBから処理済みの株一覧を取得"""
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
    """DBから次のID値を取得"""
    try:
        sample_id = con.execute("SELECT COALESCE(MAX(sample_id), -1) + 1 FROM samples").fetchone()[0]
        feature_id = con.execute("SELECT COALESCE(MAX(feature_id), -1) + 1 FROM features").fetchone()[0]
        label_id = con.execute("SELECT COALESCE(MAX(label_id), -1) + 1 FROM labels").fetchone()[0]
        return sample_id, feature_id, label_id
    except Exception:
        return 0, 0, 0


def import_strains_to_db(con, strains, strain_to_strength):
    """株マスタをDBに登録"""
    force_print("[INFO] Importing strains to DB...")
    
    for strain_id, strain_name in enumerate(tqdm(strains, desc="Registering strains")):
        strength_score = strain_to_strength.get(strain_name, 0.0)
        con.execute("""
            INSERT OR IGNORE INTO strains (strain_id, strain_name, strength_score)
            VALUES (?, ?, ?)
        """, [strain_id, strain_name, strength_score])
    
    force_print(f"[INFO] Registered {len(strains)} strains")


def write_strain_to_db(con, strain_id, strain_strength, samples_data,
                       sample_id_counter, feature_id_counter, label_id_counter):
    """株の特徴量データをDBに書き込み"""
    batch_samples = []
    batch_features = []
    batch_labels = []
    batch_size = 500
    
    for raw_path_str, path_length, max_cooc, x_features, y_targets in samples_data:
        batch_samples.append((
            sample_id_counter,
            strain_id,
            raw_path_str,
            path_length,
            max_cooc,
            0,  # split_type
            strain_strength
        ))
        
        for ts_idx, ts_events in enumerate(x_features):
            cooccurrence = len(ts_events)
            for event in ts_events:
                cat_feat, num_feat = event
                batch_features.append((
                    feature_id_counter,
                    sample_id_counter,
                    ts_idx,
                    cooccurrence,
                    pickle.dumps(cat_feat),
                    pickle.dumps(num_feat)
                ))
                feature_id_counter += 1
        
        batch_labels.append((
            label_id_counter,
            sample_id_counter,
            pickle.dumps(y_targets)
        ))
        label_id_counter += 1
        sample_id_counter += 1
        
        if len(batch_samples) >= batch_size:
            con.executemany("INSERT INTO samples VALUES (?, ?, ?, ?, ?, ?, ?)", batch_samples)
            con.executemany("INSERT INTO features VALUES (?, ?, ?, ?, ?, ?)", batch_features)
            con.executemany("INSERT INTO labels VALUES (?, ?, ?)", batch_labels)
            batch_samples = []
            batch_features = []
            batch_labels = []
    
    if batch_samples:
        con.executemany("INSERT INTO samples VALUES (?, ?, ?, ?, ?, ?, ?)", batch_samples)
        con.executemany("INSERT INTO features VALUES (?, ?, ?, ?, ?, ?)", batch_features)
        con.executemany("INSERT INTO labels VALUES (?, ?, ?)", batch_labels)
    
    return sample_id_counter, feature_id_counter, label_id_counter


def assign_splits(con, valid_ratio=0.1):
    """パス長に基づいてtrain/valid/testに分割"""
    import random
    
    force_print("[INFO] Assigning train/valid/test splits...")
    
    train_max = config.TRAIN_MAX
    valid_num = config.VALID_NUM
    overlap_start = train_max - valid_num + 1
    
    con.execute(f"UPDATE samples SET split_type = 0 WHERE path_length < {overlap_start}")
    con.execute(f"UPDATE samples SET split_type = 2 WHERE path_length > {train_max}")
    
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
    
    stats = con.execute("SELECT * FROM split_stats").fetchall()
    for split_type, split_name, count in stats:
        force_print(f"  {split_name}: {count:,} samples")


def main():
    """Step 2: キャッシュ → DB変換"""
    start_time = time.time()
    
    force_print("=" * 60)
    force_print("Step 2: Cache → DB Conversion")
    force_print("=" * 60)
    
    # 設定ハッシュ
    config_hash = get_config_hash()[:8]
    force_print(f"[INFO] Config hash: {config_hash}")
    
    # キャッシュディレクトリ確認
    cache_dir = os.path.join(config.INCREMENTAL_CACHE_DIR, 'strains')
    if not os.path.exists(cache_dir):
        force_print(f"[ERROR] Cache directory not found: {cache_dir}")
        force_print("Please run Step 1 first:")
        force_print("  python -m transformer_260224.preprocess_cache")
        return
    
    # キャッシュファイル一覧
    cache_files = [f for f in os.listdir(cache_dir) if f.startswith(f'strain_{config_hash}_') and f.endswith('.pkl')]
    if not cache_files:
        force_print(f"[ERROR] No cache files found for config hash: {config_hash}")
        return
    
    force_print(f"[INFO] Found {len(cache_files)} cache files")
    
    # 株名抽出
    strains = []
    for f in cache_files:
        # strain_{hash}_{name}.pkl → name
        name = f[len(f'strain_{config_hash}_'):-4]
        strains.append(name)
    strains = sorted(strains)
    
    # 流行度
    strain_to_strength = compute_strain_strength_from_csv()
    
    # DB初期化
    db_path = init_db()
    con = connect_db(db_path)
    
    try:
        # 株マスタ登録
        import_strains_to_db(con, strains, strain_to_strength)
        strain_id_map = {s: i for i, s in enumerate(strains)}
        
        # 処理済み株を取得
        processed_strains = get_processed_strains(con)
        if processed_strains:
            force_print(f"[INFO] Resuming: {len(processed_strains)} strains already in DB")
        
        strains_to_process = [s for s in strains if s not in processed_strains]
        force_print(f"[INFO] Strains to process: {len(strains_to_process)}")
        
        if not strains_to_process:
            force_print("[INFO] All strains already in DB!")
        else:
            sample_id, feature_id, label_id = get_next_ids(con)
            total_samples = 0
            
            pbar = tqdm(strains_to_process, desc="Writing to DB")
            for strain_name in pbar:
                # キャッシュ読み込み
                cache_path = get_strain_cache_path(strain_name, config_hash)
                samples_data = load_strain_cache(cache_path)
                
                if samples_data:
                    strain_id = strain_id_map.get(strain_name, 0)
                    strain_strength = strain_to_strength.get(strain_name, 0.0)
                    
                    sample_id, feature_id, label_id = write_strain_to_db(
                        con, strain_id, strain_strength, samples_data,
                        sample_id, feature_id, label_id
                    )
                    
                    strain_samples = len(samples_data)
                    total_samples += strain_samples
                else:
                    strain_samples = 0
                
                pbar.set_postfix({
                    'last': strain_samples,
                    'total': f'{total_samples:,}'
                })
                
                # メモリ解放
                del samples_data
                gc.collect()
            
            force_print(f"[INFO] Written: {sample_id:,} samples, {feature_id:,} features")
        
        # train/valid/test 分割
        assign_splits(con)
        
        con.close()
        
        # 統計表示
        print_db_stats(db_path)
        
        elapsed = time.time() - start_time
        force_print("=" * 60)
        force_print(f"Step 2 completed in {elapsed:.1f} seconds!")
        force_print(f"Database: {db_path}")
        force_print("=" * 60)
        
    except Exception as e:
        con.close()
        force_print(f"[ERROR] Step 2 failed: {e}")
        raise e


if __name__ == "__main__":
    main()
