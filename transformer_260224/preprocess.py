# preprocess.py
# 特徴量生成 → DuckDB保存
# Usage: python -m transformer_260224.preprocess
#
# 既存の dataset.py の特徴量生成ロジックを再利用し、
# メモリ効率的にDuckDBに保存する
#
# 機能:
# 1. バッチキャッシュ再利用（utils.pyの既存機能）
# 2. DB再開機能（処理済み株をスキップ）

import os
import pickle
import gc
import time
import pandas as pd
from tqdm import tqdm
from datetime import datetime

from . import config
from .db_utils import init_db, connect_db, get_db_path, print_db_stats, get_feature_config_hash
from .dataset import (
    import_mutation_paths,
    filter_co_occur,
    preprocess_static_data,
    Feature_path_fast,
    DNA2Protein,
)
from .utils import (
    get_config_hash,
    get_batch_cache_path,
    load_batch_cache,
    save_batch_cache,
)


def force_print(msg):
    """即時出力"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)


def load_static_data():
    """
    静的データ（コドン、頻度、物性値）を読み込み
    """
    force_print("[INFO] Loading static data...")
    
    df_codon = pd.read_csv(config.Codon_csv)
    df_freq = pd.read_csv(config.Freq_csv, index_col=0)
    df_dissimilarity = pd.read_csv(config.Disimilarity_csv)
    df_pam250 = pd.read_csv(config.PAM250_csv, index_col=0)
    
    # 高速化用にDict/Listに変換
    codon_data, freq_dict, dissim_dict, pam250_dict = preprocess_static_data(
        df_codon, df_freq, df_dissimilarity, df_pam250
    )
    
    return codon_data, freq_dict, dissim_dict, pam250_dict


def compute_strain_strength_from_csv():
    """
    CSVから株の流行度を計算
    """
    csv_path = os.path.join(config.DATA_BASE_DIR, '../sequences-241017.csv')
    
    if not os.path.exists(csv_path):
        force_print(f"[WARNING] Sequences CSV not found: {csv_path}")
        return {}
    
    import math
    df = pd.read_csv(csv_path)
    strain_counts = df.groupby('Pango lineage').size()
    
    strain_to_strength = {
        strain: float(f"{math.log(count + 1):.2f}")
        for strain, count in strain_counts.items()
    }
    
    force_print(f"[INFO] Computed strength for {len(strain_to_strength)} strains from CSV")
    return strain_to_strength


def get_processed_strains(con):
    """
    DBから処理済みの株一覧を取得（再開機能用）
    """
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
    """
    DBから次のID値を取得（再開機能用）
    """
    try:
        sample_id = con.execute("SELECT COALESCE(MAX(sample_id), -1) + 1 FROM samples").fetchone()[0]
        feature_id = con.execute("SELECT COALESCE(MAX(feature_id), -1) + 1 FROM features").fetchone()[0]
        label_id = con.execute("SELECT COALESCE(MAX(label_id), -1) + 1 FROM labels").fetchone()[0]
        return sample_id, feature_id, label_id
    except Exception:
        return 0, 0, 0


def import_strains_to_db(con, strain_to_strength):
    """
    株マスタをDBに登録
    """
    force_print("[INFO] Importing strains to DB...")
    
    usher_dir = config.DATA_BASE_DIR  # '../usher_output/' そのものがusher_output
    strains = sorted([f for f in os.listdir(usher_dir) if not f.startswith('.')])
    
    for strain_id, strain_name in enumerate(tqdm(strains, desc="Registering strains")):
        strength_score = strain_to_strength.get(strain_name, 0.0)
        con.execute("""
            INSERT OR IGNORE INTO strains (strain_id, strain_name, strength_score)
            VALUES (?, ?, ?)
        """, [strain_id, strain_name, strength_score])
    
    force_print(f"[INFO] Registered {len(strains)} strains")
    return strains


def get_strain_cache_path(strain_name, config_hash):
    """株ごとのキャッシュパスを取得"""
    cache_dir = os.path.join(config.INCREMENTAL_CACHE_DIR, 'strains')
    os.makedirs(cache_dir, exist_ok=True)
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


def save_strain_cache(data, cache_path):
    """株キャッシュを保存"""
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        pass  # キャッシュ保存失敗は無視


def process_strain_features(strain_name, codon_data, freq_dict, dissim_dict, pam250_dict, config_hash):
    """
    1つの株の特徴量を生成（キャッシュ対応）
    
    Returns:
        list of (path_str, path_length, max_cooc, feature_path, y_targets)
    """
    # キャッシュ確認
    cache_path = get_strain_cache_path(strain_name, config_hash)
    cached = load_strain_cache(cache_path)
    if cached is not None:
        return cached
    
    usher_dir = config.DATA_BASE_DIR
    file_paths = import_mutation_paths(usher_dir, strain_name)
    
    if not file_paths:
        return []
    
    max_cooccur = config.MAX_CO_OCCURRENCE
    results = []
    
    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding="utf-8_sig") as f:
                datalist = f.readlines()
        except Exception:
            continue
        
        for line in datalist[1:]:
            data = line.strip().split('\t')
            if len(data) < 3:
                continue
            
            path_str = data[2].split('>')
            path_length = len(path_str)
            
            # 共起数フィルタ
            if not filter_co_occur(path_str, max_cooccur):
                continue
            
            # 最大共起数を計算
            max_cooc = max(len(m.split(',')) for m in path_str)
            
            # 特徴量生成（既存ロジックを使用）
            try:
                feature_path = Feature_path_fast(
                    path_str, codon_data, freq_dict, dissim_dict, pam250_dict
                )
            except Exception:
                continue
            
            if len(feature_path) <= config.TARGET_LEN:
                continue
            
            # X (入力) と Y (ラベル) に分割
            x_features = feature_path[:-config.TARGET_LEN]
            y_features = feature_path[-config.TARGET_LEN:]
            
            # ラベルを抽出
            y_targets = []
            for ts_events in y_features:
                for event in ts_events:
                    cat_feat = event[0]
                    region_id = cat_feat[7]
                    position_id = cat_feat[1]
                    aa_pos_id = cat_feat[5]
                    codon_pos_id = cat_feat[3]
                    is_synonymous = 1 if cat_feat[4] == cat_feat[6] else 0
                    y_targets.append((region_id, position_id, aa_pos_id, codon_pos_id, is_synonymous))
            
            if not y_targets:
                continue
            
            raw_path_str = data[2][:500]  # 長すぎる場合は切り詰め
            results.append((raw_path_str, path_length, max_cooc, x_features, y_targets))
    
    # キャッシュに保存
    save_strain_cache(results, cache_path)
    
    return results


def write_strain_to_db(con, strain_id, strain_strength, samples_data,
                       sample_id_counter, feature_id_counter, label_id_counter):
    """
    株の特徴量データをDBに書き込み
    """
    batch_samples = []
    batch_features = []
    batch_labels = []
    batch_size = 500
    
    for raw_path_str, path_length, max_cooc, x_features, y_targets in samples_data:
        # サンプル登録
        batch_samples.append((
            sample_id_counter,
            strain_id,
            raw_path_str,
            path_length,
            max_cooc,
            0,  # split_type (後で設定)
            strain_strength
        ))
        
        # 特徴量登録
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
        
        # ラベル登録
        batch_labels.append((
            label_id_counter,
            sample_id_counter,
            pickle.dumps(y_targets)
        ))
        label_id_counter += 1
        sample_id_counter += 1
        
        # バッチがたまったら書き込み
        if len(batch_samples) >= batch_size:
            con.executemany("INSERT INTO samples VALUES (?, ?, ?, ?, ?, ?, ?)", batch_samples)
            con.executemany("INSERT INTO features VALUES (?, ?, ?, ?, ?, ?)", batch_features)
            con.executemany("INSERT INTO labels VALUES (?, ?, ?)", batch_labels)
            batch_samples = []
            batch_features = []
            batch_labels = []
    
    # 残りを書き込み
    if batch_samples:
        con.executemany("INSERT INTO samples VALUES (?, ?, ?, ?, ?, ?, ?)", batch_samples)
        con.executemany("INSERT INTO features VALUES (?, ?, ?, ?, ?, ?)", batch_features)
        con.executemany("INSERT INTO labels VALUES (?, ?, ?)", batch_labels)
    
    return sample_id_counter, feature_id_counter, label_id_counter


def assign_splits(con, train_ratio=0.8, valid_ratio=0.1):
    """
    パス長に基づいてtrain/valid/testに分割
    """
    import random
    
    force_print("[INFO] Assigning train/valid/test splits based on path length...")
    
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
        force_print(f"  {split_name}: {count:,} samples")
def process_strain_wrapper(args):
    """
    並列処理用ラッパー（picklableにするため外部関数）
    """
    strain_name, config_hash = args
    
    # 静的データは各プロセスで読み込む（共有メモリ使用も可能だが複雑になる）
    # キャッシュがあれば読み込みのみで高速
    cache_path = get_strain_cache_path(strain_name, config_hash)
    cached = load_strain_cache(cache_path)
    if cached is not None:
        return strain_name, cached, True  # キャッシュヒット
    
    # キャッシュがない場合、静的データを読み込んで処理
    try:
        codon_data, freq_dict, dissim_dict, pam250_dict = load_static_data_minimal()
        result = process_strain_features_core(
            strain_name, codon_data, freq_dict, dissim_dict, pam250_dict, config_hash
        )
        return strain_name, result, False
    except Exception as e:
        return strain_name, [], False


def load_static_data_minimal():
    """
    静的データを読み込み（並列処理用、ログなし）
    """
    import pandas as pd
    df_codon = pd.read_csv(config.Codon_csv)
    df_freq = pd.read_csv(config.Freq_csv, index_col=0)
    df_dissimilarity = pd.read_csv(config.Disimilarity_csv)
    df_pam250 = pd.read_csv(config.PAM250_csv, index_col=0)
    
    codon_data, freq_dict, dissim_dict, pam250_dict = preprocess_static_data(
        df_codon, df_freq, df_dissimilarity, df_pam250, silent=True
    )
    return codon_data, freq_dict, dissim_dict, pam250_dict


def process_strain_features_core(strain_name, codon_data, freq_dict, dissim_dict, pam250_dict, config_hash):
    """
    1つの株の特徴量を生成（キャッシュ保存込み）
    """
    usher_dir = config.DATA_BASE_DIR
    file_paths = import_mutation_paths(usher_dir, strain_name)
    
    if not file_paths:
        return []
    
    max_cooccur = config.MAX_CO_OCCURRENCE
    results = []
    
    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding="utf-8_sig") as f:
                datalist = f.readlines()
        except Exception:
            continue
        
        for line in datalist[1:]:
            data = line.strip().split('\t')
            if len(data) < 3:
                continue
            
            path_str = data[2].split('>')
            path_length = len(path_str)
            
            if not filter_co_occur(path_str, max_cooccur):
                continue
            
            max_cooc = max(len(m.split(',')) for m in path_str)
            
            try:
                feature_path = Feature_path_fast(
                    path_str, codon_data, freq_dict, dissim_dict, pam250_dict
                )
            except Exception:
                continue
            
            if len(feature_path) <= config.TARGET_LEN:
                continue
            
            x_features = feature_path[:-config.TARGET_LEN]
            y_features = feature_path[-config.TARGET_LEN:]
            
            y_targets = []
            for ts_events in y_features:
                for event in ts_events:
                    cat_feat = event[0]
                    region_id = cat_feat[7]
                    position_id = cat_feat[1]
                    aa_pos_id = cat_feat[5]
                    codon_pos_id = cat_feat[3]
                    is_synonymous = 1 if cat_feat[4] == cat_feat[6] else 0
                    y_targets.append((region_id, position_id, aa_pos_id, codon_pos_id, is_synonymous))
            
            if not y_targets:
                continue
            
            raw_path_str = data[2][:500]
            results.append((raw_path_str, path_length, max_cooc, x_features, y_targets))
    
    # キャッシュに保存
    cache_path = get_strain_cache_path(strain_name, config_hash)
    save_strain_cache(results, cache_path)
    
    return results


def main():
    """前処理メイン（並列処理版）"""
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor, as_completed
    
    start_time = time.time()
    
    force_print("=" * 60)
    force_print("Starting preprocessing (DuckDB mode, PARALLEL)...")
    force_print("=" * 60)
    
    # 設定ハッシュ（キャッシュ無効化用）
    config_hash = get_config_hash()[:8]
    force_print(f"[INFO] Config hash: {config_hash}")
    
    # 並列数
    num_workers = min(multiprocessing.cpu_count(), 4)
    force_print(f"[INFO] Using {num_workers} parallel workers")
    
    # 静的データ読み込み（メインプロセス用、表示のため）
    codon_data, freq_dict, dissim_dict, pam250_dict = load_static_data()
    
    # 流行度計算
    strain_to_strength = compute_strain_strength_from_csv()
    
    # DB初期化
    db_path = init_db()
    con = connect_db(db_path)
    
    try:
        # 株マスタをインポート
        strains = import_strains_to_db(con, strain_to_strength)
        
        # 処理済み株を取得（再開機能）
        processed_strains = get_processed_strains(con)
        if processed_strains:
            force_print(f"[INFO] Resuming: {len(processed_strains)} strains already processed in DB")
        
        # 未処理株をフィルタ
        strains_to_process = [(s, config_hash) for s in strains if s not in processed_strains]
        force_print(f"[INFO] Strains to process: {len(strains_to_process)}")
        
        if not strains_to_process:
            force_print("[INFO] All strains already processed!")
        else:
            # ストリーミング処理: 完了した株を即座にDBに書き込み
            force_print("[INFO] Processing strains (parallel generate + streaming write)...")
            
            strain_id_map = {s: i for i, s in enumerate(strains)}
            sample_id, feature_id, label_id = get_next_ids(con)
            
            cache_hits = 0
            processed = 0
            total_samples = 0
            
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = {executor.submit(process_strain_wrapper, args): args[0] 
                          for args in strains_to_process}
                
                pbar = tqdm(as_completed(futures), total=len(futures), desc="Processing")
                for future in pbar:
                    strain_name = futures[future]
                    try:
                        result_name, samples_data, from_cache = future.result()
                        
                        if from_cache:
                            cache_hits += 1
                        
                        # 即座にDBに書き込み（ストリーミング）
                        if samples_data:
                            strain_id = strain_id_map.get(result_name, 0)
                            strain_strength = strain_to_strength.get(result_name, 0.0)
                            
                            sample_id, feature_id, label_id = write_strain_to_db(
                                con, strain_id, strain_strength, samples_data,
                                sample_id, feature_id, label_id
                            )
                            
                            strain_samples = len(samples_data)
                            total_samples += strain_samples
                        else:
                            strain_samples = 0
                        
                        # 結果を明示的に解放
                        del samples_data
                        
                        processed += 1
                        
                        # tqdmに進捗表示
                        pbar.set_postfix({
                            'last': strain_samples,
                            'total': f'{total_samples:,}',
                            'cache': cache_hits
                        })
                        
                    except Exception as e:
                        force_print(f"[WARNING] Failed to process {strain_name}: {e}")
                    
                    # futureを辞書から削除してメモリ解放
                    del futures[future]
                    
                    # 定期的にメモリ解放
                    if processed % 50 == 0:
                        gc.collect()
            
            force_print(f"[INFO] Complete: {processed} strains, {total_samples:,} samples, {cache_hits} cache hits")
        
        # train/valid/test 分割
        assign_splits(con)
        
        con.close()
        
        # 統計表示
        print_db_stats(db_path)
        
        elapsed = time.time() - start_time
        force_print("=" * 60)
        force_print(f"Preprocessing completed in {elapsed:.1f} seconds!")
        force_print(f"Database: {db_path}")
        force_print("=" * 60)
        
    except Exception as e:
        con.close()
        force_print(f"[ERROR] Preprocessing failed: {e}")
        raise e


if __name__ == "__main__":
    main()
