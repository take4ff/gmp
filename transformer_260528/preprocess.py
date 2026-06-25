# --- preprocess.py ---
# 特徴量生成 → DuckDB保存
# Usage: python -m transformer_260528.preprocess
#
# 機能:
#   1. バッチキャッシュ再利用（utils/io.pyの既存機能）
#   2. DB再開機能（処理済み株をスキップ）

import os
import pickle
import gc
import time
import pandas as pd
from tqdm import tqdm
from datetime import datetime

from . import config
from .utils.logging import force_print
from .db.connection import init_db, connect_db, get_db_path, print_db_stats, get_feature_config_hash, create_db_indexes
from .db.queries import assign_splits, assign_splits_auto, get_processed_strains, get_next_ids
from .utils.io import get_config_hash
from .legacy.dataset import (
    import_mutation_paths,
    filter_co_occur,
    preprocess_static_data,
    Feature_path_fast,
    DNA2Protein,
)


def load_static_data(silent=False):
    """静的データ（コドン、頻度、物性値、コドン特徴量）を読み込む。"""
    if not silent:
        force_print("[INFO] Loading static data...")

    df_codon = pd.read_csv(config.CODON_CSV)
    df_freq = pd.read_csv(config.FREQ_CSV, index_col=0)
    df_dissimilarity = pd.read_csv(config.DISSIMILARITY_CSV)
    df_pam250 = pd.read_csv(config.PAM250_CSV, index_col=0)
    df_log_ratio = pd.read_csv(config.CODON_LOG_RATIO_CSV)
    df_human_rscu = pd.read_csv(config.HUMAN_CODON_RSCU_CSV)
    df_scv2_rscu = pd.read_csv(config.SCV2_CODON_RSCU_CSV)

    return preprocess_static_data(
        df_codon, df_freq, df_dissimilarity, df_pam250, df_log_ratio, df_human_rscu, df_scv2_rscu,
        silent=silent
    )


def load_collection_dates():
    """SEQUENCES_CSV から Accession と Collection_Date のマッピングを読み込む。"""
    csv_path = config.SEQUENCES_CSV
    if not os.path.exists(csv_path):
        force_print(f"[WARNING] Sequences CSV not found: {csv_path}")
        return {}
    
    force_print("[INFO] Loading collection dates from CSV...")
    start = time.time()
    df = pd.read_csv(csv_path, usecols=['Accession', 'Collection_Date'])
    df['Collection_Date'] = df['Collection_Date'].fillna('')
    mapping = dict(zip(df['Accession'], df['Collection_Date']))
    elapsed = time.time() - start
    force_print(f"[INFO] Loaded {len(mapping):,} collection dates in {elapsed:.1f} seconds")
    return mapping


def compute_strain_strength_from_csv():
    """CSVから株の流行度を計算する。"""
    csv_path = config.SEQUENCES_CSV

    if not os.path.exists(csv_path):
        force_print(f"[WARNING] Sequences CSV not found: {csv_path}")
        return {}

    import math
    df = pd.read_csv(csv_path)
    
    column = getattr(config, 'STRENGTH_CSV_COLUMN', 'Pango lineage')
    if column not in df.columns:
        force_print(f"[WARNING] Column '{column}' not found in CSV. Falling back to first column.")
        column = df.columns[0]
        
    strain_counts = df.groupby(column).size()

    calc_method = getattr(config, 'STRENGTH_CALC_METHOD', 'log1p').lower()
    
    strain_to_strength = {}
    for strain, count in strain_counts.items():
        if calc_method == 'log1p':
            val = math.log(count + 1)
        elif calc_method == 'sqrt':
            val = math.sqrt(count)
        elif calc_method == 'raw':
            val = float(count)
        elif calc_method == 'log10':
            val = math.log10(count + 1)
        else:
            raise ValueError(f"Unknown STRENGTH_CALC_METHOD: {calc_method}")
            
        strain_to_strength[strain] = float(f"{val:.2f}")

    force_print(f"[INFO] Computed strength for {len(strain_to_strength)} strains from CSV using method '{calc_method}'")
    return strain_to_strength


def import_strains_to_db(con, strain_to_strength):
    """株マスタをDBに登録する。"""
    force_print("[INFO] Importing strains to DB...")

    # Accession -> Pangolin(NCBI) のマッピングをロード
    csv_path = config.SEQUENCES_CSV
    acc_to_pango = {}
    if os.path.exists(csv_path):
        force_print("[INFO] Loading Accession to Pangolin mapping from CSV...")
        start_t = time.time()
        # メモリ節約のため必要なカラムのみロード
        df_seq = pd.read_csv(csv_path, usecols=['Accession', 'Pangolin'])
        df_seq['Pangolin'] = df_seq['Pangolin'].fillna('')
        acc_to_pango = dict(zip(df_seq['Accession'], df_seq['Pangolin']))
        elapsed = time.time() - start_t
        force_print(f"[INFO] Loaded {len(acc_to_pango):,} mappings in {elapsed:.1f} seconds")
    else:
        force_print(f"[WARNING] Sequences CSV not found for mapping: {csv_path}")

    usher_dir = config.DATA_BASE_DIR
    strains = sorted([f for f in os.listdir(usher_dir) if not f.startswith('.')])

    from collections import Counter
    import math

    for strain_id, strain_name in enumerate(tqdm(strains, desc="Registering strains")):
        clade_counts = Counter()
        ncbi_name_counts = Counter()
        usher_sample_count = 0
        
        # すべての timestep フォルダを探索
        strain_path = os.path.join(usher_dir, strain_name)
        if os.path.exists(strain_path):
            timesteps = [d for d in os.listdir(strain_path) if d.isdigit()]
            for ts in timesteps:
                clades_file = os.path.join(strain_path, ts, 'clades.txt')
                if os.path.exists(clades_file):
                    try:
                        with open(clades_file, 'r', encoding='utf-8') as f:
                            for line in f:
                                parts = line.strip().split('\t')
                                if len(parts) >= 2:
                                    # clade情報 (2カラム目)
                                    clade_counts[parts[1]] += 1
                                    # NCBIでの系統名マッピング (1カラム目の Accession を引く)
                                    acc_id = parts[0]
                                    ncbi_name = acc_to_pango.get(acc_id, '')
                                    if ncbi_name:
                                        ncbi_name_counts[ncbi_name] += 1
                                    usher_sample_count += 1
                    except Exception:
                        pass
        
        # 最頻値の取得
        most_common_clade = clade_counts.most_common(1)[0][0] if clade_counts else None
        most_common_ncbi_name = ncbi_name_counts.most_common(1)[0][0] if ncbi_name_counts else strain_name
        
        # 流行度スコアの計算
        strength_score_usher = float(f"{math.log(usher_sample_count + 1):.2f}") if usher_sample_count > 0 else 0.0
        # NCBI 流行度は、決定した NCBI 系統名をもとに CSV から集計された流行度を引く
        strength_score_ncbi = strain_to_strength.get(most_common_ncbi_name, 0.0)
        
        # config の STRENGTH_SOURCE に応じたデフォルト流行度の設定
        strength_source = getattr(config, 'STRENGTH_SOURCE', 'ncbi').lower()
        if strength_source == 'usher':
            strength_score = strength_score_usher
        else:
            strength_score = strength_score_ncbi

        con.execute("""
            INSERT OR IGNORE INTO strains (
                strain_id, strain_name, strain_name_ncbi, strain_name_usher,
                sample_count, clade, strength_score_ncbi, strength_score_usher, strength_score
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            strain_id, strain_name, most_common_ncbi_name, strain_name,
            usher_sample_count, most_common_clade, strength_score_ncbi, strength_score_usher, strength_score
        ])

    force_print(f"[INFO] Registered {len(strains)} strains")
    return strains


def get_strain_cache_path(strain_name, config_hash):
    """株ごとのキャッシュパスを取得する。"""
    cache_dir = os.path.join(config.INCREMENTAL_CACHE_DIR, 'strains')
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f'strain_{config_hash}_{strain_name}.pkl')


def get_strain_chunk_cache_path(strain_name, config_hash, chunk_idx):
    """チャンクごとのキャッシュパスを取得する。"""
    cache_dir = os.path.join(config.INCREMENTAL_CACHE_DIR, 'strains_chunks')
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f'strain_{config_hash}_{strain_name}_chunk_{chunk_idx}.pkl')


def get_strain_meta_cache_path(strain_name, config_hash):
    """株のチャンク情報を保持するメタキャッシュのパス。"""
    cache_dir = os.path.join(config.INCREMENTAL_CACHE_DIR, 'strains_meta')
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f'strain_{config_hash}_{strain_name}_meta.pkl')


def load_strain_cache(cache_path):
    """株キャッシュをロードする。"""
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception:
            return None
    return None


def save_strain_cache(data, cache_path):
    """株キャッシュを保存する。"""
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        pass  # キャッシュ保存失敗は無視


def process_strain_features_core_chunked(strain_name, codon_data, freq_dict, dissim_dict, pam250_dict,
                                         log_ratio_dict, human_rscu_dict, scv2_rscu_dict, config_hash):
    """1つの株の特徴量をチャンクに分割しながら生成し、キャッシュに保存する。

    Returns:
        tuple: (list of chunk_cache_paths, dict of stats)
    """
    usher_dir = config.DATA_BASE_DIR
    file_paths = import_mutation_paths(usher_dir, strain_name)

    stats = {
        'total_raw_lines': 0,
        'excluded_format': 0,
        'excluded_cooccur': 0,
        'excluded_feature_gen_error': 0,
        'excluded_short_path': 0,
        'valid_samples': 0
    }

    if not file_paths:
        return [], stats

    max_cooccur = config.MAX_CO_OCCURRENCE
    chunk_size = getattr(config, 'PREPROCESS_CHUNK_SIZE', 1000)  # 1チャンクあたりのサンプル数上限
    
    chunk_paths = []
    current_chunk_data = []
    chunk_idx = 0

    def save_current_chunk(data_list, idx):
        path = get_strain_chunk_cache_path(strain_name, config_hash, idx)
        save_strain_cache(data_list, path)
        return path

    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding="utf-8_sig") as f:
                # 行ごとのストリーミング読み込みでメモリを節約
                f.readline()  # ヘッダーを読み飛ばす
                for line in f:
                    stats['total_raw_lines'] += 1
                    data = line.strip().split('\t')
                    if len(data) < 3:
                        stats['excluded_format'] += 1
                        continue

                    path_str = data[2].split('>')
                    path_length = len(path_str)

                    if not filter_co_occur(path_str, max_cooccur):
                        stats['excluded_cooccur'] += 1
                        continue

                    max_cooc = max(len(m.split(',')) for m in path_str)

                    try:
                        feature_path = Feature_path_fast(
                            path_str, codon_data, freq_dict, dissim_dict, pam250_dict,
                            log_ratio_dict, human_rscu_dict, scv2_rscu_dict
                        )
                    except Exception:
                        stats['excluded_feature_gen_error'] += 1
                        continue

                    if len(feature_path) <= config.TARGET_LEN:
                        stats['excluded_short_path'] += 1
                        continue

                    y_targets = []
                    for ts_events in feature_path[-config.TARGET_LEN:]:
                        for event in ts_events:
                            cat_feat = event[0]
                            region_id = cat_feat[7]
                            position_id = cat_feat[1]
                            aa_pos_id = cat_feat[5]
                            codon_pos_id = cat_feat[3]
                            is_synonymous = 1 if cat_feat[4] == cat_feat[6] else 0
                            y_targets.append((region_id, position_id, aa_pos_id, codon_pos_id, is_synonymous))

                    if not y_targets:
                        stats['excluded_short_path'] += 1
                        continue

                    raw_path_str = data[2][:500]
                    current_chunk_data.append((data[0], raw_path_str, path_length, max_cooc, feature_path[:-config.TARGET_LEN], y_targets))
                    stats['valid_samples'] += 1

                    # チャンクサイズに達したらキャッシュ保存してメモリ解放
                    if len(current_chunk_data) >= chunk_size:
                        p = save_current_chunk(current_chunk_data, chunk_idx)
                        chunk_paths.append(p)
                        current_chunk_data = []
                        chunk_idx += 1
                        gc.collect()
        except Exception:
            continue

    # 残ったデータの保存
    if current_chunk_data:
        p = save_current_chunk(current_chunk_data, chunk_idx)
        chunk_paths.append(p)
        chunk_idx += 1

    # メタデータを保存（統計データもキャッシュ内に保存）
    meta_path = get_strain_meta_cache_path(strain_name, config_hash)
    save_strain_cache({'num_chunks': chunk_idx, 'stats': stats}, meta_path)

    return chunk_paths, stats


def process_strain_wrapper(args):
    """並列処理用ラッパー（picklableにするため外部関数）。"""
    strain_name, config_hash = args

    # 1. すでに新しいチャンク形式のキャッシュが存在するか確認
    meta_path = get_strain_meta_cache_path(strain_name, config_hash)
    if os.path.exists(meta_path):
        try:
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
            num_chunks = meta.get('num_chunks', 0)
            stats = meta.get('stats', None)
            chunk_paths = [get_strain_chunk_cache_path(strain_name, config_hash, i) for i in range(num_chunks)]
            if all(os.path.exists(p) for p in chunk_paths):
                if stats is None:
                    stats = {
                        'total_raw_lines': 0, 'excluded_format': 0, 'excluded_cooccur': 0,
                        'excluded_feature_gen_error': 0, 'excluded_short_path': 0, 'valid_samples': 0
                    }
                return strain_name, chunk_paths, True, stats  # キャッシュヒット
        except Exception:
            pass

    # 2. 以前の古い形式のキャッシュが存在する場合、新しいチャンク形式へ移行（マイグレーション）する
    old_cache_path = get_strain_cache_path(strain_name, config_hash)
    if os.path.exists(old_cache_path):
        try:
            old_data = load_strain_cache(old_cache_path)
            if old_data is not None:
                # 概算統計
                stats = {
                    'total_raw_lines': len(old_data), 'excluded_format': 0, 'excluded_cooccur': 0,
                    'excluded_feature_gen_error': 0, 'excluded_short_path': 0, 'valid_samples': len(old_data)
                }
                # チャンク分割して保存
                chunk_size = 5000
                chunk_paths = []
                chunk_idx = 0
                for i in range(0, len(old_data), chunk_size):
                    chunk_data = old_data[i:i + chunk_size]
                    p = get_strain_chunk_cache_path(strain_name, config_hash, chunk_idx)
                    save_strain_cache(chunk_data, p)
                    chunk_paths.append(p)
                    chunk_idx += 1
                
                # メタデータ保存
                save_strain_cache({'num_chunks': chunk_idx, 'stats': stats}, meta_path)
                return strain_name, chunk_paths, True, stats  # キャッシュヒット扱い
        except Exception:
            pass

    # 3. どちらのキャッシュもない場合は新規に再計算
    try:
        codon_data, freq_dict, dissim_dict, pam250_dict, log_ratio_dict, human_rscu_dict, scv2_rscu_dict = load_static_data(silent=True)
        chunk_paths, stats = process_strain_features_core_chunked(
            strain_name, codon_data, freq_dict, dissim_dict, pam250_dict,
            log_ratio_dict, human_rscu_dict, scv2_rscu_dict, config_hash
        )
        return strain_name, chunk_paths, False, stats
    except Exception as e:
        empty_stats = {
            'total_raw_lines': 0, 'excluded_format': 0, 'excluded_cooccur': 0,
            'excluded_feature_gen_error': 0, 'excluded_short_path': 0, 'valid_samples': 0
        }
        return strain_name, [], False, empty_stats


def flush_buffers_to_db(con, samples_buffer, features_buffer, labels_buffer):
    """バッファにあるデータを Pandas DataFrame を介して DuckDB に高速バルクインサートする。"""
    if not samples_buffer:
        return

    # トランザクションを明示的に開始して Flush 回数を激減させる
    con.execute("BEGIN TRANSACTION")

    try:
        # Pandas DataFrame 経由で一括インサート (DuckDBのネイティブ登録スキャン機能により極めて高速)
        df_samples = pd.DataFrame(samples_buffer, columns=[
            'sample_id', 'strain_id', 'raw_path', 'path_length', 
            'max_cooccurrence', 'split_type', 'split_type_date', 'strength_score', 'collection_date'
        ])
        con.execute("INSERT INTO samples SELECT * FROM df_samples")

        df_features = pd.DataFrame(features_buffer, columns=[
            'feature_id', 'sample_id', 'timestep', 'cooccurrence', 'cat_features', 'num_features'
        ])
        con.execute("INSERT INTO features SELECT * FROM df_features")

        df_labels = pd.DataFrame(labels_buffer, columns=[
            'label_id', 'sample_id', 'targets'
        ])
        con.execute("INSERT INTO labels SELECT * FROM df_labels")

        con.execute("COMMIT")
    except Exception as e:
        con.execute("ROLLBACK")
        force_print(f"[ERROR] Failed to flush buffers to database: {e}")
        raise e


def append_strain_to_buffers(strain_id, strain_strength, samples_data,
                             collection_date_dict,
                             samples_buffer, features_buffer, labels_buffer,
                             sample_id_counter, feature_id_counter, label_id_counter):
    """株の特徴量データをメモリ上のバッファへ追加する。

    Returns:
        Tuple[int, int, int]: 更新後の (sample_id, feature_id, label_id) カウンタ
    """
    for sample_name, raw_path_str, path_length, max_cooc, x_features, y_targets in samples_data:
        collection_date = collection_date_dict.get(sample_name, '')
        samples_buffer.append((
            sample_id_counter,
            strain_id,
            raw_path_str,
            path_length,
            max_cooc,
            0,  # split_type
            0,  # split_type_date
            strain_strength,
            collection_date
        ))

        for ts_idx, ts_events in enumerate(x_features):
            cooccurrence = len(ts_events)
            for event in ts_events:
                cat_feat, num_feat = event
                features_buffer.append((
                    feature_id_counter,
                    sample_id_counter,
                    ts_idx,
                    cooccurrence,
                    pickle.dumps(cat_feat),
                    pickle.dumps(num_feat)
                ))
                feature_id_counter += 1

        labels_buffer.append((
            label_id_counter,
            sample_id_counter,
            pickle.dumps(y_targets)
        ))
        label_id_counter += 1
        sample_id_counter += 1

    return sample_id_counter, feature_id_counter, label_id_counter


def main():
    """前処理メイン（並列処理版）"""
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import psutil

    start_time = time.time()

    force_print("=" * 60)
    force_print("Starting preprocessing (DuckDB mode, PARALLEL)...")
    force_print("=" * 60)

    config_hash = get_config_hash()[:8]
    force_print(f"[INFO] Config hash: {config_hash}")

    num_workers = min(multiprocessing.cpu_count(), config.NUM_PREPROCESS_WORKERS)
    force_print(f"[INFO] Using {num_workers} parallel workers")

    codon_data, freq_dict, dissim_dict, pam250_dict, log_ratio_dict, human_rscu_dict, scv2_rscu_dict = load_static_data()
    strain_to_strength = compute_strain_strength_from_csv()
    
    # 採取日マッピングをロード
    collection_date_dict = load_collection_dates()

    # psutilを用いた動的メモリ監視の初期化
    try:
        current_process = psutil.Process()
        total_memory = psutil.virtual_memory().total
        # システム総メモリの50%を強制フラッシュの閾値とする
        memory_limit = total_memory * 0.5
        force_print(f"[INFO] Dynamic memory monitoring enabled. Total memory: {total_memory / (1024**3):.1f} GB, Limit: {memory_limit / (1024**3):.1f} GB (50%)")
    except Exception as e:
        current_process = None
        memory_limit = 0
        force_print(f"[WARNING] Failed to initialize psutil monitoring: {e}. Falling back to size-only threshold.")

    db_path = init_db()
    con = connect_db(db_path)

    try:
        strains = import_strains_to_db(con, strain_to_strength)

        processed_strains = get_processed_strains(con)
        if processed_strains:
            force_print(f"[INFO] Resuming: {len(processed_strains)} strains already processed in DB")

        strains_to_process = [(s, config_hash) for s in strains if s not in processed_strains]
        force_print(f"[INFO] Strains to process: {len(strains_to_process)}")

        if not strains_to_process:
            force_print("[INFO] All strains already processed!")
        else:
            force_print("[INFO] Processing strains (parallel generate + streaming write)...")

            # バッファの初期化
            samples_buffer = []
            features_buffer = []
            labels_buffer = []
            
            # バルク書き込みの閾値 (samples 件数基準で config.DB_WRITE_BATCH_SIZE を使用)
            FLUSH_SAMPLE_THRESHOLD = getattr(config, 'DB_WRITE_BATCH_SIZE', 10000)

            # 統計用グローバルカウンター
            global_stats = {
                'total_raw_lines': 0,
                'excluded_format': 0,
                'excluded_cooccur': 0,
                'excluded_feature_gen_error': 0,
                'excluded_short_path': 0,
                'valid_samples': 0
            }

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
                        result_name, chunk_paths, from_cache, stats = future.result()

                        if from_cache:
                            cache_hits += 1

                        # 統計カウンターの合算
                        for k, v in stats.items():
                            global_stats[k] += v

                        if chunk_paths:
                            strain_id = strain_id_map.get(result_name, 0)
                            strain_strength = strain_to_strength.get(result_name, 0.0)

                            # 各チャンクファイルを1つずつディスクからロードして処理
                            strain_samples = 0
                            for chunk_path in chunk_paths:
                                samples_data = load_strain_cache(chunk_path)
                                if samples_data:
                                    strain_samples += len(samples_data)
                                    total_samples += len(samples_data)

                                    # バッファに追加
                                    sample_id, feature_id, label_id = append_strain_to_buffers(
                                        strain_id, strain_strength, samples_data,
                                        collection_date_dict,
                                        samples_buffer, features_buffer, labels_buffer,
                                        sample_id, feature_id, label_id
                                    )

                                    del samples_data

                                    # メモリ制限（システム総メモリの50%）チェック
                                    is_mem_limit = False
                                    if current_process is not None:
                                        try:
                                            process_rss = current_process.memory_info().rss
                                            if process_rss > memory_limit:
                                                force_print(f"[INFO] Memory limit (50%) exceeded: {process_rss / (1024**3):.2f} GB / {total_memory / (1024**3):.2f} GB. Forcing flush to DuckDB.")
                                                is_mem_limit = True
                                        except Exception:
                                            pass

                                    # チャンクごとにしきい値判定し、達していたら即DBへフラッシュしてメモリ解放
                                    if len(samples_buffer) >= FLUSH_SAMPLE_THRESHOLD or is_mem_limit:
                                        flush_buffers_to_db(con, samples_buffer, features_buffer, labels_buffer)
                                        samples_buffer.clear()
                                        features_buffer.clear()
                                        labels_buffer.clear()
                                        gc.collect()
                        else:
                            strain_samples = 0

                        processed += 1

                        pbar.set_postfix({
                            'last': strain_samples,
                            'total': f'{total_samples:,}',
                            'cache': cache_hits
                        })

                    except Exception as e:
                        force_print(f"[WARNING] Failed to process {strain_name}: {e}")

                    del futures[future]

                    if processed % 50 == 0:
                        gc.collect()

            # 残りのバッファを書き込み
            if samples_buffer:
                flush_buffers_to_db(con, samples_buffer, features_buffer, labels_buffer)
                samples_buffer.clear()
                features_buffer.clear()
                labels_buffer.clear()
                gc.collect()

            force_print(f"[INFO] Complete: {processed} strains, {total_samples:,} samples, {cache_hits} cache hits")

        # インデックスを一括作成（インサートがすべて終わった後に行うことで劇的に高速化）
        create_db_indexes(con)

        assign_splits_auto(con)
        con.close()

        # 統計レポートの表示
        force_print("=" * 60)
        force_print("📊 PREPROCESSING FILTER STATISTICS REPORT")
        force_print("=" * 60)
        total_raw = global_stats['total_raw_lines']
        valid = global_stats['valid_samples']
        
        def print_stat_row(label, val, total):
            pct = (val / total * 100) if total > 0 else 0.0
            force_print(f"  {label:<30} : {val:>9,} ({pct:>6.2f}%)")

        force_print(f"  Total Raw Samples Read         : {total_raw:>9,}")
        print_stat_row("Excluded by Format Error", global_stats['excluded_format'], total_raw)
        print_stat_row("Excluded by Co-occurrence Lim", global_stats['excluded_cooccur'], total_raw)
        print_stat_row("Excluded by Feature Gen Error", global_stats['excluded_feature_gen_error'], total_raw)
        print_stat_row("Excluded by Short Path (<=1 ts)", global_stats['excluded_short_path'], total_raw)
        force_print("  " + "-" * 56)
        print_stat_row("Final Valid Samples Written", valid, total_raw)
        force_print("=" * 60)

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
