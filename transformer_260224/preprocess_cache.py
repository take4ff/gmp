# preprocess_cache.py
# Step 1: 特徴量生成 → 株ごとのキャッシュ保存
# Usage: python -m transformer_260224.preprocess_cache
#
# 旧実装と同等の速度で特徴量を生成し、株ごとにpickleキャッシュに保存する。
# DBへの書き込みは行わない（Step 2で行う）。

import os
import pickle
import gc
import time
import pandas as pd
from tqdm import tqdm
from datetime import datetime

from . import config
from .dataset import (
    import_mutation_paths,
    filter_co_occur,
    preprocess_static_data,
    Feature_path_fast,
)
from .utils import get_config_hash


def force_print(msg):
    """即時出力"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)


def load_static_data():
    """静的データ（コドン、頻度、物性値）を読み込み"""
    force_print("[INFO] Loading static data...")
    
    df_codon = pd.read_csv(config.Codon_csv)
    df_freq = pd.read_csv(config.Freq_csv, index_col=0)
    df_dissimilarity = pd.read_csv(config.Disimilarity_csv)
    df_pam250 = pd.read_csv(config.PAM250_csv, index_col=0)
    
    codon_data, freq_dict, dissim_dict, pam250_dict = preprocess_static_data(
        df_codon, df_freq, df_dissimilarity, df_pam250
    )
    
    return codon_data, freq_dict, dissim_dict, pam250_dict


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
    
    force_print(f"[INFO] Computed strength for {len(strain_to_strength)} strains")
    return strain_to_strength


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
        force_print(f"[WARNING] Failed to save cache: {e}")


def process_strain(strain_name, codon_data, freq_dict, dissim_dict, pam250_dict, config_hash):
    """
    1つの株の特徴量を生成（キャッシュ対応）
    
    Returns:
        list of (path_str, path_length, max_cooc, x_features, y_targets)
    """
    # キャッシュ確認
    cache_path = get_strain_cache_path(strain_name, config_hash)
    cached = load_strain_cache(cache_path)
    if cached is not None:
        return cached, True  # (data, from_cache)
    
    usher_dir = config.DATA_BASE_DIR
    file_paths = import_mutation_paths(usher_dir, strain_name)
    
    if not file_paths:
        return [], False
    
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
    save_strain_cache(results, cache_path)
    
    return results, False


def main():
    """Step 1: 特徴量生成 → キャッシュ保存"""
    start_time = time.time()
    
    force_print("=" * 60)
    force_print("Step 1: Generating features → Cache")
    force_print("=" * 60)
    
    # 設定ハッシュ
    config_hash = get_config_hash()[:8]
    force_print(f"[INFO] Config hash: {config_hash}")
    
    # 静的データ読み込み
    codon_data, freq_dict, dissim_dict, pam250_dict = load_static_data()
    
    # 流行度計算
    strain_to_strength = compute_strain_strength_from_csv()
    
    # 株一覧を取得
    usher_dir = config.DATA_BASE_DIR
    strains = sorted([f for f in os.listdir(usher_dir) if not f.startswith('.')])
    force_print(f"[INFO] Found {len(strains)} strains")
    
    # 処理
    total_samples = 0
    cache_hits = 0
    processed = 0
    
    pbar = tqdm(strains, desc="Generating features")
    for strain_name in pbar:
        samples_data, from_cache = process_strain(
            strain_name, codon_data, freq_dict, dissim_dict, pam250_dict, config_hash
        )
        
        strain_samples = len(samples_data)
        total_samples += strain_samples
        
        if from_cache:
            cache_hits += 1
        
        processed += 1
        
        pbar.set_postfix({
            'last': strain_samples,
            'total': f'{total_samples:,}',
            'cache': cache_hits
        })
        
        # メモリ解放
        del samples_data
        if processed % 100 == 0:
            gc.collect()
    
    elapsed = time.time() - start_time
    force_print("=" * 60)
    force_print(f"Step 1 completed in {elapsed:.1f} seconds!")
    force_print(f"Total: {processed} strains, {total_samples:,} samples")
    force_print(f"Cache hits: {cache_hits}")
    force_print(f"Cache dir: {os.path.join(config.INCREMENTAL_CACHE_DIR, 'strains')}")
    force_print("=" * 60)
    force_print("")
    force_print("Next: Run Step 2 to convert cache to DB")
    force_print("  python -m transformer_260224.preprocess_db")


if __name__ == "__main__":
    main()
