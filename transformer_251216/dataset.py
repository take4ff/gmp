# --- dataset.py ---
import os
import time
import pickle
import hashlib
import math
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

# ローカルモジュール
from . import config
from .utils import force_print,get_config_hash, get_batch_cache_path, load_batch_cache, save_batch_cache

# ==========================================
# 1. 定数・ヘルパーデータ
# ==========================================

DNA2Protein = {
    'TTT': 'F', 'TCT': 'S', 'TAT': 'Y', 'TGT': 'C',
    'TTC': 'F', 'TCC': 'S', 'TAC': 'Y', 'TGC': 'C',
    'TTA': 'L', 'TCA': 'S', 'TAA': '*', 'TGA': '*',
    'TTG': 'L', 'TCG': 'S', 'TAG': '*', 'TGG': 'W',

    'CTT': 'L', 'CCT': 'P', 'CAT': 'H', 'CGT': 'R',
    'CTC': 'L', 'CCC': 'P', 'CAC': 'H', 'CGC': 'R',
    'CTA': 'L', 'CCA': 'P', 'CAA': 'Q', 'CGA': 'R',
    'CTG': 'L', 'CCG': 'P', 'CAG': 'Q', 'CGG' : 'R',

    'ATT': 'I', 'ACT': 'T', 'AAT': 'N', 'AGT': 'S',
    'ATC': 'I', 'ACC': 'T', 'AAC': 'N', 'AGC': 'S',
    'ATA': 'I', 'ACA': 'T', 'AAA': 'K', 'AGA': 'R',
    'ATG': 'M', 'ACG': 'T', 'AAG': 'K', 'AGG': 'R',

    'GTT': 'V', 'GCT': 'A', 'GAT': 'D', 'GGT': 'G',
    'GTC': 'V', 'GCC': 'A', 'GAC': 'D', 'GGC': 'G',
    'GTA': 'V', 'GCA': 'A', 'GAA': 'E', 'GGA': 'G',
    'GTG': 'V', 'GCG': 'A', 'GAG': 'E', 'GGG': 'G',
    'nnn': 'n'
}

# ==========================================
# 2. データ読み込み・フィルタリング関数
# (ここは Pandas を使っても回数が少ないので変更なし)
# ==========================================

def import_mutation_paths(base_dir, strain):
    base_dir = os.path.expanduser(base_dir)
    strain_dir = os.path.join(base_dir, strain)
    file_paths = []
    
    file_path = os.path.join(strain_dir, "mutation_paths.tsv")
    if os.path.exists(file_path):
        file_paths.append(file_path)
    else:
        if os.path.exists(strain_dir) and os.path.isdir(strain_dir):
            num_dirs = [d for d in os.listdir(strain_dir) if d.isdigit()]
            num_dirs.sort(key=int)
            for num in num_dirs:
                file_path = os.path.join(strain_dir, num, "mutation_paths.tsv")
                if os.path.exists(file_path):
                    file_paths.append(file_path)
    return file_paths

def filter_co_occur(path, max_co_occur):
    for mutations in path:
        if len(mutations.split(',')) > max_co_occur:
            return False
    return True

def filter_unique(names, lengths, mutation_paths, strains):
    df = pd.DataFrame({
        'name': names, 
        'original_len': lengths, 
        'path': mutation_paths, 
        'strain': strains
    })
    force_print(f"Initial samples loaded: {len(df)}")

    df['path_tuple'] = df['path'].apply(tuple)
    df_unique = df.drop_duplicates(subset='path_tuple', keep='first')
    force_print(f"Samples after removing duplicate sequences: {len(df_unique)}")
    
    return df_unique.drop(columns=['path_tuple'])

def sort_strain_by_num_and_filter_strain(df, max_strain_num):
    force_print(f"[INFO] Filtering strains to top {max_strain_num} by occurrence count...")
    strain_counts = df['strain'].value_counts()
    top_strains = strain_counts.nlargest(max_strain_num).index.tolist()
    
    filtered_df = df[df['strain'].isin(top_strains)].reset_index(drop=True)
    force_print(f"[INFO] Samples after filtering: {len(filtered_df)}")
    return filtered_df

def filter_num_per_strain(df, max_num_per_strain):
    force_print(f"[INFO] Filtering samples to max {max_num_per_strain} per strain...")
    filtered_dfs = []
    for _, group in df.groupby('strain'):
        if len(group) > max_num_per_strain:
            sampled_group = group.sample(n=max_num_per_strain, random_state=config.SEED)
            filtered_dfs.append(sampled_group)
        else:
            filtered_dfs.append(group)
    
    filtered_df = pd.concat(filtered_dfs).reset_index(drop=True)
    force_print(f"[INFO] Samples after filtering: {len(filtered_df)}")
    return filtered_df

def sample_proportionally(df, target_num):
    """
    各株の比率を維持しながら指定サンプル数を抽出する（比率サンプリング）
    
    Args:
        df: データフレーム（'strain'列を含む）
        target_num: 目標サンプル数
    
    Returns:
        サンプリング後のDataFrame
    """
    total = len(df)
    if total <= target_num:
        force_print(f"[INFO] 比率サンプリング: 全件({total})がtarget_num({target_num})以下のため、全件を使用")
        return df
    
    force_print(f"[INFO] 比率サンプリング: {total}件から{target_num}件を抽出（各株の比率を維持）")
    
    # 各株のカウントと比率
    strain_counts = df['strain'].value_counts()
    
    # 各株から抽出するサンプル数を計算（比率に基づく）
    strain_samples = {}
    remaining = target_num
    
    # 小さい株から順に処理（端数処理のため）
    sorted_strains = strain_counts.sort_values().index.tolist()
    
    for i, strain in enumerate(sorted_strains):
        count = strain_counts[strain]
        # 残りの株数
        remaining_strains = len(sorted_strains) - i
        
        # この株から抽出する比率
        ratio = count / strain_counts[sorted_strains[i:]].sum()
        samples_from_this = min(count, max(1, int(remaining * ratio)))
        
        # 最後の株は残り全てを割り当て
        if i == len(sorted_strains) - 1:
            samples_from_this = min(count, remaining)
        
        strain_samples[strain] = samples_from_this
        remaining -= samples_from_this
    
    # 各株からサンプリング
    sampled_dfs = []
    for strain, n_samples in strain_samples.items():
        strain_df = df[df['strain'] == strain]
        if len(strain_df) > n_samples:
            sampled_df = strain_df.sample(n=n_samples, random_state=config.SEED)
        else:
            sampled_df = strain_df
        sampled_dfs.append(sampled_df)
    
    result_df = pd.concat(sampled_dfs).reset_index(drop=True)
    force_print(f"[INFO] 比率サンプリング完了: {len(result_df)}件, {len(strain_samples)}株")
    
    return result_df

def info_import(paths, lengths, over_num):
    force_print(f"[INFO] 全件読み込み完了(共起数フィルタ適用後): {len(paths)} サンプル")
    force_print(f"[INFO] 共起数が最大値を超えたため除外されたサンプル数: {over_num}")

    print("\n[INFO] シーケンス長ごとのサンプル数:")
    if lengths:
        length_counts = pd.Series(lengths).value_counts().sort_index()
        with pd.option_context('display.max_rows', None):
            print(length_counts)
    else:
        print("データがありません。")
    
    # 共起数別の統計
    if paths:
        max_cooccur_per_sample = []
        for path in paths:
            max_co = max(len(mutations.split(',')) for mutations in path)
            max_cooccur_per_sample.append(max_co)
        
        print("\n[INFO] 最大共起数ごとのサンプル数:")
        cooccur_counts = pd.Series(max_cooccur_per_sample).value_counts().sort_index()
        with pd.option_context('display.max_rows', None):
            print(cooccur_counts)


def print_strength_distribution(strains, strain_to_strength):
    """強度スコアのカテゴリ別サンプル数を表示（動的閾値、1刻みのヒストグラム付き）"""
    import math
    from collections import Counter
    
    # 全サンプルのスコアを収集
    scores = [strain_to_strength.get(strain, 0.0) for strain in strains]
    
    if not scores:
        force_print("[INFO] サンプルがありません")
        return
    
    # 最小値・最大値から動的に閾値を計算（整数に丸める）
    min_score = min(scores)
    max_score = max(scores)
    score_range = max_score - min_score
    
    # 3等分の閾値を整数で計算
    low_max = int(min_score + score_range / 3) + 1  # 切り上げ気味
    med_max = int(min_score + 2 * score_range / 3) + 1  # 切り上げ気味
    
    low_count = 0
    med_count = 0
    high_count = 0
    
    # 1刻みのヒストグラム用
    score_bins = Counter()
    
    for score in scores:
        if score < low_max:
            low_count += 1
        elif score < med_max:
            med_count += 1
        else:
            high_count += 1
        
        # 1刻みでビン化（例: 5.3 → 5）
        bin_key = int(score)
        score_bins[bin_key] += 1
    
    print(f"\n[INFO] 強度スコア別サンプル数 (動的閾値: Low<{low_max}, Medium<{med_max}, High≥{med_max}):")
    print(f"  スコア範囲: {min_score:.1f} ~ {max_score:.1f}")
    print(f"  Low: {low_count}")
    print(f"  Medium: {med_count}")
    print(f"  High: {high_count}")
    
    # 1刻みのヒストグラム表示
    print(f"\n[INFO] 強度スコア分布 (1刻み):")
    for bin_key in sorted(score_bins.keys()):
        count = score_bins[bin_key]
        print(f"  {bin_key}-{bin_key+1}: {count}")


def compute_dynamic_thresholds(strain_to_strength):
    """
    全データの強度スコアから動的閾値を計算
    
    Returns:
        (low_max, med_max): 整数の閾値
    """
    if not strain_to_strength:
        return 3, 5  # デフォルト値
    
    scores = list(strain_to_strength.values())
    min_score = min(scores)
    max_score = max(scores)
    score_range = max_score - min_score
    
    # 3等分の閾値を整数で計算
    low_max = int(min_score + score_range / 3) + 1
    med_max = int(min_score + 2 * score_range / 3) + 1
    
    return low_max, med_max

def import_strains(usher_dir, max_num=None, max_cooccur=10):
    if not os.path.exists(usher_dir):
        raise FileNotFoundError(f"Directory not found: {usher_dir}")

    strains = sorted([f for f in os.listdir(usher_dir) if not f.startswith('.')])
    force_print(f"[INFO] データ読み込み開始: {len(strains)} strains found in {usher_dir}")

    names, lengths, paths, paths_strain = [], [], [], []
    over_num = 0

    for strain in strains:
        file_paths = import_mutation_paths(usher_dir, strain)
        for file_path in file_paths:
            try:
                with open(file_path, 'r', encoding="utf-8_sig") as f:
                    datalist = f.readlines()
            except Exception as e:
                force_print(f"[WARNING] Failed to read {file_path}: {e}")
                continue

            for line in datalist[1:]: 
                data = line.strip().split('\t')
                if len(data) < 3: continue
                path_str = data[2].split('>')
                
                if filter_co_occur(path_str, max_cooccur):
                    names.append(data[0])
                    lengths.append(int(data[1]))
                    paths.append(path_str)
                    paths_strain.append(strain)
                else:
                    over_num += 1
                
                if max_num is not None and len(paths) >= max_num:
                    info_import(paths, lengths, over_num)
                    return names, lengths, paths, paths_strain

    info_import(paths, lengths, over_num)
    return names, lengths, paths, paths_strain

# ==========================================
# 3. 特徴量計算ロジック (Pandas排除・高速化)
# ==========================================

def preprocess_static_data(df_codon, df_freq, df_dissimilarity, df_pam250):
    """
    DataFrameを高速アクセス可能なPythonネイティブ型(Dict, List)に変換する
    """
    force_print("[INFO] Converting DataFrames to optimized structures...")
    
    # 1. Codon Data (更新用Stateの初期値)
    # 各カラムをリストとして保持 (インデックスアクセス用)
    codon_data = {
        'base': df_codon['base'].tolist(),
        'protein': df_codon['protein'].tolist(),
        'protein_pos': df_codon['protein_pos'].tolist(),
        'codon': df_codon['codon'].tolist(),
        'codon_pos': df_codon['codon_pos'].tolist()
    }
    
    # 2. Frequency Data (参照用)
    # { "A->T": [freq_pos0, freq_pos1, ...], ... }
    freq_dict = df_freq.to_dict(orient='list')
    
    # 3. Dissimilarity Data (参照用)
    # { (wt, mut): {hydro: ..., charge: ...}, ... }
    dissim_dict = {}
    # itertuplesは高速
    for row in df_dissimilarity.itertuples():
        # row.wt, row.mut, row.eisenberg_weiss_diff ...
        dissim_dict[(row.wt, row.mut)] = {
            'hydro': row.eisenberg_weiss_diff,
            'charge': row.charge_diff,
            'size': row.size_diff,
            'blsm': row.blsm62_diff
        }

    # { (wt, mut): score }
    # IndexとColumnsがアミノ酸1文字表記であると仮定
    pam250_dict = {}
    # 行(index)と列(columns)でループ
    for aa1 in df_pam250.index:
        for aa2 in df_pam250.columns:
            pam250_dict[(aa1, aa2)] = float(df_pam250.at[aa1, aa2])
        
    return codon_data, freq_dict, dissim_dict, pam250_dict

def Feature_from_csv_fast(mutation, codon_state, freq_dict, dissim_dict, pam250_dict):
    """
    高速化版: Pandasを使わずList/Dictのみで計算
    codon_state: { 'base': [...], 'codon': [...], ... } (ミュータブル)
    """
    # パース
    base_pos = int(mutation[1:-1])
    bef = mutation[0]
    aft = mutation[-1]
    
    idx = base_pos - 1 # 0-index
    
    # 状態取得 (リストアクセスは爆速)
    base = str(codon_state['base'][idx])
    protein = str(codon_state['protein'][idx])
    protein_pos = int(codon_state['protein_pos'][idx])
    codon = str(codon_state['codon'][idx])
    codon_pos = int(codon_state['codon_pos'][idx])

    new_codon = codon
    
    # コドン更新ロジック
    if bef == base and codon != "none":
        # 1. 塩基の更新
        codon_state['base'][idx] = aft
        
        # 2. コドンの更新
        if 1 <= codon_pos <= 3:
            # 文字列はイミュータブルなのでリスト化して置換
            c_list = list(codon)
            c_list[codon_pos - 1] = aft
            new_codon = "".join(c_list)
            
            # 3. 影響する3つの位置すべてのコドン情報を更新
            # コドンの開始位置 (0-indexed)
            start_idx = idx - (codon_pos - 1)
            
            # 範囲チェックしつつ更新
            limit = len(codon_state['codon'])
            for k in range(3):
                curr = start_idx + k
                if 0 <= curr < limit:
                    codon_state['codon'][curr] = new_codon
    
    # 頻度取得 (Dict access)
    mut_key = f"{bef}->{aft}"
    freq = freq_dict[mut_key][idx] if mut_key in freq_dict else 0.0
    
    # 物性値取得 (Dict access)
    bef_aa = DNA2Protein.get(codon, 'X')
    aft_aa = DNA2Protein.get(new_codon, 'X')
    
    metrics = dissim_dict.get((bef_aa, aft_aa))
    if metrics:
        hydro = metrics['hydro']
        charge = metrics['charge']
        size = metrics['size']
        blsm = metrics['blsm']
    else:
        hydro = charge = size = blsm = 0.0
    
    pam250 = pam250_dict.get((bef_aa, aft_aa), 0.0)

    return codon, new_codon, codon_pos, protein, protein_pos, freq, hydro, charge, size, blsm, pam250

def Mutation_features_fast(mutations_str, codon_state, freq_dict, dissim_dict, pam250_dict):
    """高速版特徴量生成"""
    features = []
    for mutation in mutations_str.split(','):
        base_pos = int(mutation[1:-1])
        bef = mutation[0]
        aft = mutation[-1]

        # 高速版関数を呼び出し
        codon, new_codon, codon_pos, protein, protein_pos, freq, hydro, charge, size, blsm, pam250 = \
            Feature_from_csv_fast(mutation, codon_state, freq_dict, dissim_dict, pam250_dict)
        
        if codon == 'none': codon = 'nnn'
        if new_codon == 'none': new_codon = 'nnn'
        
        bef_token = config.BASE_VOCABS.get(bef, config.BASE_VOCABS['n'])
        aft_token = config.BASE_VOCABS.get(aft, config.BASE_VOCABS['n'])
        aa_bef_token = config.AA_VOCABS.get(DNA2Protein.get(codon, 'n'), config.AA_VOCABS['n'])
        aa_aft_token = config.AA_VOCABS.get(DNA2Protein.get(new_codon, 'n'), config.AA_VOCABS['n'])
        protein_token = config.PROTEIN_VOCABS.get(protein, config.PROTEIN_VOCABS['PAD'])

        cat_feat = [bef_token, base_pos, aft_token, codon_pos, 
                    aa_bef_token, protein_pos, aa_aft_token, protein_token]
        num_feat = [freq, hydro, charge, size, blsm, pam250]

        features.append((cat_feat, num_feat))
        
    return features

def Feature_path_fast(mutation_path, codon_state_template, freq_dict, dissim_dict, pam250_dict):
    """
    1パス分の処理 (高速版)
    codon_state_template: 初期状態の辞書(リスト含む)
    """
    # Stateのディープコピー (リストのスライス[:]は高速)
    # 各列をコピーして新しい辞書を作る
    local_state = {
        'base': codon_state_template['base'][:],
        'protein': codon_state_template['protein'], # 変更されないなら参照でもいいが、念のためコピー推奨
        'protein_pos': codon_state_template['protein_pos'],
        'codon': codon_state_template['codon'][:], # コドンは更新されるのでコピー必須
        'codon_pos': codon_state_template['codon_pos']
    }
    
    # protein, protein_pos, codon_pos はロジック上更新されないなら
    # コピーせず参照渡しにすればさらに高速化可能 (上記は安全策)

    path_features = []
    for mutations_str in mutation_path:
        ts_features = Mutation_features_fast(mutations_str, local_state, freq_dict, dissim_dict, pam250_dict)
        path_features.append(ts_features)
        
    return path_features

# ==========================================
# 4. 並列・インクリメンタル処理ヘルパー
# ==========================================

def process_batch_paths(paths, codon_data, freq_dict, dissim_dict, pam250_dict):
    """シングルプロセスでのバッチ処理 (高速版)"""
    batch_features = []
    for path in paths:
        features = Feature_path_fast(path, codon_data, freq_dict, dissim_dict, pam250_dict)
        batch_features.append(features)
    return batch_features

def process_batch_parallel(paths, codon_data, freq_dict, dissim_dict, pam250_dict):
    """マルチスレッドでのバッチ処理"""
    results = []
    max_workers = min(multiprocessing.cpu_count(), 8)
    
    chunk_size = (len(paths) + max_workers - 1) // max_workers
    chunks = [paths[i:i + chunk_size] for i in range(0, len(paths), chunk_size)]
    
    # ThreadPoolExecutorを使用 (Dict/ListはGIL下でも共有メモリで高速に読めるため相性が良い)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for chunk in chunks:
            futures.append(
                executor.submit(process_batch_paths, chunk, codon_data, freq_dict, dissim_dict, pam250_dict)
            )
        
        for future in futures:
            try:
                results.extend(future.result())
            except Exception as e:
                force_print(f"[ERROR] Parallel worker failed: {e}")
                raise e
    return results

def get_mutation_data(names, lengths, paths, df_codon, df_freq, df_dissimilarity, df_pam250):
    """
    メイン関数: DataFrameを最適化してから処理を開始
    """
    force_print(f"[INFO] Pre-processing data structures for speed...")
    # ここでDataFrameを高速なDict/Listに変換
    codon_data, freq_dict, dissim_dict, pam250_dict = preprocess_static_data(df_codon, df_freq, df_dissimilarity, df_pam250)

    force_print(f"[INFO] Generating mutation features for {len(paths)} paths (Incremental Cache Mode)...")
    config_hash = get_config_hash()
    
    batch_size = getattr(config, 'BATCH_SIZE_FEATURE_GEN', 5000)
    total_samples = len(paths)
    total_batches = (total_samples + batch_size - 1) // batch_size
    
    all_features_paths = []
    start_time = time.time()
    
    for i in range(total_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, total_samples)
        cache_path = get_batch_cache_path(i, total_batches, config_hash)
        
        # キャッシュ確認
        batch_data = None
        if not config.FORCE_REPROCESS:
            batch_data = load_batch_cache(cache_path)
            
        if batch_data is not None:
            if (i + 1) % 5 == 0 or i == 0:
                force_print(f"[INFO] Batch {i+1}/{total_batches}: Loaded from cache.")
            all_features_paths.extend(batch_data)
        else:
            force_print(f"[INFO] Batch {i+1}/{total_batches}: Processing {start_idx} to {end_idx}...")
            batch_paths = paths[start_idx:end_idx]
            
            # 高速化されたデータ構造を渡す
            if config.ENABLE_PARALLEL_PROCESSING:
                batch_data = process_batch_parallel(batch_paths, codon_data, freq_dict, dissim_dict, pam250_dict)
            else:
                batch_data = process_batch_paths(batch_paths, codon_data, freq_dict, dissim_dict, pam250_dict)
            
            save_batch_cache(batch_data, cache_path)
            all_features_paths.extend(batch_data)
            
            import gc
            gc.collect()

    end_time = time.time()
    force_print(f"[INFO] Feature generation completed in {end_time - start_time:.2f} seconds")
    
    return names, lengths, all_features_paths

# ==========================================
# 5. 強度スコア計算
# ==========================================

def compute_strain_strength(strains):
    """
    株別サンプル数から強度スコア (log(1 + count)) を計算する
    Returns: strain_to_strength dict
    """
    from collections import Counter
    strain_counts = Counter(strains)
    
    # log(1 + count) で正規化
    strain_to_strength = {}
    for strain, count in strain_counts.items():
        strain_to_strength[strain] = math.log(1 + count)
    
    force_print(f"[INFO] Strain strength scores computed for {len(strain_to_strength)} strains")
    force_print(f"[INFO] Strength score range: {min(strain_to_strength.values()):.2f} - {max(strain_to_strength.values()):.2f}")
    
    return strain_to_strength

# ==========================================
# 6. データセット分割・整形
# ==========================================
def split_data_by_length(df, train_len, valid_num, valid_ratio, seed):
    overlap_start_len = train_len - valid_num + 1
    train_df = df[df['original_len'] < overlap_start_len].copy()
    test_df = df[df['original_len'] > train_len].copy()
    overlap_df = df[
        (df['original_len'] >= overlap_start_len) &
        (df['original_len'] <= train_len)
    ].copy()

    overlap_df = overlap_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    n_valid_samples = int(len(overlap_df) * valid_ratio)
    
    valid_part_df = overlap_df.iloc[:n_valid_samples]
    train_part_df = overlap_df.iloc[n_valid_samples:]

    valid_df = valid_part_df
    train_df = pd.concat([train_df, train_part_df], ignore_index=True)

    force_print(f"Train samples: {len(train_df)}")
    force_print(f"Validation samples: {len(valid_df)}")
    force_print(f"Test samples: {len(test_df)}")
    return train_df, valid_df, test_df

def separate_XY(feature_paths, original_lengths, raw_paths, strain_names, strain_to_strength, max_x_len, ylen=1):
    """
    シーケンスを特徴量(X)とラベル(Y)に分割し、メタデータ(Strain, 強度スコア等)も保持する
    
    Args:
        strain_to_strength: {strain_name: strength_score} の辞書
    
    Returns:
        list of (x, y_targets, original_len, raw_path, strain, strength_score)
        y_targets: list of (region_id, position_id, protein_pos_id)
    """
    x_y_len_list = []
    
    force_print(f"[INFO] Splitting data into X and Y...")
    # zipにstrain_namesを追加
    for item, original_len, raw_path, strain in zip(feature_paths, original_lengths, raw_paths, strain_names):
        if len(item) > ylen:
            # X: 最後を除くすべて
            x = item[:-ylen]
            if len(x) > max_x_len:
                x = x[-max_x_len:]

            # Y: 最後のタイムステップ
            y_timestep = item[-ylen:][0]
            
            y_targets = []
            for event in y_timestep:
                cat_features = event[0]
                region_id = cat_features[7]        # タンパク質領域
                position_id = cat_features[1]      # 塩基位置
                protein_pos_id = cat_features[5]   # タンパク質位置
                y_targets.append((region_id, position_id, protein_pos_id))

            # 強度スコアを取得
            strength_score = strain_to_strength.get(strain, 0.0)
            
            # Strain情報と強度スコアもタプルに追加
            x_y_len_list.append((x, y_targets, original_len, raw_path, strain, strength_score))
            
    return x_y_len_list

# ==========================================
# 7. PyTorch Dataset & DataLoader
# ==========================================
class ViralDataset(Dataset):
    def __init__(self, data):
        """
        Args:
            data: list of (x, y_targets, original_len, raw_path, strain, strength_score)
        """
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        sequence, targets, original_len, raw_path, strain, strength_score = self.data[idx]

        target_len = config.TARGET_LEN
        
        if len(raw_path) > target_len:
            input_path_str = raw_path[:-target_len] # 末尾以外を入力とする
            target_path_str = raw_path[-target_len:] # 末尾を正解とする
        else:
            # 万が一パスが短すぎる場合
            input_path_str = []
            target_path_str = raw_path

        # モデルの入力長制限 (MAX_SEQ_LEN) に合わせて、文字列リストも切り捨てる
        # これにより、モデルが実際に見た範囲と一致させる
        if len(input_path_str) > config.MAX_SEQ_LEN:
            # 末尾（最新）から MAX_SEQ_LEN 分だけ残す
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
                if c >= config.MAX_CO_OCCURRENCE: break 

                # 元データを壊さないようコピーを作成
                cat = list(cat_origin) 
                num = list(num_origin)

                # cat構造: [bef(0), pos(1), aft(2), c_pos(3), aa_b(4), p_pos(5), aa_a(6), region(7)]
                # 注: p_pos(5)はタンパク質位置で予測対象のためマスク対象外
                
                # 1. コドン位置のマスク
                if config.ABLATION_MASKS['CODON_POS']:
                    cat[3] = 0 # PAD
                
                # 2. アミノ酸変異(前後)のマスク
                if config.ABLATION_MASKS['AA_MUTATION']:
                    cat[4] = 0 # PAD
                    cat[6] = 0 # PAD
                
                # 3. 数値特徴量のマスク
                if config.ABLATION_MASKS['CHEM_FEATURES']:
                    num = [0.0] * len(num)

                # 4. 共起情報のマスク
                # (共起変異がある場合、2つ目以降(c>0)を強制的にパディング扱いにすることで無視する)
                if config.ABLATION_MASKS['CO_OCCURRENCE'] and c > 0:
                    # 全て0にすればEmbeddingも0になり、パディングと同じ扱いになる
                    cat = [0] * len(cat)
                    num = [0.0] * len(num)

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

def create_dataloader(data, batch_size, shuffle=True):
    dataset = ViralDataset(data)
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

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

# ==========================================
# 8. 統合データ準備関数
# ==========================================

def prepare_all_data():
    """
    全データの読み込み・前処理・分割を一括で行う
    
    強度スコアは全データベースから計算され、キャッシュ利用時も
    最新のスコアが適用される。
    
    Returns:
        train, valid, test: 分割されたデータセット
        data_info: {train_min_len, train_max_len, val_min_len, val_max_len, test_min_len, test_max_len}
    """
    from .utils import load_cache, save_cache, get_config_hash
    
    config_hash = get_config_hash()
    cache_path = os.path.join(config.CACHE_DIR, f"data_cache_{config_hash}.pkl")
    
    # デフォルト値
    data_info = {
        'train_min_len': 0, 'train_max_len': 0,
        'val_min_len': 0, 'val_max_len': 0,
        'test_min_len': 0, 'test_max_len': 0
    }
    
    # =================================================================
    # Step 1: 全件読み込み（キャッシュ有無に関わらず実行）
    # =================================================================
    force_print("--- Loading All Data for Statistics and Strength Calculation ---")
    
    # 生データのインポート（全件読み込み）
    all_names, all_lengths, all_paths, all_strains = import_strains(
        usher_dir=config.DATA_BASE_DIR, 
        max_num=None,  # 全件読み込み
        max_cooccur=config.MAX_CO_OCCURRENCE
    )
    
    # 全データで強度スコアを計算
    force_print("[INFO] Computing strain strength scores from ALL data...")
    strain_to_strength = compute_strain_strength(all_strains)
    
    # 動的閾値を計算して data_info に追加
    low_max, med_max = compute_dynamic_thresholds(strain_to_strength)
    data_info['strength_low_max'] = low_max
    data_info['strength_med_max'] = med_max
    force_print(f"[INFO] 動的閾値を設定: Low<{low_max}, Medium<{med_max}, High≥{med_max}")
    
    # 全データの統計を表示
    print("\n[INFO] === 全データベースの強度スコア分布 ===")
    print_strength_distribution(all_strains, strain_to_strength)
    
    # =================================================================
    # Step 2: キャッシュ確認・データ取得
    # =================================================================
    cached_data = None
    if not config.FORCE_REPROCESS:
        cached_data = load_cache(cache_path)
    
    if cached_data:
        train, valid, test = cached_data
        force_print("[INFO] Loaded split datasets from cache.")
        
        # キャッシュから読み込んだデータの強度スコアを全データベースのものに更新
        force_print("[INFO] Updating strength scores with global database values...")
        train = _update_strength_scores(train, strain_to_strength)
        valid = _update_strength_scores(valid, strain_to_strength)
        test = _update_strength_scores(test, strain_to_strength)
        
        # 長さ情報の取得
        if len(train) > 0:
            train_lens = [item[2] for item in train]
            data_info['train_min_len'], data_info['train_max_len'] = min(train_lens), max(train_lens)
        if len(valid) > 0:
            val_lens = [item[2] for item in valid]
            data_info['val_min_len'], data_info['val_max_len'] = min(val_lens), max(val_lens)
        if len(test) > 0:
            test_lens = [item[2] for item in test]
            data_info['test_min_len'], data_info['test_max_len'] = min(test_lens), max(test_lens)
        
        # 使用データの統計を表示
        train_strains = [item[4] for item in train]
        valid_strains = [item[4] for item in valid]
        test_strains = [item[4] for item in test]
        
        print("\n[INFO] === Train データの強度スコア分布 ===")
        print_strength_distribution(train_strains, strain_to_strength)
        print("\n[INFO] === Validation データの強度スコア分布 ===")
        print_strength_distribution(valid_strains, strain_to_strength)
        print("\n[INFO] === Test データの強度スコア分布 ===")
        print_strength_distribution(test_strains, strain_to_strength)
        
    else:
        # フルプロセス実行
        force_print("--- Processing Features (no cache) ---")
        
        # 参照データのロード
        df_freq = pd.read_csv(config.Freq_csv)
        df_dissimilarity = pd.read_csv(config.Disimilarity_csv)
        df_codon = pd.read_csv(config.Codon_csv)
        df_pam250 = pd.read_csv(config.PAM250_csv, index_col=0)
        
        # フィルタリング（重複除去）
        df_unique = filter_unique(all_names, all_lengths, all_paths, all_strains)
        
        # サンプリングモードに基づくフィルタリング
        sampling_mode = getattr(config, 'SAMPLING_MODE', 'proportional')
        force_print(f"[INFO] サンプリングモード: {sampling_mode}")
        
        if sampling_mode == 'proportional':
            # モードA: 比率サンプリング
            df_unique = sample_proportionally(df_unique, config.MAX_NUM)
        elif sampling_mode == 'fixed_per_strain':
            # モードB: 株数×サンプル数制限
            if config.MAX_STRAIN_NUM is not None:
                df_unique = sort_strain_by_num_and_filter_strain(df_unique, config.MAX_STRAIN_NUM)
            if config.MAX_NUM_PER_STRAIN is not None:
                df_unique = filter_num_per_strain(df_unique, config.MAX_NUM_PER_STRAIN)
        else:
            force_print(f"[WARNING] 不明なサンプリングモード: {sampling_mode}. 全件を使用します。")
        
        # 訓練/評価/テストへの分割
        train_df, valid_df, test_df = split_data_by_length(
            df_unique, config.TRAIN_MAX, config.VALID_NUM, config.VALID_RATIO, config.SEED
        )
        
        # 長さ情報の設定
        if not train_df.empty:
            data_info['train_min_len'] = int(train_df['original_len'].min())
            data_info['train_max_len'] = int(train_df['original_len'].max())
        if not valid_df.empty:
            data_info['val_min_len'] = int(valid_df['original_len'].min())
            data_info['val_max_len'] = int(valid_df['original_len'].max())
        if not test_df.empty:
            data_info['test_min_len'] = int(test_df['original_len'].min())
            data_info['test_max_len'] = int(test_df['original_len'].max())
        
        # 特徴量生成
        force_print("[INFO] Processing Train features...")
        _, _, train_feats = get_mutation_data(
            train_df['name'].tolist(), train_df['original_len'].tolist(), train_df['path'].tolist(),
            df_codon, df_freq, df_dissimilarity, df_pam250
        )
        force_print("[INFO] Processing Validation features...")
        _, _, valid_feats = get_mutation_data(
            valid_df['name'].tolist(), valid_df['original_len'].tolist(), valid_df['path'].tolist(),
            df_codon, df_freq, df_dissimilarity, df_pam250
        )
        force_print("[INFO] Processing Test features...")
        _, _, test_feats = get_mutation_data(
            test_df['name'].tolist(), test_df['original_len'].tolist(), test_df['path'].tolist(),
            df_codon, df_freq, df_dissimilarity, df_pam250
        )
        
        # 強度スコア別の分布を表示
        force_print("\n[INFO] === Train データの強度スコア分布 ===")
        print_strength_distribution(train_df['strain'].tolist(), strain_to_strength)
        force_print("\n[INFO] === Validation データの強度スコア分布 ===")
        print_strength_distribution(valid_df['strain'].tolist(), strain_to_strength)
        force_print("\n[INFO] === Test データの強度スコア分布 ===")
        print_strength_distribution(test_df['strain'].tolist(), strain_to_strength)
        
        # 入力(X)と正解(Y)への分割（全データベースのstrain_to_strengthを使用）
        force_print("[INFO] Separating X and Y...")
        train = separate_XY(train_feats, train_df['original_len'].tolist(), train_df['path'].tolist(), 
                            train_df['strain'].tolist(), strain_to_strength,
                            config.MAX_SEQ_LEN, config.TARGET_LEN)
        valid = separate_XY(valid_feats, valid_df['original_len'].tolist(), valid_df['path'].tolist(), 
                            valid_df['strain'].tolist(), strain_to_strength,
                            config.MAX_SEQ_LEN, config.TARGET_LEN)
        test = separate_XY(test_feats, test_df['original_len'].tolist(), test_df['path'].tolist(), 
                           test_df['strain'].tolist(), strain_to_strength,
                           config.MAX_SEQ_LEN, config.TARGET_LEN)
        
        # キャッシュ保存
        save_cache([train, valid, test], cache_path)
    
    return train, valid, test, data_info


def _update_strength_scores(data_list, strain_to_strength):
    """
    データリスト内の強度スコアを全データベースのものに更新
    
    Args:
        data_list: list of (x, y_targets, original_len, raw_path, strain, strength_score)
        strain_to_strength: 全データベースから計算した {strain: strength_score}
    
    Returns:
        更新されたdata_list
    """
    updated_list = []
    for item in data_list:
        x, y_targets, original_len, raw_path, strain, old_score = item
        new_score = strain_to_strength.get(strain, old_score)
        updated_list.append((x, y_targets, original_len, raw_path, strain, new_score))
    return updated_list