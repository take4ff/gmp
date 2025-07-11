#処理がそもそも違う
import module.input_mutation_path as imp
import os
import sys
import time
from datetime import datetime

# 効率化された関数をここに挿入
# ... (上記の最適化関数をコピー)

def force_print(message):
    """タイムスタンプ付きで強制出力"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")
    sys.stdout.flush()

# データセット設定
dataset_config = {
    'strains': ['B.1.1.7','P.1','BA.2','BA.1.1','BA.1','B.1.617.2','B.1.351','B.1.1.529'],
    'usher_dir': '../usher_output/',
}

data_config = {
    'nmax': 100000000,
    'nmax_per_strain': 1000000
}

def is_sublist_optimized(sublist, mainlist):
    """最適化されたサブリストチェック"""
    if not sublist:
        return True
    if len(sublist) > len(mainlist):
        return False
    
    # KMP法の簡易版（効率的な文字列検索アルゴリズム）
    for i in range(len(mainlist) - len(sublist) + 1):
        if mainlist[i:i+len(sublist)] == sublist:
            return True
    return False

def get_non_encompassed_sequences_fast(sequences):
    """最高速版：ハッシュベース"""
    if not sequences:
        return []
    
    # 重複除去とタプル化
    unique_tuples = list(dict.fromkeys(tuple(path) for path in sequences))
    
    if len(unique_tuples) <= 1:
        return [list(t) for t in unique_tuples]
    
    # 長さでグループ化
    length_groups = {}
    for seq_tuple in unique_tuples:
        length = len(seq_tuple)
        if length not in length_groups:
            length_groups[length] = []
        length_groups[length].append(seq_tuple)
    
    # 長さ順にソート
    sorted_lengths = sorted(length_groups.keys())
    
    non_encompassed = []
    encompassed_sets = set()  # 内包されたシーケンスを記録
    
    for length in sorted_lengths:
        group = length_groups[length]
        
        for seq_tuple in group:
            if seq_tuple in encompassed_sets:
                continue
                
            is_encompassed = False
            seq_list = list(seq_tuple)
            
            # より短いシーケンスでの内包チェック
            for shorter_tuple in non_encompassed:
                if is_sublist_optimized(list(shorter_tuple), seq_list):
                    is_encompassed = True
                    break
            
            if not is_encompassed:
                non_encompassed.append(seq_tuple)
                
                # この新しいシーケンスが内包する可能性のある長いシーケンスをマーク
                for longer_length in range(length + 1, max(sorted_lengths) + 1):
                    if longer_length in length_groups:
                        for longer_tuple in length_groups[longer_length]:
                            if is_sublist_optimized(seq_list, list(longer_tuple)):
                                encompassed_sets.add(longer_tuple)
    
    return [list(t) for t in non_encompassed]

def save_paths_to_file(paths, filename):
    """パスをファイルに保存"""
    with open(filename, 'w') as f:
        for path in paths:
            f.write(','.join(path) + '\n')
    force_print(f"保存完了: {filename}")

# メイン処理（効率化版）
force_print("=== 効率化版フィルタリング開始 ===")

for strain in dataset_config['strains']:
    force_print(f"{strain} 処理開始...")
    
    start_time = time.time()
    
    names, lengths, base_HGVS_paths = imp.input(
        [strain], 
        dataset_config['usher_dir'], 
        nmax=data_config['nmax'], 
        nmax_per_strain=data_config['nmax_per_strain']
    )
    
    load_time = time.time() - start_time
    force_print(f"{strain}: データ読み込み完了 ({load_time:.1f}秒) - {len(base_HGVS_paths)}パス")
    
    # 重複除去
    dedup_start = time.time()
    set_list1 = [list(item) for item in dict.fromkeys(tuple(path) for path in base_HGVS_paths)]
    dedup_time = time.time() - dedup_start
    force_print(f"{strain}: 重複除去完了 ({dedup_time:.1f}秒) - {len(set_list1)}パス")
    
    # 効率化された非内包処理
    filter_start = time.time()
    set_list2 = get_non_encompassed_sequences_fast(base_HGVS_paths)  # 効率化版使用
    filter_time = time.time() - filter_start
    force_print(f"{strain}: 非内包処理完了 ({filter_time:.1f}秒) - {len(set_list2)}パス")
    
    total_time = time.time() - start_time
    force_print(f"{strain}: 処理完了 (総時間: {total_time:.1f}秒)")

# 全株統合処理
force_print("\n=== 全株統合処理 ===")
start_time = time.time()

names, lengths, base_HGVS_paths = imp.input(
    dataset_config['strains'],
    dataset_config['usher_dir'], 
    nmax=data_config['nmax'], 
    nmax_per_strain=data_config['nmax_per_strain']
)

force_print(f"統合データ読み込み完了: {len(base_HGVS_paths)}パス")

set_list1 = [list(item) for item in dict.fromkeys(tuple(path) for path in base_HGVS_paths)]
force_print(f"統合重複除去完了: {len(set_list1)}パス")

# 効率化版で処理
set_list2 = get_non_encompassed_sequences_fast(base_HGVS_paths)
force_print(f"統合非内包処理完了: {len(set_list2)}パス")

# ファイル保存
file_path1 = os.path.join(dataset_config['usher_dir'], 'unique_mutaion_paths.tsv')
file_path2 = os.path.join(dataset_config['usher_dir'], 'non_encompassed_mutaion_paths.tsv')

save_paths_to_file(set_list1, file_path1)
save_paths_to_file(set_list2, file_path2)

total_time = time.time() - start_time
force_print(f"=== 全処理完了 (総時間: {total_time/60:.1f}分) ===")