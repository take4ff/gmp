import module.input_mutation_path as imp
import os
import sys
import time
from datetime import datetime

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

output_dir = os.path.join(dataset_config['usher_dir'], 'filter3')
os.makedirs(output_dir, exist_ok=True)

def is_sublist_optimized(sublist, mainlist):
    """効率化されたis_sublist関数"""
    m = len(sublist)
    n = len(mainlist)
    
    if m == 0:
        return True
    if m > n:
        return False
    
    # 最適化1: 最初と最後の要素で事前チェック
    if sublist[0] not in mainlist or sublist[-1] not in mainlist:
        return False
    
    # 最適化2: スライドウィンドウ方式
    for i in range(n - m + 1):
        if mainlist[i] == sublist[0]:  # 最初の要素が一致した場合のみチェック
            if mainlist[i:i+m] == sublist:
                return True
    return False

def get_non_encompassed_sequences_improved(sequences):
    """無印版アルゴリズムの効率化版"""
    if not sequences:
        return []

    force_print("重複除去開始...")
    # 重複除去
    unique_sequences = [list(item) for item in dict.fromkeys(tuple(path) for path in sequences)]
    force_print(f"重複除去完了: {len(sequences)} -> {len(unique_sequences)}")
    
    if len(unique_sequences) <= 1:
        return unique_sequences
    
    # 長さでソート（短いものが先に来るように）
    force_print("長さソート開始...")
    sorted_sequences = sorted(unique_sequences, key=len)
    force_print("長さソート完了")

    non_encompassed = []
    total = len(sorted_sequences)
    
    # 最適化1: 長さベースの事前フィルタリング
    length_groups = {}
    for seq in sorted_sequences:
        length = len(seq)
        if length not in length_groups:
            length_groups[length] = []
        length_groups[length].append(seq)
    
    force_print("非内包処理開始...")
    processed = 0
    
    for current_seq in sorted_sequences:
        processed += 1
        if processed % 5000 == 0:  # 5000個ごとに進捗表示
            progress = (processed / total) * 100
            force_print(f"非内包処理進捗: {progress:.1f}% ({processed}/{total})")
        
        is_encompassed = False
        current_len = len(current_seq)
        
        # 最適化2: 現在のシーケンスより長いもののみチェック
        for check_len in range(current_len + 1, max(length_groups.keys()) + 1):
            if check_len in length_groups:
                for other_seq in length_groups[check_len]:
                    if is_sublist_optimized(current_seq, other_seq):
                        is_encompassed = True
                        break
                if is_encompassed:
                    break

        if not is_encompassed:
            non_encompassed.append(current_seq)

    force_print(f"非内包処理完了: {len(non_encompassed)}個")
    return non_encompassed

def get_non_encompassed_sequences_chunk(sequences, chunk_size=50000):
    """チャンク処理による効率化"""
    if not sequences:
        return []
    
    force_print("重複除去開始...")
    unique_sequences = [list(item) for item in dict.fromkeys(tuple(path) for path in sequences)]
    force_print(f"重複除去完了: {len(sequences)} -> {len(unique_sequences)}")
    
    if len(unique_sequences) <= chunk_size:
        return get_non_encompassed_sequences_improved(unique_sequences)
    
    force_print(f"大規模データのためチャンク処理開始 (チャンクサイズ: {chunk_size})...")
    
    # 長さでソート
    sorted_sequences = sorted(unique_sequences, key=len)
    
    # チャンクに分割
    chunks = []
    for i in range(0, len(sorted_sequences), chunk_size):
        chunks.append(sorted_sequences[i:i+chunk_size])
    
    force_print(f"チャンク数: {len(chunks)}")
    
    all_results = []
    
    # 各チャンクを処理
    for i, chunk in enumerate(chunks):
        force_print(f"チャンク {i+1}/{len(chunks)} 処理中...")
        chunk_result = get_non_encompassed_sequences_improved(chunk)
        all_results.extend(chunk_result)
    
    # 最終的な統合処理
    force_print("チャンク結果の統合処理...")
    final_result = get_non_encompassed_sequences_improved(all_results)
    
    return final_result

def save_paths_to_file(paths, filename):
    """パスをファイルに保存"""
    force_print(f"ファイル保存開始: {filename}")
    with open(filename, 'w') as f:
        for path in paths:
            f.write(','.join(path) + '\n')
    force_print(f"ファイル保存完了: {filename}")

# メイン処理（効率化された無印版）
force_print("=== 効率化無印版フィルタリング開始 ===")

for strain in dataset_config['strains']:
    force_print(f"\n{strain} 処理開始...")
    
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
    
    # 効率化された無印版の非内包処理
    filter_start = time.time()
    
    # データサイズに応じて処理方法を選択
    if len(set_list1) > 50000:
        force_print(f"{strain}: 大規模データのためチャンク処理を使用")
        set_list2 = get_non_encompassed_sequences_chunk(base_HGVS_paths, chunk_size=25000)
    else:
        set_list2 = get_non_encompassed_sequences_improved(base_HGVS_paths)
    
    filter_time = time.time() - filter_start
    force_print(f"{strain}: 非内包処理完了 ({filter_time:.1f}秒) - {len(set_list2)}パス")
    
    total_time = time.time() - start_time
    force_print(f"{strain}: 処理完了 (総時間: {total_time:.1f}秒)")

# 全株統合処理
force_print(f"\n{'='*50}")
force_print("=== 全株統合処理 ===")
start_time = time.time()

names, lengths, base_HGVS_paths = imp.input(
    dataset_config['strains'],
    dataset_config['usher_dir'], 
    nmax=data_config['nmax'], 
    nmax_per_strain=data_config['nmax_per_strain']
)

force_print(f"統合データ読み込み完了: {len(base_HGVS_paths)}パス")

# 統合重複除去
dedup_start = time.time()
set_list1 = [list(item) for item in dict.fromkeys(tuple(path) for path in base_HGVS_paths)]
dedup_time = time.time() - dedup_start
force_print(f"統合重複除去完了 ({dedup_time:.1f}秒): {len(set_list1)}パス")

# 統合非内包処理（効率化された無印版）
filter_start = time.time()

# 大規模データの場合はチャンク処理
if len(set_list1) > 100000:
    force_print("大規模統合データのためチャンク処理を使用")
    set_list2 = get_non_encompassed_sequences_chunk(base_HGVS_paths, chunk_size=50000)
else:
    set_list2 = get_non_encompassed_sequences_improved(base_HGVS_paths)

filter_time = time.time() - filter_start
force_print(f"統合非内包処理完了 ({filter_time:.1f}秒): {len(set_list2)}パス")

# ファイル保存
file_path1 = os.path.join(output_dir, 'unique_mutaion_paths.tsv')
file_path2 = os.path.join(output_dir, 'non_encompassed_mutaion_paths.tsv')

save_paths_to_file(set_list1, file_path1)
save_paths_to_file(set_list2, file_path2)

total_time = time.time() - start_time
force_print(f"=== 全処理完了 (総時間: {total_time/60:.1f}分) ===")