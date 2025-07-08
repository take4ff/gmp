import random
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

output_dir = os.path.join(dataset_config['usher_dir'], 'filter4')
os.makedirs(output_dir, exist_ok=True)

# ハッシュ衝突を減らすための複数の素数と基数
PRIMES = [1_000_000_007, 1_000_000_009, 1_000_000_021]
BASES = [31, 37, 41] 

def save_paths_to_file(paths, filename):
    """パスをファイルに保存"""
    force_print(f"ファイル保存開始: {filename}")
    with open(filename, 'w') as f:
        for path in paths:
            f.write(','.join(path) + '\n')
    force_print(f"ファイル保存完了: {filename}")

def compute_rolling_hashes_optimized(sequence_tuple, prime, base):
    """
    効率的なローリングハッシュ計算 - O(n²)
    全ての部分シーケンスのハッシュ値を効率的に計算
    """
    n = len(sequence_tuple)
    if n == 0:
        return set()
    
    all_hashes = set()
    
    # 各開始位置から
    for i in range(n):
        current_hash = 0
        power = 1
        
        # その位置から全ての長さの部分シーケンスを計算
        for j in range(i, n):
            if j == i:
                current_hash = hash(sequence_tuple[i]) % prime
            else:
                current_hash = (current_hash * base + hash(sequence_tuple[j])) % prime
            
            all_hashes.add(current_hash)
    
    return all_hashes

def is_subsequence_hash_based(short_seq, long_seq, prime, base):
    """
    ハッシュベースで短いシーケンスが長いシーケンスに含まれているかチェック
    O(n + m) - nは長いシーケンス長、mは短いシーケンス長
    """
    if len(short_seq) > len(long_seq):
        return False
    if len(short_seq) == 0:
        return True
    
    # 短いシーケンスのハッシュ値を計算
    target_hash = 0
    for item in short_seq:
        target_hash = (target_hash * base + hash(item)) % prime
    
    # 長いシーケンスでローリングハッシュを使用して検索
    if len(short_seq) == 1:
        # 長さ1の場合は単純チェック
        return any(hash(item) % prime == target_hash for item in long_seq)
    
    # ローリングハッシュで効率的に検索
    window_size = len(short_seq)
    if len(long_seq) < window_size:
        return False
    
    # 最初のウィンドウのハッシュ値を計算
    current_hash = 0
    power = 1
    
    for i in range(window_size):
        current_hash = (current_hash * base + hash(long_seq[i])) % prime
        if i < window_size - 1:
            power = (power * base) % prime
    
    if current_hash == target_hash:
        # ハッシュが一致した場合、実際の比較で確認
        if list(long_seq[:window_size]) == list(short_seq):
            return True
    
    # ローリングハッシュでスライドしながらチェック
    for i in range(window_size, len(long_seq)):
        # 古い文字を削除し、新しい文字を追加
        current_hash = (current_hash - hash(long_seq[i - window_size]) * power) % prime
        current_hash = (current_hash * base + hash(long_seq[i])) % prime
        current_hash = (current_hash + prime) % prime  # 負数を避ける
        
        if current_hash == target_hash:
            # ハッシュが一致した場合、実際の比較で確認
            if list(long_seq[i - window_size + 1:i + 1]) == list(short_seq):
                return True
    
    return False

def get_non_encompassed_sequences_2d_hashed_optimized(sequences_2d):
    """
    自動サイズ判定でチャンク化対応のハッシュベース非内包処理
    """
    if not sequences_2d:
        return []
    
    # データサイズに応じて処理方法を自動選択
    if len(sequences_2d) > 100000:
        force_print("大規模データのためチャンク処理を使用")
        return get_non_encompassed_sequences_2d_hashed_optimized_chunked(sequences_2d, chunk_size=50000)
    else:
        force_print("中規模データのため通常処理を使用")
        return get_non_encompassed_sequences_2d_hashed_optimized_standard(sequences_2d)

def get_non_encompassed_sequences_2d_hashed_optimized_standard(sequences_2d):
    """
    標準版のハッシュベース非内包処理（中規模データ用）
    """
    if not sequences_2d:
        return []

    force_print("重複除去開始...")
    # 重複除去
    unique_sequences_tuples = []
    seen_tuples = set()
    for seq in sequences_2d:
        seq_tuple = tuple(seq)
        if seq_tuple not in seen_tuples:
            seen_tuples.add(seq_tuple)
            unique_sequences_tuples.append(seq_tuple)
    
    force_print(f"重複除去完了: {len(sequences_2d)} -> {len(unique_sequences_tuples)}")
    
    if len(unique_sequences_tuples) <= 1:
        return [list(seq) for seq in unique_sequences_tuples]
    
    # 長さでソート（短いものが先）
    force_print("長さソート開始...")
    sorted_sequences = sorted(unique_sequences_tuples, key=len)
    force_print("長さソート完了")
    
    non_encompassed = []
    total = len(sorted_sequences)
    
    force_print("非内包処理開始...")
    
    for i, current_seq in enumerate(sorted_sequences):
        if (i + 1) % 5000 == 0:
            progress = ((i + 1) / total) * 100
            force_print(f"非内包処理進捗: {progress:.1f}% ({i + 1}/{total})")
        
        is_encompassed = False
        
        # 現在のシーケンスより長いシーケンスのみをチェック
        for j in range(i + 1, total):
            other_seq = sorted_sequences[j]
            
            # 長さが同じ場合はスキップ（既に重複除去済み）
            if len(other_seq) == len(current_seq):
                continue
            
            # 複数のハッシュ関数で確認（ハッシュ衝突対策）
            is_contained = True
            for prime, base in zip(PRIMES, BASES):
                if not is_subsequence_hash_based(current_seq, other_seq, prime, base):
                    is_contained = False
                    break
            
            if is_contained:
                # 最終確認：実際の部分シーケンスチェック
                if is_sublist_exact(list(current_seq), list(other_seq)):
                    is_encompassed = True
                    break
        
        if not is_encompassed:
            non_encompassed.append(list(current_seq))
    
    force_print(f"非内包処理完了: {len(non_encompassed)}個")
    
    # 結果を元の順序でソート
    original_order_map = {tuple(seq): i for i, seq in enumerate(sequences_2d)}
    return sorted(non_encompassed, key=lambda x: original_order_map.get(tuple(x), float('inf')))

def is_sublist_exact(sublist, mainlist):
    """正確な部分シーケンスチェック（最終確認用）"""
    if len(sublist) > len(mainlist):
        return False
    if len(sublist) == 0:
        return True
    
    for i in range(len(mainlist) - len(sublist) + 1):
        if mainlist[i:i+len(sublist)] == sublist:
            return True
    return False

def get_non_encompassed_sequences_2d_hashed_optimized_chunked(sequences_2d, chunk_size=50000):
    """
    チャンク化対応の最適化ハッシュベース非内包シーケンス抽出
    大規模データに対してメモリ効率を改善
    """
    if not sequences_2d:
        return []

    force_print("重複除去開始...")
    # 重複除去
    unique_sequences_tuples = []
    seen_tuples = set()
    for seq in sequences_2d:
        seq_tuple = tuple(seq)
        if seq_tuple not in seen_tuples:
            seen_tuples.add(seq_tuple)
            unique_sequences_tuples.append(seq_tuple)
    
    force_print(f"重複除去完了: {len(sequences_2d)} -> {len(unique_sequences_tuples)}")
    
    if len(unique_sequences_tuples) <= 1:
        return [list(seq) for seq in unique_sequences_tuples]
    
        # 小規模データの場合は通常処理
        if len(unique_sequences_tuples) <= chunk_size:
            force_print("小規模データのため通常処理を使用")
            return get_non_encompassed_sequences_2d_hashed_optimized_simple(unique_sequences_tuples)
    
    force_print(f"大規模データのためチャンク処理開始 (チャンクサイズ: {chunk_size})")
    
    # 長さでソート
    sorted_sequences = sorted(unique_sequences_tuples, key=len)
    
    # チャンクに分割
    chunks = []
    for i in range(0, len(sorted_sequences), chunk_size):
        chunks.append(sorted_sequences[i:i+chunk_size])
    
    force_print(f"チャンク数: {len(chunks)}")
    
    all_results = []
    
    # 各チャンクを処理
    for i, chunk in enumerate(chunks):
        force_print(f"チャンク {i+1}/{len(chunks)} 処理中...")
        chunk_result = get_non_encompassed_sequences_2d_hashed_optimized_simple(chunk)
        all_results.extend(chunk_result)
    
    # 最終的な統合処理
    force_print("チャンク結果の統合処理...")
    final_result = get_non_encompassed_sequences_2d_hashed_optimized_simple(all_results)
    
    return final_result

def get_non_encompassed_sequences_2d_hashed_optimized_simple(sequences_tuples):
    """
    シンプル版のハッシュベース非内包処理（チャンク内処理用）
    """
    if not sequences_tuples:
        return []
    
    if len(sequences_tuples) <= 1:
        return [list(seq) for seq in sequences_tuples]
    
    # 長さでソート（短いものが先）
    sorted_sequences = sorted(sequences_tuples, key=len)
    
    non_encompassed = []
    total = len(sorted_sequences)
    
    for i, current_seq in enumerate(sorted_sequences):
        is_encompassed = False
        
        # 現在のシーケンスより長いシーケンスのみをチェック
        for j in range(i + 1, total):
            other_seq = sorted_sequences[j]
            
            # 長さが同じ場合はスキップ
            if len(other_seq) == len(current_seq):
                continue
            
            # 複数のハッシュ関数で確認
            is_contained = True
            for prime, base in zip(PRIMES, BASES):
                if not is_subsequence_hash_based(current_seq, other_seq, prime, base):
                    is_contained = False
                    break
            
            if is_contained:
                # 最終確認：実際の部分シーケンスチェック
                if is_sublist_exact(list(current_seq), list(other_seq)):
                    is_encompassed = True
                    break
        
        if not is_encompassed:
            non_encompassed.append(list(current_seq))
    
    return non_encompassed

# メイン処理（最適化ハッシュベース版）
force_print("=== 最適化ハッシュベース版フィルタリング開始 ===")

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
    
    # 最適化ハッシュベースの非内包処理
    filter_start = time.time()
    set_list2 = get_non_encompassed_sequences_2d_hashed_optimized(base_HGVS_paths)
    filter_time = time.time() - filter_start
    force_print(f"{strain}: 最適化ハッシュベース非内包処理完了 ({filter_time:.1f}秒) - {len(set_list2)}パス")
    
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

# 統合非内包処理（最適化ハッシュベース版）
filter_start = time.time()
set_list2 = get_non_encompassed_sequences_2d_hashed_optimized(base_HGVS_paths)
filter_time = time.time() - filter_start
force_print(f"統合最適化ハッシュベース非内包処理完了 ({filter_time:.1f}秒): {len(set_list2)}パス")

# ファイル保存
file_path1 = os.path.join(output_dir, 'unique_mutaion_paths_optimized.tsv')
file_path2 = os.path.join(output_dir, 'non_encompassed_mutaion_paths_optimized.tsv')

# 重複除去結果も保存
unique_paths = [list(item) for item in dict.fromkeys(tuple(path) for path in base_HGVS_paths)]
save_paths_to_file(unique_paths, file_path1)
save_paths_to_file(set_list2, file_path2)

total_time = time.time() - start_time
force_print(f"=== 全処理完了 (総時間: {total_time/60:.1f}分) ===")
