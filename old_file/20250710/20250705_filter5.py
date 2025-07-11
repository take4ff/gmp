import module.input_mutation_path as imp
import os
import sys
import time
from datetime import datetime
from collections import defaultdict

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

output_dir = os.path.join(dataset_config['usher_dir'], 'filter5')
os.makedirs(output_dir, exist_ok=True)

def save_paths_to_file(paths, filename):
    """パスをファイルに保存"""
    force_print(f"ファイル保存開始: {filename}")
    with open(filename, 'w') as f:
        for path in paths:
            f.write(','.join(path) + '\n')
    force_print(f"ファイル保存完了: {filename}")

def is_sublist_ultrafast(sublist, mainlist):
    """超高速サブリストチェック（最適化済み）"""
    if not sublist:
        return True
    if len(sublist) > len(mainlist):
        return False
    
    # 長さ1の場合は高速チェック
    if len(sublist) == 1:
        return sublist[0] in mainlist
    
    # 最初と最後の要素の事前チェック
    first_item = sublist[0]
    last_item = sublist[-1]
    
    if first_item not in mainlist or last_item not in mainlist:
        return False
    
    # Boyer-Moore風の最適化：最初の要素の位置を効率的に検索
    sublist_len = len(sublist)
    mainlist_len = len(mainlist)
    
    for i in range(mainlist_len - sublist_len + 1):
        if mainlist[i] == first_item:
            # 最後の要素もチェック
            if mainlist[i + sublist_len - 1] == last_item:
                # 全体の比較
                if mainlist[i:i + sublist_len] == sublist:
                    return True
    
    return False

def get_non_encompassed_sequences_ultrafast(sequences):
    """超高速版非内包シーケンス抽出"""
    if not sequences:
        return []
    
    force_print("超高速版処理開始...")
    start_time = time.time()
    
    # Step 1: 重複除去
    force_print("重複除去開始...")
    unique_sequences = [list(item) for item in dict.fromkeys(tuple(path) for path in sequences)]
    force_print(f"重複除去完了: {len(sequences)} -> {len(unique_sequences)} ({time.time() - start_time:.1f}秒)")
    
    if len(unique_sequences) <= 1:
        return unique_sequences
    
    # Step 2: 長さによるグループ化
    force_print("長さ別グループ化開始...")
    length_groups = defaultdict(list)
    for seq in unique_sequences:
        length_groups[len(seq)].append(seq)
    
    sorted_lengths = sorted(length_groups.keys())
    force_print(f"長さ別グループ化完了: {len(sorted_lengths)}種類の長さ ({time.time() - start_time:.1f}秒)")
    
    # Step 3: 長さの昇順で処理（短いものから）
    non_encompassed = []
    total_processed = 0
    total_sequences = len(unique_sequences)
    
    force_print("非内包処理開始...")
    
    for current_length in sorted_lengths:
        current_group = length_groups[current_length]
        group_start_time = time.time()
        force_print(f"長さ {current_length}: {len(current_group)}個処理中...")
        
        for seq_idx, seq in enumerate(current_group):
            total_processed += 1
            
            # 進捗表示（1000個ごと）
            if total_processed % 1000 == 0:
                progress = (total_processed / total_sequences) * 100
                elapsed = time.time() - start_time
                force_print(f"全体進捗: {progress:.1f}% ({total_processed}/{total_sequences}) - {elapsed:.1f}秒経過")
            
            is_encompassed = False
            
            # セット事前チェック用の最適化
            seq_set = set(seq)
            
            # より長いグループとのみ比較
            for longer_length in sorted_lengths:
                if longer_length <= current_length:
                    continue
                
                longer_group = length_groups[longer_length]
                
                # 各長いシーケンスとの比較
                for longer_seq in longer_group:
                    # 高速事前チェック1: セット包含
                    longer_set = set(longer_seq)
                    if not seq_set.issubset(longer_set):
                        continue
                    
                    # 高速事前チェック2: 最初と最後の要素
                    if seq[0] not in longer_seq or seq[-1] not in longer_seq:
                        continue
                    
                    # 実際の順序チェック
                    if is_sublist_ultrafast(seq, longer_seq):
                        is_encompassed = True
                        break
                
                if is_encompassed:
                    break
            
            if not is_encompassed:
                non_encompassed.append(seq)
        
        group_time = time.time() - group_start_time
        force_print(f"長さ {current_length} 完了: {group_time:.1f}秒")
    
    total_time = time.time() - start_time
    force_print(f"超高速版処理完了: {len(non_encompassed)}個 (総時間: {total_time:.1f}秒)")
    
    return non_encompassed

def get_non_encompassed_sequences_smart_chunk(sequences, max_chunk_size=20000):
    """スマートチャンク版：小さなチャンクで超高速処理"""
    if not sequences:
        return []
    
    force_print("スマートチャンク版処理開始...")
    start_time = time.time()
    
    # 重複除去
    unique_sequences = [list(item) for item in dict.fromkeys(tuple(path) for path in sequences)]
    force_print(f"重複除去完了: {len(sequences)} -> {len(unique_sequences)}")
    
    # 小規模データは直接処理
    if len(unique_sequences) <= max_chunk_size:
        force_print("小規模データのため直接処理")
        return get_non_encompassed_sequences_ultrafast(unique_sequences)
    
    # 長さでソートしてから分割
    force_print("長さソート開始...")
    sorted_sequences = sorted(unique_sequences, key=len)
    force_print("長さソート完了")
    
    # チャンク分割
    chunks = []
    for i in range(0, len(sorted_sequences), max_chunk_size):
        chunks.append(sorted_sequences[i:i+max_chunk_size])
    
    force_print(f"スマートチャンク分割完了: {len(chunks)}個のチャンク (最大サイズ: {max_chunk_size})")
    
    # 各チャンクを処理
    results = []
    for i, chunk in enumerate(chunks):
        chunk_start_time = time.time()
        force_print(f"チャンク {i+1}/{len(chunks)} 処理開始 (サイズ: {len(chunk)})")
        
        chunk_result = get_non_encompassed_sequences_ultrafast(chunk)
        results.extend(chunk_result)
        
        chunk_time = time.time() - chunk_start_time
        force_print(f"チャンク {i+1} 完了: {len(chunk_result)}個 ({chunk_time:.1f}秒)")
    
    # 最終統合処理
    force_print("最終統合処理開始...")
    final_start_time = time.time()
    final_result = get_non_encompassed_sequences_ultrafast(results)
    final_time = time.time() - final_start_time
    
    total_time = time.time() - start_time
    force_print(f"スマートチャンク版完了: {len(final_result)}個 (統合: {final_time:.1f}秒, 総時間: {total_time:.1f}秒)")
    
    return final_result

# メイン処理（超高速版）
force_print("=== 超高速版フィルタリング開始 ===")

set_list2s = []

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
    
    # データサイズに応じて処理方法を選択
    filter_start = time.time()
    
    if len(base_HGVS_paths) > 100000:
        force_print(f"{strain}: 大規模データのためスマートチャンク処理を使用")
        set_list2 = get_non_encompassed_sequences_smart_chunk(base_HGVS_paths, max_chunk_size=15000)
    else:
        force_print(f"{strain}: 中規模データのため超高速処理を使用")
        set_list2 = get_non_encompassed_sequences_ultrafast(base_HGVS_paths)
    set_list2s.append(set_list2)
    
    filter_time = time.time() - filter_start
    force_print(f"{strain}: 超高速非内包処理完了 ({filter_time:.1f}秒) - {len(set_list2)}パス")
    
    total_time = time.time() - start_time
    force_print(f"{strain}: 処理完了 (総時間: {total_time:.1f}秒)")

# 全株統合処理
force_print(f"\n{'='*50}")
force_print("=== 全株統合処理 ===")
start_time = time.time()

# 各株の結果を統合
all_sequences = []
for strain_results in set_list2s:
    all_sequences.extend(strain_results)

force_print(f"統合データ準備完了: {len(all_sequences)}パス (各株の結果を統合)")

# 統合非内包処理（超高速版）
filter_start = time.time()

if len(all_sequences) > 200000:
    force_print("大規模統合データのためスマートチャンク処理を使用")
    set_list2 = get_non_encompassed_sequences_smart_chunk(all_sequences, max_chunk_size=20000)
else:
    force_print("中規模統合データのため超高速処理を使用")
    set_list2 = get_non_encompassed_sequences_ultrafast(all_sequences)

filter_time = time.time() - filter_start
force_print(f"統合超高速非内包処理完了 ({filter_time:.1f}秒): {len(set_list2)}パス")

# ファイル保存
file_path1 = os.path.join(output_dir, 'unique_mutaion_paths_ultrafast.tsv')
file_path2 = os.path.join(output_dir, 'non_encompassed_mutaion_paths_ultrafast.tsv')

# 重複除去結果も保存（統合データから）
unique_paths = [list(item) for item in dict.fromkeys(tuple(path) for path in all_sequences)]
save_paths_to_file(unique_paths, file_path1)
save_paths_to_file(set_list2, file_path2)

total_time = time.time() - start_time
force_print(f"=== 全処理完了 (総時間: {total_time/60:.1f}分) ===")
