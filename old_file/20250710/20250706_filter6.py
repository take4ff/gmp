import module.input_mutation_path as imp
import os
import sys
import time
from datetime import datetime
from collections import defaultdict
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import tempfile
import pickle

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

output_dir = os.path.join(dataset_config['usher_dir'], 'filter6')
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

def process_length_group_batch(args):
    """長さグループの並列処理用関数"""
    current_group, longer_groups_data, current_length, batch_id = args
    
    non_encompassed = []
    processed_count = 0
    
    for seq in current_group:
        processed_count += 1
        is_encompassed = False
        seq_set = set(seq)
        
        # より長いグループとのみ比較
        for longer_length, longer_group in longer_groups_data:
            if longer_length <= current_length:
                continue
            
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
    
    return non_encompassed, processed_count, batch_id

def get_non_encompassed_sequences_parallel(sequences, num_processes=None):
    """並列処理版非内包シーケンス抽出"""
    if not sequences:
        return []
    
    if num_processes is None:
        num_processes = min(mp.cpu_count() - 1, 6)  # 最大6プロセスに制限
    
    force_print(f"並列処理版開始 (プロセス数: {num_processes})...")
    start_time = time.time()
    
    # 既に重複除去済みかチェック
    input_tuples = [tuple(seq) for seq in sequences]
    unique_tuples = list(dict.fromkeys(input_tuples))
    
    if len(unique_tuples) == len(input_tuples):
        force_print("入力データは既に重複除去済み")
        unique_sequences = sequences
    else:
        force_print("重複除去開始...")
        unique_sequences = [list(item) for item in unique_tuples]
        force_print(f"重複除去完了: {len(sequences)} -> {len(unique_sequences)} ({time.time() - start_time:.1f}秒)")
    
    if len(unique_sequences) <= 1:
        return unique_sequences
    
    # Step 2: 長さによるグループ化
    force_print("長さ別グループ化開始...")
    length_groups = defaultdict(list)
    for seq in unique_sequences:
        length_groups[len(seq)].append(seq)
    
    sorted_lengths = sorted(length_groups.keys())
    force_print(f"長さ別グループ化完了: {len(sorted_lengths)}種類の長さ")
    
    # Step 3: 並列処理用のタスク作成
    tasks = []
    batch_size = max(1000, len(unique_sequences) // (num_processes * 2))  # バッチサイズを大きく
    
    for current_length in sorted_lengths:
        current_group = length_groups[current_length]
        
        # より長いグループのデータを準備
        longer_groups_data = []
        for longer_length in sorted_lengths:
            if longer_length > current_length:
                longer_groups_data.append((longer_length, length_groups[longer_length]))
        
        # 大きなグループはバッチに分割
        if len(current_group) > batch_size:
            for i in range(0, len(current_group), batch_size):
                batch = current_group[i:i+batch_size]
                batch_id = f"length_{current_length}_batch_{i//batch_size}"
                tasks.append((batch, longer_groups_data, current_length, batch_id))
        else:
            batch_id = f"length_{current_length}_single"
            tasks.append((current_group, longer_groups_data, current_length, batch_id))
    
    force_print(f"並列タスク作成完了: {len(tasks)}個のタスク")
    
    # Step 4: 並列実行
    non_encompassed = []
    total_processed = 0
    
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        # タスクを送信
        future_to_task = {executor.submit(process_length_group_batch, task): task for task in tasks}
        
        # 結果を収集
        completed = 0
        for future in as_completed(future_to_task):
            result, processed_count, batch_id = future.result()
            non_encompassed.extend(result)
            total_processed += processed_count
            completed += 1
            
            progress = (completed / len(tasks)) * 100
            elapsed = time.time() - start_time
            force_print(f"並列処理進捗: {progress:.1f}% ({completed}/{len(tasks)}) - {elapsed:.1f}秒経過 - {batch_id}")
    
    total_time = time.time() - start_time
    force_print(f"並列処理版完了: {len(non_encompassed)}個 (総時間: {total_time:.1f}秒)")
    
    return non_encompassed

def get_non_encompassed_sequences_stage_divide(sequences, max_chunk_size=30000):
    """段階分割統合版：メモリ効率を重視した分割処理"""
    if not sequences:
        return []
    
    force_print("段階分割統合版処理開始...")
    start_time = time.time()
    
    # 既に重複除去済みかチェック
    input_tuples = [tuple(seq) for seq in sequences]
    unique_tuples = list(dict.fromkeys(input_tuples))
    
    if len(unique_tuples) == len(input_tuples):
        force_print("入力データは既に重複除去済み")
        unique_sequences = sequences
    else:
        force_print("重複除去開始...")
        unique_sequences = [list(item) for item in unique_tuples]
        force_print(f"重複除去完了: {len(sequences)} -> {len(unique_sequences)}")
    
    # 小規模データは並列処理で直接処理
    if len(unique_sequences) <= max_chunk_size:
        force_print("中規模データのため並列処理で直接処理")
        return get_non_encompassed_sequences_parallel(unique_sequences)
    
    # 段階1: 長さでソートしてから分割
    force_print("長さソート開始...")
    sorted_sequences = sorted(unique_sequences, key=len)
    force_print("長さソート完了")
    
    # 段階2: 小チャンクで前処理（各チャンク内で非内包処理）
    force_print("段階2: 小チャンク前処理開始...")
    small_chunk_size = max_chunk_size // 3
    stage1_results = []
    
    total_chunks = (len(sorted_sequences) + small_chunk_size - 1) // small_chunk_size
    
    for i in range(0, len(sorted_sequences), small_chunk_size):
        chunk = sorted_sequences[i:i+small_chunk_size]
        chunk_num = i // small_chunk_size + 1
        
        force_print(f"前処理チャンク {chunk_num}/{total_chunks} (サイズ: {len(chunk)})")
        
        # 小チャンクは並列処理
        chunk_result = get_non_encompassed_sequences_parallel(chunk)
        stage1_results.extend(chunk_result)
        
        force_print(f"前処理チャンク {chunk_num} 完了: {len(chunk_result)}個")
    
    stage1_time = time.time() - start_time
    force_print(f"段階2完了: {len(stage1_results)}個 ({stage1_time:.1f}秒)")
    
    # 段階3: 中間統合（前処理結果をさらに統合）
    if len(stage1_results) > max_chunk_size:
        force_print("段階3: 中間統合処理開始...")
        
        intermediate_results = []
        for i in range(0, len(stage1_results), max_chunk_size):
            chunk = stage1_results[i:i+max_chunk_size]
            chunk_num = i // max_chunk_size + 1
            total_inter_chunks = (len(stage1_results) + max_chunk_size - 1) // max_chunk_size
            
            force_print(f"中間統合チャンク {chunk_num}/{total_inter_chunks} (サイズ: {len(chunk)})")
            
            chunk_result = get_non_encompassed_sequences_parallel(chunk)
            intermediate_results.extend(chunk_result)
            
            force_print(f"中間統合チャンク {chunk_num} 完了: {len(chunk_result)}個")
        
        stage1_results = intermediate_results
        stage2_time = time.time() - start_time
        force_print(f"段階3完了: {len(stage1_results)}個 ({stage2_time - stage1_time:.1f}秒)")
    
    # 段階4: 最終統合
    force_print("段階4: 最終統合処理開始...")
    final_start_time = time.time()
    final_result = get_non_encompassed_sequences_parallel(stage1_results)
    final_time = time.time() - final_start_time
    
    total_time = time.time() - start_time
    force_print(f"段階分割統合版完了: {len(final_result)}個 (最終統合: {final_time:.1f}秒, 総時間: {total_time:.1f}秒)")
    
    return final_result

def get_non_encompassed_sequences_adaptive(sequences):
    """適応的処理選択版（修正版）"""
    if not sequences:
        return []
    
    data_size = len(sequences)
    force_print(f"データサイズ: {data_size}")
    
    # 閾値を実データサイズに合わせて調整
    if data_size <= 100000:
        force_print("小規模データ: 並列処理版を使用")
        return get_non_encompassed_sequences_parallel(sequences)
    elif data_size <= 500000:
        force_print("中規模データ: 段階分割統合版を使用")
        return get_non_encompassed_sequences_stage_divide(sequences, max_chunk_size=80000)
    else:
        force_print("大規模データ: 段階分割統合版（小チャンク）を使用")
        return get_non_encompassed_sequences_stage_divide(sequences, max_chunk_size=50000)

# メイン処理（超々高速版）
force_print("=== 超々高速版フィルタリング開始 (filter6) ===")

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

    # 重複除去
    force_print("重複除去開始...")
    unique_paths = [list(item) for item in dict.fromkeys(tuple(path) for path in base_HGVS_paths)]
    force_print(f"重複除去完了: {len(base_HGVS_paths)} -> {len(unique_paths)} ({time.time() - load_time:.1f}秒)")  
    
    # 適応的処理選択
    filter_start = time.time()
    set_list2 = get_non_encompassed_sequences_adaptive(unique_paths)
    set_list2s.append(set_list2)
    
    filter_time = time.time() - filter_start
    force_print(f"{strain}: 適応的非内包処理完了 ({filter_time:.1f}秒) - {len(set_list2)}パス")
    
    total_time = time.time() - start_time
    force_print(f"{strain}: 処理完了 (総時間: {total_time:.1f}秒)")

    # ファイル保存
    file_path1 = os.path.join(output_dir, f'unique_mutaion_paths_ultrafast_{strain}.tsv')
    file_path2 = os.path.join(output_dir, f'non_encompassed_mutaion_paths_ultrafast_{strain}.tsv')

    # 重複除去結果も保存（統合データから）
    save_paths_to_file(unique_paths, file_path1)
    save_paths_to_file(set_list2, file_path2)

# 全株統合処理
force_print(f"\n{'='*50}")
force_print("=== 全株統合処理 ===")
start_time = time.time()

# 各株の結果を統合
all_sequences = []
for strain_results in set_list2s:
    all_sequences.extend(strain_results)

force_print(f"統合データ準備完了: {len(all_sequences)}パス (各株の結果を統合)")

# 統合非内包処理（適応的選択）
filter_start = time.time()
set_list2 = get_non_encompassed_sequences_adaptive(all_sequences)
filter_time = time.time() - filter_start

force_print(f"統合適応的非内包処理完了 ({filter_time:.1f}秒): {len(set_list2)}パス")

# ファイル保存
file_path1 = os.path.join(output_dir, 'unique_mutaion_paths_ultrafast.tsv')
file_path2 = os.path.join(output_dir, 'non_encompassed_mutaion_paths_ultrafast.tsv')

# 重複除去結果も保存（統合データから）
unique_paths = [list(item) for item in dict.fromkeys(tuple(path) for path in all_sequences)]
save_paths_to_file(unique_paths, file_path1)
save_paths_to_file(set_list2, file_path2)

total_time = time.time() - start_time
force_print(f"=== 全処理完了 (総時間: {total_time/60:.1f}分) ===")
force_print(f"filter6による大幅高速化完了！")
