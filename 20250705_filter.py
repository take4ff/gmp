#遅い
import module.input_mutation_path as imp
import os
import sys

# データセット設定
dataset_config = {
    'strains': ['B.1.1.7','P.1','BA.2','BA.1.1','BA.1','B.1.617.2','B.1.351','B.1.1.529'],
    'usher_dir': '../usher_output/',
}

data_config = {
    'nmax': 100000000,
    'nmax_per_strain': 1000000
}

def is_sublist(sublist, mainlist):
    n = len(mainlist)
    m = len(sublist)
    if m == 0:  # 空のリストは常に任意のリストの部分リストとみなす
        return True
    if m > n:   # 部分リストがメインリストより長い場合は、部分リストになりえない
        return False
    
    # メインリストの各開始位置から部分リストと一致するかチェック
    for i in range(n - m + 1):
        if mainlist[i : i + m] == sublist:
            return True
    return False

def get_non_encompassed_sequences(sequences):
    if not sequences:
        return []

    # set() を使うために、内部リストをタプルに変換してハッシュ可能にする
    # 重複を除去し、長さに応じてソート
    unique_sequences = [list(item) for item in dict.fromkeys(tuple(path) for path in sequences)]
    
    # 長さでソート（短いものが先に来るように）
    sorted_sequences = sorted(unique_sequences, key=len)

    non_encompassed = []

    for i, current_seq in enumerate(sorted_sequences):
        is_encompassed = False
        
        # 現在のシーケンスが他のどのシーケンスに内包されていないかを確認
        for j, other_seq in enumerate(sorted_sequences):
            if i == j:  # 自分自身との比較はスキップ
                continue

            # current_seq が other_seq に完全に内包されているか
            # is_sublist ヘルパー関数を使用
            if is_sublist(current_seq, other_seq):
                is_encompassed = True
                break # 一つでも内包されていれば、それは非内包ではないので次のシーケンスへ

        if not is_encompassed:
            non_encompassed.append(current_seq)

    return non_encompassed

def save_paths_to_file(paths, filename):
    """パスをファイルに保存"""
    with open(filename, 'w') as f:
        for path in paths:
            f.write(','.join(path) + '\n')
    print(f"保存完了: {filename}")

for strain in dataset_config['strains']:
    names, lengths, base_HGVS_paths = imp.input(
        [strain], 
        dataset_config['usher_dir'], 
        nmax=data_config['nmax'], 
        nmax_per_strain=data_config['nmax_per_strain']
    )
    set_list1 = [list(item) for item in dict.fromkeys(tuple(path) for path in base_HGVS_paths)]
    print(f"\n{strain} のデータ:")
    print(f"重複除去後の長さ: {len(set_list1)}")
    sys.stdout.flush()

    set_list2 = get_non_encompassed_sequences(base_HGVS_paths)
    print(f"最大パス保持後の長さ: {len(set_list2)}")

names, lengths, base_HGVS_paths = imp.input(
        dataset_config['strains'],
        dataset_config['usher_dir'], 
        nmax=data_config['nmax'], 
        nmax_per_strain=data_config['nmax_per_strain']
)
set_list1 = [list(item) for item in dict.fromkeys(tuple(path) for path in base_HGVS_paths)]
print(f"重複除去後の長さ: {len(set_list1)}")
sys.stdout.flush()

set_list2 = get_non_encompassed_sequences(base_HGVS_paths)
print(f"最大パス保持後の長さ: {len(set_list2)}")

file_path1 = os.path.join(dataset_config['usher_dir'], 'unique_mutaion_paths.tsv')
file_path2 = os.path.join(dataset_config['usher_dir'], 'non_encompassed_mutaion_paths.tsv')

save_paths_to_file(set_list1, file_path1)
save_paths_to_file(set_list2, file_path2)