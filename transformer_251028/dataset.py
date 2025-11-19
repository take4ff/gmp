# --- dataset.py ---
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from . import config
from .utils import force_print
import os
import pandas as pd

def import_mutation_paths(base_dir, strain):
    # ホームディレクトリを展開
    base_dir = os.path.expanduser(base_dir)
    strain_dir = os.path.join(base_dir, strain)

    # strain直下のファイルパスを確認
    file_paths = []
    file_path = os.path.join(strain_dir, f"mutation_paths.tsv")
    #file_path = os.path.join(strain_dir, f"mutation_paths_{strain}.tsv")
    if os.path.exists(file_path):
        file_paths.append(file_path)
    
    # strain/numサブディレクトリを探索
    else:
        if os.path.exists(strain_dir) and os.path.isdir(strain_dir):
            num_dirs = [d for d in os.listdir(strain_dir) if d.isdigit()]
            num_dirs.sort(key=int)  # 数字順にソート

            for num in num_dirs:
                file_path = os.path.join(strain_dir, num, f"mutation_paths.tsv")
                #file_path = os.path.join(strain_dir, num, f"mutation_paths_{strain}.tsv")
                if os.path.exists(file_path):
                    file_paths.append(file_path)

    if not file_paths:
        raise FileNotFoundError(f"mutation_paths_{strain}.tsvが{strain_dir}内に見つかりませんでした。")

    return file_paths

def filter_co_occur(path,max_co_occur):
    compare = 0
    for i in range(len(path)):
        mutation = path[i].split(',')
        if compare < len(mutation):
            compare = len(mutation)
    if compare <= max_co_occur:
        return True
    else:
        return False
    
def filter_unique(names, lengths, mutation_paths, strains):
    df = pd.DataFrame({'name': names, 'original_len': lengths, 'path': mutation_paths, 'strain': strains})
    print(f"Initial samples loaded: {len(df)}")

    # リストをタプルに変換した列を一時的に作成
    df['path_tuple'] = df['path'].apply(tuple)
    
    # 'path_tuple' 列に基づいて重複を削除 (最初の出現を残す)
    df_unique = df.drop_duplicates(subset='path_tuple', keep='first')
    print(f"Samples after removing duplicate sequences: {len(df_unique)}")
    
    # 一時的な列を削除
    df_filtered = df_unique.drop(columns=['path_tuple'])

    return df_filtered

def sort_strain_by_num_and_filter_strain(df, max_strain_num):
    # 各strainの出現数をカウントし、上位max_num_strain個のstrainを取得
    force_print(f"[INFO] Filtering strains to top {max_strain_num} by occurrence count...")
    strain_counts = df['strain'].value_counts()
    top_strains = strain_counts.nlargest(max_strain_num).index.tolist()
    
    filtered_df = df[df['strain'].isin(top_strains)].reset_index(drop=True)
    force_print(f"[INFO] Samples after filtering by top {max_strain_num} strains: {len(filtered_df)}")
    return filtered_df

def filter_num_per_strain(df, max_num_per_strain):
    force_print(f"[INFO] Filtering samples to max {max_num_per_strain} per strain...")
    filtered_dfs = []
    for strain, group in df.groupby('strain'):
        if len(group) > max_num_per_strain:
            sampled_group = group.sample(n=max_num_per_strain, random_state=config.SEED)
            filtered_dfs.append(sampled_group)
        else:
            filtered_dfs.append(group)
    
    filtered_df = pd.concat(filtered_dfs).reset_index(drop=True)
    force_print(f"[INFO] Samples after filtering by strain (max {max_num_per_strain} per strain): {len(filtered_df)}")
    return filtered_df

def info_import(paths, lengths, over_num):
    force_print(f"[INFO] 全件読み込み完了(共起数フィルタ適用後): {len(paths)} サンプル")
    force_print(f"[INFO] 共起数が最大値を超えたサンプル数: {over_num} サンプル")

    force_print("\n[INFO] シーケンス長ごとのサンプル数:")
    if lengths:
        length_counts = pd.Series(lengths).value_counts().sort_index()
        with pd.option_context('display.max_rows', None):
            force_print(length_counts)
    else:
        force_print("データがありません。")

def import_strains(usher_dir='../usher_output/', max_num=None, max_cooccur=10):
    # --- データ読み込み・前処理 ---
    strains =  sorted([filename for filename in os.listdir(usher_dir) if not filename.startswith('.')])
    force_print(f"[INFO] データ読み込み開始: {len(strains)} strains found in {usher_dir}")

    # 全件データの読み込み
    names = []
    lengths = []
    paths = []
    paths_strain = []
    
    over_num = 0
    for strain in strains:
        file_paths = import_mutation_paths(usher_dir,strain)
        for file_path in file_paths:
            print(f"[INFO]import: {file_path}")
            f = open(file_path, 'r',encoding="utf-8_sig")
            datalist = f.readlines()
            f.close()

            data_num = len(datalist)

            for i in range(1,data_num):
                data = datalist[i].split('\t')
                path = data[2].rstrip().split('>')
                if filter_co_occur(path,max_co_occur=max_cooccur):
                    names.append(data[0])
                    lengths.append(int(data[1]))
                    paths.append(path)
                    paths_strain.append(strain)
                else:
                    over_num += 1
                
                if max_num is not None:
                    if max_num <= len(paths):
                        info_import(paths, lengths, over_num)
                        return names, lengths, paths, paths_strain

    info_import(paths, lengths, over_num)

    return names, lengths, paths, paths_strain    

DNA2Protein = {
        'TTT' : 'F', 'TCT' : 'S', 'TAT' : 'Y', 'TGT' : 'C',
        'TTC' : 'F', 'TCC' : 'S', 'TAC' : 'Y', 'TGC' : 'C',
        'TTA' : 'L', 'TCA' : 'S', 'TAA' : '*', 'TGA' : '*',
        'TTG' : 'L', 'TCG' : 'S', 'TAG' : '*', 'TGG' : 'W',

        'CTT' : 'L', 'CCT' : 'P', 'CAT' : 'H', 'CGT' : 'R',
        'CTC' : 'L', 'CCC' : 'P', 'CAC' : 'H', 'CGC' : 'R',
        'CTA' : 'L', 'CCA' : 'P', 'CAA' : 'Q', 'CGA' : 'R',
        'CTG' : 'L', 'CCG' : 'P', 'CAG' : 'Q', 'CGG' : 'R',

        'ATT' : 'I', 'ACT' : 'T', 'AAT' : 'N', 'AGT' : 'S',
        'ATC' : 'I', 'ACC' : 'T', 'AAC' : 'N', 'AGC' : 'S',
        'ATA' : 'I', 'ACA' : 'T', 'AAA' : 'K', 'AGA' : 'R',
        'ATG' : 'M', 'ACG' : 'T', 'AAG' : 'K', 'AGG' : 'R',

        'GTT' : 'V', 'GCT' : 'A', 'GAT' : 'D', 'GGT' : 'G',
        'GTC' : 'V', 'GCC' : 'A', 'GAC' : 'D', 'GGC' : 'G',
        'GTA' : 'V', 'GCA' : 'A', 'GAA' : 'E', 'GGA' : 'G',
        'GTG' : 'V', 'GCG' : 'A', 'GAG' : 'E', 'GGG' : 'G',
        'nnn':'n'
}

# >A,>T,>G,>C行の更新は無し
def Feature_from_csv(mutation, df_codon, df_freq, df_dissimilarity):

    base_pos = int(mutation[1:-1])
    bef = str(mutation[0])
    aft = str(mutation[-1])

    base = str(df_codon["base"][base_pos-1])
    protein = str(df_codon["protein"][base_pos-1])
    protein_pos = int(df_codon["protein_pos"][base_pos-1])
    codon = str(df_codon["codon"][base_pos-1])
    codon_pos = int(df_codon["codon_pos"][base_pos-1])
    #print(f"base_pos: {base_pos}, bef: {bef}, aft: {aft}, base: {base}, protein: {protein}, protein_pos: {protein_pos}, codon: {codon}, codon_pos: {codon_pos}")

    if bef == base and codon != "none":
        df_codon.at[base_pos-1, "base"] = aft
        if codon_pos == 1:
            new_codon = aft + codon[1:3]
            base1, base2 = 1, 2
        elif codon_pos == 2:
            new_codon = codon[0] + aft + codon[2]
            base1, base2 = -1, 1
        elif codon_pos == 3:
            new_codon = codon[0:2] + aft
            base1, base2 = -2, -1
        else:
            new_codon = codon  # 念のため
            base1 = base2 = 0

        df_codon.at[base_pos-1, "codon"] = new_codon
        if protein_pos == df_codon["protein_pos"][base_pos-1+base1]:
            df_codon.at[base_pos-1+base1, "codon"] = new_codon
        if protein_pos == df_codon["protein_pos"][base_pos-1+base2]:
            df_codon.at[base_pos-1+base2, "codon"] = new_codon
    else:
        new_codon = codon

    freq = df_freq[bef+'->'+aft][base_pos-1]

    bef_aa = DNA2Protein[codon]
    aft_aa = DNA2Protein[new_codon]
    dissimilarity = df_dissimilarity.query('wt==@bef_aa & mut==@aft_aa')

    hydro = dissimilarity.iloc[0]['eisenberg_weiss_diff']
    charge = dissimilarity.iloc[0]['charge_diff']
    size = dissimilarity.iloc[0]['size_diff']
    blsm = dissimilarity.iloc[0]['blsm62_diff']

    return codon, new_codon, codon_pos, protein, protein_pos, freq, hydro, charge, size, blsm

def Mutation_features(mutations, df_codon, df_freq, df_dissimilarity):
    features = []
    for mutation in mutations.split(','):
        base_pos = int(mutation[1:-1])
        bef = str(mutation[0])
        aft = str(mutation[-1])

        codon, new_codon, codon_pos, protein, protein_pos, freq, hydro, charge, size, blsm = Feature_from_csv(mutation, df_codon, df_freq, df_dissimilarity)
        if(codon=='none'):
            codon = 'nnn'
        if(new_codon=='none'):
            new_codon = 'nnn'

        bef_token = config.BASE_VOCABS[bef]
        aft_token = config.BASE_VOCABS[aft]
        aa_bef_token = config.AA_VOCABS[DNA2Protein[codon]]
        aa_aft_token = config.AA_VOCABS[DNA2Protein[new_codon]]
        protein_token = config.PROTEIN_VOCABS[protein]
        feature = ([bef_token, base_pos, aft_token, codon_pos, aa_bef_token, 
                    protein_pos, aa_aft_token, protein_token],
                   [freq, hydro, charge, size, blsm])

        features.append(feature)
    return features

def Feature_path(mutation_path, df_codon, df_freq, df_dissimilarity):
    df_codon1 = df_codon.copy()
    features_path = []
    for mutations in mutation_path:
        features = Mutation_features(mutations, df_codon1, df_freq, df_dissimilarity)
        features_path.append(features)

    return features_path

def get_mutation_data(names, lengths, paths, df_codon, df_freq, df_dissimilarity):

    features_paths = []
    force_print(f"[INFO] Generating mutation features for paths : Num {len(paths)}...")
    count = 0
    for path in paths:
        features_paths.append(Feature_path(path, df_codon, df_freq, df_dissimilarity))
        count += 1
        if count % config.PRINT_COUNT == 0:
            force_print(f"[INFO] Processed {count} paths...")

    return names, lengths, features_paths

def split_data_by_length(df, train_len, valid_num, valid_ratio, seed):
    """
    元のシーケンス長に基づいてデータを訓練、評価、テスト用に分割する。
    重複期間のデータは VALID_RATIO_FOR_EVAL に基づいて排他的に分割される。
    """
    # タイムステップの境界を定義
    overlap_start_len = train_len - valid_num + 1

    # 1. 訓練データ (重複期間より前)
    train_df = df[df['original_len'] < overlap_start_len].copy()

    # 2. テストデータ
    test_df = df[df['original_len'] > train_len].copy()

    # 3. 重複候補のデータを抽出
    overlap_df = df[
        (df['original_len'] >= overlap_start_len) &
        (df['original_len'] <= train_len)
    ].copy()

    # 4. 重複候補データをシャッフルし、評価用と訓練用に分割
    #    インデックスをリセットしてシャッフル可能にする
    overlap_df = overlap_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    # 評価データとして使用するサンプル数を計算
    n_valid_samples = int(len(overlap_df) * valid_ratio)
    
    # 分割
    valid_part_df = overlap_df.iloc[:n_valid_samples]
    train_part_df = overlap_df.iloc[n_valid_samples:]

    # 5. 評価データを確定
    valid_df = valid_part_df

    # 6. 訓練データに重複期間の残り部分を追加
    train_df = pd.concat([train_df, train_part_df], ignore_index=True)

    force_print(f"Train samples: {len(train_df)}")
    force_print(f"Validation samples: {len(valid_df)}")
    force_print(f"Test samples: {len(test_df)}")
    
    return train_df, valid_df, test_df

def separate_XY(feature_paths, original_lengths, raw_paths, max_x_len, ylen=1):
    """シーケンスを特徴量(X), ラベル(Y), 元の長さ(original_len)に分割する。"""
    x_y_len_list = []

    # 特徴量パスと元の長さを一緒にループ処理
    force_print(f"[INFO] Splitting data into X and Y with max_x_len={max_x_len} and ylen={ylen}...")
    for item, original_len, raw_path in zip(feature_paths, original_lengths, raw_paths):
        if len(item) > ylen:
            # X: 最後のylen個(例: 1個)を除くすべて
            x = item[:-ylen]
            if len(x) > max_x_len:
                x = x[-max_x_len:]  # 最新のmax_x_len個を使用

            # Y: 最後のylen個(例: 1個)のタイムステップ
            y_timestep = item[-ylen:][0]
            
            # Yを (領域, 位置) タプルのリストに変換
            y_targets = []
            for event in y_timestep:
                cat_features = event[0]
                region_id = cat_features[7]
                position_id = cat_features[1]
                y_targets.append((region_id, position_id))

            # (x, y, original_len) のタプルとしてリストに追加
            x_y_len_list.append((x, y_targets, original_len,raw_path))
    force_print(f"[INFO] Completed splitting data: {len(x_y_len_list)} samples prepared.")

    return x_y_len_list

'''
[
  # --- 1サンプル目 ---
  (
    # 1. sequence (時系列長 T=2)
    [
      # --- TimeStep 1 (共起数 C=2) ---
      [
        ([1045, 1, 3], [0.11, 0.22, 0.33]), # 変異1 (ID, 化学)
        ([20111, 4, 1], [0.44, 0.55, 0.66])  # 変異2 (ID, 化学)
      ],
      
      # --- TimeStep 2 (共起数 C=1) ---
      [
        ([500, 2, 2], [0.77, 0.88, 0.99])   # 変異1 (ID, 化学)
      ]
    ],
    
    # 2. targets (次の変異, 共起数2)
    [3, 8] # 領域3 と 領域8
  )
]
'''

# --- モックデータ（ダミーデータ）の生成 ---
def get_mock_data(num_samples, min_len=1, max_len=config.TRAIN_MAX):
    data = []
    for _ in range(num_samples):
        seq_len = np.random.randint(min_len, max_len + 1)
        sequence = []
        for _ in range(seq_len):
            co_occurrence = np.random.randint(1, 4)
            timestep_events = []
            for _ in range(co_occurrence):
                cat_features = [
                    np.random.randint(1, config.VOCAB_SIZE_POSITION),
                    np.random.randint(1, config.VOCAB_SIZE_BASE),
                    np.random.randint(1, config.VOCAB_SIZE_BASE),
                    np.random.randint(1, config.VOCAB_SIZE_AA), 
                    np.random.randint(1, config.VOCAB_SIZE_AA), 
                    np.random.randint(1, config.NUM_REGIONS),
                ]
                num_features = np.random.rand(config.NUM_CHEM_FEATURES).astype(np.float32)
                timestep_events.append((cat_features, num_features))
            sequence.append(timestep_events)

        # ターゲット（正解ラベル）を (領域ID, 位置ID) のタプルに変更
        num_targets = np.random.randint(1, 3)
        targets = []
        for _ in range(num_targets):
            # 領域ID (0 ~ NUM_REGIONS-1)
            target_region = np.random.randint(0, config.NUM_REGIONS)
            # 位置ID (1 ~ VOCAB_SIZE_POSITION-1)
            target_position = np.random.randint(1, config.VOCAB_SIZE_POSITION)
            targets.append((target_region, target_position))
        # 例: targets = [(2, 1000), (5, 25000)]

        data.append((sequence, targets))
    
    print("mock_data:",data[0])
    return data
# --- モックデータ生成 終了 ---

class ViralDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 1. データを取得 (original_lenも受け取る)
        sequence, targets, original_len, raw_path = self.data[idx]
        
        # パディング/切り捨て前の「元の時系列長」は、渡されたものをそのまま使う
        # original_len = len(sequence) + config.TARGET_LEN # この行を削除
        
        # もしシーケンス長がMAX_SEQ_LENを超える場合 (例: 31~40)
        # (注: ここのSEQ_LENはconfig.TRAIN_MAXとは異なる、モデル入力長の最大値)
        if len(sequence) > config.SEQ_LEN:
            # 最新のSEQ_LEN個のデータだけを使う (スライディングウィンドウ)
            start_index = len(sequence) - config.SEQ_LEN
            sequence_to_pad = sequence[start_index:]
            # パディングマスク計算用の長さも更新
            active_seq_len = config.SEQ_LEN
        else:
            sequence_to_pad = sequence
            active_seq_len = len(sequence)

        # 2. パディング処理 (カテゴリカル)
        padded_cat = np.zeros(
            (config.SEQ_LEN, config.MAX_CO_OCCUR, config.NUM_FEATURE_STRING), dtype=np.int64
        )
        # 2. パディング処理 (数値)
        padded_num = np.zeros(
            (config.SEQ_LEN, config.MAX_CO_OCCUR, config.NUM_CHEM_FEATURES),
            dtype=np.float32
        )
        
        # 3. パディングマスク
        padding_mask = np.ones(config.SEQ_LEN, dtype=bool)
        # 実際のデータがある部分はFalse
        padding_mask[:active_seq_len] = False 
        
        for t, timestep_events in enumerate(sequence_to_pad):
            # (t は 0 から active_seq_len-1 まで)
            co_len = len(timestep_events)
            for c, (cat, num) in enumerate(timestep_events):
                if c >= config.MAX_CO_OCCUR: break # 共起数も上限を超える場合は切り捨て
                padded_cat[t, c] = cat
                padded_num[t, c] = num
        
        return {
            "x_cat": torch.tensor(padded_cat, dtype=torch.long),
            "x_num": torch.tensor(padded_num, dtype=torch.float),
            "mask": torch.tensor(padding_mask, dtype=torch.bool),
            "y": targets, # yは (領域, 位置) タプルのリスト
            "original_len": original_len, # 元の長さを返す
            "raw_input": sequence, # 元の入力シーケンスも返す
            "raw_path": raw_path
        }

def create_dataloader(data, batch_size, shuffle=True):
    dataset = ViralDataset(data)
    
    def collate_fn(batch):
        batch_x_cat = torch.stack([item['x_cat'] for item in batch])
        batch_x_num = torch.stack([item['x_num'] for item in batch])
        batch_mask = torch.stack([item['mask'] for item in batch])

        # yは (領域, 位置) タプルのリスト のリスト
        batch_y = [item['y'] for item in batch]

        # 元の時系列長もバッチ化
        batch_lens = [item['original_len'] for item in batch]

        # 元の入力シーケンスもバッチ化
        batch_raw_inputs = [item['raw_input'] for item in batch]
        batch_raw_paths = [item['raw_path'] for item in batch]

        return (batch_x_cat, batch_x_num, batch_mask), batch_y, batch_lens, batch_raw_inputs, batch_raw_paths

    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        collate_fn=collate_fn
    )