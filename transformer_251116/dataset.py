import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import os
import re
from . import config
from .utils import force_print

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

# --- 1. 生データ読み込み ---

def import_mutation_paths(base_dir, strain):
    """
    指定された株（strain）のディレクトリ以下を再帰的に探索し、
    全ての 'mutation_paths.tsv' ファイルへのパスのリストを返す。
    """
    base_dir = os.path.expanduser(base_dir)
    strain_dir = os.path.join(base_dir, strain)
    file_paths = []
    for root, dirs, files in os.walk(strain_dir):
        if "mutation_paths.tsv" in files:
            file_paths.append(os.path.join(root, "mutation_paths.tsv"))
    return file_paths

def import_strains(usher_dir, max_strain_num=None):
    """
    指定されたディレクトリから株のデータを読み込み、DataFrameとして返す。
    ['name', 'original_len', 'path', 'strain'] のカラムを持つ。
    """
    force_print(f"[INFO] Data import started from {usher_dir}")
    
    # ディレクトリのみを株（strain）としてリストアップする
    try:
        all_entries = sorted(os.listdir(usher_dir))
        strain_dirs = [d for d in all_entries if os.path.isdir(os.path.join(usher_dir, d)) and not d.startswith('.')]
    except FileNotFoundError:
        raise FileNotFoundError(f"The specified directory does not exist: {usher_dir}")

    if max_strain_num:
        strain_dirs = strain_dirs[:max_strain_num]
    
    force_print(f"[INFO] Found {len(strain_dirs)} strains.")

    all_data = []
    for strain in strain_dirs:
        file_paths = import_mutation_paths(usher_dir, strain)
        force_print(f"[INFO]  Loading {len(file_paths)} files for strain: {strain}")
        for file_path in file_paths:
            df = pd.read_csv(file_path, sep='\t', header=0, names=['name', 'original_len', 'path'])
            df['strain'] = strain
            all_data.append(df)
    
    if not all_data:
        raise FileNotFoundError(f"No mutation data found in any strain subdirectories of {usher_dir}")

    df_all = pd.concat(all_data, ignore_index=True)
    df_all['path'] = df_all['path'].str.split('>')
    
    force_print(f"[INFO] Total samples loaded: {len(df_all)}")
    return df_all

# --- 2. データフィルタリング・分割 ---

def filter_by_max_co_occurrence(df, max_co_occur):
    """
    パス内の各タイムステップで、共起変異数が最大値を超えるサンプルを除外する。
    """
    if max_co_occur is None:
        return df

    force_print(f"[INFO] Filtering samples by max co-occurrence ({max_co_occur})...")
    initial_count = len(df)

    def check_co_occurrence(path):
        for mutations in path:
            if len(mutations.split(',')) > max_co_occur:
                return False
        return True

    mask = df['path'].apply(check_co_occurrence)
    filtered_df = df[mask]
    
    removed_count = initial_count - len(filtered_df)
    force_print(f"[INFO] Samples after co-occurrence filtering: {len(filtered_df)} ({removed_count} removed)")
    return filtered_df

def sample_by_total_count(df, max_samples, seed):
    """
    データセット全体の株の分布を維持しながら、総サンプル数を制限する（層化サンプリング）。
    """
    if max_samples is None:
        return df

    total_count = len(df)
    if total_count <= max_samples:
        return df

    force_print(f"[INFO] Stratified sampling to {max_samples} samples...")

    sampled_df = df.groupby('strain', group_keys=False).apply(
        lambda x: x.sample(frac=max_samples/total_count, random_state=seed)
    )
    
    force_print(f"[INFO] Samples after stratified sampling: {len(sampled_df)}")
    return sampled_df

def sort_strain_by_num_and_filter_strain(df, max_top_strains):
    """
    株の出現頻度でソートし、上位N株のみにデータをフィルタリングする。
    """
    if max_top_strains is None:
        return df
    
    force_print(f"[INFO] Filtering strains to top {max_top_strains} by occurrence count...")
    strain_counts = df['strain'].value_counts()
    top_strains = strain_counts.nlargest(max_top_strains).index
    filtered_df = df[df['strain'].isin(top_strains)].copy()
    force_print(f"[INFO] Samples after filtering by top strains: {len(filtered_df)}")
    return filtered_df

def filter_num_per_strain(df, max_num_per_strain, seed):
    """
    各株から最大N個のサンプルをランダムに抽出する。
    """
    if max_num_per_strain is None:
        return df
        
    force_print(f"[INFO] Filtering samples to max {max_num_per_strain} per strain...")
    
    filtered_df = df.groupby('strain', group_keys=False).apply(
        lambda x: x.sample(n=min(len(x), max_num_per_strain), random_state=seed)
    )
    
    force_print(f"[INFO] Samples after filtering per strain: {len(filtered_df)}")
    return filtered_df.reset_index(drop=True)

def filter_unique_paths(df):
    """変異パスの重複を削除する"""
    df = df.copy()
    initial_count = len(df)
    df['path_tuple'] = df['path'].apply(tuple)
    df = df.drop_duplicates(subset='path_tuple', keep='first').drop(columns=['path_tuple'])
    force_print(f"[INFO] Samples after removing duplicate paths: {len(df)} ({initial_count - len(df)} removed)")
    return df

def split_features_by_length(features_df, train_max_len, valid_num, valid_ratio, seed):
    """
    特徴量生成後のDataFrameを元のシーケンス長に基づいて訓練、検証、テスト用に分割する。
    効率化版: 特徴量生成後に分割を実行。
    """
    force_print("[INFO] Splitting features by original sequence length...")
    
    overlap_start_len = train_max_len - valid_num + 1

    train_mask = features_df['original_len'] < overlap_start_len
    test_mask = features_df['original_len'] > train_max_len
    overlap_mask = ((features_df['original_len'] >= overlap_start_len) & 
                   (features_df['original_len'] <= train_max_len))

    train_features = features_df[train_mask]
    test_features = features_df[test_mask]
    overlap_features = features_df[overlap_mask].sample(frac=1, random_state=seed)

    n_valid_samples = int(len(overlap_features) * valid_ratio)
    valid_features = overlap_features.iloc[:n_valid_samples]
    train_part_features = overlap_features.iloc[n_valid_samples:]

    train_features = pd.concat([train_features, train_part_features], ignore_index=True)

    # DataFrameから辞書形式に変換
    def df_to_dict(df):
        return {
            'input_cat_seq': df['input_cat_seq'].tolist(),
            'input_num_seq': df['input_num_seq'].tolist(), 
            'target_protein': df['target_protein'].tolist(),
            'target_pos': df['target_pos'].tolist(),
            'original_len': df['original_len'].tolist()
        }

    train_data = df_to_dict(train_features)
    valid_data = df_to_dict(valid_features)
    test_data = df_to_dict(test_features)

    force_print(f"Train samples: {len(train_data['target_protein'])}")
    force_print(f"Validation samples: {len(valid_data['target_protein'])}")
    force_print(f"Test samples: {len(test_data['target_protein'])}")
    
    return train_data, valid_data, test_data

# --- 3. 特徴量生成 (効率化版) ---

def parse_mutation(mutations):
    """
    'A123G' のような単一または複数の変異文字列をパースする。
    """
    mutation_list = mutations.split(',')
    parsed = []
    for mut in mutation_list:
        match = re.match(r'^([ACGTN])(\d+)([ACGTN])$', mut)
        if match:
            parsed.append((match.group(1), int(match.group(2)), match.group(3)))
    return parsed

def get_parent_path(path_str):
    """ 'A.B.C' -> 'A.B' """
    parts = path_str.split('.')
    if len(parts) <= 1:
        return None
    return '.'.join(parts[:-1])

def get_mutations_from_path_str(path_str):
    """ 'A.B.C' -> ['A', 'B', 'C'] """
    return path_str.split('.')

def update_codon_table(df_codon_current, mutation):
    """
    単一の変異情報を用いてdf_codonを更新する。
    """
    bef, pos, aft = mutation
    
    target_indices = df_codon_current[df_codon_current['base_pos'] == pos].index
    if target_indices.empty:
        return df_codon_current

    for idx in target_indices:
        original_codon = df_codon_current.at[idx, 'codon']
        codon_pos = df_codon_current.at[idx, 'codon_pos']
        
        if pd.isna(original_codon) or original_codon == 'none' or len(original_codon) != 3:
            continue

        new_codon_list = list(original_codon)
        if 1 <= codon_pos <= 3:
            new_codon_list[codon_pos - 1] = aft
        new_codon = "".join(new_codon_list)

        new_aa = config.DNA2PROTEIN.get(new_codon, '*')

        codon_start_pos = df_codon_current.at[idx, 'codon_start']
        codon_indices = df_codon_current[df_codon_current['codon_start'] == codon_start_pos].index
        
        df_codon_current.loc[codon_indices, 'codon'] = new_codon
        df_codon_current.loc[codon_indices, 'amino_acid'] = new_aa
        
    return df_codon_current

def get_new_codon(codon, codon_pos, aft_base):
    """補助関数: 変異後のコドンを計算"""
    if pd.isna(codon) or codon == 'none': return 'nnn'
    if codon_pos == 1: return aft_base + codon[1:]
    if codon_pos == 2: return codon[0] + aft_base + codon[2]
    if codon_pos == 3: return codon[:2] + aft_base
    return codon

def process_feature_batch_optimized(df_batch, df_codon, df_freq, df_dissimilarity):
    """
    効率化された特徴量生成関数。
    親子関係を活用した最適化を含む。
    """
    # 1. 入力(X)と正解(Y)にパスを分離
    df_batch['x_path_str'] = df_batch['path'].apply(lambda p: '.'.join(p[:-1]))
    df_batch['y_path_str'] = df_batch['path'].apply(lambda p: p[-1])
    
    # 2. 正解(Y)を共起単位で分割 (explode)
    df_batch['y_mutations'] = df_batch['y_path_str'].str.split(',')
    df_exploded = df_batch.explode('y_mutations')

    # 3. パースして (bef, pos, aft) を抽出
    df_exploded['y_parsed'] = df_exploded['y_mutations'].apply(lambda m: re.match(r'^([ACGTN])(\d+)([ACGTN])$', m))
    df_exploded = df_exploded.dropna(subset=['y_parsed'])
    df_exploded['y_bef'] = df_exploded['y_parsed'].apply(lambda m: m.group(1))
    df_exploded['y_pos'] = df_exploded['y_parsed'].apply(lambda m: int(m.group(2)))
    df_exploded['y_aft'] = df_exploded['y_parsed'].apply(lambda m: m.group(3))

    # 4. ユニークなx_pathを取得し、階層的にソート
    unique_x_paths = sorted(df_exploded['x_path_str'].unique())

    # 5. キャッシュと結果リストの初期化
    codon_cache = {} # key: path_str, value: df_codon
    feature_cache = {} # key: path_str, value: list of feature_records
    all_x_features = [] # 特徴量レコードを格納
    
    # 6. 効率的な階層処理による特徴量生成
    for i, path_str in enumerate(unique_x_paths):
        if (i + 1) % config.PRINT_COUNT == 0:
            force_print(f"[INFO]  - Processing path {i + 1}/{len(unique_x_paths)}: {path_str}")
            
        # 親パスの情報を取得して効率化
        parent_path_str = get_parent_path(path_str)
        
        if parent_path_str and parent_path_str in codon_cache:
            # 親パスの結果を再利用
            df_codon_current = codon_cache[parent_path_str].copy()
            parent_features = feature_cache.get(parent_path_str, [])
            
            # 親パスの特徴量をコピー（path_strを更新）
            for feature in parent_features:
                feature_copy = feature.copy()
                feature_copy['path_str'] = path_str
                all_x_features.append(feature_copy)
            
            # 差分変異のみ処理
            parent_mutations = get_mutations_from_path_str(parent_path_str)
            current_mutations = get_mutations_from_path_str(path_str)
            diff_mutations = current_mutations[len(parent_mutations):]
            start_t = len(parent_mutations)
        else:
            # 親パスがない場合は最初から処理
            df_codon_current = df_codon.copy()
            current_mutations = get_mutations_from_path_str(path_str)
            diff_mutations = current_mutations
            start_t = 0
        
        # 現在のパス用の特徴量リスト
        current_path_features = []

        # 差分変異を時系列で処理
        for t, mutations_str in enumerate(diff_mutations, start=start_t):
            parsed_muts = parse_mutation(mutations_str)
            
            for bef, pos, aft in parsed_muts:
                # a. 特徴量取得 (更新前のdf_codon_currentを使用)
                codon_info_rows = df_codon_current[df_codon_current['base_pos'] == pos]
                if codon_info_rows.empty: continue
                codon_info = codon_info_rows.iloc[0]
                
                # b. 特徴量レコード作成
                feature_record = {
                    'path_str': path_str,
                    'timestep_idx': t,
                    'bef_id': config.BASE_VOCABS.get(bef, 0),
                    'pos_id': pos,
                    'aft_id': config.BASE_VOCABS.get(aft, 0),
                    'codon_pos_id': codon_info['codon_pos'],
                    'aa_b_id': config.AA_VOCABS.get(codon_info['amino_acid'], 0),
                    'protein_pos_id': codon_info['protein_pos'],
                    'protein_id': config.PROTEIN_VOCABS.get(codon_info['protein'], 0)
                }
                
                # 変異後のアミノ酸を計算
                new_codon = get_new_codon(codon_info['codon'], codon_info['codon_pos'], aft)
                feature_record['aa_a_id'] = config.AA_VOCABS.get(config.DNA2PROTEIN.get(new_codon, '*'), 0)

                # 数値特徴量
                freq_col = f"{bef}->{aft}"
                freq_val = df_freq.loc[pos, freq_col] if pos in df_freq.index and freq_col in df_freq.columns else 0
                
                codon = codon_info['codon']
                bef_aa = DNA2Protein.get(codon, 'X')
                aft_aa = DNA2Protein.get(new_codon, 'X')
                dissim_row = df_dissimilarity[(df_dissimilarity['wt'] == bef_aa) & (df_dissimilarity['mut'] == aft_aa)]
                
                feature_record.update({
                    'freq': freq_val,
                    'hydro': dissim_row['eisenberg_weiss_diff'].iloc[0] if not dissim_row.empty else 0,
                    'charge': dissim_row['charge_diff'].iloc[0] if not dissim_row.empty else 0,
                    'size': dissim_row['size_diff'].iloc[0] if not dissim_row.empty else 0,
                    'blsm': dissim_row['blsm62_diff'].iloc[0] if not dissim_row.empty else 0,
                })
                
                all_x_features.append(feature_record)
                current_path_features.append(feature_record.copy())

                # c. df_codon更新
                df_codon_current = update_codon_table(df_codon_current, (bef, pos, aft))

        # 処理済みのdf_codonと特徴量をキャッシュ
        codon_cache[path_str] = df_codon_current
        # 現在のパスの全特徴量を保存（親パスの特徴量 + 新しい特徴量）
        if parent_path_str and parent_path_str in feature_cache:
            all_path_features = feature_cache[parent_path_str] + current_path_features
        else:
            all_path_features = current_path_features
        feature_cache[path_str] = all_path_features

    if not all_x_features:
        return {
            'input_cat_seq': [], 'input_num_seq': [], 'target_protein': [],
            'target_pos': [], 'original_len': []
        }

    # 7. 特徴量シーケンスの再構成
    df_features = pd.DataFrame(all_x_features)
    grouped_features = df_features.groupby('path_str')

    # ユニークな入力パスごとにシーケンスを事前に構築し、キャッシュする
    sequence_cache = {}
    for path_str, path_features in grouped_features:
        timestep_groups = path_features.groupby('timestep_idx')
        
        cat_seq = []
        num_seq = []
        for _, group in sorted(timestep_groups):
            cat_events = group[['bef_id', 'pos_id', 'aft_id', 'codon_pos_id', 'aa_b_id', 'protein_pos_id', 'aa_a_id', 'protein_id']].values.tolist()
            num_events = group[['freq', 'hydro', 'charge', 'size', 'blsm']].values.tolist()
            cat_seq.append(cat_events)
            num_seq.append(num_events)
        
        sequence_cache[path_str] = (cat_seq, num_seq)

    # df_explodedの各行（サンプル）に対応するシーケンスを割り当てる
    final_cat_sequences = []
    final_num_sequences = []
    empty_sequence = ([], [])

    for path_str in df_exploded['x_path_str']:
        cat_seq, num_seq = sequence_cache.get(path_str, empty_sequence)
        final_cat_sequences.append(cat_seq)
        final_num_sequences.append(num_seq)

    target_protein_ids = []
    target_pos_ids = []
    
    for _, row in df_exploded.iterrows():
        path_str = row['x_path_str']
        y_pos = row['y_pos']
        
        final_codon_table = codon_cache.get(path_str, df_codon)
        
        target_info = final_codon_table[final_codon_table['base_pos'] == y_pos]
        if not target_info.empty:
            protein_id = target_info['protein_id'].iloc[0]
            target_protein_ids.append(int(protein_id) if pd.notna(protein_id) else 0)
            target_pos_ids.append(y_pos)
        else:
            target_protein_ids.append(0)
            target_pos_ids.append(y_pos)

    return {
        'input_cat_seq': final_cat_sequences,
        'input_num_seq': final_num_sequences,
        'target_protein': target_protein_ids,
        'target_pos': target_pos_ids,
        'original_len': df_exploded['original_len'].values
    }

# --- 4. PyTorch Dataset & DataLoader ---

class ViralMutationDataset(Dataset):
    def __init__(self, data_dict):
        self.input_cat_seqs = data_dict['input_cat_seq']
        self.input_num_seqs = data_dict['input_num_seq']
        self.target_proteins = data_dict['target_protein']
        self.target_pos = data_dict['target_pos']
        self.original_lens = data_dict['original_len']

    def __len__(self):
        return len(self.target_proteins)

    def __getitem__(self, idx):
        input_cat_seq = self.input_cat_seqs[idx]
        input_num_seq = self.input_num_seqs[idx]
        
        # モデルに入力するシーケンス長をSEQ_LENに基づいて決定
        seq_len = config.SEQ_LEN if config.SEQ_LEN is not None else config.TRAIN_MAX_LEN

        # パスの先頭からSEQ_LEN分を切り出す（全体を含むように変更）
        if len(input_cat_seq) > seq_len:
            input_cat_seq = input_cat_seq[-seq_len:]
            input_num_seq = input_num_seq[-seq_len:]
        
        active_len = len(input_cat_seq)
        
        # パディングはSEQ_LENの長さに合わせる
        padded_cat = np.zeros((seq_len, config.MAX_CO_OCCUR, config.NUM_CAT_FEATURES), dtype=np.int64)
        padded_num = np.zeros((seq_len, config.MAX_CO_OCCUR, config.NUM_CHEM_FEATURES), dtype=np.float32)
        
        # 共起変異のパディングマスク
        x_cat_mask = np.ones((seq_len, config.MAX_CO_OCCUR), dtype=bool)

        for t, (cat_events, num_events) in enumerate(zip(input_cat_seq, input_num_seq)):
            co_len = min(len(cat_events), config.MAX_CO_OCCUR)
            x_cat_mask[t, :co_len] = False
            for c in range(co_len):
                padded_cat[t, c, :] = cat_events[c]
                padded_num[t, c, :] = num_events[c]

        # シーケンス長のパディングマスク
        seq_mask = np.ones(seq_len, dtype=bool)
        seq_mask[:active_len] = False

        return {
            "x_cat": torch.tensor(padded_cat, dtype=torch.long),
            "x_num": torch.tensor(padded_num, dtype=torch.float),
            "x_cat_mask": torch.tensor(x_cat_mask, dtype=torch.bool),
            "seq_mask": torch.tensor(seq_mask, dtype=torch.bool),
            "y_protein": torch.tensor(self.target_proteins[idx], dtype=torch.long),
            "y_pos": torch.tensor(self.target_pos[idx], dtype=torch.long),
            "original_len": self.original_lens[idx]
        }

def create_dataloader(data_dict, batch_size, shuffle=True):
    dataset = ViralMutationDataset(data_dict)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=2,
        pin_memory=True
    )

# --- ダミーデータと補助辞書の定義 ---

def load_aux_data():
    force_print("[INFO] Loading auxiliary data (codon, freq, etc.)...")
    df_codon = pd.read_csv(config.CODON_CSV_PATH)
    protein_map = {name: i + 1 for i, name in enumerate(df_codon['protein'].unique()) if pd.notna(name)}
    df_codon['protein_id'] = df_codon['protein'].map(protein_map).fillna(0)

    df_codon['amino_acid'] = df_codon['codon'].apply(lambda c: config.DNA2PROTEIN.get(c, '*'))
    df_codon['codon_start'] = df_codon['base_pos'] - (df_codon['codon_pos'] - 1)
    
    df_freq = pd.read_csv(config.FREQ_CSV_PATH, index_col=0)
    df_dissimilarity = pd.read_csv(config.DISSIMILARITY_CSV_PATH)

    return df_codon, df_freq, df_dissimilarity