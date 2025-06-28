import os
import sys
import random
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict
from tqdm import tqdm
import pickle
import re
from multiprocessing import Pool, cpu_count
import functools
try:
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    mpl.rcParams['font.family'] = 'Noto Sans CJK JP'
except ImportError:
    plt = None
from utils_20250602_transf1_AllFea import (
    count_numeric_subfolders,
    select_by_idx,
    pad_and_trim,
    build_vocab,
    encode_seqs,
    make_train_val_idx,
    extract_features_from_mutation,
    extract_feature_lists_with_path
)

# codon_to_aa辞書の定義
codon_to_aa = {
    'AAT': 'N', 'TAT': 'Y', 'GAT': 'D', 'CAT': 'H',
    'AAC': 'N', 'TAC': 'Y', 'GAC': 'D', 'CAC': 'H',
    'AAA': 'K', 'TAA': '*', 'GAA': 'E', 'CAA': 'Q',
    'AAG': 'K', 'TAG': '*', 'GAG': 'E', 'CAG': 'Q',
    'ACT': 'T', 'TCT': 'S', 'GCT': 'A', 'CCT': 'P',
    'ACC': 'T', 'TCC': 'S', 'GCC': 'A', 'CCC': 'P',
    'ACA': 'T', 'TCA': 'S', 'GCA': 'A', 'CCA': 'P',
    'ACG': 'T', 'TCG': 'S', 'GCG': 'A', 'CCG': 'P',
    'ATT': 'I', 'TTT': 'F', 'GTT': 'V', 'CTT': 'L',
    'ATC': 'I', 'TTC': 'F', 'GTC': 'V', 'CTC': 'L',
    'ATA': 'I', 'TTA': 'L', 'GTA': 'V', 'CTA': 'L',
    'ATG': 'M', 'TTG': 'L', 'GTG': 'V', 'CTG': 'L',
    'AGT': 'S', 'TGT': 'C', 'GGT': 'G', 'CGT': 'R',
    'AGC': 'S', 'TGC': 'C', 'GGC': 'G', 'CGC': 'R',
    'AGA': 'R', 'TGA': '*', 'GGA': 'G', 'CGA': 'R',
    'AGG': 'R', 'TGG': 'W', 'GGG': 'G', 'CGG': 'R'
}

def process_single_sample(args):
    path, protein, protein_pos, codon_pos, amino_change, amino_change_flag, codon_df, codon_to_aa = args
    proteins = []
    protein_positions = []
    codon_positions = []
    amino_changes = []
    amino_change_flags = []
    
    for mutation in path:
        if mutation == 'PAD':
            proteins.append('PAD')
            protein_positions.append('PAD')
            codon_positions.append('PAD')
            amino_changes.append('PAD')
            amino_change_flags.append('PAD')
            continue
            
        # カンマ区切りの複数変異を1つのトークンとして処理
        mutations = mutation.split(',')
        combined_proteins = []
        combined_positions = []
        combined_codon_positions = []
        combined_amino_changes = []
        combined_flags = []
        
        for single_mutation in mutations:
            m = re.match(r'([ACGT])([0-9]+)([ACGT])', single_mutation)
            if not m:
                continue
                
            orig_base, position, new_base = m.groups()
            position = int(position)
            mutation_info = codon_df[codon_df['base_pos'] == position]
            
            if len(mutation_info) > 0:
                info = mutation_info.iloc[0]
                combined_proteins.append(info['protein'])
                combined_positions.append(str(info['protein_pos']))
                combined_codon_positions.append(str(info['codon_pos']))
                orig_codon = info['codon']
                codon_pos = int(info['codon_pos'])
                mut_codon = list(orig_codon)
                mut_codon[codon_pos-1] = new_base
                mut_codon = ''.join(mut_codon)
                orig_aa = codon_to_aa.get(orig_codon, '')
                mut_aa = codon_to_aa.get(mut_codon, '')
                
                if orig_aa and mut_aa:
                    # アミノ酸変異の表記を変更（N501Y → N>Y）
                    combined_amino_changes.append(f"{orig_aa}>{mut_aa}")
                    is_synonymous = orig_aa == mut_aa
                    combined_flags.append('synonymous' if is_synonymous else 'non_synonymous')
                else:
                    combined_amino_changes.append('')
                    combined_flags.append('non_synonymous')
            else:
                combined_proteins.append('')
                combined_positions.append('')
                combined_codon_positions.append('-1')
                combined_amino_changes.append('')
                combined_flags.append('non_synonymous')
        
        # 共起変異を1つのトークンとして結合
        if combined_proteins:
            # 各特徴量をカンマで結合
            proteins.append(','.join(filter(None, combined_proteins)))  # 空文字を除外
            protein_positions.append(','.join(filter(None, combined_positions)))
            codon_positions.append(','.join(filter(None, combined_codon_positions)))
            amino_changes.append(','.join(filter(None, combined_amino_changes)))
            # フラグは、1つでもnon_synonymousがあればnon_synonymousとする
            amino_change_flags.append('non_synonymous' if 'non_synonymous' in combined_flags else 'synonymous')
        else:
            proteins.append('')
            protein_positions.append('')
            codon_positions.append('-1')
            amino_changes.append('')
            amino_change_flags.append('non_synonymous')
            
    return proteins, protein_positions, codon_positions, amino_changes, amino_change_flags

CONFIG = {
    'SEED': 42,
    'output_root': '../model/20250602_output/transf1_AllFeaV10/',
    'num_epochs': 20,
    'early_stopping_patience': 5,
    'learning_rate': 1e-3,
    'batch_size': 256,
    'val_batch_size': 256,
    'maxlen': 30,
    'use_data_num': 1000,
    'val_ratio': 0.2,
    # データリーク対策オプション
    'remove_data_leak_samples': True,      # データリークサンプルを除去する
    'filter_vocabulary_overlap': False,    # 語彙レベルでのフィルタリング（注意：大幅にデータが減る可能性）
    'use_stratified_split': True,          # ラベルベース層化分割を使用する
}

def set_seed(seed: int):
    # 環境変数を設定（CUDA CuBLASの決定論的動作のため）
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # より安全な決定論的アルゴリズムの設定
    if hasattr(torch, 'use_deterministic_algorithms'):
        try:
            torch.use_deterministic_algorithms(True)
        except RuntimeError as e:
            print(f"[WARNING] Could not enable full deterministic algorithms: {e}")
            print("[INFO] Continuing with partial determinism settings")
            # 部分的な決定論的設定のみ使用

def make_output_dirs(base_dir: str) -> Dict[str, str]:
    now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    out_root = os.path.join(base_dir, f"run_{now}")
    os.makedirs(out_root, exist_ok=True)
    dirs = {
        'root': out_root,
        'model': os.path.join(out_root, 'model'),
        'fig': os.path.join(out_root, 'fig'),
        'vocab': os.path.join(out_root, 'vocab'),
        'log': os.path.join(out_root, 'log')
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs

class EarlyStopping:
    def __init__(self, patience=5, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
    def save_checkpoint(self, val_loss, model, path):
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss

class MutationDatasetWithPathLast(Dataset):
    def __init__(self, path, protein, protein_pos, codon_pos, amino_change, amino_change_flag, labels):
        self.path = path
        self.protein = protein
        self.protein_pos = protein_pos
        self.codon_pos = codon_pos
        self.amino_change = amino_change
        self.amino_change_flag = amino_change_flag
        self.labels = labels

    def __len__(self):
        return len(self.path)

    def __getitem__(self, idx):
        # amino_change_flagの変換を修正
        amino_change_flag_tensor = torch.tensor(
            [1 if x == 'synonymous' else 2 if x == 'non_synonymous' else 0 
             for x in self.amino_change_flag[idx]], 
            dtype=torch.long
        )

        return {
            'path': torch.tensor(self.path[idx], dtype=torch.long),
            'protein': torch.tensor(self.protein[idx], dtype=torch.long),
            'protein_pos': torch.tensor(self.protein_pos[idx], dtype=torch.long),
            'codon_pos': torch.tensor(self.codon_pos[idx], dtype=torch.long),
            'amino_change': torch.tensor(self.amino_change[idx], dtype=torch.long),
            'amino_change_flag': amino_change_flag_tensor,
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# データ準備時の変換関数を追加
def convert_amino_change_flag(flag_list):
    """
    amino_change_flagを数値に変換する関数
    
    Args:
        flag_list: 文字列のリスト（'synonymous', 'non_synonymous', 'PAD'）
    
    Returns:
        数値のリスト（1: synonymous, 2: non_synonymous, 0: PAD）
    """
    return [1 if x == 'synonymous' else 2 if x == 'non_synonymous' else 0 for x in flag_list]

# デバッグ用の確認関数を追加
def check_amino_change_flag(data_list, name="データ"):
    print(f"\n{name}のamino_change_flag確認:")
    print(f"サンプル数: {len(data_list)}")
    if len(data_list) > 0:
        print(f"最初のサンプルの長さ: {len(data_list[0])}")
        print(f"最初のサンプルの内容: {data_list[0]}")
        print(f"データ型: {type(data_list[0][0])}")

class MutationTransformerWithPath(nn.Module):
    def __init__(self, vocab_sizes: Dict[str, int], d_model=128, nhead=4, num_layers=2, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.embedding_path = nn.Embedding(vocab_sizes['path'], d_model)
        self.embedding_protein = nn.Embedding(vocab_sizes['protein'], d_model)
        self.embedding_protein_pos = nn.Embedding(vocab_sizes['protein_pos'], d_model)
        self.embedding_codon_pos = nn.Embedding(vocab_sizes['codon_pos'], d_model)
        self.embedding_amino_change = nn.Embedding(vocab_sizes['amino_change'], d_model)
        self.embedding_amino_change_flag = nn.Embedding(3, d_model)  # 0/1/2のみ
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True),
            num_layers=num_layers
        )
        self.fc = nn.Linear(d_model, vocab_sizes['path'])
    def forward(self, path, protein, protein_pos, codon_pos, amino_change, amino_change_flag):
        x = self.embedding_path(path) \
            + self.embedding_protein(protein) \
            + self.embedding_protein_pos(protein_pos) \
            + self.embedding_codon_pos(codon_pos) \
            + self.embedding_amino_change(amino_change) \
            + self.embedding_amino_change_flag(amino_change_flag)
        mask = (path == 0).to(torch.bool).contiguous()
        out = self.transformer(x, src_key_padding_mask=mask)
        last_out = out[:, -1, :]
        out = self.fc(last_out)
        return out

def make_laststep_samples(path_list, protein_list, protein_pos_list, codon_pos_list, amino_change_list, maxlen):
    """最後のステップを予測するためのサンプルを作成（末尾から切り取り）"""
    X_path, X_protein, X_protein_pos, X_codon_pos, X_amino_change, Y = [], [], [], [], [], []
    Y_base_before, Y_pos, Y_base_after = [], [], []
    
    for idx, (p, pr, prp, cp, ac) in enumerate(zip(path_list, protein_list, protein_pos_list, codon_pos_list, amino_change_list)):
        if len(p) < 2:
            continue  # ラベルが作れない
        
        label = p[-1]
        m = re.match(r"([A-Za-z]+)([0-9]+)([A-Za-z]+)", label)
        if m:
            Y_base_before.append(m.group(1))
            Y_pos.append(m.group(2))
            Y_base_after.append(m.group(3))
        else:
            Y_base_before.append("")
            Y_pos.append("")
            Y_base_after.append("")
        
        input_path = p[max(0, len(p)-1-maxlen):len(p)-1]
        input_protein = pr[max(0, len(pr)-1-maxlen):len(pr)-1]
        input_protein_pos = prp[max(0, len(prp)-1-maxlen):len(prp)-1]
        input_codon_pos = cp[max(0, len(cp)-1-maxlen):len(cp)-1]
        input_amino_change = ac[max(0, len(ac)-1-maxlen):len(ac)-1]
        
        pad_len = maxlen - len(input_path)
        X_path.append([0]*pad_len + input_path)
        X_protein.append([0]*pad_len + input_protein)
        X_protein_pos.append([0]*pad_len + input_protein_pos)
        X_codon_pos.append([0]*pad_len + input_codon_pos)
        X_amino_change.append([0]*pad_len + input_amino_change)
        Y.append(label)
    
    return X_path, X_protein, X_protein_pos, X_codon_pos, X_amino_change, Y, Y_base_before, Y_pos, Y_base_after

def make_laststep_samples_from_head(path_list, protein_list, protein_pos_list, codon_pos_list, amino_change_list, target_len):
    """先頭から指定長まで切り取って最後のステップを予測するためのサンプルを作成"""
    X_path, X_protein, X_protein_pos, X_codon_pos, X_amino_change, Y = [], [], [], [], [], []
    Y_base_before, Y_pos, Y_base_after = [], [], []
    
    for idx, (p, pr, prp, cp, ac) in enumerate(zip(path_list, protein_list, protein_pos_list, codon_pos_list, amino_change_list)):
        # 先頭からtarget_len+1まで切り取り（target_len個の入力 + 1個のラベル）
        if len(p) < target_len + 1:
            continue  # データが不足している場合はスキップ
        
        # 先頭からtarget_len+1個取得
        trimmed_p = p[:target_len + 1]
        trimmed_pr = pr[:target_len + 1]
        trimmed_prp = prp[:target_len + 1]
        trimmed_cp = cp[:target_len + 1]
        trimmed_ac = ac[:target_len + 1]
        
        # 最後の要素をラベルとして使用
        label = trimmed_p[-1]
        m = re.match(r"([A-Za-z]+)([0-9]+)([A-Za-z]+)", label)
        if m:
            Y_base_before.append(m.group(1))
            Y_pos.append(m.group(2))
            Y_base_after.append(m.group(3))
        else:
            Y_base_before.append("")
            Y_pos.append("")
            Y_base_after.append("")
        
        # 入力は最後の1個を除いたもの（つまりtarget_len個）
        input_path = trimmed_p[:-1]
        input_protein = trimmed_pr[:-1]
        input_protein_pos = trimmed_prp[:-1]
        input_codon_pos = trimmed_cp[:-1]
        input_amino_change = trimmed_ac[:-1]
        
        # パディング（先頭に0を追加）
        pad_len = target_len - len(input_path)
        X_path.append([0]*pad_len + input_path)
        X_protein.append([0]*pad_len + input_protein)
        X_protein_pos.append([0]*pad_len + input_protein_pos)
        X_codon_pos.append([0]*pad_len + input_codon_pos)
        X_amino_change.append([0]*pad_len + input_amino_change)
        Y.append(label)
    
    return X_path, X_protein, X_protein_pos, X_codon_pos, X_amino_change, Y, Y_base_before, Y_pos, Y_base_after

def make_amino_change_flag(amino_change_list):
    # amino_change: 例 "D614G" など。"-"なら変化なし。
    flag_list = []
    for seq in amino_change_list:
        flag_seq = []
        for ac in seq:
            if ac == "-":
                flag_seq.append("synonymous")
            else:
                flag_seq.append("non_synonymous")
        flag_list.append(flag_seq)
    return flag_list

def filter_co_occur(data, sample_name, data_len, max_co_occur, out_num):
    """
    元の関数と同じ動作：各サンプル内の最大同時変異数でフィルタリング
    
    Args:
        data: 変異パスのリスト（各要素は変異のリスト）
        sample_name: サンプル名のリスト
        data_len: データ長のリスト  
        max_co_occur: 最大同時変異数の閾値
        out_num: 出力する最大サンプル数
    
    Returns:
        フィルタ後のdata, sample_name, data_lenのタプル
    """
    filted_data = []
    filted_sample_name = []
    filted_data_len = []
    
    for i in range(len(data)):
        compare = 0
        for j in range(len(data[i])):
            mutation = data[i][j].split(',')
            if compare < len(mutation):
                compare = len(mutation)
        
        if compare <= max_co_occur:
            filted_data.append(data[i])
            filted_sample_name.append(sample_name[i])
            filted_data_len.append(data_len[i])
            
            if len(filted_data) >= out_num:
                break
    
    print(f"[INFO] co-occurrence フィルタリング結果:")
    print(f"  元のサンプル数: {len(data)}")
    print(f"  フィルタ後のサンプル数: {len(filted_data)}")
    print(f"  除去されたサンプル数: {len(data) - len(filted_data)}")
    print(f"  保持率: {len(filted_data)/len(data)*100:.1f}%")
    
    return filted_data, filted_sample_name, filted_data_len

def remove_data_leak_samples(X_path, X_protein, X_protein_pos, X_codon_pos, X_amino_change, Y, Y_base_before, Y_pos, Y_base_after, path_vocab, amino_change_vocab):
    """
    データリークの原因となるサンプルを除去する
    
    Args:
        X_path, X_protein, X_protein_pos, X_codon_pos, X_amino_change: 入力データリスト
        Y, Y_base_before, Y_pos, Y_base_after: ラベルデータリスト
        path_vocab, amino_change_vocab: 語彙辞書
        debug: デバッグ情報を出力するかどうか
    
    Returns:
        データリークサンプルを除去した後のデータ
    """
    # 語彙の逆引き辞書を作成
    path_vocab_reverse = {v: k for k, v in path_vocab.items()}
    amino_change_vocab_reverse = {v: k for k, v in amino_change_vocab.items()}
    
    clean_indices = []
    removed_count = 0
    overlap_anywhere_count = 0
    overlap_at_end_count = 0
    
    for i in range(len(Y)):
        input_path = X_path[i]
        input_amino_change = X_amino_change[i]
        label = Y[i]
        
        # ラベルを文字列に変換
        if label in amino_change_vocab_reverse:
            label_str = amino_change_vocab_reverse[label]
        else:
            # 不明なラベルの場合はそのまま保持
            clean_indices.append(i)
            continue
        
        # 入力シーケンスを文字列に変換（PADトークンを除く）
        input_amino_change_strs = []
        for ac_id in input_amino_change:
            if ac_id != 0 and ac_id in amino_change_vocab_reverse:  # 0はPADトークン
                input_amino_change_strs.append(amino_change_vocab_reverse[ac_id])
        
        # データリークチェック
        has_anywhere_overlap = label_str in input_amino_change_strs
        has_end_overlap = len(input_amino_change_strs) > 0 and input_amino_change_strs[-1] == label_str
        
        if has_anywhere_overlap:
            overlap_anywhere_count += 1
            removed_count += 1
        elif has_end_overlap:
            overlap_at_end_count += 1
            removed_count += 1
        else:
            clean_indices.append(i)

    
    # クリーンなサンプルのみを抽出
    clean_X_path = [X_path[i] for i in clean_indices]
    clean_X_protein = [X_protein[i] for i in clean_indices]
    clean_X_protein_pos = [X_protein_pos[i] for i in clean_indices]
    clean_X_codon_pos = [X_codon_pos[i] for i in clean_indices]
    clean_X_amino_change = [X_amino_change[i] for i in clean_indices]
    clean_Y = [Y[i] for i in clean_indices]
    clean_Y_base_before = [Y_base_before[i] for i in clean_indices]
    clean_Y_pos = [Y_pos[i] for i in clean_indices]
    clean_Y_base_after = [Y_base_after[i] for i in clean_indices]
    
    return clean_X_path, clean_X_protein, clean_X_protein_pos, clean_X_codon_pos, clean_X_amino_change, clean_Y, clean_Y_base_before, clean_Y_pos, clean_Y_base_after, clean_indices

def create_stratified_split_by_labels(Y, train_ratio=0.8, val_ratio=0.1, seed=42):
    """
    ラベルに基づく層化分割を実行してクロスデータセット汚染を防ぐ
    
    Args:
        Y: ラベルリスト
        train_ratio: 訓練データの比率
        val_ratio: 検証データの比率 (残りはテスト用)
        seed: ランダムシード
    
    Returns:
        train_indices, val_indices, test_indices: 各データセットのインデックス
    """
    import numpy as np
    from collections import Counter
    
    np.random.seed(seed)
    
    # ラベルの分布を確認
    label_counts = Counter(Y)
    unique_labels = list(label_counts.keys())
    
    print(f"[INFO] 層化分割処理開始...")
    print(f"  ユニークラベル数: {len(unique_labels)}")
    print(f"  分割比率 - 訓練:{train_ratio:.1f}, 検証:{val_ratio:.1f}, テスト:{1-train_ratio-val_ratio:.1f}")
    
    train_indices = []
    val_indices = []
    test_indices = []
    
    for label in unique_labels:
        # このラベルを持つ全インデックスを取得
        label_indices = [i for i, y in enumerate(Y) if y == label]
        n_samples = len(label_indices)
        
        # シャッフル
        np.random.shuffle(label_indices)
        
        # 分割点を計算
        n_train = max(1, int(n_samples * train_ratio))  # 最低1サンプルは訓練に
        n_val = max(1 if n_samples > 2 else 0, int(n_samples * val_ratio))  # サンプルが2個以下なら検証は0
        
        # サンプル数が少ない場合の調整
        if n_train + n_val > n_samples:
            if n_samples == 1:
                n_train, n_val = 1, 0
            elif n_samples == 2:
                n_train, n_val = 1, 1
            else:
                n_val = n_samples - n_train
        
        # 分割実行
        train_indices.extend(label_indices[:n_train])
        val_indices.extend(label_indices[n_train:n_train + n_val])
        test_indices.extend(label_indices[n_train + n_val:])
    
    # シャッフル
    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)
    np.random.shuffle(test_indices)
    
    print(f"[INFO] 層化分割結果:")
    print(f"  訓練データ: {len(train_indices)} サンプル")
    print(f"  検証データ: {len(val_indices)} サンプル")
    print(f"  テストデータ: {len(test_indices)} サンプル")
    
    # クロス汚染チェック
    train_labels = set(Y[i] for i in train_indices)
    val_labels = set(Y[i] for i in val_indices)
    test_labels = set(Y[i] for i in test_indices)
    
    train_val_overlap = len(train_labels & val_labels)
    train_test_overlap = len(train_labels & test_labels)
    val_test_overlap = len(val_labels & test_labels)
    
    print(f"[INFO] クロスデータセット汚染チェック:")
    print(f"  訓練-検証ラベル重複: {train_val_overlap}")
    print(f"  訓練-テストラベル重複: {train_test_overlap}")
    print(f"  検証-テストラベル重複: {val_test_overlap}")
    
    return train_indices, val_indices, test_indices


def filter_vocabulary_overlap(X_amino_change, Y, amino_change_vocab):
    """
    語彙レベルでのデータリークを解決するため、入力とラベルで重複する語彙を除去
    
    Args:
        X_amino_change: 入力のアミノ酸変化シーケンス
        Y: ラベル
        amino_change_vocab: アミノ酸変化語彙辞書
        debug: デバッグ情報を出力するかどうか
    
    Returns:
        フィルタリング後のデータと新しい語彙辞書
    """
    # 語彙の逆引き辞書
    vocab_reverse = {v: k for k, v in amino_change_vocab.items()}
    
    # 入力で使用される語彙を収集
    input_vocab_ids = set()
    for seq in X_amino_change:
        for token_id in seq:
            if token_id != 0:  # PADトークンを除く
                input_vocab_ids.add(token_id)
    
    # ラベルで使用される語彙を収集
    label_vocab_ids = set(Y)
    
    # 重複語彙を特定
    overlap_vocab_ids = input_vocab_ids & label_vocab_ids
    
    # 重複語彙を含むサンプルを除去するか、語彙をマスクするか選択
    # ここでは保守的にサンプル除去を選択
    clean_indices = []
    removed_samples = 0
    
    for i in range(len(Y)):
        label = Y[i]
        input_seq = X_amino_change[i]
        
        # ラベルが重複語彙に含まれているかチェック
        if label in overlap_vocab_ids:
            removed_samples += 1
            continue
        
        # 入力シーケンスに重複語彙が含まれているかチェック
        has_overlap = any(token_id in overlap_vocab_ids for token_id in input_seq if token_id != 0)
        if has_overlap:
            removed_samples += 1
            continue
        
        clean_indices.append(i)

    return clean_indices

def main():
    config = CONFIG
    set_seed(config['SEED'])
    dirs = make_output_dirs(config['output_root'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # --- データ読み込み・前処理 ---
    name = []
    length = []
    mutation_paths = []
    strain = ['B.1.1.7']
    
    # 全件データの読み込み
    print("[INFO] 全件データの読み込み開始...")
    for s in strain:
        dir = 'sequences_20241017_'+s+'_random/'
        folder_num = count_numeric_subfolders(dir)
        for num in range(folder_num):
            f = open(dir+str(num)+'/mutation_paths_'+s+'.tsv', 'r',encoding="utf-8_sig")
            print(dir+str(num)+'/mutation_paths_'+s+'.tsv')
            datalist = f.readlines()
            f.close()
            for i in range(1,len(datalist)):
                data = datalist[i].split('\t')
                name.append(data[0])
                length.append(int(data[1]))
                mutation_paths.append(data[2].rstrip().split('>'))
    
    print(f"[INFO] 全件読み込み完了: {len(mutation_paths)} サンプル")
    
    # --- filter_co_occurによるフィルタリング ---
    print("\n[INFO] 共起数フィルタリング開始...")
    max_co_occur = 5
    out_num = config['use_data_num']   # 全件フィルタリング
    #out_num = len(mutation_paths)  # 全件フィルタリング
    
    filtered_mutation_paths, filtered_name, filtered_length = filter_co_occur(
        mutation_paths, name, length, max_co_occur, out_num)
    
    print(f"[INFO] フィルタリング完了:")
    print(f"  フィルタ前: {len(mutation_paths)} サンプル")
    print(f"  フィルタ後: {len(filtered_mutation_paths)} サンプル")
    
    # --- 学習用データの抽出（use_data_num分）---
    use_data_num = config['use_data_num']
    final_mutation_paths = filtered_mutation_paths[:use_data_num]
    final_name = filtered_name[:use_data_num]
    final_length = filtered_length[:use_data_num]
    
    print(f"[INFO] 学習用データ抽出: {len(final_mutation_paths)} サンプル")
    
    # 特徴量抽出（フィルタ後の全件から）
    print(f"[INFO] 特徴量抽出中（フィルタ後全件: {len(filtered_mutation_paths)} サンプル）...")
    all_path_list, all_protein_list, all_protein_pos_list, all_codon_pos_list, all_amino_change_list = extract_feature_lists_with_path(
        filtered_mutation_paths, extract_features_from_mutation)
    all_amino_change_flag_list = make_amino_change_flag(all_amino_change_list)
    
    # 学習用データの特徴量抽出
    path_list, protein_list, protein_pos_list, codon_pos_list, amino_change_list = extract_feature_lists_with_path(
        final_mutation_paths, extract_features_from_mutation)
    amino_change_flag_list = make_amino_change_flag(amino_change_list)

    # --- 新しい系列長による分割（データ拡張版） ---
    maxlen = config['maxlen']
    
    print("\n" + "="*60)
    print("新しい系統長による分割（データ拡張版）")
    print("="*60)
    
    # 長さ分布の詳細確認
    from collections import Counter, defaultdict
    length_counter = Counter(final_length)
    
    print("[詳細分析] 全データの長さ分布:")
    for length_val in sorted(length_counter.keys()):
        count = length_counter[length_val]
        print(f"  長さ {length_val}: {count} サンプル")
    print()
    
    # 訓練データ: 全データ（≤maxlen + >maxlenから先頭maxlenまで切り出し）
    trainval_idx = list(range(len(final_length)))  # 全データを訓練候補に
    
    # テストデータセット: 系統長ごとに分割
    test_datasets = {}
    max_length = max(final_length)
    
    print(f"[設定] maxlen = {maxlen}")
    print(f"[新分割ルール]:")
    print(f"  訓練データ: 全データ（長いデータは{maxlen}まで先頭切り取り）")
    print(f"  テストデータ: 系統長{maxlen+1}〜{max_length}まで、各長さごとに分割（先頭から切り取り）")
    print()
    
    # 各テスト系統長に対してデータセットを構築
    for target_len in range(maxlen + 1, max_length + 1):
        if target_len in length_counter:
            # この系統長のデータのインデックス
            exact_len_indices = [i for i, l in enumerate(final_length) if l == target_len]
            
            # より長いデータから切り出したもののインデックス
            longer_indices = [i for i, l in enumerate(final_length) if l > target_len]
            
            # 合計のインデックス
            test_indices = exact_len_indices + longer_indices
            
            test_datasets[target_len] = {
                'indices': test_indices,
                'exact_count': len(exact_len_indices),
                'trimmed_count': len(longer_indices),
                'total_count': len(test_indices)
            }
            
            print(f"  テスト長{target_len}: {len(exact_len_indices)}サンプル + {len(longer_indices)}切り取り = {len(test_indices)}合計")
    
    print()
    print(f"[結果] 全データ数: {len(final_length)}")
    print(f"[結果] 訓練候補: {len(trainval_idx)} サンプル")
    print(f"[結果] テストセット数: {len(test_datasets)} 種類")
    
    print("="*60)
    print("新分割の構築完了")
    print("="*60 + "\n")
    
    # 訓練データの詳細分析
    print("[INFO] 訓練データの詳細分析...")
    trainval_lengths = [final_length[i] for i in trainval_idx]
    trainval_short_count = len([l for l in trainval_lengths if l <= maxlen])  # 元から30以下
    trainval_long_count = len([l for l in trainval_lengths if l > maxlen])   # 30より長い（切り取り対象）
    
    print(f"  訓練長{maxlen}: {trainval_short_count}サンプル + {trainval_long_count}切り取り = {trainval_short_count + trainval_long_count}合計")
    print()
    
    # サンプル化（訓練・検証用：元データと切り取りデータを分けて処理）
    trainval_path_list = [path_list[i] for i in trainval_idx]
    trainval_protein_list = [protein_list[i] for i in trainval_idx]
    trainval_protein_pos_list = [protein_pos_list[i] for i in trainval_idx]
    trainval_codon_pos_list = [codon_pos_list[i] for i in trainval_idx]
    trainval_amino_change_list = [amino_change_list[i] for i in trainval_idx]
    trainval_amino_change_flag_list = [amino_change_flag_list[i] for i in trainval_idx]
    
    # 元から≤maxlenのデータ（切り取り不要）
    short_indices = [i for i, l in enumerate(trainval_lengths) if l <= maxlen]
    short_path_list = [trainval_path_list[i] for i in short_indices]
    short_protein_list = [trainval_protein_list[i] for i in short_indices]
    short_protein_pos_list = [trainval_protein_pos_list[i] for i in short_indices]
    short_codon_pos_list = [trainval_codon_pos_list[i] for i in short_indices]
    short_amino_change_list = [trainval_amino_change_list[i] for i in short_indices]
    short_amino_change_flag_list = [trainval_amino_change_flag_list[i] for i in short_indices]
    
    # 元データの処理（末尾予測形式に変換、入力長をmaxlen-1に統一）
    short_X_path, short_X_protein, short_X_protein_pos, short_X_codon_pos, short_X_amino_change = [], [], [], [], []
    short_Y, short_Y_base_before, short_Y_pos, short_Y_base_after = [], [], [], []
    
    for i in range(len(short_path_list)):
        p = short_path_list[i]
        pr = short_protein_list[i]
        prp = short_protein_pos_list[i]
        cp = short_codon_pos_list[i]
        ac = short_amino_change_list[i]
        
        if len(p) >= 2:  # 最低限の長さチェック
            # 最後の要素をラベルとして使用
            input_seq = p[:-1]  # 最後以外の要素
            label = p[-1]
            
            # 入力長をmaxlen-1に統一（パディングまたは切り取り）
            if len(input_seq) > maxlen - 1:
                # 末尾からmaxlen-1個を取り出し
                input_seq = input_seq[-(maxlen-1):]
                input_pr = pr[-(maxlen-1):]
                input_prp = prp[-(maxlen-1):]
                input_cp = cp[-(maxlen-1):]
                input_ac = ac[-(maxlen-1):]
            else:
                # パディングで長さを統一
                pad_len = (maxlen - 1) - len(input_seq)
                input_seq = ["PAD"] * pad_len + input_seq
                input_pr = ["PAD"] * pad_len + pr[:-1]
                input_prp = ["PAD"] * pad_len + prp[:-1]
                input_cp = ["PAD"] * pad_len + cp[:-1]
                input_ac = ["PAD"] * pad_len + ac[:-1]
            
            short_X_path.append(input_seq)
            short_X_protein.append(input_pr)
            short_X_protein_pos.append(input_prp)
            short_X_codon_pos.append(input_cp)
            short_X_amino_change.append(input_ac)
            
            short_Y.append(label)
            
            # ラベルの分解
            import re
            m = re.match(r"([A-Za-z]+)([0-9]+)([A-Za-z]+)", label)
            if m:
                short_Y_base_before.append(m.group(1))
                short_Y_pos.append(m.group(2))
                short_Y_base_after.append(m.group(3))
            else:
                short_Y_base_before.append("")
                short_Y_pos.append("")
                short_Y_base_after.append("")
    
    # amino_change_flagの処理（入力長をmaxlen-1に統一）
    short_X_amino_change_flag = []
    for i in range(len(short_path_list)):
        flag_seq = short_amino_change_flag_list[i][:-1]  # ラベル部分を除去
        
        # 入力長をmaxlen-1に統一（パディングまたは切り取り）
        if len(flag_seq) > maxlen - 1:
            # 末尾からmaxlen-1個を取り出し
            flag_seq = flag_seq[-(maxlen-1):]
        else:
            # パディングで長さを統一
            pad_len = (maxlen - 1) - len(flag_seq)
            flag_seq = ["PAD"] * pad_len + flag_seq
        
        short_X_amino_change_flag.append(flag_seq)
    
    # >maxlenのデータ（切り取り必要）
    long_indices = [i for i, l in enumerate(trainval_lengths) if l > maxlen]
    long_path_list = [trainval_path_list[i] for i in long_indices]
    long_protein_list = [trainval_protein_list[i] for i in long_indices]
    long_protein_pos_list = [trainval_protein_pos_list[i] for i in long_indices]
    long_codon_pos_list = [trainval_codon_pos_list[i] for i in long_indices]
    long_amino_change_list = [trainval_amino_change_list[i] for i in long_indices]
    long_amino_change_flag_list = [trainval_amino_change_flag_list[i] for i in long_indices]
    
    # 切り取りデータの処理
    long_X_path, long_X_protein, long_X_protein_pos, long_X_codon_pos, long_X_amino_change, long_Y, long_Y_base_before, long_Y_pos, long_Y_base_after = make_laststep_samples_from_head(
        long_path_list, long_protein_list, long_protein_pos_list, long_codon_pos_list, long_amino_change_list, maxlen)
    
    # amino_change_flagの処理（入力長をmaxlen-1に統一）
    long_X_amino_change_flag = []
    for i in range(len(long_amino_change_flag_list)):
        if len(long_amino_change_flag_list[i]) < maxlen + 1:
            continue  # データが不足している場合はスキップ
            
        # 先頭からmaxlen個取得し、最後の要素を除く
        flag_seq = long_amino_change_flag_list[i][:maxlen + 1][:-1]
        
        # 長さをmaxlen-1に調整
        if len(flag_seq) > maxlen - 1:
            # 先頭からmaxlen-1個を取り出す
            flag_seq = flag_seq[:maxlen-1]
        
        # パディング（先頭に"PAD"を追加して長さをmaxlen-1に統一）
        pad_len = (maxlen - 1) - len(flag_seq)
        long_X_amino_change_flag.append(["PAD"]*pad_len + flag_seq)
    
    # 全訓練データを結合
    X_path = short_X_path + long_X_path
    X_protein = short_X_protein + long_X_protein
    X_protein_pos = short_X_protein_pos + long_X_protein_pos
    X_codon_pos = short_X_codon_pos + long_X_codon_pos
    X_amino_change = short_X_amino_change + long_X_amino_change
    X_amino_change_flag = short_X_amino_change_flag + long_X_amino_change_flag
    Y = short_Y + long_Y
    Y_base_before = short_Y_base_before + long_Y_base_before
    Y_pos = short_Y_pos + long_Y_pos
    Y_base_after = short_Y_base_after + long_Y_base_after
    
    print(f"[INFO] 訓練データサンプル化完了:")
    print(f"  サンプル化後の訓練データ数: {len(X_path)}")
    print(f"  元から≤{maxlen}: {trainval_short_count}件 → サンプル化後: {trainval_short_count}件")
    print(f"  >{maxlen}から切り取り: {trainval_long_count}件 → サンプル化後: {trainval_long_count}件")
    print(f"  合計: {len(X_path)}件")
    print()
    
    # --- テストデータセットのサンプル化（各系統長ごと） ---
    test_datasets_sampled = {}
    
    for target_len, dataset_info in test_datasets.items():
        print(f"[INFO] テスト長{target_len}のサンプル化中...")
        
        test_indices = dataset_info['indices']
        exact_count = dataset_info['exact_count']
        trimmed_count = dataset_info['trimmed_count']
        
        # 元の系統長データ（切り取り不要）
        exact_indices = test_indices[:exact_count]
        exact_path_list = [path_list[i] for i in exact_indices]
        exact_protein_list = [protein_list[i] for i in exact_indices]
        exact_protein_pos_list = [protein_pos_list[i] for i in exact_indices]
        exact_codon_pos_list = [codon_pos_list[i] for i in exact_indices]
        exact_amino_change_list = [amino_change_list[i] for i in exact_indices]
        exact_amino_change_flag_list = [amino_change_flag_list[i] for i in exact_indices]
        
        # 元データの処理（切り取りなし - 直接使用）
        if exact_count > 0:
            # 元データは末尾予測形式でサンプル化（元の長さ-1まで入力、末尾をラベル）
            exact_X_path, exact_X_protein, exact_X_protein_pos, exact_X_codon_pos, exact_X_amino_change = [], [], [], [], []
            exact_Y, exact_Y_base_before, exact_Y_pos, exact_Y_base_after = [], [], [], []
            
            for i in range(exact_count):
                p = exact_path_list[i]
                pr = exact_protein_list[i]
                prp = exact_protein_pos_list[i]
                cp = exact_codon_pos_list[i]
                ac = exact_amino_change_list[i]
                
                if len(p) >= 2:  # 最低限の長さチェック
                    # 末尾からmaxlen個を取り出し、最後の要素をラベルとして使用
                    if len(p) > maxlen:
                        # 末尾からmaxlen個を取り出し
                        trimmed_p = p[-maxlen:]
                        trimmed_pr = pr[-maxlen:]
                        trimmed_prp = prp[-maxlen:]
                        trimmed_cp = cp[-maxlen:]
                        trimmed_ac = ac[-maxlen:]
                    else:
                        # そのまま使用
                        trimmed_p = p
                        trimmed_pr = pr
                        trimmed_prp = prp
                        trimmed_cp = cp
                        trimmed_ac = ac
                    
                    # 最後の要素をラベルとして使用
                    exact_X_path.append(trimmed_p[:-1])  # 最後以外の要素
                    exact_X_protein.append(trimmed_pr[:-1])
                    exact_X_protein_pos.append(trimmed_prp[:-1])
                    exact_X_codon_pos.append(trimmed_cp[:-1])
                    exact_X_amino_change.append(trimmed_ac[:-1])
                    
                    # ラベル
                    label = trimmed_p[-1]
                    exact_Y.append(label)
                    
                    # ラベルの分解
                    import re
                    m = re.match(r"([A-Za-z]+)([0-9]+)([A-Za-z]+)", label)
                    if m:
                        exact_Y_base_before.append(m.group(1))
                        exact_Y_pos.append(m.group(2))
                        exact_Y_base_after.append(m.group(3))
                    else:
                        exact_Y_base_before.append("")
                        exact_Y_pos.append("")
                        exact_Y_base_after.append("")
            
            # amino_change_flagの処理
            exact_X_amino_change_flag = []
            for i in range(len(exact_X_path)):
                # 元のフラグリストから対応するものを取得
                flag_seq = exact_amino_change_flag_list[i]
                
                # trimmed_pと同様に処理
                if len(flag_seq) > maxlen:
                    # 末尾からmaxlen個を取り出し
                    flag_seq = flag_seq[-maxlen:]
                
                # 最後の要素（ラベル）を除く
                flag_seq = flag_seq[:-1]
                
                # 長さをmaxlen-1に調整
                if len(flag_seq) > maxlen - 1:
                    flag_seq = flag_seq[-(maxlen-1):]
                else:
                    # パディングで長さを統一
                    pad_len = (maxlen - 1) - len(flag_seq)
                    flag_seq = ["PAD"] * pad_len + flag_seq
                
                exact_X_amino_change_flag.append(flag_seq)
        else:
            exact_X_path, exact_X_protein, exact_X_protein_pos, exact_X_codon_pos, exact_X_amino_change = [], [], [], [], []
            exact_X_amino_change_flag, exact_Y, exact_Y_base_before, exact_Y_pos, exact_Y_base_after = [], [], [], [], []
        
        # より長いデータ（切り取り必要）
        trimmed_indices = test_indices[exact_count:]
        trimmed_path_list = [path_list[i] for i in trimmed_indices]
        trimmed_protein_list = [protein_list[i] for i in trimmed_indices]
        trimmed_protein_pos_list = [protein_pos_list[i] for i in trimmed_indices]
        trimmed_codon_pos_list = [codon_pos_list[i] for i in trimmed_indices]
        trimmed_amino_change_list = [amino_change_list[i] for i in trimmed_indices]
        trimmed_amino_change_flag_list = [amino_change_flag_list[i] for i in trimmed_indices]
        
        # 切り取りデータのサンプル化（target_lenまで先頭から切り取り）
        trimmed_X_path, trimmed_X_protein, trimmed_X_protein_pos, trimmed_X_codon_pos, trimmed_X_amino_change, trimmed_Y, trimmed_Y_base_before, trimmed_Y_pos, trimmed_Y_base_after = make_laststep_samples_from_head(
            trimmed_path_list, trimmed_protein_list, trimmed_protein_pos_list, trimmed_codon_pos_list, trimmed_amino_change_list, target_len)
        
        # amino_change_flagの処理：各サンプルの先頭からtarget_len個取得し、最後の要素を除く（ラベル）
        trimmed_X_amino_change_flag = []
        for i, flag_list in enumerate(trimmed_amino_change_flag_list):
            if len(flag_list) < target_len + 1:
                continue  # データが不足している場合はスキップ
                
            # 先頭からtarget_len+1個取得し、最後の要素を除く
            flag_seq = flag_list[:target_len + 1][:-1]
            
            # 長さをtarget_len-1に調整
            if len(flag_seq) > target_len - 1:
                # 先頭からtarget_len-1個を取り出す
                flag_seq = flag_seq[:target_len-1]
            
            # パディング（先頭に"PAD"を追加して長さをtarget_len-1に統一）
            pad_len = (target_len - 1) - len(flag_seq)
            trimmed_X_amino_change_flag.append(["PAD"]*pad_len + flag_seq)
            
            # インデックスの一貫性チェック（デバッグ用、必要に応じて有効化）
            #if len(trimmed_X_amino_change_flag) != len(trimmed_X_path):
            #    print(f"警告: インデックス{i}でamino_change_flagとpathのサンプル数が一致しません")
        
        # 全データを結合
        test_X_path = exact_X_path + trimmed_X_path
        test_X_protein = exact_X_protein + trimmed_X_protein
        test_X_protein_pos = exact_X_protein_pos + trimmed_X_protein_pos
        test_X_codon_pos = exact_X_codon_pos + trimmed_X_codon_pos
        test_X_amino_change = exact_X_amino_change + trimmed_X_amino_change
        test_X_amino_change_flag = exact_X_amino_change_flag + trimmed_X_amino_change_flag
        test_Y = exact_Y + trimmed_Y
        test_Y_base_before = exact_Y_base_before + trimmed_Y_base_before
        test_Y_pos = exact_Y_pos + trimmed_Y_pos
        test_Y_base_after = exact_Y_base_after + trimmed_Y_base_after
        
        test_datasets_sampled[target_len] = {
            'path': test_X_path,
            'protein': test_X_protein,
            'protein_pos': test_X_protein_pos,
            'codon_pos': test_X_codon_pos,
            'amino_change': test_X_amino_change,
            'amino_change_flag': test_X_amino_change_flag,
            'labels': test_Y,
            'labels_base_before': test_Y_base_before,
            'labels_pos': test_Y_pos,
            'labels_base_after': test_Y_base_after,
            'info': dataset_info
        }
        
        print(f"  サンプル数: {len(test_X_path)}")
        print(f"  元データ: {exact_count}件, 切り取りデータ: {trimmed_count}件")
    
    # --- フィルタ後全件のサンプル化（語彙構築のため） ---
    print("\n[INFO] フィルタ後全件のサンプル化中（語彙構築用）...")
    all_X_path, all_X_protein, all_X_protein_pos, all_X_codon_pos, all_X_amino_change, all_Y, all_Y_base_before, all_Y_pos, all_Y_base_after = make_laststep_samples(
        all_path_list, all_protein_list, all_protein_pos_list, all_codon_pos_list, all_amino_change_list, maxlen)
    
    # --- 語彙構築（フィルタ後の全件データから構築） ---
    print("\n[INFO] 語彙構築中（フィルタ後全件データ使用）...")
    # vocabはフィルタ後の全件データ（入力・ラベル含む）から構築
    vocab_path_data = all_X_path + [[y] for y in all_Y]  # 入力シーケンス + ラベル
    vocab_protein_data = all_X_protein
    vocab_protein_pos_data = all_X_protein_pos
    vocab_codon_pos_data = all_X_codon_pos
    vocab_amino_change_data = all_X_amino_change
    
    path_vocab = build_vocab(vocab_path_data)
    protein_vocab = build_vocab(vocab_protein_data)
    protein_pos_vocab = build_vocab(vocab_protein_pos_data)
    codon_pos_vocab = build_vocab(vocab_codon_pos_data)
    amino_change_vocab = build_vocab(vocab_amino_change_data)
    amino_change_flag_vocab = {"PAD": 0, "synonymous": 1, "non_synonymous": 2}
    
    # amino_change_flagの準備
    X_amino_change_flag = pad_and_trim(trainval_amino_change_flag_list, maxlen, pad_token="PAD")
    X_amino_change_flag = encode_seqs(X_amino_change_flag, amino_change_flag_vocab)
    
    print(f"語彙構築完了:")
    print(f"  path_vocab: {len(path_vocab)} tokens")
    print(f"  protein_vocab: {len(protein_vocab)} tokens")
    print(f"  protein_pos_vocab: {len(protein_pos_vocab)} tokens")
    print(f"  codon_pos_vocab: {len(codon_pos_vocab)} tokens")
    print(f"  amino_change_vocab: {len(amino_change_vocab)} tokens")
    print(f"  amino_change_flag_vocab: {len(amino_change_flag_vocab)} tokens")
    
    # --- データリーク対策の適用 ---
    print("\n" + "="*60)
    print("データリーク（考慮済み）対策処理開始")
    print("="*60)
    
    # オプション1: データリークサンプルの除去
    if config.get('remove_data_leak_samples', True):
        print("[STEP 1] データリークサンプルの除去...")
        X_path, X_protein, X_protein_pos, X_codon_pos, X_amino_change, Y, Y_base_before, Y_pos, Y_base_after, clean_indices = remove_data_leak_samples(
            X_path, X_protein, X_protein_pos, X_codon_pos, X_amino_change, Y, Y_base_before, Y_pos, Y_base_after, 
            path_vocab, amino_change_vocab)
        # amino_change_flagも同期して更新
        X_amino_change_flag = [X_amino_change_flag[i] for i in clean_indices]
    
    # オプション2: 語彙レベルでのフィルタリング（オプション、サンプル数が大幅に減る可能性あり）
    if config.get('filter_vocabulary_overlap', False):
        print("[STEP 2] 語彙レベルでのデータリーク対策...")
        vocab_clean_indices = filter_vocabulary_overlap(X_amino_change, Y, amino_change_vocab)
        # 必要に応じてここでサンプルをさらにフィルタリング
        if len(vocab_clean_indices) < len(Y) * 0.5:  # 50%以上のサンプルが除去される場合は警告
            print(f"[警告] 語彙フィルタリングで{len(Y) - len(vocab_clean_indices)}サンプルが除去されます。")
            print(f"これによりデータセットが{len(vocab_clean_indices)}サンプルに減少します。")
            print("語彙フィルタリングをスキップします。より良い解決策が必要です。")
        else:
            # 語彙フィルタリングを適用
            X_path = [X_path[i] for i in vocab_clean_indices]
            X_protein = [X_protein[i] for i in vocab_clean_indices]
            X_protein_pos = [X_protein_pos[i] for i in vocab_clean_indices]
            X_codon_pos = [X_codon_pos[i] for i in vocab_clean_indices]
            X_amino_change = [X_amino_change[i] for i in vocab_clean_indices]
            Y = [Y[i] for i in vocab_clean_indices]
            Y_base_before = [Y_base_before[i] for i in vocab_clean_indices]
            Y_pos = [Y_pos[i] for i in vocab_clean_indices]
            Y_base_after = [Y_base_after[i] for i in vocab_clean_indices]
            X_amino_change_flag = [X_amino_change_flag[i] for i in vocab_clean_indices]
    
    # 改善されたデータ分割（ラベルベース層化分割）
    print(f"[INFO] データ分割前の状況:")
    print(f"  総データ数: {len(X_path)}")
    print(f"  総ラベル数: {len(set(Y))}")
    print(f"  設定val_ratio: {config['val_ratio']}")
    
    if config.get('use_stratified_split', True) and len(set(Y)) >= 10:  # ラベル数が十分にある場合のみ層化分割
        print("[STEP 3] ラベルベース層化分割の適用...")
        try:
            train_idx, val_idx, internal_test_idx = create_stratified_split_by_labels(
                Y, train_ratio=1-config['val_ratio'], val_ratio=config['val_ratio'], seed=config['SEED'])
            print(f"  層化分割結果: train={len(train_idx)}, val={len(val_idx)}")
        except Exception as e:
            print(f"[警告] 層化分割に失敗しました: {e}")
            print("従来の分割方法にフォールバックします...")
            idx = list(range(len(X_path)))
            train_idx, val_idx = make_train_val_idx(idx, train_size=len(idx), val_ratio=config['val_ratio'], seed=config['SEED'])
            print(f"  従来分割結果: train={len(train_idx)}, val={len(val_idx)}")
    else:
        # 従来の分割方法
        print("[STEP 3] 従来の分割方法を適用...")
        idx = list(range(len(X_path)))
        train_idx, val_idx = make_train_val_idx(idx, train_size=len(idx), val_ratio=config['val_ratio'], seed=config['SEED'])
        print(f"  従来分割結果: train={len(train_idx)}, val={len(val_idx)}")
    
    # 検証データが0件の場合の対策
    if len(val_idx) == 0:
        print("[緊急対策] 検証データが0件です。最小限の検証データを確保します...")
        # 最低でも1件は検証データとして確保
        min_val_size = max(1, int(len(X_path) * 0.1))  # 最低10%または1件
        if len(X_path) > min_val_size:
            val_idx = list(range(len(X_path) - min_val_size, len(X_path)))
            train_idx = list(range(len(X_path) - min_val_size))
            print(f"  緊急分割結果: train={len(train_idx)}, val={len(val_idx)}")
        else:
            print(f"  [警告] データ数が極めて少ないため({len(X_path)}件)、検証データを作成できません。")
            print("  検証なしで訓練を実行します。")
            val_idx = []
    
    # データリーク対策後の最終確認
    print("\n[最終確認] データリーク対策後の検証...")
    if len(val_idx) > 0:
        train_Y = [Y[i] for i in train_idx]
        val_Y = [Y[i] for i in val_idx]
        train_labels_set = set(train_Y)
        val_labels_set = set(val_Y)
        overlap_count = len(train_labels_set & val_labels_set)
        print(f"  訓練データ数: {len(train_idx)}")
        print(f"  検証データ数: {len(val_idx)}")
        print(f"  訓練データラベル数: {len(train_labels_set)}")
        print(f"  検証データラベル数: {len(val_labels_set)}")
        print(f"  訓練-検証ラベル重複: {overlap_count}")
        if overlap_count == 0:
            print("  [成功] 訓練-検証間のラベル重複が解消されました！")
        else:
            print(f"  [注意] まだ{overlap_count}個のラベル重複があります。")
    else:
        print(f"  訓練データ数: {len(train_idx)}")
        print(f"  検証データ数: 0 (検証データなし)")
        print("  [警告] 検証データがありません。訓練のみ実行されます。")
    
    print("="*60)
    print("データリーク対策処理完了")
    print("="*60 + "\n")

    # --- ID変換 ---
    print("\n[INFO] データの次元数と内容の確認")
    print("="*60)
    
    # 特徴量の定義を確認
    print("\n特徴量の定義:")
    print("1. タンパク質領域 (protein): 変異が発生したタンパク質の領域名（例：S, N, M, E, ORF1a, ORF1b等）")
    print("2. タンパク質位置 (protein_pos): タンパク質領域内でのアミノ酸位置")
    print("3. コドン位置 (codon_pos): コドン内での位置（1,2,3）")
    print("4. アミノ酸変異 (amino_change): 元のアミノ酸から変異後のアミノ酸への変化（例：D614G）")
    print("5. 変異タイプ (amino_change_flag): シノニマス変異かノンシノニマス変異か")
    print("6. 変異パス (path): 変異の系列")
    
    # codon_mutation.csvから特徴量を読み込む
    print("\n[INFO] codon_mutation.csvから特徴量を読み込み中...")


    def update_features(path_list, protein_list, protein_pos_list, codon_pos_list, amino_change_list, amino_change_flag_list):
        if codon_df is None:
            return protein_list, protein_pos_list, codon_pos_list, amino_change_list, amino_change_flag_list
        num_cores = cpu_count() - 1
        print(f"[INFO] {num_cores}コアで並列処理を開始します...")
        input_data = [(p, pr, prp, cp, ac, acf, codon_df, codon_to_aa)
                    for p, pr, prp, cp, ac, acf in zip(path_list, protein_list, protein_pos_list,
                                                        codon_pos_list, amino_change_list, amino_change_flag_list)]
        with Pool(num_cores) as pool:
            results = list(tqdm(pool.imap(process_single_sample, input_data), total=len(input_data), desc="特徴量更新中"))
        updated_protein = []
        updated_protein_pos = []
        updated_codon_pos = []
        updated_amino_change = []
        updated_amino_change_flag = []
        for result in results:
            proteins, protein_positions, codon_positions, amino_changes, amino_change_flags = result
            updated_protein.append(proteins)
            updated_protein_pos.append(protein_positions)
            updated_codon_pos.append(codon_positions)
            updated_amino_change.append(amino_changes)
            updated_amino_change_flag.append(amino_change_flags)
        return updated_protein, updated_protein_pos, updated_codon_pos, updated_amino_change, updated_amino_change_flag

    try:
        import pandas as pd
        # codon_mutation.csvを読み込み
        codon_df = pd.read_csv('data/codon_mutation.csv', comment='#')
        print(f"[INFO] codon_mutation.csvから{len(codon_df)}行のデータを読み込みました")
        print(f"[INFO] カラム: {codon_df.columns.tolist()}")
        
        # 塩基配列の初期状態を設定
        base_sequence = pd.Series(['A'] * 30000)  # 十分な長さの塩基配列を初期化

    except FileNotFoundError:
        print("[警告] codon_mutation.csvが見つかりません。デフォルト値を使用します。")
        # デフォルトのupdate_features関数を定義

    # 訓練データの特徴量を更新
    print("\n[INFO] 訓練データの特徴量を更新中...")
    X_protein, X_protein_pos, X_codon_pos, X_amino_change, X_amino_change_flag = update_features(
        X_path, X_protein, X_protein_pos, X_codon_pos, X_amino_change, X_amino_change_flag)
    
    # テストデータの特徴量を更新
    print("[INFO] テストデータの特徴量を更新中...")
    for target_len, dataset_data in test_datasets_sampled.items():
        dataset_data['protein'], dataset_data['protein_pos'], dataset_data['codon_pos'], \
        dataset_data['amino_change'], dataset_data['amino_change_flag'] = update_features(
            dataset_data['path'], dataset_data['protein'], dataset_data['protein_pos'],
            dataset_data['codon_pos'], dataset_data['amino_change'], dataset_data['amino_change_flag'])
    
    # 訓練データの次元数確認
    print("\n訓練データの次元数:")
    print(f"X_path: {len(X_path)} サンプル, 各サンプル最大 {max(len(x) for x in X_path)} 要素")
    print(f"X_protein: {len(X_protein)} サンプル, 各サンプル最大 {max(len(x) for x in X_protein)} 要素")
    print(f"X_protein_pos: {len(X_protein_pos)} サンプル, 各サンプル最大 {max(len(x) for x in X_protein_pos)} 要素")
    print(f"X_codon_pos: {len(X_codon_pos)} サンプル, 各サンプル最大 {max(len(x) for x in X_codon_pos)} 要素")
    print(f"X_amino_change: {len(X_amino_change)} サンプル, 各サンプル最大 {max(len(x) for x in X_amino_change)} 要素")
    print(f"X_amino_change_flag: {len(X_amino_change_flag)} サンプル, 各サンプル最大 {max(len(x) for x in X_amino_change_flag)} 要素")
    print(f"Y: {len(Y)} サンプル")
    
    # シーケンス長の確認
    print("\nシーケンス長の確認:")
    path_lengths = [len(x) for x in X_path]
    print(f"X_path の長さ: 最小={min(path_lengths)}, 最大={max(path_lengths)}, 平均={sum(path_lengths)/len(path_lengths):.2f}")
    
    # 長さが30のデータを確認
    print("\n長さが30のデータの確認:")
    for i, length in enumerate(path_lengths):
        if length == 30:
            print(f"\nサンプル {i}:")
            print(f"変異パス (path): {X_path[i]}")
            print(f"タンパク質領域 (protein): {X_protein[i]}")
            print(f"タンパク質位置 (protein_pos): {X_protein_pos[i]}")
            print(f"コドン位置 (codon_pos): {X_codon_pos[i]}")
            print(f"アミノ酸変異 (amino_change): {X_amino_change[i]}")
            print(f"変異タイプ (amino_change_flag): {X_amino_change_flag[i]}")
            print(f"予測ラベル (Y): {Y[i]}")
    
    # パディング処理の修正
    def pad_sequence(seq, maxlen, pad_token="PAD"):
        if len(seq) > maxlen - 1:  # maxlen-1に修正
            return seq[-(maxlen-1):]  # 末尾からmaxlen-1個を取り出す
        return [pad_token] * ((maxlen-1) - len(seq)) + seq  # 先頭にパディング
    
    # 訓練データのパディング
    X_path = [pad_sequence(x, maxlen) for x in X_path]
    X_protein = [pad_sequence(x, maxlen) for x in X_protein]
    X_protein_pos = [pad_sequence(x, maxlen) for x in X_protein_pos]
    X_codon_pos = [pad_sequence(x, maxlen) for x in X_codon_pos]
    X_amino_change = [pad_sequence(x, maxlen) for x in X_amino_change]
    X_amino_change_flag = [pad_sequence(x, maxlen) for x in X_amino_change_flag]
    
    # パディング後の長さ確認
    print("\nパディング後のシーケンス長:")
    path_lengths = [len(x) for x in X_path]
    print(f"X_path の長さ: 最小={min(path_lengths)}, 最大={max(path_lengths)}, 平均={sum(path_lengths)/len(path_lengths):.2f}")
    
    # 訓練データの内容確認（最初の3サンプル）
    print("\n訓練データの内容（最初の3サンプル）:")
    for i in range(min(3, len(X_path))):
        print(f"\nサンプル {i+1}:")
        print(f"変異パス (path): {X_path[i]}")
        print(f"タンパク質領域 (protein): {X_protein[i]}")
        print(f"タンパク質位置 (protein_pos): {X_protein_pos[i]}")
        print(f"コドン位置 (codon_pos): {X_codon_pos[i]}")
        print(f"アミノ酸変異 (amino_change): {X_amino_change[i]}")
        print(f"変異タイプ (amino_change_flag): {X_amino_change_flag[i]}")
        print(f"予測ラベル (Y): {Y[i]}")
    
    # テストデータのパディングと次元数確認（各系統長ごと）
    print("\nテストデータの次元数（系統長ごと）:")
    for target_len, dataset_data in test_datasets_sampled.items():
        print(f"\n系統長 {target_len}:")
        # テストデータのパディング
        dataset_data['path'] = [pad_sequence(x, maxlen) for x in dataset_data['path']]
        dataset_data['protein'] = [pad_sequence(x, maxlen) for x in dataset_data['protein']]
        dataset_data['protein_pos'] = [pad_sequence(x, maxlen) for x in dataset_data['protein_pos']]
        dataset_data['codon_pos'] = [pad_sequence(x, maxlen) for x in dataset_data['codon_pos']]
        dataset_data['amino_change'] = [pad_sequence(x, maxlen) for x in dataset_data['amino_change']]
        dataset_data['amino_change_flag'] = [pad_sequence(x, maxlen) for x in dataset_data['amino_change_flag']]
        
        print(f"変異パス (path): {len(dataset_data['path'])} サンプル, 各サンプル最大 {max(len(x) for x in dataset_data['path'])} 要素")
        print(f"タンパク質領域 (protein): {len(dataset_data['protein'])} サンプル, 各サンプル最大 {max(len(x) for x in dataset_data['protein'])} 要素")
        print(f"タンパク質位置 (protein_pos): {len(dataset_data['protein_pos'])} サンプル, 各サンプル最大 {max(len(x) for x in dataset_data['protein_pos'])} 要素")
        print(f"コドン位置 (codon_pos): {len(dataset_data['codon_pos'])} サンプル, 各サンプル最大 {max(len(x) for x in dataset_data['codon_pos'])} 要素")
        print(f"アミノ酸変異 (amino_change): {len(dataset_data['amino_change'])} サンプル, 各サンプル最大 {max(len(x) for x in dataset_data['amino_change'])} 要素")
        print(f"変異タイプ (amino_change_flag): {len(dataset_data['amino_change_flag'])} サンプル, 各サンプル最大 {max(len(x) for x in dataset_data['amino_change_flag'])} 要素")
        print(f"予測ラベル (labels): {len(dataset_data['labels'])} サンプル")
        
        # 最初のサンプルの内容表示
        if len(dataset_data['path']) > 0:
            print(f"\n系統長 {target_len} の最初のサンプル:")
            print(f"変異パス (path): {dataset_data['path'][0]}")
            print(f"タンパク質領域 (protein): {dataset_data['protein'][0]}")
            print(f"タンパク質位置 (protein_pos): {dataset_data['protein_pos'][0]}")
            print(f"コドン位置 (codon_pos): {dataset_data['codon_pos'][0]}")
            print(f"アミノ酸変異 (amino_change): {dataset_data['amino_change'][0]}")
            print(f"変異タイプ (amino_change_flag): {dataset_data['amino_change_flag'][0]}")
            print(f"予測ラベル (label): {dataset_data['labels'][0]}")
    
    print("\n" + "="*60)
    
    train_path = encode_seqs([X_path[i] for i in train_idx], path_vocab)
    train_protein = encode_seqs([X_protein[i] for i in train_idx], protein_vocab)
    train_protein_pos = encode_seqs([X_protein_pos[i] for i in train_idx], protein_pos_vocab)
    train_codon_pos = encode_seqs([X_codon_pos[i] for i in train_idx], codon_pos_vocab)
    train_amino_change = encode_seqs([X_amino_change[i] for i in train_idx], amino_change_vocab)
    train_amino_change_flag = [X_amino_change_flag[i] for i in train_idx]
    train_label = [Y[i] for i in train_idx]
    train_label = [path_vocab[x] for x in train_label]
    
    # 検証データの準備（検証データがある場合のみ）
    if len(val_idx) > 0:
        val_path = encode_seqs([X_path[i] for i in val_idx], path_vocab)
        val_protein = encode_seqs([X_protein[i] for i in val_idx], protein_vocab)
        val_protein_pos = encode_seqs([X_protein_pos[i] for i in val_idx], protein_pos_vocab)
        val_codon_pos = encode_seqs([X_codon_pos[i] for i in val_idx], codon_pos_vocab)
        val_amino_change = encode_seqs([X_amino_change[i] for i in val_idx], amino_change_vocab)
        val_amino_change_flag = [X_amino_change_flag[i] for i in val_idx]
        val_label = [Y[i] for i in val_idx]
        val_label = [path_vocab[x] for x in val_label]
    else:
        # 検証データがない場合は空のリストを設定
        val_path, val_protein, val_protein_pos, val_codon_pos, val_amino_change = [], [], [], [], []
        val_amino_change_flag, val_label = [], []

    # テストデータのID変換は評価ループ内で行う（既存ロジックのまま）

    # 訓練データの準備
    train_amino_change_flag = [convert_amino_change_flag(flags) for flags in X_amino_change_flag]
    check_amino_change_flag(train_amino_change_flag, "訓練データ")

    # 検証データの準備
    if len(val_idx) > 0:
        val_amino_change_flag = [convert_amino_change_flag(flags) for flags in val_amino_change_flag]
        check_amino_change_flag(val_amino_change_flag, "検証データ")
    else:
        val_amino_change_flag = []
        print("検証データがないため、検証データのamino_change_flagチェックをスキップします。")

    # テストデータの準備
    for target_len, dataset_data in test_datasets_sampled.items():
        dataset_data['amino_change_flag'] = [
            convert_amino_change_flag(flags) 
            for flags in dataset_data['amino_change_flag']
        ]
        check_amino_change_flag(dataset_data['amino_change_flag'], f"テストデータ（長さ{target_len}）")

    # DataLoader
    train_dataset = MutationDatasetWithPathLast(train_path, train_protein, train_protein_pos, train_codon_pos, train_amino_change, train_amino_change_flag, train_label)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    
    # 検証データローダーの作成（検証データがある場合のみ）
    if len(val_idx) > 0:
        val_dataset = MutationDatasetWithPathLast(val_path, val_protein, val_protein_pos, val_codon_pos, val_amino_change, val_amino_change_flag, val_label)
        val_loader = DataLoader(val_dataset, batch_size=config['val_batch_size'], shuffle=False)
        print(f"[INFO] データセットサイズ:")
        print(f"  訓練データ: {len(train_dataset)} サンプル")
        print(f"  検証データ: {len(val_dataset)} サンプル")
        print(f"  訓練バッチ数: {len(train_loader)}")
        print(f"  検証バッチ数: {len(val_loader)}")
    else:
        val_loader = None
        print(f"[INFO] データセットサイズ:")
        print(f"  訓練データ: {len(train_dataset)} サンプル")
        print(f"  検証データ: 0 サンプル (検証なし)")
        print(f"  訓練バッチ数: {len(train_loader)}")
        print("[警告] 検証データが0件です。検証をスキップします。")
    vocab_sizes = {
        'path': len(path_vocab),
        'protein': len(protein_vocab),
        'protein_pos': len(protein_pos_vocab),
        'codon_pos': len(codon_pos_vocab),
        'amino_change': len(amino_change_vocab)
    }
    model = MutationTransformerWithPath(vocab_sizes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    vocab_dict = {
        'path_vocab': path_vocab,
        'protein_vocab': protein_vocab,
        'protein_pos_vocab': protein_pos_vocab,
        'codon_pos_vocab': codon_pos_vocab,
        'amino_change_vocab': amino_change_vocab
    }
    early_stopping = EarlyStopping(patience=config.get('early_stopping_patience', 5), verbose=True)
    log = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_top3': [], 'train_acc': [], 'train_top3': []}
    for epoch in range(config['num_epochs']):
        model.train()
        total_loss = 0
        total_correct1 = 0
        total_correct3 = 0
        total_count = 0
        for batch in tqdm(train_loader):
            optimizer.zero_grad()
            for k in batch:
                batch[k] = batch[k].to(device)
            output = model(batch['path'], batch['protein'], batch['protein_pos'], batch['codon_pos'], batch['amino_change'], batch['amino_change_flag'])
            target = batch['label']  # 末尾
            loss = criterion(output, target)
            pred1 = output.argmax(dim=-1)
            top3 = output.topk(3, dim=-1).indices
            correct1 = (pred1 == target)
            correct3 = torch.any(top3 == target.unsqueeze(1), dim=1)
            total_correct1 += correct1.sum().item()
            total_correct3 += correct3.sum().item()
            total_count += target.size(0)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_loss = total_loss / len(train_loader)
        train_acc = total_correct1 / total_count if total_count > 0 else 0
        train_top3 = total_correct3 / total_count if total_count > 0 else 0
        # 検証
        model.eval()
        total_loss = 0
        total_correct1 = 0
        total_correct3 = 0
        total_count = 0
        
        if val_loader is not None and len(val_loader) > 0:
            with torch.no_grad():
                for batch in val_loader:
                    for k in batch:
                        batch[k] = batch[k].to(device)
                    output = model(batch['path'], batch['protein'], batch['protein_pos'], batch['codon_pos'], batch['amino_change'], batch['amino_change_flag'])
                    target = batch['label']
                    loss = criterion(output, target)
                    pred1 = output.argmax(dim=-1)
                    top3 = output.topk(3, dim=-1).indices
                    correct1 = (pred1 == target)
                    correct3 = torch.any(top3 == target.unsqueeze(1), dim=1)
                    total_correct1 += correct1.sum().item()
                    total_correct3 += correct3.sum().item()
                    total_count += target.size(0)
                    total_loss += loss.item()
            val_loss = total_loss / len(val_loader)
            val_acc = total_correct1 / total_count if total_count > 0 else 0
            val_top3 = total_correct3 / total_count if total_count > 0 else 0
        else:
            print("[警告] 検証データがないため、検証をスキップします。")
            val_loss = float('inf')  # 検証データがない場合のデフォルト値
            val_acc = 0.0
            val_top3 = 0.0
        log['train_loss'].append(train_loss)
        log['val_loss'].append(val_loss)
        log['val_acc'].append(val_acc)
        log['val_top3'].append(val_top3)
        log['train_acc'].append(train_acc)
        log['train_top3'].append(train_top3)
        print(f"Epoch {epoch+1}/{config['num_epochs']} train_loss: {train_loss:.4f} val_loss: {val_loss:.4f} train_acc: {train_acc:.4f} val_acc: {val_acc:.4f} val_top3: {val_top3:.4f}")
        
        # Early stoppingは検証データがある場合のみ実行
        if val_loader is not None and len(val_loader) > 0:
            early_stopping(val_loss, model, os.path.join(dirs['model'], 'best_model.pt'))
            if early_stopping.early_stop:
                print("Early stopping!")
                break
        else:
            # 検証データがない場合は、定期的にモデルを保存
            if (epoch + 1) % 5 == 0:
                torch.save(model.state_dict(), os.path.join(dirs['model'], f'model_epoch_{epoch+1}.pt'))
                print(f"モデルを保存しました: model_epoch_{epoch+1}.pt")
    
    # 最終モデルの保存
    if val_loader is None or len(val_loader) == 0:
        torch.save(model.state_dict(), os.path.join(dirs['model'], 'final_model.pt'))
        print("最終モデルを保存しました: final_model.pt")
    
    with open(os.path.join(dirs['log'], 'train_log.json'), 'w') as f:
        import json
        json.dump(log, f, indent=2)
    # --- 追加: エポックごとの損失・精度をCSVでも保存 ---
    try:
        import pandas as pd
        log_df = pd.DataFrame({
            'epoch': list(range(1, len(log['train_loss'])+1)),
            'train_loss': log['train_loss'],
            'val_loss': log['val_loss'],
            'train_acc': log['train_acc'],
            'val_acc': log['val_acc'],
            'train_top3': log['train_top3'],
            'val_top3': log['val_top3']
        })
        log_df.to_csv(os.path.join(dirs['log'], 'train_log.csv'), index=False)
    except ImportError:
        pass
    if plt is not None:
        plt.figure()
        plt.plot(log['train_loss'], label="train_loss")
        plt.plot(log['val_loss'], label="val_loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Loss Curve (末尾予測)")
        plt.savefig(os.path.join(dirs['fig'], "loss_curve_laststep.png"))
        plt.close()
        plt.figure()
        plt.plot(log['val_acc'], label="val_acc (Top-1)")
        plt.plot(log['val_top3'], label="val_top3 (Top-3)")
        plt.plot(log['train_acc'], label="train_acc (Top-1)")
        plt.plot(log['train_top3'], label="train_top3 (Top-3)")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.title("Validation/Train Accuracy Curve (末尾予測)")
        plt.savefig(os.path.join(dirs['fig'], "val_acc_curve_laststep.png"))
        plt.close()
    with open(os.path.join(dirs['vocab'], "vocab.pkl"), "wb") as f:
        pickle.dump(vocab_dict, f)
    print(f"vocabを保存しました: {os.path.join(dirs['vocab'], 'vocab.pkl')}")
    print(f"最良モデル: {os.path.join(dirs['model'], 'best_model.pt')}")
    # --- 系統長ごとのテスト評価（新データ構造） ---
    print("\n[INFO] 系統長ごとのテスト評価開始...")
    
    test_acc1_bylen = {}
    test_acc3_bylen = {}
    test_base_before_acc_bylen = {}
    test_pos_acc_bylen = {}
    test_base_after_acc_bylen = {}
    test_n_bylen = {}
    test_base_before_corr_bylen = {}
    test_pos_corr_bylen = {}
    test_base_after_corr_bylen = {}
    
    for target_len, dataset_data in test_datasets_sampled.items():
        print(f"\n[評価] 系統長{target_len}のテストデータ評価中...")
        
        # データの取得
        test_X_path = dataset_data['path']
        test_X_protein = dataset_data['protein']
        test_X_protein_pos = dataset_data['protein_pos']
        test_X_codon_pos = dataset_data['codon_pos']
        test_X_amino_change = dataset_data['amino_change']
        test_X_amino_change_flag = dataset_data['amino_change_flag']
        test_Y = dataset_data['labels']
        test_Y_base_before = dataset_data['labels_base_before']
        test_Y_pos = dataset_data['labels_pos']
        test_Y_base_after = dataset_data['labels_base_after']
        
        if len(test_X_path) == 0:
            print(f"  スキップ: データが空です")
            continue
            
        print(f"  サンプル数: {len(test_X_path)}")
        print(f"  元データ: {dataset_data['info']['exact_count']}件")
        print(f"  切り取りデータ: {dataset_data['info']['trimmed_count']}件")
        
        # エンコード
        test_path = encode_seqs(test_X_path, path_vocab)
        test_protein = encode_seqs(test_X_protein, protein_vocab)
        test_protein_pos = encode_seqs(test_X_protein_pos, protein_pos_vocab)
        test_codon_pos = encode_seqs(test_X_codon_pos, codon_pos_vocab)
        test_amino_change = encode_seqs(test_X_amino_change, amino_change_vocab)
        test_amino_change_flag_encoded = encode_seqs(test_X_amino_change_flag, amino_change_flag_vocab)
        test_label = [path_vocab[x] for x in test_Y]
        
        # データセットとローダー作成
        test_dataset = MutationDatasetWithPathLast(test_path, test_protein, test_protein_pos, test_codon_pos, test_amino_change, test_amino_change_flag_encoded, test_label)
        test_loader = DataLoader(test_dataset, batch_size=config['val_batch_size'], shuffle=False)
        
        # 評価実行
        model.eval()
        total_correct1 = 0
        total_correct3 = 0
        total_count = 0
        # 部分ごとの正解率用
        total_base_before = 0
        total_pos = 0
        total_base_after = 0
        correct_base_before = 0
        correct_pos = 0
        correct_base_after = 0
        
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                for k in batch:
                    batch[k] = batch[k].to(device)
                output = model(batch['path'], batch['protein'], batch['protein_pos'], batch['codon_pos'], batch['amino_change'], batch['amino_change_flag'])
                target = batch['label']
                pred1 = output.argmax(dim=-1)
                top3 = output.topk(3, dim=-1).indices
                correct1 = (pred1 == target)
                correct3 = torch.any(top3 == target.unsqueeze(1), dim=1)
                total_correct1 += correct1.sum().item()
                total_correct3 += correct3.sum().item()
                total_count += target.size(0)
                
                # --- 部分ごとの正解率 ---
                for j in range(target.size(0)):
                    # 正解ラベル
                    true_label = test_Y[ i * config['val_batch_size'] + j ]
                    m_true = re.match(r"([A-Za-z]+)([0-9]+)([A-Za-z]+)", true_label)
                    if m_true:
                        true_b, true_p, true_a = m_true.group(1), m_true.group(2), m_true.group(3)
                    else:
                        continue
                    # 予測ラベル
                    pred_label_id = pred1[j].item()
                    pred_label = None
                    for k_, v_ in path_vocab.items():
                        if v_ == pred_label_id:
                            pred_label = k_
                            break
                    if pred_label is None:
                        continue
                    m_pred = re.match(r"([A-Za-z]+)([0-9]+)([A-Za-z]+)", pred_label)
                    if m_pred:
                        pred_b, pred_p, pred_a = m_pred.group(1), m_pred.group(2), m_pred.group(3)
                    else:
                        continue
                    total_base_before += 1
                    total_pos += 1
                    total_base_after += 1
                    if true_b == pred_b:
                        correct_base_before += 1
                    if true_p == pred_p:
                        correct_pos += 1
                    if true_a == pred_a:
                        correct_base_after += 1
        
        # 精度計算
        acc1 = total_correct1 / total_count if total_count > 0 else 0
        acc3 = total_correct3 / total_count if total_count > 0 else 0
        acc_base_before = correct_base_before / total_base_before if total_base_before > 0 else 0
        acc_pos = correct_pos / total_pos if total_pos > 0 else 0
        acc_base_after = correct_base_after / total_base_after if total_base_after > 0 else 0
        
        # 結果保存
        test_acc1_bylen[target_len] = acc1
        test_acc3_bylen[target_len] = acc3
        test_base_before_acc_bylen[target_len] = acc_base_before
        test_pos_acc_bylen[target_len] = acc_pos
        test_base_after_acc_bylen[target_len] = acc_base_after
        test_n_bylen[target_len] = total_count
        test_base_before_corr_bylen[target_len] = (correct_base_before, total_base_before)
        test_pos_corr_bylen[target_len] = (correct_pos, total_pos)
        test_base_after_corr_bylen[target_len] = (correct_base_after, total_base_after)
        
        print(f"[Test] length={target_len}  Top-1 acc={acc1:.4f}  Top-3 acc={acc3:.4f}  base_before_acc={acc_base_before:.4f}  pos_acc={acc_pos:.4f}  base_after_acc={acc_base_after:.4f}  n={total_count}")
    
    print(f"\n[INFO] 全テスト評価完了")
    # --- 系統長ごとのテスト精度をCSVとグラフで保存 ---
    try:
        import pandas as pd
        test_acc_df = pd.DataFrame({
            'length': list(test_acc1_bylen.keys()),
            'top1_acc': list(test_acc1_bylen.values()),
            'top3_acc': list(test_acc3_bylen.values()),
            'base_before_acc': list(test_base_before_acc_bylen.values()),
            'pos_acc': list(test_pos_acc_bylen.values()),
            'base_after_acc': list(test_base_after_acc_bylen.values()),
            'n': [test_n_bylen[target_len] for target_len in test_acc1_bylen.keys()],
            'base_before_corr': [f"{test_base_before_corr_bylen[target_len][0]}/{test_base_before_corr_bylen[target_len][1]}" for target_len in test_acc1_bylen.keys()],
            'pos_corr': [f"{test_pos_corr_bylen[target_len][0]}/{test_pos_corr_bylen[target_len][1]}" for target_len in test_acc1_bylen.keys()],
            'base_after_corr': [f"{test_base_after_corr_bylen[target_len][0]}/{test_base_after_corr_bylen[target_len][1]}" for target_len in test_acc1_bylen.keys()]
        })
        test_acc_csv_path = os.path.join(dirs['log'], 'test_acc_by_length.csv')
        test_acc_df.to_csv(test_acc_csv_path, index=False)
        print(f"[INFO] 系列長ごとのテスト精度を {test_acc_csv_path} に保存しました")
        if plt is not None:
            plt.figure(figsize=(8,6))
            ax1 = plt.gca()
            ax2 = ax1.twinx()
            ax1.plot(test_acc_df['length'], test_acc_df['top1_acc'], label='Top-1 acc', color='C0')
            ax1.plot(test_acc_df['length'], test_acc_df['top3_acc'], label='Top-3 acc', color='C1')
            ax1.plot(test_acc_df['length'], test_acc_df['base_before_acc'], label='base_before_acc', color='C2')
            ax1.plot(test_acc_df['length'], test_acc_df['pos_acc'], label='pos_acc', color='C3')
            ax1.plot(test_acc_df['length'], test_acc_df['base_after_acc'], label='base_after_acc', color='C4')
            ax2.bar(test_acc_df['length'], test_acc_df['n'], alpha=0.2, color='gray', label='n (sample count)')
            ax1.set_xlabel('Sequence Length')
            ax1.set_ylabel('Accuracy')
            ax2.set_ylabel('Sample count (n)')
            ax1.set_title('Test Accuracy by Sequence Length')
            ax1.legend(loc='upper left')
            ax2.legend(loc='upper right')
            plt.grid(True)
            test_acc_fig_path = os.path.join(dirs['fig'], 'test_acc_by_length.png')
            plt.savefig(test_acc_fig_path)
            plt.close()
            print(f"[INFO] 系列長ごとのテスト精度グラフを {test_acc_fig_path} に保存しました")
    except ImportError:
        print("[WARN] pandasがインストールされていないため、CSV保存・グラフ描画をスキップします。")

    # --- summary.txtにamino_change_flag_vocabも明記 ---
    summary_path = os.path.join(dirs['log'], 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"path_vocab: {len(path_vocab)}\n")
        f.write(f"protein_vocab: {len(protein_vocab)}\n")
        f.write(f"protein_pos_vocab: {len(protein_pos_vocab)}\n")
        f.write(f"codon_pos_vocab: {len(codon_pos_vocab)}\n")
        f.write(f"amino_change_vocab: {len(amino_change_vocab)}\n")
        f.write(f"amino_change_flag_vocab: {amino_change_flag_vocab}\n")
        f.write(f"train+val: {len(train_idx)+len(val_idx)}\n")
        f.write(f"test: {sum([test_n_bylen[l] for l in test_n_bylen])}\n")
        f.write(f"config: {config}\n")
    print(f"[INFO] summary.txtを {summary_path} に保存しました")

    # --- train_log.csvも保存（loss/acc曲線のCSV） ---
    train_log_csv_path = os.path.join(dirs['log'], 'train_log.csv')
    import csv
    with open(train_log_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_acc', 'val_top3'])
        for i in range(len(log['train_loss'])):
            writer.writerow([i+1, log['train_loss'][i], log['val_loss'][i], log['val_acc'][i], log['val_top3'][i]])
    print(f"[INFO] 訓練・検証のloss/acc曲線を {train_log_csv_path} に保存しました")

if __name__ == "__main__":
    main()
