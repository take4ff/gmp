from sklearn.model_selection import train_test_split
import module.input_mutation_path as imp # 外部モジュール
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import os
from Bio import SeqIO # pip install biopython
import torch.optim as optim
from datetime import datetime
import time
import sys

# モデル保存用のディレクトリとパス
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
folder_name = "../model/20250719_learn2/"
MODEL_SAVE_DIR = os.path.join(folder_name, current_time)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
BEST_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, 'best_model.pth')

# === 1. 設定パラメータ ===
data_config = {
    'train_end': 45,
    'test_start': 40,
    'ylen': 1, # 予測する未来のステップ数（通常は1）
    'val_ratio': 0.2, # 検証セットの割合
    'mix_ratio': 0.2, # train_endとtest_startが重複する範囲でのテスト分割割合
    'frag_len': 10, # 変異履歴のフラグメント長（モデル入力の時系列長）
    'max_co_occur': 20, # 各変異ステップで許容する最大共起変異数
    'nmax': 100000000, # 全データから取得する最大パス数
    'nmax_per_strain': 1000000 # 各株から取得する最大パス数
}

dataset_config = {
    'strains': ['B.1.1.7','P.1','BA.2','BA.1.1','BA.1','B.1.617.2','B.1.351','B.1.1.529'],
    'usher_dir': '../usher_output/', # Usherデータのディレクトリ
    'bunpu_csv': "table_heatmap/250621/table_set/table_set.csv", # 分布CSVパス
    'codon_csv': 'meta_data/codon_mutation4.csv', # コドン変異CSVパス
    'cache_dir': '../cache', # 特徴データキャッシュ用ディレクトリ
}

model_config = {
    'epochs': 30,
    'batch_size': 128,
    'embed_dim': 64, # 埋め込み次元
    'sequence_length': 30000, # ゲノム長（NC_045512.2の長さは約29903bp）
    'vocab_size': 4, # 塩基A,C,G,T（パディングは別途扱う）
    'num_heads': 8, # Transformerのヘッド数
    'num_encoder_layers': 4, # TransformerEncoderの層数
    'dropout': 0.1,
    'lr': 0.001,
    'weight_decay': 0.01 # AdamWのウェイト減衰強度
}

# --- 2. グローバル定数とゲノム配列の読み込み ---
BASE_TO_INT = {'A': 0, 'T': 1, 'C': 2, 'G': 3, '<PAD_BASE>': 4}
INT_TO_BASE = {v: k for k, v in BASE_TO_INT.items()}
PAD_BASE = BASE_TO_INT['<PAD_BASE>']
# 位置のパディング値はゲノム長+1 (0-indexed position + 1 for padding token)
PAD_POS = model_config['sequence_length'] + 1 

GENOME_FASTA_PATH = 'meta_data/NC_045512.2.fasta' # 実際のパスに合わせてください

def force_print(message):
    """タイムスタンプ付きで強制出力"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")
    sys.stdout.flush()

def load_and_tokenize_genome(fasta_path, base_to_int_map, target_length):
    genome_sequence = ""
    try:
        # Biopythonを使用してFASTAファイルをパース
        for record in SeqIO.parse(fasta_path, "fasta"):
            genome_sequence = str(record.seq).upper()
            break # 最初のシーケンスのみを使用
    except Exception as e:
        print(f"Error reading FASTA file {fasta_path}: {e}. Trying fallback.")
        # Biopythonがない場合や問題がある場合のフォールバック（簡易版）
        with open(fasta_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if not line.startswith('>'):
                    genome_sequence += line.strip()
        genome_sequence = genome_sequence.upper()

    if not genome_sequence:
        raise ValueError(f"No sequence found in FASTA file: {fasta_path}")

    # 塩基配列を数値インデックスに変換
    tokenized_genome = [base_to_int_map.get(base, base_to_int_map['<PAD_BASE>']) for base in genome_sequence]
    genome_tensor = torch.tensor(tokenized_genome, dtype=torch.long)

    # ゲノム長が target_length と一致しない場合の処理（切り捨て/パディング）
    if len(genome_tensor) != target_length:
        print(f"WARNING: Genome length ({len(genome_tensor)}) does not match target_length ({target_length}). Truncating/padding.")
        if len(genome_tensor) > target_length:
            genome_tensor = genome_tensor[:target_length]
        else:
            padding = torch.full((target_length - len(genome_tensor),), PAD_BASE, dtype=torch.long)
            genome_tensor = torch.cat([genome_tensor, padding])
            
    return genome_tensor

# ゲノム配列をロードし、固定のテンソルとして保持 (すべてのサンプルで共有)
try:
    GLOBAL_GENOME_TENSOR = load_and_tokenize_genome(GENOME_FASTA_PATH, BASE_TO_INT, model_config['sequence_length'])
    print(f"Loaded genome sequence tensor of shape: {GLOBAL_GENOME_TENSOR.shape}")
except (FileNotFoundError, ValueError) as e:
    print(f"CRITICAL ERROR: Could not load genome sequence: {e}")
    print("Please ensure 'meta_data/NC_045512.2.fasta' exists and contains valid FASTA data.")
    GLOBAL_GENOME_TENSOR = torch.full((model_config['sequence_length'],), PAD_BASE, dtype=torch.long) # エラー時のダミーデータ

# --- 3. データ前処理関数群 ---
def filter_co_occur(data,sample_name,data_len,max_co_occur,out_num=None):
    filted_data = []
    filted_sample_name =[]
    filted_data_len = []
    for i in range(len(data)):
        compare = 0
        for j in range(len(data[i])):
            mutation = data[i][j].split(',')
            if(compare < len(mutation)):
                compare = len(mutation)
        if(compare <= max_co_occur):
            filted_data.append(data[i])
            filted_sample_name.append(sample_name[i])
            filted_data_len.append(data_len[i])
        if(out_num is not None and len(filted_data)>=out_num):
            break
    return filted_data,filted_sample_name,filted_data_len

def unique_path(data):
    return [list(item) for item in dict.fromkeys(tuple(path) for path in data)]

def data_by_ts(data):
    data_ts = {}
    for i in range(len(data)):
        length = len(data[i])
        if(data_ts.get(length) is None):
            data_ts[length] = []
        data_ts[length].append(data[i])
    return data_ts

def fragmentation(data,frag_len,end_opt=False):
    frag_data = []
    for i in range(len(data)):
        if end_opt:
            start = len(data[i])-frag_len
        else:
            start = 0
        for j in range(start,len(data[i])-frag_len+1):
            frag_data.append(data[i][j:j+frag_len])
    return frag_data

def separate_XY(paths,ylen):
    X = []
    Y = []
    for path in paths:
        X.append(path[:-ylen])
        Y.append(path[-ylen:])
    return X,Y

def separete_HGVS(hgvs_paths):
    new_hgvs_paths = []
    for hgvs_path in hgvs_paths:
        new_hgvs_path = []
        for hgvss in hgvs_path:
            new_hgvs = []
            for hgvs in hgvss.split(','):
                new_hgvs.append([hgvs[0],hgvs[1:-1],hgvs[-1]])
            new_hgvs_path.append(new_hgvs)
        new_hgvs_paths.append(new_hgvs_path)
    return new_hgvs_paths

def dataset_by_ts(data, train_end, test_start, mix_ratio=0.2, val_ratio=0.2, frag_len=None, unique=False, ylen=1):
    train = []
    test = {}
    data_ts = data_by_ts(data)
    keys = sorted(list(data_ts.keys()))
    print("keys:",keys)

    if train_end >= test_start:
        train_test_mix = [test_start, train_end]
    else:
        train_test_mix = None
    
    if train_test_mix is None:
        for k in keys:
            items = data_ts[k]
            if k <= train_end:
                train.extend(items)
            if test_start <= k:
                test[k] = items.copy()
    else:
        for k in keys:
            items = data_ts[k]
            if k < test_start:
                train.extend(items)
            elif train_end < k:
                test[k] = items.copy()
            else:
                train_temp, test[k] = train_test_split(items, test_size=mix_ratio)
                train.extend(train_temp)
    if frag_len is not None:
        train, valid = train_test_split(train, test_size=val_ratio)
        train = fragmentation(train,frag_len=frag_len)
        valid = fragmentation(valid,frag_len=frag_len,end_opt=True)
        for k in sorted(list(test.keys())):
            test[k] = fragmentation(test[k],frag_len=frag_len,end_opt=True)
    
    if unique:
        train = unique_path(train)
        valid = unique_path(valid)
        for k in sorted(list(test.keys())):
            test[k] = unique_path(test[k])
    
    train_x,train_y = separate_XY(train,ylen)
    valid_x,valid_y = separate_XY(valid,ylen)
    test_x = {}
    test_y = {}
    for k in sorted(list(test.keys())):
        tx,ty = separate_XY(test[k],ylen)
        test_x[k] = tx
        test_y[k] = ty

    train_x = separete_HGVS(train_x)
    train_y = separete_HGVS(train_y)
    valid_x = separete_HGVS(valid_x)
    valid_y = separete_HGVS(valid_y)
    for k in sorted(list(test.keys())):
        test_x[k] = separete_HGVS(test_x[k])
        test_y[k] = separete_HGVS(test_y[k])

    return train_x,train_y, valid_x,valid_y, test_x,test_y

# --- `add_x_by_y` 関数 (訓練/検証データで共起変異を単一ターゲットに分解) ---
def add_x_by_y(x, y):
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    
    new_x, new_y = [], []
    for i in range(len(y)):
        for fea in y[i]:
            new_x.append(x[i])
            new_y.append(fea) 
    return new_x, new_y

# --- 4. データロードと前処理の実行 ---
names, lengths, base_HGVS_paths = imp.input(
dataset_config['strains'], 
dataset_config['usher_dir'], 
nmax=data_config['nmax'], 
nmax_per_strain=data_config['nmax_per_strain']
)
print("元データ",len(base_HGVS_paths))

filted_data, temp, temp = filter_co_occur(base_HGVS_paths,names,lengths,data_config['max_co_occur'])
print("共起フィルタ",len(filted_data))

data = unique_path(filted_data)
print("ユニークパス",len(data))

train_x,train_y, valid_x,valid_y, test_x,test_y = dataset_by_ts(data,train_end=data_config['train_end'],test_start=data_config['test_start'],mix_ratio=data_config['mix_ratio'],
                                                               val_ratio=data_config['val_ratio'],frag_len=data_config['frag_len'],unique=True,ylen=data_config['ylen'])

train_x_split, train_y_split = add_x_by_y(train_x, train_y)
valid_x_split, valid_y_split = add_x_by_y(valid_x, valid_y)

# === 5. PyTorchモデル構築・学習用コード ===

# --- カスタムデータセットクラス ---
class MutationPathDataset(Dataset):
    def __init__(self, X_data, Y_data, sequence_length, max_co_occur, frag_len, is_test_data=False, genome_tensor=None):
        self.X_data = X_data
        self.Y_data = Y_data 
        self.sequence_length = sequence_length
        self.max_co_occur = max_co_occur
        self.frag_len = frag_len 
        self.is_test_data = is_test_data
        
        if genome_tensor is None:
            raise ValueError("genome_tensor must be provided to MutationPathDataset.")
        self.genome_tensor = genome_tensor 

        self.processed_X = self._process_hgvs_paths_X(self.X_data)
        self.processed_Y = self._process_hgvs_paths_Y(self.Y_data, is_test_data)

    def _hgvs_to_int(self, mutation_list_input):
        """
        ['C', '200', 'G'] -> [2, 200, 3]
        この関数は、引数が ['塩基', '位置', '塩基'] の形式であることを期待します。
        データが [['C', '15279', 'T']] のようにネストされている場合にも対応します。
        """
        if isinstance(mutation_list_input[0], list) and len(mutation_list_input[0]) == 3 and isinstance(mutation_list_input[0][0], str):
            # print(f"WARNING: _hgvs_to_int received nested list: {mutation_list_input}. Attempting to flatten.") # デバッグ用
            mutation_list_input = mutation_list_input[0]

        if not all(isinstance(x, str) for x in mutation_list_input[:3]):
             print(f"ERROR: _hgvs_to_int received non-string elements or incorrect format: {mutation_list_input}")
             raise TypeError("Expected string elements for base and position in _hgvs_to_int")

        original_base_int = BASE_TO_INT[mutation_list_input[0]]
        position_int = int(mutation_list_input[1])
        new_base_int = BASE_TO_INT[mutation_list_input[2]]
        return [original_base_int, position_int, new_base_int]

    def _process_hgvs_paths_X(self, hgvs_paths_list):
        processed_tensors = []
        for path in hgvs_paths_list:
            step_tensors = []
            for step_mutations in path:
                co_mut_tensors = []
                for mutation_str_list in step_mutations:
                    co_mut_tensors.append(self._hgvs_to_int(mutation_str_list))
                
                while len(co_mut_tensors) < self.max_co_occur:
                    co_mut_tensors.append([PAD_BASE, PAD_POS, PAD_BASE])
                
                step_tensors.append(co_mut_tensors[:self.max_co_occur])

            while len(step_tensors) < self.frag_len:
                step_tensors.append([[PAD_BASE, PAD_POS, PAD_BASE]] * self.max_co_occur)
            
            processed_tensors.append(torch.tensor(step_tensors, dtype=torch.long))
        return processed_tensors

    def _process_hgvs_paths_Y(self, hgvs_paths_list, is_test_data):
        processed_tensors = []
        for y_element in hgvs_paths_list:
            if is_test_data:
                temp_list = [self._hgvs_to_int(m) for m in y_element]
                while len(temp_list) < self.max_co_occur:
                     temp_list.append([PAD_BASE, PAD_POS, PAD_BASE])
                processed_tensors.append(torch.tensor(temp_list[:self.max_co_occur], dtype=torch.long))
            else: 
                processed_tensors.append(torch.tensor(self._hgvs_to_int(y_element), dtype=torch.long))
        return processed_tensors

    def __len__(self):
        return len(self.X_data) 

    def __getitem__(self, idx):
        return self.processed_X[idx], self.processed_Y[idx], self.genome_tensor

# --- DataLoaderのcollate_fn ---
def custom_collate_fn(batch):
    X_batch = torch.stack([s[0] for s in batch])
    Y_batch_samples = [s[1] for s in batch]
    # ゲノムテンソルはバッチ内の全サンプルで共通なので、最初のサンプルのものを複製して使用
    genome_batch = batch[0][2].unsqueeze(0).repeat(len(batch), 1) 

    if len(Y_batch_samples[0].shape) > 1: # テストデータのY: (MAX_COOCCURRING_MUTATIONS, 3)
        Y_batch = torch.stack(Y_batch_samples)
    else: # 訓練/検証データのY: (3,)
        Y_batch = torch.stack(Y_batch_samples)
    
    return X_batch, Y_batch, genome_batch

# --- 6. Positional Encoding クラス (Transformer用) ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # peの形状は (max_len, d_model) のままにして、forwardでunsqueeze(0)して加算する。
        # または、Transformerの入力に合わせて (1, max_len, d_model)
        self.register_buffer('pe', pe.unsqueeze(0)) # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is (batch_size, seq_len, d_model) for batch_first=True Transformer
        # PEは (1, seq_len, d_model) なので、ブロードキャストされる
        # x.size(1) は seq_len
        return x + self.pe[:, :x.size(1), :]
        
# --- 7. Mutation Predictor モデル ---
class MutationPredictor(nn.Module):
    def __init__(self, embed_dim, vocab_size, sequence_length, max_co_occur, num_heads, num_encoder_layers, dropout_rate=0.1):
        super(MutationPredictor, self).__init__()
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.max_co_occur = max_co_occur

        # 変異履歴処理部分
        self.original_base_embedding = nn.Embedding(vocab_size + 1, embed_dim)
        self.position_embedding = nn.Embedding(sequence_length + 2, embed_dim)
        self.new_base_embedding = nn.Embedding(vocab_size + 1, embed_dim)
        self.mutation_embedding_projection = nn.Linear(embed_dim * 3, embed_dim)
        # 履歴のPositionalEncodingは、`frag_len` (時系列のステップ数) に適用
        self.history_positional_encoding = PositionalEncoding(embed_dim, max_len=data_config['frag_len']) 
        
        # TransformerEncoderLayer に batch_first=True が設定されていることを確認
        self.history_encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout_rate, batch_first=True)
        # TransformerEncoder に batch_first=True が設定されていることを確認
        self.transformer_encoder = nn.TransformerEncoder(self.history_encoder_layer, num_layers=num_encoder_layers)
        # enable_nested_tensor=False は一部のPyTorchバージョンでbatch_first=Trueと併用する際のトラブルシューティングになります。

        # 初期ゲノム配列処理部分
        self.genome_base_embedding = nn.Embedding(vocab_size + 1, embed_dim) 
        self.context_window_size = 64 # 予測位置の周辺ゲノムコンテキストサイズ
        
        # ローカルゲノムコンテキスト処理用のエンコーダ (Conv1D)
        self.local_genome_context_encoder = nn.Sequential(
            nn.Conv1d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1) # 最終的に1つのベクトルに集約
        )

        # ゲノム情報と変異履歴情報の統合レイヤー
        self.integration_layer = nn.Linear(embed_dim * 2, embed_dim) 

        # 予測ヘッド
        self.predicted_original_base_head = nn.Linear(embed_dim, vocab_size)
        self.position_head = nn.Linear(embed_dim, sequence_length + 1)
        self.predicted_new_base_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, mutation_history_input, genome_input, target_pos=None):
        # 変異履歴処理
        original_base_indices = mutation_history_input[..., 0]
        position_indices = mutation_history_input[..., 1]
        new_base_indices = mutation_history_input[..., 2]

        original_emb = self.original_base_embedding(original_base_indices)
        position_emb = self.position_embedding(position_indices)
        new_base_emb = self.new_base_embedding(new_base_indices)

        single_mutation_embedding = torch.cat([original_emb, position_emb, new_base_emb], dim=-1)
        single_mutation_embedding = self.mutation_embedding_projection(single_mutation_embedding)
        
        co_occurrence_mask = (original_base_indices == PAD_BASE).float()
        sum_embeddings = (single_mutation_embedding * co_occurrence_mask.unsqueeze(-1)).sum(dim=-2)
        count_embeddings = co_occurrence_mask.sum(dim=-1).unsqueeze(-1)
        co_mutation_context = sum_embeddings / (count_embeddings + 1e-9)

        # 履歴のPositional Encoding適用 (出力形状は (batch_size, seq_len, embed_dim) のまま)
        co_mutation_context = self.history_positional_encoding(co_mutation_context) # ここは変更なし
        
        # ... (src_key_padding_mask の計算) ...
        # src_key_padding_mask の形状は既に (batch_size, MAX_MUTATION_STEPS)
        # これが TransformerEncoder の batch_first=True の期待する形状
        is_padded_co_mutation = (original_base_indices == PAD_BASE)
        src_key_padding_mask = is_padded_co_mutation.all(dim=-1) # (batch_size, frag_len)

        # TransformerEncoder にマスクを渡す
        history_encoder_output = self.transformer_encoder(co_mutation_context, src_key_padding_mask=src_key_padding_mask)
        # 出力形状は (batch_size, frag_len, embed_dim)
        # PE適用後に再びpermuteして入力に合うようにする
        last_history_output = history_encoder_output[:, -1, :] # 最後のステップの出力を取得

        # ゲノム配列処理
        if target_pos is not None: # 訓練/検証時: ターゲット位置が与えられる
            local_genome_tensors = []
            half_window = self.context_window_size // 2
            for i in range(genome_input.shape[0]):
                center_pos = target_pos[i].item()
                start_idx = max(0, center_pos - half_window)
                end_idx = min(self.sequence_length, center_pos + half_window) # minはゲノム長を超えないように
                local_seq = genome_input[i, start_idx:end_idx]
                
                # コンテキストウィンドウサイズに合わせてパディング/切り捨て
                if len(local_seq) < self.context_window_size:
                    pad_len = self.context_window_size - len(local_seq)
                    local_seq = torch.cat([local_seq, torch.full((pad_len,), PAD_BASE, dtype=torch.long, device=local_seq.device)])
                elif len(local_seq) > self.context_window_size:
                    local_seq = local_seq[:self.context_window_size] # 切り捨て
                
                local_genome_tensors.append(local_seq)
            
            local_genome_batch = torch.stack(local_genome_tensors) # (batch_size, context_window_size)
            
            genome_emb_local = self.genome_base_embedding(local_genome_batch) # (batch_size, context_window_size, embed_dim)
            genome_emb_local = genome_emb_local.permute(0, 2, 1) # (batch_size, embed_dim, context_window_size) for Conv1d
            local_genome_summary_vector = self.local_genome_context_encoder(genome_emb_local).squeeze(-1) # (batch_size, embed_dim)

        else: # 推論時: target_posがNoneの場合 (ゲノム全体の平均を使用)
            genome_emb_full = self.genome_base_embedding(genome_input) # (batch_size, sequence_length, embed_dim)
            local_genome_summary_vector = genome_emb_full.mean(dim=1) # (batch_size, embed_dim)

        # ゲノム情報と変異履歴情報の統合
        combined_features = torch.cat([last_history_output, local_genome_summary_vector], dim=-1)
        final_prediction_input = self.integration_layer(combined_features)

        # 予測ヘッド
        predicted_original_base_logits = self.predicted_original_base_head(final_prediction_input)
        predicted_position_logits = self.position_head(final_prediction_input)
        predicted_new_base_logits = self.predicted_new_base_head(final_prediction_input)

        return predicted_original_base_logits, predicted_position_logits, predicted_new_base_logits

# --- 8. DataLoaderのcollate_fn ---
def custom_collate_fn(batch):
    X_batch = torch.stack([s[0] for s in batch])
    Y_batch_samples = [s[1] for s in batch]
    genome_batch = batch[0][2].unsqueeze(0).repeat(len(batch), 1)

    if len(Y_batch_samples[0].shape) > 1: # テストデータのY: (MAX_COOCCURRING_MUTATIONS, 3)
        Y_batch = torch.stack(Y_batch_samples)
    else: # 訓練/検証データのY: (3,)
        Y_batch = torch.stack(Y_batch_samples)
    
    return X_batch, Y_batch, genome_batch

# --- 9. データセットとデータローダーの作成 ---
train_dataset = MutationPathDataset(
    train_x_split, train_y_split, 
    model_config['sequence_length'], data_config['max_co_occur'], data_config['frag_len'],
    is_test_data=False, genome_tensor=GLOBAL_GENOME_TENSOR
)
val_dataset = MutationPathDataset(
    valid_x_split, valid_y_split, 
    model_config['sequence_length'], data_config['max_co_occur'], data_config['frag_len'],
    is_test_data=False, genome_tensor=GLOBAL_GENOME_TENSOR
)

test_dataloaders = {}
for k, v in test_x.items():
    test_dataset = MutationPathDataset(
        v, test_y[k], 
        model_config['sequence_length'], data_config['max_co_occur'], data_config['frag_len'],
        is_test_data=True, genome_tensor=GLOBAL_GENOME_TENSOR
    )
    test_dataloaders[k] = DataLoader(test_dataset, batch_size=model_config['batch_size'], shuffle=False, collate_fn=custom_collate_fn)

train_dataloader = DataLoader(train_dataset, batch_size=model_config['batch_size'], shuffle=True, collate_fn=custom_collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=model_config['batch_size'], shuffle=False, collate_fn=custom_collate_fn)

# --- 10. モデルのインスタンス化 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MutationPredictor(
    embed_dim=model_config['embed_dim'],
    vocab_size=model_config['vocab_size'],
    sequence_length=model_config['sequence_length'],
    max_co_occur=data_config['max_co_occur'],
    num_heads=model_config['num_heads'],
    num_encoder_layers=model_config['num_encoder_layers'],
    dropout_rate=model_config['dropout']
).to(device)

# --- 11. 損失関数と最適化手法 ---
criterion_orig = nn.CrossEntropyLoss(ignore_index=PAD_BASE)
criterion_pos = nn.CrossEntropyLoss(ignore_index=PAD_POS)
criterion_new = nn.CrossEntropyLoss(ignore_index=PAD_BASE) 

optimizer = optim.AdamW(model.parameters(), lr=model_config['lr'], weight_decay=model_config['weight_decay'])

# --- 12. 学習ループ ---
def train_model(model, train_loader, val_loader, criterion_orig, criterion_pos, criterion_new, optimizer, epochs, device, model_save_path):
    best_val_pos_acc = -1.0 # 最良の検証位置精度を追跡
    
    for epoch in range(epochs):
        epoch_start = time.time()  # エポック開始時刻
        model.train()
        total_loss = 0
        for batch_idx, (X_batch, Y_batch, genome_batch) in enumerate(train_loader):
            X_batch = X_batch.to(device)
            genome_batch = genome_batch.to(device)

            target_original_base = Y_batch[:, 0].to(device)
            target_position = Y_batch[:, 1].to(device)
            target_new_base = Y_batch[:, 2].to(device)

            optimizer.zero_grad()
            pred_orig_logits, pred_pos_logits, pred_new_logits = model(X_batch, genome_batch, target_pos=target_position)

            loss_orig = criterion_orig(pred_orig_logits, target_original_base)
            loss_pos = criterion_pos(pred_pos_logits, target_position)
            loss_new = criterion_new(pred_new_logits, target_new_base)
            
            loss = loss_orig + loss_pos + loss_new
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        avg_train_loss = total_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        correct_orig = 0; total_orig = 0
        correct_pos = 0; total_pos = 0
        correct_new = 0; total_new = 0

        with torch.no_grad():
            for X_batch, Y_batch, genome_batch in val_loader:
                X_batch = X_batch.to(device)
                genome_batch = genome_batch.to(device)

                target_original_base = Y_batch[:, 0].to(device)
                target_position = Y_batch[:, 1].to(device)
                target_new_base = Y_batch[:, 2].to(device)

                pred_orig_logits, pred_pos_logits, pred_new_logits = model(X_batch, genome_batch, target_pos=target_position)

                loss_orig = criterion_orig(pred_orig_logits, target_original_base)
                loss_pos = criterion_pos(pred_pos_logits, target_position)
                loss_new = criterion_new(pred_new_logits, target_new_base)
                val_loss += (loss_orig + loss_pos + loss_new).item()

                _, predicted_orig = torch.max(pred_orig_logits, 1)
                valid_orig_mask = (target_original_base != PAD_BASE)
                correct_orig += (predicted_orig[valid_orig_mask] == target_original_base[valid_orig_mask]).sum().item()
                total_orig += valid_orig_mask.sum().item()

                _, predicted_pos = torch.max(pred_pos_logits, 1)
                valid_pos_mask = (target_position != PAD_POS)
                correct_pos += (predicted_pos[valid_pos_mask] == target_position[valid_pos_mask]).sum().item()
                total_pos += valid_pos_mask.sum().item()
                
                _, predicted_new = torch.max(pred_new_logits, 1)
                valid_new_mask = (target_new_base != PAD_BASE)
                correct_new += (predicted_new[valid_new_mask] == target_new_base[valid_new_mask]).sum().item()
                total_new += valid_new_mask.sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_orig_acc = correct_orig / total_orig if total_orig > 0 else 0
        val_pos_acc = correct_pos / total_pos if total_pos > 0 else 0
        val_new_acc = correct_new / total_new if total_new > 0 else 0

        force_print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
              f"Val Orig Acc: {val_orig_acc:.4f}, Val Pos Acc: {val_pos_acc:.4f}, Val New Acc: {val_new_acc:.4f}")

        # 最良モデルの保存ロジック (検証位置精度に基づいて)
        if val_pos_acc > best_val_pos_acc:
            best_val_pos_acc = val_pos_acc
            torch.save(model.state_dict(), model_save_path)
            force_print(f"--- Epoch {epoch+1}: Validation Position Accuracy improved to {best_val_pos_acc:.4f}. Saving model to {model_save_path} ---")
        epoch_end = time.time()  # エポック終了時刻
        force_print(f"--- Epoch {epoch+1} took {epoch_end - epoch_start:.2f} seconds ---")

# --- 学習の実行 ---
print("--- PyTorchモデル学習開始 ---")
train_model(model, train_dataloader, val_dataloader, criterion_orig, criterion_pos, criterion_new, optimizer, model_config['epochs'], device, BEST_MODEL_PATH)

# --- 13. テストデータでの共起ラベル評価 ---
def evaluate_multi_label(model, test_dataloaders, device, model_path_to_load):
    # 最良モデルの重みをロード
    if os.path.exists(model_path_to_load):
        model.load_state_dict(torch.load(model_path_to_load, map_location=device))
        print(f"--- Loaded best model from {model_path_to_load} for evaluation ---")
    else:
        print(f"WARNING: Best model not found at {model_path_to_load}. Evaluating with current model weights.")

    model.eval() # 評価モードに設定
    results = {}
    with torch.no_grad(): # 勾配計算を無効化
        for strain_name, data_loader in test_dataloaders.items():
            all_recalls = []
            all_precisions = []
            all_f1_scores = []
            all_hit_counts_in_top_n = [] # Top-Nでの正解数
            all_strict_match_accuracies = [] # 厳密な一致正解率

            for X_batch, Y_batch_true_raw, genome_batch in data_loader:
                X_batch = X_batch.to(device)
                genome_batch = genome_batch.to(device)
                
                # 1回目のモデル呼び出し: ゲノム全体の平均情報を使って位置予測を行う
                # target_pos=None を渡すと、forwardのelseブロックに入り、local_genome_summary_vector がゲノム全体の平均として計算される
                _, pred_pos_logits_initial, _ = model(X_batch, genome_batch, target_pos=None) 
                
                # 1回目の予測からTop-1位置を決定
                _, top1_predicted_pos_idx = torch.max(pred_pos_logits_initial, 1)
                
                # 2回目のモデル呼び出し: 1回目の予測位置の周辺コンテキストを使って最終予測を行う
                # target_pos に top1_predicted_pos_idx を渡すと、forwardのif target_pos is not Noneブロックに入り、ローカルコンテキストが使われる
                pred_orig_logits, pred_pos_logits, pred_new_logits = model(X_batch, genome_batch, target_pos=top1_predicted_pos_idx)
                
                orig_probs = torch.softmax(pred_orig_logits, dim=1).cpu().numpy()
                pos_probs = torch.softmax(pred_pos_logits, dim=1).cpu().numpy()
                new_probs = torch.softmax(pred_new_logits, dim=1).cpu().numpy()

                for i in range(Y_batch_true_raw.shape[0]): # バッチ内の各サンプルについて
                    true_mutations = []
                    for j in range(Y_batch_true_raw.shape[1]): # 共起変異の数だけループ
                        orig_t, pos_t, new_t = Y_batch_true_raw[i, j].tolist()
                        if orig_t != PAD_BASE: # パディングされていない変異のみ有効な正解とする
                            true_mutations.append((orig_t, pos_t, new_t)) 
                    
                    if not true_mutations: # 正解が空の場合（パディングのみの場合）
                        all_recalls.append(0); all_precisions.append(0); all_f1_scores.append(0)
                        all_hit_counts_in_top_n.append(0); all_strict_match_accuracies.append(0)
                        continue

                    n_true = len(true_mutations) # 正解の共起数
                    true_set = set(true_mutations) # 正解変異の集合

                    # 全ての可能な予測候補を生成 (膨大になるので注意)
                    all_possible_predictions = []
                    for orig_idx in range(model_config['vocab_size']):
                        for pos_idx in range(model_config['sequence_length'] + 1): # ゲノム上の全位置 + PAD_POS
                            for new_idx in range(model_config['vocab_size']):
                                # パディング値を含む組み合わせは予測候補から除外
                                if orig_idx == PAD_BASE or pos_idx == PAD_POS or new_idx == PAD_BASE:
                                    continue 
                                # 予測スコアの積を計算
                                score = orig_probs[i, orig_idx] * pos_probs[i, pos_idx] * new_probs[i, new_idx]
                                all_possible_predictions.append((score, (orig_idx, pos_idx, new_idx)))
                    
                    if not all_possible_predictions: # 予測候補が一つも生成されない場合
                        all_recalls.append(0); all_precisions.append(0); all_f1_scores.append(0)
                        all_hit_counts_in_top_n.append(0); all_strict_match_accuracies.append(0)
                        continue

                    # スコアの高い順にソート
                    all_possible_predictions.sort(key=lambda x: x[0], reverse=True)
                    
                    # Top-N (N = n_true) の予測された変異を選択
                    n_pred = n_true # 予測する変異の数を正解の共起数と同じに設定
                    top_n_preds_raw = all_possible_predictions[:n_pred]
                    predicted_set = set(pred_tuple for _, pred_tuple in top_n_preds_raw) # 予測変異の集合

                    # --- 評価指標の計算 ---
                    num_correct_in_top_n = len(predicted_set & true_set) # 予測と正解の共通部分の数

                    recall = num_correct_in_top_n / n_true # Recall = (正解かつ予測された数) / (正解の総数)
                    all_recalls.append(recall)

                    precision = num_correct_in_top_n / n_pred if n_pred > 0 else 0 # Precision = (正解かつ予測された数) / (予測の総数)
                    all_precisions.append(precision)

                    if precision + recall == 0:
                        f1 = 0
                    else:
                        f1 = 2 * (precision * recall) / (precision + recall)
                    all_f1_scores.append(f1)

                    # Average Hit Count in Top-N: Top-N予測に含まれる正解の個数
                    all_hit_counts_in_top_n.append(num_correct_in_top_n)

                    # Strict Match Accuracy: 予測されたTop-Nのセットが正解のセットと完全に一致するかどうか
                    strict_match = 1 if predicted_set == true_set else 0
                    all_strict_match_accuracies.append(strict_match)
            
            # 各ストレインの全サンプルにおける平均値を計算
            avg_recall = np.mean(all_recalls) if all_recalls else 0
            avg_precision = np.mean(all_precisions) if all_precisions else 0
            avg_f1_score = np.mean(all_f1_scores) if all_f1_scores else 0
            avg_hit_count_in_top_n = np.mean(all_hit_counts_in_top_n) if all_hit_counts_in_top_n else 0
            avg_strict_match_accuracy = np.mean(all_strict_match_accuracies) if all_strict_match_accuracies else 0
            
            results[strain_name] = {
                'recall': avg_recall,
                'precision': avg_precision,
                'f1_score': avg_f1_score,
                'average_hit_count_in_top_n': avg_hit_count_in_top_n,
                'strict_match_accuracy': avg_strict_match_accuracy
            }
            force_print(f"Test Strain {strain_name}: "
                  f"Avg Hits in Top-N: {avg_hit_count_in_top_n:.2f}, "
                  f"Strict Match Acc: {avg_strict_match_accuracy:.4f}, "
                  f"Recall: {avg_recall:.4f}, "
                  f"Precision: {avg_precision:.4f}, "
                  f"F1-score: {avg_f1_score:.4f}")
    return results

# --- 14. 評価の実行 ---
print("\n--- テストデータでの評価 ---")
test_results = evaluate_multi_label(model, test_dataloaders, device, BEST_MODEL_PATH)
print(test_results)