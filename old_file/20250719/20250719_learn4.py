from sklearn.model_selection import train_test_split
import module.input_mutation_path as imp
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import os
from Bio import SeqIO
import torch.optim as optim
from datetime import datetime
import time
import sys

# モデル保存用のディレクトリとパス
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
folder_name = "../model/20250719_learn4/"
MODEL_SAVE_DIR = os.path.join(folder_name, current_time)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
BEST_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, 'best_model.pth')

# === 1. 設定パラメータ ===
data_config = {
    'train_end': 45,
    'test_start': 40,
    'ylen': 1,
    'val_ratio': 0.2,
    'mix_ratio': 0.2,
    'frag_len': 10,
    'max_co_occur': 20,
    'nmax': 20000,
    'nmax_per_strain': 1000000
}

dataset_config = {
    'strains': ['B.1.1.7','P.1','BA.2','BA.1.1','BA.1','B.1.617.2','B.1.351','B.1.1.529'],
    'usher_dir': '../usher_output/',
    'bunpu_csv': "table_heatmap/250621/table_set/table_set.csv",
    'codon_csv': 'meta_data/codon_mutation4.csv',
    'cache_dir': '../cache',
}

model_config = {
    'epochs': 30,
    'batch_size': 128,
    'embed_dim': 64,
    'sequence_length': 30000,
    'vocab_size': 4,
    'num_heads': 8,
    'num_encoder_layers': 4,
    'dropout': 0.1,
    'lr': 0.001,
    'weight_decay': 0.01
}

# --- 2. グローバル定数とゲノム配列の読み込み ---
BASE_TO_INT = {'A': 1, 'T': 2, 'C': 3, 'G': 4, '<PAD_BASE>': 0}
INT_TO_BASE = {v: k for k, v in BASE_TO_INT.items()}
PAD_BASE = BASE_TO_INT['<PAD_BASE>']
# 位置のパディング値を0に設定
PAD_POS = 0

GENOME_FASTA_PATH = 'meta_data/NC_045512.2.fasta'

def force_print(message):
    """タイムスタンプ付きで強制出力"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")
    sys.stdout.flush()

def load_and_tokenize_genome(fasta_path, base_to_int_map, target_length):
    genome_sequence = ""
    try:
        for record in SeqIO.parse(fasta_path, "fasta"):
            genome_sequence = str(record.seq).upper()
            break
    except Exception as e:
        print(f"Error reading FASTA file {fasta_path}: {e}. Trying fallback.")
        with open(fasta_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if not line.startswith('>'):
                    genome_sequence += line.strip()
        genome_sequence = genome_sequence.upper()

    if not genome_sequence:
        raise ValueError(f"No sequence found in FASTA file: {fasta_path}")

    tokenized_genome = [base_to_int_map.get(base, base_to_int_map['<PAD_BASE>']) for base in genome_sequence]
    genome_tensor = torch.tensor(tokenized_genome, dtype=torch.long)

    if len(genome_tensor) != target_length:
        print(f"WARNING: Genome length ({len(genome_tensor)}) does not match target_length ({target_length}). Truncating/padding.")
        if len(genome_tensor) > target_length:
            genome_tensor = genome_tensor[:target_length]
        else:
            padding = torch.full((target_length - len(genome_tensor),), PAD_BASE, dtype=torch.long)
            genome_tensor = torch.cat([genome_tensor, padding])
            
    return genome_tensor

try:
    GLOBAL_GENOME_TENSOR = load_and_tokenize_genome(GENOME_FASTA_PATH, BASE_TO_INT, model_config['sequence_length'])
    print(f"Loaded genome sequence tensor of shape: {GLOBAL_GENOME_TENSOR.shape}")
except (FileNotFoundError, ValueError) as e:
    print(f"CRITICAL ERROR: Could not load genome sequence: {e}")
    print("Please ensure 'meta_data/NC_045512.2.fasta' exists and contains valid FASTA data.")
    GLOBAL_GENOME_TENSOR = torch.full((model_config['sequence_length'],), PAD_BASE, dtype=torch.long)

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

# --- `add_x_by_y` is no longer needed for training data as per improvement #3 ---

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

# Improvement #3: Training data is now processed in the same way as test data.
train_x,train_y, valid_x,valid_y, test_x,test_y = dataset_by_ts(data,train_end=data_config['train_end'],test_start=data_config['test_start'],mix_ratio=data_config['mix_ratio'],
                                                               val_ratio=data_config['val_ratio'],frag_len=data_config['frag_len'],unique=True,ylen=data_config['ylen'])

# Improvement #3: `add_x_by_y` is no longer called for train and validation sets.
# train_x_split, train_y_split = add_x_by_y(train_x, train_y)
# valid_x_split, valid_y_split = add_x_by_y(valid_x, valid_y)

# === 5. PyTorchモデル構築・学習用コード ===

# --- カスタムデータセットクラス ---
class MutationPathDataset(Dataset):
    def __init__(self, X_data, Y_data, sequence_length, max_co_occur, frag_len, genome_tensor=None):
        self.X_data = X_data
        self.Y_data = Y_data 
        self.sequence_length = sequence_length
        self.max_co_occur = max_co_occur
        self.frag_len = frag_len - 1 # X has length frag_len - 1
        
        if genome_tensor is None:
            raise ValueError("genome_tensor must be provided to MutationPathDataset.")
        self.genome_tensor = genome_tensor 

        self.processed_X = self._process_hgvs_paths_X(self.X_data)
        # Improvement #3: Y processing is now unified.
        self.processed_Y = self._process_hgvs_paths_Y(self.Y_data)

    def _hgvs_to_int(self, mutation_list_input):
        if isinstance(mutation_list_input[0], list) and len(mutation_list_input[0]) == 3 and isinstance(mutation_list_input[0][0], str):
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

    def _process_hgvs_paths_Y(self, hgvs_paths_list):
        # Improvement #3: Simplified Y processing, handles list of mutations directly.
        processed_tensors = []
        for y_element in hgvs_paths_list:
            # y_element is a list of mutations, e.g., [[['C', '15279', 'T']], [['G', '1120', 'A']]]
            # We need to flatten it to [['C', '15279', 'T'], ['G', '1120', 'A']]
            flat_y = [item[0] for item in y_element]
            
            temp_list = [self._hgvs_to_int(m) for m in flat_y]
            while len(temp_list) < self.max_co_occur:
                 temp_list.append([PAD_BASE, PAD_POS, PAD_BASE])
            processed_tensors.append(torch.tensor(temp_list[:self.max_co_occur], dtype=torch.long))
            
        return processed_tensors

    def __len__(self):
        return len(self.X_data) 

    def __getitem__(self, idx):
        return self.processed_X[idx], self.processed_Y[idx], self.genome_tensor

# --- DataLoaderのcollate_fn ---
def custom_collate_fn(batch):
    X_batch = torch.stack([s[0] for s in batch])
    Y_batch = torch.stack([s[1] for s in batch]) # Y is now always a list
    genome_batch = batch[0][2].unsqueeze(0).repeat(len(batch), 1)
    
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
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        self.original_base_embedding = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=PAD_BASE)
        self.position_embedding = nn.Embedding(sequence_length + 1, embed_dim, padding_idx=PAD_POS)
        self.new_base_embedding = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=PAD_BASE)
        self.mutation_embedding_projection = nn.Linear(embed_dim * 3, embed_dim)
        self.history_positional_encoding = PositionalEncoding(embed_dim, max_len=data_config['frag_len']) 
        
        self.history_encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout_rate, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.history_encoder_layer, num_layers=num_encoder_layers)

        # Improvement #1 & #5: Global Genome Context Encoder
        self.genome_base_embedding = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=PAD_BASE) 
        self.global_genome_encoder = nn.Sequential(
            nn.Conv1d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=11, padding=5),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        # ゲノム情報と変異履歴情報の統合レイヤー
        self.integration_layer = nn.Linear(embed_dim * 2, embed_dim) 
        
        # 予測ヘッド (Improvement #3: Predicts multiple mutations)
        self.prediction_feature_expander = nn.Linear(embed_dim, embed_dim * max_co_occur)
        
        self.predicted_original_base_head = nn.Linear(embed_dim, vocab_size + 1)
        self.position_head = nn.Linear(embed_dim, sequence_length + 1)
        self.predicted_new_base_head = nn.Linear(embed_dim, vocab_size + 1)

    # Improvement #1: Simplified forward pass, no target_pos needed
    def forward(self, mutation_history_input, genome_input):
        original_base_indices = mutation_history_input[..., 0]
        position_indices = mutation_history_input[..., 1]
        new_base_indices = mutation_history_input[..., 2]

        original_emb = self.original_base_embedding(original_base_indices)
        position_emb = self.position_embedding(position_indices)
        new_base_emb = self.new_base_embedding(new_base_indices)

        single_mutation_embedding = torch.cat([original_emb, position_emb, new_base_emb], dim=-1)
        single_mutation_embedding = self.mutation_embedding_projection(single_mutation_embedding)
        
        # Average embeddings for co-occurring mutations at each step
        co_occurrence_mask = (position_indices != PAD_POS).float().unsqueeze(-1)
        sum_embeddings = (single_mutation_embedding * co_occurrence_mask).sum(dim=2)
        count_embeddings = co_occurrence_mask.sum(dim=2)
        co_mutation_context = sum_embeddings / (count_embeddings + 1e-9)

        co_mutation_context = self.history_positional_encoding(co_mutation_context)
        
        # Create padding mask for the transformer encoder
        src_key_padding_mask = (count_embeddings == 0).squeeze(-1)

        history_encoder_output = self.transformer_encoder(co_mutation_context, src_key_padding_mask=src_key_padding_mask)
        # Use the output corresponding to the last time step
        last_history_output = history_encoder_output[:, -1, :]

        # Improvement #1 & #5: Process genome with global context encoder
        genome_emb = self.genome_base_embedding(genome_input)
        genome_emb = genome_emb.permute(0, 2, 1) # (batch, embed_dim, seq_len)
        global_genome_summary_vector = self.global_genome_encoder(genome_emb).squeeze(-1)

        # Combine history and genome context
        combined_features = torch.cat([last_history_output, global_genome_summary_vector], dim=-1)
        final_prediction_input = self.integration_layer(combined_features)

        # Improvement #3: Expand features to predict max_co_occur mutations
        expanded_features = self.prediction_feature_expander(final_prediction_input)
        # Reshape to (batch_size, max_co_occur, embed_dim)
        expanded_features = expanded_features.view(-1, self.max_co_occur, self.embed_dim)

        # Generate predictions for each of the max_co_occur possible mutations
        predicted_original_base_logits = self.predicted_original_base_head(expanded_features)
        predicted_position_logits = self.position_head(expanded_features)
        predicted_new_base_logits = self.predicted_new_base_head(expanded_features)

        return predicted_original_base_logits, predicted_position_logits, predicted_new_base_logits

# --- 9. データセットとデータローダーの作成 ---
train_dataset = MutationPathDataset(
    train_x, train_y, 
    model_config['sequence_length'], data_config['max_co_occur'], data_config['frag_len'],
    genome_tensor=GLOBAL_GENOME_TENSOR
)
val_dataset = MutationPathDataset(
    valid_x, valid_y, 
    model_config['sequence_length'], data_config['max_co_occur'], data_config['frag_len'],
    genome_tensor=GLOBAL_GENOME_TENSOR
)

test_dataloaders = {}
for k, v in test_x.items():
    test_y_k = test_y[k]
    test_dataset = MutationPathDataset(
        v, test_y_k, 
        model_config['sequence_length'], data_config['max_co_occur'], data_config['frag_len'],
        genome_tensor=GLOBAL_GENOME_TENSOR
    )
    test_dataloaders[k] = DataLoader(test_dataset, batch_size=model_config['batch_size'], shuffle=False, collate_fn=custom_collate_fn)

train_dataloader = DataLoader(train_dataset, batch_size=model_config['batch_size'], shuffle=True, collate_fn=custom_collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=model_config['batch_size'], shuffle=False, collate_fn=custom_collate_fn)


# --- 10. モデルのインスタンス化 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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
    best_val_pos_acc = -1.0
    
    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        total_loss = 0
        for batch_idx, (X_batch, Y_batch, genome_batch) in enumerate(train_loader):
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            genome_batch = genome_batch.to(device)

            # Targets have shape (batch, max_co_occur, 3)
            target_original_base = Y_batch[..., 0]
            target_position = Y_batch[..., 1]
            target_new_base = Y_batch[..., 2]

            optimizer.zero_grad()
            
            # Improvement #1: Single forward pass
            pred_orig_logits, pred_pos_logits, pred_new_logits = model(X_batch, genome_batch)
            
            # Predictions have shape (batch, max_co_occur, num_classes)
            # Reshape for loss calculation: (batch * max_co_occur, num_classes)
            # Targets need to be reshaped to (batch * max_co_occur)
            
            # Improvement #4: Correctly handle padding in loss calculation
            loss_orig = criterion_orig(pred_orig_logits.reshape(-1, model_config['vocab_size'] + 1), target_original_base.reshape(-1))
            loss_pos = criterion_pos(pred_pos_logits.reshape(-1, model_config['sequence_length'] + 1), target_position.reshape(-1))
            loss_new = criterion_new(pred_new_logits.reshape(-1, model_config['vocab_size'] + 1), target_new_base.reshape(-1))
            
            loss = loss_orig + loss_pos + loss_new
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        avg_train_loss = total_loss / len(train_loader)
        
        model.eval()
        val_loss = 0
        correct_pos = 0; total_pos = 0

        with torch.no_grad():
            for X_batch, Y_batch, genome_batch in val_loader:
                X_batch = X_batch.to(device)
                Y_batch = Y_batch.to(device)
                genome_batch = genome_batch.to(device)

                target_position = Y_batch[..., 1]

                # Single forward pass
                _, pred_pos_logits, _ = model(X_batch, genome_batch)

                # Position accuracy is a good indicator of performance
                _, predicted_pos = torch.max(pred_pos_logits, 2)
                
                valid_pos_mask = (target_position != PAD_POS)
                correct_pos += (predicted_pos[valid_pos_mask] == target_position[valid_pos_mask]).sum().item()
                total_pos += valid_pos_mask.sum().item()

        val_pos_acc = correct_pos / total_pos if total_pos > 0 else 0

        force_print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Pos Acc: {val_pos_acc:.4f}")

        if val_pos_acc > best_val_pos_acc:
            best_val_pos_acc = val_pos_acc
            torch.save(model.state_dict(), model_save_path)
            force_print(f"--- Epoch {epoch+1}: Validation Position Accuracy improved to {best_val_pos_acc:.4f}. Saving model to {model_save_path} ---")
        epoch_end = time.time()
        force_print(f"--- Epoch {epoch+1} took {epoch_end - epoch_start:.2f} seconds ---")

print("--- PyTorchモデル学習開始 ---")
train_model(model, train_dataloader, val_dataloader, criterion_orig, criterion_pos, criterion_new, optimizer, model_config['epochs'], device, BEST_MODEL_PATH)

print("\n--- テストデータでの評価 ---")
# Improvement #2: Simplified evaluation metrics, focusing on F1-score
def evaluate_multi_label(model, test_dataloaders, device, model_path_to_load):
    if os.path.exists(model_path_to_load):
        model.load_state_dict(torch.load(model_path_to_load, map_location=device))
        print(f"--- Loaded best model from {model_path_to_load} for evaluation ---")
    else:
        print(f"WARNING: Best model not found at {model_path_to_load}. Evaluating with current model weights.")

    model.eval()
    results = {}
    with torch.no_grad():
        for strain_name, data_loader in test_dataloaders.items():
            all_recalls = []
            all_precisions = []
            all_f1_scores = []

            for X_batch, Y_batch_true_raw, genome_batch in data_loader:
                X_batch = X_batch.to(device)
                genome_batch = genome_batch.to(device)
                
                # Single forward pass
                pred_orig_logits, pred_pos_logits, pred_new_logits = model(X_batch, genome_batch)
                
                # Get top prediction for each slot
                _, pred_orig_idx = torch.max(pred_orig_logits, 2)
                _, pred_pos_idx = torch.max(pred_pos_logits, 2)
                _, pred_new_idx = torch.max(pred_new_logits, 2)
                
                # Loop through each sample in the batch
                for i in range(Y_batch_true_raw.shape[0]):
                    # Build the true set of mutations, ignoring padding
                    true_set = set()
                    for j in range(Y_batch_true_raw.shape[1]):
                        orig_t, pos_t, new_t = Y_batch_true_raw[i, j].tolist()
                        if pos_t != PAD_POS: # Use position padding to check validity
                            true_set.add((orig_t, pos_t, new_t))
                    
                    if not true_set:
                        continue # Skip if no true mutations

                    # Build the predicted set of mutations, ignoring padding
                    predicted_set = set()
                    for j in range(pred_pos_idx.shape[1]):
                        pos_p = pred_pos_idx[i, j].item()
                        if pos_p != PAD_POS:
                            predicted_set.add((
                                pred_orig_idx[i, j].item(),
                                pos_p,
                                pred_new_idx[i, j].item()
                            ))
                    
                    num_correct = len(predicted_set & true_set)
                    n_true = len(true_set)
                    n_pred = len(predicted_set)

                    recall = num_correct / n_true if n_true > 0 else 0
                    precision = num_correct / n_pred if n_pred > 0 else 0
                    
                    if precision + recall == 0:
                        f1 = 0.0
                    else:
                        f1 = 2 * (precision * recall) / (precision + recall)
                    
                    all_recalls.append(recall)
                    all_precisions.append(precision)
                    all_f1_scores.append(f1)

            avg_recall = np.mean(all_recalls) if all_recalls else 0
            avg_precision = np.mean(all_precisions) if all_precisions else 0
            avg_f1_score = np.mean(all_f1_scores) if all_f1_scores else 0
            
            results[strain_name] = {
                'recall': avg_recall,
                'precision': avg_precision,
                'f1_score': avg_f1_score,
            }
            force_print(f"Test Strain {strain_name}: "
                  f"Recall: {avg_recall:.4f}, "
                  f"Precision: {avg_precision:.4f}, "
                  f"F1-score: {avg_f1_score:.4f}")
    return results

test_results = evaluate_multi_label(model, test_dataloaders, device, BEST_MODEL_PATH)
print("--- Final Test Results ---")
print(test_results)