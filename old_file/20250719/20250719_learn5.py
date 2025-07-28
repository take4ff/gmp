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
folder_name = "../model/20250719_learn5/"
MODEL_SAVE_DIR = os.path.join(folder_name, current_time)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
BEST_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, 'best_model.pth')

# === 1. 設定パラメータ ===
data_config = {
    'train_end': 40,
    'test_start': 35,
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
    'weight_decay': 0.01,
    'top_k_eval': 100
}

# --- 2. グローバル定数とゲノム配列の読み込み ---
BASE_TO_INT = {'A': 1, 'T': 2, 'C': 3, 'G': 4, '<PAD_BASE>': 0}
INT_TO_BASE = {v: k for k, v in BASE_TO_INT.items()}
PAD_BASE = BASE_TO_INT['<PAD_BASE>']
PAD_POS = 0
GENOME_FASTA_PATH = 'meta_data/NC_045512.2.fasta'

def force_print(message):
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
            genome_sequence = "".join([line.strip() for line in lines if not line.startswith('>')])
        genome_sequence = genome_sequence.upper()
    if not genome_sequence: raise ValueError(f"No sequence found in FASTA file: {fasta_path}")
    tokenized_genome = [base_to_int_map.get(base, PAD_BASE) for base in genome_sequence]
    genome_tensor = torch.tensor(tokenized_genome, dtype=torch.long)
    if len(genome_tensor) != target_length:
        print(f"WARNING: Genome length ({len(genome_tensor)}) does not match target_length ({target_length}). Adjusting.")
        if len(genome_tensor) > target_length: genome_tensor = genome_tensor[:target_length]
        else: genome_tensor = torch.cat([genome_tensor, torch.full((target_length - len(genome_tensor),), PAD_BASE, dtype=torch.long)])
    return genome_tensor

try:
    GLOBAL_GENOME_TENSOR = load_and_tokenize_genome(GENOME_FASTA_PATH, BASE_TO_INT, model_config['sequence_length'])
    print(f"Loaded genome sequence tensor of shape: {GLOBAL_GENOME_TENSOR.shape}")
except (FileNotFoundError, ValueError) as e:
    print(f"CRITICAL ERROR: Could not load genome sequence: {e}")
    GLOBAL_GENOME_TENSOR = torch.full((model_config['sequence_length'],), PAD_BASE, dtype=torch.long)

# --- 3. データ前処理関数群 (ユーザー提供のものをそのまま使用) ---
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
    train_x = separete_HGVS(train_x); train_y = separete_HGVS(train_y)
    valid_x = separete_HGVS(valid_x); valid_y = separete_HGVS(valid_y)
    for k in sorted(list(test.keys())):
        test_x[k] = separete_HGVS(test_x[k]); test_y[k] = separete_HGVS(test_y[k])
    return train_x,train_y, valid_x,valid_y, test_x,test_y

# [変更点] ラベル設定を戻すため、add_x_by_y関数を追加
def add_x_by_y(x, y):
    if len(x) != len(y): raise ValueError("x and y must have the same length")
    new_x, new_y = [], []
    for i in range(len(y)):
        list_of_co_mutations = y[i][0]
        for single_mutation in list_of_co_mutations:
            new_x.append(x[i]); new_y.append(single_mutation)
    return new_x, new_y

# --- 4. データロードと前処理の実行 ---
names, lengths, base_HGVS_paths = imp.input(dataset_config['strains'], dataset_config['usher_dir'], nmax=data_config['nmax'], nmax_per_strain=data_config['nmax_per_strain'])
print("元データ",len(base_HGVS_paths)); filted_data, _, _ = filter_co_occur(base_HGVS_paths, names, lengths, data_config['max_co_occur']); print("共起フィルタ",len(filted_data))
data = unique_path(filted_data); print("ユニークパス",len(data))
train_x, train_y, valid_x, valid_y, test_x, test_y = dataset_by_ts(data, train_end=data_config['train_end'], test_start=data_config['test_start'], mix_ratio=data_config['mix_ratio'], val_ratio=data_config['val_ratio'], frag_len=data_config['frag_len'], unique=True, ylen=data_config['ylen'])

# [変更点] ラベル設定を戻すため、訓練・検証データを分解
print("Splitting co-occurrences for training and validation sets...")
train_x_split, train_y_split = add_x_by_y(train_x, train_y); valid_x_split, valid_y_split = add_x_by_y(valid_x, valid_y)
print(f"Original train samples: {len(train_x)}, Split train samples: {len(train_x_split)}"); print(f"Original valid samples: {len(valid_x)}, Split valid samples: {len(valid_x_split)}")

# --- 5. PyTorch Dataset Class ---
class MutationPathDataset(Dataset):
    # [変更点] is_test_dataフラグを再導入
    def __init__(self, X_data, Y_data, sequence_length, max_co_occur, frag_len, genome_tensor, is_test_data=False):
        self.X_data, self.Y_data = X_data, Y_data; self.sequence_length, self.max_co_occur, self.frag_len = sequence_length, max_co_occur, frag_len -1; self.genome_tensor, self.is_test_data = genome_tensor, is_test_data
        self.processed_X = self._process_hgvs_paths_X(self.X_data); self.processed_Y = self._process_hgvs_paths_Y(self.Y_data)
    
    def _hgvs_to_int(self, m):
        pos_str = m[1]
        if not pos_str.isdigit() or not (0 <= int(pos_str) <= self.sequence_length):
            return [PAD_BASE, PAD_POS, PAD_BASE]
        orig_int = BASE_TO_INT.get(m[0], PAD_BASE); new_int = BASE_TO_INT.get(m[2], PAD_BASE)
        if orig_int == PAD_BASE or new_int == PAD_BASE:
            return [PAD_BASE, PAD_POS, PAD_BASE]
        return [orig_int, int(pos_str), new_int]

    def _process_hgvs_paths_X(self, hgvs_paths_list):
        processed_tensors = []
        for path in hgvs_paths_list:
            step_tensors = []
            for step_mutations in path:
                co_mut_tensors = [self._hgvs_to_int(m) for m in step_mutations]
                while len(co_mut_tensors) < self.max_co_occur: co_mut_tensors.append([PAD_BASE, PAD_POS, PAD_BASE])
                step_tensors.append(co_mut_tensors[:self.max_co_occur])
            while len(step_tensors) < self.frag_len: step_tensors.append([[PAD_BASE, PAD_POS, PAD_BASE]] * self.max_co_occur)
            processed_tensors.append(torch.tensor(step_tensors, dtype=torch.long))
        return processed_tensors

    # [変更点] is_test_dataフラグに応じて処理を分岐
    def _process_hgvs_paths_Y(self, hgvs_paths_list):
        processed_tensors = []
        if self.is_test_data:
            for y_element in hgvs_paths_list:
                flat_y = [item[0] for item in y_element]; temp_list = [self._hgvs_to_int(m) for m in flat_y]
                while len(temp_list) < self.max_co_occur: temp_list.append([PAD_BASE, PAD_POS, PAD_BASE])
                processed_tensors.append(torch.tensor(temp_list[:self.max_co_occur], dtype=torch.long))
        else:
            for y_element in hgvs_paths_list: processed_tensors.append(torch.tensor(self._hgvs_to_int(y_element), dtype=torch.long))
        return processed_tensors
    def __len__(self): return len(self.X_data)
    def __getitem__(self, idx): return self.processed_X[idx], self.processed_Y[idx], self.genome_tensor

# --- 6. Collate Function & Positional Encoding ---
def custom_collate_fn(batch):
    X_batch = torch.stack([s[0] for s in batch]); Y_batch = torch.stack([s[1] for s in batch]); genome_batch = batch[0][2].unsqueeze(0).repeat(len(batch), 1)
    return X_batch, Y_batch, genome_batch
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model); position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term); pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x: torch.Tensor) -> torch.Tensor: return x + self.pe[:, :x.size(1), :]

# --- 7. Mutation Predictor モデル ---
class MutationPredictor(nn.Module):
    # [変更点] 単一変異予測モデルに戻す
    def __init__(self, embed_dim, vocab_size, sequence_length, max_co_occur, num_heads, num_encoder_layers, dropout_rate=0.1):
        super(MutationPredictor, self).__init__()
        self.original_base_embedding = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=PAD_BASE)
        self.position_embedding = nn.Embedding(sequence_length + 1, embed_dim, padding_idx=PAD_POS)
        self.new_base_embedding = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=PAD_BASE)
        self.mutation_embedding_projection = nn.Linear(embed_dim * 3, embed_dim)
        self.history_positional_encoding = PositionalEncoding(embed_dim, max_len=data_config['frag_len']) 
        self.history_encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout_rate, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.history_encoder_layer, num_layers=num_encoder_layers)
        self.genome_base_embedding = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=PAD_BASE) 
        self.global_genome_encoder = nn.Sequential(
            nn.Conv1d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=11, padding=5),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.integration_layer = nn.Linear(embed_dim * 2, embed_dim) 
        self.predicted_original_base_head = nn.Linear(embed_dim, vocab_size + 1)
        self.position_head = nn.Linear(embed_dim, sequence_length + 1)
        self.predicted_new_base_head = nn.Linear(embed_dim, vocab_size + 1)

    def forward(self, mutation_history_input, genome_input):
        original_base_indices = mutation_history_input[..., 0]; position_indices = mutation_history_input[..., 1]; new_base_indices = mutation_history_input[..., 2]
        original_emb = self.original_base_embedding(original_base_indices); position_emb = self.position_embedding(position_indices); new_base_emb = self.new_base_embedding(new_base_indices)
        single_mutation_embedding = self.mutation_embedding_projection(torch.cat([original_emb, position_emb, new_base_emb], dim=-1))
        co_occurrence_mask = (position_indices != PAD_POS).float().unsqueeze(-1)
        co_mutation_context = (single_mutation_embedding * co_occurrence_mask).sum(dim=2) / (co_occurrence_mask.sum(dim=2) + 1e-9)
        co_mutation_context = self.history_positional_encoding(co_mutation_context)
        src_key_padding_mask = (co_occurrence_mask.sum(dim=2) == 0).squeeze(-1)
        history_encoder_output = self.transformer_encoder(co_mutation_context, src_key_padding_mask=src_key_padding_mask)
        history_summary_vector = history_encoder_output[:, -1, :]
        genome_emb = self.genome_base_embedding(genome_input).permute(0, 2, 1)
        global_genome_summary_vector = self.global_genome_encoder(genome_emb).squeeze(-1)
        combined_features = torch.cat([history_summary_vector, global_genome_summary_vector], dim=-1)
        final_prediction_input = self.integration_layer(combined_features)
        predicted_original_base_logits = self.predicted_original_base_head(final_prediction_input)
        predicted_position_logits = self.position_head(final_prediction_input)
        predicted_new_base_logits = self.predicted_new_base_head(final_prediction_input)
        return predicted_original_base_logits, predicted_position_logits, predicted_new_base_logits

# --- 8. Dataloaders ---
train_dataset = MutationPathDataset(train_x_split, train_y_split, model_config['sequence_length'], data_config['max_co_occur'], data_config['frag_len'], GLOBAL_GENOME_TENSOR, is_test_data=False)
val_dataset = MutationPathDataset(valid_x_split, valid_y_split, model_config['sequence_length'], data_config['max_co_occur'], data_config['frag_len'], GLOBAL_GENOME_TENSOR, is_test_data=False)
test_dataloaders = {k: DataLoader(MutationPathDataset(v, test_y[k], model_config['sequence_length'], data_config['max_co_occur'], data_config['frag_len'], GLOBAL_GENOME_TENSOR, is_test_data=True), batch_size=model_config['batch_size'], shuffle=False, collate_fn=custom_collate_fn) for k, v in test_x.items()}
train_dataloader = DataLoader(train_dataset, batch_size=model_config['batch_size'], shuffle=True, collate_fn=custom_collate_fn); val_dataloader = DataLoader(val_dataset, batch_size=model_config['batch_size'], shuffle=False, collate_fn=custom_collate_fn)

# --- 9. Model, Loss, Optimizer ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); print(f"Using device: {device}")
model = MutationPredictor(embed_dim=model_config['embed_dim'], vocab_size=model_config['vocab_size'], sequence_length=model_config['sequence_length'], max_co_occur=data_config['max_co_occur'], num_heads=model_config['num_heads'], num_encoder_layers=model_config['num_encoder_layers'], dropout_rate=model_config['dropout']).to(device)
criterion_orig = nn.CrossEntropyLoss(ignore_index=PAD_BASE); criterion_pos = nn.CrossEntropyLoss(ignore_index=PAD_POS); criterion_new = nn.CrossEntropyLoss(ignore_index=PAD_BASE)
optimizer = optim.AdamW(model.parameters(), lr=model_config['lr'], weight_decay=model_config['weight_decay'])

# --- 10. 学習ループ ---
# [変更点] 単一予測モデル用に単純化
def train_model(model, train_loader, val_loader, criterion_orig, criterion_pos, criterion_new, optimizer, epochs, device, model_save_path):
    best_val_pos_acc = -1.0
    for epoch in range(epochs):
        epoch_start = time.time(); model.train(); total_loss = 0
        for X_batch, Y_batch, genome_batch in train_loader:
            X_batch, Y_batch, genome_batch = X_batch.to(device), Y_batch.to(device), genome_batch.to(device)
            target_original_base, target_position, target_new_base = Y_batch[:, 0], Y_batch[:, 1], Y_batch[:, 2]
            optimizer.zero_grad()
            pred_orig_logits, pred_pos_logits, pred_new_logits = model(X_batch, genome_batch)
            loss = criterion_orig(pred_orig_logits, target_original_base) + criterion_pos(pred_pos_logits, target_position) + criterion_new(pred_new_logits, target_new_base)
            loss.backward(); optimizer.step(); total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)
        model.eval(); correct_pos, total_pos = 0, 0
        with torch.no_grad():
            for X_batch, Y_batch, genome_batch in val_loader:
                X_batch, Y_batch, genome_batch = X_batch.to(device), Y_batch.to(device), genome_batch.to(device)
                target_position = Y_batch[:, 1]; _, pred_pos_logits, _ = model(X_batch, genome_batch); _, predicted_pos = torch.max(pred_pos_logits, 1)
                valid_pos_mask = (target_position != PAD_POS); correct_pos += (predicted_pos[valid_pos_mask] == target_position[valid_pos_mask]).sum().item(); total_pos += valid_pos_mask.sum().item()
        val_pos_acc = correct_pos / total_pos if total_pos > 0 else 0
        force_print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Pos Acc: {val_pos_acc:.4f}")
        if val_pos_acc > best_val_pos_acc:
            best_val_pos_acc = val_pos_acc; torch.save(model.state_dict(), model_save_path); force_print(f"  -> Val Pos Acc improved. Saving model to {model_save_path}")
        force_print(f"  Epoch took {time.time() - epoch_start:.2f} seconds.")

# --- 11. 評価ループ ---
# [変更点] 単一予測モデル用にランキングベースの評価方法に変更
def evaluate_model(model, test_dataloaders, device, model_path_to_load):
    if os.path.exists(model_path_to_load): model.load_state_dict(torch.load(model_path_to_load, map_location=device)); force_print(f"Loaded best model from {model_path_to_load} for evaluation.")
    else: force_print(f"WARNING: Model file not found at {model_path_to_load}. Evaluating with current model.")
    model.eval(); results = {}
    with torch.no_grad():
        for strain_name, data_loader in test_dataloaders.items():
            all_recalls, all_precisions, all_f1s = [], [], []
            for X_batch, Y_batch_true_raw, genome_batch in data_loader:
                X_batch, genome_batch = X_batch.to(device), genome_batch.to(device)
                orig_logits, pos_logits, new_logits = model(X_batch, genome_batch)
                orig_probs, pos_probs, new_probs = torch.softmax(orig_logits, dim=1), torch.softmax(pos_logits, dim=1), torch.softmax(new_logits, dim=1)
                top_k_pos_probs, top_k_pos_indices = torch.topk(pos_probs, model_config['top_k_eval'], dim=1)
                for i in range(Y_batch_true_raw.shape[0]):
                    true_mutations = {tuple(m) for m in Y_batch_true_raw[i].tolist() if m[1] != PAD_POS}
                    if not true_mutations: continue
                    n_true = len(true_mutations); candidates = []
                    for k in range(model_config['top_k_eval']):
                        pos_idx, pos_prob = top_k_pos_indices[i, k].item(), top_k_pos_probs[i, k].item()
                        for orig_idx in range(1, model_config['vocab_size'] + 1):
                            for new_idx in range(1, model_config['vocab_size'] + 1):
                                if orig_idx == new_idx: continue
                                score = pos_prob * orig_probs[i, orig_idx].item() * new_probs[i, new_idx].item()
                                candidates.append((score, (orig_idx, pos_idx, new_idx)))
                    candidates.sort(key=lambda x: x[0], reverse=True); top_n_preds = {pred for _, pred in candidates[:n_true]}
                    num_correct = len(top_n_preds & true_mutations); precision = num_correct / n_true if n_true > 0 else 0; recall = precision
                    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                    all_precisions.append(precision); all_recalls.append(recall); all_f1s.append(f1)
            results[strain_name] = {'recall': np.mean(all_recalls) if all_recalls else 0, 'precision': np.mean(all_precisions) if all_precisions else 0, 'f1_score': np.mean(all_f1s) if all_f1s else 0}
            force_print(f"Test Strain {strain_name} | Recall@k: {results[strain_name]['recall']:.4f} | F1@k: {results[strain_name]['f1_score']:.4f}")
    return results

# --- 12. Execution ---
force_print("--- PyTorchモデル学習開始 ---")
train_model(model, train_dataloader, val_dataloader, criterion_orig, criterion_pos, criterion_new, optimizer, model_config['epochs'], device, BEST_MODEL_PATH)

force_print("\n--- テストデータでの評価 ---")
test_results = evaluate_model(model, test_dataloaders, device, BEST_MODEL_PATH)
print("\n--- Final Test Results ---")
print(test_results)