from sklearn.model_selection import train_test_split
import module.input_mutation_path as imp

data_config = {
    'train_end': 37,
    'test_start': 34,
    'ylen': 1,
    'val_ratio': 0.2,
    'mix_ratio': 0.2, 
    'frag_len': 10,
    'max_co_occur': 20,
    'nmax': 10000,
    'nmax_per_strain': 1000000
}

dataset_config = {
    'strains': ['B.1.1.7','P.1','BA.2','BA.1.1','BA.1','B.1.617.2','B.1.351','B.1.1.529'],
    'usher_dir': '../usher_output/',
    'bunpu_csv': "table_heatmap/250621/table_set/table_set.csv",
    'codon_csv': 'meta_data/codon_mutation4.csv',
    'cache_dir': '../cache',  # 特徴データキャッシュ用ディレクトリ
}

# === モデル関連パラメータの設定 ===
model_config = {
    'epochs': 10,
    'batch_size': 32,
    'embed_dim': 32,         # 埋め込み次元
    'sequence_length': 30000, # ゲノム長など（適宜修正）
    'vocab_size': 4,         # 塩基A,C,G,Tなら4
    'num_heads': 4,          # Transformerのヘッド数
    'num_encoder_layers': 2, # Transformerの層数
}

# HGVS表記から数値インデックスへのマッピング定義
BASE_TO_INT = {'A': 0, 'T': 1, 'C': 2, 'G': 3, '<PAD_BASE>': 4}
INT_TO_BASE = {v: k for k, v in BASE_TO_INT.items()}
PAD_BASE = BASE_TO_INT['<PAD_BASE>']
PAD_POS = model_config['sequence_length'] + 1 

names, lengths, base_HGVS_paths = imp.input(
dataset_config['strains'], 
dataset_config['usher_dir'], 
nmax=data_config['nmax'], 
nmax_per_strain=data_config['nmax_per_strain']
)

print("元データ",len(base_HGVS_paths))

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

filted_data, temp, temp  = filter_co_occur(base_HGVS_paths,names,lengths,data_config['max_co_occur'])
print("共起フィルタ",len(filted_data))

data = unique_path(filted_data)
print("ユニークパス",len(data))

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
    """
    print("train",len(train))
    print("valid",len(valid))
    for k in sorted(list(test.keys())):
        print(f"test:{k} {len(test[k])}")
        print(test[k])
    """
    train_x,train_y = separate_XY(train,ylen)
    valid_x,valid_y = separate_XY(valid,ylen)
    test_x = {}
    test_y = {}
    for k in sorted(list(test.keys())):
        if(test_x.get(k) is None):
            test_x[k] = []
        if(test_y.get(k) is None):
            test_y[k] = []
        tx,ty = separate_XY(test[k],ylen)
        test_x[k].extend(tx)
        test_y[k].extend(ty)

    train_x = separete_HGVS(train_x)
    train_y = separete_HGVS(train_y) # ラベルにも適用
    valid_x = separete_HGVS(valid_x)
    valid_y = separete_HGVS(valid_y) # ラベルにも適用
    for k in sorted(list(test.keys())):
        test_x[k] = separete_HGVS(test_x[k])
        test_y[k] = separete_HGVS(test_y[k]) # ラベルにも適用

    return train_x,train_y, valid_x,valid_y, test_x,test_y

train_x,train_y, valid_x,valid_y, test_x,test_y = dataset_by_ts(data,train_end=data_config['train_end'],test_start=data_config['test_start'],mix_ratio=data_config['mix_ratio'],
                                val_ratio=data_config['val_ratio'],frag_len=data_config['frag_len'],unique=True,ylen=data_config['ylen'])


def add_x_by_y(x, y):
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    
    new_x, new_y = [], []
    for i in range(len(y)):
        for fea in y[i]:
            new_x.append(x[i])
            new_y.append(fea)
    return new_x, new_y

train_x_split, train_y_split = add_x_by_y(train_x, train_y)
valid_x_split, valid_y_split = add_x_by_y(valid_x, valid_y)


print("\n--- Debugging Data Structure ---")
print(f"Type of train_y_split: {type(train_y_split)}")
if train_y_split:
    print(f"Length of train_y_split: {len(train_y_split)}")
    print(f"Sample train_y_split[0]: {train_y_split[0]}")
    print(f"Type of train_y_split[0]: {type(train_y_split[0])}")

    # さらに深く潜って確認
    if isinstance(train_y_split[0], list) and train_y_split[0]:
        print(f"Sample train_y_split[0][0]: {train_y_split[0][0]}")
        print(f"Type of train_y_split[0][0]: {type(train_y_split[0][0])}")
        if isinstance(train_y_split[0][0], list) and train_y_split[0][0]:
            print(f"Sample train_y_split[0][0][0]: {train_y_split[0][0][0]}")
            print(f"Type of train_y_split[0][0][0]: {type(train_y_split[0][0][0])}")
print("------------------------------\n")

print(f"train_x_split sample: {train_x_split[0]}")
print(f"train_y_split sample: {train_y_split[0]}")
print(f"train_y_split type: {type(train_y_split[0])}")

"""
print("train_x",len(train_x))
print("train_y",len(train_y))
print("valid_x",len(valid_x))
print("valid_y",len(valid_y))
for k in sorted(list(test_x.keys())):
    print(f"test_x:{k} {len(test_x[k])}")
    print(test_x[k])
for k in sorted(list(test_y.keys())):
    print(f"test_y:{k} {len(test_y[k])}")
    print(test_y[k])
"""

# === ここからモデル構築・学習用コード ===
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
# --- カスタムデータセットクラスの調整 ---
class MutationPathDataset(Dataset):
    def __init__(self, X_data, Y_data, sequence_length, max_co_occur, frag_len, is_test_data=False):
        self.X_data = X_data
        self.Y_data = Y_data 
        self.sequence_length = sequence_length
        self.max_co_occur = max_co_occur
        self.frag_len = frag_len 
        self.is_test_data = is_test_data

        self.processed_X = self._process_hgvs_paths_X(self.X_data)
        self.processed_Y = self._process_hgvs_paths_Y(self.Y_data, is_test_data)

    def _hgvs_to_int(self, mutation_list_input):
        """
        ['C', '200', 'G'] -> [2, 200, 3]
        この関数は、引数が ['塩基', '位置', '塩基'] の形式であることを強く期待します。
        """
        # --- ここをさらに強化する ---
        # 念のため、要素が文字列であることを確認し、型が違うなら強制的に変換を試みる
        if isinstance(mutation_list_input[0], list):
            # もしここに来るなら、これまでのデバッグ出力と矛盾。しかし念のため
            #print(f"WARNING: _hgvs_to_int received nested list: {mutation_list_input}. Attempting to flatten.")
            mutation_list_input = mutation_list_input[0] # ネストを剥がす

        if not all(isinstance(x, str) for x in mutation_list_input[:3]):
             # 少なくとも最初の3要素が文字列でない場合、エラーの可能性が高い
             print(f"ERROR: _hgvs_to_int received non-string elements: {mutation_list_input}")
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
            # デバッグ出力 (前回追加したもの)
            # print(f"DEBUG Y pre-loop: {y_element}, type: {type(y_element)}")
            
            if is_test_data:
                temp_list = [self._hgvs_to_int(m) for m in y_element]
                while len(temp_list) < self.max_co_occur:
                     temp_list.append([PAD_BASE, PAD_POS, PAD_BASE])
                processed_tensors.append(torch.tensor(temp_list[:self.max_co_occur], dtype=torch.long))
            else: 
                current_element = y_element
                while isinstance(current_element, list) and current_element and isinstance(current_element[0], list):
                    current_element = current_element[0]
                
                # print(f"DEBUG Y post-loop: {current_element}, type: {type(current_element)}") # デバッグ出力
                processed_tensors.append(torch.tensor(self._hgvs_to_int(current_element), dtype=torch.long))

        return processed_tensors

    def __len__(self):
        return len(self.X_data) 

    def __getitem__(self, idx):
        return self.processed_X[idx], self.processed_Y[idx]

# PositionalEncoding クラスの定義
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        """
        Args:
            d_model: 埋め込みベクトルの次元数 (embed_dim)
            max_len: シーケンスの最大長。これより長いシーケンスには対応できない
        """
        super(PositionalEncoding, self).__init__()
        
        # PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        # PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # div_term = 1 / (10000^(2i/d_model)) の計算
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # peの形状を (max_len, 1, d_model) に変更し、バッチ次元に対応できるようにする
        # nn.TransformerEncoderはデフォルトで (seq_len, batch_size, feature_dim) を期待するため
        pe = pe.unsqueeze(0).transpose(0, 1) # (max_len, 1, d_model)
        
        # peをモデルのstate_dictに登録し、パラメータとして学習されないようにする
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 入力テンソル (seq_len, batch_size, d_model) または (batch_size, seq_len, d_model)
               `batch_first=True` の TransformerEncoderLayer に合わせる場合は (batch_size, seq_len, d_model)
        Returns:
            位置エンコーディングが加算された入力テンソル
        """
        # x.size(0) が seq_len に相当 (batch_first=Falseの場合)
        # または x.size(1) が seq_len に相当 (batch_first=Trueの場合)
        
        # ここでは、`MutationPredictor`内で`batch_first=True`で処理しているため、
        # `co_mutation_context`は `(batch_size, MAX_MUTATION_STEPS, embed_dim)`
        # `permute(1, 0, 2)` で `(MAX_MUTATION_STEPS, batch_size, embed_dim)` に変換してPE適用
        # `permute(1, 0, 2)` で元に戻す
        
        # xの形状が (seq_len, batch_size, d_model) であることを想定
        return x + self.pe[:x.size(0), :]

# --- MutationPredictor モデルクラスの定義 (出力層の変更) ---
# Yが個々の変異を予測するので、出力は [batch_size, sequence_length], [batch_size, vocab_size]
# Original Baseも予測するなら、もう一つ出力層を追加
class MutationPredictor(nn.Module):
    def __init__(self, embed_dim, vocab_size, sequence_length, max_co_occur, num_heads, num_encoder_layers, dropout_rate=0.1):
        super(MutationPredictor, self).__init__()
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.max_co_occur = max_co_occur

        self.original_base_embedding = nn.Embedding(vocab_size + 1, embed_dim) # +1 for PAD_BASE
        self.position_embedding = nn.Embedding(sequence_length + 2, embed_dim) # +2 for 0-indexing and PAD_POS
        self.new_base_embedding = nn.Embedding(vocab_size + 1, embed_dim)
        
        self.mutation_embedding_projection = nn.Linear(embed_dim * 3, embed_dim)

        self.positional_encoding = PositionalEncoding(embed_dim, max_len=data_config['frag_len'])

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout_rate, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # 予測ヘッド: 今回は Original Base も予測に含める
        self.predicted_original_base_head = nn.Linear(embed_dim, vocab_size) # 元の塩基
        self.position_head = nn.Linear(embed_dim, sequence_length + 1) # 位置 (+1 for 0-indexing)
        self.predicted_new_base_head = nn.Linear(embed_dim, vocab_size) # 新しい塩基

    def forward(self, mutation_history_input):
        original_base_indices = mutation_history_input[..., 0]
        position_indices = mutation_history_input[..., 1]
        new_base_indices = mutation_history_input[..., 2]

        original_emb = self.original_base_embedding(original_base_indices)
        position_emb = self.position_embedding(position_indices)
        new_base_emb = self.new_base_embedding(new_base_indices)

        single_mutation_embedding = torch.cat([original_emb, position_emb, new_base_emb], dim=-1)
        single_mutation_embedding = self.mutation_embedding_projection(single_mutation_embedding)
        
        # 共起変異の集約
        co_occurrence_mask = (original_base_indices != PAD_BASE).float()
        sum_embeddings = (single_mutation_embedding * co_occurrence_mask.unsqueeze(-1)).sum(dim=-2)
        count_embeddings = co_occurrence_mask.sum(dim=-1).unsqueeze(-1)
        co_mutation_context = sum_embeddings / (count_embeddings + 1e-9)

        # 位置エンコーディング
        co_mutation_context = self.positional_encoding(co_mutation_context.permute(1, 0, 2)).permute(1, 0, 2)

        # 各共起変異がPADかどうか: (batch_size, MAX_MUTATION_STEPS, MAX_COOCCURRING_MUTATIONS)
        is_padded_co_mutation = (original_base_indices == PAD_BASE)
        
        # Transformerのsrc_key_padding_mask
        # 各タイムステップで少なくとも1つの変異がPADなら、そのタイムステップ全体はPAD
        src_key_padding_mask = is_padded_co_mutation.all(dim=-1)

        encoder_output = self.transformer_encoder(co_mutation_context, src_key_padding_mask=src_key_padding_mask)

        # 予測ヘッド: 最後のタイムステップの出力を利用
        last_step_output = encoder_output[:, -1, :] 

        predicted_original_base_logits = self.predicted_original_base_head(last_step_output)
        predicted_position_logits = self.position_head(last_step_output)
        predicted_new_base_logits = self.predicted_new_base_head(last_step_output)

        return predicted_original_base_logits, predicted_position_logits, predicted_new_base_logits

# --- DataLoaderのcollate_fn ---
# MutationPathDataset.__getitem__ が返すのは個別のテンソルなので、DataLoaderが自動でスタックする
# その際、X_dataとY_dataで形状が異なる可能性があるため、明示的にcollate_fnを定義する
# --- DataLoaderのcollate_fn ---
def custom_collate_fn(batch):
    X_batch = torch.stack([s[0] for s in batch])
    Y_batch_samples = [s[1] for s in batch]

    # Y_batchのパディング
    # Y_batch_samplesの最初の要素の次元数で判定
    # 訓練/検証データ: Y_sample は (3,)
    # テストデータ: Y_sample は (MAX_COOCCURRING_MUTATIONS, 3)
    
    if len(Y_batch_samples[0].shape) > 1: # テストデータのY: (MAX_COOCCURRING_MUTATIONS, 3)
        Y_batch = torch.stack(Y_batch_samples)
    else: # 訓練/検証データのY: (3,)
        Y_batch = torch.stack(Y_batch_samples)
    
    return X_batch, Y_batch

# --- データセットとデータローダーの作成 ---
train_dataset = MutationPathDataset(
    train_x_split, 
    train_y_split, 
    model_config['sequence_length'], 
    data_config['max_co_occur'], 
    data_config['frag_len'],
    is_test_data=False # 訓練データはFalse
)
val_dataset = MutationPathDataset(
    valid_x_split, 
    valid_y_split, 
    model_config['sequence_length'], 
    data_config['max_co_occur'], 
    data_config['frag_len'],
    is_test_data=False # 検証データもFalse
)

test_dataloaders = {}
for k, v in test_x.items():
    test_dataset = MutationPathDataset(
        v, 
        test_y[k], # test_yは add_x_by_y を適用しない
        model_config['sequence_length'], 
        data_config['max_co_occur'], 
        data_config['frag_len'],
        is_test_data=True # テストデータはTrue
    )
    test_dataloaders[k] = DataLoader(test_dataset, batch_size=model_config['batch_size'], shuffle=False, collate_fn=custom_collate_fn)

train_dataloader = DataLoader(train_dataset, batch_size=model_config['batch_size'], shuffle=True, collate_fn=custom_collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=model_config['batch_size'], shuffle=False, collate_fn=custom_collate_fn)

# --- モデルのインスタンス化 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MutationPredictor(
    embed_dim=model_config['embed_dim'],
    vocab_size=model_config['vocab_size'],
    sequence_length=model_config['sequence_length'],
    max_co_occur=data_config['max_co_occur'],
    num_heads=model_config['num_heads'],
    num_encoder_layers=model_config['num_encoder_layers'],
    dropout_rate=0.1
).to(device)

# 損失関数と最適化手法
# Yの各要素（original_base, position, new_base）は個別のCrossEntropyLoss
criterion_orig = nn.CrossEntropyLoss(ignore_index=PAD_BASE)
criterion_pos = nn.CrossEntropyLoss(ignore_index=PAD_POS) 
criterion_new = nn.CrossEntropyLoss(ignore_index=PAD_BASE) 

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# --- 学習ループの調整 ---
def train_model(model, train_loader, val_loader, criterion_orig, criterion_pos, criterion_new, optimizer, epochs, device):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_idx, (X_batch, Y_batch) in enumerate(train_loader):
            X_batch = X_batch.to(device)
            # Y_batchは (batch_size, 3) のテンソルを想定 (add_x_by_y 適用後)
            target_original_base = Y_batch[:, 0].to(device)
            target_position = Y_batch[:, 1].to(device)
            target_new_base = Y_batch[:, 2].to(device)

            optimizer.zero_grad()
            pred_orig_logits, pred_pos_logits, pred_new_logits = model(X_batch)

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
            for X_batch, Y_batch in val_loader:
                X_batch = X_batch.to(device)
                target_original_base = Y_batch[:, 0].to(device)
                target_position = Y_batch[:, 1].to(device)
                target_new_base = Y_batch[:, 2].to(device)

                pred_orig_logits, pred_pos_logits, pred_new_logits = model(X_batch)

                loss_orig = criterion_orig(pred_orig_logits, target_original_base)
                loss_pos = criterion_pos(pred_pos_logits, target_position)
                loss_new = criterion_new(pred_new_logits, target_new_base)
                val_loss += (loss_orig + loss_pos + loss_new).item()

                # 予測の精度計算
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

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
              f"Val Orig Acc: {val_orig_acc:.4f}, Val Pos Acc: {val_pos_acc:.4f}, Val New Acc: {val_new_acc:.4f}")

# 学習の実行
print("--- PyTorchモデル学習開始 ---")
train_model(model, train_dataloader, val_dataloader, criterion_orig, criterion_pos, criterion_new, optimizer, model_config['epochs'], device)

# --- テストデータでの共起ラベル評価 ---
def evaluate_multi_label(model, test_dataloaders, device):
    model.eval()
    results = {}
    with torch.no_grad():
        for strain_name, data_loader in test_dataloaders.items():
            all_recalls = []
            all_precisions = []
            all_f1_scores = []
            
            for X_batch, Y_batch_true_raw in data_loader:
                X_batch = X_batch.to(device)
                
                pred_orig_logits, pred_pos_logits, pred_new_logits = model(X_batch)

                orig_probs = torch.softmax(pred_orig_logits, dim=1).cpu().numpy()
                pos_probs = torch.softmax(pred_pos_logits, dim=1).cpu().numpy()
                new_probs = torch.softmax(pred_new_logits, dim=1).cpu().numpy()

                for i in range(Y_batch_true_raw.shape[0]):
                    true_mutations = []
                    for j in range(Y_batch_true_raw.shape[1]):
                        orig_t, pos_t, new_t = Y_batch_true_raw[i, j].tolist()
                        if orig_t != PAD_BASE: # パディングされていない変異のみ
                            true_mutations.append((orig_t, pos_t, new_t)) 
                    
                    if not true_mutations: # 正解が空の場合、評価スキップ（または0を記録）
                        all_recalls.append(0)
                        all_precisions.append(0)
                        all_f1_scores.append(0)
                        continue

                    n_true = len(true_mutations) # 正解の共起数
                    true_set = set(true_mutations)

                    # Top-N 予測の生成 (Nは正解の共起数と同じに設定)
                    # 全ての可能な予測候補を生成
                    all_possible_predictions = []
                    for orig_idx in range(model_config['vocab_size']):
                        for pos_idx in range(model_config['sequence_length'] + 1):
                            for new_idx in range(model_config['vocab_size']):
                                if orig_idx == PAD_BASE or pos_idx == PAD_POS or new_idx == PAD_BASE:
                                    continue # パディング値は予測候補から除外
                                
                                score = orig_probs[i, orig_idx] * pos_probs[i, pos_idx] * new_probs[i, new_idx]
                                all_possible_predictions.append((score, (orig_idx, pos_idx, new_idx)))
                    
                    # スコアの高い順にソートし、Top-N個を選択
                    all_possible_predictions.sort(key=lambda x: x[0], reverse=True)
                    
                    # Top-N (N = n_true) の予測された変異
                    # モデルが出力する予測の数 n_pred も n_true と同じにする
                    n_pred = n_true 
                    top_n_preds_raw = all_possible_predictions[:n_pred]
                    predicted_set = set(pred_tuple for _, pred_tuple in top_n_preds_raw)

                    # --- Recall, Precision, F1-score の計算 ---
                    num_correct_in_top_n = len(predicted_set & true_set) # 予測と正解の共通部分の数

                    # Recall: 実際に正解であるもののうち、どれだけモデルが正しく予測できたか
                    # = (正解かつ予測された数) / (正解の総数)
                    recall = num_correct_in_top_n / n_true if n_true > 0 else 0
                    all_recalls.append(recall)

                    # Precision: モデルが正解だと予測したもののうち、実際にどれだけが正解だったか
                    # = (正解かつ予測された数) / (予測の総数)
                    precision = num_correct_in_top_n / n_pred if n_pred > 0 else 0
                    all_precisions.append(precision)

                    # F1-score
                    if precision + recall == 0:
                        f1 = 0
                    else:
                        f1 = 2 * (precision * recall) / (precision + recall)
                    all_f1_scores.append(f1)
            
            # 各ストレインの平均値を計算
            avg_recall = np.mean(all_recalls) if all_recalls else 0
            avg_precision = np.mean(all_precisions) if all_precisions else 0
            avg_f1_score = np.mean(all_f1_scores) if all_f1_scores else 0
            
            results[strain_name] = {
                'recall': avg_recall,
                'precision': avg_precision,
                'f1_score': avg_f1_score
            }
            print(f"Test Strain {strain_name}: "
                  f"Recall@{n_true if n_true > 0 else 'N/A'}: {avg_recall:.4f}, " # N_trueはサンプルごと異なるので注意
                  f"Precision@{n_pred if n_pred > 0 else 'N/A'}: {avg_precision:.4f}, "
                  f"F1-score: {avg_f1_score:.4f}")
    return results

print("\n--- テストデータでの評価 ---")
test_results = evaluate_multi_label(model, test_dataloaders, device)
print(test_results)