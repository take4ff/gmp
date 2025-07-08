# %%
# 必要なライブラリのインポート
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import importlib
from datetime import datetime
from torch.utils.data import DataLoader
from torch import nn, optim
from sklearn.metrics import classification_report
import sys

# matplotlib日本語フォント設定
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial Unicode MS', 'Hiragino Sans', 'Yu Gothic', 'Meiryo', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']

# カスタムモジュールのインポート
import module.input_mutation_path as imp
import module.get_feature as gfea
import module.mutation_transformer3 as mt
import module.make_dataset as mds
import module.evaluation2 as ev
import module.save2 as save

# デバイス設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %%
# モジュールの再読み込み（開発時のみ）
importlib.reload(imp)
importlib.reload(gfea)
importlib.reload(mt)
importlib.reload(mds)
importlib.reload(ev)
importlib.reload(save)

# %%
# 実験設定とハイパーパラメータ
# =============================================================================

# 保存ディレクトリの設定
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
folder_name = "../model/20250707_train5/bert/"
save_dir = os.path.join(folder_name, current_time)
os.makedirs(save_dir, exist_ok=True)

# モデルハイパーパラメータ
model_config = {
    'num_epochs': 30,
    'batch_size': 256,
    'd_model': 256,
    'nhead': 8,
    'num_layers': 4,
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'auto_adjust': True,  # パラメータ自動調整機能
    'use_pretrained_bert': True,  # 事前学習済みBERTを使用するかどうか
    'bert_model_name': 'bert-base-uncased',  # 使用する事前学習済みBERTモデル
    'freeze_bert_layers': 2  # 下位何層をフリーズするか（0なら全て学習）
}

# データ分割設定
data_config = {
    'test_start': 36,
    'ylen': 1,
    'val_ratio': 0.2,
    'feature_idx': 6,  # protein特徴量のインデックス
    'nmax': 100000000,
    'nmax_per_strain': 1000000
}

# データセット設定
dataset_config = {
    'strains': ['B.1.1.7','P.1','BA.2','BA.1.1','BA.1','B.1.617.2','B.1.351','B.1.1.529'],
    'usher_dir': '../usher_output/',
    'bunpu_csv': "table_heatmap/250621/table_set/table_set.csv",
    'codon_csv': 'meta_data/codon_mutation4.csv',
    'cache_dir': '../cache',  # 特徴データキャッシュ用ディレクトリ
    'filter_options': 'unique'
}

def force_print(message):
    """タイムスタンプ付きで強制出力"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")
    sys.stdout.flush()

print(f"実験設定完了 - 保存先: {save_dir}")
print(f"対象変異株: {dataset_config['strains']}")
print(f"モデル設定: d_model={model_config['d_model']}, nhead={model_config['nhead']}, num_layers={model_config['num_layers']}")

names, lengths, base_HGVS_paths = imp.input(
dataset_config['strains'], 
dataset_config['usher_dir'], 
nmax=data_config['nmax'], 
nmax_per_strain=data_config['nmax_per_strain']
)
if dataset_config['filter_options'] == 'unique':
    base_HGVS_paths = [list(item) for item in dict.fromkeys(tuple(path) for path in base_HGVS_paths)]

print(f"対象変異株のデータ数: {len(base_HGVS_paths)}")

bunpu_df = pd.read_csv(dataset_config['bunpu_csv'])
codon_df = pd.read_csv(dataset_config['codon_csv'])

def separate_data(base_HGVS_paths):
    datas = []
    for i in range(0, len(base_HGVS_paths)):
        base_HGVS_path = base_HGVS_paths[i]


        data_ts = {}
        for i in range(len(base_HGVS_path)):
            for mutation in base_HGVS_path[i].split(','):
                if mutation != '':
                    if data_ts.get(i+1) is None:
                        data_ts[i+1] = []
                    data_ts[i+1].append(mutation)
            
        datas.append(data_ts)
    return datas

data = separate_data(base_HGVS_paths)

# データ分割の実行
train_x, train_y, val_x, val_y, test_x, test_y = mds.create_time_aware_split_modified(
    data, data_config['test_start'], data_config['ylen'], data_config['val_ratio']
)

def extract_protein(y, codon_df, bunpu_df):
    # タイムステップごとにプロテイン特徴量を抽出
    new_y = []
    for mutations in y:
        proteins = []
        for mutation in mutations:
            temp, temp, protein, temp, temp = gfea.Mutation_features(mutation, codon_df, bunpu_df)
            proteins.append(protein)
        new_y.append(proteins)
    return new_y

# プロテイン特徴量の抽出
train_y_protein = extract_protein(train_y, codon_df, bunpu_df)
val_y_protein = extract_protein(val_y, codon_df, bunpu_df)

# データとラベルの結合
train_x2, train_y2 = mds.add_x_by_y(train_x, train_y_protein)
val_x2, val_y2 = mds.add_x_by_y(val_x, val_y_protein)

print(f"データ分割完了:")
print(f"  訓練データ: {len(train_x2)} サンプル")
print(f"  検証データ: {len(val_x2)} サンプル")
print(f"  テストデータ: {len(test_x)} タイムステップ")
# %%
# BERT風の語彙構築とデータセット作成
# =============================================================================

def create_mutation_vocabulary():
    """塩基変異パターンのみの語彙を作成"""
    base = ['A', 'C', 'G', 'T']
    
    # 特別トークンを追加
    vocab = {
        '<PAD>': 0,
        '<UNK>': 1,
        '<CLS>': 2,
        '<SEP>': 3
    }
    
    # 塩基変異パターンを生成 (例: A1C, A1G, A1T, ...)
    # 4種類の塩基 × 3種類の変異 × 30,000位置 = 360,000パターン
    mutations = []
    for b1 in base:
        for b2 in base:
            if b1 != b2:  # 同じ塩基への変異は除外
                for pos in range(1, 30001):  # 位置1-30000
                    mutations.append(f"{b1}{pos}{b2}")
    
    # 変異を語彙に追加
    for i, mutation in enumerate(sorted(mutations), start=4):
        vocab[mutation] = i
    
    return vocab

def create_protein_vocabulary():
    """プロテイン語彙を作成"""
    protein_names = [
        "non_coding1", "nsp1", "nsp2", "nsp3", "nsp4", "nsp5", "nsp6", "nsp7", "nsp8", "nsp9", "nsp10",
        "nsp12", "nsp13", "nsp14", "nsp15", "nsp16", "non_coding2", "S", "non_coding3", "ORF3a", 
        "non_coding4", "E", "non_coding5", "M", "non_coding6", "ORF6", "non_coding7", "ORF7a", 
        "ORF7b", "non_coding8", "ORF8", "non_coding9", "N", "non_coding10", "ORF10", "non_coding11"
    ]
    
    vocab = {
        '<PAD>': 0,
        '<UNK>': 1
    }
    
    for i, protein in enumerate(protein_names, start=2):
        vocab[protein] = i
    
    return vocab

# 語彙辞書を作成
print("語彙辞書を構築中...")
mutation_vocab = create_mutation_vocabulary()
protein_vocab = create_protein_vocabulary()

print(f"変異語彙サイズ: {len(mutation_vocab):,}")
print(f"プロテイン語彙サイズ: {len(protein_vocab):,}")

# データセットクラスの定義
class MutationBERTDataset(torch.utils.data.Dataset):
    def __init__(self, x_data, y_data, mutation_vocab, protein_vocab, max_length=512):
        self.x_data = x_data
        self.y_data = y_data
        self.mutation_vocab = mutation_vocab
        self.protein_vocab = protein_vocab
        self.max_length = max_length
        
        # ラベルエンコーダーの作成（簡易版）
        all_proteins = []
        for proteins in y_data:
            all_proteins.extend(proteins)
        
        unique_proteins = sorted(list(set(all_proteins)))
        self.classes_ = unique_proteins
        self.protein_to_idx = {protein: idx for idx, protein in enumerate(unique_proteins)}
        self.num_classes = len(unique_proteins)
        
    def __len__(self):
        return len(self.x_data)
    
    def __getitem__(self, idx):
        # 入力シーケンスの構築
        x_sample = self.x_data[idx]
        y_sample = self.y_data[idx]
        
        # 変異シーケンスを構築
        mutation_tokens = [self.mutation_vocab['<CLS>']]
        
        # x_sampleが辞書かリストかを判定
        if isinstance(x_sample, dict):
            # 辞書の場合：{タイムステップ: [変異リスト]}
            for ts, mutations in sorted(x_sample.items()):
                for mutation in mutations:
                    token_id = self.mutation_vocab.get(mutation, self.mutation_vocab['<UNK>'])
                    mutation_tokens.append(token_id)
                mutation_tokens.append(self.mutation_vocab['<SEP>'])
        elif isinstance(x_sample, list):
            # リストの場合：[変異1, 変異2, ...]
            for mutation in x_sample:
                token_id = self.mutation_vocab.get(mutation, self.mutation_vocab['<UNK>'])
                mutation_tokens.append(token_id)
            mutation_tokens.append(self.mutation_vocab['<SEP>'])
        else:
            # その他の場合はエラー
            raise ValueError(f"Unsupported x_sample type: {type(x_sample)}")
        
        # パディングまたは切り詰め
        if len(mutation_tokens) > self.max_length:
            mutation_tokens = mutation_tokens[:self.max_length]
        else:
            padding_length = self.max_length - len(mutation_tokens)
            mutation_tokens.extend([self.mutation_vocab['<PAD>']] * padding_length)
        
        # アテンションマスク
        attention_mask = [1 if token != self.mutation_vocab['<PAD>'] else 0 for token in mutation_tokens]
        
        # ラベル（最初のプロテインを使用）
        label = self.protein_to_idx[y_sample[0]]
        
        return {
            'input_ids': torch.tensor(mutation_tokens, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# データセットの作成
print("データセットを作成中...")
train_dataset = MutationBERTDataset(train_x2, train_y2, mutation_vocab, protein_vocab)
val_dataset = MutationBERTDataset(val_x2, val_y2, mutation_vocab, protein_vocab, train_dataset.max_length)

# データローダー
train_loader = DataLoader(train_dataset, batch_size=model_config['batch_size'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=model_config['batch_size'], shuffle=False)

print(f"訓練データセット: {len(train_dataset)} サンプル")
print(f"検証データセット: {len(val_dataset)} サンプル")
print(f"クラス数: {train_dataset.num_classes}")
print(f"最大シーケンス長: {train_dataset.max_length}")
print(f"クラス: {train_dataset.classes_}")

# %%
# BERT風のTransformerモデルの定義
# =============================================================================

class MutationBERTModel(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=4, num_classes=36, max_seq_length=512):
        super(MutationBERTModel, self).__init__()
        
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        
        # Embedding層
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_length, d_model)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 分類ヘッド
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, num_classes)
        )
        
    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_length = input_ids.shape
        
        # Position IDs
        position_ids = torch.arange(seq_length, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        token_embeds = self.token_embedding(input_ids)
        position_embeds = self.position_embedding(position_ids)
        embeddings = token_embeds + position_embeds
        
        # Attention mask for transformer (inverted)
        if attention_mask is not None:
            # True for masked positions
            transformer_mask = (attention_mask == 0)
        else:
            transformer_mask = None
        
        # Transformer
        hidden_states = self.transformer(embeddings, src_key_padding_mask=transformer_mask)
        
        # CLSトークンの出力を使用（最初のトークン）
        cls_output = hidden_states[:, 0, :]
        
        # 分類
        logits = self.classifier(cls_output)
        
        return logits

class PretrainedBERTModel(nn.Module):
    def __init__(self, bert_model_name, num_classes, mutation_vocab, freeze_layers=0):
        super(PretrainedBERTModel, self).__init__()
        
        try:
            import transformers
            from transformers import AutoModel, AutoConfig, AutoTokenizer
            self.use_pretrained = True
            
            # BERTのトークナイザーと語彙を取得
            print(f"🤗 transformersライブラリを使用してBERTモデルを読み込みます: {bert_model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
            original_vocab_size = len(self.tokenizer)
            print(f"元のBERT語彙サイズ: {original_vocab_size:,}")
            
            # カスタム変異トークンをBERT語彙に追加
            print("変異トークンをBERT語彙に追加中...")
            mutation_tokens = []
            for token, _ in sorted(mutation_vocab.items(), key=lambda x: x[1]):
                if token not in ['<PAD>', '<UNK>', '<CLS>', '<SEP>']:  # 特別トークンは除く
                    mutation_tokens.append(token)
            
            print(f"追加するトークン数: {len(mutation_tokens):,}")
            
            # 新しいトークンを追加（バッチ処理で高速化）
            num_added_tokens = self.tokenizer.add_tokens(mutation_tokens)
            print(f"BERT語彙に {num_added_tokens:,} 個の変異トークンを追加しました")
            print(f"語彙サイズ: {original_vocab_size:,} → {len(self.tokenizer):,}")
            
            # 語彙マッピングを作成（カスタム→BERT）- 高速化版
            print("語彙マッピングを作成中...")
            self.vocab_mapping = {}
            
            # 特別トークンの処理
            special_tokens = {
                '<PAD>': self.tokenizer.pad_token_id or self.tokenizer.unk_token_id,
                '<UNK>': self.tokenizer.unk_token_id or 0,
                '<CLS>': self.tokenizer.cls_token_id or self.tokenizer.unk_token_id,
                '<SEP>': self.tokenizer.sep_token_id or self.tokenizer.unk_token_id
            }
            
            for token, custom_id in mutation_vocab.items():
                if token in special_tokens:
                    self.vocab_mapping[custom_id] = special_tokens[token]
                else:
                    # 変異トークンは新しく追加されているので、直接IDを取得
                    bert_id = self.tokenizer.convert_tokens_to_ids(token)
                    self.vocab_mapping[custom_id] = bert_id
            
            print(f"語彙マッピング完了: {len(self.vocab_mapping):,} トークン")
            
            # 高速変換用のテンソルマッピングを作成
            print("高速変換テーブルを作成中...")
            max_custom_id = max(mutation_vocab.values())
            self.vocab_mapping_tensor = torch.full(
                (max_custom_id + 1,), 
                self.tokenizer.unk_token_id, 
                dtype=torch.long
            )
            
            for custom_id, bert_id in self.vocab_mapping.items():
                self.vocab_mapping_tensor[custom_id] = bert_id
                
            # テンソルをパラメータとして登録（GPUに自動移動）
            self.register_buffer('vocab_mapping_tensor', self.vocab_mapping_tensor)
            print("高速変換テーブル作成完了")
            
            # BERTモデルの設定を取得
            config = AutoConfig.from_pretrained(bert_model_name)
            self.d_model = config.hidden_size
            
            # BERTモデルを読み込み
            self.bert = AutoModel.from_pretrained(bert_model_name)
            
            # 語彙サイズが変更されたので、埋め込み層をリサイズ
            self.bert.resize_token_embeddings(len(self.tokenizer))
            print(f"BERT埋め込み層をリサイズ: {original_vocab_size:,} → {len(self.tokenizer):,}")
            
            # 下位層をフリーズ
            if freeze_layers > 0:
                for i, layer in enumerate(self.bert.encoder.layer):
                    if i < freeze_layers:
                        for param in layer.parameters():
                            param.requires_grad = False
                print(f"BERT下位{freeze_layers}層をフリーズしました")
            
            # 分類ヘッド
            self.classifier = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(self.d_model, self.d_model // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.d_model // 2, num_classes)
            )
            
        except ImportError:
            print("⚠️ transformersライブラリが見つかりません。オリジナルBERTモデルを使用します。")
            self.use_pretrained = False
            
    def forward(self, input_ids, attention_mask=None):
        if not self.use_pretrained:
            raise RuntimeError("事前学習済みBERTが利用できません")
        
        # カスタム語彙IDをBERT語彙IDに変換（高速化版）
        # テンソル全体を一度に変換
        input_ids_flat = input_ids.flatten()
        
        # 範囲外のIDをunk_token_idにクランプ
        valid_mask = input_ids_flat < len(self.vocab_mapping_tensor)
        clamped_ids = torch.clamp(input_ids_flat, 0, len(self.vocab_mapping_tensor)-1)
        
        bert_input_ids_flat = torch.where(
            valid_mask,
            self.vocab_mapping_tensor[clamped_ids],
            torch.tensor(self.tokenizer.unk_token_id, device=input_ids.device, dtype=input_ids.dtype)
        )
        
        bert_input_ids = bert_input_ids_flat.view(input_ids.shape)
            
        # BERT forward
        outputs = self.bert(input_ids=bert_input_ids, attention_mask=attention_mask)
        
        # CLSトークンの出力を使用
        cls_output = outputs.last_hidden_state[:, 0, :]
        
        # 分類
        logits = self.classifier(cls_output)
        
        return logits

def create_model(model_config, vocab_size, num_classes, max_seq_length, mutation_vocab=None):
    """モデル設定に基づいてモデルを作成"""
    
    if model_config['use_pretrained_bert']:
        print("🤗 事前学習済みBERTモデルを使用します")
        print(f"モデル: {model_config['bert_model_name']}")
        print(f"フリーズ層数: {model_config['freeze_bert_layers']}")
        
        if mutation_vocab is None:
            raise ValueError("事前学習済みBERTを使用する場合、mutation_vocabが必要です")
        
        try:
            model = PretrainedBERTModel(
                bert_model_name=model_config['bert_model_name'],
                num_classes=num_classes,
                mutation_vocab=mutation_vocab,
                freeze_layers=model_config['freeze_bert_layers']
            )
            
            if not model.use_pretrained:
                # フォールバック: オリジナルモデルを使用
                print("🔄 オリジナルBERTモデルにフォールバック")
                model = MutationBERTModel(
                    vocab_size=vocab_size,
                    d_model=model_config['d_model'],
                    nhead=model_config['nhead'],
                    num_layers=model_config['num_layers'],
                    num_classes=num_classes,
                    max_seq_length=max_seq_length
                )
                
        except Exception as e:
            print(f"⚠️ 事前学習済みBERTの読み込みに失敗: {e}")
            print("🔄 オリジナルBERTモデルにフォールバック")
            model = MutationBERTModel(
                vocab_size=vocab_size,
                d_model=model_config['d_model'],
                nhead=model_config['nhead'],
                num_layers=model_config['num_layers'],
                num_classes=num_classes,
                max_seq_length=max_seq_length
            )
    else:
        print("🔧 オリジナルBERTモデルを使用します")
        model = MutationBERTModel(
            vocab_size=vocab_size,
            d_model=model_config['d_model'],
            nhead=model_config['nhead'],
            num_layers=model_config['num_layers'],
            num_classes=num_classes,
            max_seq_length=max_seq_length
        )
    
    return model

# %%
# モデル初期化と訓練設定
# =============================================================================

print("モデルを初期化中...")

# モデルの作成
model = create_model(
    model_config=model_config,
    vocab_size=len(mutation_vocab),
    num_classes=train_dataset.num_classes,
    max_seq_length=train_dataset.max_length,
    mutation_vocab=mutation_vocab
).to(device)

# 損失関数とオプティマイザー
criterion = nn.CrossEntropyLoss()

# 事前学習済みBERTの場合は学習率を調整
if model_config['use_pretrained_bert'] and hasattr(model, 'use_pretrained') and model.use_pretrained:
    # BERTパラメータと分類ヘッドで異なる学習率を設定
    bert_params = []
    classifier_params = []
    
    for name, param in model.named_parameters():
        if 'bert' in name:
            bert_params.append(param)
        else:
            classifier_params.append(param)
    
    optimizer = optim.AdamW([
        {'params': bert_params, 'lr': model_config['learning_rate'] * 0.1},  # BERTは小さい学習率
        {'params': classifier_params, 'lr': model_config['learning_rate']}    # 分類ヘッドは通常の学習率
    ], weight_decay=model_config['weight_decay'])
    
    print(f"差分学習率設定:")
    print(f"  BERT層: {model_config['learning_rate'] * 0.1:.2e}")
    print(f"  分類ヘッド: {model_config['learning_rate']:.2e}")
else:
    optimizer = optim.AdamW(model.parameters(), lr=model_config['learning_rate'], weight_decay=model_config['weight_decay'])
    print(f"統一学習率: {model_config['learning_rate']:.2e}")

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

print(f"モデルのパラメータ数: {sum(p.numel() for p in model.parameters()):,}")
print(f"学習可能パラメータ数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
print(f"語彙サイズ: {len(mutation_vocab):,}")
print(f"クラス数: {train_dataset.num_classes}")
print(f"最大シーケンス長: {train_dataset.max_length}")

# モデルタイプの表示
if hasattr(model, 'use_pretrained') and model.use_pretrained:
    print(f"モデルタイプ: 事前学習済みBERT ({model_config['bert_model_name']})")
    if hasattr(model, 'd_model') and model.d_model:
        print(f"隠れ層サイズ: {model.d_model}")
    print(f"フリーズ層数: {model_config.get('freeze_bert_layers', 0)}")
else:
    print(f"モデルタイプ: オリジナルBERT")
    print(f"d_model: {model_config['d_model']}, nhead: {model_config['nhead']}, num_layers: {model_config['num_layers']}")

# %%
# 訓練関数の定義
# =============================================================================

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(train_loader), correct / total

def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
    
    return total_loss / len(val_loader), correct / total, all_preds, all_targets

# %%
# モデル訓練
# =============================================================================

# モデルの訓練
best_val_acc = 0
best_model_state = None
train_losses = []
val_losses = []
train_accs = []
val_accs = []
epoch_times = []

print("訓練を開始します...")
training_start_time = time.time()

try:
    for epoch in range(model_config['num_epochs']):
        epoch_start_time = time.time()
        force_print(f"\nEpoch {epoch+1}/{model_config['num_epochs']}")
        
        # 訓練
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # 検証
        val_loss, val_acc, val_preds, val_targets = evaluate(model, val_loader, criterion, device)
        
        # スケジューラを更新
        scheduler.step(val_loss)
        
        # エポック終了時間計算
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_duration)
        
        # 結果を記録
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # 時間情報を含む出力
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"Epoch Time: {epoch_duration:.2f}s ({epoch_duration/60:.1f}min)")
        
        # 累積時間と推定残り時間
        total_elapsed = sum(epoch_times)
        avg_epoch_time = total_elapsed / len(epoch_times)
        remaining_epochs = model_config['num_epochs'] - (epoch + 1)
        estimated_remaining = avg_epoch_time * remaining_epochs
        
        print(f"Elapsed: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min), "
              f"ETA: {estimated_remaining:.1f}s ({estimated_remaining/60:.1f}min)")
        
        # 最良モデルを保存
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            print(f"新しい最良モデル (Val Acc: {val_acc:.4f})")

    # 訓練完了時の統計
    training_end_time = time.time()
    total_training_time = training_end_time - training_start_time
    
    print(f"\n=== 訓練完了! ===")
    print(f"最良検証精度: {best_val_acc:.4f}")
    print(f"総訓練時間: {total_training_time:.1f}s ({total_training_time/60:.1f}min)")
    print(f"平均エポック時間: {np.mean(epoch_times):.2f}s")
    print(f"最速エポック: {min(epoch_times):.2f}s")
    print(f"最遅エポック: {max(epoch_times):.2f}s")

    # 最良モデルをロード
    if best_model_state:
        model.load_state_dict(best_model_state)
        
except Exception as e:
    print(f"訓練中にエラーが発生しました: {e}")
    import traceback
    traceback.print_exc()

# %%
# 訓練結果の分析と可視化
# =============================================================================

# 訓練結果の可視化
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(train_accs, label='Train Acc')
plt.plot(val_accs, label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 3, 3)
# 検証データでの最終評価
val_loss, val_acc, val_preds, val_targets = evaluate(model, val_loader, criterion, device)
print(f"最終検証精度: {val_acc:.4f}")

# クラス名とラベルの対応を確認
class_names = train_dataset.classes_
print(f"全クラス数: {len(class_names)}")
print(f"検証データに含まれるユニークなクラス数: {len(set(val_targets))}")
print(f"予測に含まれるユニークなクラス数: {len(set(val_preds))}")

# 実際に使用されているクラスのみを取得
unique_labels = sorted(set(val_targets) | set(val_preds))
actual_class_names = [class_names[i] for i in unique_labels]

print(f"実際に使用されているクラス: {actual_class_names}")

# 分類レポート（実際に使用されているクラスのみ）
print("\n分類レポート:")
print(classification_report(
    val_targets, 
    val_preds, 
    labels=unique_labels,
    target_names=actual_class_names, 
    zero_division=0
))

plt.tight_layout()
plt.show()

# %%
# 簡易テスト評価とモデル保存
# =============================================================================

# 保存ディレクトリの作成
os.makedirs(save_dir, exist_ok=True)

print("=== モデルと結果の保存を開始 ===")

# 1. モデルの保存
model_save_path = os.path.join(save_dir, "best_model.pth")
save_data = {
    'model_state_dict': best_model_state,
    'model_config': model_config,
    'mutation_vocab': mutation_vocab,
    'protein_vocab': protein_vocab,
    'label_encoder': train_dataset.protein_to_idx,
    'num_classes': train_dataset.num_classes,
    'max_length': train_dataset.max_length,
    'model_type': 'pretrained_bert' if (hasattr(model, 'use_pretrained') and model.use_pretrained) else 'original_bert'
}

# 事前学習済みBERTの場合は追加情報を保存
if hasattr(model, 'use_pretrained') and model.use_pretrained:
    save_data['bert_model_name'] = model_config['bert_model_name']
    save_data['freeze_layers'] = model_config['freeze_bert_layers']

torch.save(save_data, model_save_path)
print(f"モデル保存完了: {model_save_path}")

# 2. 訓練履歴の保存
import json

results = {
    'train_losses': train_losses,
    'val_losses': val_losses,
    'train_accs': train_accs,
    'val_accs': val_accs,
    'best_val_acc': best_val_acc,
    'total_training_time': total_training_time,
    'model_config': model_config,
    'data_config': data_config,
    'dataset_config': dataset_config
}

results_path = os.path.join(save_dir, "training_results.json")
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"訓練結果保存完了: {results_path}")

# 3. 訓練グラフの保存
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Train Acc')
plt.plot(val_accs, label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.tight_layout()
plot_path = os.path.join(save_dir, "training_history.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"訓練グラフ保存完了: {plot_path}")

print(f"\n=== 全ての保存が完了しました ===")
print(f"保存先: {os.path.abspath(save_dir)}")
print(f"最良検証精度: {best_val_acc:.4f}")
print(f"総エポック数: {model_config['num_epochs']}")
print(f"語彙サイズ: {len(mutation_vocab):,}")
print(f"クラス数: {train_dataset.num_classes}")

print("\n=== 実験完了 ===")

# %%
# テストデータ評価機能の追加
# =============================================================================

def evaluate_test_data_timestep(model, test_x, test_y, mutation_vocab, protein_vocab, train_dataset, device):
    """
    タイムステップごとにテストデータでの予測精度を評価
    train4のevaluate_test_data_timestepと同様の計算方法を実装
    
    Args:
        model: 訓練済みモデル
        test_x: テストデータの入力（タイムステップ辞書）
        test_y: テストデータのラベル（タイムステップ辞書）
        mutation_vocab: 変異語彙辞書
        protein_vocab: プロテイン語彙辞書
        train_dataset: 訓練データセット（クラス情報含む）
        device: デバイス
    """
    print("\n=== タイムステップごとのテスト評価 ===")
    
    timestep_results = {}
    
    # 訓練データセットのクラス情報を取得
    train_classes = set(train_dataset.classes_)
    train_protein_to_idx = train_dataset.protein_to_idx
    
    # プロテイン特徴量抽出関数
    def extract_protein_from_test_labels(test_labels, codon_df, bunpu_df):
        proteins = []
        for mutations in test_labels:
            seq_proteins = []
            for mutation in mutations:
                _, _, protein, _, _ = gfea.Mutation_features(mutation, codon_df, bunpu_df)
                seq_proteins.append(protein)
            proteins.append(seq_proteins)
        return proteins
    
    for timestep in sorted(test_x.keys()):
        print(f"\nタイムステップ {timestep} の評価中...")
        
        # テストデータの準備
        test_sequences = test_x[timestep]
        test_labels = test_y[timestep]
        
        if len(test_sequences) == 0:
            print(f"  タイムステップ {timestep}: データなし")
            continue
        
        # プロテイン名を抽出
        test_y_protein = extract_protein_from_test_labels(test_labels, codon_df, bunpu_df)
        test_x_expanded, test_y_expanded = mds.add_x_by_y(test_sequences, test_y_protein)
        
        # 未知のクラスをフィルタリング
        filtered_x = []
        filtered_y = []
        filtered_protein_labels = []
        
        for i, (x, y, orig_label) in enumerate(zip(test_x_expanded, test_y_expanded, test_y_protein)):
            label = y[0] if isinstance(y, list) and len(y) > 0 else y
            if str(label) in train_classes:
                filtered_x.append(x)
                filtered_y.append(y)
                if i < len(test_y_protein):
                    filtered_protein_labels.append(orig_label)
        
        if len(filtered_x) == 0:
            print(f"  タイムステップ {timestep}: 既知のクラスがありません")
            continue
        
        print(f"  フィルタリング: {len(test_x_expanded)} -> {len(filtered_x)} サンプル")
        
        # テストデータセットの作成
        test_dataset = MutationBERTDataset(
            filtered_x, 
            filtered_y, 
            mutation_vocab, 
            protein_vocab, 
            train_dataset.max_length
        )
        
        # テストデータローダー
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
        
        # 評価実行
        model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        criterion = torch.nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs  # PretrainedBERTModelは既にlogitsを返している
                
                loss = criterion(logits, labels)
                total_loss += loss.item()
                
                predictions = torch.argmax(logits, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
        
        test_loss = total_loss / len(test_loader)
        
        # シーケンス単位での予測と真のラベルを準備
        predictions_per_sequence = []
        true_labels_per_sequence = []
        
        for seq_idx, orig_labels in enumerate(filtered_protein_labels):
            if isinstance(orig_labels, list):
                true_labels_set = set(orig_labels)
            else:
                true_labels_set = {orig_labels}
            
            true_labels_per_sequence.append(true_labels_set)
            
            # 対応する予測を取得（範囲チェック付き）
            if seq_idx < len(all_predictions):
                pred_idx = all_predictions[seq_idx]
                if 0 <= pred_idx < len(train_dataset.classes_):
                    pred_class_name = train_dataset.classes_[pred_idx]
                    predictions_per_sequence.append([pred_class_name])
                else:
                    print(f"警告: 予測インデックス {pred_idx} が範囲外 (0-{len(train_dataset.classes_)-1})")
                    predictions_per_sequence.append(['<UNK>'])
            else:
                predictions_per_sequence.append(['<UNK>'])
        
        # シーケンス単位の精度計算（2つの指標）
        strict_sequence_accuracy = calculate_strict_sequence_accuracy(
            predictions_per_sequence, true_labels_per_sequence
        )
        flexible_sequence_accuracy = calculate_flexible_sequence_accuracy(
            predictions_per_sequence, true_labels_per_sequence
        )
        
        # 結果を保存
        timestep_results[timestep] = {
            'samples': len(test_sequences),
            'expanded_samples': len(test_x_expanded),
            'filtered_samples': len(filtered_x),
            'loss': test_loss,
            'strict_sequence_accuracy': strict_sequence_accuracy,
            'flexible_sequence_accuracy': flexible_sequence_accuracy,
            'predictions': all_predictions,
            'targets': all_targets,
            'original_labels': filtered_protein_labels,
            'predictions_per_sequence': predictions_per_sequence,
            'true_labels_per_sequence': [list(labels) for labels in true_labels_per_sequence]
        }
        
        print(f"  サンプル数: {len(test_sequences)} (展開後: {len(test_x_expanded)}, フィルタ後: {len(filtered_x)})")
        print(f"  Loss: {test_loss:.4f}")
        print(f"  厳密シーケンス精度: {strict_sequence_accuracy:.4f}")
        print(f"  柔軟シーケンス精度: {flexible_sequence_accuracy:.4f}")
    
    return timestep_results

def calculate_strict_sequence_accuracy(predictions_per_sequence, true_labels_per_sequence):
    """
    厳密正解率：単一ラベルの場合のみ正解、複数ラベルは必ず不正解
    """
    correct_sequences = 0
    total_sequences = len(true_labels_per_sequence)
    
    for pred_list, true_set in zip(predictions_per_sequence, true_labels_per_sequence):
        pred = pred_list[0]  # 単一予測
        
        # 真のラベルが1つの場合のみ正解の可能性あり
        if len(true_set) == 1 and pred in true_set:
            correct_sequences += 1
        # 複数ラベルの場合は必ず不正解
    
    return correct_sequences / total_sequences if total_sequences > 0 else 0.0

def calculate_flexible_sequence_accuracy(predictions_per_sequence, true_labels_per_sequence):
    """
    柔軟正解率：予測が真のラベルのいずれかに含まれていれば正解
    """
    correct_sequences = 0
    total_sequences = len(true_labels_per_sequence)
    
    for pred_list, true_set in zip(predictions_per_sequence, true_labels_per_sequence):
        pred = pred_list[0]  # 単一予測
        
        # 予測が真のラベルセットに含まれていれば正解
        if pred in true_set:
            correct_sequences += 1
    
    return correct_sequences / total_sequences if total_sequences > 0 else 0.0

def report_timestep_results(timestep_results):
    """
    タイムステップ評価結果のサマリーレポートを表示
    """
    print("\n=== タイムステップ評価結果サマリー ===")
    
    total_samples = 0
    total_filtered = 0
    weighted_strict_acc = 0.0
    weighted_flexible_acc = 0.0
    
    print(f"{'Timestep':<10} {'Samples':<8} {'Filtered':<8} {'Loss':<8} {'Strict Acc':<12} {'Flexible Acc':<12}")
    print("-" * 70)
    
    for timestep in sorted(timestep_results.keys()):
        result = timestep_results[timestep]
        samples = result['filtered_samples']
        loss = result['loss']
        strict_acc = result['strict_sequence_accuracy']
        flexible_acc = result['flexible_sequence_accuracy']
        
        print(f"{timestep:<10} {result['samples']:<8} {samples:<8} {loss:<8.4f} {strict_acc:<12.4f} {flexible_acc:<12.4f}")
        
        total_samples += result['samples']
        total_filtered += samples
        weighted_strict_acc += strict_acc * samples
        weighted_flexible_acc += flexible_acc * samples
    
    if total_filtered > 0:
        weighted_strict_acc /= total_filtered
        weighted_flexible_acc /= total_filtered
    
    print("-" * 70)
    print(f"{'Total':<10} {total_samples:<8} {total_filtered:<8} {'---':<8} {weighted_strict_acc:<12.4f} {weighted_flexible_acc:<12.4f}")
    print()
    print(f"全体加重平均 - 厳密精度: {weighted_strict_acc:.4f}, 柔軟精度: {weighted_flexible_acc:.4f}")

# %%
# テストデータの評価実行（オプション）
# =============================================================================

# テストデータ評価の実行
print("\n=== テストデータ評価の実行 ===")

# モデルを最良の重みに復元
model.load_state_dict(best_model_state)

# テストデータ評価
timestep_results = evaluate_test_data_timestep(
    model=model,
    test_x=test_x,
    test_y=test_y,
    mutation_vocab=mutation_vocab,
    protein_vocab=protein_vocab,
    train_dataset=train_dataset,
    device=device
)

# 結果レポート
report_timestep_results(timestep_results)

# テストデータ正解率の可視化
def plot_test_accuracy_by_timestep(timestep_results, save_dir):
    """
    タイムステップ別のテストデータ正解率をグラフ化
    """
    if not timestep_results:
        print("テスト結果がありません。グラフをスキップします。")
        return
    
    timesteps = sorted(timestep_results.keys())
    strict_accuracies = []
    flexible_accuracies = []
    sample_counts = []
    filtered_counts = []
    
    for ts in timesteps:
        result = timestep_results[ts]
        strict_accuracies.append(result['strict_sequence_accuracy'])
        flexible_accuracies.append(result['flexible_sequence_accuracy'])
        sample_counts.append(result['samples'])
        filtered_counts.append(result['filtered_samples'])
    
    # グラフの作成
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. 正解率の推移
    ax1.plot(timesteps, strict_accuracies, 'o-', label='Strict Accuracy', color='red', linewidth=2)
    ax1.plot(timesteps, flexible_accuracies, 's-', label='Flexible Accuracy', color='blue', linewidth=2)
    ax1.set_xlabel('Timestep')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Test Accuracy by Timestep')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # 2. サンプル数の推移
    ax2.bar(timesteps, sample_counts, alpha=0.7, label='Original Samples', color='lightblue')
    ax2.bar(timesteps, filtered_counts, alpha=0.9, label='Filtered Samples', color='darkblue')
    ax2.set_xlabel('Timestep')
    ax2.set_ylabel('Sample Count')
    ax2.set_title('Sample Count by Timestep')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 正解率とサンプル数の相関（厳密正解率）
    ax3.scatter(filtered_counts, strict_accuracies, alpha=0.7, s=60, color='red')
    ax3.set_xlabel('Filtered Sample Count')
    ax3.set_ylabel('Strict Accuracy')
    ax3.set_title('Sample Count vs Strict Accuracy')
    ax3.grid(True, alpha=0.3)
    
    # 各点にタイムステップラベルを追加
    for i, ts in enumerate(timesteps):
        ax3.annotate(f'T{ts}', (filtered_counts[i], strict_accuracies[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # 4. 正解率とサンプル数の相関（柔軟正解率）
    ax4.scatter(filtered_counts, flexible_accuracies, alpha=0.7, s=60, color='blue')
    ax4.set_xlabel('Filtered Sample Count')
    ax4.set_ylabel('Flexible Accuracy')
    ax4.set_title('Sample Count vs Flexible Accuracy')
    ax4.grid(True, alpha=0.3)
    
    # 各点にタイムステップラベルを追加
    for i, ts in enumerate(timesteps):
        ax4.annotate(f'T{ts}', (filtered_counts[i], flexible_accuracies[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.tight_layout()
    
    # グラフの保存
    test_plot_path = os.path.join(save_dir, "test_accuracy_by_timestep.png")
    plt.savefig(test_plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"テスト正解率グラフ保存完了: {test_plot_path}")
    
    # 統計情報の表示
    print(f"\n=== テスト正解率統計 ===")
    print(f"厳密正解率 - 最高: {max(strict_accuracies):.4f}, 最低: {min(strict_accuracies):.4f}, 平均: {np.mean(strict_accuracies):.4f}")
    print(f"柔軟正解率 - 最高: {max(flexible_accuracies):.4f}, 最低: {min(flexible_accuracies):.4f}, 平均: {np.mean(flexible_accuracies):.4f}")
    print(f"タイムステップ数: {len(timesteps)}")
    print(f"総フィルタ後サンプル数: {sum(filtered_counts)}")

# テストデータ正解率の可視化を実行
plot_test_accuracy_by_timestep(timestep_results, save_dir)

# ===== train4と同等の保存機能を追加 =====

# 結果の保存（詳細版）
import json
import pickle

# JSON保存用にnumpy配列を変換
for timestep in timestep_results:
    result = timestep_results[timestep]
    if 'predictions' in result:
        result['predictions'] = [int(x) for x in result['predictions']]
    if 'targets' in result:
        result['targets'] = [int(x) for x in result['targets']]

test_results_path = os.path.join(save_dir, "test_results_timestep.json")
with open(test_results_path, 'w') as f:
    json.dump(timestep_results, f, indent=2, ensure_ascii=False)
print(f"テスト結果保存完了: {test_results_path}")

# 4. 設定ファイルの保存 (config.json)
config_data = {
    'model_config': model_config,
    'data_config': data_config,
    'dataset_config': dataset_config,
    'feature_mask': None,  # BERT風モデルでは使用しない
    'model_type': 'bert_style',
    'mutation_vocab_size': len(mutation_vocab),
    'protein_vocab_size': len(protein_vocab),
    'num_classes': train_dataset.num_classes,
    'max_length': train_dataset.max_length,
    'class_names': train_dataset.classes_,
    'training_statistics': {
        'best_val_acc': best_val_acc,
        'total_training_time': total_training_time,
        'final_train_acc': train_accs[-1] if train_accs else 0,
        'final_val_acc': val_accs[-1] if val_accs else 0,
        'epochs_completed': len(train_accs)
    }
}

config_path = os.path.join(save_dir, "config.json")
with open(config_path, 'w') as f:
    json.dump(config_data, f, indent=2, ensure_ascii=False)
print(f"設定ファイル保存完了: {config_path}")

# 5. 語彙辞書の保存 (vocabularies.pkl)
vocab_data = {
    'mutation_vocab': mutation_vocab,
    'protein_vocab': protein_vocab,
    'label_encoder': train_dataset.protein_to_idx,
    'reverse_label_encoder': {v: k for k, v in train_dataset.protein_to_idx.items()},
    'class_names': train_dataset.classes_,
    'vocab_type': 'bert_style'
}

vocab_path = os.path.join(save_dir, "vocabularies.pkl")
with open(vocab_path, 'wb') as f:
    pickle.dump(vocab_data, f)
print(f"語彙辞書保存完了: {vocab_path}")

# 6. ラベルエンコーダーの保存 (label_encoder.pkl) 
label_encoder_path = os.path.join(save_dir, "label_encoder.pkl")
with open(label_encoder_path, 'wb') as f:
    pickle.dump(train_dataset.protein_to_idx, f)
print(f"ラベルエンコーダー保存完了: {label_encoder_path}")

# 7. README.mdの生成と保存
# f-string用の値を事前計算
final_train_acc = train_accs[-1] if train_accs else 0.0
final_val_acc = val_accs[-1] if val_accs else 0.0

readme_content = f"""# BERT風変異予測モデル実験結果

## 実験概要
- **実験日時**: {current_time}
- **モデルタイプ**: {'事前学習済みBERT' if (hasattr(model, 'use_pretrained') and model.use_pretrained) else 'オリジナルBERT'}
- **対象変異株**: {', '.join(dataset_config['strains'])}
- **予測対象**: プロテイン変異 (36クラス)

## モデル設定
- **アーキテクチャ**: BERT風Transformer
- **語彙サイズ**: {len(mutation_vocab):,} (変異語彙)
- **d_model**: {model_config['d_model']}
- **num_heads**: {model_config['nhead']}
- **num_layers**: {model_config['num_layers']}
- **最大シーケンス長**: {train_dataset.max_length}

## データセット
- **訓練データ**: {len(train_dataset):,} サンプル
- **検証データ**: {len(val_dataset):,} サンプル  
- **テストデータ**: {len(test_x)} タイムステップ
- **クラス数**: {train_dataset.num_classes}

## 訓練結果
- **エポック数**: {model_config['num_epochs']}
- **最良検証精度**: {best_val_acc:.4f}
- **最終訓練精度**: {final_train_acc:.4f}
- **最終検証精度**: {final_val_acc:.4f}
- **総訓練時間**: {total_training_time/60:.1f}分

## テスト評価結果
"""

# テスト結果のサマリーを追加
if timestep_results:
    total_samples = sum(result['samples'] for result in timestep_results.values())
    total_filtered = sum(result['filtered_samples'] for result in timestep_results.values())
    
    if total_filtered > 0:
        weighted_strict = sum(result['strict_sequence_accuracy'] * result['filtered_samples'] 
                             for result in timestep_results.values()) / total_filtered
        weighted_flexible = sum(result['flexible_sequence_accuracy'] * result['filtered_samples'] 
                               for result in timestep_results.values()) / total_filtered
        
        readme_content += f"""
- **評価タイムステップ数**: {len(timestep_results)}
- **総テストサンプル数**: {total_samples:,} → {total_filtered:,} (フィルタ後)
- **全体厳密正解率**: {weighted_strict:.4f}
- **全体柔軟正解率**: {weighted_flexible:.4f}
"""

readme_content += f"""

## ファイル構成
- `best_model.pth`: 最良モデルの重み
- `training_results.json`: 訓練履歴データ
- `training_history.png`: 訓練・検証のLoss/Accuracy推移
- `test_results_timestep.json`: タイムステップ別テスト結果
- `test_accuracy_by_timestep.png`: テスト正解率の可視化
- `config.json`: 実験設定とハイパーパラメータ
- `vocabularies.pkl`: 語彙辞書
- `label_encoder.pkl`: ラベルエンコーダー

## 実験の特徴
- **アプローチ**: 変異を自然言語のトークンとして扱うBERT風モデル
- **語彙構築**: 塩基変異パターン (例: A1234C, G5678T)
- **特別トークン**: <PAD>, <UNK>, <CLS>, <SEP>
- **分類方法**: CLSトークンを用いた分類

## 従来モデルとの違い
- **従来**: 多次元特徴量 (8種類) → 構造化アプローチ
- **BERT風**: 単一変異語彙 → 言語モデルアプローチ
- **メモリ**: より大きな語彙サイズ ({len(mutation_vocab):,} vs 数千)
- **解釈性**: 暗黙的パターン学習 vs 明示的生物学的特徴
"""

readme_path = os.path.join(save_dir, "README.md")
with open(readme_path, 'w', encoding='utf-8') as f:
    f.write(readme_content)
print(f"README保存完了: {readme_path}")

# 8. 実験サマリーの保存 (experiment_summary.json)
summary_data = {
    'experiment_name': current_time,
    'model_type': 'bert_style_transformer',
    'dataset': {
        'strains': dataset_config['strains'],
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset),
        'test_timesteps': len(test_x),
        'num_classes': train_dataset.num_classes,
        'class_names': train_dataset.classes_
    },
    'model_architecture': {
        'type': 'original_bert' if not (hasattr(model, 'use_pretrained') and model.use_pretrained) else 'pretrained_bert',
        'vocab_size': len(mutation_vocab),
        'd_model': model_config['d_model'],
        'nhead': model_config['nhead'],
        'num_layers': model_config['num_layers'],
        'max_seq_length': train_dataset.max_length,
        'total_parameters': sum(p.numel() for p in model.parameters()),
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
    },
    'training_config': {
        'epochs': model_config['num_epochs'],
        'batch_size': model_config['batch_size'],
        'learning_rate': model_config['learning_rate'],
        'weight_decay': model_config['weight_decay']
    },
    'results': {
        'best_val_accuracy': best_val_acc,
        'final_train_accuracy': train_accs[-1] if train_accs else 0,
        'final_val_accuracy': val_accs[-1] if val_accs else 0,
        'training_time_minutes': total_training_time / 60,
        'epochs_completed': len(train_accs)
    }
}

# テスト結果を追加
if timestep_results:
    total_samples = sum(result['samples'] for result in timestep_results.values())
    total_filtered = sum(result['filtered_samples'] for result in timestep_results.values())
    
    if total_filtered > 0:
        weighted_strict = sum(result['strict_sequence_accuracy'] * result['filtered_samples'] 
                             for result in timestep_results.values()) / total_filtered
        weighted_flexible = sum(result['flexible_sequence_accuracy'] * result['filtered_samples'] 
                               for result in timestep_results.values()) / total_filtered
        
        summary_data['test_results'] = {
            'evaluated_timesteps': len(timestep_results),
            'total_test_samples': total_samples,
            'filtered_test_samples': total_filtered,
            'overall_strict_accuracy': weighted_strict,
            'overall_flexible_accuracy': weighted_flexible,
            'per_timestep_results': {
                str(ts): {
                    'samples': result['samples'],
                    'filtered_samples': result['filtered_samples'],
                    'strict_accuracy': result['strict_sequence_accuracy'],
                    'flexible_accuracy': result['flexible_sequence_accuracy']
                }
                for ts, result in timestep_results.items()
            }
        }

summary_path = os.path.join(save_dir, "experiment_summary.json")
with open(summary_path, 'w') as f:
    json.dump(summary_data, f, indent=2, ensure_ascii=False)
print(f"実験サマリー保存完了: {summary_path}")

print(f"\n=== train4と同等の保存が完了しました ===")
print(f"保存先: {os.path.abspath(save_dir)}")

# 最終的なファイル確認
if os.path.exists(save_dir):
    files = os.listdir(save_dir)
    print(f"保存ファイル数: {len(files)}")
    print("保存されたファイル:")
    for file in sorted(files):
        file_path = os.path.join(save_dir, file)
        if os.path.isfile(file_path):
            size = os.path.getsize(file_path)
            print(f"  {file}: {size:,} bytes")
    
    # 保存完了の確認
    expected_files = [
        "best_model.pth", "training_results.json", "training_history.png", 
        "test_results_timestep.json", "test_accuracy_by_timestep.png",
        "config.json", "vocabularies.pkl", "label_encoder.pkl", "README.md", 
        "experiment_summary.json"
    ]
    
    missing_files = [f for f in expected_files if f not in files]
    if missing_files:
        print(f"\n警告: 以下のファイルが見つかりません: {missing_files}")
    else:
        print(f"\n✅ 全ての重要ファイルが正常に保存されました (train4と同等)")
else:
    print(f"❌ 保存ディレクトリが見つかりません: {save_dir}")

print("\n=== 実験完了 ===")
print(f"実験名: {current_time}")
print(f"対象変異株: {dataset_config['strains']}")
print(f"最良検証精度: {best_val_acc:.4f}")
print(f"総エポック数: {model_config['num_epochs']}")
print(f"語彙サイズ: {len(mutation_vocab):,}")
print(f"クラス数: {train_dataset.num_classes}")

# %%
