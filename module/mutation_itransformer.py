# 下位互換
# 機械学習ライブラリのインポート
import torch
import torch.nn as nn
import math

class MutationITransformer(nn.Module):
    """
    iTransformer for mutation prediction
    特徴量間のアテンションを計算する変異予測用iTransformer
    """
    def __init__(self, feature_vocabs, d_model=256, nhead=8, num_layers=6, num_classes=36, max_seq_length=100, feature_mask=None):
        super().__init__()
        
        # 特徴量マスク（どの特徴量を使用するか）
        self.feature_mask = feature_mask if feature_mask is not None else [True] * (len(feature_vocabs) + 1)
        
        self.d_model = d_model
        self.feature_vocabs = feature_vocabs
        self.num_categorical_features = len(feature_vocabs)
        self.total_features = self.num_categorical_features + 1  # +1 for count
        self.max_seq_length = max_seq_length
        
        # 特徴量ごとの埋め込み（統一次元）
        embed_dim = d_model // self.total_features
        if d_model % self.total_features != 0:
            embed_dim = d_model // self.total_features + 1
        
        self.embed_dim = embed_dim
        
        self.categorical_embeddings = nn.ModuleList([
            nn.Embedding(len(vocab), embed_dim) 
            for vocab in feature_vocabs
        ])
        self.count_projection = nn.Linear(1, embed_dim)
        
        # 実際のd_model（パディング考慮）
        self.actual_d_model = embed_dim * self.total_features
        
        # 次元調整（必要に応じて）
        self.feature_projection = None
        if self.actual_d_model != d_model:
            self.feature_projection = nn.Linear(self.actual_d_model, d_model)
            self.actual_d_model = d_model
        
        # iTransformer: 特徴量間アテンション
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,  # 各特徴量の次元
            nhead=max(1, embed_dim // 64),  # 適切なhead数
            dim_feedforward=embed_dim * 2,
            dropout=0.1,
            batch_first=True
        )
        self.feature_transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # 特徴量位置エンコーディング
        self.feature_pos_encoding = nn.Parameter(torch.randn(self.total_features, embed_dim))
        
        # 時系列集約
        self.temporal_pooling = nn.AdaptiveAvgPool1d(1)
        
        # 分類ヘッド
        self.classifier = nn.Sequential(
            nn.Linear(self.actual_d_model, self.actual_d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.actual_d_model // 2, num_classes)
        )
        
        self.init_weights()
    
    def init_weights(self):
        """重みの初期化"""
        # 埋め込み層の初期化
        for embedding in self.categorical_embeddings:
            nn.init.xavier_uniform_(embedding.weight)
        
        # 線形層の初期化
        nn.init.xavier_uniform_(self.count_projection.weight)
        nn.init.zeros_(self.count_projection.bias)
        
        # 特徴量投影層の初期化（存在する場合）
        if self.feature_projection is not None:
            nn.init.xavier_uniform_(self.feature_projection.weight)
            nn.init.zeros_(self.feature_projection.bias)
        
        # 特徴量位置エンコーディングの初期化
        nn.init.normal_(self.feature_pos_encoding, std=0.02)
        
        # 分類器の初期化
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, categorical_x, count_x, padding_mask=None):
        batch_size, num_features, seq_len = categorical_x.shape
        
        # 特徴量埋め込み（マスク適用）: [batch, seq_len, num_features, embed_dim]
        embedded_features = []
        for i in range(self.num_categorical_features):
            if self.feature_mask[i]:  # この特徴量を使用する場合
                embedded = self.categorical_embeddings[i](categorical_x[:, i, :])  # [batch, seq_len, embed_dim]
            else:  # この特徴量を無効化する場合
                embedded = torch.zeros(batch_size, seq_len, self.embed_dim, 
                                     device=categorical_x.device, dtype=torch.float32)
            embedded_features.append(embedded)
        
        # count特徴量（マスク適用）
        if self.feature_mask[-1]:  # count特徴量を使用する場合
            count_embedded = self.count_projection(count_x.unsqueeze(-1))  # [batch, seq_len, embed_dim]
        else:  # count特徴量を無効化する場合
            count_embedded = torch.zeros(batch_size, seq_len, self.embed_dim, 
                                       device=count_x.device, dtype=torch.float32)
        embedded_features.append(count_embedded)
        
        # 特徴量次元でスタック: [batch, seq_len, total_features, embed_dim]
        x = torch.stack(embedded_features, dim=2)
        
        # iTransformerの核心: 各時点で特徴量間アテンション
        # [batch, seq_len, total_features, embed_dim] -> [batch*seq_len, total_features, embed_dim]
        x_reshaped = x.view(-1, self.total_features, self.embed_dim)
        
        # 特徴量位置エンコーディングを追加
        x_reshaped = x_reshaped + self.feature_pos_encoding
        
        # 特徴量間アテンション
        attended = self.feature_transformer(x_reshaped)  # [batch*seq_len, total_features, embed_dim]
        
        # 元の形状に戻す: [batch, seq_len, total_features, embed_dim]
        attended = attended.view(batch_size, seq_len, self.total_features, self.embed_dim)
        
        # 特徴量次元を結合: [batch, seq_len, total_features * embed_dim]
        attended = attended.view(batch_size, seq_len, -1)
        
        # 次元調整（必要に応じて）
        if self.feature_projection is not None:
            attended = self.feature_projection(attended)
        
        # 時系列集約: [batch, d_model]
        pooled = self.temporal_pooling(attended.transpose(1, 2)).squeeze(-1)
        
        return self.classifier(pooled)


def train_epoch(model, dataloader, criterion, optimizer, device):
    """1エポックの訓練を実行"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, batch in enumerate(dataloader):
        categorical_data = batch['categorical'].to(device)
        count_data = batch['count'].to(device)
        labels = batch['label'].to(device)
        
        # 勾配をゼロに
        optimizer.zero_grad()
        
        # 順伝播
        outputs = model(categorical_data, count_data)
        loss = criterion(outputs, labels)
        
        # 逆伝播
        loss.backward()
        optimizer.step()
        
        # 統計を更新
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        if batch_idx % 50 == 0:
            print(f'Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%')
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    
    return avg_loss, accuracy

def evaluate(model, dataloader, criterion, device):
    """モデルの評価を実行"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in dataloader:
            categorical_data = batch['categorical'].to(device)
            count_data = batch['count'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(categorical_data, count_data)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    
    return avg_loss, accuracy, all_preds, all_targets
