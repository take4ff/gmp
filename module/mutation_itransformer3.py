# 機械学習ライブラリのインポート
import torch
import torch.nn as nn
import math

class MutationITransformer(nn.Module):
    """
    iTransformer for mutation prediction
    特徴量間のアテンションを計算する変異予測用iTransformer
    """
    
    @staticmethod
    def _auto_adjust_params(d_model, nhead, total_features):
        """
        特徴量数に応じてパラメータを自動調整
        
        Args:
            d_model: 指定されたモデル次元
            nhead: 希望するhead数
            total_features: 総特徴量数
            
        Returns:
            tuple: (embed_dim_per_feature, actual_nhead)
        """
        # 最小埋め込み次元（head数の倍数である必要がある）
        min_embed_dim = max(16, nhead)  # 最低16次元、かつnhead以上
        
        # 理想的な特徴量あたりの埋め込み次元
        ideal_embed_dim = d_model // total_features
        
        # 最小次元を下回る場合は調整
        if ideal_embed_dim < min_embed_dim:
            embed_dim_per_feature = min_embed_dim
            print(f"Warning: embed_dim_per_feature adjusted from {ideal_embed_dim} to {embed_dim_per_feature} "
                  f"(min required: {min_embed_dim})")
        else:
            embed_dim_per_feature = ideal_embed_dim
        
        # nheadをembed_dim_per_featureに適合するよう調整
        actual_nhead = min(nhead, embed_dim_per_feature)
        
        # embed_dim_per_featureがnheadで割り切れるように調整
        if embed_dim_per_feature % actual_nhead != 0:
            # 適切なhead数を見つける（降順で検索）
            for h in range(actual_nhead, 0, -1):
                if embed_dim_per_feature % h == 0:
                    actual_nhead = h
                    break
        
        # 調整された値を報告
        if actual_nhead != nhead:
            print(f"Info: nhead adjusted from {nhead} to {actual_nhead} "
                  f"(embed_dim_per_feature: {embed_dim_per_feature})")
        
        return embed_dim_per_feature, actual_nhead
    
    def __init__(self, feature_vocabs, d_model=256, nhead=8, num_layers=6, num_classes=36, max_seq_length=100, feature_mask=None, auto_adjust=True):
        super().__init__()
        
        # 特徴量マスク（どの特徴量を使用するか）
        self.feature_mask = feature_mask if feature_mask is not None else [True] * (len(feature_vocabs) + 1)
        
        self.d_model = d_model  # 指定されたd_modelを保持
        self.feature_vocabs = feature_vocabs
        self.num_categorical_features = len(feature_vocabs)
        self.total_features = self.num_categorical_features + 1  # +1 for count
        self.max_seq_length = max_seq_length
        
        # 特徴量数に応じて適切なパラメータを自動調整
        if auto_adjust:
            embed_dim_per_feature, actual_nhead = self._auto_adjust_params(d_model, nhead, self.total_features)
            # Transformerとの統一性のため、d_modelも調整
            adjusted_d_model = embed_dim_per_feature * self.total_features
            if adjusted_d_model != d_model:
                print(f"Info: d_model adjusted from {d_model} to {adjusted_d_model} "
                      f"(for consistency with embed_dim_per_feature: {embed_dim_per_feature})")
                d_model = adjusted_d_model
        else:
            # d_modelがnheadで割り切れることを確認
            if d_model % nhead != 0:
                raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")
            embed_dim_per_feature = d_model // self.total_features
            actual_nhead = nhead
        
        self.embed_dim = embed_dim_per_feature
        self.actual_nhead = actual_nhead
        self.actual_d_model = d_model  # 調整後のd_model
        
        # カテゴリカル特徴量用の埋め込み層
        self.categorical_embeddings = nn.ModuleList([
            nn.Embedding(len(vocab), embed_dim_per_feature) 
            for vocab in feature_vocabs
        ])
        
        # count特徴量用の線形変換
        self.count_projection = nn.Linear(1, embed_dim_per_feature)
        
        # 特徴量を結合した後の総次元数
        total_embed_dim = embed_dim_per_feature * self.total_features
        
        # 指定されたd_modelに調整する投影層
        if total_embed_dim != d_model:
            self.feature_projection = nn.Linear(total_embed_dim, d_model)
        else:
            self.feature_projection = None
        
        # 実際のd_modelは指定された値を使用
        self.actual_d_model = d_model
        
        # iTransformer用のTransformerEncoder
        # 特徴量間のアテンションを計算するため、embed_dim_per_featureを使用
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim_per_feature,  # 各特徴量の次元
            nhead=actual_nhead,  # 自動調整されたhead数
            dim_feedforward=embed_dim_per_feature * 2,
            dropout=0.1,
            batch_first=True
        )
        self.feature_transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # 実際の値を保存（上で設定済み）
        self.actual_num_layers = num_layers
        
        # 特徴量位置エンコーディング
        self.feature_pos_encoding = nn.Parameter(torch.randn(self.total_features, embed_dim_per_feature))
        
        # 時系列集約
        self.temporal_pooling = nn.AdaptiveAvgPool1d(1)
        
        # 分類ヘッド（指定されたd_modelを使用）
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, num_classes)
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
        
        # 特徴量埋め込み
        embedded_features = []
        for i in range(self.num_categorical_features):
            if self.feature_mask[i]:
                embedded = self.categorical_embeddings[i](categorical_x[:, i, :])
            else:
                embedded = torch.zeros(batch_size, seq_len, self.embed_dim, 
                                    device=categorical_x.device, dtype=torch.float32)
            embedded_features.append(embedded)
        
        if self.feature_mask[-1]:
            count_embedded = self.count_projection(count_x.unsqueeze(-1))
        else:
            count_embedded = torch.zeros(batch_size, seq_len, self.embed_dim, 
                                    device=count_x.device, dtype=torch.float32)
        embedded_features.append(count_embedded)
        
        # iTransformer: 時系列の各時点で特徴量間アテンション
        # [batch, seq_len, total_features, embed_dim]
        x = torch.stack(embedded_features, dim=2)
        
        # 各時点で特徴量間アテンション
        batch_seq, total_feat, embed_dim = x.size(0) * x.size(1), x.size(2), x.size(3)
        x_reshaped = x.view(batch_seq, total_feat, embed_dim)
        
        # 特徴量位置エンコーディング追加
        x_reshaped = x_reshaped + self.feature_pos_encoding
        
        # 特徴量間アテンション
        attended = self.feature_transformer(x_reshaped)
        
        # 元の形状に戻す
        attended = attended.view(batch_size, seq_len, total_feat, embed_dim)
        
        # 特徴量次元を結合  
        attended = attended.view(batch_size, seq_len, -1)
        
        # 時系列集約
        pooled = self.temporal_pooling(attended.transpose(1, 2)).squeeze(-1)
        
        # 投影（必要に応じて）
        if self.feature_projection is not None:
            pooled = self.feature_projection(pooled)
        
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
