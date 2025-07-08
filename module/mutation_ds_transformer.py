# 下位互換
# Dual-Stream Transformer for Mutation Data
# 特徴量と時系列の重みづけは不可能
import torch
import torch.nn as nn

class MutationDSTransformer(nn.Module):
    
    @staticmethod
    def _auto_adjust_params(d_model, nhead, total_features):
        """
        Dual-Stream用のパラメータを自動調整
        """
        # 各ストリーム用の埋め込み次元
        embed_dim_per_feature = max(16, d_model // total_features)
        
        # 時系列ストリーム用のhead数調整
        temporal_nhead = min(nhead, d_model)
        while d_model % temporal_nhead != 0:
            temporal_nhead -= 1
        
        # 特徴量ストリーム用のhead数調整
        feature_nhead = min(nhead, embed_dim_per_feature)
        while embed_dim_per_feature % feature_nhead != 0:
            feature_nhead -= 1
        
        if temporal_nhead != nhead or feature_nhead != nhead:
            print(f"Info: nhead adjusted - temporal: {temporal_nhead}, feature: {feature_nhead}")
        
        return embed_dim_per_feature, temporal_nhead, feature_nhead
    
    def __init__(self, feature_vocabs, d_model=256, nhead=8, num_layers=6, num_classes=36, max_seq_length=100, feature_mask=None, auto_adjust=True):
        super().__init__()
        
        # 特徴量マスク（どの特徴量を使用するか）
        self.feature_mask = feature_mask if feature_mask is not None else [True] * (len(feature_vocabs) + 1)
        
        self.d_model = d_model
        self.feature_vocabs = feature_vocabs
        self.num_categorical_features = len(feature_vocabs)
        self.total_features = self.num_categorical_features + 1
        self.max_seq_length = max_seq_length
        
        # パラメータ自動調整
        if auto_adjust:
            embed_dim_per_feature, temporal_nhead, feature_nhead = self._auto_adjust_params(d_model, nhead, self.total_features)
        else:
            embed_dim_per_feature = d_model // self.total_features
            temporal_nhead = nhead
            feature_nhead = nhead
        
        self.embed_dim = embed_dim_per_feature
        self.actual_nhead = temporal_nhead  # メイン（時系列）のhead数
        self.actual_d_model = d_model
        self.actual_num_layers = num_layers
        
        # カテゴリカル特徴量用の埋め込み層
        self.categorical_embeddings = nn.ModuleList([
            nn.Embedding(len(vocab), embed_dim_per_feature) 
            for vocab in feature_vocabs
        ])
        
        # count特徴量用の線形変換
        self.count_projection = nn.Linear(1, embed_dim_per_feature)
        
        # Stream 1: 時系列アテンション用
        total_embed_dim = embed_dim_per_feature * self.total_features
        if total_embed_dim != d_model:
            self.temporal_projection = nn.Linear(total_embed_dim, d_model)
        else:
            self.temporal_projection = None
        
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=temporal_nhead,
            dim_feedforward=d_model * 2,
            dropout=0.1,
            batch_first=True
        )
        self.temporal_encoder = nn.TransformerEncoder(temporal_layer, num_layers)
        
        # Stream 2: 特徴量アテンション用
        feature_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim_per_feature,
            nhead=feature_nhead,
            dim_feedforward=embed_dim_per_feature * 2,
            dropout=0.1,
            batch_first=True
        )
        self.feature_encoder = nn.TransformerEncoder(feature_layer, num_layers)
        
        # 特徴量ストリームの投影
        feature_final_dim = embed_dim_per_feature * self.total_features
        if feature_final_dim != d_model:
            self.feature_projection = nn.Linear(feature_final_dim, d_model)
        else:
            self.feature_projection = None
        
        # 位置エンコーディング
        self.pos_encoding = nn.Parameter(torch.randn(max_seq_length, d_model))
        #self.feature_pos_encoding = nn.Parameter(torch.randn(self.total_features, embed_dim_per_feature))
        
        # 融合機構
        self.fusion_layer = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=temporal_nhead,
            batch_first=True
        )
        
        # 分類ヘッド（2つのストリームを結合）
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, num_classes)
        )
        
        self.init_weights()
    
    def init_weights(self):
        """重みの初期化（既存と同じパターン）"""
        # 埋め込み層の初期化
        for embedding in self.categorical_embeddings:
            nn.init.xavier_uniform_(embedding.weight)
        
        # 線形層の初期化
        nn.init.xavier_uniform_(self.count_projection.weight)
        nn.init.zeros_(self.count_projection.bias)
        
        # 投影層の初期化
        if self.temporal_projection is not None:
            nn.init.xavier_uniform_(self.temporal_projection.weight)
            nn.init.zeros_(self.temporal_projection.bias)
        
        if self.feature_projection is not None:
            nn.init.xavier_uniform_(self.feature_projection.weight)
            nn.init.zeros_(self.feature_projection.bias)
        
        # 位置エンコーディングの初期化
        nn.init.normal_(self.pos_encoding, std=0.02)
        #nn.init.normal_(self.feature_pos_encoding, std=0.02)
        
        # 分類器の初期化
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, categorical_x, count_x, padding_mask=None):
        batch_size, num_features, seq_len = categorical_x.shape
        
        # カテゴリカル特徴量を埋め込み（マスク適用）
        embedded_features = []
        for i in range(self.num_categorical_features):
            embedded = self.categorical_embeddings[i](categorical_x[:, i, :])
            if not self.feature_mask[i]:
                embedded = embedded * 0  # より効率的
            embedded_features.append(embedded)
        
        count_embedded = self.count_projection(count_x.unsqueeze(-1))
        if not self.feature_mask[-1]:
            count_embedded = count_embedded * 0
        embedded_features.append(count_embedded)
        
        # Stream 1: 時系列アテンション
        temporal_input = torch.cat(embedded_features, dim=-1)
        if self.temporal_projection is not None:
            temporal_input = self.temporal_projection(temporal_input)
        
        # 位置エンコーディング追加
        temporal_input = temporal_input + self.pos_encoding[:seq_len].unsqueeze(0)
        temporal_output = self.temporal_encoder(temporal_input, src_key_padding_mask=padding_mask)
        temporal_pooled = temporal_output.mean(dim=1)
        
        # Stream 2: 特徴量アテンション - 修正が必要
        feature_input = torch.stack(embedded_features, dim=2)  # [batch, seq_len, features, embed_dim]
        batch_size, seq_len, num_features, embed_dim = feature_input.shape
        feature_input = feature_input.permute(0, 2, 1, 3).contiguous()  # [batch, features, seq_len, embed_dim]
        feature_input = feature_input.view(batch_size * num_features, seq_len, embed_dim)
        
        # 特徴量位置エンコーディングは削除（系列方向の処理なので）
        feature_output = self.feature_encoder(feature_input)
        
        # 形状復元
        feature_output = feature_output.view(batch_size, num_features, seq_len, embed_dim)
        feature_output = feature_output.permute(0, 2, 1, 3)  # [batch, seq_len, features, embed_dim]
        feature_output = feature_output.contiguous().view(batch_size, seq_len, -1)
        feature_pooled = feature_output.mean(dim=1)
        if self.feature_projection is not None:
            feature_pooled = self.feature_projection(feature_pooled)
        
        # 融合: Cross-attention between streams
        # 修正版：双方向のcross-attentionまたはより適切な融合
        # Option 1: 双方向fusion
        temporal_to_feature, _ = self.fusion_layer(
            temporal_pooled.unsqueeze(1),
            feature_pooled.unsqueeze(1),
            feature_pooled.unsqueeze(1)
        )
        feature_to_temporal, _ = self.fusion_layer(
            feature_pooled.unsqueeze(1),
            temporal_pooled.unsqueeze(1),
            temporal_pooled.unsqueeze(1)
        )
        fused_output = (temporal_to_feature + feature_to_temporal) / 2
        
        # 最終結合　重みづけは1:1
        combined = torch.cat([temporal_pooled, fused_output.squeeze(1)], dim=-1)
        return self.classifier(combined)
    
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
