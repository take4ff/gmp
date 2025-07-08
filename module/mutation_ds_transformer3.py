# 複数の高度な結合法
# Dual-Stream Transformer v3 for Mutation Data with Advanced Fusion
import torch
import torch.nn as nn
import torch.nn.functional as F

class AdvancedFusionModule(nn.Module):
    """高度な融合機構"""
    def __init__(self, d_model, nhead=8, fusion_type='cross_attention'):
        super().__init__()
        self.fusion_type = fusion_type
        self.d_model = d_model
        
        if fusion_type == 'cross_attention':
            # Cross-Attention融合
            self.temporal_to_feature = nn.MultiheadAttention(
                embed_dim=d_model, num_heads=nhead, batch_first=True
            )
            self.feature_to_temporal = nn.MultiheadAttention(
                embed_dim=d_model, num_heads=nhead, batch_first=True
            )
            self.fusion_norm1 = nn.LayerNorm(d_model)
            self.fusion_norm2 = nn.LayerNorm(d_model)
            
        elif fusion_type == 'gated':
            # Gated融合
            self.gate_network = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
                nn.Sigmoid()
            )
            
        elif fusion_type == 'multi_modal':
            # Multi-Modal融合
            self.modal_projection1 = nn.Linear(d_model, d_model // 2)
            self.modal_projection2 = nn.Linear(d_model, d_model // 2)
            self.cross_modal = nn.MultiheadAttention(
                embed_dim=d_model // 2, num_heads=max(1, nhead // 2), batch_first=True
            )
            self.fusion_mlp = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model)
            )
        
        # 共通の最終投影
        self.final_projection = nn.Linear(d_model, d_model)
        
    def forward(self, temporal_features, feature_features):
        """
        高度な融合を実行
        Args:
            temporal_features: [batch, d_model] - 時系列ストリームの出力
            feature_features: [batch, d_model] - 特徴量ストリームの出力
        Returns:
            fused_features: [batch, d_model] - 融合された特徴量
        """
        batch_size = temporal_features.size(0)
        
        if self.fusion_type == 'cross_attention':
            # Cross-Attention融合
            # temporal -> feature のアテンション
            temp_unsqueezed = temporal_features.unsqueeze(1)  # [batch, 1, d_model]
            feat_unsqueezed = feature_features.unsqueeze(1)   # [batch, 1, d_model]
            
            temp_attended, _ = self.temporal_to_feature(
                query=temp_unsqueezed,
                key=feat_unsqueezed,
                value=feat_unsqueezed
            )
            temp_enhanced = self.fusion_norm1(temporal_features + temp_attended.squeeze(1))
            
            # feature -> temporal のアテンション
            feat_attended, _ = self.feature_to_temporal(
                query=feat_unsqueezed,
                key=temp_unsqueezed,
                value=temp_unsqueezed
            )
            feat_enhanced = self.fusion_norm2(feature_features + feat_attended.squeeze(1))
            
            # 融合
            fused = (temp_enhanced + feat_enhanced) / 2
            
        elif self.fusion_type == 'gated':
            # Gated融合
            combined_input = torch.cat([temporal_features, feature_features], dim=-1)
            gate = self.gate_network(combined_input)  # [batch, d_model]
            
            fused = gate * temporal_features + (1 - gate) * feature_features
            
        elif self.fusion_type == 'multi_modal':
            # Multi-Modal融合
            temp_proj = self.modal_projection1(temporal_features)  # [batch, d_model//2]
            feat_proj = self.modal_projection2(feature_features)   # [batch, d_model//2]
            
            # Cross-modal attention
            temp_unsqueezed = temp_proj.unsqueeze(1)
            feat_unsqueezed = feat_proj.unsqueeze(1)
            
            cross_attended, _ = self.cross_modal(
                query=temp_unsqueezed,
                key=feat_unsqueezed,
                value=feat_unsqueezed
            )
            
            # 結合とMLP
            combined = torch.cat([temp_proj, cross_attended.squeeze(1)], dim=-1)
            fused = self.fusion_mlp(combined)
            
        else:  # 'concat' (fallback)
            fused = torch.cat([temporal_features, feature_features], dim=-1)
            fused = self.final_projection(fused)
            return fused
        
        # 最終投影
        return self.final_projection(fused)


class MutationDSTransformer(nn.Module):
    """Dual-Stream Transformer v3 with Advanced Fusion"""
    
    @staticmethod
    def _auto_adjust_params(d_model, nhead, total_features):
        """Dual-Stream用のパラメータを自動調整"""
        # 各ストリーム用の埋め込み次元
        embed_dim_per_feature = max(16, d_model // total_features)
        
        # 時系列ストリーム用のhead数調整
        temporal_nhead = min(nhead, d_model)
        while d_model % temporal_nhead != 0 and temporal_nhead > 1:
            temporal_nhead -= 1
        
        # 特徴量ストリーム用のhead数調整
        feature_nhead = min(nhead, embed_dim_per_feature)
        while embed_dim_per_feature % feature_nhead != 0 and feature_nhead > 1:
            feature_nhead -= 1
        
        if temporal_nhead != nhead or feature_nhead != nhead:
            print(f"Info: nhead adjusted - temporal: {temporal_nhead}, feature: {feature_nhead}")
        
        return embed_dim_per_feature, temporal_nhead, feature_nhead
    
    def __init__(self, feature_vocabs, d_model=256, nhead=8, num_layers=6, num_classes=36, 
                 max_seq_length=100, feature_mask=None, auto_adjust=True, 
                 fusion_type='cross_attention'):
        super().__init__()
        
        # 特徴量マスク
        self.feature_mask = feature_mask if feature_mask is not None else [True] * (len(feature_vocabs) + 1)
        
        self.d_model = d_model
        self.feature_vocabs = feature_vocabs
        self.num_categorical_features = len(feature_vocabs)
        self.total_features = self.num_categorical_features + 1
        self.max_seq_length = max_seq_length
        self.fusion_type = fusion_type
        
        # パラメータ自動調整
        if auto_adjust:
            embed_dim_per_feature, temporal_nhead, feature_nhead = self._auto_adjust_params(d_model, nhead, self.total_features)
        else:
            embed_dim_per_feature = d_model // self.total_features
            temporal_nhead = nhead
            feature_nhead = nhead
        
        self.embed_dim = embed_dim_per_feature
        self.actual_nhead = temporal_nhead
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
        
        # 高度な融合機構
        self.advanced_fusion = AdvancedFusionModule(d_model, temporal_nhead, fusion_type)
        
        # 分類ヘッド（融合後は d_model 次元）
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
        
        # 投影層の初期化
        if self.temporal_projection is not None:
            nn.init.xavier_uniform_(self.temporal_projection.weight)
            nn.init.zeros_(self.temporal_projection.bias)
        
        if self.feature_projection is not None:
            nn.init.xavier_uniform_(self.feature_projection.weight)
            nn.init.zeros_(self.feature_projection.bias)
        
        # 位置エンコーディングの初期化
        nn.init.normal_(self.pos_encoding, std=0.02)
        
        # 分類器の初期化
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, categorical_x, count_x, padding_mask=None):
        """v3では特徴量処理を最適化し、高度な融合を適用"""
        batch_size, num_features, seq_len = categorical_x.shape
        
        # 各時点での特徴量ベクトルを作成（v3最適化）
        timestep_features = []
        for t in range(seq_len):
            features_at_t = []
            
            # カテゴリカル特徴量（マスク適用）
            for i in range(self.num_categorical_features):
                if self.feature_mask[i]:
                    embedded = self.categorical_embeddings[i](categorical_x[:, i, t])
                    features_at_t.append(embedded)
            
            # count特徴量（マスク適用）
            if self.feature_mask[-1]:
                count_embedded = self.count_projection(count_x[:, t:t+1])
                features_at_t.append(count_embedded)
            
            # 特徴量を連結
            if features_at_t:
                timestep_feature = torch.cat(features_at_t, dim=-1)
                timestep_features.append(timestep_feature)
            else:
                # フォールバック
                zero_feature = torch.zeros(batch_size, self.embed_dim * len([i for i, m in enumerate(self.feature_mask) if m]), 
                                         device=categorical_x.device)
                timestep_features.append(zero_feature)
        
        # [batch, seq_len, feature_dim]
        x = torch.stack(timestep_features, dim=1)
        
        # Stream 1: 時系列アテンション
        temporal_input = x
        if self.temporal_projection is not None:
            temporal_input = self.temporal_projection(temporal_input)
        
        temporal_input = temporal_input + self.pos_encoding[:seq_len].unsqueeze(0)
        temporal_output = self.temporal_encoder(temporal_input, src_key_padding_mask=padding_mask)
        temporal_pooled = temporal_output.mean(dim=1)  # [batch, d_model]
        
        # Stream 2: 特徴量アテンション
        # 元の埋め込みを再構築
        embedded_features = []
        for i in range(self.num_categorical_features):
            if self.feature_mask[i]:
                embedded = self.categorical_embeddings[i](categorical_x[:, i, :])
                embedded_features.append(embedded)
        
        if self.feature_mask[-1]:
            count_embedded = self.count_projection(count_x.unsqueeze(-1))
            embedded_features.append(count_embedded)
        
        # 各特徴量を個別に処理
        feature_outputs = []
        for embedded in embedded_features:
            feature_out = self.feature_encoder(embedded)
            feature_outputs.append(feature_out.mean(dim=1))  # 時系列次元でプール
        
        if feature_outputs:
            feature_combined = torch.cat(feature_outputs, dim=-1)
            if self.feature_projection is not None:
                feature_pooled = self.feature_projection(feature_combined)
            else:
                feature_pooled = feature_combined
        else:
            # フォールバック
            feature_pooled = torch.zeros(batch_size, self.d_model, device=categorical_x.device)
        
        # 高度な融合
        fused_features = self.advanced_fusion(temporal_pooled, feature_pooled)
        
        return self.classifier(fused_features)


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
        
        optimizer.zero_grad()
        
        outputs = model(categorical_data, count_data)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
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
