class BiologyAwareTransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, num_features):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_features = num_features
        
        # 標準的なMultiHeadAttention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        
        # 特徴量重要度ゲート
        self.feature_gate = nn.Sequential(
            nn.Linear(d_model, num_features),
            nn.Sigmoid()
        )
        
        # Layer Norm and FFN
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(0.1)
        )
        
    def forward(self, x):
        # Self-attention
        attn_output, attn_weights = self.self_attn(x, x, x)
        x = self.norm1(x + attn_output)
        
        # 特徴量重要度を計算（時系列平均）
        feature_importance = self.feature_gate(x.mean(dim=1, keepdim=True))
        
        # 重要度を適用
        x = x * feature_importance
        
        # FFN
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        
        return x, feature_importance

# MutationTransformerのforwardメソッドを改良
def forward(self, categorical_x, count_x, padding_mask=None):
    batch_size, num_features, seq_len = categorical_x.shape
    
    # 既存の特徴量埋め込み処理...
    embedded_features = []
    for i in range(self.num_categorical_features):
        if self.feature_mask[i]:
            embedded = self.categorical_embeddings[i](categorical_x[:, i, :])
        else:
            embedded = torch.zeros_like(self.categorical_embeddings[i](categorical_x[:, i, :]))
        embedded_features.append(embedded)
    
    if self.feature_mask[-1]:
        count_embedded = self.count_projection(count_x.unsqueeze(-1))
    else:
        count_embedded = torch.zeros_like(self.count_projection(count_x.unsqueeze(-1)))
    embedded_features.append(count_embedded)
    
    x = torch.cat(embedded_features, dim=-1)
    
    if self.feature_projection is not None:
        x = self.feature_projection(x)
    
    x = x + self.pos_encoding[:seq_len].unsqueeze(0)
    
    # カスタムTransformer層を適用
    feature_importances = []
    for layer in self.custom_layers:
        x, importance = layer(x)
        feature_importances.append(importance)
    
    # Global Average Pooling
    x = x.mean(dim=1)
    
    output = self.classifier(x)
    
    # 特徴量重要度も返す（解析用）
    return output, torch.stack(feature_importances, dim=1)