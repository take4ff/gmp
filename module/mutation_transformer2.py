# 下位互換
# 機械学習ライブラリのインポート
import torch
import torch.nn as nn

class MutationTransformer(nn.Module):
    
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
        
        self.d_model = d_model
        self.feature_vocabs = feature_vocabs
        self.num_categorical_features = len(feature_vocabs)
        self.total_features = self.num_categorical_features + 1
        self.max_seq_length = max_seq_length
        
        # 特徴量数に応じて適切なパラメータを自動調整
        if auto_adjust:
            embed_dim_per_feature, actual_nhead = self._auto_adjust_params(d_model, nhead, self.total_features)
            # d_modelもactual_nheadで割り切れるように調整
            if d_model % actual_nhead != 0:
                adjusted_d_model = (d_model // actual_nhead) * actual_nhead
                if adjusted_d_model != d_model:
                    print(f"Info: d_model adjusted from {d_model} to {adjusted_d_model} "
                          f"(to be divisible by nhead: {actual_nhead})")
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
        
        
        # 標準的なTransformerEncoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,  # 指定されたd_modelを使用
            nhead=actual_nhead,  # 自動調整されたhead数
            dim_feedforward=d_model * 2,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # 実際の値を保存（上で設定済み）
        self.actual_num_layers = num_layers
        
        # 位置エンコーディング
        self.pos_encoding = nn.Parameter(torch.randn(max_seq_length, d_model))
        
        # 分類ヘッド
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
        
        # 位置エンコーディングの初期化
        nn.init.normal_(self.pos_encoding, std=0.02)
        
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
            if self.feature_mask[i]:  # この特徴量を使用する場合
                embedded = self.categorical_embeddings[i](categorical_x[:, i, :])
            else:  # この特徴量を無効化する場合
                embedded = torch.zeros_like(self.categorical_embeddings[i](categorical_x[:, i, :]))
            embedded_features.append(embedded)
        
        # count特徴量を変換（マスク適用）
        if self.feature_mask[-1]:  # count特徴量を使用する場合
            count_embedded = self.count_projection(count_x.unsqueeze(-1))
        else:  # count特徴量を無効化する場合
            count_embedded = torch.zeros_like(self.count_projection(count_x.unsqueeze(-1)))
        embedded_features.append(count_embedded)
        
        # 全特徴量を連結
        x = torch.cat(embedded_features, dim=-1)
        
        # d_modelに投影（必要な場合）
        if self.feature_projection is not None:
            x = self.feature_projection(x)
        
        # 位置エンコーディング追加
        x = x + self.pos_encoding[:seq_len].unsqueeze(0)
        
        # Transformer処理
        x = self.transformer(x, src_key_padding_mask=padding_mask)
        x = x.mean(dim=1)  # Global Average Pooling
        
        return self.classifier(x)

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