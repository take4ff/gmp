# 機械学習ライブラリのインポート
import torch
import torch.nn as nn

class MutationTransformer(nn.Module):
    def __init__(self, feature_vocabs, d_model=256, nhead=8, num_layers=6, num_classes=36, max_seq_length=100, feature_mask=None, auto_adjust=True):
        super(MutationTransformer, self).__init__()

        # 特徴量マスク（どの特徴量を使用するか）
        self.feature_mask = feature_mask if feature_mask is not None else [True] * (len(feature_vocabs) + 1)
        
        self.d_model = d_model
        self.feature_vocabs = feature_vocabs
        self.num_categorical_features = len(feature_vocabs)  # 8つのカテゴリカル特徴量
        
        # パラメータ自動調整
        if auto_adjust:
            # 特徴量数に応じて適切なパラメータを自動調整
            total_features = self.num_categorical_features + 1  # +1 for count
            embed_dim_per_feature = max(16, min(64, d_model // total_features))
            
            # nheadの調整
            total_embed_dim = embed_dim_per_feature * total_features
            if total_embed_dim % nhead != 0:
                self.actual_nhead = max(1, total_embed_dim // (total_embed_dim // nhead))
            else:
                self.actual_nhead = nhead
            
            self.actual_num_layers = num_layers
        else:
            embed_dim_per_feature = 32
            self.actual_nhead = nhead
            self.actual_num_layers = num_layers
        
        # カテゴリカル特徴量用の埋め込み層
        self.categorical_embeddings = nn.ModuleList([
            nn.Embedding(len(vocab), embed_dim_per_feature) 
            for vocab in feature_vocabs
        ])
        
        # count特徴量用の線形変換
        self.count_projection = nn.Linear(1, embed_dim_per_feature)
        
        # 全特徴量の埋め込み次元
        total_embed_dim = embed_dim_per_feature * (self.num_categorical_features + 1)  # +1 for count
        
        # 次元調整
        if total_embed_dim % self.actual_nhead != 0:
            self.actual_d_model = ((total_embed_dim // self.actual_nhead) + 1) * self.actual_nhead
            self.feature_projection = nn.Linear(total_embed_dim, self.actual_d_model)
        else:
            self.actual_d_model = total_embed_dim
            self.feature_projection = None
        
        # 位置エンコーディング
        self.pos_encoding = nn.Parameter(torch.randn(max_seq_length, self.actual_d_model))
        
        # Transformer Encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=self.actual_d_model, 
            nhead=self.actual_nhead, 
            dim_feedforward=self.actual_d_model * 2,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layers, self.actual_num_layers)
        
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
        
        # 位置エンコーディングの初期化
        nn.init.normal_(self.pos_encoding, std=0.02)
        
        # 分類器の初期化
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, categorical_x, count_x, padding_mask=None):
        batch_size, num_features, seq_len = categorical_x.shape
        
        # 各時点での特徴量ベクトルを作成
        timestep_features = []
        for t in range(seq_len):
            features_at_t = []
            
            # カテゴリカル特徴量
            for i in range(self.num_categorical_features):
                if self.feature_mask[i]:
                    embedded = self.categorical_embeddings[i](categorical_x[:, i, t])
                    features_at_t.append(embedded)
            
            # count特徴量
            if self.feature_mask[-1]:
                count_embedded = self.count_projection(count_x[:, t:t+1])
                features_at_t.append(count_embedded)
            
            # 特徴量を連結
            timestep_feature = torch.cat(features_at_t, dim=-1)
            timestep_features.append(timestep_feature)
        
        # [batch, seq_len, feature_dim]
        x = torch.stack(timestep_features, dim=1)
        
        # 必要に応じて次元調整
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