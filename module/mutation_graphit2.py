# 画像化機能あり　不要
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GraphAttentionLayer(nn.Module):
    """グラフアテンション層 - 最適化版"""
    def __init__(self, in_features, out_features, dropout=0.1, alpha=0.2):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, h, adj):
        batch_size, N = h.size(0), h.size(1)
        
        # 線形変換
        Wh = torch.matmul(h, self.W)
        
        # アテンション重み計算（最適化版）
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = torch.matmul(a_input, self.a).squeeze(-1)
        
        # マスク適用（隣接行列）
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        # アテンション適用
        h_prime = torch.matmul(attention, Wh)
        
        return F.elu(h_prime)

    def _prepare_attentional_mechanism_input(self, Wh):
        """最適化されたアテンション入力準備"""
        batch_size, N, features = Wh.size()
        
        # より効率的な実装
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=1)
        Wh_repeated_alternating = Wh.repeat(1, N, 1)
        
        all_combinations_matrix = torch.cat([
            Wh_repeated_in_chunks, 
            Wh_repeated_alternating
        ], dim=-1)
        
        return all_combinations_matrix.view(batch_size, N, N, 2 * self.out_features)

class GraphiTLayer(nn.Module):
    """GraphiT層（グラフアテンション + Transformer）- 最適化版"""
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        
        # グラフアテンション
        self.graph_attention = GraphAttentionLayer(d_model, d_model, dropout)
        
        # Transformerアテンション
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # フィードフォワード
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # 正規化とドロップアウト
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, src, adj_matrix, src_mask=None, src_key_padding_mask=None):
        # 1. グラフアテンション
        graph_out = self.graph_attention(src, adj_matrix)
        graph_out = self.norm1(src + self.dropout1(graph_out))
        
        # 2. Transformerセルフアテンション
        attn_out, _ = self.self_attn(graph_out, graph_out, graph_out,
                                   attn_mask=src_mask,
                                   key_padding_mask=src_key_padding_mask)
        attn_out = self.norm2(graph_out + self.dropout2(attn_out))
        
        # 3. フィードフォワード
        ff_out = self.linear2(self.dropout(F.relu(self.linear1(attn_out))))
        output = self.norm3(attn_out + self.dropout3(ff_out))
        
        return output


class MutationGraphiT(nn.Module):
    """変異予測用GraphiT v3 - 最適化版"""
    
    @staticmethod
    def _auto_adjust_params(d_model, nhead, total_features):
        """特徴量数に応じてパラメータを自動調整"""
        min_embed_dim = max(16, nhead)
        ideal_embed_dim = d_model // total_features
        
        if ideal_embed_dim < min_embed_dim:
            embed_dim_per_feature = min_embed_dim
            print(f"Warning: embed_dim_per_feature adjusted from {ideal_embed_dim} to {embed_dim_per_feature} "
                  f"(min required: {min_embed_dim})")
        else:
            embed_dim_per_feature = ideal_embed_dim
        
        actual_nhead = min(nhead, embed_dim_per_feature)
        
        if embed_dim_per_feature % actual_nhead != 0:
            for h in range(actual_nhead, 0, -1):
                if embed_dim_per_feature % h == 0:
                    actual_nhead = h
                    break
        
        if actual_nhead != nhead:
            print(f"Info: nhead adjusted from {nhead} to {actual_nhead} "
                  f"(embed_dim_per_feature: {embed_dim_per_feature})")
        
        return embed_dim_per_feature, actual_nhead
    
    def __init__(self, feature_vocabs, d_model=256, nhead=8, num_layers=6, num_classes=36, 
                 max_seq_length=100, feature_mask=None, auto_adjust=True, 
                 similarity_threshold=0.6, graph_config=None):
        super().__init__()
        
        # 基本設定
        self.feature_mask = feature_mask if feature_mask is not None else [True] * (len(feature_vocabs) + 1)
        self.similarity_threshold = similarity_threshold
        
        # グラフ構築設定（新機能）
        self.graph_config = graph_config or {
            'enable_graph_cache': True,
            'max_graph_connections': 10,
            'batch_graph_construction': True,
            'use_sparse_attention': False
        }
        
        self.d_model = d_model
        self.feature_vocabs = feature_vocabs
        self.num_categorical_features = len(feature_vocabs)
        self.total_features = self.num_categorical_features + 1
        self.max_seq_length = max_seq_length
        
        # パラメータ自動調整
        if auto_adjust:
            embed_dim_per_feature, actual_nhead = self._auto_adjust_params(d_model, nhead, self.total_features)
            if d_model % actual_nhead != 0:
                adjusted_d_model = (d_model // actual_nhead) * actual_nhead
                if adjusted_d_model != d_model:
                    print(f"Info: d_model adjusted from {d_model} to {adjusted_d_model}")
                    d_model = adjusted_d_model
        else:
            if d_model % nhead != 0:
                raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")
            embed_dim_per_feature = d_model // self.total_features
            actual_nhead = nhead
        
        self.embed_dim = embed_dim_per_feature
        self.actual_nhead = actual_nhead
        self.actual_d_model = d_model
        
        # 埋め込み層
        self.categorical_embeddings = nn.ModuleList([
            nn.Embedding(len(vocab), embed_dim_per_feature) 
            for vocab in feature_vocabs
        ])
        
        # count特徴量用の線形変換
        self.count_projection = nn.Linear(1, embed_dim_per_feature)
        
        # 特徴量投影層
        total_embed_dim = embed_dim_per_feature * self.total_features
        if total_embed_dim != d_model:
            self.feature_projection = nn.Linear(total_embed_dim, d_model)
        else:
            self.feature_projection = None
        
        # GraphiT層
        self.graphit_layers = nn.ModuleList([
            GraphiTLayer(d_model, actual_nhead, d_model * 2, dropout=0.1)
            for _ in range(num_layers)
        ])
        
        # 位置エンコーディング
        self.pos_encoding = nn.Parameter(torch.randn(max_seq_length, d_model))
        
        # 分類ヘッド
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, num_classes)
        )
        
        # グラフキャッシュ（最適化用）
        if self.graph_config.get('enable_graph_cache', False):
            self.graph_cache = {}
        
        self.init_weights()
    
    def init_weights(self):
        """重みの初期化"""
        # 埋め込み層の初期化
        for embedding in self.categorical_embeddings:
            nn.init.xavier_uniform_(embedding.weight)
        
        # 線形層の初期化
        nn.init.xavier_uniform_(self.count_projection.weight)
        nn.init.zeros_(self.count_projection.bias)
        
        # 特徴量投影層の初期化
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

    def create_mutation_graph_optimized(self, categorical_x, count_x, similarity_threshold=0.6):
        """変異データからグラフ構造を作成（最適化版）"""
        batch_size, num_features, seq_len = categorical_x.shape
        device = categorical_x.device
        
        # バッチ全体で一度に処理
        adj_matrices = torch.eye(seq_len, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
        
        # 隣接時点を接続（ベクトル化）
        if seq_len > 1:
            # 前向き接続
            adj_matrices[:, torch.arange(seq_len-1), torch.arange(1, seq_len)] = 1
            # 後向き接続
            adj_matrices[:, torch.arange(1, seq_len), torch.arange(seq_len-1)] = 1
        
        # 特徴量類似性に基づく接続（効率化）
        max_connections = self.graph_config.get('max_graph_connections', 10)
        
        for i in range(seq_len):
            end_j = min(i + max_connections, seq_len)
            if end_j > i + 1:
                # カテゴリカル特徴量の類似性（ベクトル化）
                cat_features_i = categorical_x[:, :, i:i+1]  # [batch, features, 1]
                cat_features_j = categorical_x[:, :, i+1:end_j]  # [batch, features, range]
                cat_sim = (cat_features_i == cat_features_j).float().mean(dim=1)  # [batch, range]
                
                # count特徴量の類似性（ベクトル化）
                count_i = count_x[:, i:i+1]  # [batch, 1]
                count_j = count_x[:, i+1:end_j]  # [batch, range]
                count_diff = torch.abs(count_i - count_j)
                count_sim = torch.exp(-count_diff / 100.0)
                
                # 総合類似性
                similarity = (cat_sim + count_sim) / 2
                mask = similarity > similarity_threshold
                
                # 隣接行列に反映（バッチ処理）
                for b in range(batch_size):
                    for local_j, global_j in enumerate(range(i+1, end_j)):
                        if mask[b, local_j]:
                            sim_val = similarity[b, local_j].item()
                            adj_matrices[b, i, global_j] = sim_val
                            adj_matrices[b, global_j, i] = sim_val
        
        return adj_matrices

    def forward(self, categorical_x, count_x, padding_mask=None):
        """前向き計算 - v3では特徴量処理を最適化"""
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
            if features_at_t:  # マスクで除外されていない特徴量がある場合
                timestep_feature = torch.cat(features_at_t, dim=-1)
                timestep_features.append(timestep_feature)
            else:
                # 全特徴量がマスクされている場合のフォールバック
                zero_feature = torch.zeros(batch_size, self.actual_d_model, device=categorical_x.device)
                timestep_features.append(zero_feature)
        
        # [batch, seq_len, feature_dim]
        x = torch.stack(timestep_features, dim=1)
        
        # 必要に応じて次元調整
        if self.feature_projection is not None:
            x = self.feature_projection(x)
        
        # 位置エンコーディング追加
        x = x + self.pos_encoding[:seq_len].unsqueeze(0)
        
        # グラフ構造を作成（最適化版使用）
        if self.graph_config.get('batch_graph_construction', True):
            adj_matrix = self.create_mutation_graph_optimized(categorical_x, count_x, self.similarity_threshold)
        else:
            adj_matrix = self.create_mutation_graph_legacy(categorical_x, count_x, self.similarity_threshold)
        
        # GraphiT層を通す
        for layer in self.graphit_layers:
            x = layer(x, adj_matrix, src_key_padding_mask=padding_mask)
        
        # Global Average Pooling
        x = x.mean(dim=1)
        
        return self.classifier(x)

    def create_mutation_graph_legacy(self, categorical_x, count_x, similarity_threshold=0.6):
        """従来のグラフ構築方法（互換性のため）"""
        batch_size, num_features, seq_len = categorical_x.shape
        
        adj_matrices = []
        
        for b in range(batch_size):
            # 基本的な時系列グラフ
            adj = torch.eye(seq_len, device=categorical_x.device)
            
            # 隣接時点を接続
            for i in range(seq_len - 1):
                adj[i, i + 1] = 1
                adj[i + 1, i] = 1
            
            # 特徴量類似性に基づく接続
            for i in range(seq_len):
                for j in range(i + 1, min(i + 5, seq_len)):
                    # カテゴリカル特徴量の類似性
                    cat_sim = (categorical_x[b, :, i] == categorical_x[b, :, j]).float().mean()
                    # count特徴量の類似性
                    count_diff = torch.abs(count_x[b, i] - count_x[b, j])
                    count_sim = torch.exp(-count_diff / 100.0)
                    
                    # 類似性が閾値を超えれば接続
                    similarity = (cat_sim + count_sim) / 2
                    if similarity > similarity_threshold:
                        adj[i, j] = similarity
                        adj[j, i] = similarity
            
            adj_matrices.append(adj)
        
        return torch.stack(adj_matrices)


# 既存の訓練・評価関数（変更なし）
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
