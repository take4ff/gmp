# --- model.py ---
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math
from . import config


class PositionalEncoding(nn.Module):
    """Transformer用位置エンコーディング"""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = pe.permute(1, 0, 2)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class InputEmbedding(nn.Module):
    """8つのID特徴量 + 5つの数値特徴量 を受け取り、FEATURE_DIM次元に射影"""
    def __init__(self):
        super().__init__()
        self.pos_embed = nn.Embedding(config.VOCAB_SIZE_POSITION, config.EMBED_DIM_POS, padding_idx=0)
        self.base_embed = nn.Embedding(config.VOCAB_SIZE_BASE, config.EMBED_DIM_BASE, padding_idx=0)
        self.aa_embed = nn.Embedding(config.VOCAB_SIZE_AA, config.EMBED_DIM_AA, padding_idx=0)
        self.region_embed = nn.Embedding(config.NUM_REGIONS, config.EMBED_DIM_REGION, padding_idx=0)
        self.codon_pos_embed = nn.Embedding(config.VOCAB_SIZE_CODON_POS, config.EMBED_DIM_CODON_POS, padding_idx=0)
        self.prot_pos_embed = nn.Embedding(config.VOCAB_SIZE_AA_POS, config.EMBED_DIM_AA_POS, padding_idx=0)
        self.synonymous_embed = nn.Embedding(
            config.VOCAB_SIZE_SYNONYMOUS,
            getattr(config, 'EMBED_DIM_SYNONYMOUS', 8),
            padding_idx=0
        )

        self.num_norm = nn.LayerNorm(config.NUM_CHEM_FEATURES)

        total_embed_dim = (config.EMBED_DIM_POS + (config.EMBED_DIM_BASE * 2) +
                           (config.EMBED_DIM_AA * 2) + config.EMBED_DIM_REGION +
                           config.EMBED_DIM_CODON_POS + config.EMBED_DIM_AA_POS +
                           config.NUM_CHEM_FEATURES +
                           getattr(config, 'EMBED_DIM_SYNONYMOUS', 8))

        self.projection = nn.Linear(total_embed_dim, config.FEATURE_DIM)

    def forward(self, x_cat, x_num):
        # x_cat: [B, T, C, 8]
        base_before = self.base_embed(x_cat[..., 0])
        pos = self.pos_embed(x_cat[..., 1])
        base_after = self.base_embed(x_cat[..., 2])
        codon_pos = self.codon_pos_embed(x_cat[..., 3])
        aa_before = self.aa_embed(x_cat[..., 4])
        aa_pos = self.prot_pos_embed(x_cat[..., 5])
        aa_after = self.aa_embed(x_cat[..., 6])
        region = self.region_embed(x_cat[..., 7])
        synonymous = self.synonymous_embed(x_cat[..., 8])

        num = self.num_norm(x_num)

        combined = torch.cat([
            pos, base_before, base_after,
            aa_before, aa_after,
            region, codon_pos, aa_pos, synonymous,
            num
        ], dim=-1)

        return self.projection(combined)


class CoOccurrenceAttention(nn.Module):
    """共起変異をAttentionで集約"""
    def __init__(self):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, config.FEATURE_DIM))
        self.attention = nn.MultiheadAttention(
            embed_dim=config.FEATURE_DIM,
            num_heads=config.N_HEADS,
            dropout=config.DROPOUT,
            batch_first=True
        )

    def forward(self, x):
        # x: [B, T, C, F]
        B, T, C, F = x.shape
        x_flat = x.reshape(B * T, C, F)
        q_flat = self.query.repeat(B * T, 1, 1)
        attn_output, _ = self.attention(q_flat, x_flat, x_flat)
        output = attn_output.reshape(B, T, F)
        return output


class CausalConv1d(nn.Module):
    """因果的1D畳み込み (局所的な文脈学習)"""
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.padding = (kernel_size - 1)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=0)
        self.act = nn.GELU()  # ReLU -> GeLU に変更

    def forward(self, x):
        # x: [B, T, F] -> [B, F, T]
        x = x.permute(0, 2, 1)
        x = torch.nn.functional.pad(x, (self.padding, 0))  # 左側パディング
        x = self.conv(x)
        x = self.act(x)
        x = x.permute(0, 2, 1)
        return x


class OriginAttention(nn.Module):
    """
    現在の時系列(Query)から、初期状態(Key/Value)を参照するCross-Attention
    これにより、モデルは「現在の変異」が「原点（Wuhan株）」からどれくらい離れているかを常に計算できる
    """
    def __init__(self):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=config.FEATURE_DIM,
            num_heads=getattr(config, 'ORIGIN_ATTENTION_HEADS', 4),
            dropout=config.DROPOUT,
            batch_first=True
        )
        self.norm = nn.LayerNorm(config.FEATURE_DIM)
        self.dropout = nn.Dropout(config.DROPOUT)

    def forward(self, x_seq, x_origin):
        """
        x_seq:    [Batch, Time, Dim] (現在の変異パス)
        x_origin: [Batch, 1, Dim]    (初期盤面/Wuhan株)

        Returns:
            attn_out: [Batch, Time, Dim] (原点との比較情報、残差結合は呼び出し側で行う)
        """
        # x_origin を Key と Value に使う
        attn_out, _ = self.attn(
            query=x_seq,
            key=x_origin,
            value=x_origin
        )
        # 純粋なAttention出力のみを返す（残差結合は呼び出し側で行う）
        return self.norm(self.dropout(attn_out))


class HierarchicalTransformer(nn.Module):
    def __init__(self):
        super().__init__()

        self.input_embed = InputEmbedding()
        self.co_attn = CoOccurrenceAttention()

        # 局所的な文脈情報を抽出するConv1d (Ablation Study用に切り替え可能)
        self.use_local_conv = getattr(config, 'USE_LOCAL_CONV1D', True)
        if self.use_local_conv:
            self.local_feature_extractor = CausalConv1d(
                in_channels=config.FEATURE_DIM,
                out_channels=config.FEATURE_DIM,
                kernel_size=config.LOCAL_CONTEXT_KERNEL_SIZE
            )
        else:
            self.local_feature_extractor = None

        # Origin Attention: 原点（Wuhan株）を常に参照 (Ablation Study用に切り替え可能)
        self.use_origin_attention = getattr(config, 'USE_ORIGIN_ATTENTION', True)
        if self.use_origin_attention:
            self.origin_attn = OriginAttention()
            # 学習可能なOrigin埋め込み（「変異なし」の原点を表す専用ベクトル）
            self.origin_embedding = nn.Parameter(
                torch.randn(1, 1, config.FEATURE_DIM) * 0.02
            )
        else:
            self.origin_attn = None
            self.origin_embedding = None

        self.pos_encoder = PositionalEncoding(config.FEATURE_DIM, config.DROPOUT)

        # Transformer Encoder (Pre-Norm & GeLU 採用)
        encoder_layer = TransformerEncoderLayer(
            d_model=config.FEATURE_DIM,
            nhead=config.N_HEADS,
            dim_feedforward=config.FEATURE_DIM * 4,
            dropout=config.DROPOUT,
            batch_first=True,
            activation="gelu",  # GeLU採用
            norm_first=True     # Pre-Norm採用 (学習安定化)
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=config.N_LAYERS)

        # 予測ヘッド (6タスク)
        # 1. Region予測ヘッド
        self.output_head = nn.Sequential(
            nn.Linear(config.FEATURE_DIM, config.FEATURE_DIM),
            nn.GELU(),
            nn.LayerNorm(config.FEATURE_DIM),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.FEATURE_DIM, config.NUM_REGIONS)
        )

        # 2. 塩基位置予測ヘッド
        self.position_head = nn.Sequential(
            nn.Linear(config.FEATURE_DIM, config.FEATURE_DIM),
            nn.GELU(),
            nn.LayerNorm(config.FEATURE_DIM),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.FEATURE_DIM, config.VOCAB_SIZE_POSITION)
        )

        # 3. アミノ酸配列位置予測ヘッド
        self.aa_pos_head = nn.Sequential(
            nn.Linear(config.FEATURE_DIM, config.FEATURE_DIM),
            nn.GELU(),
            nn.LayerNorm(config.FEATURE_DIM),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.FEATURE_DIM, config.VOCAB_SIZE_AA_POS)
        )

        # 4. 流行度予測ヘッド (回帰)
        self.strength_head = nn.Sequential(
            nn.Linear(config.FEATURE_DIM, config.FEATURE_DIM),
            nn.GELU(),
            nn.LayerNorm(config.FEATURE_DIM),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.FEATURE_DIM, 1)
        )

        # 5. コドン位置予測ヘッド (1, 2, 3 の3クラス + PAD=0 で6クラス)
        self.codon_pos_head = nn.Sequential(
            nn.Linear(config.FEATURE_DIM, config.FEATURE_DIM // 2),
            nn.GELU(),
            nn.LayerNorm(config.FEATURE_DIM // 2),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.FEATURE_DIM // 2, config.VOCAB_SIZE_CODON_POS)
        )

        # 6. シノニマス/ノンシノニマス予測ヘッド (2クラス分類)
        self.synonymous_head = nn.Sequential(
            nn.Linear(config.FEATURE_DIM, config.FEATURE_DIM // 4),
            nn.GELU(),
            nn.LayerNorm(config.FEATURE_DIM // 4),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.FEATURE_DIM // 4, config.VOCAB_SIZE_SYNONYMOUS)
        )

    def forward(self, x_cat, x_num, src_mask=None, src_key_padding_mask=None):
        # 1. 入力埋め込み
        x = self.input_embed(x_cat, x_num)

        # 2. 共起集約 (ベース情報)
        x_agg = self.co_attn(x)

        # 3. 局所特徴抽出 (文脈情報) - Ablation Study用に条件分岐
        if self.use_local_conv and self.local_feature_extractor is not None:
            x_context = self.local_feature_extractor(x_agg)
            # 残差結合 (ベース + 文脈)
            x_combined = x_agg + x_context
        else:
            # Conv1D層をスキップ
            x_combined = x_agg

        # 4. Origin Attention: 原点との比較情報を注入 - Ablation Study用に条件分岐
        if self.use_origin_attention and self.origin_attn is not None:
            batch_size = x_combined.size(0)
            origin_emb = self.origin_embedding.expand(batch_size, -1, -1)  # [B, 1, Dim]
            origin_context = self.origin_attn(x_seq=x_combined, x_origin=origin_emb)
            x_combined = x_combined + origin_context

        # 5. Transformer (大局的文脈)
        x = self.pos_encoder(x_combined)
        x = self.transformer_encoder(
            x,
            mask=src_mask,
            src_key_padding_mask=src_key_padding_mask
        )

        # 6. 予測 (6ヘッド)
        latest_context = x[:, -1, :]
        output_region = self.output_head(latest_context)
        output_position = self.position_head(latest_context)
        output_aa_pos = self.aa_pos_head(latest_context)
        output_strength = self.strength_head(latest_context).squeeze(-1)  # [B, 1] -> [B]
        output_codon_pos = self.codon_pos_head(latest_context)
        output_synonymous = self.synonymous_head(latest_context)

        return output_region, output_position, output_aa_pos, output_strength, output_codon_pos, output_synonymous


class MultiTaskLoss(nn.Module):
    """
    複数のタスクの損失を、不確実性(Uncertainty)に基づいて自動重み付けする層。
    Alex Kendall et al. "Multi-Task Learning Using Uncertainty to Weigh Losses"
    """
    def __init__(self, num_tasks=4):
        super().__init__()
        # log_vars: 損失の重みを制御する学習可能パラメータ (初期値0)
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, *losses):
        """
        任意の数の損失を受け取り、自動重み付けして結合する
        losses: (loss1, loss2, ...) - 各タスクの損失
        """
        loss_sum = 0
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            loss_sum += precision * loss + self.log_vars[i]
        return loss_sum

    def get_weights(self):
        """現在の損失重みを返す (デバッグ用)"""
        with torch.no_grad():
            return [torch.exp(-lv).item() for lv in self.log_vars]


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification (Lin et al., 2017)
    クラス不均衡問題に対応し、難しいサンプルに重点を置いた学習を行う。

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    - gamma > 0: 簡単なサンプル（高確信度）の重みを下げる
    - alpha: クラス重み (オプション)
    """
    def __init__(self, gamma=2.0, alpha=None, reduction='none', label_smoothing=0.0):
        """
        Args:
            gamma: focusing parameter (default: 2.0)
                   gamma=0 で通常のCrossEntropyLossと等価
            alpha: クラス重み (Tensor or None)
            reduction: 'none', 'mean', 'sum'
            label_smoothing: ラベルスムージング係数 (0.0-1.0)
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        """
        Args:
            inputs: [N, C] - 未正規化のロジット (softmax前)
            targets: [N] - 正解クラスのインデックス
        Returns:
            loss: スカラー or [N] (reduction による)
        """
        num_classes = inputs.size(-1)

        # log_softmax for numerical stability
        log_probs = torch.nn.functional.log_softmax(inputs, dim=-1)

        if self.label_smoothing > 0:
            # Smooth labels: (1 - eps) * one_hot + eps / num_classes
            with torch.no_grad():
                smooth_targets = torch.zeros_like(log_probs)
                smooth_targets.fill_(self.label_smoothing / num_classes)
                smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing + self.label_smoothing / num_classes)

            # 標準的なCross Entropy (smoothed)
            ce_loss = -(smooth_targets * log_probs).sum(dim=-1)
            # Focal weightは正解クラスの確率に基づく
            probs = torch.exp(log_probs)
            pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        else:
            # 標準的なCross Entropy
            ce_loss = torch.nn.functional.nll_loss(log_probs, targets, reduction='none')
            # 正解クラスの確率
            probs = torch.exp(log_probs)
            pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        # Focal weight: (1 - pt)^gamma
        focal_weight = (1 - pt) ** self.gamma

        # Alpha weighting (クラスごとの重み)
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha
            else:
                alpha_t = self.alpha.gather(0, targets)
            focal_weight = alpha_t * focal_weight

        # Focal Loss
        loss = focal_weight * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss
