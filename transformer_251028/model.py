# --- model.py ---
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from . import config
from .utils import PositionalEncoding

class InputEmbedding(nn.Module):
    """
    dataset.py の Mutation_features が返す
    8つのID特徴量 + 5つの数値特徴量 を受け取る
    """
    def __init__(self, vocab_pos, vocab_base, vocab_aa, vocab_region, 
                 vocab_codon_pos, vocab_prot_pos,
                 num_chem, 
                 embed_pos, embed_base, embed_aa, embed_region,
                 embed_codon_pos, embed_prot_pos,
                 feature_dim):
        super().__init__()
        # 1. カテゴリカルEmbedding層 (8特徴量に対応)
        self.pos_embed = nn.Embedding(vocab_pos, embed_pos, padding_idx=0)         # base_pos (idx:1)
        self.base_embed = nn.Embedding(vocab_base, embed_base, padding_idx=0)      # bef/aft_token (idx:0, 2)
        self.aa_embed = nn.Embedding(vocab_aa, embed_aa, padding_idx=0)            # aa_bef/aft_token (idx:4, 6)
        self.region_embed = nn.Embedding(vocab_region, embed_region, padding_idx=0)# protein_token (idx:7)
        self.codon_pos_embed = nn.Embedding(vocab_codon_pos, embed_codon_pos, padding_idx=0) # codon_pos (idx:3)
        self.prot_pos_embed = nn.Embedding(vocab_prot_pos, embed_prot_pos, padding_idx=0)  # protein_pos (idx:5)
        
        # 2. 数値 (化学的性質) のための正規化層
        self.num_norm = nn.LayerNorm(num_chem)
        
        # 3. 結合後の次元を最終的な特徴量次元(FEATURE_DIM)に合わせる線形層
        # (Pos + Base*2 + AA*2 + Region + CodonPos + ProtPos + Chem)
        total_embed_dim = (embed_pos + (embed_base * 2) + (embed_aa * 2) + 
                           embed_region + embed_codon_pos + embed_prot_pos + 
                           num_chem)
        
        self.projection = nn.Linear(total_embed_dim, feature_dim)

    def forward(self, x_cat, x_num):
        # x_cat: [B, T, C, 8] (dataset.pyのMutation_featuresの出力順)
        
        # 8つのIDをそれぞれの辞書(Embedding)でベクトル化
        # [bef_token, base_pos, aft_token, codon_pos, aa_bef_token, protein_pos, aa_aft_token, protein_token]
        base_before = self.base_embed(x_cat[..., 0])
        pos = self.pos_embed(x_cat[..., 1])
        base_after = self.base_embed(x_cat[..., 2])
        codon_pos = self.codon_pos_embed(x_cat[..., 3])
        aa_before = self.aa_embed(x_cat[..., 4])
        protein_pos = self.prot_pos_embed(x_cat[..., 5])
        aa_after = self.aa_embed(x_cat[..., 6])
        region = self.region_embed(x_cat[..., 7])
        
        # 数値データを正規化 (x_num: [B, T, C, 5])
        num = self.num_norm(x_num)
        
        # すべてのベクトルを結合(Concatenate)
        combined = torch.cat([
            pos, base_before, base_after, 
            aa_before, aa_after, 
            region, 
            codon_pos, protein_pos,
            num
        ], dim=-1)
        
        # 最終的な特徴量ベクトルに射影
        return self.projection(combined)


class CoOccurrenceAttention(nn.Module):
    # (変更なし)
    def __init__(self, feature_dim):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, feature_dim))
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim, 
            num_heads=config.N_HEADS, 
            dropout=config.DROPOUT,
            batch_first=True
        )
    def forward(self, x):
        B, T, C, F = x.shape
        x_flat = x.reshape(B * T, C, F)
        q_flat = self.query.repeat(B * T, 1, 1)
        attn_output, _ = self.attention(q_flat, x_flat, x_flat)
        output = attn_output.reshape(B, T, F)
        return output


class HierarchicalTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 1. 入力層 (configから新しい設定を渡す)
        self.input_embed = InputEmbedding(
            config.VOCAB_SIZE_POSITION, config.VOCAB_SIZE_BASE, 
            config.VOCAB_SIZE_AA,
            config.NUM_REGIONS,         # vocab_region
            config.VOCAB_SIZE_CODON_POS,
            config.VOCAB_SIZE_PROTEIN_POS,
            config.NUM_CHEM_FEATURES, 
            config.EMBED_DIM_POS, config.EMBED_DIM_BASE,
            config.EMBED_DIM_AA,
            config.EMBED_DIM_REGION,
            config.EMBED_DIM_CODON_POS,
            config.EMBED_DIM_PROTEIN_POS,
            config.FEATURE_DIM
        )
        
        # 2. 共起集約層 (変更なし)
        self.co_attn = CoOccurrenceAttention(config.FEATURE_DIM)
        
        # 3. 時系列エンコーダ (変更なし)
        self.pos_encoder = PositionalEncoding(config.FEATURE_DIM, config.DROPOUT)
        encoder_layers = TransformerEncoderLayer(
            d_model=config.FEATURE_DIM, 
            nhead=config.N_HEADS, 
            dim_feedforward=config.HIDDEN_DIM * 2,
            dropout=config.DROPOUT,
            batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(
            encoder_layers, num_layers=config.N_LAYERS
        )
        
        # 4. 予測ヘッド (変更なし)
        self.output_head = nn.Linear(config.FEATURE_DIM, config.NUM_REGIONS)
        self.position_head = nn.Linear(config.FEATURE_DIM, config.VOCAB_SIZE_POSITION)

    def forward(self, x_cat, x_num, src_mask=None, src_key_padding_mask=None):
        # 1-3 (変更なし)
        x = self.input_embed(x_cat, x_num)
        x = self.co_attn(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(
            x, 
            mask=src_mask, 
            src_key_padding_mask=src_key_padding_mask
        )
        
        # 4. 予測ヘッド (変更なし)
        latest_context = x[:, -1, :] 
        output_region = self.output_head(latest_context)
        output_position = self.position_head(latest_context)
        
        return output_region, output_position