import torch
import torch.nn as nn
from . import config

class HierarchicalTransformer(nn.Module):
    """
    共起変異と時系列の両方を扱う階層的Transformerモデル。
    1. Timestep-wise Transformer: 各タイムステップ内の共起変異を処理
    2. Sequence Transformer: タイムステップ間の関係を処理
    """
    def __init__(self):
        super().__init__()

        # --- 1. 埋め込み層 ---
        # 各カテゴリカル特徴量のための埋め込み層を定義
        self.embed_base = nn.Embedding(config.VOCAB_SIZE_BASE, config.EMBED_DIM_BASE, padding_idx=0)
        self.embed_pos = nn.Embedding(config.VOCAB_SIZE_POS, config.EMBED_DIM_POS, padding_idx=0)
        self.embed_codon_pos = nn.Embedding(config.VOCAB_SIZE_CODON_POS, config.EMBED_DIM_CODON_POS, padding_idx=0)
        self.embed_aa = nn.Embedding(config.VOCAB_SIZE_AA, config.EMBED_DIM_AA, padding_idx=0)
        self.embed_protein = nn.Embedding(config.VOCAB_SIZE_PROTEIN, config.EMBED_DIM_PROTEIN, padding_idx=0)
        self.embed_protein_pos = nn.Embedding(config.VOCAB_SIZE_PROTEIN_POS, config.EMBED_DIM_PROTEIN_POS, padding_idx=0)

        # 特徴量を結合した後の次元をTransformerの入力次元に合わせる線形層
        # ここでは単純化のため、入力次元はconfig.FEATURE_DIMと仮定
        self.feature_projection = nn.Linear(
            config.CAT_FEATURE_DIM + config.NUM_CHEM_FEATURES,
            config.FEATURE_DIM
        )
        
        # --- 2. 階層的Transformer ---
        # ★ 共起変異用 (Timestep-wise) Transformer
        timestep_encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.FEATURE_DIM,
            nhead=config.N_HEADS_TIMESTEP,
            dim_feedforward=config.FEATURE_DIM * config.HIDDEN_DIM_MULTIPLIER,
            dropout=config.DROPOUT,
            batch_first=True  # --- batch_first=True を追加 ---
        )
        self.timestep_transformer_encoder = nn.TransformerEncoder(
            encoder_layer=timestep_encoder_layer,
            num_layers=config.N_LAYERS_TIMESTEP
        )

        # ★ 時系列用 (Sequence) Transformer
        sequence_encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.FEATURE_DIM,
            nhead=config.N_HEADS_SEQUENCE,
            dim_feedforward=config.FEATURE_DIM * config.HIDDEN_DIM_MULTIPLIER,
            dropout=config.DROPOUT,
            batch_first=True  # --- batch_first=True を追加 ---
        )
        self.sequence_transformer_encoder = nn.TransformerEncoder(
            encoder_layer=sequence_encoder_layer,
            num_layers=config.N_LAYERS_SEQUENCE
        )        # --- 3. 出力ヘッド ---
        # CLSトークンに対応する出力を受け取り、各タスクのクラス数に射影する
        self.protein_head = nn.Linear(config.FEATURE_DIM, config.VOCAB_SIZE_PROTEIN)
        self.position_head = nn.Linear(config.FEATURE_DIM, config.VOCAB_SIZE_POS)

    def forward(self, x_cat, x_num, x_cat_mask, seq_mask): # ★ 引数リストを修正
        # --- デバッグコード ---
        """
        print(f"x_cat shape: {x_cat.shape}")
        print(f"base (bef) max: {x_cat[..., 0].max()}, min: {x_cat[..., 0].min()}")
        print(f"pos max: {x_cat[..., 1].max()}, min: {x_cat[..., 1].min()}")
        print(f"base (aft) max: {x_cat[..., 2].max()}, min: {x_cat[..., 2].min()}")
        print(f"codon_pos max: {x_cat[..., 3].max()}, min: {x_cat[..., 3].min()}")
        print(f"aa (b) max: {x_cat[..., 4].max()}, min: {x_cat[..., 4].min()}")
        print(f"prot_pos max: {x_cat[..., 5].max()}, min: {x_cat[..., 5].min()}")
        print(f"aa (a) max: {x_cat[..., 6].max()}, min: {x_cat[..., 6].min()}")
        print(f"protein max: {x_cat[..., 7].max()}, min: {x_cat[..., 7].min()}")
        """
        # --- デバッグコード終了 ---
        # 0. 入力データの形状を取得
        batch_size, seq_len, co_occur_len, _ = x_cat.shape

        # 1. 特徴量の埋め込みと結合
        # 各カテゴリカル特徴量を埋め込み
        emb_bef = self.embed_base(x_cat[..., 0])
        emb_pos = self.embed_pos(x_cat[..., 1])
        emb_aft = self.embed_base(x_cat[..., 2])
        emb_codon_pos = self.embed_codon_pos(x_cat[..., 3])
        emb_aa_b = self.embed_aa(x_cat[..., 4])
        emb_prot_pos = self.embed_protein_pos(x_cat[..., 5]) # protein_pos_idではなくprotein_id
        emb_aa_a = self.embed_aa(x_cat[..., 6])
        emb_protein = self.embed_protein(x_cat[..., 7])
        
        # 全ての特徴量を結合
        x_combined_raw = torch.cat([
            emb_bef, emb_pos, emb_aft, 
            emb_codon_pos, emb_aa_b, emb_prot_pos, emb_aa_a, emb_protein, x_num
        ], dim=-1)
        
        # 結合した特徴量をFEATURE_DIM (128) に射影
        # (batch_size, seq_len, max_co_occur, feature_dim)
        x_projected = self.feature_projection(x_combined_raw)

        # --- Timestep-wise Transformer ---
        # (batch_size * seq_len, max_co_occur, feature_dim)
        batch_size, seq_len, max_co_occur, _ = x_projected.shape
        x_projected_reshaped = x_projected.view(batch_size * seq_len, max_co_occur, -1)
        # (batch_size * seq_len, max_co_occur)
        x_cat_mask_reshaped = x_cat_mask.view(batch_size * seq_len, max_co_occur)

        # encoded_timesteps: (batch_size * seq_len, max_co_occur, feature_dim)
        encoded_timesteps = self.timestep_transformer_encoder(x_projected_reshaped, src_key_padding_mask=x_cat_mask_reshaped)

        # (batch_size, seq_len, max_co_occur, feature_dim)
        encoded_timesteps = encoded_timesteps.view(batch_size, seq_len, max_co_occur, -1)

        # --- 時点ごとの特徴量集約 ---
        # マスクを適用してパディング部分を0にする
        encoded_timesteps = encoded_timesteps * x_cat_mask.unsqueeze(-1)
        # (batch_size, seq_len, feature_dim)
        sequence_features = encoded_timesteps.sum(dim=2)

        # --- 時系列 Transformer ---
        # encoded_sequence: (batch_size, seq_len, feature_dim)
        encoded_sequence = self.sequence_transformer_encoder(sequence_features, src_key_padding_mask=seq_mask)        # --- 時点ごとの特徴量集約 ---
        # マスクを適用してパディング部分を0にする
        encoded_timesteps = encoded_timesteps * x_cat_mask.unsqueeze(-1)

        # 各タイムステップの表現ベクトルを集約 (平均プーリング)
        # マスクを考慮して平均を計算
        encoded_timesteps[x_cat_mask] = 0 # パディング部分は0
        sum_vectors = encoded_timesteps.sum(dim=2)
        num_vectors = (~x_cat_mask).sum(dim=2).unsqueeze(-1).clamp(min=1)
        sequence_input = sum_vectors / num_vectors

        # 3. 時系列の処理 (Sequence Transformer)
        sequence_output = self.sequence_transformer_encoder(sequence_input, src_key_padding_mask=seq_mask)

        # 4. 最終的な予測
        # シーケンスの最後の有効なタイムステップの出力を取得
        # 各バッチ項目ごとに有効な長さを取得
        active_lengths = (~seq_mask).sum(dim=1)
        last_step_indices = active_lengths - 1
        
        # gatherを使って各バッチから最後の有効な出力を抽出
        last_output = sequence_output[torch.arange(batch_size), last_step_indices]

        protein_logits = self.protein_head(last_output)
        pos_logits = self.position_head(last_output)

        return protein_logits, pos_logits