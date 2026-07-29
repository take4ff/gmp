# --- model.py ---
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
import math
from . import config
from .utils import logging as _log


def get_activation():
    act_str = getattr(config, 'ACTIVATION', 'gelu').lower()
    if act_str == 'gelu':
        return nn.GELU()
    elif act_str == 'relu':
        return nn.ReLU()
    elif act_str == 'silu':
        return nn.SiLU()
    elif act_str == 'elu':
        return nn.ELU()
    elif act_str == 'selu':
        return nn.SELU()
    else:
        raise ValueError(f"Unknown activation function: {act_str}")


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


class ALibiPositionalBias(nn.Module):
    """ALiBi (Attention with Linear Biases) - Press et al., 2021

    学習パラメータなしで相対位置バイアスを Self-Attention に注入する。
    各ヘッド h に小さな slope m_h を割り当て、遠い覆とのアテンションを掳り込む。

        bias[h, i, j] = -m_h * |i - j|

    正弦波 PE をスキップし、このバイアスを TransformerEncoder の attn_mask に渡す。
    """
    def __init__(self, num_heads: int, max_len: int = 5000):
        super().__init__()
        slopes = self._get_slopes(num_heads)            # [n_heads]
        positions = torch.arange(max_len, dtype=torch.float32)
        # |i - j| の行列: [max_len, max_len]
        distances = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs()
        # biases[h, i, j] = -slopes[h] * |i - j|: [n_heads, max_len, max_len]
        biases = -slopes.view(-1, 1, 1) * distances.unsqueeze(0)
        self.register_buffer('biases', biases)          # [n_heads, max_len, max_len]

    @staticmethod
    def _get_slopes(n: int) -> torch.Tensor:
        """ALiBi の slope m_h を計算する。Press et al. の実装に準拠。"""
        def _slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            return [start * (start ** i) for i in range(n)]

        if math.log2(n) == int(math.log2(n)):
            return torch.tensor(_slopes_power_of_2(n))
        # n が 2 の累乗でない場合は補間
        p = 2 ** math.floor(math.log2(n))
        base = _slopes_power_of_2(p)
        extra = _slopes_power_of_2(2 * p)[0::2][:n - p]
        return torch.tensor(base + extra)

    def forward(self, seq_len: int, batch_size: int, n_heads: int, device) -> torch.Tensor:
        """attn_mask として渡せるバイアスを返す。

        Returns:
            Tensor [B * n_heads, T, T] — TransformerEncoder の mask 引数に渡せる形式
        """
        bias = self.biases[:n_heads, :seq_len, :seq_len].to(device)  # [n_heads, T, T]
        # [B, n_heads, T, T] → [B*n_heads, T, T]
        return bias.unsqueeze(0).expand(batch_size, -1, -1, -1).reshape(batch_size * n_heads, seq_len, seq_len)


class InputEmbedding(nn.Module):
    """(9 + 2*CONTEXT_WINDOW) のカテゴリ特徴量 + 32の数値特徴量を受け取り、FEATURE_DIM次元に射影

    カテゴリ特徴量 (9 + 2*CONTEXT_WINDOW): base_before[0], position[1], base_after[2], codon_pos[3],
                         aa_before[4], aa_pos[5], aa_after[6], region[7], synonymous[8],
                         left{W}[9]..left1[9+W-1], right1[9+W]..right{W}[9+2W-1]  (W=CONTEXT_WINDOW)
    数値特徴量    (32): freq[0], hydro[1], charge[2], size[3], blsm[4], pam250[5],
                         host_distance_log_ratio_diff[6], host_distance_log_ratio_before[7],
                         human_RSCU_diff[8], SCV2_RSCU_diff[9],
                         optimal_to_optimal[10], non_optimal_to_optimal[11], optimal_to_non_optimal[12],
                         is_transition[13], transition_human_RSCU_diff[14], transition_SCV2_RSCU_diff[15],
                         CpG_diff[16], UpA_diff[17],
                         human_RSCU_before[18], SCV2_RSCU_before[19],
                         human_freq_before[20], human_freq_diff[21], SCV2_freq_before[22], SCV2_freq_diff[23],
                         human_CAI_before[24], human_CAI_diff[25], SCV2_CAI_before[26], SCV2_CAI_diff[27],
                         host_distance_RSCU_ratio_before[28], host_distance_RSCU_ratio_diff[29],
                         cum_syn[30], cum_nonsyn[31]
    """
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
            padding_idx=0,
        )

        self.num_norm = nn.LayerNorm(config.NUM_CHEM_FEATURES)

        _ctx_w = getattr(config, 'CONTEXT_WINDOW', 3)
        total_embed_dim = (config.EMBED_DIM_POS + (config.EMBED_DIM_BASE * (2 + 2 * _ctx_w)) +
                           (config.EMBED_DIM_AA * 2) + config.EMBED_DIM_REGION +
                           config.EMBED_DIM_CODON_POS + config.EMBED_DIM_AA_POS +
                           config.NUM_CHEM_FEATURES +
                           getattr(config, 'EMBED_DIM_SYNONYMOUS', 8))

        self.projection = nn.Linear(total_embed_dim, config.FEATURE_DIM)

    def forward(self, x_cat, x_num):
        # x_cat: [B, T, C, 15]
        base_before = self.base_embed(x_cat[..., 0])
        pos = self.pos_embed(x_cat[..., 1])
        base_after = self.base_embed(x_cat[..., 2])
        codon_pos = self.codon_pos_embed(x_cat[..., 3])
        aa_before = self.aa_embed(x_cat[..., 4])
        aa_pos = self.prot_pos_embed(x_cat[..., 5])
        aa_after = self.aa_embed(x_cat[..., 6])
        region = self.region_embed(x_cat[..., 7])
        synonymous = self.synonymous_embed(x_cat[..., 8])
        ctx_w = getattr(config, 'CONTEXT_WINDOW', 3)
        context_embeds = [self.base_embed(x_cat[..., 9 + i]) for i in range(2 * ctx_w)]

        num = self.num_norm(x_num)

        combined = torch.cat([
            pos, base_before, base_after,
            aa_before, aa_after,
            region, codon_pos, aa_pos, synonymous,
            *context_embeds,
            num
        ], dim=-1)

        return self.projection(combined)


class CoOccurrenceAttention(nn.Module):
    """共起変異を Attention で集約。

    CO_ATTN_N_LAYERS > 1 の場合:
        最初の N-1 層で Self-Attention（変異間の多段相互作用）を行い、
        最終層で学習クエリによる Cross-Attention でスカラーへ集約する。
    CO_ATTN_DIM != FEATURE_DIM の場合:
        in_proj で広い次元に投影して集約し、out_proj で FEATURE_DIM に戻す。
    """
    def __init__(self):
        super().__init__()
        n_heads  = getattr(config, 'CO_ATTN_N_HEADS', config.N_HEADS)
        n_layers = getattr(config, 'CO_ATTN_N_LAYERS', 1)
        attn_dim = getattr(config, 'CO_ATTN_DIM', config.FEATURE_DIM)

        self.attn_dim = attn_dim
        self.n_layers = n_layers

        # 次元変換: FEATURE_DIM → attn_dim → FEATURE_DIM
        if attn_dim != config.FEATURE_DIM:
            self.in_proj  = nn.Linear(config.FEATURE_DIM, attn_dim)
            self.out_proj = nn.Linear(attn_dim, config.FEATURE_DIM)
        else:
            self.in_proj  = None
            self.out_proj = None

        # 集約用学習クエリ
        self.query = nn.Parameter(torch.randn(1, 1, attn_dim))

        # Self-Attention 層（最初の N-1 層: 変異間相互作用）
        n_self = max(0, n_layers - 1)
        self.self_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(attn_dim, n_heads, dropout=config.DROPOUT, batch_first=True)
            for _ in range(n_self)
        ])
        self.self_attn_norms = nn.ModuleList([
            nn.LayerNorm(attn_dim) for _ in range(n_self)
        ])

        # 最終 Cross-Attention 層（クエリ → 変異集合 → スカラー集約）
        self.attention = nn.MultiheadAttention(
            embed_dim=attn_dim,
            num_heads=n_heads,
            dropout=config.DROPOUT,
            batch_first=True,
        )

    def forward(self, x, co_occur_mask=None, need_weights=False, freq_penalty=None):
        # x: [B, T, C, F]
        # co_occur_mask: [B, T, C] bool, True=PAD（base_before==0 の位置）
        # freq_penalty: [B, T, C] float, 提案(USE_COATTN_FREQUENCY_PENALTY)有効時のみ非None。
        #   値が大きいほど「系統非依存の定番変異」であることを示し、Cross-Attentionスコアから
        #   減点する（頻度に依らず勝ってしまう創始変異への機械的な固着を抑制する対策）。
        B, T, C, F = x.shape
        x_flat = x.reshape(B * T, C, F)  # [B*T, C, F]

        # key_padding_mask: [B*T, C]（MHA の key/value 側 PAD を無視させる）
        kpm = co_occur_mask.reshape(B * T, C) if co_occur_mask is not None else None

        # 次元変換 (FEATURE_DIM → attn_dim)
        if self.in_proj is not None:
            x_flat = self.in_proj(x_flat)

        # Self-Attention 層（変異間相互作用）
        for sa, norm in zip(self.self_attn_layers, self.self_attn_norms):
            sa_out, _ = sa(x_flat, x_flat, x_flat, key_padding_mask=kpm, need_weights=False)
            x_flat = norm(x_flat + sa_out)

        # Cross-Attention（クエリ → 変異集合 → スカラーへ集約）
        q_flat = self.query.repeat(B * T, 1, 1)  # [B*T, 1, attn_dim]

        if freq_penalty is None:
            # 既存コードパスを完全維持（頻度ペナルティ無効時は挙動不変）
            attn_out, attn_weights = self.attention(
                q_flat, x_flat, x_flat,
                key_padding_mask=kpm,
                need_weights=need_weights,
                average_attn_weights=True,
            )  # attn_out: [B*T, 1, attn_dim]
        else:
            # bool key_padding_mask と float attn_mask の混在はPyTorchのdeprecation対象
            # （実機確認済み）のため、PADマスクも同じfloat加算形式に統合してから渡す。
            bias = -freq_penalty.reshape(B * T, 1, C)  # 頻度が高いほど減点
            if kpm is not None:
                bias = bias.masked_fill(kpm.unsqueeze(1), float('-inf'))
            attn_mask = bias.repeat_interleave(self.attention.num_heads, dim=0)  # [B*T*heads, 1, C]
            attn_out, attn_weights = self.attention(
                q_flat, x_flat, x_flat,
                attn_mask=attn_mask,
                need_weights=need_weights,
                average_attn_weights=True,
            )  # attn_out: [B*T, 1, attn_dim]

        output = attn_out.reshape(B, T, self.attn_dim)  # [B, T, attn_dim]

        # attn_dim → FEATURE_DIM に戻す
        if self.out_proj is not None:
            output = self.out_proj(output)

        # 全PADタイムステップ（kpm 行が全 True）の集約は不定値になりうるため 0 で埋める。
        # cross-attention 出力を直接ガードすることで CO_ATTN_N_LAYERS の値に依らず全設定を保護する。
        output = torch.nan_to_num(output, nan=0.0)

        # need_weights=True のとき: attn_weights shape は [B*T, 1, C]
        return (output, attn_weights) if need_weights else output


class CausalConv1d(nn.Module):
    """因果的1D畳み込み (局所的な文脈学習)"""
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.padding = (kernel_size - 1)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=0)
        self.act = get_activation()  # 活性化関数をconfigから選択

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


class BroadcastBackAttention(nn.Module):
    """USE_BROADCAST_BACK_ATTENTION=True 時、集約前の個々の変異embeddingが、時系列
    Transformer Encoder適用後の他タイムステップの代表ベクトル群に直接Cross-Attentionする。

    CoOccurrenceAttentionは各タイムステップの共起変異集合を1本のベクトルへ潰すため、
    その後の時系列Transformer Encoderは代表ベクトル同士でしかAttentionを取れず、
    「ある共起変異群の1つの変異」と「他タイムステップの変異」間の個別粒度での
    Attentionが構造的に失われる。本モジュールはその是正として、個々の変異embedding
    （Query）が時系列文脈化済みの代表ベクトル列 r'_1..r'_T（Key/Value）へ直接
    Cross-Attentionすることで、個別粒度の情報を一度取り戻す（出力は呼び出し側で
    CoOccurrenceAttentionにより再集約される想定）。
    """
    def __init__(self):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=config.FEATURE_DIM,
            num_heads=getattr(config, 'BROADCAST_BACK_ATTENTION_HEADS', None) or config.N_HEADS,
            dropout=config.DROPOUT,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(config.FEATURE_DIM)
        self.dropout = nn.Dropout(config.DROPOUT)

    def forward(self, x_mutations, r_context, context_key_padding_mask=None):
        """
        x_mutations: [B, T, C, F] 集約前の個々の変異embedding（Query）
        r_context:   [B, T', F]   時系列Transformer Encoder適用後の代表ベクトル列（Key/Value）
                     （T'はTEMPORAL_POOLING='cls'でCLSトークンを含む場合 T+1）
        context_key_padding_mask: [B, T'] bool, True=PAD（r_context側のPADタイムステップ）

        Returns:
            [B, T, C, F] 個々の変異ごとに他タイムステップの文脈を取り込んだembedding
        """
        B, T, C, F = x_mutations.shape
        Tc = r_context.size(1)
        q_flat = x_mutations.reshape(B * T, C, F)

        # r_context [B, Tc, F] を、B*T個の各クエリグループが同一batch内の同じキー系列を
        # 参照できるよう [B*T, Tc, F] に展開する（クエリ側のtに依らずbが同じなら同じ系列）。
        kv = r_context.unsqueeze(1).expand(B, T, Tc, F).reshape(B * T, Tc, F)
        kpm = None
        if context_key_padding_mask is not None:
            kpm = context_key_padding_mask.unsqueeze(1).expand(B, T, Tc).reshape(B * T, Tc)

        attn_out, _ = self.attn(q_flat, kv, kv, key_padding_mask=kpm, need_weights=False)
        attn_out = self.norm(self.dropout(attn_out))
        return attn_out.reshape(B, T, C, F)


class MixturePositionHead(nn.Module):
    """提案8: 多峰対応の Position 出力ヘッド（Mixture of Softmax Experts）。

    K 個のエキスパート softmax をゲートで混合し、混合確率の対数を疑似ロジットとして返す。
    混合確率は語彙上で総和 1 のため log_softmax は恒等（logsumexp=0）となり、
    下流の CrossEntropyLoss / topk / softmax をそのまま利用できる。
    負荷分散正則化（ゲート崩壊防止）を forward 毎に self.aux_loss へ格納する。
    """
    def __init__(self, in_dim, vocab_size, k, dropout, load_balance_weight):
        super().__init__()
        self.k = k
        self.load_balance_weight = load_balance_weight
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, in_dim),
                get_activation(),
                nn.LayerNorm(in_dim),
                nn.Dropout(dropout),
                nn.Linear(in_dim, vocab_size),
            ) for _ in range(k)
        ])
        self.gate = nn.Linear(in_dim, k)
        self.aux_loss = None

    def forward(self, h):
        # h: [B, in_dim]
        gate = torch.softmax(self.gate(h), dim=-1)                              # [B, K]
        log_probs = torch.stack(
            [torch.log_softmax(e(h), dim=-1) for e in self.experts], dim=1
        )                                                                       # [B, K, V]
        mixed_log = torch.logsumexp(
            torch.log(gate.clamp_min(1e-9)).unsqueeze(-1) + log_probs, dim=1
        )                                                                       # [B, V]
        # 負荷分散: バッチ平均ゲートを一様分布に近づける
        gate_mean = gate.mean(dim=0)                                            # [K]
        uniform = torch.full_like(gate_mean, 1.0 / self.k)
        self.aux_loss = self.load_balance_weight * ((gate_mean - uniform) ** 2).sum()
        return mixed_log


class HierarchicalTransformer(nn.Module):
    def __init__(self):
        super().__init__()

        self.input_embed = InputEmbedding()
        self.co_attn = CoOccurrenceAttention()

        # ③ USE_FLAT_COATTN: 共起集約をスキップし全変異を独立トークンとして Transformer に渡す
        self.use_flat_coattn = getattr(config, 'USE_FLAT_COATTN', False)
        self.flat_1d_pe = getattr(config, 'FLAT_COATTN_1D_PE', False)
        # 2D PE（タイムステップ + 共起内インデックス）は FLAT_COATTN_1D_PE=False のときのみ使用
        if self.use_flat_coattn and not self.flat_1d_pe:
            self.flat_ts_embed = nn.Embedding(config.MAX_SEQ_LEN + 2, config.FEATURE_DIM)
            self.flat_co_embed = nn.Embedding(config.MAX_CO_OCCURRENCE + 1, config.FEATURE_DIM)
        else:
            self.flat_ts_embed = None
            self.flat_co_embed = None

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
                torch.randn(1, 1, config.FEATURE_DIM) * getattr(config, 'INITIALIZATION_SCALE', 0.02)
            )
        else:
            self.origin_attn = None
            self.origin_embedding = None

        # Broadcast-back Cross-Attention（Ablation Study用に切り替え可能）
        self.use_broadcast_back = getattr(config, 'USE_BROADCAST_BACK_ATTENTION', False)
        if self.use_broadcast_back:
            self.broadcast_back_attn = BroadcastBackAttention()
            # 学習可能ゲート（0初期化）: 新規パスが既存の latest_context を初手から
            # 乱さないよう、tanh(gate)=0 スタートから寄与を徐々に学習させる
            # （Flamingoのgated cross-attention等と同様の安定化トリック）。
            self.broadcast_back_gate = nn.Parameter(torch.zeros(1))
        else:
            self.broadcast_back_attn = None
            self.broadcast_back_gate = None

        # [CLS] トークン（BERT方式）- TEMPORAL_POOLING='cls' の場合に使用
        # シーケンス先頭に追加し、Self-Attention を通じてシーケンス全体の情報を集約する
        # モード切り替えのたびにモデルを作り直す必要を避けるため、常に定義しておく
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.FEATURE_DIM) * getattr(config, 'INITIALIZATION_SCALE', 0.02))

        # Shared Trunk（全ヘッド共通の中間層）（Ablation Study用）
        # latest_context → Shared Trunk → 各ヘッドにわたるスイッチを容易にするため常に定義
        if getattr(config, 'USE_SHARED_TRUNK', False):
            self.shared_trunk = nn.Sequential(
                nn.Linear(config.FEATURE_DIM, config.FEATURE_DIM),
                get_activation(),
                nn.LayerNorm(config.FEATURE_DIM),
            )
        else:
            self.shared_trunk = None

        self.pos_encoder = PositionalEncoding(config.FEATURE_DIM, config.DROPOUT)

        # ALiBi 相対位置エンコーディング（USE_RPE=True の場合に正弦波 PE を置き換え）
        if getattr(config, 'USE_RPE', False):
            self.alibi = ALibiPositionalBias(num_heads=config.N_HEADS)
        else:
            self.alibi = None


        # Transformer Encoder (Pre-Norm & GeLU 採用)
        encoder_layer = TransformerEncoderLayer(
            d_model=config.FEATURE_DIM,
            nhead=config.N_HEADS,
            dim_feedforward=config.FEATURE_DIM * getattr(config, 'FFN_RATIO', 4),
            dropout=config.DROPOUT,
            batch_first=True,
            activation=getattr(config, 'ACTIVATION', 'gelu').lower(),
            norm_first=getattr(config, 'NORM_FIRST', True)
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=config.N_LAYERS)

        # 予測ヘッド (6タスク)
        # 1. Region予測ヘッド
        self.output_head = nn.Sequential(
            nn.Linear(config.FEATURE_DIM, config.FEATURE_DIM),
            get_activation(),
            nn.LayerNorm(config.FEATURE_DIM),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.FEATURE_DIM, config.NUM_REGIONS)
        )

        # 2. 塩基位置予測ヘッド
        # 提案8: USE_MIXTURE_POSITION_HEAD=True で多峰対応の Mixture ヘッドに差し替え。
        # 出力 shape は [B, VOCAB_SIZE_POSITION] で変わらないため下流は無改変。
        if getattr(config, 'USE_MIXTURE_POSITION_HEAD', False):
            self.position_head = MixturePositionHead(
                in_dim=config.FEATURE_DIM,
                vocab_size=config.VOCAB_SIZE_POSITION,
                k=getattr(config, 'POSITION_MIXTURE_K', 4),
                dropout=config.DROPOUT,
                load_balance_weight=getattr(config, 'MIXTURE_LOAD_BALANCE_WEIGHT', 0.01),
            )
        else:
            self.position_head = nn.Sequential(
                nn.Linear(config.FEATURE_DIM, config.FEATURE_DIM),
                get_activation(),
                nn.LayerNorm(config.FEATURE_DIM),
                nn.Dropout(config.DROPOUT),
                nn.Linear(config.FEATURE_DIM, config.VOCAB_SIZE_POSITION)
            )

        # 3. アミノ酸配列位置予測ヘッド
        self.aa_pos_head = nn.Sequential(
            nn.Linear(config.FEATURE_DIM, config.FEATURE_DIM),
            get_activation(),
            nn.LayerNorm(config.FEATURE_DIM),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.FEATURE_DIM, config.VOCAB_SIZE_AA_POS)
        )

        # 4. 流行度予測ヘッド (回帰)
        self.strength_head = nn.Sequential(
            nn.Linear(config.FEATURE_DIM, config.FEATURE_DIM),
            get_activation(),
            nn.LayerNorm(config.FEATURE_DIM),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.FEATURE_DIM, 1)
        )

        # 5. コドン位置予測ヘッド (1, 2, 3 の3クラス + PAD=0 で6クラス)
        self.codon_pos_head = nn.Sequential(
            nn.Linear(config.FEATURE_DIM, config.FEATURE_DIM // 2),
            get_activation(),
            nn.LayerNorm(config.FEATURE_DIM // 2),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.FEATURE_DIM // 2, config.VOCAB_SIZE_CODON_POS)
        )

        # 6. シノニマス/ノンシノニマス予測ヘッド (2クラス分類)
        self.synonymous_head = nn.Sequential(
            nn.Linear(config.FEATURE_DIM, config.FEATURE_DIM // 4),
            get_activation(),
            nn.LayerNorm(config.FEATURE_DIM // 4),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.FEATURE_DIM // 4, config.VOCAB_SIZE_SYNONYMOUS)
        )

        # --- Item 5: Substitution Prediction Head ---
        # base_after (塩基変化先: VOCAB_SIZE_BASE=7) と aa_after (AA変化先: VOCAB_SIZE_AA=23) を予測
        # USE_SUBSTITUTION_HEAD=False の場合 None を返すが、ヘッド自体は定義しておく
        if getattr(config, 'USE_SUBSTITUTION_HEAD', False):
            self.base_after_head = nn.Sequential(
                nn.Linear(config.FEATURE_DIM, config.FEATURE_DIM // 4),
                get_activation(),
                nn.LayerNorm(config.FEATURE_DIM // 4),
                nn.Dropout(config.DROPOUT),
                nn.Linear(config.FEATURE_DIM // 4, config.VOCAB_SIZE_BASE)
            )
            self.aa_after_head = nn.Sequential(
                nn.Linear(config.FEATURE_DIM, config.FEATURE_DIM // 2),
                get_activation(),
                nn.LayerNorm(config.FEATURE_DIM // 2),
                nn.Dropout(config.DROPOUT),
                nn.Linear(config.FEATURE_DIM // 2, config.VOCAB_SIZE_AA)
            )
        else:
            self.base_after_head = None
            self.aa_after_head   = None

        # --- Item 4: Autoregressive Decoder ---
        # 各タスク用のクエリ埋め込みを TransformerDecoder に通し、タスク別の特徴を生成する
        # N_TASKS = 6 (base 6 tasks; 拡張時は対応するスライスも更新が必要)
        scale = getattr(config, 'INITIALIZATION_SCALE', 0.02)
        if getattr(config, 'USE_AUTOREGRESSIVE_DECODER', False):
            self.task_queries = nn.Parameter(torch.randn(6, 1, config.FEATURE_DIM) * scale)
            decoder_layer = TransformerDecoderLayer(
                d_model=config.FEATURE_DIM,
                nhead=getattr(config, 'AR_DECODER_HEADS', 4),
                dim_feedforward=config.FEATURE_DIM * getattr(config, 'FFN_RATIO', 4),
                dropout=config.DROPOUT,
                batch_first=True,
                norm_first=getattr(config, 'NORM_FIRST', True),
            )
            self.ar_decoder = TransformerDecoder(
                decoder_layer,
                num_layers=getattr(config, 'AR_DECODER_LAYERS', 1),
            )
        else:
            self.task_queries = None
            self.ar_decoder   = None

        # --- Item 12: SupCon Projection Head ---
        # latest_context [B, FEATURE_DIM] → 正規化された射影ベクトル [B, SUPCON_PROJECTION_DIM]
        # forward() 後に self._last_projections にセットされる (train.py から参照)
        if getattr(config, 'USE_SUPCON', False):
            proj_dim = getattr(config, 'SUPCON_PROJECTION_DIM', 128)
            self.supcon_projector = nn.Sequential(
                nn.Linear(config.FEATURE_DIM, config.FEATURE_DIM),
                nn.ReLU(),
                nn.Linear(config.FEATURE_DIM, proj_dim),
            )
        else:
            self.supcon_projector = None
        self._last_projections = None  # train.py が参照するための属性

        # --- 提案9: Abstention（棄却）ヘッド ---
        # latest_context → スカラー: サンプルの多義性 log(1+ターゲット数) を回帰する。
        # forward() 後に self._last_uncertainty にセット（train.py / evaluate.py が参照）。
        if getattr(config, 'USE_ABSTENTION_HEAD', False):
            self.abstention_head = nn.Sequential(
                nn.Linear(config.FEATURE_DIM, config.FEATURE_DIM // 4),
                get_activation(),
                nn.Linear(config.FEATURE_DIM // 4, 1),
            )
        else:
            self.abstention_head = None
        self._last_uncertainty = None
        self._last_mixture_aux = None   # 提案8: 負荷分散正則化項（train.py が加算）

        # --- 提案11: 分子時計（枝長）補助ヘッド ---
        # latest_context → スカラー: 次の変異までの枝長を回帰。
        # 【要 preprocess 再実行】学習ターゲット(枝長)が labels に無い場合、train.py 側で
        # 損失加算はスキップされる（ヘッドは定義のみで無害）。
        if getattr(config, 'USE_BRANCH_LENGTH_AUX', False):
            self.branch_length_head = nn.Sequential(
                nn.Linear(config.FEATURE_DIM, config.FEATURE_DIM // 4),
                get_activation(),
                nn.Linear(config.FEATURE_DIM // 4, 1),
            )
        else:
            self.branch_length_head = None
        self._last_branch_length = None

        # --- 提案4: 主要系統クレード埋め込み（分布シフト対策）---
        # strain 名から写像した clade_id の Embedding を pooled 表現(latest_context)へ加算。
        # padding_idx=0（unknown）はゼロベクトルなので、未知系統では現行挙動と一致。
        # clade_ids は forward() の引数で受け取る（None のときは加算しない）。
        if getattr(config, 'USE_CLADE_EMBEDDING', False):
            from .utils.clade import NUM_CLADES
            self.clade_embed = nn.Embedding(NUM_CLADES, config.FEATURE_DIM, padding_idx=0)
        else:
            self.clade_embed = None

        # --- 提案10: ホモプラシー（収斂進化）事前分布 ---
        # 位置別再発回数 CSV を読み込み log(1+r) を Position ロジットへ加算するバイアスを構築。
        # 学習可能スケール係数付き。CSV が無ければバイアス無効（現行と同一）にフォールバック。
        self.homoplasy_scale = None
        self.register_buffer('homoplasy_bias', None, persistent=False)
        if getattr(config, 'USE_HOMOPLASY_PRIOR', False):
            bias_vec = self._load_homoplasy_bias('USE_HOMOPLASY_PRIOR')
            if bias_vec is not None:
                self.homoplasy_bias = bias_vec
                self.homoplasy_scale = nn.Parameter(
                    torch.tensor(float(getattr(config, 'HOMOPLASY_PRIOR_SCALE', 1.0)))
                )

        # --- 提案: Co-occurrence Attentionへの頻度ペナルティ ---
        # 大きな共起グループ(オミクロン系統等)で、ほぼ全系統に共通する創始変異
        # (nsp12:P323L等)にAttentionが機械的に収束する現象を
        # verify_coattn_frequency_confound.py で確認した（embeddingノルムとの相関は
        # verify_position_embedding_norm_confound.py で否定済み=機構不問の対策とする）。
        # ホモプラシー再発回数(HOMOPLASY_CSVを流用)が高い変異ほど、Cross-Attentionの
        # スコアに学習可能スケール付きの負バイアスを加算し、系統非依存の"定番"変異への
        # 注目を抑制する。USE_HOMOPLASY_PRIOR(position headへの正バイアス)と対称的な設計。
        self.coattn_freq_penalty_scale = None
        self.register_buffer('coattn_freq_penalty_bias', None, persistent=False)
        if getattr(config, 'USE_COATTN_FREQUENCY_PENALTY', False):
            bias_vec = self._load_homoplasy_bias('USE_COATTN_FREQUENCY_PENALTY')
            if bias_vec is not None:
                self.coattn_freq_penalty_bias = bias_vec
                self.coattn_freq_penalty_scale = nn.Parameter(
                    torch.tensor(float(getattr(config, 'COATTN_FREQUENCY_PENALTY_SCALE', 1.0)))
                )

        # --- Items 8/9/10: 外部特徴量スタブ ---
        # ESM-2: preprocess.py で抽出した埋め込みを受け取り、FEATURE_DIM 互換の64次元に射影する
        # 実際の特徴量は forward() の esm2_features 引数 (Optional Tensor [B, T, ESM2_EMBED_DIM]) で渡す予定
        # 現状はスタブ実装のみ。有効化時は InputEmbedding.forward() の引数拡張と連携が必要。
        if getattr(config, 'USE_ESM2', False):
            esm2_dim = getattr(config, 'ESM2_EMBED_DIM', 320)
            self.esm2_projection = nn.Linear(esm2_dim, 64)
        else:
            self.esm2_projection = None
        # USE_STRUCTURE_FEATURES / USE_EVESCAPE: num_norm の入力次元を拡張する予定
        # 現状は config フラグのみ定義。実際の統合は preprocess.py での特徴量付与が先決。

    def _load_homoplasy_bias(self, context='USE_HOMOPLASY_PRIOR'):
        """位置別再発回数 CSV から log(1+r) バイアスベクトルを構築する（提案10）。

        USE_HOMOPLASY_PRIOR（position headへの正バイアス）と
        USE_COATTN_FREQUENCY_PENALTY（Co-occurrence Attentionへの負バイアス）の
        双方が同じCSV・同じ log(1+r) ベクトルを共有する（用途ごとに別スケールで符号反転）。
        CSV は 'position_id','recurrence_count' 列を持つ。存在しない/読めない場合は
        None を返し、呼び出し元の機能を無効化する（現行挙動と同一）。
        """
        import os
        path = getattr(config, 'HOMOPLASY_CSV', '')
        if not path or not os.path.exists(path):
            _log.force_print(f"[WARN] {context}=True but CSV not found: {path}. Disabled.")
            return None
        try:
            import pandas as pd
            df = pd.read_csv(path)
            bias = torch.zeros(config.VOCAB_SIZE_POSITION, dtype=torch.float)
            for pos_id, rec in zip(df['position_id'].astype(int), df['recurrence_count'].astype(float)):
                if 0 <= pos_id < config.VOCAB_SIZE_POSITION:
                    bias[pos_id] = math.log1p(max(rec, 0.0))
            _log.force_print(f"[INFO] Homoplasy bias loaded ({context}): {len(df)} sites from {path}")
            return bias
        except Exception as e:
            _log.force_print(f"[WARN] Failed to load homoplasy CSV ({path}) for {context}: {e}. Disabled.")
            return None

    def forward(self, x_cat, x_num, src_mask=None, src_key_padding_mask=None, clade_ids=None):
        # 1. 入力埋め込み
        x = self.input_embed(x_cat, x_num)

        if self.use_flat_coattn:
            # ③ Flat Co-occurrence: 全変異を独立トークンとして Transformer に渡す
            # [B, T, C, F] → [B, T*C, F]
            B, T, C, F_dim = x.shape
            x_flat_seq = x.reshape(B, T * C, F_dim)

            if self.flat_1d_pe:
                # 1D 正弦波 PE: 共起グループを識別しない = PETra相当の純粋時系列扱い
                x_flat_seq = self.pos_encoder(x_flat_seq)
            else:
                # 2D PE: タイムステップ次元 + 共起内インデックス次元（共起グループを識別）
                ts_idx = torch.arange(T, device=x.device)
                co_idx = torch.arange(C, device=x.device)
                pos_2d = (self.flat_ts_embed(ts_idx).unsqueeze(1) +
                          self.flat_co_embed(co_idx).unsqueeze(0)).reshape(T * C, F_dim)
                x_flat_seq = x_flat_seq + pos_2d.unsqueeze(0)
                x_flat_seq = self.pos_encoder.dropout(x_flat_seq)

            # PAD マスク: x_cat[..., 0] == 0 は PAD 変異（base_before=PAD トークン）
            mutation_pad  = (x_cat[..., 0] == 0)            # [B, T, C]
            flat_pad_mask = mutation_pad.reshape(B, T * C)   # [B, T*C]

            x_enc = self.transformer_encoder(
                x_flat_seq,
                mask=None,
                src_key_padding_mask=flat_pad_mask,
            )  # [B, T*C, F]

            # 最終タイムステップの変異ベクトルを PAD 除外平均して latest_context を作成
            last_start  = (T - 1) * C
            last_tokens = x_enc[:, last_start:last_start + C, :]              # [B, C, F]
            last_valid  = (~flat_pad_mask[:, last_start:last_start + C]).float().unsqueeze(-1)  # [B, C, 1]
            latest_context = (last_tokens * last_valid).sum(1) / last_valid.sum(1).clamp(min=1)

        else:
            # 2. 共起集約 (ベース情報)
            # base_before==0 は PAD トークン（BASE_VOCABS['PAD']=0）
            co_occur_mask = (x_cat[..., 0] == 0)  # [B, T, C] True=PAD
            # Broadcast-back Cross-Attention用に集約前の個々の変異embeddingを保持しておく
            # （下の x_agg = self.co_attn(x, ...) 以降、変数名 x は上書きされていくため）
            x_mutations = x  # [B, T, C, F]

            # 提案: 頻度ペナルティ無効時は引数を一切追加せず既存呼び出しのまま
            # （USE_FLAT_COATTN分岐と対称に、co_attn.forwardのシグネチャ互換性を壊さないため）
            co_attn_kwargs = {}
            if self.coattn_freq_penalty_bias is not None:
                pos_ids = x_cat[..., 1]  # POS_COL、共起スロットのゲノム位置ID [B, T, C]
                co_attn_kwargs['freq_penalty'] = (
                    self.coattn_freq_penalty_scale * self.coattn_freq_penalty_bias[pos_ids])
            x_agg = self.co_attn(x, co_occur_mask=co_occur_mask, **co_attn_kwargs)

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
            #    TEMPORAL_POOLING='cls' の場合: pos_encoder の前に [CLS] トークンを先頭に差し込む
            pooling = getattr(config, 'TEMPORAL_POOLING', 'last')
            key_padding_mask = src_key_padding_mask  # forward 引数を破壊しないようローカル変数に

            if pooling == 'cls':
                B = x_combined.size(0)
                cls_tokens = self.cls_token.expand(B, -1, -1)              # [B, 1, F]
                x_combined = torch.cat([cls_tokens, x_combined], dim=1)   # [B, T+1, F]
                if key_padding_mask is not None:
                    cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=key_padding_mask.device)
                    key_padding_mask = torch.cat([cls_mask, key_padding_mask], dim=1)  # [B, T+1]

            # ALiBi RPE: USE_RPE=True の場合は正弦波 PE をスキップし、
            #            ALiBi バイアスを attn_mask に注入する
            if self.alibi is not None:
                B_ = x_combined.size(0)
                T_ = x_combined.size(1)
                # Dropout は正弦波 PE に内包されているので、RPE 時は別途 Dropout を適用
                x = self.pos_encoder.dropout(x_combined)
                alibi_mask = self.alibi(T_, B_, config.N_HEADS, x.device)  # [B*n_heads, T, T]
                effective_mask = src_mask if src_mask is not None else alibi_mask
                if src_mask is not None:
                    # src_mask と alibi を加算で合成
                    effective_mask = src_mask + alibi_mask
            else:
                x = self.pos_encoder(x_combined)
                effective_mask = src_mask

            x = self.transformer_encoder(
                x,
                mask=effective_mask,
                src_key_padding_mask=key_padding_mask
            )

            # 6. 予測ヘッドへの入力 vector を TEMPORAL_POOLING に従って選択
            if pooling == 'mean':
                if key_padding_mask is not None:
                    mask_float = (~key_padding_mask).float().unsqueeze(-1)  # [B, T, 1]
                    latest_context = (x * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1)
                else:
                    latest_context = x.mean(dim=1)
            elif pooling == 'cls':
                latest_context = x[:, 0, :]
            else:  # 'last'
                latest_context = x[:, -1, :]

            # Broadcast-back Cross-Attention: 個々の変異embedding（Query）が時系列
            # Transformer Encoder適用後の代表ベクトル列 r'_1..r'_T（Key/Value）へ
            # 直接Cross-Attentionし、CoOccurrenceAttentionで再集約した r''_t を
            # 学習可能ゲート付き残差として latest_context に加算する。
            if self.use_broadcast_back and self.broadcast_back_attn is not None:
                if pooling == 'cls':
                    # CLSトークンは変異timestepではないためbroadcast-back先から除く
                    r_context = x[:, 1:, :]
                    ctx_kpm = key_padding_mask[:, 1:] if key_padding_mask is not None else None
                else:
                    r_context = x
                    ctx_kpm = key_padding_mask

                h = self.broadcast_back_attn(x_mutations, r_context, context_key_padding_mask=ctx_kpm)
                x_refined_agg = self.co_attn(h, co_occur_mask=co_occur_mask, **co_attn_kwargs)  # [B, T, F]

                if pooling == 'mean':
                    if ctx_kpm is not None:
                        mask_float = (~ctx_kpm).float().unsqueeze(-1)
                        refined_context = (x_refined_agg * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1)
                    else:
                        refined_context = x_refined_agg.mean(dim=1)
                elif pooling == 'cls':
                    # CLS自体には対応するタイムステップが無いため全タイムステップの平均を側注として加算
                    refined_context = x_refined_agg.mean(dim=1)
                else:  # 'last'
                    refined_context = x_refined_agg[:, -1, :]

                latest_context = latest_context + torch.tanh(self.broadcast_back_gate) * refined_context

        # 提案4: 主要系統クレード埋め込みを pooled 表現へ加算（分布シフト対策）
        # clade_ids [B] が渡され USE_CLADE_EMBEDDING=True のときのみ作用する。
        if self.clade_embed is not None and clade_ids is not None:
            latest_context = latest_context + self.clade_embed(clade_ids.to(latest_context.device))

        # Shared Trunk: 実験時は USE_SHARED_TRUNK=True に変更
        if self.shared_trunk is not None:
            latest_context = self.shared_trunk(latest_context)

        # 提案12: kNN データストア構築・検索のためエンコーダ表現を公開（非破壊）
        self._last_context = latest_context

        # --- Item 12: SupCon 射影ベクトルの計算 ---
        # forward 後に self._last_projections にセット (train.py から参照)
        if self.supcon_projector is not None:
            proj = self.supcon_projector(latest_context)           # [B, proj_dim]
            self._last_projections = nn.functional.normalize(proj, dim=1)
        else:
            self._last_projections = None

        # --- Item 4: Autoregressive Decoder ---
        # タスク別クエリを Decoder に通し、各タスク固有の特徴を生成する
        # decoded[:, i, :] を各タスクヘッドに渡す（0=region, 1=position, ...）
        if self.ar_decoder is not None:
            B_size = latest_context.size(0)
            memory = latest_context.unsqueeze(1)                          # [B, 1, F]
            queries = self.task_queries.expand(-1, B_size, -1).permute(1, 0, 2)  # [B, 6, F]
            decoded = self.ar_decoder(queries, memory)                    # [B, 6, F]
            ctx_region    = decoded[:, 0, :]
            ctx_position  = decoded[:, 1, :]
            ctx_aa_pos    = decoded[:, 2, :]
            ctx_strength  = decoded[:, 3, :]
            ctx_codon_pos = decoded[:, 4, :]
            ctx_synonymous= decoded[:, 5, :]
        else:
            ctx_region = ctx_position = ctx_aa_pos = ctx_strength = ctx_codon_pos = ctx_synonymous = latest_context

        output_region    = self.output_head(ctx_region)
        output_position  = self.position_head(ctx_position)
        # 提案8: Mixture ヘッドの負荷分散正則化項を公開（train.py が total_loss に加算）
        self._last_mixture_aux = getattr(self.position_head, 'aux_loss', None)
        # 提案10: ホモプラシー事前分布を Position ロジットへ加算（後続 softmax/CE で再正規化）
        if self.homoplasy_bias is not None and self.homoplasy_scale is not None:
            output_position = output_position + self.homoplasy_scale * self.homoplasy_bias
        output_aa_pos    = self.aa_pos_head(ctx_aa_pos)
        output_strength  = self.strength_head(ctx_strength).squeeze(-1)  # [B, 1] -> [B]
        output_codon_pos = self.codon_pos_head(ctx_codon_pos)
        output_synonymous= self.synonymous_head(ctx_synonymous)

        # --- Item 5: Substitution Prediction Head ---
        # USE_SUBSTITUTION_HEAD=True の場合のみ計算、False なら None を返す
        if self.base_after_head is not None:
            output_base_after = self.base_after_head(latest_context)  # [B, VOCAB_SIZE_BASE]
            output_aa_after   = self.aa_after_head(latest_context)    # [B, VOCAB_SIZE_AA]
        else:
            output_base_after = None
            output_aa_after   = None

        # 提案9: 不確実性（棄却）スコアを公開（train.py が回帰損失、evaluate.py が棄却判定に使用）
        if self.abstention_head is not None:
            self._last_uncertainty = self.abstention_head(latest_context).squeeze(-1)  # [B]
        else:
            self._last_uncertainty = None

        # 提案11: 枝長予測を公開（train.py が枝長ターゲット存在時のみ損失加算）
        if self.branch_length_head is not None:
            self._last_branch_length = self.branch_length_head(latest_context).squeeze(-1)  # [B]
        else:
            self._last_branch_length = None

        return (output_region, output_position, output_aa_pos, output_strength,
                output_codon_pos, output_synonymous, output_base_after, output_aa_after)


class MultiTaskLoss(nn.Module):
    """
    複数のタスクの損失を、不確実性(Uncertainty)に基づいて自動重み付けする層。
    Alex Kendall et al. "Multi-Task Learning Using Uncertainty to Weigh Losses"
    """
    def __init__(self, num_tasks=6):
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
    """Focal Loss for multi-class classification (Lin et al., 2017)
    クラス不均衡問題に対応し、難しいサンプルに重点を置いた学習を行う。

    [260417] Soft Target（確率分配ベクトル）に対応した Approach A を実装。

    ● Hard Target モード (targets: LongTensor [N])
        FL = -(1 - p_t)^γ × log(p_t)
        従来と同じ動作。

    ● Soft Target モード (targets: FloatTensor [N, C])
        FL = -sum_c [ soft_target[c] × (1 - softmax[c])^γ × log_softmax[c] ]
        各クラスに独立した focal weight (1 - p_c)^γ を掛け、
        soft_target[c] との内積でスカラー損失を計算する。
        これにより「各クラスの予測が難しいほど、そのクラスへの確信度を重視」する。
    """
    def __init__(self, gamma=2.0, alpha=None, reduction='none', label_smoothing=0.0):
        """
        Args:
            gamma: focusing parameter (default: 2.0)
                   gamma=0 で通常のCrossEntropyLossと等価
            alpha: クラス重み (Tensor [C] or float or None)
            reduction: 'none', 'mean', 'sum'
            label_smoothing: ラベルスムージング係数 (0.0-1.0, Hard Targetモードのみ有効)
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        """
        Args:
            inputs:  [N, C] - 未正規化のロジット (softmax前)
            targets: [N]    - 正解クラスのインデックス (Hard Target, LongTensor)
                     [N, C] - 確率分配ベクトル       (Soft Target, FloatTensor, sum≈1.0)
        Returns:
            loss: スカラー or [N] (reduction による)
        """
        log_probs = torch.nn.functional.log_softmax(inputs, dim=-1)  # [N, C]
        probs     = torch.exp(log_probs)                              # [N, C]

        # ── Soft Target モード (Approach A) ─────────────────────────────
        if targets.dtype == torch.float32 or targets.dtype == torch.float64:
            # targets: [N, C] の確率ベクトル
            # focal_weight[n, c] = (1 - softmax[n, c])^γ (各クラス独立)
            focal_weight = (1.0 - probs) ** self.gamma  # [N, C]

            # クラス重み alpha の適用 (Tensor [C] の場合)
            if self.alpha is not None:
                if isinstance(self.alpha, (float, int)):
                    focal_weight = self.alpha * focal_weight
                else:
                    # alpha: [C] → broadcast to [N, C]
                    alpha_t = self.alpha.to(inputs.device).unsqueeze(0)
                    focal_weight = alpha_t * focal_weight  # [N, C]

            # loss[n] = -sum_c [ soft_target[n,c] × focal_weight[n,c] × log_softmax[n,c] ]
            loss = -(targets * focal_weight * log_probs).sum(dim=-1)  # [N]

        # ── Hard Target モード（従来互換） ───────────────────────────────
        else:
            num_classes = inputs.size(-1)

            if self.label_smoothing > 0:
                # Smooth targets で CE を計算
                with torch.no_grad():
                    smooth_targets = torch.zeros_like(log_probs)
                    smooth_targets.fill_(self.label_smoothing / num_classes)
                    smooth_targets.scatter_(
                        1, targets.unsqueeze(1),
                        1.0 - self.label_smoothing + self.label_smoothing / num_classes
                    )
                ce_loss = -(smooth_targets * log_probs).sum(dim=-1)
                pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
            else:
                ce_loss = torch.nn.functional.nll_loss(log_probs, targets, reduction='none')
                pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

            # focal weight は「正解クラスの確率」のみで計算 (従来通り)
            focal_weight = (1.0 - pt) ** self.gamma

            if self.alpha is not None:
                if isinstance(self.alpha, (float, int)):
                    focal_weight = self.alpha * focal_weight
                else:
                    alpha_t = self.alpha.gather(0, targets)
                    focal_weight = alpha_t * focal_weight

            loss = focal_weight * ce_loss  # [N]

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

