# petra/model.py — Decoder-only Transformer (PETra相当、RoPE版)
#
# 原論文(petra/PETra)はMegatronのGPTModelを`position_embedding_type`/`rotary_percent`
# 引数付きで呼んでおり、RoPE(回転位置埋め込み)を使っている可能性が高い。学習可能な絶対
# 位置埋め込み(nn.Embedding)よりも原論文に近づけるため、本実装はRoPEを採用する。

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RotaryEmbedding(nn.Module):
    """RoPE の cos/sin テーブルを事前計算するモジュール。"""

    def __init__(self, dim: int, max_seq_len: int = 2048, base: int = 10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len).float()
        freqs = torch.einsum('i,j->ij', t, inv_freq)   # (max_seq_len, dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)        # (max_seq_len, dim)
        self.register_buffer('cos_cached', emb.cos(), persistent=False)
        self.register_buffer('sin_cached', emb.sin(), persistent=False)

    def forward(self, seq_len: int):
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def _apply_rotary_pos_emb(q, k, cos, sin):
    """q, k: (B, n_heads, T, d_head)。cos, sin: (T, d_head)。"""
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_rot = (q * cos) + (_rotate_half(q) * sin)
    k_rot = (k * cos) + (_rotate_half(k) * sin)
    return q_rot, k_rot


class CausalSelfAttentionRoPE(nn.Module):
    """RoPEを適用した因果自己注意（`F.scaled_dot_product_attention`で計算）。"""

    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, 'd_model must be divisible by n_heads'
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = dropout
        self.rotary = RotaryEmbedding(self.d_head, max_seq_len=max_seq_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                       # (B, n_heads, T, d_head)

        cos, sin = self.rotary(T)
        cos, sin = cos.to(x.device, x.dtype), sin.to(x.device, x.dtype)
        q, k = _apply_rotary_pos_emb(q, k, cos, sin)

        attn_out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(attn_out)


class PetraDecoderBlock(nn.Module):
    """Pre-LN の decoder ブロック（RoPE自己注意 + GELU FFN）。"""

    def __init__(self, d_model: int, n_heads: int, ffn_dim: int, max_seq_len: int,
                 dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttentionRoPE(d_model, n_heads, max_seq_len, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.attn(self.ln1(x)))
        x = x + self.drop(self.ffn(self.ln2(x)))
        return x


class PetraDecoder(nn.Module):
    """Decoder-only Transformer（因果自己注意のみ、クロスアテンションなし、RoPE版）。"""

    def __init__(self, vocab_size: int, d_model: int, n_heads: int,
                 n_layers: int, ffn_dim: int, max_seq_len: int,
                 dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            PetraDecoderBlock(d_model, n_heads, ffn_dim, max_seq_len, dropout)
            for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # 埋め込みとヘッドの重み共有
        self.head.weight = self.embed.weight

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (B, T)
        Returns:
            logits: (B, T, vocab_size)
        """
        x = self.drop(self.embed(input_ids))   # 位置情報はRoPEで各層の自己注意内に注入される
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)

    @torch.no_grad()
    def topk_next(self, prompt_ids: torch.Tensor, top_k: int) -> torch.Tensor:
        """プロンプト末尾の次トークン Top-K を返す。

        Args:
            prompt_ids: (B, T)
            top_k: 何位まで取るか
        Returns:
            topk_ids: (B, top_k)
        """
        self.eval()
        logits = self(prompt_ids)          # B, T, V
        next_logits = logits[:, -1, :]     # B, V
        return next_logits.topk(top_k, dim=-1).indices

    @torch.no_grad()
    def greedy_generate(self, prompt_ids: torch.Tensor, max_new_tokens: int,
                        eos_id: int) -> torch.Tensor:
        """Greedy 生成（評価・デモ用）。

        Args:
            prompt_ids: (B, T_prompt)
            max_new_tokens: 最大生成トークン数
            eos_id: EOS トークン ID
        Returns:
            generated: (B, T_prompt + generated_len)
        """
        self.eval()
        generated = prompt_ids.clone()
        for _ in range(max_new_tokens):
            logits = self(generated)
            next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # B, 1
            generated = torch.cat([generated, next_tok], dim=1)
            # 全バッチが EOS を出力したら終了
            if (next_tok.squeeze(-1) == eos_id).all():
                break
        return generated
