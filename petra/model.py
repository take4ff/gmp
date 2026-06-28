# petra/model.py — Decoder-only Transformer (PETra相当)

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PetraDecoder(nn.Module):
    """Decoder-only Transformer（因果自己注意のみ、クロスアテンションなし）。

    PyTorch の TransformerEncoderLayer に is_causal=True を使うことで
    GPT-style の decoder-only Transformer を実現する。
    """

    def __init__(self, vocab_size: int, d_model: int, n_heads: int,
                 n_layers: int, ffn_dim: int, max_seq_len: int,
                 dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,    # Pre-LN（GPT-2 スタイル）
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers, enable_nested_tensor=False
        )

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
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device)

        x = self.drop(self.embed(input_ids) + self.pos_embed(pos))

        # 因果マスク（float: 0 = attend, -inf = block）
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=input_ids.device)
        x = self.transformer(x, mask=causal_mask, is_causal=True)
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
