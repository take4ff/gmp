# petra/train.py — CLM 学習・検証ループ
#
# AMP（自動混合精度、bfloat16）対応: v2は語彙拡大(132,335)に加え変異1件が6トークンに
# 展開されるため実効系列長も約6倍になり、logitsテンソル(B,T,V)の活性化メモリが
# v1比で大幅に増える（推定: batch=64,T=512上限で v1=12.1GB -> v2=17.35GB）。
# AMPで概ね半減させる。bfloat16はfp32と同じ指数範囲を持つため、fp16と異なり
# GradScalerによる損失スケーリングは不要。

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def train_epoch(model, loader: DataLoader, optimizer, scheduler,
                device: torch.device, grad_clip: float = 1.0,
                use_amp: bool = True) -> float:
    model.train()
    total_loss = 0.0
    total_tokens = 0
    amp_enabled = use_amp and device.type == 'cuda'

    for input_ids, target_ids, _mask, _countries, _dates, _is_leaked in loader:
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        optimizer.zero_grad()

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
            logits = model(input_ids)  # B, T, V
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target_ids.reshape(-1),
                ignore_index=-100,
            )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        n_tok = (target_ids != -100).sum().item()
        total_loss += loss.item() * n_tok
        total_tokens += n_tok

    return total_loss / total_tokens if total_tokens > 0 else 0.0


@torch.no_grad()
def evaluate_loss(model, loader: DataLoader, device: torch.device,
                  use_amp: bool = True) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    amp_enabled = use_amp and device.type == 'cuda'

    for input_ids, target_ids, _mask, _countries, _dates, _is_leaked in loader:
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
            logits = model(input_ids)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target_ids.reshape(-1),
                ignore_index=-100,
            )

        n_tok = (target_ids != -100).sum().item()
        total_loss += loss.item() * n_tok
        total_tokens += n_tok

    return total_loss / total_tokens if total_tokens > 0 else 0.0
