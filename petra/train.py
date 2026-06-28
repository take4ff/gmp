# petra/train.py — CLM 学習・検証ループ

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def train_epoch(model, loader: DataLoader, optimizer, scheduler,
                device: torch.device, grad_clip: float = 1.0) -> float:
    model.train()
    total_loss = 0.0
    total_tokens = 0

    for input_ids, target_ids, _ in loader:
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        logits = model(input_ids)  # B, T, V

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target_ids.reshape(-1),
            ignore_index=-100,
        )

        optimizer.zero_grad()
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
def evaluate_loss(model, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for input_ids, target_ids, _ in loader:
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

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
