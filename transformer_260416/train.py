# --- train.py ---
import torch
from tqdm import tqdm
from . import config
from .utils.losses import compute_task_losses


def train_one_epoch(model, dataloader, optimizer, loss_fn, loss_wrapper=None):
    """1エポック分の学習を実行する。

    Args:
        model: 学習対象モデル
        dataloader: 学習用DataLoader
        optimizer: 最適化アルゴリズム (AdamW等)
        loss_fn: 分類損失関数 (FocalLoss or CrossEntropyLoss)
        loss_wrapper: MultiTaskLoss インスタンス。Noneの場合は固定重みを使用

    Returns:
        float: エポック平均損失
    """
    model.train()
    total_epoch_loss = 0
    batches_processed = 0

    for (x_cat, x_num, mask), y_batch_list, _, _, batch_strength_scores, _, _, _ in tqdm(dataloader, desc="Training"):

        x_cat = x_cat.to(config.DEVICE)
        x_num = x_num.to(config.DEVICE)
        mask  = mask.to(config.DEVICE)

        predictions_tuple = model(x_cat, x_num, src_key_padding_mask=mask)

        losses = compute_task_losses(
            predictions_tuple, y_batch_list, loss_fn, batch_strength_scores
        )
        if losses is None:
            continue

        optimizer.zero_grad()

        if loss_wrapper is not None:
            # 自動重み付け (MultiTaskLoss) — 6タスク
            total_loss = loss_wrapper(
                losses['region'], losses['position'], losses['aa_pos'],
                losses['strength'], losses['codon_pos'], losses['synonymous'],
            )
        else:
            total_loss = losses['total']

        total_loss.backward()

        max_norm = getattr(config, "MAX_GRAD_NORM", None)
        if max_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        optimizer.step()

        total_epoch_loss += total_loss.item()
        batches_processed += 1

    return total_epoch_loss / batches_processed if batches_processed > 0 else 0
