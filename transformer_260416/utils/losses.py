# --- utils/losses.py ---
# 損失計算の共通ロジック (train.py / evaluate.py 共用)

import torch
import torch.nn as nn
from . import logging as _log
from .. import config


def build_loss_fn():
    """設定に応じた分類損失関数を構築する。

    Returns:
        nn.Module: FocalLoss または CrossEntropyLoss
    """
    from ..model import FocalLoss

    if config.USE_FOCAL_LOSS:
        label_smooth = config.LABEL_SMOOTHING_FACTOR if config.USE_LABEL_SMOOTHING else 0.0
        _log.force_print(
            f"[INFO] Using Focal Loss (gamma={config.FOCAL_LOSS_GAMMA}, "
            f"label_smoothing={label_smooth})"
        )
        return FocalLoss(
            gamma=config.FOCAL_LOSS_GAMMA,
            reduction='none',
            label_smoothing=label_smooth,
        )
    elif config.USE_LABEL_SMOOTHING:
        _log.force_print(
            f"[INFO] Using CrossEntropyLoss with Label Smoothing "
            f"(factor={config.LABEL_SMOOTHING_FACTOR})"
        )
        return nn.CrossEntropyLoss(
            reduction='none',
            label_smoothing=config.LABEL_SMOOTHING_FACTOR,
        )
    else:
        _log.force_print("[INFO] Using CrossEntropyLoss")
        return nn.CrossEntropyLoss(reduction='none')


def compute_task_losses(
    predictions_tuple,
    y_batch_list,
    loss_fn,
    batch_strength_scores,
):
    """バッチ内の全サンプルに対して各タスクの損失を計算する。

    train.py・evaluate.py の共通ロジックを集約する。

    Args:
        predictions_tuple: モデルが返す 6タプル
            (pred_region, pred_position, pred_aa_pos,
             pred_strength, pred_codon_pos, pred_synonymous)
        y_batch_list: バッチ内の正解リスト (list of list of tuples)
        loss_fn: 分類損失関数 (FocalLoss or CrossEntropyLoss, reduction='none')
        batch_strength_scores: バッチ内の流行度スコア (list[float])

    Returns:
        dict with keys:
            'region', 'position', 'aa_pos', 'codon_pos', 'synonymous' — タスク平均損失 (Tensor)
            'strength' — MSE損失 (Tensor)
            'total' — 加重合計損失 (Tensor)
            'num_targets' — 有効ターゲット数 (int), 0 の場合は None を返す
        None: バッチ内に有効ターゲットがない場合
    """
    (predictions_region, predictions_position, predictions_aa_pos,
     predictions_strength, predictions_codon_pos, predictions_synonymous) = predictions_tuple

    mse_fn = nn.MSELoss()

    loss_region_total = 0.0
    loss_position_total = 0.0
    loss_aa_pos_total = 0.0
    loss_codon_pos_total = 0.0
    loss_synonymous_total = 0.0
    num_targets_in_batch = 0

    for i, targets_tuples in enumerate(y_batch_list):
        if not targets_tuples:
            continue

        num_targets = len(targets_tuples)

        # (region_id, position_id, aa_pos_id, codon_pos_id, is_synonymous)
        targets_region   = torch.tensor([t[0] for t in targets_tuples], dtype=torch.long, device=config.DEVICE)
        targets_position = torch.tensor([t[1] for t in targets_tuples], dtype=torch.long, device=config.DEVICE)
        targets_aa_pos   = torch.tensor([t[2] for t in targets_tuples], dtype=torch.long, device=config.DEVICE)
        targets_codon    = torch.tensor([t[3] for t in targets_tuples], dtype=torch.long, device=config.DEVICE)
        targets_syn      = torch.tensor([t[4] for t in targets_tuples], dtype=torch.long, device=config.DEVICE)

        loss_region_total     += loss_fn(predictions_region[i].expand(num_targets, -1),   targets_region).sum()
        loss_position_total   += loss_fn(predictions_position[i].expand(num_targets, -1), targets_position).sum()
        loss_aa_pos_total     += loss_fn(predictions_aa_pos[i].expand(num_targets, -1),   targets_aa_pos).sum()
        loss_codon_pos_total  += loss_fn(predictions_codon_pos[i].expand(num_targets, -1),targets_codon).sum()
        loss_synonymous_total += loss_fn(predictions_synonymous[i].expand(num_targets, -1),targets_syn).sum()
        num_targets_in_batch  += num_targets

    if num_targets_in_batch == 0:
        return None

    avg_region   = loss_region_total   / num_targets_in_batch
    avg_position = loss_position_total / num_targets_in_batch
    avg_aa_pos   = loss_aa_pos_total   / num_targets_in_batch
    avg_codon    = loss_codon_pos_total / num_targets_in_batch
    avg_syn      = loss_synonymous_total / num_targets_in_batch

    # 流行度 MSE
    target_strength = torch.tensor(
        batch_strength_scores, dtype=torch.float, device=config.DEVICE
    )
    loss_strength = mse_fn(predictions_strength, target_strength)

    # 合計損失
    total_loss = (
        config.LOSS_WEIGHT_REGION     * avg_region   +
        config.LOSS_WEIGHT_POSITION   * avg_position +
        config.LOSS_WEIGHT_AA_POS     * avg_aa_pos   +
        config.LOSS_WEIGHT_CODON_POS  * avg_codon    +
        config.LOSS_WEIGHT_SYNONYMOUS * avg_syn      +
        config.LOSS_WEIGHT_STRENGTH   * loss_strength
    )

    return {
        'region':      avg_region,
        'position':    avg_position,
        'aa_pos':      avg_aa_pos,
        'codon_pos':   avg_codon,
        'synonymous':  avg_syn,
        'strength':    loss_strength,
        'total':       total_loss,
        'num_targets': num_targets_in_batch,
    }
