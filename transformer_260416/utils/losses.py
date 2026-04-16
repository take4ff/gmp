# --- utils/losses.py ---
# 損失計算の共通ロジック (train.py / evaluate.py 共用)

import torch
import torch.nn as nn
from . import logging as _log
from .. import config


def get_class_weights(counts, loss_type, beta=0.9999):
    """クラスの出現頻度から損失の重みを計算する"""
    counts_float = counts.float()
    if loss_type == 'wce':
        # 1 / n_t
        weights = 1.0 / torch.clamp(counts_float, min=1.0)
    elif loss_type in ['cbce', 'cb_focal']:
        # (1 - beta) / (1 - beta^n_t)
        weights = (1.0 - beta) / (1.0 - torch.pow(beta, torch.clamp(counts_float, min=1.0)))
        # counts == 0 の場合は重みを0にするなど安全策
        weights[counts == 0] = 0.0
    else:
        weights = torch.ones_like(counts_float)

    if getattr(config, 'NORMALIZE_LOSS_WEIGHTS', True) and weights.sum() > 0:
        weights = weights / weights.mean()

    return weights


def build_loss_fn(class_counts_dict=None):
    """設定に応じた分類損失関数を構築する。

    Args:
        class_counts_dict: {'region': ..., 'position': ...} 各タスクのクラス頻度テンソル
    Returns:
        dict: タスクごとの損失関数 {'region': nn.Module, ...}、または単一のModule
    """
    from ..model import FocalLoss

    loss_type = getattr(config, 'LOSS_FUNCTION_TYPE', 'ce').lower()
    label_smooth = config.LABEL_SMOOTHING_FACTOR if config.USE_LABEL_SMOOTHING else 0.0

    _log.force_print(f"[INFO] Using Loss Function: {loss_type.upper()} "
                     f"(label_smoothing={label_smooth})")

    tasks = ['region', 'position', 'aa_pos', 'codon_pos', 'synonymous']
    loss_fns = {}

    for task in tasks:
        alpha = None
        if class_counts_dict is not None and task in class_counts_dict:
            counts = class_counts_dict[task].to(config.DEVICE)
            if loss_type in ['wce', 'cbce', 'cb_focal']:
                beta = getattr(config, 'CLASS_BALANCED_BETA', 0.9999)
                alpha = get_class_weights(counts, loss_type, beta=beta)
        
        if loss_type == 'focal' and alpha is None:
            alpha = getattr(config, 'FOCAL_LOSS_ALPHA', None)
        
        if loss_type in ['focal', 'cb_focal']:
            gamma = getattr(config, 'FOCAL_LOSS_GAMMA', 1.0)
            if task == tasks[0]: # 1回だけprint
                _log.force_print(f"[INFO] Focal Loss parameters: gamma={gamma}, "
                                 f"weighted={alpha is not None}")
            loss_fns[task] = FocalLoss(
                gamma=gamma,
                alpha=alpha,
                reduction='none',
                label_smoothing=label_smooth,
            )
        else: # 'ce', 'wce', 'cbce'
            if task == tasks[0]:
                _log.force_print(f"[INFO] Cross Entropy based loss, "
                                 f"weighted={alpha is not None}")
            if alpha is not None:
                # 組み込みのCrossEntropyLossは weight parameter を受け取るが、
                # reduction='none', label_smoothingありと組み合わせる。
                loss_fns[task] = nn.CrossEntropyLoss(
                    weight=alpha,
                    reduction='none',
                    label_smoothing=label_smooth,
                )
            else:
                loss_fns[task] = nn.CrossEntropyLoss(
                    reduction='none',
                    label_smoothing=label_smooth,
                )
    
    return loss_fns


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
        loss_fn: 分類損失関数 (FocalLoss or CrossEntropyLoss またはそれらのdict)
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

    def get_lf(task_name):
        return loss_fn[task_name] if isinstance(loss_fn, dict) else loss_fn

    lf_reg = get_lf('region')
    lf_pos = get_lf('position')
    lf_aa = get_lf('aa_pos')
    lf_cod = get_lf('codon_pos')
    lf_syn = get_lf('synonymous')

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

        loss_region_total     += lf_reg(predictions_region[i].expand(num_targets, -1),   targets_region).sum()
        loss_position_total   += lf_pos(predictions_position[i].expand(num_targets, -1), targets_position).sum()
        loss_aa_pos_total     += lf_aa(predictions_aa_pos[i].expand(num_targets, -1),    targets_aa_pos).sum()
        loss_codon_pos_total  += lf_cod(predictions_codon_pos[i].expand(num_targets, -1), targets_codon).sum()
        loss_synonymous_total += lf_syn(predictions_synonymous[i].expand(num_targets, -1),targets_syn).sum()
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
