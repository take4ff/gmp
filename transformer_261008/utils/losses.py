# --- utils/losses.py ---
# 損失計算の共通ロジック (train.py / evaluate.py 共用)
# Soft Target（確率分配ベクトル）対応版
#   - FocalLoss を廃止し nn.CrossEntropyLoss(weight=class_weights) に統一
#   - compute_task_losses は Soft Target dict または Hard Target list のどちらも受け付ける
#   - expand() / sum() によるGradient Clash を排除した 1対1 のLoss計算

import torch
import torch.nn as nn
from . import logging as _log
from .. import config
from .bio_smooth import gaussian_smooth_target


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss (Khosla et al., 2020)

    同一 strain ラベルのサンプルをポジティブペアとして引き付け、
    異なる strain のサンプルを遠ざける対照損失。

    Args:
        temperature: ソフトマックスの温度パラメータ (デフォルト 0.07)

    Input:
        features: [B, D] — 正規化済み射影ベクトル
        labels:   [B]    — strain ラベル (整数値)

    Returns:
        スカラー損失
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        features: [B, D] normalized projection vectors
        labels:   [B] strain labels (LongTensor)
        """
        B = features.size(0)
        device = features.device

        # 特徴量を正規化 (L2 norm)
        features = nn.functional.normalize(features, dim=1)  # [B, D]

        # コサイン類似度行列: [B, B]
        sim_matrix = torch.matmul(features, features.T) / self.temperature

        # 同一ラベルペアのマスク: [B, B]（自己ペアを除外）
        labels = labels.view(-1)
        pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1))  # [B, B]
        diag_mask = torch.eye(B, dtype=torch.bool, device=device)
        pos_mask = pos_mask & ~diag_mask  # 対角を除いた正例マスク

        # ポジティブペアが存在しないサンプルはスキップ
        has_positive = pos_mask.any(dim=1)  # [B]
        if not has_positive.any():
            return torch.tensor(0.0, device=device, requires_grad=True)

        # 分母: 自己ペアを除いた全ペアの log-sum-exp
        sim_matrix_masked = sim_matrix.masked_fill(diag_mask, float('-inf'))
        log_denom = torch.logsumexp(sim_matrix_masked, dim=1)  # [B]

        # 各アンカーのポジティブペアに対する損失平均（ベクトル化）。
        # log_prob[i, j] = sim[i, j] - log_denom[i]
        # loss_i = -mean_{j: pos_mask[i,j]} log_prob[i, j]
        #        = -(Σ_j pos_mask[i,j] * log_prob[i,j]) / n_pos[i]
        # 最終損失 = ポジティブを持つアンカーにわたる平均（旧・逐次実装と数値一致）。
        log_prob = sim_matrix - log_denom.unsqueeze(1)              # [B, B]
        pos_mask_f = pos_mask.float()
        n_pos = pos_mask_f.sum(dim=1)                               # [B]
        loss_per_anchor = -(pos_mask_f * log_prob).sum(dim=1) / n_pos.clamp(min=1)  # [B]
        loss_valid = loss_per_anchor[has_positive]
        if loss_valid.numel() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        return loss_valid.mean()


def get_class_weights(counts, loss_type, beta=0.9999):
    """クラスの出現頻度から損失の重みを計算する。

    Args:
        counts: 各クラスの出現回数 (LongTensor)
        loss_type: 'ce', 'wce', 'cbce'
        beta: Class-Balanced用パラメータ

    Returns:
        FloatTensor: 正規化済みクラス重み
    """
    counts_float = counts.float()
    if loss_type == 'wce':
        # Inverse frequency: 1 / n_t
        weights = 1.0 / torch.clamp(counts_float, min=1.0)
    elif loss_type == 'cbce':
        # Class-Balanced: (1 - beta) / (1 - beta^n_t)
        weights = (1.0 - beta) / (1.0 - torch.pow(beta, torch.clamp(counts_float, min=1.0)))
        weights[counts == 0] = 0.0
    else:  # 'ce'
        weights = torch.ones_like(counts_float)

    if getattr(config, 'NORMALIZE_LOSS_WEIGHTS', False) and weights.sum() > 0:
        # [Fix] weights.mean() は0クラスを含む全体平均のため、大語彙タスク（position等）では
        # ほぼ全クラスが0件 → mean≈0 → weights/mean が inf/nan になる問題を修正。
        # 非ゼロクラスのみの mean で正規化することで数値安定性を確保する。
        non_zero_mask = weights > 0
        if non_zero_mask.any():
            weights = weights / weights[non_zero_mask].mean()

    return weights


def build_loss_fn(class_counts_dict=None):
    """設定に応じた分類損失関数を構築する。

    Args:
        class_counts_dict: {'region': Tensor, 'position': Tensor, ...} 各タスクのクラス頻度
    Returns:
        dict: タスクごとの損失関数 {'region': nn.Module, ...}
    """
    from ..model import FocalLoss

    loss_type = getattr(config, 'LOSS_FUNCTION_TYPE', 'ce').lower()
    label_smooth = config.LABEL_SMOOTHING_FACTOR if config.USE_LABEL_SMOOTHING else 0.0

    alpha = getattr(config, 'HYBRID_ALPHA', 1.0)
    mode_str = (
        'Soft(alpha=1.0)' if alpha == 1.0
        else ('Hard(alpha=0.0)' if alpha == 0.0 else f'Hybrid(alpha={alpha})')
    )
    _log.force_print(
        f"[INFO] Using Loss Function: {loss_type.upper()} "
        f"(label_smoothing={label_smooth}, mode={mode_str})"
    )

    tasks = ['region', 'position', 'aa_pos', 'codon_pos', 'synonymous']
    # USE_SUBSTITUTION_HEAD=True の場合は base_after / aa_after も追加
    if getattr(config, 'USE_SUBSTITUTION_HEAD', False):
        tasks = tasks + ['base_after', 'aa_after']
    loss_fns = {}

    for task in tasks:
        class_weight = None
        if class_counts_dict is not None and task in class_counts_dict:
            counts = class_counts_dict[task].to(config.DEVICE)
            beta = getattr(config, 'CLASS_BALANCED_BETA', 0.9999)
            class_weight = get_class_weights(counts, loss_type, beta=beta)

        if task == tasks[0]:
            _log.force_print(
                f"[INFO] Loss type={loss_type.upper()}, class_weighted={class_weight is not None}"
            )

        if loss_type in ['focal', 'cb_focal']:
            # ── Focal Loss（Soft Target アプローチA 対応版） ──────────────
            gamma = getattr(config, 'FOCAL_LOSS_GAMMA', 1.0)
            # cb_focal の場合は class_weight を alpha として使用
            # focal の場合は config.FOCAL_LOSS_ALPHA を使用 (None可)
            if loss_type == 'cb_focal':
                alpha = class_weight  # Class-Balanced重み
            else:
                alpha = getattr(config, 'FOCAL_LOSS_ALPHA', None)
                if alpha is None and class_weight is not None:
                    alpha = class_weight  # class_counts_dictがある場合はそちらを使用

            if task == tasks[0]:
                _log.force_print(
                    f"[INFO] FocalLoss: gamma={gamma}, alpha={'CB-weight' if loss_type=='cb_focal' else alpha}"
                )
            loss_fns[task] = FocalLoss(
                gamma=gamma,
                alpha=alpha,
                reduction='mean',       # Soft Target モードでは mean で直接スカラーを返す
                label_smoothing=label_smooth,  # Hard Target モードのみ有効
            )
        else:
            # ── CrossEntropyLoss ('ce', 'wce', 'cbce') ───────────────────
            # [260417] Soft Target では reduction='mean' かつ target を float ベクトルで渡す
            # label_smoothing と weight の同時使用は PyTorch 1.10+ でサポート
            if class_weight is not None:
                loss_fns[task] = nn.CrossEntropyLoss(
                    weight=class_weight,
                    reduction='mean',
                    label_smoothing=label_smooth,
                )
            else:
                loss_fns[task] = nn.CrossEntropyLoss(
                    reduction='mean',
                    label_smoothing=label_smooth,
                )

    return loss_fns



def compute_task_losses(
    predictions_tuple,
    y_batch,
    loss_fn,
    batch_strength_scores,
    raw_y_batch=None,
):
    """バッチ内の全サンプルに対して各タスクの損失を計算する。

    [260417] HYBRID_ALPHA 対応版。
        - HYBRID_ALPHA = 1.0: Soft Target のみ（y_batch が list[dict] の場合）
        - HYBRID_ALPHA = 0.0: Hard Target のみ（y_batch が list[list[tuple]] の場合）
        - 0 < HYBRID_ALPHA < 1: ハイブリッド
            loss = α × Soft_CE + (1-α) × Hard_CE
            raw_y_batch に Hard Target の tuples が必要

    Args:
        predictions_tuple: モデルが返す 6タプル
            (pred_region, pred_position, pred_aa_pos,
             pred_strength, pred_codon_pos, pred_synonymous)
        y_batch: list of (dict or list of tuples)
        loss_fn: dict of nn.CrossEntropyLoss (タスク名→損失関数)
        batch_strength_scores: list[float]
        raw_y_batch: list of list of tuples (Hard Target)。
                     ハイブリッドモード (0 < alpha < 1) で使用。
                     None の場合はハイブリッド計算をスキップ。

    Returns:
        dict with keys: 'region', 'position', 'aa_pos', 'codon_pos', 'synonymous',
                        'strength', 'total', 'num_targets'
        None: バッチ内に有効ターゲットがない場合
    """
    # 8-tuple への拡張に対応（base_after / aa_after は None の場合あり）
    (predictions_region, predictions_position, predictions_aa_pos,
     predictions_strength, predictions_codon_pos, predictions_synonymous,
     *_extra_preds) = predictions_tuple
    predictions_base_after = _extra_preds[0] if len(_extra_preds) > 0 else None
    predictions_aa_after   = _extra_preds[1] if len(_extra_preds) > 1 else None

    def get_lf(task_name):
        return loss_fn[task_name] if isinstance(loss_fn, dict) else loss_fn

    lf_reg = get_lf('region')
    lf_pos = get_lf('position')
    lf_aa  = get_lf('aa_pos')
    lf_cod = get_lf('codon_pos')
    lf_syn = get_lf('synonymous')

    # Soft Target モードか判定（最初の要素の型で判断）
    is_soft = (
        len(y_batch) > 0 and
        isinstance(y_batch[0], dict) and
        'position' in y_batch[0]
    )

    # HYBRID_ALPHA を取得（0.0=純粋Hard, 1.0=純粋Soft, 中間=ハイブリッド）
    alpha = getattr(config, 'HYBRID_ALPHA', 1.0)
    use_hybrid = is_soft and (0.0 < alpha < 1.0) and (raw_y_batch is not None)

    loss_region_total     = torch.tensor(0.0, device=config.DEVICE)
    loss_position_total   = torch.tensor(0.0, device=config.DEVICE)
    loss_aa_pos_total     = torch.tensor(0.0, device=config.DEVICE)
    loss_codon_pos_total  = torch.tensor(0.0, device=config.DEVICE)
    loss_synonymous_total = torch.tensor(0.0, device=config.DEVICE)
    num_valid_samples = 0
    # avg_* は各パスで確定させる（fast パスは直接 mean を代入、逐次パスは末尾で total/N）。
    avg_region = avg_position = avg_aa_pos = avg_codon = avg_syn = None

    _bio_on = getattr(config, 'USE_BIO_INFORMED_LOSS', False)
    # ─── 高速パス: 純粋 Soft Target（非ハイブリッド・非 Bio-smoothing）─────────
    # 旧実装はサンプルごとに unsqueeze(0) して極小 Loss を B×5 回呼んでいた
    # （B=512 で ~2,560 カーネル起動）。有効サンプルを stack して各タスク 1 回の
    # reduction='mean' 呼び出しに集約する。mean over Nv は旧実装の Σ_i/Nv と一致。
    if is_soft and not use_hybrid and not _bio_on:
        valid_idx = [i for i, st in enumerate(y_batch)
                     if isinstance(st, dict) and float(st['position'].sum()) > 0]
        num_valid_samples = len(valid_idx)
        if num_valid_samples > 0:
            idx_t = torch.tensor(valid_idx, dtype=torch.long, device=config.DEVICE)

            def _stack_soft(task):
                return torch.stack([y_batch[i][task] for i in valid_idx]).to(config.DEVICE)

            avg_region   = lf_reg(predictions_region.index_select(0, idx_t),    _stack_soft('region'))
            avg_position = lf_pos(predictions_position.index_select(0, idx_t),  _stack_soft('position'))
            avg_aa_pos   = lf_aa(predictions_aa_pos.index_select(0, idx_t),     _stack_soft('aa_pos'))
            avg_codon    = lf_cod(predictions_codon_pos.index_select(0, idx_t), _stack_soft('codon_pos'))
            avg_syn      = lf_syn(predictions_synonymous.index_select(0, idx_t),_stack_soft('synonymous'))

    elif is_soft:
        # ─── Soft Target モード / ハイブリッドモード（逐次: bio-smooth/hybrid 用）───
        # 各サンプルについて、Soft Target ベクトルのsum > 0 のもののみ計算
        for i, soft_target in enumerate(y_batch):
            # ターゲットが空（全ゼロ）のサンプルはスキップ
            if soft_target['position'].sum() == 0:
                continue

            # Soft Target を device に転送（1次元FloatTensor、sum=1.0）
            st_reg = soft_target['region'].to(config.DEVICE)
            st_pos = soft_target['position'].to(config.DEVICE)
            st_aa  = soft_target['aa_pos'].to(config.DEVICE)
            st_cod = soft_target['codon_pos'].to(config.DEVICE)
            st_syn = soft_target['synonymous'].to(config.DEVICE)

            # Biologically Informed Loss: 位置近傍にガウス分布でスムージング
            # USE_BIO_INFORMED_LOSS=True かつ Soft Target モードのみ有効
            if getattr(config, 'USE_BIO_INFORMED_LOSS', False):
                sigma  = getattr(config, 'BIO_INFORMED_SIGMA', 2.0)
                radius = getattr(config, 'BIO_INFORMED_RADIUS', 10)
                tasks  = getattr(config, 'BIO_INFORMED_TASKS', ['position', 'aa_pos'])
                if 'position' in tasks:
                    st_pos = gaussian_smooth_target(st_pos, sigma, radius)
                if 'aa_pos' in tasks:
                    st_aa = gaussian_smooth_target(st_aa, sigma, radius)
                if 'region' in tasks:
                    st_reg = gaussian_smooth_target(st_reg, sigma, radius)

            # 予測ベクトル（1次元）を unsqueeze して [1, C] にし、target も [1, C] に
            # → CrossEntropyLoss(reduction='mean') で 1対1 のスカラー損失を得る
            pred_reg_i = predictions_region[i].unsqueeze(0)    # [1, NUM_REGIONS]
            pred_pos_i = predictions_position[i].unsqueeze(0)  # [1, VOCAB_SIZE_POSITION]
            pred_aa_i  = predictions_aa_pos[i].unsqueeze(0)    # [1, VOCAB_SIZE_AA_POS]
            pred_cod_i = predictions_codon_pos[i].unsqueeze(0) # [1, VOCAB_SIZE_CODON_POS]
            pred_syn_i = predictions_synonymous[i].unsqueeze(0)# [1, VOCAB_SIZE_SYNONYMOUS]

            soft_reg = lf_reg(pred_reg_i, st_reg.unsqueeze(0))
            soft_pos = lf_pos(pred_pos_i, st_pos.unsqueeze(0))
            soft_aa  = lf_aa(pred_aa_i,   st_aa.unsqueeze(0))
            soft_cod = lf_cod(pred_cod_i, st_cod.unsqueeze(0))
            soft_syn = lf_syn(pred_syn_i, st_syn.unsqueeze(0))

            if use_hybrid:
                # ─── ハイブリッド: Hard CE も計算して alpha で混合 ──────────
                targets_tuples = raw_y_batch[i] if i < len(raw_y_batch) else []
                if targets_tuples:
                    num_t = len(targets_tuples)
                    t_reg = torch.tensor([t[0] for t in targets_tuples], dtype=torch.long, device=config.DEVICE)
                    t_pos = torch.tensor([t[1] for t in targets_tuples], dtype=torch.long, device=config.DEVICE)
                    t_aa  = torch.tensor([t[2] for t in targets_tuples], dtype=torch.long, device=config.DEVICE)
                    t_cod = torch.tensor([t[3] for t in targets_tuples], dtype=torch.long, device=config.DEVICE)
                    t_syn = torch.tensor([t[4] for t in targets_tuples], dtype=torch.long, device=config.DEVICE)

                    hard_reg = lf_reg(predictions_region[i].expand(num_t, -1),    t_reg).mean()
                    hard_pos = lf_pos(predictions_position[i].expand(num_t, -1),  t_pos).mean()
                    hard_aa  = lf_aa(predictions_aa_pos[i].expand(num_t, -1),     t_aa).mean()
                    hard_cod = lf_cod(predictions_codon_pos[i].expand(num_t, -1), t_cod).mean()
                    hard_syn = lf_syn(predictions_synonymous[i].expand(num_t, -1),t_syn).mean()

                    loss_region_total     += alpha * soft_reg + (1 - alpha) * hard_reg
                    loss_position_total   += alpha * soft_pos + (1 - alpha) * hard_pos
                    loss_aa_pos_total     += alpha * soft_aa  + (1 - alpha) * hard_aa
                    loss_codon_pos_total  += alpha * soft_cod + (1 - alpha) * hard_cod
                    loss_synonymous_total += alpha * soft_syn + (1 - alpha) * hard_syn
                else:
                    # Hard Target が空の場合は Soft のみ
                    loss_region_total     += soft_reg
                    loss_position_total   += soft_pos
                    loss_aa_pos_total     += soft_aa
                    loss_codon_pos_total  += soft_cod
                    loss_synonymous_total += soft_syn
            else:
                # ─── 純粋 Soft Target（alpha=1.0）─────────────────────────
                loss_region_total     += soft_reg
                loss_position_total   += soft_pos
                loss_aa_pos_total     += soft_aa
                loss_codon_pos_total  += soft_cod
                loss_synonymous_total += soft_syn

            num_valid_samples += 1

    else:
        # ─── 純粋 Hard Target モード（alpha=0.0, 後方互換） ──────────────────
        for i, targets_tuples in enumerate(y_batch):
            if not targets_tuples:
                continue

            num_targets = len(targets_tuples)
            targets_region   = torch.tensor([t[0] for t in targets_tuples], dtype=torch.long, device=config.DEVICE)
            targets_position = torch.tensor([t[1] for t in targets_tuples], dtype=torch.long, device=config.DEVICE)
            targets_aa_pos   = torch.tensor([t[2] for t in targets_tuples], dtype=torch.long, device=config.DEVICE)
            targets_codon    = torch.tensor([t[3] for t in targets_tuples], dtype=torch.long, device=config.DEVICE)
            targets_syn      = torch.tensor([t[4] for t in targets_tuples], dtype=torch.long, device=config.DEVICE)

            loss_region_total     += lf_reg(predictions_region[i].expand(num_targets, -1),    targets_region).mean()
            loss_position_total   += lf_pos(predictions_position[i].expand(num_targets, -1),  targets_position).mean()
            loss_aa_pos_total     += lf_aa(predictions_aa_pos[i].expand(num_targets, -1),     targets_aa_pos).mean()
            loss_codon_pos_total  += lf_cod(predictions_codon_pos[i].expand(num_targets, -1), targets_codon).mean()
            loss_synonymous_total += lf_syn(predictions_synonymous[i].expand(num_targets, -1),targets_syn).mean()
            num_valid_samples += 1

    if num_valid_samples == 0:
        return None

    # 逐次パス（hybrid/hard/bio-smooth）は total を N で割って mean を得る。
    # fast パスは既に avg_* を確定済みのためスキップする。
    if avg_region is None:
        avg_region   = loss_region_total   / num_valid_samples
        avg_position = loss_position_total / num_valid_samples
        avg_aa_pos   = loss_aa_pos_total   / num_valid_samples
        avg_codon    = loss_codon_pos_total / num_valid_samples
        avg_syn      = loss_synonymous_total / num_valid_samples

    # 流行度損失 (STRENGTH_LOSS_TYPE で切り替え可能)
    target_strength = torch.tensor(
        batch_strength_scores, dtype=torch.float, device=config.DEVICE
    )
    strength_loss_type = getattr(config, 'STRENGTH_LOSS_TYPE', 'mse').lower()
    if strength_loss_type == 'mae':
        loss_strength = torch.nn.functional.l1_loss(predictions_strength, target_strength)
    elif strength_loss_type == 'huber':
        delta = getattr(config, 'STRENGTH_HUBER_DELTA', 1.0)
        loss_strength = torch.nn.functional.huber_loss(
            predictions_strength, target_strength, delta=delta
        )
    else:  # 'mse' (デフォルト・従来動作)
        loss_strength = nn.MSELoss()(predictions_strength, target_strength)

    # --- Substitution Head 損失 (Item 5) ---
    # USE_SUBSTITUTION_HEAD=True かつ predictions_base_after/aa_after が存在する場合に計算
    use_sub = getattr(config, 'USE_SUBSTITUTION_HEAD', False)
    loss_base_after = torch.tensor(0.0, device=config.DEVICE)
    loss_aa_after   = torch.tensor(0.0, device=config.DEVICE)

    if use_sub and predictions_base_after is not None and predictions_aa_after is not None:
        lf_base_after = get_lf('base_after') if isinstance(loss_fn, dict) and 'base_after' in loss_fn \
                        else nn.CrossEntropyLoss(reduction='mean')
        lf_aa_after   = get_lf('aa_after') if isinstance(loss_fn, dict) and 'aa_after' in loss_fn \
                        else nn.CrossEntropyLoss(reduction='mean')

        loss_base_after_total = torch.tensor(0.0, device=config.DEVICE)
        loss_aa_after_total   = torch.tensor(0.0, device=config.DEVICE)
        n_sub = 0

        # ターゲットは 7 要素タプル: (..., base_after_token=t[5], aa_after_token=t[6])
        source = raw_y_batch if raw_y_batch is not None else y_batch
        if not is_soft:
            source = y_batch

        for i, targets_tuples in enumerate(source):
            if not targets_tuples:
                continue
            # tuples が dict (Soft Target) の場合はスキップ
            if isinstance(targets_tuples, dict):
                continue
            # 7 要素タプルかどうか確認
            sample_tuples = targets_tuples if not isinstance(targets_tuples, dict) else []
            valid_tuples = [t for t in sample_tuples if len(t) >= 7]
            if not valid_tuples:
                continue

            num_t = len(valid_tuples)
            t_base_after = torch.tensor([t[5] for t in valid_tuples], dtype=torch.long, device=config.DEVICE)
            t_aa_after   = torch.tensor([t[6] for t in valid_tuples], dtype=torch.long, device=config.DEVICE)

            loss_base_after_total += lf_base_after(
                predictions_base_after[i].expand(num_t, -1), t_base_after
            ).mean()
            loss_aa_after_total += lf_aa_after(
                predictions_aa_after[i].expand(num_t, -1), t_aa_after
            ).mean()
            n_sub += 1

        if n_sub > 0:
            loss_base_after = loss_base_after_total / n_sub
            loss_aa_after   = loss_aa_after_total   / n_sub

    # 加重合計損失
    total_loss = (
        config.LOSS_WEIGHT_REGION     * avg_region   +
        config.LOSS_WEIGHT_POSITION   * avg_position +
        config.LOSS_WEIGHT_AA_POS     * avg_aa_pos   +
        config.LOSS_WEIGHT_CODON_POS  * avg_codon    +
        config.LOSS_WEIGHT_SYNONYMOUS * avg_syn      +
        config.LOSS_WEIGHT_STRENGTH   * loss_strength
    )

    if use_sub:
        w_base = getattr(config, 'LOSS_WEIGHT_BASE_AFTER', 0.05)
        w_aa   = getattr(config, 'LOSS_WEIGHT_AA_AFTER',   0.05)
        total_loss = total_loss + w_base * loss_base_after + w_aa * loss_aa_after

    result = {
        'region':      avg_region,
        'position':    avg_position,
        'aa_pos':      avg_aa_pos,
        'codon_pos':   avg_codon,
        'synonymous':  avg_syn,
        'strength':    loss_strength,
        'total':       total_loss,
        'num_targets': num_valid_samples,
    }

    if use_sub:
        result['base_after'] = loss_base_after
        result['aa_after']   = loss_aa_after

    return result

