# --- train.py ---
import torch
from tqdm import tqdm
from . import config
from .utils.losses import compute_task_losses


def _log_rss_train(tag: str):
    """[調査用 2026-07-31] train_one_epoch()バッチループ中のRSS推移を追跡し、
    fold_4学習中クラッシュ(persistent train workerでの共有group_label_cacheの
    COW+refcountタッチによる緩やかな重複増加の疑い)を切り分けるための一時計測。"""
    try:
        with open('/proc/self/status') as f:
            vmrss_kb = None
            for line in f:
                if line.startswith('VmRSS:'):
                    vmrss_kb = int(line.split()[1])
                    break
        cur_gb = (vmrss_kb / 1024 / 1024) if vmrss_kb is not None else float('nan')
        from .utils.logging import force_print
        force_print(f"[MEM] {tag}: current_rss={cur_gb:.2f} GiB")
    except Exception as e:
        from .utils.logging import force_print
        force_print(f"[MEM] {tag}: (failed to read RSS: {e})")


def train_one_epoch(model, dataloader, optimizer, loss_fn, loss_wrapper=None):
    """1エポック分の学習を実行する。

    Args:
        model: 学習対象モデル
        dataloader: 学習用DataLoader
        optimizer: 最適化アルゴリズム (AdamW等)
        loss_fn: 分類損失関数 (dict of nn.CrossEntropyLoss)
        loss_wrapper: MultiTaskLoss インスタンス。Noneの場合は固定重みを使用

    Returns:
        float: エポック平均損失
    """
    model.train()
    total_epoch_loss = 0
    batches_processed = 0

    # SupCon 設定
    use_supcon = getattr(config, 'USE_SUPCON', False)
    supcon_weight = getattr(config, 'SUPCON_WEIGHT', 0.1)
    if use_supcon:
        from .utils.losses import SupConLoss
        supcon_temperature = getattr(config, 'SUPCON_TEMPERATURE', 0.07)
        supcon_loss_fn = SupConLoss(temperature=supcon_temperature)

    # collate_fn の戻り値に raw_y が追加
    # Soft Target モード: y_batch = list of dict (Soft Target ベクトル)
    # Hard Target モード: y_batch = list of list of tuples
    for _train_batch_idx, ((x_cat, x_num, mask), y_batch, _, batch_strains, batch_strength_scores, _, _, _, _raw_y, *extra) in enumerate(tqdm(dataloader, desc="Training")):
        if _train_batch_idx == 0 or _train_batch_idx % 300 == 0:
            _log_rss_train(f"train_one_epoch() batch {_train_batch_idx}")

        x_cat = x_cat.to(config.DEVICE)
        x_num = x_num.to(config.DEVICE)
        mask  = mask.to(config.DEVICE)

        # 提案4: 主要系統クレード埋め込み（OFF 時は None → モデル側で無視）
        clade_ids = None
        if getattr(config, 'USE_CLADE_EMBEDDING', False):
            from .utils.clade import clade_ids_tensor
            clade_ids = clade_ids_tensor(batch_strains, device=config.DEVICE)

        predictions_tuple = model(x_cat, x_num, src_key_padding_mask=mask, clade_ids=clade_ids)

        losses = compute_task_losses(
            predictions_tuple, y_batch, loss_fn, batch_strength_scores,
            raw_y_batch=_raw_y,
        )

        if losses is None:
            continue

        optimizer.zero_grad()

        if loss_wrapper is not None:
            # 自動重み付け (MultiTaskLoss) — 引数の順序は MultiTaskLoss の log_vars 添字と対応
            # 順序: region[0], position[1], aa_pos[2], strength[3], codon_pos[4], synonymous[5]
            total_loss = loss_wrapper(
                losses['region'],    # [0]
                losses['position'],  # [1]
                losses['aa_pos'],    # [2]
                losses['strength'],  # [3]
                losses['codon_pos'], # [4]
                losses['synonymous'],# [5]
            )
            # MultiTaskLoss は6タスク固定のため substitution head 損失を別途加算
            if getattr(config, 'USE_SUBSTITUTION_HEAD', False):
                w_base = getattr(config, 'LOSS_WEIGHT_BASE_AFTER', 0.05)
                w_aa   = getattr(config, 'LOSS_WEIGHT_AA_AFTER',   0.05)
                total_loss = total_loss + w_base * losses['base_after'] + w_aa * losses['aa_after']
        else:
            total_loss = losses['total']

        # --- Item 12: Supervised Contrastive Loss ---
        # USE_SUPCON=True かつ モデルが射影ベクトルを持つ場合に追加
        if use_supcon and getattr(model, '_last_projections', None) is not None:
            projections = model._last_projections  # [B, proj_dim]
            # strain 名を整数ハッシュ化してラベルとして使用
            strain_labels = torch.tensor(
                [hash(s) % (2 ** 31) for s in batch_strains],
                dtype=torch.long,
                device=config.DEVICE,
            )
            sc_loss = supcon_loss_fn(projections, strain_labels)
            total_loss = total_loss + supcon_weight * sc_loss

        # --- 提案8: Mixture ヘッドの負荷分散正則化 ---
        mix_aux = getattr(model, '_last_mixture_aux', None)
        if mix_aux is not None:
            total_loss = total_loss + mix_aux

        # --- 提案9: Abstention（棄却）ヘッドの回帰損失 ---
        # ターゲット = log(1 + サンプルのターゲット数)。raw_y / soft dict / hard list から算出。
        unc = getattr(model, '_last_uncertainty', None)
        if unc is not None:
            import math as _math
            def _cardinality(i):
                if _raw_y is not None and i < len(_raw_y) and _raw_y[i]:
                    return len(_raw_y[i])
                yi = y_batch[i]
                if isinstance(yi, dict):
                    return int((yi['position'] > 0).sum().item())
                return len(yi) if yi else 0
            targets_unc = torch.tensor(
                [_math.log1p(_cardinality(i)) for i in range(unc.size(0))],
                dtype=torch.float, device=config.DEVICE,
            )
            w_abs = getattr(config, 'LOSS_WEIGHT_ABSTENTION', 0.05)
            total_loss = total_loss + w_abs * torch.nn.functional.mse_loss(unc, targets_unc)

        # --- 提案11: 枝長補助損失（枝長ターゲットがバッチに存在する場合のみ）---
        bl_pred = getattr(model, '_last_branch_length', None)
        batch_branch_len = None
        if extra:
            # collate_fn が将来 branch_length を返す場合、extra の先頭に入る想定
            batch_branch_len = extra[0]
        if bl_pred is not None and batch_branch_len is not None:
            target_bl = torch.as_tensor(batch_branch_len, dtype=torch.float, device=config.DEVICE)
            if target_bl.numel() == bl_pred.numel():
                w_bl = getattr(config, 'LOSS_WEIGHT_BRANCH_LENGTH', 0.02)
                total_loss = total_loss + w_bl * torch.nn.functional.mse_loss(bl_pred, target_bl.view_as(bl_pred))

        total_loss.backward()

        max_norm = getattr(config, "MAX_GRAD_NORM", None)
        if max_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        optimizer.step()

        total_epoch_loss += total_loss.item()
        batches_processed += 1

    return total_epoch_loss / batches_processed if batches_processed > 0 else 0
