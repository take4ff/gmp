# --- scripts/pretrain.py ---
"""MLM / CLM 事前学習スクリプト (Item 3)

使い方:
    python -m transformer_260707.scripts.train.pretrain

config.USE_PRETRAINING=True かつ本スクリプトを実行することで
HierarchicalTransformer の事前学習を行い、チェックポイントを保存する。

事前学習モード:
    PRETRAINING_MODE='mlm' : Masked Language Model
        - タイムステップの cat/num 特徴量を PRETRAINING_MASK_RATIO の確率でゼロマスク
        - マスクされたタイムステップの region / position を再予測
    PRETRAINING_MODE='clm' : Causal Language Model
        - 各タイムステップで次のタイムステップの region / position を予測
        - src_mask（因果マスク）を Transformer Encoder に適用

出力: OUTPUT_DIR/pretrain/pretrain_<mode>_epoch<N>.pth
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datetime import datetime
from tqdm import tqdm

from ... import config
from ...model import HierarchicalTransformer
from ...utils.logging import force_print


# ─────────────────────────────────────────────────────────────
# 1. PretrainingModel ラッパー
# ─────────────────────────────────────────────────────────────

class PretrainingModel(nn.Module):
    """HierarchicalTransformer に MLM/CLM 用の補助ヘッドを追加したラッパー。

    補助ヘッドは region と position の 2 タスクのみ（最も情報量が大きいため）。
    事前学習完了後は backbone（HierarchicalTransformer）の重みのみを保存し、
    main.py で fine-tuning に使用する。
    """

    def __init__(self, backbone: HierarchicalTransformer):
        super().__init__()
        self.backbone = backbone
        feature_dim = config.FEATURE_DIM

        # MLM/CLM 補助ヘッド: region + position を直接 latest_context から予測する代わりに
        # Transformer Encoder 出力全体 [B, T, F] を使って各タイムステップを予測する
        # backbone の各予測ヘッドを再利用
        self.mlm_region_head   = nn.Linear(feature_dim, config.NUM_REGIONS)
        self.mlm_position_head = nn.Linear(feature_dim, config.VOCAB_SIZE_POSITION)

    def forward(self, x_cat, x_num, src_mask=None, src_key_padding_mask=None):
        """Transformer Encoder の全タイムステップ出力を返す（MLM/CLM 用）。

        Returns:
            (enc_out, [B, T, F]): Encoder の全タイムステップ出力
            (logits_region, [B, T, NUM_REGIONS]): 各タイムステップの region ロジット
            (logits_position, [B, T, VOCAB_SIZE_POSITION]): 各タイムステップの position ロジット
        """
        b = self.backbone

        # InputEmbedding + CoOccurrence Attention
        x = b.input_embed(x_cat, x_num)             # [B, T, C, F]
        x_agg = b.co_attn(x)                         # [B, T, F]

        # Local Conv1d (任意)
        if b.use_local_conv and b.local_feature_extractor is not None:
            x_combined = x_agg + b.local_feature_extractor(x_agg)
        else:
            x_combined = x_agg

        # Origin Attention (任意)
        if b.use_origin_attention and b.origin_attn is not None:
            B_ = x_combined.size(0)
            origin_emb = b.origin_embedding.expand(B_, -1, -1)
            x_combined = x_combined + b.origin_attn(x_seq=x_combined, x_origin=origin_emb)

        # Positional Encoding + Transformer Encoder
        if b.alibi is not None:
            B_, T_ = x_combined.shape[:2]
            x = b.pos_encoder.dropout(x_combined)
            alibi_mask = b.alibi(T_, B_, config.N_HEADS, x.device)
            effective_mask = alibi_mask if src_mask is None else src_mask + alibi_mask
        else:
            x = b.pos_encoder(x_combined)
            effective_mask = src_mask

        enc_out = b.transformer_encoder(
            x,
            mask=effective_mask,
            src_key_padding_mask=src_key_padding_mask,
        )  # [B, T, F]

        # 全タイムステップに MLM 補助ヘッドを適用
        logits_region   = self.mlm_region_head(enc_out)    # [B, T, NUM_REGIONS]
        logits_position = self.mlm_position_head(enc_out)  # [B, T, VOCAB_SIZE_POSITION]

        return enc_out, logits_region, logits_position


# ─────────────────────────────────────────────────────────────
# 2. MLM マスキング
# ─────────────────────────────────────────────────────────────

def apply_mlm_mask(x_cat, x_num, padding_mask, mask_ratio, mask_mode='random', span_len=3):
    """タイムステップ単位で cat/num 特徴量をゼロマスクし、マスク位置のインデックスを返す。

    padding_mask: [B, T] — True = PAD (マスク対象外)
    mask_mode: 'random'=各タイムステップ独立、'span'=連続 span_len タイムステップをまとめて
               マスク（SpanBERT 風）。span の期待マスク率が mask_ratio に一致するよう、
               スパン開始確率を mask_ratio/span_len に設定する。
    Returns:
        x_cat_masked, x_num_masked: マスク適用後テンソル
        mlm_mask:      [B, T] bool  — True = MLM マスクされたタイムステップ
    """
    B, T = padding_mask.shape
    device = x_cat.device

    # PAD 以外の有効タイムステップのみをマスク対象とする
    valid_mask = ~padding_mask  # [B, T] True = 有効

    if mask_mode == 'span' and span_len > 1:
        # スパン開始をサンプリングし、右方向へ span_len 分だけ塗り広げる。
        # 開始確率 = mask_ratio / span_len で総マスク率が概ね mask_ratio になる。
        start_prob = mask_ratio / float(span_len)
        starts = torch.rand(B, T, device=device) < start_prob  # [B, T]
        mlm_mask = torch.zeros_like(valid_mask)
        for off in range(span_len):
            if off == 0:
                mlm_mask |= starts
            else:
                shifted = torch.zeros_like(starts)
                shifted[:, off:] = starts[:, :T - off]
                mlm_mask |= shifted
        mlm_mask &= valid_mask
    else:
        # 各タイムステップを mask_ratio の確率でマスク (PAD は除外)
        rand = torch.rand(B, T, device=device)
        mlm_mask = (rand < mask_ratio) & valid_mask  # [B, T]

    x_cat_masked = x_cat.clone()
    x_num_masked = x_num.clone()

    # マスク位置の cat/num を 0 に
    mlm_mask_4d_cat = mlm_mask.unsqueeze(-1).unsqueeze(-1)  # [B, T, 1, 1]
    mlm_mask_4d_num = mlm_mask.unsqueeze(-1).unsqueeze(-1)
    x_cat_masked = x_cat_masked.masked_fill(mlm_mask_4d_cat.expand_as(x_cat), 0)
    x_num_masked = x_num_masked.masked_fill(mlm_mask_4d_num.expand_as(x_num), 0.0)

    return x_cat_masked, x_num_masked, mlm_mask


# ─────────────────────────────────────────────────────────────
# 3. 因果マスク (CLM 用)
# ─────────────────────────────────────────────────────────────

def make_causal_mask(T, device):
    """上三角を -inf にした因果マスク [T, T] を返す。"""
    mask = torch.triu(torch.full((T, T), float('-inf'), device=device), diagonal=1)
    return mask


# ─────────────────────────────────────────────────────────────
# 4. 事前学習ループ
# ─────────────────────────────────────────────────────────────

def run_pretraining():
    """メイン事前学習ループ。"""
    if not getattr(config, 'USE_PRETRAINING', False):
        force_print("[INFO] USE_PRETRAINING=False — スキップします。")
        return

    mode           = getattr(config, 'PRETRAINING_MODE', 'mlm').lower()
    mask_ratio     = getattr(config, 'PRETRAINING_MASK_RATIO', 0.15)
    mask_mode      = getattr(config, 'PRETRAINING_MASK_MODE', 'random').lower()
    span_len       = int(getattr(config, 'PRETRAINING_SPAN_LEN', 3))
    epochs         = getattr(config, 'PRETRAINING_EPOCHS', 5)
    lr             = getattr(config, 'PRETRAINING_LR', 1e-4)
    warmup_epochs  = getattr(config, 'PRETRAINING_WARMUP_EPOCHS', 1)
    eta_min        = getattr(config, 'PRETRAINING_ETA_MIN', 1e-6)

    force_print(
        f"[INFO] 事前学習開始: mode={mode}, mask_ratio={mask_ratio}, "
        f"mask_mode={mask_mode}" + (f"(span_len={span_len})" if mask_mode == 'span' else "") +
        f", epochs={epochs}, lr={lr}, warmup={warmup_epochs}, eta_min={eta_min}"
    )

    # データローダー構築
    from ...db.connection import check_db_exists, get_db_path
    from ...db.dataset import create_db_dataloader

    db_exists, msg = check_db_exists()
    if not db_exists:
        raise RuntimeError(f"Database not found: {msg}")
    db_path = get_db_path()

    # walk_forward モードでは run_walk_forward() が事前にフォールドの splits を
    # assign_wf_splits() で設定してから run_pretraining() を呼ぶため、
    # get_split_col() 経由で正しい split_type_wf が参照される。
    # 他のモードでは split_type（timestep 基準）が使われる。
    train_loader = create_db_dataloader(
        db_path=db_path,
        split_type=0,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        max_cooccurrence=config.MAX_CO_OCCURRENCE,
    )

    # モデル構築
    backbone = HierarchicalTransformer().to(config.DEVICE)
    pt_model = PretrainingModel(backbone).to(config.DEVICE)

    optimizer = torch.optim.AdamW(
        pt_model.parameters(), lr=lr, weight_decay=config.WEIGHT_DECAY
    )

    # LinearWarmup + CosineAnnealing（main.py と同じパターン）
    if warmup_epochs > 0 and epochs > warmup_epochs:
        warmup_sched = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0 / max(warmup_epochs, 1),
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        cosine_sched = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(epochs - warmup_epochs, 1),
            eta_min=eta_min,
        )
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_sched, cosine_sched],
            milestones=[warmup_epochs],
        )
        force_print(f"[INFO] Scheduler: LinearWarmup({warmup_epochs} epochs) + CosineAnnealingLR(eta_min={eta_min})")
    elif epochs > 0:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=eta_min)
        force_print(f"[INFO] Scheduler: CosineAnnealingLR(eta_min={eta_min})")
    else:
        scheduler = None

    ce_loss = nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')

    # 出力ディレクトリ（SPLIT_MODE を含めて区別）
    split_mode = getattr(config, 'SPLIT_MODE', 'timestep')
    pretrain_dir = os.path.join(config.OUTPUT_DIR, 'pretrain', split_mode)
    os.makedirs(pretrain_dir, exist_ok=True)

    for epoch in range(epochs):
        pt_model.train()
        total_loss = 0.0
        n_batches  = 0
        t_start    = time.time()

        pbar = tqdm(train_loader, desc=f"Pretrain Epoch {epoch+1}/{epochs}", dynamic_ncols=True)
        for (x_cat, x_num, mask), y_batch, *_ in pbar:
            x_cat = x_cat.to(config.DEVICE)
            x_num = x_num.to(config.DEVICE)
            mask  = mask.to(config.DEVICE)  # [B, T] True=PAD

            if mode == 'mlm':
                # ── MLM ──────────────────────────────────────────────────
                x_cat_in, x_num_in, mlm_mask = apply_mlm_mask(
                    x_cat, x_num, mask, mask_ratio,
                    mask_mode=mask_mode, span_len=span_len,
                )

                _, logits_region, logits_position = pt_model(
                    x_cat_in, x_num_in, src_key_padding_mask=mask
                )  # logits: [B, T, C]

                # マスクされた位置のみで損失計算
                # target: x_cat の region (index 7) と position (index 1) の最初の共起変異
                B, T = mlm_mask.shape
                # x_cat shape: [B, T, MAX_CO_OCCURRENCE, 15]
                target_region   = x_cat[:, :, 0, 7].long()    # [B, T]
                target_position = x_cat[:, :, 0, 1].long()    # [B, T]

                # MLM マスク外は ignore_index=-1 に
                target_region   = target_region.masked_fill(~mlm_mask, -1)
                target_position = target_position.masked_fill(~mlm_mask, -1)

                # [B, T, C] → [B*T, C]
                loss_reg = ce_loss(
                    logits_region.reshape(-1, config.NUM_REGIONS),
                    target_region.reshape(-1)
                )
                loss_pos = ce_loss(
                    logits_position.reshape(-1, config.VOCAB_SIZE_POSITION),
                    target_position.reshape(-1)
                )
                batch_loss = loss_reg + loss_pos

            else:
                # ── CLM ──────────────────────────────────────────────────
                T_seq = x_cat.size(1)
                causal_mask = make_causal_mask(T_seq, config.DEVICE)

                _, logits_region, logits_position = pt_model(
                    x_cat, x_num, src_mask=causal_mask, src_key_padding_mask=mask
                )

                # CLM: 各タイムステップ t が t+1 を予測
                # ロジット: logits[:, :-1, :]  ターゲット: x_cat[:, 1:, 0, :]
                if T_seq < 2:
                    continue

                pred_reg = logits_region[:, :-1, :].reshape(-1, config.NUM_REGIONS)
                pred_pos = logits_position[:, :-1, :].reshape(-1, config.VOCAB_SIZE_POSITION)

                tgt_reg = x_cat[:, 1:, 0, 7].long().reshape(-1)
                tgt_pos = x_cat[:, 1:, 0, 1].long().reshape(-1)

                # PAD タイムステップの予測は無視
                valid_flat = ~mask[:, 1:].reshape(-1)
                tgt_reg  = tgt_reg.masked_fill(~valid_flat, -1)
                tgt_pos  = tgt_pos.masked_fill(~valid_flat, -1)

                loss_reg = ce_loss(pred_reg, tgt_reg)
                loss_pos = ce_loss(pred_pos, tgt_pos)
                batch_loss = loss_reg + loss_pos

            optimizer.zero_grad()
            batch_loss.backward()
            if getattr(config, 'MAX_GRAD_NORM', None) is not None:
                torch.nn.utils.clip_grad_norm_(pt_model.parameters(), config.MAX_GRAD_NORM)
            optimizer.step()

            total_loss += batch_loss.item()
            n_batches  += 1
            pbar.set_postfix(loss=f"{total_loss / n_batches:.4f}")

        elapsed = time.time() - t_start
        avg_loss = total_loss / max(n_batches, 1)
        current_lr = optimizer.param_groups[0]['lr']
        force_print(f"[Pretrain] Epoch {epoch+1}/{epochs} — loss={avg_loss:.4f}, lr={current_lr:.2e} ({elapsed:.1f}s)")

        if scheduler is not None:
            scheduler.step()

        # チェックポイント保存
        ckpt_path = os.path.join(
            pretrain_dir,
            f"pretrain_{mode}_epoch{epoch+1}.pth"
        )
        torch.save({
            'epoch': epoch + 1,
            'backbone_state_dict': backbone.state_dict(),
            'pt_model_state_dict': pt_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'loss': avg_loss,
            'mode': mode,
        }, ckpt_path)
        force_print(f"[Pretrain] Checkpoint saved: {ckpt_path}")

    force_print("[Pretrain] 事前学習完了。")
    final_path = os.path.join(pretrain_dir, f"pretrain_{mode}_final.pth")
    torch.save({'backbone_state_dict': backbone.state_dict(), 'mode': mode}, final_path)
    force_print(f"[Pretrain] 最終 backbone 重み保存: {final_path}")


if __name__ == '__main__':
    run_pretraining()
