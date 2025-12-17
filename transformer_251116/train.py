import torch
from tqdm import tqdm
from . import config
from torch.amp import autocast, GradScaler

def train_one_epoch(model, dataloader, optimizer, loss_fn, scaler):
    """1エポック分の訓練を行う"""
    model.train()
    total_epoch_loss = 0.0

    for batch in tqdm(dataloader, desc="Training"):
        # データをデバイスに転送
        x_cat = batch['x_cat'].to(config.DEVICE)
        x_num = batch['x_num'].to(config.DEVICE)
        x_cat_mask = batch['x_cat_mask'].to(config.DEVICE)
        seq_mask = batch['seq_mask'].to(config.DEVICE)
        y_protein = batch['y_protein'].to(config.DEVICE)
        y_pos = batch['y_pos'].to(config.DEVICE)

        optimizer.zero_grad()

        # 混合精度計算 (AMP)
        with autocast(device_type=config.DEVICE):
            # 順伝播
            protein_logits, pos_logits = model(x_cat, x_num, x_cat_mask, seq_mask)
            
            # 損失計算 (ベクトル化)
            loss_protein = loss_fn(protein_logits, y_protein)
            loss_pos = loss_fn(pos_logits, y_pos)
            
            # 重み付けして合計
            total_loss = (config.LOSS_WEIGHT_REGION * loss_protein) + \
                         (config.LOSS_WEIGHT_POSITION * loss_pos)

        # 勾配のスケーリングと逆伝播
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_epoch_loss += total_loss.item()

    return total_epoch_loss / len(dataloader)