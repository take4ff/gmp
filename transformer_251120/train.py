# --- train.py ---
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from . import config

def train_one_epoch(model, dataloader, optimizer, loss_fn, loss_wrapper=None):
    model.train()
    total_epoch_loss = 0
    batches_processed = 0

    # (x,y以外のメタデータは訓練ループ内では使わないため _ で無視する)
    # 戻り値: (inputs), y, lens, strains, in_strs, tgt_strs, full_paths (計7個)
    for (x_cat, x_num, mask), y_batch_list, _, _, _, _, _ in tqdm(dataloader, desc="Training"):
    
        x_cat = x_cat.to(config.DEVICE)
        x_num = x_num.to(config.DEVICE)
        mask = mask.to(config.DEVICE)

        predictions_region, predictions_position = model(x_cat, x_num, src_key_padding_mask=mask)
        
        loss_region_total = 0
        loss_position_total = 0
        num_targets_in_batch = 0
        
        for i in range(len(y_batch_list)):
            pred_region_single = predictions_region[i]
            pred_position_single = predictions_position[i]
            
            targets_tuples = y_batch_list[i]
            
            if not targets_tuples: continue
            
            targets_region_list = [t[0] for t in targets_tuples]
            targets_position_list = [t[1] for t in targets_tuples]
            
            targets_region_tensor = torch.tensor(targets_region_list, dtype=torch.long).to(config.DEVICE)
            targets_position_tensor = torch.tensor(targets_position_list, dtype=torch.long).to(config.DEVICE)
            
            num_targets = len(targets_tuples)
            
            loss_r = loss_fn(pred_region_single.expand(num_targets, -1), targets_region_tensor)
            loss_p = loss_fn(pred_position_single.expand(num_targets, -1), targets_position_tensor)

            loss_region_total += loss_r.sum()
            loss_position_total += loss_p.sum()

            num_targets_in_batch += num_targets

        if num_targets_in_batch > 0:
            optimizer.zero_grad()
            
            avg_loss_r = loss_region_total / num_targets_in_batch
            avg_loss_p = loss_position_total / num_targets_in_batch
            
            if loss_wrapper is not None:
                # 自動重み付け (MultiTaskLoss)
                total_loss = loss_wrapper(avg_loss_r, avg_loss_p)
            else:
                # 固定重み付け (Config)
                total_loss = (config.LOSS_WEIGHT_REGION * avg_loss_r) + \
                             (config.LOSS_WEIGHT_POSITION * avg_loss_p)
            
            total_loss.backward()
            optimizer.step()
            
            total_epoch_loss += total_loss.item()
            batches_processed += 1

    return total_epoch_loss / batches_processed if batches_processed > 0 else 0