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
    
    # 強度予測用のMSE損失
    mse_loss_fn = nn.MSELoss()

    # 戻り値: (inputs), y, lens, strains, strength_scores, in_strs, tgt_strs, full_paths
    for (x_cat, x_num, mask), y_batch_list, _, _, batch_strength_scores, _, _, _ in tqdm(dataloader, desc="Training"):
    
        x_cat = x_cat.to(config.DEVICE)
        x_num = x_num.to(config.DEVICE)
        mask = mask.to(config.DEVICE)

        # モデル出力: 4つの予測
        predictions_region, predictions_position, predictions_protein_pos, predictions_strength = model(
            x_cat, x_num, src_key_padding_mask=mask
        )
        
        loss_region_total = 0
        loss_position_total = 0
        loss_protein_pos_total = 0
        num_targets_in_batch = 0
        
        for i in range(len(y_batch_list)):
            pred_region_single = predictions_region[i]
            pred_position_single = predictions_position[i]
            pred_protein_pos_single = predictions_protein_pos[i]
            
            targets_tuples = y_batch_list[i]
            
            if not targets_tuples: continue
            
            # y_targets: list of (region_id, position_id, protein_pos_id)
            targets_region_list = [t[0] for t in targets_tuples]
            targets_position_list = [t[1] for t in targets_tuples]
            targets_protein_pos_list = [t[2] for t in targets_tuples]
            
            targets_region_tensor = torch.tensor(targets_region_list, dtype=torch.long).to(config.DEVICE)
            targets_position_tensor = torch.tensor(targets_position_list, dtype=torch.long).to(config.DEVICE)
            targets_protein_pos_tensor = torch.tensor(targets_protein_pos_list, dtype=torch.long).to(config.DEVICE)
            
            num_targets = len(targets_tuples)
            
            loss_r = loss_fn(pred_region_single.expand(num_targets, -1), targets_region_tensor)
            loss_p = loss_fn(pred_position_single.expand(num_targets, -1), targets_position_tensor)
            loss_protein_pos = loss_fn(pred_protein_pos_single.expand(num_targets, -1), targets_protein_pos_tensor)

            loss_region_total += loss_r.sum()
            loss_position_total += loss_p.sum()
            loss_protein_pos_total += loss_protein_pos.sum()

            num_targets_in_batch += num_targets

        if num_targets_in_batch > 0:
            optimizer.zero_grad()
            
            avg_loss_r = loss_region_total / num_targets_in_batch
            avg_loss_p = loss_position_total / num_targets_in_batch
            avg_loss_protein_pos = loss_protein_pos_total / num_targets_in_batch
            
            # 強度スコアのMSE損失 (バッチ全体)
            target_strength = torch.tensor(batch_strength_scores, dtype=torch.float).to(config.DEVICE)
            loss_strength = mse_loss_fn(predictions_strength, target_strength)
            
            if loss_wrapper is not None:
                # 自動重み付け (MultiTaskLoss) - 4タスク
                total_loss = loss_wrapper(avg_loss_r, avg_loss_p, avg_loss_protein_pos, loss_strength)
            else:
                # 固定重み付け (Config)
                total_loss = (config.LOSS_WEIGHT_REGION * avg_loss_r) + \
                             (config.LOSS_WEIGHT_POSITION * avg_loss_p) + \
                             (config.LOSS_WEIGHT_PROTEIN_POS * avg_loss_protein_pos) + \
                             (config.LOSS_WEIGHT_STRENGTH * loss_strength)
            
            total_loss.backward()
            optimizer.step()
            
            total_epoch_loss += total_loss.item()
            batches_processed += 1

    return total_epoch_loss / batches_processed if batches_processed > 0 else 0