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
    
    # 流行度予測用のMSE損失
    mse_loss_fn = nn.MSELoss()

    # 戻り値: (inputs), y, lens, strains, strength_scores, in_strs, tgt_strs, full_paths
    for (x_cat, x_num, mask), y_batch_list, _, _, batch_strength_scores, _, _, _ in tqdm(dataloader, desc="Training"):
    
        x_cat = x_cat.to(config.DEVICE)
        x_num = x_num.to(config.DEVICE)
        mask = mask.to(config.DEVICE)

        # モデル出力: 6つの予測
        predictions_region, predictions_position, predictions_aa_pos, predictions_strength, predictions_codon_pos, predictions_synonymous = model(
            x_cat, x_num, src_key_padding_mask=mask
        )
        
        loss_region_total = 0
        loss_position_total = 0
        loss_aa_pos_total = 0
        loss_codon_pos_total = 0
        loss_synonymous_total = 0
        num_targets_in_batch = 0
        
        for i in range(len(y_batch_list)):
            pred_region_single = predictions_region[i]
            pred_position_single = predictions_position[i]
            pred_aa_pos_single = predictions_aa_pos[i]
            pred_codon_pos_single = predictions_codon_pos[i]
            pred_synonymous_single = predictions_synonymous[i]
            
            targets_tuples = y_batch_list[i]
            
            if not targets_tuples: continue
            
            # y_targets: list of (region_id, position_id, aa_pos_id, codon_pos_id, is_synonymous)
            targets_region_list = [t[0] for t in targets_tuples]
            targets_position_list = [t[1] for t in targets_tuples]
            targets_aa_pos_list = [t[2] for t in targets_tuples]
            targets_codon_pos_list = [t[3] for t in targets_tuples]
            targets_synonymous_list = [t[4] for t in targets_tuples]
            
            targets_region_tensor = torch.tensor(targets_region_list, dtype=torch.long).to(config.DEVICE)
            targets_position_tensor = torch.tensor(targets_position_list, dtype=torch.long).to(config.DEVICE)
            targets_aa_pos_tensor = torch.tensor(targets_aa_pos_list, dtype=torch.long).to(config.DEVICE)
            targets_codon_pos_tensor = torch.tensor(targets_codon_pos_list, dtype=torch.long).to(config.DEVICE)
            targets_synonymous_tensor = torch.tensor(targets_synonymous_list, dtype=torch.long).to(config.DEVICE)
            
            num_targets = len(targets_tuples)
            
            loss_r = loss_fn(pred_region_single.expand(num_targets, -1), targets_region_tensor)
            loss_p = loss_fn(pred_position_single.expand(num_targets, -1), targets_position_tensor)
            loss_aa_pos = loss_fn(pred_aa_pos_single.expand(num_targets, -1), targets_aa_pos_tensor)
            loss_codon_pos = loss_fn(pred_codon_pos_single.expand(num_targets, -1), targets_codon_pos_tensor)
            loss_synonymous = loss_fn(pred_synonymous_single.expand(num_targets, -1), targets_synonymous_tensor)

            loss_region_total += loss_r.sum()
            loss_position_total += loss_p.sum()
            loss_aa_pos_total += loss_aa_pos.sum()
            loss_codon_pos_total += loss_codon_pos.sum()
            loss_synonymous_total += loss_synonymous.sum()

            num_targets_in_batch += num_targets

        if num_targets_in_batch > 0:
            optimizer.zero_grad()
            
            avg_loss_r = loss_region_total / num_targets_in_batch
            avg_loss_p = loss_position_total / num_targets_in_batch
            avg_loss_aa_pos = loss_aa_pos_total / num_targets_in_batch
            avg_loss_codon_pos = loss_codon_pos_total / num_targets_in_batch
            avg_loss_synonymous = loss_synonymous_total / num_targets_in_batch
            
            # 流行度のMSE損失 (バッチ全体)
            target_strength = torch.tensor(batch_strength_scores, dtype=torch.float).to(config.DEVICE)
            loss_strength = mse_loss_fn(predictions_strength, target_strength)
            
            if loss_wrapper is not None:
                # 自動重み付け (MultiTaskLoss) - 6タスク
                total_loss = loss_wrapper(avg_loss_r, avg_loss_p, avg_loss_aa_pos, loss_strength, avg_loss_codon_pos, avg_loss_synonymous)
            else:
                # 固定重み付け (Config)
                total_loss = (config.LOSS_WEIGHT_REGION * avg_loss_r) + \
                             (config.LOSS_WEIGHT_POSITION * avg_loss_p) + \
                             (config.LOSS_WEIGHT_AA_POS * avg_loss_aa_pos) + \
                             (config.LOSS_WEIGHT_CODON_POS * avg_loss_codon_pos) + \
                             (config.LOSS_WEIGHT_SYNONYMOUS * avg_loss_synonymous) + \
                             (config.LOSS_WEIGHT_STRENGTH * loss_strength)
            
            total_loss.backward()
            optimizer.step()
            
            total_epoch_loss += total_loss.item()
            batches_processed += 1

    return total_epoch_loss / batches_processed if batches_processed > 0 else 0