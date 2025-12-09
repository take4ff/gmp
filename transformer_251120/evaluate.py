# --- evaluate.py ---
import torch
from collections import defaultdict
from .utils import calculate_metrics
from . import config

def evaluate(model, dataloader, loss_fn):
    model.eval()
    total_epoch_loss = 0
    batches_processed = 0
    
    # タイムステップ別の集計用
    results_by_timestep = defaultdict(lambda: {
        "preds_region": [], "targets_region": [],
        "preds_position": [], "targets_position": []
    })

    # 保存用の詳細結果リスト
    detailed_results = []

    example_printed = False

    with torch.no_grad():
        # dataset.py の collate_fn の戻り値に合わせて受け取る
        for (x_cat, x_num, mask), y_batch_list, batch_lens, batch_strains, batch_input_strs, batch_target_strs, batch_full_paths in dataloader:
        
            x_cat = x_cat.to(config.DEVICE)
            x_num = x_num.to(config.DEVICE)
            mask = mask.to(config.DEVICE)

            predictions_region, predictions_position = model(x_cat, x_num, src_key_padding_mask=mask)
            
            topk_indices_region = torch.topk(predictions_region, config.TOP_K_EVAL, dim=1).indices
            topk_indices_position = torch.topk(predictions_position, config.TOP_K_EVAL, dim=1).indices
            
            loss_region_total = 0
            loss_position_total = 0
            num_targets_in_batch = 0
            
            for i in range(len(y_batch_list)):
                ts_len = batch_lens[i]
                strain = batch_strains[i]
                targets_tuples = y_batch_list[i]

                input_str = batch_input_strs[i]   # モデルに入力された範囲の文字列
                target_str = batch_target_strs[i] # 予測対象の文字列
                full_path = batch_full_paths[i]   # 全体の文字列

                pred_set_region = set(topk_indices_region[i].cpu().tolist())
                target_set_region = set([t[0] for t in targets_tuples])
                
                pred_set_position = set(topk_indices_position[i].cpu().tolist())
                target_set_position = set([t[1] for t in targets_tuples])

                hit_region = len(pred_set_region.intersection(target_set_region)) > 0
                hit_position = len(pred_set_position.intersection(target_set_position)) > 0

                detailed_results.append({
                    'len': ts_len,
                    'strain': strain,
                    'raw_path': full_path,
                    'targets_region': target_set_region,
                    'preds_region': pred_set_region,
                    'hit_region': hit_region,
                    'targets_position': target_set_position,
                    'preds_position': pred_set_position,
                    'hit_position': hit_position
                })

                if not example_printed and targets_tuples:
                    print("\n--- Evaluation Example ---")
                    print(f"  Strain: {strain}")
                    print(f"  Len: {ts_len}")
                    print(f"  [Original Path]: {full_path}")
                    print(f"  [Input Sequence]: {input_str}")
                    print(f"  [Target Sequence]: {target_str}")
                    print(f"  [Target IDs]: {targets_tuples}")
                    print(f"  Region Preds: {pred_set_region}")
                    print(f"  Position Preds: {pred_set_position}")
                    print("--------------------------")
                    example_printed = True

                results_by_timestep[ts_len]["preds_region"].append(pred_set_region)
                results_by_timestep[ts_len]["targets_region"].append(target_set_region)
                results_by_timestep[ts_len]["preds_position"].append(pred_set_position)
                results_by_timestep[ts_len]["targets_position"].append(target_set_position)

                if not targets_tuples: continue
                
                targets_region_tensor = torch.tensor([t[0] for t in targets_tuples], dtype=torch.long).to(config.DEVICE)
                targets_position_tensor = torch.tensor([t[1] for t in targets_tuples], dtype=torch.long).to(config.DEVICE)
                num_targets = len(targets_tuples)

                loss_r = loss_fn(predictions_region[i].expand(num_targets, -1), targets_region_tensor)
                loss_p = loss_fn(predictions_position[i].expand(num_targets, -1), targets_position_tensor)
                
                loss_region_total += loss_r.sum()
                loss_position_total += loss_p.sum()
                num_targets_in_batch += num_targets

            if num_targets_in_batch > 0:
                avg_loss_r = loss_region_total / num_targets_in_batch
                avg_loss_p = loss_position_total / num_targets_in_batch
                total_loss = (config.LOSS_WEIGHT_REGION * avg_loss_r) + \
                             (config.LOSS_WEIGHT_POSITION * avg_loss_p)
                
                total_epoch_loss += total_loss.item()
                batches_processed += 1

    final_metrics_by_ts = {}
    for ts_len, data in results_by_timestep.items():
        hit_reg, prec_reg, recall_reg, f1_reg = calculate_metrics(data["preds_region"], data["targets_region"])
        hit_pos, prec_pos, recall_pos, f1_pos = calculate_metrics(data["preds_position"], data["targets_position"])
        
        final_metrics_by_ts[ts_len] = {
            "num_samples": len(data["preds_region"]),
            "region_hit_rate": hit_reg,
            "region_precision": prec_reg,
            "region_recall": recall_reg,
            "region_f1": f1_reg,
            "position_hit_rate": hit_pos,
            "position_precision": prec_pos,
            "position_recall": recall_pos,
            "position_f1": f1_pos
        }

    avg_loss = total_epoch_loss / batches_processed if batches_processed > 0 else 0
    
    return avg_loss, final_metrics_by_ts, detailed_results