# --- evaluate.py ---
import torch
from collections import defaultdict
from .utils import calculate_topk_hit_rate # utils.pyからインポート
from . import config

def evaluate(model, dataloader, loss_fn):
    model.eval() # 評価モード
    total_epoch_loss = 0
    batches_processed = 0
    
    # 辞書構造を変更し、領域(region)と位置(position)の両方を格納
    results_by_timestep = defaultdict(lambda: {
        "preds_region": [], "targets_region": [],
        "preds_position": [], "targets_position": []
    })

    example_printed = False # デバック用

    with torch.no_grad():
        # raw_inputsとraw_pathsはデバック用
        for (x_cat, x_num, mask), y_batch_list, batch_lens, raw_inputs, raw_paths in dataloader:
        
            x_cat = x_cat.to(config.DEVICE)
            x_num = x_num.to(config.DEVICE)
            mask = mask.to(config.DEVICE)

            # 1. 予測 (2つの出力を受け取る)
            predictions_region, predictions_position = model(x_cat, x_num, src_key_padding_mask=mask)
            
            # 2. Top-K 予測セットの生成
            topk_indices_region = torch.topk(predictions_region, config.TOP_K_EVAL, dim=1).indices
            topk_indices_position = torch.topk(predictions_position, config.TOP_K_EVAL, dim=1).indices
            
            # 3. 損失計算と結果の辞書への格納
            loss_region_total = 0
            loss_position_total = 0
            num_targets_in_batch = 0
            
            for i in range(len(y_batch_list)):
                ts_len = batch_lens[i]
                targets_tuples = y_batch_list[i] # [(r1, p1), (r2, p2)]

                # 領域のセットを作成
                pred_set_region = set(topk_indices_region[i].cpu().tolist())
                target_set_region = set([t[0] for t in targets_tuples])
                
                # 位置のセットを作成
                pred_set_position = set(topk_indices_position[i].cpu().tolist())
                target_set_position = set([t[1] for t in targets_tuples])

                if not example_printed and targets_tuples:
                    hit_region = len(pred_set_region.intersection(target_set_region)) > 0
                    hit_position = len(pred_set_position.intersection(target_set_position)) > 0

                    input_sequence_example = raw_inputs[i]
                    raw_path_example = raw_paths[i]
                    
                    print("\n--- Evaluation Example (First Sample of First Batch) ---")
                    print(f"  Len: {ts_len}")
                    print(f"  Raw Path: {raw_path_example}")
                    print(f"  Input Sequence (Features): {input_sequence_example}")
                    print(f"  Targets (Region, Position): {targets_tuples}")
                    print(f"  Region Preds (Top-{config.TOP_K_EVAL}): {pred_set_region}")
                    print(f"  Position Preds (Top-{config.TOP_K_EVAL}): {pred_set_position}")
                    print(f"  Hit? (Region): {hit_region}")
                    print(f"  Hit? (Position): {hit_position}")
                    print("--------------------------------------------------------")
                    example_printed = True

                # 辞書に両方の結果を格納
                results_by_timestep[ts_len]["preds_region"].append(pred_set_region)
                results_by_timestep[ts_len]["targets_region"].append(target_set_region)
                results_by_timestep[ts_len]["preds_position"].append(pred_set_position)
                results_by_timestep[ts_len]["targets_position"].append(target_set_position)

                if not targets_tuples: continue
                
                # 損失計算 (train.pyと同様)
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

    # 4. 最終的なメトリクスをタイムステップ別に集計
    final_metrics_by_ts = {}
    
    for ts_len, data in results_by_timestep.items():
        # 領域のヒット率を計算
        hit_rate_region = calculate_topk_hit_rate(data["preds_region"], data["targets_region"])
        # 位置のヒット率を計算
        hit_rate_position = calculate_topk_hit_rate(data["preds_position"], data["targets_position"])
        
        final_metrics_by_ts[ts_len] = {
            "region_hit_rate_percent": hit_rate_region,
            "position_hit_rate_percent": hit_rate_position,
            "num_samples": len(data["preds_region"])
        }

    avg_loss = total_epoch_loss / batches_processed if batches_processed > 0 else 0
    
    return avg_loss, final_metrics_by_ts