# --- evaluate.py ---
import torch
from collections import defaultdict
from .utils import calculate_metrics
from . import config

def evaluate(model, dataloader, loss_fn, strength_thresholds=None):
    """
    Args:
        model: 評価対象のモデル
        dataloader: データローダー
        loss_fn: 損失関数
        strength_thresholds: (low_max, med_max) の動的閾値タプル。Noneの場合はconfigの値を使用
    """
    model.eval()
    total_epoch_loss = 0
    batches_processed = 0
    
    # 動的閾値の設定
    if strength_thresholds is not None:
        low_max, med_max = strength_thresholds
    else:
        low_max = config.STRENGTH_CATEGORY_LOW_MAX
        med_max = config.STRENGTH_CATEGORY_MED_MAX
    
    # タイムステップ別の集計用 (タンパク質位置追加)
    results_by_timestep = defaultdict(lambda: {
        "preds_region": [], "targets_region": [],
        "preds_position": [], "targets_position": [],
        "preds_protein_pos": [], "targets_protein_pos": [],
        "preds_strength": [], "targets_strength": []  # 強度スコア予測
    })
    
    # 強度カテゴリ別 (timestep, category) -> {preds_strength, targets_strength, ...}
    results_by_ts_and_category = defaultdict(lambda: defaultdict(lambda: {
        "preds_strength": [], "targets_strength": [],
        "preds_region": [], "targets_region": [],
        "preds_position": [], "targets_position": [],
        "preds_protein_pos": [], "targets_protein_pos": []
    }))

    # 保存用の詳細結果リスト
    detailed_results = []
    
    # 強度フィルタの統計
    total_samples = 0
    filtered_samples = 0
    
    def get_strength_category(strength_score):
        """強度スコアからカテゴリを判定（動的閾値を使用）"""
        if strength_score < low_max:
            return 'low'
        elif strength_score < med_max:
            return 'medium'
        else:
            return 'high'

    example_printed = False

    with torch.no_grad():
        # dataset.py の collate_fn の戻り値に合わせて受け取る (8個)
        for (x_cat, x_num, mask), y_batch_list, batch_lens, batch_strains, batch_strength_scores, batch_input_strs, batch_target_strs, batch_full_paths in dataloader:
        
            x_cat = x_cat.to(config.DEVICE)
            x_num = x_num.to(config.DEVICE)
            mask = mask.to(config.DEVICE)

            # モデル出力: 4つの予測
            predictions_region, predictions_position, predictions_protein_pos, predictions_strength = model(x_cat, x_num, src_key_padding_mask=mask)
            
            topk_indices_region = torch.topk(predictions_region, config.TOP_K_EVAL, dim=1).indices
            topk_indices_position = torch.topk(predictions_position, config.TOP_K_EVAL, dim=1).indices
            topk_indices_protein_pos = torch.topk(predictions_protein_pos, config.TOP_K_EVAL, dim=1).indices
            
            loss_region_total = 0
            loss_position_total = 0
            loss_protein_pos_total = 0
            num_targets_in_batch = 0
            
            for i in range(len(y_batch_list)):
                ts_len = batch_lens[i]
                strain = batch_strains[i]
                strength_score = batch_strength_scores[i]
                targets_tuples = y_batch_list[i]

                input_str = batch_input_strs[i]   # モデルに入力された範囲の文字列
                target_str = batch_target_strs[i] # 予測対象の文字列
                full_path = batch_full_paths[i]   # 全体の文字列
                
                total_samples += 1
                
                # 強度フィルタの適用
                if config.USE_STRENGTH_FILTER and strength_score < config.STRENGTH_THRESHOLD:
                    filtered_samples += 1
                    continue

                pred_set_region = set(topk_indices_region[i].cpu().tolist())
                target_set_region = set([t[0] for t in targets_tuples])
                
                pred_set_position = set(topk_indices_position[i].cpu().tolist())
                target_set_position = set([t[1] for t in targets_tuples])
                
                pred_set_protein_pos = set(topk_indices_protein_pos[i].cpu().tolist())
                target_set_protein_pos = set([t[2] for t in targets_tuples])

                hit_region = len(pred_set_region.intersection(target_set_region)) > 0
                hit_position = len(pred_set_position.intersection(target_set_position)) > 0
                hit_protein_pos = len(pred_set_protein_pos.intersection(target_set_protein_pos)) > 0
                
                # 強度スコア予測値
                pred_strength = predictions_strength[i].item()

                detailed_results.append({
                    'len': ts_len,
                    'strain': strain,
                    'strength_score': strength_score,
                    'pred_strength': pred_strength,
                    'raw_path': full_path,
                    'targets_region': target_set_region,
                    'preds_region': pred_set_region,
                    'hit_region': hit_region,
                    'targets_position': target_set_position,
                    'preds_position': pred_set_position,
                    'hit_position': hit_position,
                    'targets_protein_pos': target_set_protein_pos,
                    'preds_protein_pos': pred_set_protein_pos,
                    'hit_protein_pos': hit_protein_pos
                })

                if not example_printed and targets_tuples:
                    print("\n--- Evaluation Example ---")
                    print(f"  Strain: {strain} (Strength: {strength_score:.2f}, Pred: {pred_strength:.2f})")
                    print(f"  Len: {ts_len}")
                    print(f"  [Original Path]: {full_path}")
                    print(f"  [Input Sequence]: {input_str}")
                    print(f"  [Target Sequence]: {target_str}")
                    print(f"  [Target IDs]: {targets_tuples}")
                    print(f"  Region Preds: {pred_set_region}")
                    print(f"  Position Preds: {pred_set_position}")
                    print(f"  Protein Position Preds: {pred_set_protein_pos}")
                    print("--------------------------")
                    example_printed = True

                results_by_timestep[ts_len]["preds_region"].append(pred_set_region)
                results_by_timestep[ts_len]["targets_region"].append(target_set_region)
                results_by_timestep[ts_len]["preds_position"].append(pred_set_position)
                results_by_timestep[ts_len]["targets_position"].append(target_set_position)
                results_by_timestep[ts_len]["preds_protein_pos"].append(pred_set_protein_pos)
                results_by_timestep[ts_len]["targets_protein_pos"].append(target_set_protein_pos)
                results_by_timestep[ts_len]["preds_strength"].append(pred_strength)
                results_by_timestep[ts_len]["targets_strength"].append(strength_score)
                
                # 強度カテゴリ別にも協積
                category = get_strength_category(strength_score)
                results_by_ts_and_category[ts_len][category]["preds_strength"].append(pred_strength)
                results_by_ts_and_category[ts_len][category]["targets_strength"].append(strength_score)
                results_by_ts_and_category[ts_len][category]["preds_region"].append(pred_set_region)
                results_by_ts_and_category[ts_len][category]["targets_region"].append(target_set_region)
                results_by_ts_and_category[ts_len][category]["preds_position"].append(pred_set_position)
                results_by_ts_and_category[ts_len][category]["targets_position"].append(target_set_position)
                results_by_ts_and_category[ts_len][category]["preds_protein_pos"].append(pred_set_protein_pos)
                results_by_ts_and_category[ts_len][category]["targets_protein_pos"].append(target_set_protein_pos)

                if not targets_tuples: continue
                
                targets_region_tensor = torch.tensor([t[0] for t in targets_tuples], dtype=torch.long).to(config.DEVICE)
                targets_position_tensor = torch.tensor([t[1] for t in targets_tuples], dtype=torch.long).to(config.DEVICE)
                targets_protein_pos_tensor = torch.tensor([t[2] for t in targets_tuples], dtype=torch.long).to(config.DEVICE)
                num_targets = len(targets_tuples)

                loss_r = loss_fn(predictions_region[i].expand(num_targets, -1), targets_region_tensor)
                loss_p = loss_fn(predictions_position[i].expand(num_targets, -1), targets_position_tensor)
                loss_protein_pos = loss_fn(predictions_protein_pos[i].expand(num_targets, -1), targets_protein_pos_tensor)
                
                loss_region_total += loss_r.sum()
                loss_position_total += loss_p.sum()
                loss_protein_pos_total += loss_protein_pos.sum()
                num_targets_in_batch += num_targets

            if num_targets_in_batch > 0:
                avg_loss_r = loss_region_total / num_targets_in_batch
                avg_loss_p = loss_position_total / num_targets_in_batch
                avg_loss_protein_pos = loss_protein_pos_total / num_targets_in_batch
                total_loss = (config.LOSS_WEIGHT_REGION * avg_loss_r) + \
                             (config.LOSS_WEIGHT_POSITION * avg_loss_p) + \
                             (config.LOSS_WEIGHT_PROTEIN_POS * avg_loss_protein_pos)
                
                total_epoch_loss += total_loss.item()
                batches_processed += 1
    
    # 強度フィルタの結果表示
    if config.USE_STRENGTH_FILTER and config.STRENGTH_THRESHOLD > 0:
        print(f"[INFO] Strength filter: {filtered_samples}/{total_samples} samples excluded (threshold={config.STRENGTH_THRESHOLD:.2f})")

    final_metrics_by_ts = {}
    for ts_len, data in results_by_timestep.items():
        hit_reg, prec_reg, recall_reg, f1_reg = calculate_metrics(data["preds_region"], data["targets_region"])
        hit_pos, prec_pos, recall_pos, f1_pos = calculate_metrics(data["preds_position"], data["targets_position"])
        hit_protein_pos, prec_protein_pos, recall_protein_pos, f1_protein_pos = calculate_metrics(data["preds_protein_pos"], data["targets_protein_pos"])
        
        final_metrics_by_ts[ts_len] = {
            "num_samples": len(data["preds_region"]),
            "region_hit_rate": hit_reg,
            "region_precision": prec_reg,
            "region_recall": recall_reg,
            "region_f1": f1_reg,
            "position_hit_rate": hit_pos,
            "position_precision": prec_pos,
            "position_recall": recall_pos,
            "position_f1": f1_pos,
            "protein_pos_hit_rate": hit_protein_pos,
            "protein_pos_precision": prec_protein_pos,
            "protein_pos_recall": recall_protein_pos,
            "protein_pos_f1": f1_protein_pos,
            "strength_mae": sum(abs(p - t) for p, t in zip(data["preds_strength"], data["targets_strength"])) / len(data["preds_strength"]) if data["preds_strength"] else 0.0
        }
    
    # 強度カテゴリ別のメトリクス計算
    metrics_by_category = {}
    for ts_len in results_by_ts_and_category.keys():
        metrics_by_category[ts_len] = {}
        for category in ['low', 'medium', 'high']:
            cat_data = results_by_ts_and_category[ts_len][category]
            num_samples = len(cat_data["preds_strength"])
            if num_samples > 0:
                mae = sum(abs(p - t) for p, t in zip(cat_data["preds_strength"], cat_data["targets_strength"])) / num_samples
                hit_reg, _, _, _ = calculate_metrics(cat_data["preds_region"], cat_data["targets_region"])
                hit_pos, _, _, _ = calculate_metrics(cat_data["preds_position"], cat_data["targets_position"])
                hit_prot, _, _, _ = calculate_metrics(cat_data["preds_protein_pos"], cat_data["targets_protein_pos"])
            else:
                mae = 0.0
                hit_reg = hit_pos = hit_prot = 0.0
            
            metrics_by_category[ts_len][category] = {
                "num_samples": num_samples,
                "strength_mae": mae,
                "region_hit_rate": hit_reg,
                "position_hit_rate": hit_pos,
                "protein_pos_hit_rate": hit_prot
            }

    avg_loss = total_epoch_loss / batches_processed if batches_processed > 0 else 0
    
    return avg_loss, final_metrics_by_ts, detailed_results, metrics_by_category