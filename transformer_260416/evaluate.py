# --- evaluate.py ---
import torch
from collections import defaultdict
from .utils.logging import calculate_metrics, calculate_weighted_macro_recall
from .utils.losses import compute_task_losses
from . import config


def evaluate(model, dataloader, loss_fn, strength_thresholds=None):
    """モデルを評価し、各タスクのメトリクスと詳細結果を返す。

    Args:
        model: 評価対象のモデル
        dataloader: データローダー
        loss_fn: 損失関数
        strength_thresholds: (low_max, med_max) の動的閾値タプル。Noneの場合はconfigの値を使用

    Returns:
        Tuple[float, dict, list, dict]:
            - avg_loss: エポック平均損失
            - final_metrics_by_ts: タイムステップ別メトリクス辞書
            - detailed_results: サンプル別詳細結果リスト
            - metrics_by_category: 流行度カテゴリ別メトリクス辞書
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

    # タイムステップ別の集計用
    results_by_timestep = defaultdict(lambda: {
        "preds_region": [], "targets_region": [],
        "preds_position": [], "targets_position": [],
        "preds_aa_pos": [], "targets_aa_pos": [],
        "preds_codon_pos": [], "targets_codon_pos": [],
        "preds_synonymous": [], "targets_synonymous": [],
        "preds_strength": [], "targets_strength": []
    })

    # 流行度カテゴリ別
    results_by_ts_and_category = defaultdict(lambda: defaultdict(lambda: {
        "preds_strength": [], "targets_strength": [],
        "preds_region": [], "targets_region": [],
        "preds_position": [], "targets_position": [],
        "preds_aa_pos": [], "targets_aa_pos": [],
        "preds_codon_pos": [], "targets_codon_pos": [],
        "preds_synonymous": [], "targets_synonymous": []
    }))

    # 保存用の詳細結果リスト
    detailed_results = []

    # 流行度フィルタの統計
    total_samples = 0
    filtered_samples = 0

    def get_strength_category(strength_score):
        """流行度からカテゴリを判定（動的閾値を使用）"""
        if strength_score < low_max:
            return 'low'
        elif strength_score < med_max:
            return 'medium'
        else:
            return 'high'

    example_printed = False

    with torch.no_grad():
        for (x_cat, x_num, mask), y_batch_list, batch_lens, batch_strains, batch_strength_scores, batch_input_strs, batch_target_strs, batch_full_paths in dataloader:

            x_cat = x_cat.to(config.DEVICE)
            x_num = x_num.to(config.DEVICE)
            mask = mask.to(config.DEVICE)

            # モデル出力: 6つの予測
            predictions_tuple = model(x_cat, x_num, src_key_padding_mask=mask)
            (predictions_region, predictions_position, predictions_aa_pos,
             predictions_strength, predictions_codon_pos, predictions_synonymous) = predictions_tuple

            # Top-K 推論
            def safe_topk(tensor, k):
                """kがクラス数を超えないように調整してtopkを取得"""
                actual_k = min(k, tensor.size(1))
                return torch.topk(tensor, actual_k, dim=1).indices

            topk_indices_region    = safe_topk(predictions_region,    config.TOP_K_EVAL)
            topk_indices_position  = safe_topk(predictions_position,  config.TOP_K_EVAL)
            topk_indices_aa_pos    = safe_topk(predictions_aa_pos,    config.TOP_K_EVAL)
            topk_indices_codon_pos = safe_topk(predictions_codon_pos, config.TOP_K_EVAL)
            topk_indices_synonymous= safe_topk(predictions_synonymous,config.TOP_K_EVAL)

            # 損失計算 (utils/losses.py 共通ロジック)
            losses = compute_task_losses(
                predictions_tuple, y_batch_list, loss_fn, batch_strength_scores
            )
            if losses is not None:
                total_epoch_loss += losses['total'].item()
                batches_processed += 1

            # サンプルごとのメトリクス集計
            for i in range(len(y_batch_list)):
                ts_len = batch_lens[i]
                strain = batch_strains[i]
                strength_score = batch_strength_scores[i]
                targets_tuples = y_batch_list[i]

                input_str = batch_input_strs[i]
                target_str = batch_target_strs[i]
                full_path = batch_full_paths[i]

                total_samples += 1

                # 流行度フィルタの適用
                if config.USE_STRENGTH_FILTER and strength_score < config.STRENGTH_THRESHOLD:
                    filtered_samples += 1
                    continue

                pred_set_region    = set(topk_indices_region[i].cpu().tolist())
                target_set_region  = set(t[0] for t in targets_tuples)
                pred_set_position  = set(topk_indices_position[i].cpu().tolist())
                target_set_position= set(t[1] for t in targets_tuples)
                pred_set_aa_pos    = set(topk_indices_aa_pos[i].cpu().tolist())
                target_set_aa_pos  = set(t[2] for t in targets_tuples)
                pred_set_codon_pos = set(topk_indices_codon_pos[i].cpu().tolist())
                target_set_codon_pos= set(t[3] for t in targets_tuples)
                pred_set_synonymous= set(topk_indices_synonymous[i].cpu().tolist())
                target_set_synonymous= set(t[4] for t in targets_tuples)

                hit_region     = len(pred_set_region.intersection(target_set_region)) > 0
                hit_position   = len(pred_set_position.intersection(target_set_position)) > 0
                hit_aa_pos     = len(pred_set_aa_pos.intersection(target_set_aa_pos)) > 0
                hit_codon_pos  = len(pred_set_codon_pos.intersection(target_set_codon_pos)) > 0
                hit_synonymous = len(pred_set_synonymous.intersection(target_set_synonymous)) > 0

                pred_strength = predictions_strength[i].item()

                detailed_results.append({
                    'len': ts_len, 'strain': strain,
                    'strength_score': strength_score, 'pred_strength': pred_strength,
                    'raw_path': full_path,
                    'targets_region': target_set_region,   'preds_region': pred_set_region,   'hit_region': hit_region,
                    'targets_position': target_set_position,'preds_position': pred_set_position,'hit_position': hit_position,
                    'targets_aa_pos': target_set_aa_pos,   'preds_aa_pos': pred_set_aa_pos,   'hit_aa_pos': hit_aa_pos,
                    'targets_codon_pos': target_set_codon_pos,'preds_codon_pos': pred_set_codon_pos,'hit_codon_pos': hit_codon_pos,
                    'targets_synonymous': target_set_synonymous,'preds_synonymous': pred_set_synonymous,'hit_synonymous': hit_synonymous,
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
                    print(f"  Base Pos Preds: {pred_set_position}")
                    print(f"  AA Pos Preds: {pred_set_aa_pos}")
                    print(f"  Codon Pos Preds: {pred_set_codon_pos}")
                    print(f"  Synonymous Preds: {pred_set_synonymous}")
                    print("--------------------------")
                    example_printed = True

                # タイムステップ別の集計
                results_by_timestep[ts_len]["preds_region"].append(pred_set_region)
                results_by_timestep[ts_len]["targets_region"].append(target_set_region)
                results_by_timestep[ts_len]["preds_position"].append(pred_set_position)
                results_by_timestep[ts_len]["targets_position"].append(target_set_position)
                results_by_timestep[ts_len]["preds_aa_pos"].append(pred_set_aa_pos)
                results_by_timestep[ts_len]["targets_aa_pos"].append(target_set_aa_pos)
                results_by_timestep[ts_len]["preds_codon_pos"].append(pred_set_codon_pos)
                results_by_timestep[ts_len]["targets_codon_pos"].append(target_set_codon_pos)
                results_by_timestep[ts_len]["preds_synonymous"].append(pred_set_synonymous)
                results_by_timestep[ts_len]["targets_synonymous"].append(target_set_synonymous)
                results_by_timestep[ts_len]["preds_strength"].append(pred_strength)
                results_by_timestep[ts_len]["targets_strength"].append(strength_score)

                # 流行度カテゴリ別に蓄積
                category = get_strength_category(strength_score)
                results_by_ts_and_category[ts_len][category]["preds_strength"].append(pred_strength)
                results_by_ts_and_category[ts_len][category]["targets_strength"].append(strength_score)
                results_by_ts_and_category[ts_len][category]["preds_region"].append(pred_set_region)
                results_by_ts_and_category[ts_len][category]["targets_region"].append(target_set_region)
                results_by_ts_and_category[ts_len][category]["preds_position"].append(pred_set_position)
                results_by_ts_and_category[ts_len][category]["targets_position"].append(target_set_position)
                results_by_ts_and_category[ts_len][category]["preds_aa_pos"].append(pred_set_aa_pos)
                results_by_ts_and_category[ts_len][category]["targets_aa_pos"].append(target_set_aa_pos)
                results_by_ts_and_category[ts_len][category]["preds_codon_pos"].append(pred_set_codon_pos)
                results_by_ts_and_category[ts_len][category]["targets_codon_pos"].append(target_set_codon_pos)
                results_by_ts_and_category[ts_len][category]["preds_synonymous"].append(pred_set_synonymous)
                results_by_ts_and_category[ts_len][category]["targets_synonymous"].append(target_set_synonymous)

    # 流行度フィルタの結果表示
    if config.USE_STRENGTH_FILTER and config.STRENGTH_THRESHOLD > 0:
        print(f"[INFO] Strength filter: {filtered_samples}/{total_samples} samples excluded "
              f"(threshold={config.STRENGTH_THRESHOLD:.2f})")

    # タイムステップ別メトリクス計算
    final_metrics_by_ts = {}
    for ts_len, data in results_by_timestep.items():
        hit_reg, prec_reg, recall_reg, f1_reg = calculate_metrics(data["preds_region"], data["targets_region"])
        hit_pos, prec_pos, recall_pos, f1_pos = calculate_metrics(data["preds_position"], data["targets_position"])
        hit_aa_pos, prec_aa_pos, recall_aa_pos, f1_aa_pos = calculate_metrics(data["preds_aa_pos"], data["targets_aa_pos"])
        hit_codon_pos, prec_codon_pos, recall_codon_pos, f1_codon_pos = calculate_metrics(data["preds_codon_pos"], data["targets_codon_pos"])
        hit_synonymous, prec_synonymous, recall_synonymous, f1_synonymous = calculate_metrics(data["preds_synonymous"], data["targets_synonymous"])

        w_recall_reg,   m_recall_reg   = calculate_weighted_macro_recall(data["preds_region"],     data["targets_region"])
        w_recall_pos,   m_recall_pos   = calculate_weighted_macro_recall(data["preds_position"],   data["targets_position"])
        w_recall_aa,    m_recall_aa    = calculate_weighted_macro_recall(data["preds_aa_pos"],     data["targets_aa_pos"])
        w_recall_codon, m_recall_codon = calculate_weighted_macro_recall(data["preds_codon_pos"], data["targets_codon_pos"])
        w_recall_syn,   m_recall_syn   = calculate_weighted_macro_recall(data["preds_synonymous"], data["targets_synonymous"])

        final_metrics_by_ts[ts_len] = {
            "num_samples": len(data["preds_region"]),
            "region_hit_rate": hit_reg,
            "region_precision": prec_reg,
            "region_recall": recall_reg,
            "region_f1": f1_reg,
            "region_weighted_recall": w_recall_reg,
            "region_macro_recall": m_recall_reg,
            "position_hit_rate": hit_pos,
            "position_precision": prec_pos,
            "position_recall": recall_pos,
            "position_f1": f1_pos,
            "position_weighted_recall": w_recall_pos,
            "position_macro_recall": m_recall_pos,
            "aa_pos_hit_rate": hit_aa_pos,
            "aa_pos_precision": prec_aa_pos,
            "aa_pos_recall": recall_aa_pos,
            "aa_pos_f1": f1_aa_pos,
            "aa_pos_weighted_recall": w_recall_aa,
            "aa_pos_macro_recall": m_recall_aa,
            "codon_pos_hit_rate": hit_codon_pos,
            "codon_pos_precision": prec_codon_pos,
            "codon_pos_recall": recall_codon_pos,
            "codon_pos_f1": f1_codon_pos,
            "codon_pos_weighted_recall": w_recall_codon,
            "codon_pos_macro_recall": m_recall_codon,
            "synonymous_hit_rate": hit_synonymous,
            "synonymous_precision": prec_synonymous,
            "synonymous_recall": recall_synonymous,
            "synonymous_f1": f1_synonymous,
            "synonymous_weighted_recall": w_recall_syn,
            "synonymous_macro_recall": m_recall_syn,
            "strength_mae": (
                sum(abs(p - t) for p, t in zip(data["preds_strength"], data["targets_strength"])) /
                len(data["preds_strength"])
                if data["preds_strength"] else 0.0
            )
        }

    # 流行度カテゴリ別のメトリクス計算
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
                hit_aa, _, _, _ = calculate_metrics(cat_data["preds_aa_pos"], cat_data["targets_aa_pos"])
                hit_codon, _, _, _ = calculate_metrics(cat_data["preds_codon_pos"], cat_data["targets_codon_pos"])
                hit_syn, _, _, _ = calculate_metrics(cat_data["preds_synonymous"], cat_data["targets_synonymous"])
            else:
                mae = 0.0
                hit_reg = hit_pos = hit_aa = hit_codon = hit_syn = 0.0

            metrics_by_category[ts_len][category] = {
                "num_samples": num_samples,
                "strength_mae": mae,
                "region_hit_rate": hit_reg,
                "position_hit_rate": hit_pos,
                "aa_pos_hit_rate": hit_aa,
                "codon_pos_hit_rate": hit_codon,
                "synonymous_hit_rate": hit_syn
            }

    avg_loss = total_epoch_loss / batches_processed if batches_processed > 0 else 0

    return avg_loss, final_metrics_by_ts, detailed_results, metrics_by_category


def evaluate_topk(model, dataloader, ks=(1, 3, 5)):
    """複数の K 値で Top-K Precision / Recall を同時に計算する。

    Args:
        model: 評価対象モデル
        dataloader: DataLoader
        ks: 評価する K の値のタプル (例: (1, 3, 5))

    Returns:
        dict: {
            task_name: {k: {'precision': float, 'recall': float, 'hit_rate': float}}
        }
        task_name は 'region' / 'position' / 'aa_pos' / 'codon_pos' / 'synonymous'
    """
    model.eval()

    tasks = ['region', 'position', 'aa_pos', 'codon_pos', 'synonymous']
    # task -> k -> {'tp': int, 'fp': int, 'total_targets': int, 'hits': int, 'n': int}
    stats = {t: {k: {'tp': 0, 'fp': 0, 'total_targets': 0, 'hits': 0, 'n': 0} for k in ks} for t in tasks}

    def _topk_preds(tensor, k):
        actual_k = min(k, tensor.size(1))
        return torch.topk(tensor, actual_k, dim=1).indices

    with torch.no_grad():
        for (x_cat, x_num, mask), y_batch_list, _, _, _, _, _, _ in dataloader:
            x_cat = x_cat.to(config.DEVICE)
            x_num = x_num.to(config.DEVICE)
            mask  = mask.to(config.DEVICE)

            (pred_region, pred_position, pred_aa_pos,
             _, pred_codon_pos, pred_synonymous) = model(x_cat, x_num, src_key_padding_mask=mask)

            preds_all = {
                'region':    pred_region,
                'position':  pred_position,
                'aa_pos':    pred_aa_pos,
                'codon_pos': pred_codon_pos,
                'synonymous': pred_synonymous,
            }
            target_idx = {'region': 0, 'position': 1, 'aa_pos': 2, 'codon_pos': 3, 'synonymous': 4}

            for i, targets_tuples in enumerate(y_batch_list):
                if not targets_tuples:
                    continue
                for task in tasks:
                    t_set = set(t[target_idx[task]] for t in targets_tuples)
                    for k in ks:
                        pred_set = set(_topk_preds(preds_all[task], k)[i].cpu().tolist())
                        tp = len(pred_set & t_set)
                        stats[task][k]['tp']            += tp
                        stats[task][k]['fp']            += len(pred_set) - tp
                        stats[task][k]['total_targets'] += len(t_set)
                        stats[task][k]['hits']          += int(tp > 0)
                        stats[task][k]['n']             += 1

    results = {}
    for task in tasks:
        results[task] = {}
        for k in ks:
            s = stats[task][k]
            total_pred  = s['tp'] + s['fp']
            precision   = (s['tp'] / total_pred  * 100) if total_pred  > 0 else 0.0
            recall      = (s['tp'] / s['total_targets'] * 100) if s['total_targets'] > 0 else 0.0
            hit_rate    = (s['hits'] / s['n'] * 100) if s['n'] > 0 else 0.0
            results[task][k] = {'precision': precision, 'recall': recall, 'hit_rate': hit_rate}

    return results
