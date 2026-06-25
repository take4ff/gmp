# --- evaluate.py ---
import torch
from collections import defaultdict
from .utils.logging import calculate_metrics, calculate_weighted_macro_recall
from .utils.losses import compute_task_losses
from . import config


def _predict_tta(model, x_cat, x_num, mask, n_passes):
    """MC Dropout TTA: Dropout を有効にしたまま n_passes 回推論し、softmax 平均を返す。

    通常の eval() モードだと Dropout が無効化されるが、この関数内では
    model.train() を呼ぶことで Dropout を有効にする。完了後は model.eval() に戻す。

    Args:
        model: 推論対象モデル
        x_cat, x_num, mask: モデル入力
        n_passes: 推論回数

    Returns:
        Tuple[Tensor x 6]: 各タスクの softmax 平均確率ベクトルのタプル
    """
    model.train()  # Dropout を有効化
    all_logits = []
    with torch.no_grad():
        for _ in range(n_passes):
            out = model(x_cat, x_num, src_key_padding_mask=mask)
            # (region, position, aa_pos, strength, codon_pos, synonymous)
            probs = tuple(torch.softmax(o, dim=-1) if o.dim() > 1 else o for o in out)
            all_logits.append(probs)
    model.eval()  # 推論後は必ず eval に戻す

    # タスクごとに n_passes 平均を返す
    n_tasks = len(all_logits[0])
    averaged = []
    for task_idx in range(n_tasks):
        stacked = torch.stack([p[task_idx] for p in all_logits], dim=0)
        averaged.append(stacked.mean(dim=0))
    return tuple(averaged)


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

    # ECE 計算用: タイムステップ別に (pred_probs, hit) のリストを蓄積
    results_for_ece = defaultdict(lambda: {
        'probs_region': [], 'hits_region': [],
        'probs_position': [], 'hits_position': [],
        'probs_aa_pos': [], 'hits_aa_pos': [],
    })

    # 月別集計用 (EVAL_X_AXIS == 'date' のときのみ使用)
    eval_x_axis = getattr(config, 'EVAL_X_AXIS', 'timestep')
    results_by_yearmonth = defaultdict(lambda: {
        "preds_region": [], "targets_region": [],
        "preds_position": [], "targets_position": [],
        "preds_aa_pos": [], "targets_aa_pos": [],
        "preds_codon_pos": [], "targets_codon_pos": [],
        "preds_synonymous": [], "targets_synonymous": [],
        "preds_strength": [], "targets_strength": []
    })

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
        for (x_cat, x_num, mask), y_batch, batch_lens, batch_strains, batch_strength_scores, batch_input_strs, batch_target_strs, batch_full_paths, raw_y_batch, batch_collection_dates in dataloader:

            x_cat = x_cat.to(config.DEVICE)
            x_num = x_num.to(config.DEVICE)
            mask = mask.to(config.DEVICE)

            # モデル出力: 6つの予測 (TTA 時は softmax 平均値を返す)
            if getattr(config, 'USE_TTA', False):
                n_passes = getattr(config, 'TTA_N_PASSES', 5)
                predictions_tuple = _predict_tta(model, x_cat, x_num, mask, n_passes)
            else:
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

            # 確率の計算 (TTA時はすでにsoftmax済みのため再適用しない)
            if getattr(config, 'USE_TTA', False):
                probs_region   = predictions_region
                probs_position = predictions_position
                probs_aa_pos   = predictions_aa_pos
            else:
                probs_region   = torch.nn.functional.softmax(predictions_region,   dim=-1)
                probs_position = torch.nn.functional.softmax(predictions_position, dim=-1)
                probs_aa_pos   = torch.nn.functional.softmax(predictions_aa_pos,   dim=-1)

            # 損失計算 (utils/losses.py 共通ロジック)
            losses = compute_task_losses(
                predictions_tuple, y_batch, loss_fn, batch_strength_scores,
                raw_y_batch=raw_y_batch,
            )
            if losses is not None:
                total_epoch_loss += losses['total'].item()
                batches_processed += 1

            # サンプルごとのメトリクス集計
            # Soft Target モードでは raw_y_batch（全ルートのフラットターゲット）を評価に使用
            for i in range(len(raw_y_batch)):
                ts_len = batch_lens[i]
                strain = batch_strains[i]
                strength_score = batch_strength_scores[i]
                targets_tuples = raw_y_batch[i]  # 評価用: Hard Target (全ルートフラット)
                collection_date = batch_collection_dates[i]

                input_str = batch_input_strs[i]
                target_str = batch_target_strs[i]
                full_path = batch_full_paths[i]

                total_samples += 1

                # 流行度フィルタの適用
                if config.USE_EVAL_STRENGTH_FILTER and strength_score < config.EVAL_STRENGTH_MIN:
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

                # Top-1予測の確率を抽出
                top1_reg_idx = topk_indices_region[i, 0] if topk_indices_region.size(1) > 0 else 0
                pred_prob_region = probs_region[i, top1_reg_idx].item()

                top1_pos_idx = topk_indices_position[i, 0] if topk_indices_position.size(1) > 0 else 0
                pred_prob_position = probs_position[i, top1_pos_idx].item()

                top1_aa_idx = topk_indices_aa_pos[i, 0] if topk_indices_aa_pos.size(1) > 0 else 0
                pred_prob_aa_pos = probs_aa_pos[i, top1_aa_idx].item()

                hit_region     = len(pred_set_region.intersection(target_set_region)) > 0
                hit_position   = len(pred_set_position.intersection(target_set_position)) > 0
                hit_aa_pos     = len(pred_set_aa_pos.intersection(target_set_aa_pos)) > 0
                hit_codon_pos  = len(pred_set_codon_pos.intersection(target_set_codon_pos)) > 0
                hit_synonymous = len(pred_set_synonymous.intersection(target_set_synonymous)) > 0

                pred_strength = predictions_strength[i].item()

                detailed_results.append({
                    'len': ts_len, 'strain': strain,
                    'strength_score': strength_score, 'pred_strength': pred_strength,
                    'collection_date': collection_date,
                    'raw_path': full_path,
                    'targets_region': target_set_region,   'preds_region': pred_set_region, 'pred_prob_region': pred_prob_region,  'hit_region': hit_region,
                    'targets_position': target_set_position,'preds_position': pred_set_position, 'pred_prob_position': pred_prob_position, 'hit_position': hit_position,
                    'targets_aa_pos': target_set_aa_pos,   'preds_aa_pos': pred_set_aa_pos, 'pred_prob_aa_pos': pred_prob_aa_pos,  'hit_aa_pos': hit_aa_pos,
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

                # ECE 用データの蓄積 (SAVE_ECE=True の場合のみ)
                if getattr(config, 'SAVE_ECE', True):
                    hit_reg_int = int(len(pred_set_region.intersection(target_set_region)) > 0)
                    hit_pos_int = int(len(pred_set_position.intersection(target_set_position)) > 0)
                    hit_aa_int  = int(len(pred_set_aa_pos.intersection(target_set_aa_pos)) > 0)
                    results_for_ece[ts_len]['probs_region'].append(pred_prob_region)
                    results_for_ece[ts_len]['hits_region'].append(hit_reg_int)
                    results_for_ece[ts_len]['probs_position'].append(pred_prob_position)
                    results_for_ece[ts_len]['hits_position'].append(hit_pos_int)
                    results_for_ece[ts_len]['probs_aa_pos'].append(pred_prob_aa_pos)
                    results_for_ece[ts_len]['hits_aa_pos'].append(hit_aa_int)

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

                # 月別集計 (EVAL_X_AXIS == 'date' のときのみ)
                if eval_x_axis == 'date':
                    raw_date = collection_date if collection_date else ''
                    yearmonth = raw_date[:7] if len(raw_date) >= 7 else 'unknown'
                    results_by_yearmonth[yearmonth]["preds_region"].append(pred_set_region)
                    results_by_yearmonth[yearmonth]["targets_region"].append(target_set_region)
                    results_by_yearmonth[yearmonth]["preds_position"].append(pred_set_position)
                    results_by_yearmonth[yearmonth]["targets_position"].append(target_set_position)
                    results_by_yearmonth[yearmonth]["preds_aa_pos"].append(pred_set_aa_pos)
                    results_by_yearmonth[yearmonth]["targets_aa_pos"].append(target_set_aa_pos)
                    results_by_yearmonth[yearmonth]["preds_codon_pos"].append(pred_set_codon_pos)
                    results_by_yearmonth[yearmonth]["targets_codon_pos"].append(target_set_codon_pos)
                    results_by_yearmonth[yearmonth]["preds_synonymous"].append(pred_set_synonymous)
                    results_by_yearmonth[yearmonth]["targets_synonymous"].append(target_set_synonymous)
                    results_by_yearmonth[yearmonth]["preds_strength"].append(pred_strength)
                    results_by_yearmonth[yearmonth]["targets_strength"].append(strength_score)

    # 流行度フィルタの結果表示
    if config.USE_EVAL_STRENGTH_FILTER and config.EVAL_STRENGTH_MIN > 0:
        print(f"[INFO] Strength filter: {filtered_samples}/{total_samples} samples excluded "
              f"(threshold={config.EVAL_STRENGTH_MIN:.2f})")

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

        # ECE (Expected Calibration Error) 計算 (SAVE_ECE=True の場合のみ)
        if getattr(config, 'SAVE_ECE', True) and ts_len in results_for_ece:
            n_bins = getattr(config, 'ECE_N_BINS', 10)

            def _ece(probs, hits):
                """確信度 (probs) と結果 (hits: 0/1) から ECE を計算する。
                Equal-Width Binning: [0, 1] を n_bins 等分割し、各ビンの
                |avg_confidence - avg_accuracy| の加重平均。
                """
                if not probs:
                    return 0.0
                import math
                bin_boundaries = [i / n_bins for i in range(n_bins + 1)]
                ece_sum = 0.0
                n = len(probs)
                for b in range(n_bins):
                    lo, hi = bin_boundaries[b], bin_boundaries[b + 1]
                    in_bin = [(p, h) for p, h in zip(probs, hits) if lo <= p < hi]
                    if b == n_bins - 1:  # 最後のビンの上限を含む
                        in_bin = [(p, h) for p, h in zip(probs, hits) if lo <= p <= hi]
                    if not in_bin:
                        continue
                    avg_conf = sum(p for p, _ in in_bin) / len(in_bin)
                    avg_acc  = sum(h for _, h in in_bin) / len(in_bin)
                    ece_sum += (len(in_bin) / n) * abs(avg_conf - avg_acc)
                return ece_sum

            ece_data = results_for_ece[ts_len]
            final_metrics_by_ts[ts_len]['ece_region']   = _ece(
                ece_data['probs_region'],   ece_data['hits_region']
            )
            final_metrics_by_ts[ts_len]['ece_position']  = _ece(
                ece_data['probs_position'], ece_data['hits_position']
            )
            final_metrics_by_ts[ts_len]['ece_aa_pos']    = _ece(
                ece_data['probs_aa_pos'],   ece_data['hits_aa_pos']
            )

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

    # 月別メトリクス計算 (EVAL_X_AXIS == 'date' のときのみ)
    final_metrics_by_ym = {}
    if eval_x_axis == 'date':
        for ym, data in results_by_yearmonth.items():
            hit_reg, prec_reg, recall_reg, f1_reg = calculate_metrics(data["preds_region"], data["targets_region"])
            hit_pos, prec_pos, recall_pos, f1_pos = calculate_metrics(data["preds_position"], data["targets_position"])
            hit_aa, prec_aa, recall_aa, f1_aa = calculate_metrics(data["preds_aa_pos"], data["targets_aa_pos"])
            hit_codon, _, _, _ = calculate_metrics(data["preds_codon_pos"], data["targets_codon_pos"])
            hit_syn, _, _, _ = calculate_metrics(data["preds_synonymous"], data["targets_synonymous"])
            final_metrics_by_ym[ym] = {
                "num_samples": len(data["preds_region"]),
                "region_hit_rate": hit_reg,
                "region_precision": prec_reg,
                "region_recall": recall_reg,
                "region_f1": f1_reg,
                "position_hit_rate": hit_pos,
                "position_precision": prec_pos,
                "position_recall": recall_pos,
                "position_f1": f1_pos,
                "aa_pos_hit_rate": hit_aa,
                "aa_pos_precision": prec_aa,
                "aa_pos_recall": recall_aa,
                "aa_pos_f1": f1_aa,
                "codon_pos_hit_rate": hit_codon,
                "synonymous_hit_rate": hit_syn,
                "strength_mae": (
                    sum(abs(p - t) for p, t in zip(data["preds_strength"], data["targets_strength"])) /
                    len(data["preds_strength"]) if data["preds_strength"] else 0.0
                )
            }

    return avg_loss, final_metrics_by_ts, detailed_results, metrics_by_category, final_metrics_by_ym


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
        for (x_cat, x_num, mask), _y_batch, _, _, _, _, _, _, raw_y_batch, *extra in dataloader:
            x_cat = x_cat.to(config.DEVICE)
            x_num = x_num.to(config.DEVICE)
            mask  = mask.to(config.DEVICE)

            (pred_region, pred_position, pred_aa_pos,
             _, pred_codon_pos, pred_synonymous) = (
                _predict_tta(model, x_cat, x_num, mask, getattr(config, 'TTA_N_PASSES', 5))
                if getattr(config, 'USE_TTA', False)
                else model(x_cat, x_num, src_key_padding_mask=mask)
            )

            preds_all = {
                'region':    pred_region,
                'position':  pred_position,
                'aa_pos':    pred_aa_pos,
                'codon_pos': pred_codon_pos,
                'synonymous': pred_synonymous,
            }
            target_idx = {'region': 0, 'position': 1, 'aa_pos': 2, 'codon_pos': 3, 'synonymous': 4}

            for i, targets_tuples in enumerate(raw_y_batch):
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
