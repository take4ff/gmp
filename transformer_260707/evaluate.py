# --- evaluate.py ---
import torch
from collections import defaultdict
from .utils.logging import calculate_metrics, calculate_weighted_macro_recall
from .utils.losses import compute_task_losses
from . import config


def _predict_tta(model, x_cat, x_num, mask, n_passes, clade_ids=None):
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
            out = model(x_cat, x_num, src_key_padding_mask=mask, clade_ids=clade_ids)
            # 8-tuple に対応: base_after / aa_after が None の場合はそのまま保持
            probs = tuple(
                torch.softmax(o, dim=-1) if (o is not None and o.dim() > 1)
                else o
                for o in out
            )
            all_logits.append(probs)
    model.eval()  # 推論後は必ず eval に戻す

    # タスクごとに n_passes 平均を返す（None は None のまま）
    n_tasks = len(all_logits[0])
    averaged = []
    for task_idx in range(n_tasks):
        samples = [p[task_idx] for p in all_logits]
        if samples[0] is None:
            averaged.append(None)
        else:
            stacked = torch.stack(samples, dim=0)
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

    # 提案3: 階層的予測（Region→Position マスキング, Option A）用の写像を用意。
    # OFF 時は None のままでマスキングは一切行わない（現行挙動と一致）。
    hier_on = getattr(config, 'USE_HIERARCHICAL_PREDICTION', False)
    pos_region_map = None
    if hier_on:
        from .db.queries import get_position_region_map
        pos_region_map = get_position_region_map().to(config.DEVICE)  # [VOCAB_SIZE_POSITION]
    hier_topk_regions = max(1, int(getattr(config, 'HIERARCHICAL_TOPK_REGIONS', 3)))

    use_clade = getattr(config, 'USE_CLADE_EMBEDDING', False)

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
        "preds_strength": [], "targets_strength": [],
        "hits_base_after": [], "hits_aa_after": [],   # USE_SUBSTITUTION_HEAD 用
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

    def _push(acc, sets):
        """1サンプル分の pred/target 集合を任意のアキュムレータへ追記する。
        timestep / category / yearmonth の3系統で同じ append を繰り返していたのを集約。
        sets の並び: (reg_pred, reg_tgt, pos_pred, pos_tgt, aa_pred, aa_tgt,
                      cod_pred, cod_tgt, syn_pred, syn_tgt, strength_pred, strength_tgt)
        """
        (pr_reg, tg_reg, pr_pos, tg_pos, pr_aa, tg_aa,
         pr_cod, tg_cod, pr_syn, tg_syn, pr_str, tg_str) = sets
        acc["preds_region"].append(pr_reg);      acc["targets_region"].append(tg_reg)
        acc["preds_position"].append(pr_pos);    acc["targets_position"].append(tg_pos)
        acc["preds_aa_pos"].append(pr_aa);       acc["targets_aa_pos"].append(tg_aa)
        acc["preds_codon_pos"].append(pr_cod);   acc["targets_codon_pos"].append(tg_cod)
        acc["preds_synonymous"].append(pr_syn);  acc["targets_synonymous"].append(tg_syn)
        acc["preds_strength"].append(pr_str);    acc["targets_strength"].append(tg_str)

    example_printed = False

    with torch.no_grad():
        for (x_cat, x_num, mask), y_batch, batch_lens, batch_strains, batch_strength_scores, batch_input_strs, batch_target_strs, batch_full_paths, raw_y_batch, batch_collection_dates in dataloader:

            x_cat = x_cat.to(config.DEVICE)
            x_num = x_num.to(config.DEVICE)
            mask = mask.to(config.DEVICE)

            # 提案4: 主要系統クレード埋め込み（OFF 時は None → モデル側で無視）
            clade_ids = None
            if use_clade:
                from .utils.clade import clade_ids_tensor
                clade_ids = clade_ids_tensor(batch_strains, device=config.DEVICE)

            # モデル出力: 6 or 8 タプル (TTA 時は softmax 平均値を返す)
            if getattr(config, 'USE_TTA', False):
                n_passes = getattr(config, 'TTA_N_PASSES', 5)
                predictions_tuple = _predict_tta(model, x_cat, x_num, mask, n_passes, clade_ids=clade_ids)
            else:
                predictions_tuple = model(x_cat, x_num, src_key_padding_mask=mask, clade_ids=clade_ids)
            # 8-tuple への拡張に対応
            (predictions_region, predictions_position, predictions_aa_pos,
             predictions_strength, predictions_codon_pos, predictions_synonymous,
             *_extra_preds) = predictions_tuple

            use_sub = getattr(config, 'USE_SUBSTITUTION_HEAD', False)
            predictions_base_after = _extra_preds[0] if use_sub and len(_extra_preds) > 0 else None
            predictions_aa_after   = _extra_preds[1] if use_sub and len(_extra_preds) > 1 else None

            # Top-K 推論
            def safe_topk(tensor, k):
                """kがクラス数を超えないように調整してtopkを取得"""
                actual_k = min(k, tensor.size(1))
                return torch.topk(tensor, actual_k, dim=1).indices

            # --- 提案3: 階層的予測（Region→Position マスキング, Option A）---
            # 予測 Region 上位 hier_topk_regions 個に属さない位置を Position から除外する。
            # 損失計算には影響させないため predictions_position 自体は変更しない。
            # position_pred_scores: topk/確率の算出に使う（マスク後の）Position スコア。
            position_pred_scores = predictions_position
            allowed_position_mask = None  # [B, VOCAB] bool（None=マスク無し）
            if pos_region_map is not None:
                R = min(hier_topk_regions, predictions_region.size(1))
                topR_regions = torch.topk(predictions_region, R, dim=1).indices        # [B, R]
                pr = pos_region_map.unsqueeze(0)                                        # [1, VOCAB]
                allowed_position_mask = torch.zeros_like(
                    predictions_position, dtype=torch.bool)                            # [B, VOCAB]
                for r in range(R):
                    allowed_position_mask |= (pr == topR_regions[:, r:r + 1])
                # 全位置が除外される行（予測 Region に写像位置が皆無）は nan 回避のため
                # マスクを解除して現行挙動へフォールバックする。
                no_allowed = ~allowed_position_mask.any(dim=1, keepdim=True)
                allowed_position_mask = allowed_position_mask | no_allowed
                if not getattr(config, 'USE_TTA', False):
                    neg_inf = torch.finfo(predictions_position.dtype).min
                    position_pred_scores = predictions_position.masked_fill(
                        ~allowed_position_mask, neg_inf)

            topk_indices_region    = safe_topk(predictions_region,    config.TOP_K_EVAL)
            topk_indices_position  = safe_topk(position_pred_scores,  config.TOP_K_EVAL)
            topk_indices_aa_pos    = safe_topk(predictions_aa_pos,    config.TOP_K_EVAL)
            topk_indices_codon_pos = safe_topk(predictions_codon_pos, config.TOP_K_EVAL)
            topk_indices_synonymous= safe_topk(predictions_synonymous,config.TOP_K_EVAL)

            # 確率の計算 (TTA時はすでにsoftmax済みのため再適用しない)
            if getattr(config, 'USE_TTA', False):
                probs_region   = predictions_region
                probs_position = predictions_position
                probs_aa_pos   = predictions_aa_pos
                if allowed_position_mask is not None:
                    # TTA は確率が入っているため、除外位置を 0 にして再正規化する。
                    masked = probs_position * allowed_position_mask.float()
                    probs_position = masked / masked.sum(dim=1, keepdim=True).clamp(min=1e-12)
            else:
                probs_region   = torch.nn.functional.softmax(predictions_region,   dim=-1)
                probs_position = torch.nn.functional.softmax(position_pred_scores, dim=-1)
                probs_aa_pos   = torch.nn.functional.softmax(predictions_aa_pos,   dim=-1)

            # --- ホールドアウト NLL 用: タスク別の対数予測分布 ---
            # NLL は「真の変異集合上の一様分布（config 不変の参照ターゲット）」と
            # 「モデルの全クラス予測分布」の交差エントロピーとして per-sample で算出する。
            # 学習損失関数（focal/CB 等）に依存しない素の log-softmax を使うことで、
            # 構成間で公平に比較できる予測分布の質の指標になる。
            # Position は #3 の階層マスク前の生ロジット（predictions_position）を使い、
            # デコード戦略ではなくモデル分布そのものを評価する。TTA 時は既に確率のため log を取る。
            _tta_on = getattr(config, 'USE_TTA', False)
            def _logp(pred):
                if _tta_on:
                    return torch.log(pred.clamp(min=1e-12))
                return torch.nn.functional.log_softmax(pred, dim=-1)
            logp_region     = _logp(predictions_region)
            logp_position   = _logp(predictions_position)
            logp_aa_pos     = _logp(predictions_aa_pos)
            logp_codon_pos  = _logp(predictions_codon_pos)
            logp_synonymous = _logp(predictions_synonymous)

            def _sample_nll(logp_row, target_idx_set, vocab):
                """NLL = -(1/|T|) Σ_{t∈T} log p(t)。ターゲットが無ければ None。"""
                idxs = [j for j in target_idx_set if 0 <= j < vocab]
                if not idxs:
                    return None
                return float(-logp_row[idxs].mean().item())

            # 損失計算 (utils/losses.py 共通ロジック)
            losses = compute_task_losses(
                predictions_tuple, y_batch, loss_fn, batch_strength_scores,
                raw_y_batch=raw_y_batch,
            )
            if losses is not None:
                total_epoch_loss += losses['total'].item()
                batches_processed += 1

            # --- バッチ単位で GPU→CPU 転送を1回に集約（旧: サンプルごとに .cpu()/.item()）---
            # topk インデックス・Top-1 確率・回帰値・置換 argmax・NLL 用 log 確率を
            # ここで一括転送し、以降の per-sample ループは Python 側だけで回す。
            tk_region_l   = topk_indices_region.cpu().tolist()
            tk_position_l = topk_indices_position.cpu().tolist()
            tk_aa_pos_l   = topk_indices_aa_pos.cpu().tolist()
            tk_codon_l    = topk_indices_codon_pos.cpu().tolist()
            tk_syn_l      = topk_indices_synonymous.cpu().tolist()
            # Top-1 予測確率（top1 index = スコア argmax = probs の最大値なので max で取得）
            p_top1_region_l   = probs_region.max(dim=1).values.cpu().tolist()
            p_top1_position_l = probs_position.max(dim=1).values.cpu().tolist()
            p_top1_aa_pos_l   = probs_aa_pos.max(dim=1).values.cpu().tolist()
            strength_l        = predictions_strength.detach().cpu().tolist()
            if predictions_base_after is not None and predictions_aa_after is not None:
                pred_base_l = torch.argmax(predictions_base_after, dim=1).cpu().tolist()
                pred_aa_l   = torch.argmax(predictions_aa_after,   dim=1).cpu().tolist()
            else:
                pred_base_l = pred_aa_l = None
            # NLL 用 log 確率を CPU に移し、per-sample gather の GPU 同期を排除
            logp_region_c     = logp_region.cpu()
            logp_position_c   = logp_position.cpu()
            logp_aa_pos_c     = logp_aa_pos.cpu()
            logp_codon_pos_c  = logp_codon_pos.cpu()
            logp_synonymous_c = logp_synonymous.cpu()

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

                pred_set_region    = set(tk_region_l[i])
                target_set_region  = set(t[0] for t in targets_tuples)
                pred_set_position  = set(tk_position_l[i])
                target_set_position= set(t[1] for t in targets_tuples)
                pred_set_aa_pos    = set(tk_aa_pos_l[i])
                target_set_aa_pos  = set(t[2] for t in targets_tuples)
                pred_set_codon_pos = set(tk_codon_l[i])
                target_set_codon_pos= set(t[3] for t in targets_tuples)
                pred_set_synonymous= set(tk_syn_l[i])
                target_set_synonymous= set(t[4] for t in targets_tuples)

                # Top-1予測の確率（バッチ単位で precompute 済み）
                pred_prob_region   = p_top1_region_l[i]
                pred_prob_position = p_top1_position_l[i]
                pred_prob_aa_pos   = p_top1_aa_pos_l[i]

                hit_region     = len(pred_set_region.intersection(target_set_region)) > 0
                hit_position   = len(pred_set_position.intersection(target_set_position)) > 0
                hit_aa_pos     = len(pred_set_aa_pos.intersection(target_set_aa_pos)) > 0
                hit_codon_pos  = len(pred_set_codon_pos.intersection(target_set_codon_pos)) > 0
                hit_synonymous = len(pred_set_synonymous.intersection(target_set_synonymous)) > 0

                pred_strength = strength_l[i]

                # base_after / aa_after Top-1 ヒット判定
                hit_base_after = hit_aa_after = None
                if pred_base_l is not None and pred_aa_l is not None:
                    pred_base = pred_base_l[i]
                    pred_aa   = pred_aa_l[i]
                    target_base_set = set(t[5] for t in targets_tuples if len(t) > 5)
                    target_aa_set   = set(t[6] for t in targets_tuples if len(t) > 6)
                    hit_base_after = pred_base in target_base_set
                    hit_aa_after   = pred_aa   in target_aa_set

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
                    'hit_base_after': hit_base_after,
                    'hit_aa_after': hit_aa_after,
                    # ホールドアウト NLL（nats, タスク別）。ターゲット不在タスクは None。
                    'nll_region':     _sample_nll(logp_region_c[i],     target_set_region,     config.NUM_REGIONS),
                    'nll_position':   _sample_nll(logp_position_c[i],   target_set_position,   config.VOCAB_SIZE_POSITION),
                    'nll_aa_pos':     _sample_nll(logp_aa_pos_c[i],     target_set_aa_pos,     config.VOCAB_SIZE_AA_POS),
                    'nll_codon_pos':  _sample_nll(logp_codon_pos_c[i],  target_set_codon_pos,  config.VOCAB_SIZE_CODON_POS),
                    'nll_synonymous': _sample_nll(logp_synonymous_c[i], target_set_synonymous, config.VOCAB_SIZE_SYNONYMOUS),
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
                    if hit_base_after is not None:
                        print(f"  Base After Hit: {hit_base_after}, AA After Hit: {hit_aa_after}")
                    print("--------------------------")
                    example_printed = True

                # 3系統（timestep / category / yearmonth）へ同一サンプルを追記
                _sample_sets = (
                    pred_set_region,    target_set_region,
                    pred_set_position,  target_set_position,
                    pred_set_aa_pos,    target_set_aa_pos,
                    pred_set_codon_pos, target_set_codon_pos,
                    pred_set_synonymous,target_set_synonymous,
                    pred_strength,      strength_score,
                )

                # タイムステップ別の集計
                _push(results_by_timestep[ts_len], _sample_sets)
                if hit_base_after is not None:
                    results_by_timestep[ts_len]["hits_base_after"].append(hit_base_after)
                    results_by_timestep[ts_len]["hits_aa_after"].append(hit_aa_after)

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
                _push(results_by_ts_and_category[ts_len][category], _sample_sets)

                # 月別集計（常に実行）
                raw_date = collection_date if collection_date else ''
                yearmonth = raw_date[:7] if len(raw_date) >= 7 else 'unknown'
                _push(results_by_yearmonth[yearmonth], _sample_sets)

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
            ),
            "base_after_accuracy": (
                sum(data["hits_base_after"]) / len(data["hits_base_after"])
                if data["hits_base_after"] else None
            ),
            "aa_after_accuracy": (
                sum(data["hits_aa_after"]) / len(data["hits_aa_after"])
                if data["hits_aa_after"] else None
            ),
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

    # キャリブレーション bin データの集計（全タイムステップを結合）
    calib_bins = {}
    if getattr(config, 'SAVE_ECE', False):
        n_bins = getattr(config, 'ECE_N_BINS', 10)
        for task in ('region', 'position', 'aa_pos'):
            all_probs, all_hits = [], []
            for ece_d in results_for_ece.values():
                all_probs.extend(ece_d[f'probs_{task}'])
                all_hits.extend(ece_d[f'hits_{task}'])
            if not all_probs:
                continue
            n = len(all_probs)
            bins = []
            for b in range(n_bins):
                lo, hi = b / n_bins, (b + 1) / n_bins
                in_bin = [(p, h) for p, h in zip(all_probs, all_hits)
                          if lo <= p < hi or (b == n_bins - 1 and p == hi)]
                count = len(in_bin)
                bins.append({
                    'bin': b + 1,
                    'lower': round(lo, 4),
                    'upper': round(hi, 4),
                    'avg_confidence': round(sum(p for p, _ in in_bin) / count, 4) if count else None,
                    'avg_accuracy':   round(sum(h for _, h in in_bin) / count, 4) if count else None,
                    'count': count,
                })
            calib_bins[task] = bins

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

    # 月別メトリクス計算（常に実行）
    final_metrics_by_ym = {}
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

    return avg_loss, final_metrics_by_ts, detailed_results, metrics_by_category, final_metrics_by_ym, calib_bins


def evaluate_topk(model, dataloader, ks=(1, 3, 5), position_tolerances=None):
    """複数の K 値で Top-K Precision / Recall を同時に計算する。

    USE_R_PRECISION=True の場合、各サンプルの K=len(target_set) で計算した
    R-Precision も 'r_precision' キーとして結果に追加する。

    Args:
        model: 評価対象モデル
        dataloader: DataLoader
        ks: 評価する K の値のタプル (例: (1, 3, 5))
        position_tolerances: 指定すると 'position' タスクについて、Top-1 予測位置が
            正解位置（共起の場合は最も近いもの）から±tolerance以内なら正解とみなす
            許容誤差付き Top-1 正解率を追加計算する（例: (0, 5, 10, 50)）。
            tolerance=0 は「予測位置 == 正解位置」の完全一致（既存の position[1]['hit_rate']
            と同一の判定基準）。

    Returns:
        dict: {
            task_name: {
                k: {'precision': float, 'recall': float, 'hit_rate': float},
                ...,
                'r_precision': {'precision': float, 'recall': float, 'hit_rate': float}  # USE_R_PRECISION=True のみ
            },
            'position_tolerance': {tolerance: {'hit_rate': float, 'n': int}, ...}  # position_tolerances 指定時のみ
        }
        task_name は 'region' / 'position' / 'aa_pos' / 'codon_pos' / 'synonymous'
    """
    model.eval()

    use_r_precision = getattr(config, 'USE_R_PRECISION', False)

    tasks = ['region', 'position', 'aa_pos', 'codon_pos', 'synonymous']
    # task -> k -> {'tp': int, 'fp': int, 'total_targets': int, 'hits': int, 'n': int}
    stats = {t: {k: {'tp': 0, 'fp': 0, 'total_targets': 0, 'hits': 0, 'n': 0} for k in ks} for t in tasks}

    # R-Precision 用統計（動的 K = len(target_set)）
    if use_r_precision:
        rp_stats = {t: {'tp': 0, 'fp': 0, 'total_targets': 0, 'hits': 0, 'n': 0} for t in tasks}

    # 位置の許容誤差付き Top-1 正解率用統計
    if position_tolerances:
        pos_tol_stats = {t: {'hits': 0, 'n': 0} for t in position_tolerances}

    def _topk_preds(tensor, k):
        actual_k = min(k, tensor.size(1))
        return torch.topk(tensor, actual_k, dim=1).indices

    with torch.no_grad():
        for (x_cat, x_num, mask), _y_batch, _, _, _, _, _, _, raw_y_batch, *extra in dataloader:
            x_cat = x_cat.to(config.DEVICE)
            x_num = x_num.to(config.DEVICE)
            mask  = mask.to(config.DEVICE)

            out_tuple = (
                _predict_tta(model, x_cat, x_num, mask, getattr(config, 'TTA_N_PASSES', 5))
                if getattr(config, 'USE_TTA', False)
                else model(x_cat, x_num, src_key_padding_mask=mask)
            )
            # 8-tuple への拡張に対応
            (pred_region, pred_position, pred_aa_pos,
             _, pred_codon_pos, pred_synonymous, *_extra) = out_tuple

            preds_all = {
                'region':    pred_region,
                'position':  pred_position,
                'aa_pos':    pred_aa_pos,
                'codon_pos': pred_codon_pos,
                'synonymous': pred_synonymous,
            }
            target_idx = {'region': 0, 'position': 1, 'aa_pos': 2, 'codon_pos': 3, 'synonymous': 4}

            # 必要な最大 k（標準 ks と R-Precision の動的 k を包含）を先に決める。
            max_r_k = 0
            if use_r_precision:
                for targets_tuples in raw_y_batch:
                    if not targets_tuples:
                        continue
                    for task in tasks:
                        n_t = len(set(t[target_idx[task]] for t in targets_tuples))
                        if n_t > max_r_k:
                            max_r_k = n_t
            need_k = max(max(ks), max_r_k, 1)

            # 各タスクの topk を「バッチ×1回」だけ計算して CPU list 化する。
            # 旧実装は for i / for k の内側で毎回バッチ全体の topk を取り直していた。
            topk_cpu = {task: _topk_preds(preds_all[task], need_k).cpu().tolist()
                        for task in tasks}

            for i, targets_tuples in enumerate(raw_y_batch):
                if not targets_tuples:
                    continue
                for task in tasks:
                    t_set = set(t[target_idx[task]] for t in targets_tuples)
                    row = topk_cpu[task][i]   # スコア降順の topk インデックス（need_k 個）
                    for k in ks:
                        pred_set = set(row[:k])
                        tp = len(pred_set & t_set)
                        stats[task][k]['tp']            += tp
                        stats[task][k]['fp']            += len(pred_set) - tp
                        stats[task][k]['total_targets'] += len(t_set)
                        stats[task][k]['hits']          += int(tp > 0)
                        stats[task][k]['n']             += 1

                    # R-Precision: K = len(target_set) を動的に決定（同じ row をスライス）
                    if use_r_precision:
                        r_k = max(1, len(t_set))
                        pred_set_r = set(row[:r_k])
                        tp_r = len(pred_set_r & t_set)
                        rp_stats[task]['tp']            += tp_r
                        rp_stats[task]['fp']            += len(pred_set_r) - tp_r
                        rp_stats[task]['total_targets'] += len(t_set)
                        rp_stats[task]['hits']          += int(tp_r > 0)
                        rp_stats[task]['n']             += 1

                    # 位置の許容誤差付き Top-1 正解率（'position' タスクのみ）
                    if position_tolerances and task == 'position':
                        pred_top1 = row[0]
                        min_dist = min(abs(pred_top1 - tp) for tp in t_set)
                        for tol in position_tolerances:
                            pos_tol_stats[tol]['n']    += 1
                            pos_tol_stats[tol]['hits'] += int(min_dist <= tol)

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

        # R-Precision の追加
        if use_r_precision:
            s = rp_stats[task]
            total_pred  = s['tp'] + s['fp']
            precision   = (s['tp'] / total_pred  * 100) if total_pred  > 0 else 0.0
            recall      = (s['tp'] / s['total_targets'] * 100) if s['total_targets'] > 0 else 0.0
            hit_rate    = (s['hits'] / s['n'] * 100) if s['n'] > 0 else 0.0
            results[task]['r_precision'] = {'precision': precision, 'recall': recall, 'hit_rate': hit_rate}

    if position_tolerances:
        results['position_tolerance'] = {}
        for tol in position_tolerances:
            s = pos_tol_stats[tol]
            hit_rate = (s['hits'] / s['n'] * 100) if s['n'] > 0 else 0.0
            results['position_tolerance'][tol] = {'hit_rate': hit_rate, 'n': s['n']}

    return results
