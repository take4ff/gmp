# petra/evaluate.py — PETRA 準拠 Recall@K 評価（per-sequence macro-average + Weighted）
#
# PETRA 原論文/リポジトリ (petra/PETra/model_training/eval_mutgpt.py) を精読して確認した集計式:
#   各配列(サンプル)について、その配列内の変異トークン位置での recall_i@K
#   （hits_i@K / n_targets_i）を先に計算し、
#     Average Recall@K = 配列間の単純平均（macro-average）
#     Weighted Recall@K = representativeness 重み付き平均
#       weight = proc(population(country) / seq_count(country, yymm))
#   をそれぞれ報告する。トークン全体の Σhits/Σtotal（micro-average）ではない点に注意
#   （旧実装はこの誤りを含んでいたため修正済み。本体側 scripts/eval/petra_recall.py と同じ集計方式）。

from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate_recall_topk(model, loader: DataLoader, tokenizer,
                          device: torch.device, top_k_list: list[int],
                          weights=None) -> dict:
    """PETRA 準拠 Recall@K 評価（per-sequence macro-average）。

    各配列（サンプル）内の変異トークン位置に対する recall を配列単位で計算し、
    配列間で単純平均（Average）・representativeness 重み付き平均（Weighted）を報告する。
    コンテキストトークン（BOS/EOS/PAD/年・月・国・クレード）は評価対象外。

    Args:
        weights: PetraRepresentativenessWeights インスタンス（None なら Weighted=Average）

    Returns:
        dict: {'average': {'recall@K':...}, 'weighted': {...}, 'n_sequences': int}
    """
    model.eval()
    max_k = max(top_k_list)

    context_ids = set(
        i for i in range(tokenizer.vocab_size) if tokenizer.is_context_token(i)
    )
    context_tensor = torch.tensor(sorted(context_ids), dtype=torch.long, device=device)

    per_seq_recall = {k: [] for k in top_k_list}
    per_seq_weight = []

    for input_ids, target_ids, _mask, countries, dates in loader:
        input_ids = input_ids.to(device)
        target_ids_dev = target_ids.to(device)

        logits = model(input_ids)                                    # B, T, V
        topk_preds = logits.topk(max_k, dim=-1).indices               # B, T, max_k

        is_context = torch.isin(target_ids_dev, context_tensor)
        is_valid = (~is_context) & (target_ids_dev != -100)           # B, T

        correct_at = {
            k: (topk_preds[..., :k] == target_ids_dev.unsqueeze(-1)).any(-1) & is_valid
            for k in top_k_list
        }

        B = input_ids.shape[0]
        for b in range(B):
            n_targets = int(is_valid[b].sum().item())
            if n_targets == 0:
                continue  # ターゲット無し配列はスキップ（PETRA と同様 nummut>0 のみ集計）
            w = 1.0
            if weights is not None:
                w = weights.weight_for(countries[b], dates[b])
            per_seq_weight.append(w)
            for k in top_k_list:
                hits = int(correct_at[k][b].sum().item())
                per_seq_recall[k].append(hits / n_targets)

    n_seq = len(per_seq_weight)
    result = {'n_sequences': n_seq, 'average': {}, 'weighted': {}}
    w_arr = np.array(per_seq_weight) if n_seq else np.array([])
    for k in top_k_list:
        vals = np.array(per_seq_recall[k])
        result['average'][f'recall@{k}'] = float(vals.mean()) if n_seq else 0.0
        result['weighted'][f'recall@{k}'] = (
            float((w_arr * vals).sum() / w_arr.sum()) if n_seq and w_arr.sum() > 0 else 0.0
        )
    return result


class PetraRepresentativenessWeights:
    """PETRA の representativeness 重み（transformer_260702/scripts/eval/petra_weights.py を再利用）。

    weight(country, yymm) = proc(population(country) / seq_count(country, yymm))
    country/yymm が不明な場合は 1.0（中立、Average と同一）にフォールバックする。
    """

    def __init__(self, db_path: str, split_type: int):
        from transformer_260702.scripts.eval.petra_weights import (
            parse_country_population, get_strain_country_map,
            compute_representativeness_weight, _to_yymm,
        )
        self._compute = compute_representativeness_weight
        self._to_yymm = _to_yymm
        self.pop_dic = parse_country_population()
        # petra/dataset.py は country を直接サンプルから取得済みなので strain マップは不要だが、
        # 国・月別の配列数 count_dic は必要なため、既存ヘルパで DB から一括取得する。
        _, self.count_dic = get_strain_country_map(db_path, split_type)

    def weight_for(self, country, date_str):
        if not country:
            return 1.0
        yymm = self._to_yymm(date_str)
        if not yymm:
            return 1.0
        return self._compute(country, yymm, self.pop_dic, self.count_dic)
