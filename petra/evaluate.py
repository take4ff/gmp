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

import re
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

# S遺伝子の核酸位置範囲（本体 scripts/eval/petra_recall.py の SPIKE_START/SPIKE_END と同一。
# reference/codon/codon_mutation4.csv で protein=='S' の base_pos 範囲に対応）。
SPIKE_START, SPIKE_END = 21563, 25384
_MUT_TOKEN_RE = re.compile(r'^[ACGT-](\d+)[ACGT-]$')


def build_spike_token_ids(tokenizer) -> set[int]:
    """変異トークン（例 "A23403G"）のうち、位置がスパイク範囲に入るものの token id 集合を返す。"""
    spike_ids = set()
    for tok_id, tok in enumerate(tokenizer.id2token):
        m = _MUT_TOKEN_RE.match(tok)
        if m and SPIKE_START <= int(m.group(1)) <= SPIKE_END:
            spike_ids.add(tok_id)
    return spike_ids


def build_region_token_ids(tokenizer) -> set[int]:
    """[REGION_xxx] トークン（v2の9フィールド化で追加された遺伝子領域トークン）の
    token id 集合を返す。v1（use_region_fields=False）では空集合になる。
    """
    if not getattr(tokenizer, 'use_region_fields', False):
        return set()
    return set(i for i in range(tokenizer.vocab_size) if tokenizer.is_region_token(i))


def build_position_lookup(tokenizer) -> dict:
    """変異トークン（例 "A23403G"）の token id -> ゲノム位置(int) の辞書を返す。

    v1（フラットトークン）専用。BOS/EOS/PAD/UNK/年月/クレード/国等の非変異トークンは
    キーに含めない（position tolerance 評価で target/pred がこれらの場合は None 扱いにする）。
    """
    lookup = {}
    for tok_id, tok in enumerate(tokenizer.id2token):
        m = _MUT_TOKEN_RE.match(tok)
        if m:
            lookup[tok_id] = int(m.group(1))
    return lookup


@torch.no_grad()
def evaluate_recall_topk_tail_only(model, loader: DataLoader, tokenizer,
                                    device: torch.device, top_k_list: list[int],
                                    weights=None, spike_token_ids: set[int] = None,
                                    region_token_ids: set[int] = None,
                                    position_lookup: dict = None,
                                    position_tolerances: list = None) -> dict:
    """変異パス末尾の変異1件のみを評価する Recall@K（提案モデルの position_hit_rate と
    同じ「まだ見ぬ次の変異1件を予測する」タスクに揃えるための版）。

    通常の evaluate_recall_topk は配列内の全変異トークン位置（teacher-forcing による
    既知履歴の再構成）を平均するため、序盤の decisive な founder 変異（ほぼ全サンプル共通）
    を大量に含み精度が大きく底上げされる。本関数は各配列の「最後の有効な変異トークン位置」
    （= 実際にまだ見ぬ次の変異）のみを対象にすることで、提案モデルの評価と対応させる。

    region_token_ids を渡すと（v2の9フィールド化時のみ有効）、末尾の変異イベントに含まれる
    「最後の[REGION_xxx]トークン位置」も別途評価し、提案モデルの region_hit_rate と対応する
    「次の変異の遺伝子領域を当てられるか」を計測する。v1（region_token_idsが空）では計算しない。

    position_lookup + position_tolerances を渡すと、塩基一致を問わない「位置のみ」の
    許容誤差付き Top-1 正解率を追加計算する（build_position_lookup() 参照）。
    tolerance=0 は「予測位置 == 正解位置」（塩基不一致でも正解扱い）で、既存の
    recall@1（トークン完全一致＝位置も塩基も一致）とは判定基準が異なる点に注意。
    tolerance>0 は |予測位置 - 正解位置| <= tolerance を正解とみなす。
    """
    model.eval()
    max_k = max(top_k_list)

    context_ids = set(
        i for i in range(tokenizer.vocab_size) if tokenizer.is_context_token(i)
    )
    context_tensor = torch.tensor(sorted(context_ids), dtype=torch.long, device=device)
    spike_tensor = (torch.tensor(sorted(spike_token_ids), dtype=torch.long, device=device)
                    if spike_token_ids else None)
    region_tensor = (torch.tensor(sorted(region_token_ids), dtype=torch.long, device=device)
                     if region_token_ids else None)

    per_seq_recall = {k: [] for k in top_k_list}
    per_seq_weight = []
    spike_per_seq_recall = {k: [] for k in top_k_list}
    spike_per_seq_weight = []
    region_per_seq_recall = {k: [] for k in top_k_list}
    region_per_seq_weight = []
    # raw_pathが同一foldのtrain窓に既出（is_leaked=True）かどうかで別集計する。
    # PETRAのCLM設計ではraw_path一致でinput/targetの対応も機械的に一致するため、
    # 「本当に汎化しているか」を見るにはleakedを除いたnon_leakedの数値を見る必要がある。
    leaked_per_seq_recall = {k: [] for k in top_k_list}
    leaked_per_seq_weight = []
    nonleaked_per_seq_recall = {k: [] for k in top_k_list}
    nonleaked_per_seq_weight = []

    do_pos_tol = position_lookup is not None and position_tolerances
    if do_pos_tol:
        pos_tol_hit = {t: [] for t in position_tolerances}
        pos_tol_weight = []
        pos_tol_leaked_hit = {t: [] for t in position_tolerances}
        pos_tol_leaked_weight = []
        pos_tol_nonleaked_hit = {t: [] for t in position_tolerances}
        pos_tol_nonleaked_weight = []

    for input_ids, target_ids, _mask, countries, dates, is_leaked_flags in loader:
        input_ids = input_ids.to(device)
        target_ids_dev = target_ids.to(device)

        logits = model(input_ids)                                    # B, T, V
        topk_preds = logits.topk(max_k, dim=-1).indices               # B, T, max_k

        is_context = torch.isin(target_ids_dev, context_tensor)
        is_valid = (~is_context) & (target_ids_dev != -100)           # B, T
        is_region = (torch.isin(target_ids_dev, region_tensor) & is_valid
                     if region_tensor is not None else None)

        B, T = target_ids_dev.shape
        for b in range(B):
            valid_positions = torch.nonzero(is_valid[b], as_tuple=True)[0]
            if len(valid_positions) == 0:
                continue
            last_pos = int(valid_positions[-1].item())  # 配列末尾の変異トークン位置のみ

            w = 1.0
            if weights is not None:
                w = weights.weight_for(countries[b], dates[b])
            per_seq_weight.append(w)

            target = target_ids_dev[b, last_pos]
            hits_by_k = {}
            for k in top_k_list:
                hit = bool((topk_preds[b, last_pos, :k] == target).any().item())
                hits_by_k[k] = hit
                per_seq_recall[k].append(1.0 if hit else 0.0)

            leak_bucket = leaked_per_seq_recall if is_leaked_flags[b] else nonleaked_per_seq_recall
            leak_weight_bucket = leaked_per_seq_weight if is_leaked_flags[b] else nonleaked_per_seq_weight
            leak_weight_bucket.append(w)
            for k in top_k_list:
                leak_bucket[k].append(1.0 if hits_by_k[k] else 0.0)

            # スパイク限定: 末尾の真の変異がスパイク範囲のときのみ集計
            if spike_tensor is not None and bool(torch.isin(target, spike_tensor).item()):
                spike_per_seq_weight.append(w)
                for k in top_k_list:
                    hit = bool((topk_preds[b, last_pos, :k] == target).any().item())
                    spike_per_seq_recall[k].append(1.0 if hit else 0.0)

            # region限定: 末尾の変異イベント内の最後の[REGION_xxx]位置（=次の変異の遺伝子領域）
            if is_region is not None:
                region_positions = torch.nonzero(is_region[b], as_tuple=True)[0]
                if len(region_positions) > 0:
                    r_pos = int(region_positions[-1].item())
                    r_target = target_ids_dev[b, r_pos]
                    region_per_seq_weight.append(w)
                    for k in top_k_list:
                        hit = bool((topk_preds[b, r_pos, :k] == r_target).any().item())
                        region_per_seq_recall[k].append(1.0 if hit else 0.0)

            # 位置のみの許容誤差付き Top-1 正解率（塩基不一致でも位置が近ければ正解扱い）
            if do_pos_tol:
                target_pos = position_lookup.get(int(target.item()))
                if target_pos is not None:
                    pred_tok = int(topk_preds[b, last_pos, 0].item())
                    pred_pos = position_lookup.get(pred_tok)
                    dist = abs(pred_pos - target_pos) if pred_pos is not None else None
                    pos_tol_weight.append(w)
                    leak_pos_hit = pos_tol_leaked_hit if is_leaked_flags[b] else pos_tol_nonleaked_hit
                    leak_pos_weight = pos_tol_leaked_weight if is_leaked_flags[b] else pos_tol_nonleaked_weight
                    leak_pos_weight.append(w)
                    for t in position_tolerances:
                        hit = (dist is not None) and (dist <= t)
                        pos_tol_hit[t].append(1.0 if hit else 0.0)
                        leak_pos_hit[t].append(1.0 if hit else 0.0)

    def _finalize(recalls, weights_list):
        n = len(weights_list)
        out = {'n_sequences': n, 'average': {}, 'weighted': {}}
        w_arr = np.array(weights_list) if n else np.array([])
        for k in top_k_list:
            vals = np.array(recalls[k])
            out['average'][f'recall@{k}'] = float(vals.mean()) if n else 0.0
            out['weighted'][f'recall@{k}'] = (
                float((w_arr * vals).sum() / w_arr.sum()) if n and w_arr.sum() > 0 else 0.0
            )
        return out

    def _finalize_tolerance(hits_by_t, weights_list):
        n = len(weights_list)
        out = {'n_sequences': n, 'average': {}, 'weighted': {}}
        w_arr = np.array(weights_list) if n else np.array([])
        for t in position_tolerances:
            vals = np.array(hits_by_t[t])
            out['average'][f'position_acc@tol{t}'] = float(vals.mean()) if n else 0.0
            out['weighted'][f'position_acc@tol{t}'] = (
                float((w_arr * vals).sum() / w_arr.sum()) if n and w_arr.sum() > 0 else 0.0
            )
        return out

    result = _finalize(per_seq_recall, per_seq_weight)
    if spike_tensor is not None:
        result['spike'] = _finalize(spike_per_seq_recall, spike_per_seq_weight)
    if region_tensor is not None:
        result['region'] = _finalize(region_per_seq_recall, region_per_seq_weight)
    result['leaked'] = _finalize(leaked_per_seq_recall, leaked_per_seq_weight)
    result['non_leaked'] = _finalize(nonleaked_per_seq_recall, nonleaked_per_seq_weight)
    if do_pos_tol:
        result['position_tolerance'] = _finalize_tolerance(pos_tol_hit, pos_tol_weight)
        result['position_tolerance_leaked'] = _finalize_tolerance(pos_tol_leaked_hit, pos_tol_leaked_weight)
        result['position_tolerance_non_leaked'] = _finalize_tolerance(pos_tol_nonleaked_hit, pos_tol_nonleaked_weight)
    return result


@torch.no_grad()
def evaluate_recall_topk(model, loader: DataLoader, tokenizer,
                          device: torch.device, top_k_list: list[int],
                          weights=None, spike_token_ids: set[int] = None,
                          region_token_ids: set[int] = None) -> dict:
    """PETRA 準拠 Recall@K 評価（per-sequence macro-average）。

    各配列（サンプル）内の変異トークン位置に対する recall を配列単位で計算し、
    配列間で単純平均（Average）・representativeness 重み付き平均（Weighted）を報告する。
    コンテキストトークン（BOS/EOS/PAD/年・月・国・クレード）は評価対象外。

    region_token_ids を渡すと（v2の9フィールド化時のみ有効）、[REGION_xxx]トークン位置に
    限定した region 単位の Recall@K も 'region' キーで追加報告する。

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
    spike_tensor = (torch.tensor(sorted(spike_token_ids), dtype=torch.long, device=device)
                    if spike_token_ids else None)
    region_tensor = (torch.tensor(sorted(region_token_ids), dtype=torch.long, device=device)
                     if region_token_ids else None)

    per_seq_recall = {k: [] for k in top_k_list}
    per_seq_weight = []
    spike_per_seq_recall = {k: [] for k in top_k_list}
    spike_per_seq_weight = []
    region_per_seq_recall = {k: [] for k in top_k_list}
    region_per_seq_weight = []
    leaked_per_seq_recall = {k: [] for k in top_k_list}
    leaked_per_seq_weight = []
    nonleaked_per_seq_recall = {k: [] for k in top_k_list}
    nonleaked_per_seq_weight = []

    for input_ids, target_ids, _mask, countries, dates, is_leaked_flags in loader:
        input_ids = input_ids.to(device)
        target_ids_dev = target_ids.to(device)

        logits = model(input_ids)                                    # B, T, V
        topk_preds = logits.topk(max_k, dim=-1).indices               # B, T, max_k

        is_context = torch.isin(target_ids_dev, context_tensor)
        is_valid = (~is_context) & (target_ids_dev != -100)           # B, T
        is_spike = (torch.isin(target_ids_dev, spike_tensor) & is_valid
                    if spike_tensor is not None else None)
        is_region = (torch.isin(target_ids_dev, region_tensor) & is_valid
                     if region_tensor is not None else None)

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

            leak_bucket = leaked_per_seq_recall if is_leaked_flags[b] else nonleaked_per_seq_recall
            leak_weight_bucket = leaked_per_seq_weight if is_leaked_flags[b] else nonleaked_per_seq_weight
            leak_weight_bucket.append(w)
            for k in top_k_list:
                hits = int(correct_at[k][b].sum().item())
                leak_bucket[k].append(hits / n_targets)

            if is_spike is not None:
                n_spike_targets = int(is_spike[b].sum().item())
                if n_spike_targets > 0:
                    spike_per_seq_weight.append(w)
                    for k in top_k_list:
                        spike_hits = int((correct_at[k][b] & is_spike[b]).sum().item())
                        spike_per_seq_recall[k].append(spike_hits / n_spike_targets)

            if is_region is not None:
                n_region_targets = int(is_region[b].sum().item())
                if n_region_targets > 0:
                    region_per_seq_weight.append(w)
                    for k in top_k_list:
                        region_hits = int((correct_at[k][b] & is_region[b]).sum().item())
                        region_per_seq_recall[k].append(region_hits / n_region_targets)

    def _finalize(recalls, weights_list):
        n = len(weights_list)
        out = {'n_sequences': n, 'average': {}, 'weighted': {}}
        w_arr = np.array(weights_list) if n else np.array([])
        for k in top_k_list:
            vals = np.array(recalls[k])
            out['average'][f'recall@{k}'] = float(vals.mean()) if n else 0.0
            out['weighted'][f'recall@{k}'] = (
                float((w_arr * vals).sum() / w_arr.sum()) if n and w_arr.sum() > 0 else 0.0
            )
        return out

    result = _finalize(per_seq_recall, per_seq_weight)
    if spike_tensor is not None:
        result['spike'] = _finalize(spike_per_seq_recall, spike_per_seq_weight)
    if region_tensor is not None:
        result['region'] = _finalize(region_per_seq_recall, region_per_seq_weight)
    result['leaked'] = _finalize(leaked_per_seq_recall, leaked_per_seq_weight)
    result['non_leaked'] = _finalize(nonleaked_per_seq_recall, nonleaked_per_seq_weight)
    return result


class PetraRepresentativenessWeights:
    """PETRA の representativeness 重み（transformer_260707/scripts/eval/petra_weights.py を再利用）。

    weight(country, yymm) = proc(population(country) / seq_count(country, yymm))
    country/yymm が不明な場合は 1.0（中立、Average と同一）にフォールバックする。
    """

    def __init__(self, db_path: str, split_type: int):
        from transformer_260707.scripts.eval.petra_weights import (
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
