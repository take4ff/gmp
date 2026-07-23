# --- transformer_260723/scripts/eval/petra_recall.py ---
"""PETRA 準拠の Recall@K で提案モデルを評価する（PETRA との厳密比較 Phase 0）。

PETRA (arXiv:2511.03976 / xz-keg/PETra, petra/PETra/model_training/eval_mutgpt.py で確認済み) の
指標に合わせる:
  - 各サンプルについて、次に起きる変異位置（ターゲット集合）に対し、モデルの position ヘッドの
    ランキング上位 K に真の位置がいくつ入っているかで **サンプルごとの recall@K を先に計算**する
    （recall_i@K = hits_i@K / n_targets_i）。
  - **Average Recall@K** = 全サンプルの recall_i@K の単純平均（macro-average）。
    ※ PETRA の forward_step() で `ret_tensor[:,2]=ret_tensor[:,0]/nummut` に対応。
  - **Weighted Recall@K** = Σ(weight_i * recall_i@K) / Σ(weight_i)。
    weight_i は representativeness = proc(population(country) / seq_count(country, yymm))。
    ※ PETRA の forward_step() で `ret_tensor[:,0]=ret_tensor[:,0]*weight/nummut` に対応
      （petra/eval/weights.py に移植済み）。country は DB 再構築後（samples.country 有り）でのみ有効、
      無ければ weight=1.0 にフォールボックし Average と同一になる。
  - **核酸（全位置）** と **スパイク（核酸 21563-25384 の S 遺伝子）** を別々に報告。

粒度は **位置レベル**（PETRA は 位置×塩基 の変異レベル。位置に射影して比較。塩基込みは後続）。

Usage:
  python -m transformer_260723.scripts.eval.petra_recall \\
      --checkpoint outputs/.../models/best_model.pth --split test --ks 1 10 100
"""

import argparse
import os

import numpy as np
import torch

from transformer_260723 import config
from transformer_260723.utils.logging import force_print
from transformer_260723.scripts.analysis.xai import _xai_common as X
from petra.eval.weights import (
    parse_country_population, get_strain_country_map,
    compute_representativeness_weight, _to_yymm,
)

# S 遺伝子の核酸位置範囲（reference/codon/codon_mutation4.csv, protein=='S'）
SPIKE_START, SPIKE_END = 21563, 25384


def _sample_recall_at_ks(ranked_positions, true_positions, ks):
    """1サンプル分の recall_i@K（hits/n_targets）を K ごとに返す。ターゲット無しなら None。"""
    if not true_positions:
        return None
    tset = set(true_positions)
    n = len(tset)
    out = {}
    for k in ks:
        topk = set(ranked_positions[:k])
        out[k] = len(topk & tset) / n
    return out


class _MacroAccumulator:
    """per-sample recall_i@K を集め、Average（単純平均）と Weighted（重み付き平均）を出す。"""

    def __init__(self, ks):
        self.ks = ks
        self.recalls = {k: [] for k in ks}   # per-sample recall_i@K
        self.weights = []                     # per-sample weight（representativeness）
        self.n = 0

    def add(self, recall_dict, weight):
        if recall_dict is None:
            return
        self.n += 1
        self.weights.append(weight)
        for k in self.ks:
            self.recalls[k].append(recall_dict[k])

    def finalize(self):
        out = {'n_samples': self.n, 'average': {}, 'weighted': {}}
        for k in self.ks:
            vals = np.array(self.recalls[k])
            w = np.array(self.weights)
            out['average'][f'recall@{k}'] = float(100.0 * vals.mean()) if len(vals) else 0.0
            out['weighted'][f'recall@{k}'] = (
                float(100.0 * (w * vals).sum() / w.sum()) if len(vals) and w.sum() > 0 else 0.0
            )
        return out


def main():
    parser = argparse.ArgumentParser(description='PETRA-comparable Recall@K for the proposed model')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--ks', nargs='+', type=int, default=[1, 10, 100])
    parser.add_argument('--n_batches', type=int, default=100000, help='最大バッチ数（既定=全部）')
    parser.add_argument('--force_cpu', action='store_true')
    parser.add_argument('--output_dir', type=str, default=None)
    args = parser.parse_args()

    ks = sorted(args.ks)
    maxk = max(ks)
    out_dir = X.make_output_dir('petra_recall', args.output_dir)
    force_print(f"Output dir: {out_dir}")

    model, loader, device = X.load_model_and_loader(
        args.checkpoint, split=args.split, force_cpu=args.force_cpu)

    # representativeness 重みの準備（country が DB に無ければ空 dict → 全サンプル weight=1.0）
    split_map = {'train': 0, 'val': 1, 'test': 2}
    from transformer_260723.db.connection import get_db_path
    pop_dic = parse_country_population()
    strain_country, count_dic = get_strain_country_map(get_db_path(), split_map[args.split])
    has_country = len(strain_country) > 0
    if not has_country:
        force_print("[WARNING] samples.country が無い（未再構築DB）ため Weighted Recall は "
                    "Average と同一値（weight=1.0）になります。")

    nuc_acc = _MacroAccumulator(ks)
    spike_acc = _MacroAccumulator(ks)

    model.eval()
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if bi >= args.n_batches:
                break
            x_cat, x_num, mask = batch[0]
            out = model(x_cat.to(device), x_num.to(device), src_key_padding_mask=mask.to(device))
            logits = out[1]  # position head [B, V]
            topk_all = torch.topk(logits, min(maxk, logits.shape[1]), dim=1).indices.cpu().numpy()
            # スパイク限定ランキング: スパイク外を -inf にして上位を取る
            V = logits.shape[1]
            spike_mask = torch.full((V,), float('-inf'), device=logits.device)
            lo = max(0, SPIKE_START); hi = min(V, SPIKE_END + 1)
            spike_mask[lo:hi] = 0.0
            topk_spike = torch.topk(logits + spike_mask, min(maxk, hi - lo), dim=1).indices.cpu().numpy()

            raw_y = batch[8] if len(batch) > 8 else None
            strains = batch[3] if len(batch) > 3 else None
            dates = batch[9] if len(batch) > 9 else None
            if raw_y is None:
                continue
            for i, targets in enumerate(raw_y):
                true_pos = [int(t[1]) for t in targets]

                # representativeness 重み（country/date が無ければ 1.0）
                weight = 1.0
                if has_country and strains is not None and dates is not None:
                    country = strain_country.get(strains[i])
                    yymm = _to_yymm(dates[i]) if i < len(dates) else None
                    if country and yymm:
                        weight = compute_representativeness_weight(country, yymm, pop_dic, count_dic)

                r = _sample_recall_at_ks(list(topk_all[i]), true_pos, ks)
                nuc_acc.add(r, weight)

                true_spike = [p for p in true_pos if SPIKE_START <= p <= SPIKE_END]
                rs = _sample_recall_at_ks(list(topk_spike[i]), true_spike, ks)
                spike_acc.add(rs, weight)

    result = {
        'split': args.split,
        'ks': ks,
        'granularity': 'position-level (nucleotide position), per-sample macro-average '
                       '(matches PETRA forward_step aggregation)',
        'weighted_recall_valid': has_country,
        'nucleotide': nuc_acc.finalize(),
        'spike': spike_acc.finalize(),
        'note': 'Average = 単純平均, Weighted = representativeness(population/seq_count)重み付き平均。'
                'weighted_recall_valid=False の場合 Weighted は Average と同一（weight=1.0 フォールバック）。'
                '塩基込みの変異レベル(位置×塩基)は後続で追加予定。',
    }
    X.save_json(result, out_dir, 'petra_recall.json')

    force_print("\n===== PETRA 準拠 Recall@K（提案モデル・位置レベル・macro-average）=====")
    force_print(f"  [核酸]   Average: " + "  ".join(f"@{k}={nuc_acc.finalize()['average'][f'recall@{k}']:.2f}%" for k in ks))
    force_print(f"           Weighted: " + "  ".join(f"@{k}={nuc_acc.finalize()['weighted'][f'recall@{k}']:.2f}%" for k in ks))
    force_print(f"  [スパイク] Average: " + "  ".join(f"@{k}={spike_acc.finalize()['average'][f'recall@{k}']:.2f}%" for k in ks))
    force_print(f"           Weighted: " + "  ".join(f"@{k}={spike_acc.finalize()['weighted'][f'recall@{k}']:.2f}%" for k in ks))
    if not has_country:
        force_print("  [注意] Weighted は country 未整備のため Average と同一値です。")


if __name__ == '__main__':
    main()
