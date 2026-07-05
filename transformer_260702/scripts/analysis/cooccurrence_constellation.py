# --- transformer_260702/scripts/analysis/cooccurrence_constellation.py ---
"""③ 共起変異コンステレーションの復元検証。

モデルが1サンプルで同時に予測する上位K位置（＝予測された共起変異セット）を集め、
位置ペアの共予測頻度からモデルの「学習したコンステレーション」を抽出する。
実ターゲットの共起ペアと突き合わせ、既知ハプロタイプ（連鎖・エピスタシス）を
再発見できているかを見る。位置は遺伝子(protein)に注釈する。

出力:
  {out}/model_cooccurrence_pairs.csv   … モデル予測の共起ペア上位（pos_a,gene_a,pos_b,gene_b,count）
  {out}/target_cooccurrence_pairs.csv  … 実ターゲットの共起ペア上位
  {out}/constellation_overlap.json     … 予測ペアが実ペアと一致する割合

Usage:
  python -m transformer_260702.scripts.analysis.cooccurrence_constellation \\
      --checkpoint outputs/.../models/best_model.pth --split test --n_batches 100 --top_k 5
"""

import argparse
import json
import os
from collections import Counter
from itertools import combinations

import pandas as pd
import torch

from transformer_260702 import config
from transformer_260702.utils.logging import force_print
from transformer_260702.scripts.analysis import _xai_common as X


def _accumulate(model, loader, n_batches, device, top_k):
    V = config.VOCAB_SIZE_POSITION
    pred_pairs = Counter()
    tgt_pairs = Counter()
    model.eval()
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if bi >= n_batches:
                break
            x_cat, x_num, mask = batch[0]
            out = model(x_cat.to(device), x_num.to(device), src_key_padding_mask=mask.to(device))
            logits = out[1]  # position head [B, V]
            topk = torch.topk(logits, min(top_k, logits.shape[1]), dim=1).indices.cpu().numpy()
            for row in topk:
                for a, b in combinations(sorted(set(int(p) for p in row)), 2):
                    pred_pairs[(a, b)] += 1
            raw_y = batch[8] if len(batch) > 8 else None
            if raw_y is not None:
                for targets in raw_y:
                    pos = sorted(set(int(t[1]) for t in targets if 0 <= int(t[1]) < V))
                    for a, b in combinations(pos, 2):
                        tgt_pairs[(a, b)] += 1
    return pred_pairs, tgt_pairs


def _pairs_df(counter, pos2gene, top_n=200):
    rows = []
    for (a, b), c in counter.most_common(top_n):
        ga = pos2gene.get(a, ('unknown', '0'))[0]
        gb = pos2gene.get(b, ('unknown', '0'))[0]
        rows.append({'pos_a': a, 'gene_a': ga, 'pos_b': b, 'gene_b': gb, 'count': c})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description='Co-occurrence constellation recovery')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--n_batches', type=int, default=100)
    parser.add_argument('--top_k', type=int, default=5, help='1サンプルで共起とみなす上位予測位置数')
    parser.add_argument('--force_cpu', action='store_true')
    parser.add_argument('--output_dir', type=str, default=None)
    args = parser.parse_args()

    out_dir = X.make_output_dir('constellation', args.output_dir)
    force_print(f"Output dir: {out_dir}")

    model, loader, device = X.load_model_and_loader(
        args.checkpoint, split=args.split, force_cpu=args.force_cpu)
    pos2gene = X.load_position_gene_map()

    force_print(f"[INFO] Accumulating co-prediction pairs (top_k={args.top_k}, {args.n_batches} batches)...")
    pred_pairs, tgt_pairs = _accumulate(model, loader, args.n_batches, device, args.top_k)

    pred_df = _pairs_df(pred_pairs, pos2gene)
    tgt_df = _pairs_df(tgt_pairs, pos2gene)
    X.save_csv(pred_df, out_dir, 'model_cooccurrence_pairs.csv')
    X.save_csv(tgt_df, out_dir, 'target_cooccurrence_pairs.csv')

    # 予測上位ペアが実ペア集合にどれだけ含まれるか（コンステレーション復元率）
    tgt_set = set(tgt_pairs.keys())
    for topn in (50, 100, 200):
        pred_top = [pair for pair, _ in pred_pairs.most_common(topn)]
        overlap = sum(1 for p in pred_top if p in tgt_set)
        force_print(f"  予測上位{topn}ペア中、実共起にも存在: {overlap}/{len(pred_top)} "
                    f"({100*overlap/max(len(pred_top),1):.1f}%)")
    result = {
        'top_k': args.top_k,
        'n_pred_pairs': len(pred_pairs),
        'n_target_pairs': len(tgt_pairs),
        'overlap_top100': sum(1 for p, _ in pred_pairs.most_common(100) if p in tgt_set),
    }
    X.save_json(result, out_dir, 'constellation_overlap.json')

    force_print("\nモデル予測の共起ペア Top-10（遺伝子注釈）:")
    for _, r in pred_df.head(10).iterrows():
        force_print(f"    {r['pos_a']}({r['gene_a']}) — {r['pos_b']}({r['gene_b']}) : {r['count']}")


if __name__ == '__main__':
    main()
