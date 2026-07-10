# --- transformer_260707/scripts/analysis/probe_representation.py ---
"""⑤ 内部表現のプロービング。

予測ヘッド直前の隠れ表現から、モデルが明示的に与えられていない生物学的概念を
線形分類で復元できるか（＝内部に概念が符号化されているか）を測る。

プローブ対象（ターゲット変異から導出）:
  - target_gene   : ターゲット位置の遺伝子（多クラス）
  - in_spike      : ターゲットが Spike か（2値）
  - is_synonymous : 同義置換か（2値）

各プローブの精度を多数派ベースラインと比較する（精度 >> ベースラインなら概念を保持）。

出力: {out}/probe_results.json

Usage:
  python -m transformer_260707.scripts.analysis.xai.probe_representation \\
      --checkpoint outputs/.../models/best_model.pth --split test --n_batches 150
"""

import argparse
import json
import os

import numpy as np
import torch

from transformer_260707 import config
from transformer_260707.utils.logging import force_print
from transformer_260707.scripts.analysis.xai import _xai_common as X


def _collect(model, loader, n_batches, device, pos2gene):
    """予測ヘッド直前表現 X と、各サンプルのラベル(gene/in_spike/syn)を集める。"""
    captured = {}

    def _pre_hook(module, inp):
        captured['rep'] = inp[0].detach().cpu().numpy()

    handle = model.position_head.register_forward_pre_hook(_pre_hook)
    feats, gene_lbl, spike_lbl, syn_lbl = [], [], [], []
    model.eval()
    try:
        with torch.no_grad():
            for bi, batch in enumerate(loader):
                if bi >= n_batches:
                    break
                x_cat, x_num, mask = batch[0]
                model(x_cat.to(device), x_num.to(device), src_key_padding_mask=mask.to(device))
                rep = captured.get('rep')
                raw_y = batch[8] if len(batch) > 8 else None
                if rep is None or raw_y is None:
                    continue
                for i, targets in enumerate(raw_y):
                    if not targets:
                        continue
                    t = targets[0]                       # 代表として先頭ターゲットを使用
                    pos = int(t[1])
                    gene = pos2gene.get(pos, ('unknown', '0'))[0]
                    feats.append(rep[i])
                    gene_lbl.append(gene)
                    spike_lbl.append(1 if gene == 'S' else 0)
                    syn_lbl.append(int(t[4]) if len(t) > 4 else 0)
    finally:
        handle.remove()
    return np.array(feats), np.array(gene_lbl), np.array(spike_lbl), np.array(syn_lbl)


def _probe(X_feat, y, name):
    """線形プローブ精度と多数派ベースラインを返す。"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    classes, counts = np.unique(y, return_counts=True)
    baseline = counts.max() / counts.sum()
    if len(classes) < 2:
        return {'probe': None, 'baseline': float(baseline), 'n_classes': int(len(classes)),
                'note': 'クラスが1つのみでプローブ不能'}
    Xtr, Xte, ytr, yte = train_test_split(X_feat, y, test_size=0.3, random_state=config.SEED,
                                          stratify=y if counts.min() >= 2 else None)
    scaler = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=1000, multi_class='auto')
    clf.fit(scaler.transform(Xtr), ytr)
    acc = clf.score(scaler.transform(Xte), yte)
    return {'probe_accuracy': float(acc), 'majority_baseline': float(baseline),
            'lift': float(acc - baseline), 'n_classes': int(len(classes)),
            'n_samples': int(len(y))}


def main():
    parser = argparse.ArgumentParser(description='Linear probing of hidden representation')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--n_batches', type=int, default=150)
    parser.add_argument('--force_cpu', action='store_true')
    parser.add_argument('--output_dir', type=str, default=None)
    args = parser.parse_args()

    out_dir = X.make_output_dir('probe', args.output_dir)
    force_print(f"Output dir: {out_dir}")

    model, loader, device = X.load_model_and_loader(
        args.checkpoint, split=args.split, force_cpu=args.force_cpu)
    pos2gene = X.load_position_gene_map()

    force_print(f"[INFO] Collecting representations over {args.n_batches} batches...")
    feats, gene_lbl, spike_lbl, syn_lbl = _collect(model, loader, args.n_batches, device, pos2gene)
    force_print(f"[INFO] Collected {len(feats)} samples, rep dim={feats.shape[1] if len(feats) else 0}")

    results = {
        'n_samples': int(len(feats)),
        'rep_dim': int(feats.shape[1]) if len(feats) else 0,
        'target_gene': _probe(feats, gene_lbl, 'target_gene'),
        'in_spike': _probe(feats, spike_lbl, 'in_spike'),
        'is_synonymous': _probe(feats, syn_lbl, 'is_synonymous'),
        'note': 'probe_accuracy >> majority_baseline なら、その生物概念が表現に符号化されている。',
    }
    X.save_json(results, out_dir, 'probe_results.json')

    force_print("\n===== プローブ結果（精度 vs 多数派ベースライン）=====")
    for key in ('target_gene', 'in_spike', 'is_synonymous'):
        r = results[key]
        if r.get('probe_accuracy') is not None:
            force_print(f"  {key:14s}: acc={r['probe_accuracy']:.3f}  baseline={r['majority_baseline']:.3f}  "
                        f"lift={r['lift']:+.3f}")
        else:
            force_print(f"  {key:14s}: {r.get('note')}")


if __name__ == '__main__':
    main()
