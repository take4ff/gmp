# --- transformer_260702/scripts/analysis/local_explain.py ---
"""④ 局所説明（Integrated Gradients による1予測の帰属）。

個々のサンプルについて、モデルが予測した「次変異の位置(top-1)」を、
入力の数値特徴 x_num に対する Integrated Gradients で帰属し、
「どの過去変異（入力スロット）・どの特徴が予測を押し上げたか」を示す。

x_cat は離散埋め込みのため IG はできないが、各入力スロットの位置(x_cat[...,1])で
どの過去変異かは特定できる。数値特徴の寄与と合わせてケーススタディに使える。

出力: {out}/local_explanations.json（サンプルごとに予測位置/遺伝子・寄与上位の入力変異・特徴）

Usage:
  python -m transformer_260702.scripts.analysis.local_explain \\
      --checkpoint outputs/.../models/best_model.pth --split test --n_samples 8 --ig_steps 32
"""

import argparse
import json
import os

import numpy as np
import torch

from transformer_260702 import config
from transformer_260702.utils.logging import force_print
from transformer_260702.scripts.analysis import _xai_common as X
from transformer_260702.scripts.analysis.feature_importance import _NUM_FEATURE_NAMES


def _integrated_gradients(model, x_cat, x_num, mask, target_pos, steps):
    """x_num に対する IG を返す [B,T,C,F]。baseline=0。"""
    baseline = torch.zeros_like(x_num)
    total_grad = torch.zeros_like(x_num)
    for alpha in torch.linspace(0.0, 1.0, steps):
        xi = (baseline + alpha * (x_num - baseline)).clone().detach().requires_grad_(True)
        out = model(x_cat, xi, src_key_padding_mask=mask)
        logits = out[1]                                   # position head [B, V]
        sel = logits.gather(1, target_pos.view(-1, 1)).sum()
        grad = torch.autograd.grad(sel, xi)[0]
        total_grad = total_grad + grad
    avg_grad = total_grad / steps
    return (x_num - baseline) * avg_grad


def main():
    parser = argparse.ArgumentParser(description='Local explanation via Integrated Gradients')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--n_samples', type=int, default=8, help='説明するサンプル数（先頭から）')
    parser.add_argument('--ig_steps', type=int, default=32)
    parser.add_argument('--top', type=int, default=5, help='上位いくつを表示するか')
    parser.add_argument('--force_cpu', action='store_true')
    parser.add_argument('--output_dir', type=str, default=None)
    args = parser.parse_args()

    out_dir = X.make_output_dir('local_explain', args.output_dir)
    force_print(f"Output dir: {out_dir}")

    model, loader, device = X.load_model_and_loader(
        args.checkpoint, split=args.split, batch_size=max(8, args.n_samples), force_cpu=args.force_cpu)
    pos2gene = X.load_position_gene_map()
    feat_names = _NUM_FEATURE_NAMES[:config.NUM_CHEM_FEATURES]

    # 先頭バッチから n_samples を説明
    batch = next(iter(loader))
    x_cat, x_num, mask = batch[0]
    x_cat = x_cat.to(device); x_num = x_num.to(device); mask = mask.to(device)
    B = min(args.n_samples, x_cat.shape[0])
    x_cat, x_num, mask = x_cat[:B], x_num[:B], mask[:B]

    with torch.no_grad():
        out = model(x_cat, x_num, src_key_padding_mask=mask)
        pred_pos = torch.argmax(out[1], dim=1)            # [B] top-1 予測位置

    ig = _integrated_gradients(model, x_cat, x_num, mask, pred_pos, args.ig_steps)  # [B,T,C,F]
    ig_np = ig.detach().cpu().numpy()
    x_cat_np = x_cat.detach().cpu().numpy()
    pred_pos_np = pred_pos.cpu().numpy()

    explanations = []
    for i in range(B):
        p = int(pred_pos_np[i])
        gene = pos2gene.get(p, ('unknown', '0'))[0]
        # 特徴別寄与（|IG| を T,C,B 以外で集約）
        feat_attr = np.abs(ig_np[i]).sum(axis=(0, 1))     # [F]
        top_feats = np.argsort(feat_attr)[::-1][:args.top]
        # 入力変異スロット別寄与（|IG| を F で集約）→ そのスロットの位置(x_cat[...,1])
        slot_attr = np.abs(ig_np[i]).sum(axis=2)          # [T, C]
        valid = x_cat_np[i][..., 0] != 0                  # base_before!=0 が有効スロット
        slot_list = []
        T, C = slot_attr.shape
        for t in range(T):
            for c in range(C):
                if not valid[t, c]:
                    continue
                slot_list.append((float(slot_attr[t, c]), t, int(x_cat_np[i][t, c, 1])))
        slot_list.sort(reverse=True)
        explanations.append({
            'sample_index': i,
            'predicted_next_position': p,
            'predicted_gene': gene,
            'top_features': [
                {'feature': feat_names[j] if j < len(feat_names) else str(j),
                 'attribution': float(feat_attr[j])} for j in top_feats],
            'top_input_mutations': [
                {'timestep': t, 'position': pos, 'gene': pos2gene.get(pos, ('unknown', '0'))[0],
                 'attribution': a} for a, t, pos in slot_list[:args.top]],
        })

    X.save_json(explanations, out_dir, 'local_explanations.json')

    for e in explanations[:min(3, len(explanations))]:
        force_print(f"\n[sample {e['sample_index']}] 予測次変異位置={e['predicted_next_position']} ({e['predicted_gene']})")
        force_print("  寄与の大きい特徴: " + ", ".join(f"{f['feature']}({f['attribution']:.3g})" for f in e['top_features']))
        force_print("  寄与の大きい入力変異: " + ", ".join(
            f"{m['position']}({m['gene']})" for m in e['top_input_mutations']))


if __name__ == '__main__':
    main()
