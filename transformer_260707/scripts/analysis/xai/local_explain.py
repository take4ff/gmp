# --- transformer_260707/scripts/analysis/local_explain.py ---
"""④ 局所説明（Integrated Gradients による1予測の帰属）。

個々のサンプルについて、モデルが予測した「次変異の位置(top-1)」を、
入力の数値特徴 x_num に対する Integrated Gradients で帰属し、
「どの過去変異（入力スロット）・どの特徴が予測を押し上げたか」を示す。

x_cat は離散埋め込みのため IG はできないが、各入力スロットの位置(x_cat[...,1])で
どの過去変異かは特定できる。数値特徴の寄与と合わせてケーススタディに使える。

出力: {out}/local_explanations.json（サンプルごとに予測位置/遺伝子・寄与上位の入力変異・特徴）

既定（--voc_search、常時有効）で、D614G・N501Y等の既知VOC定義変異がtop-1予測になっている
サンプルを複数バッチから優先的に探して抽出する（--scan_batchesで走査上限を指定）。単一系統に
偏ったfold（例: fold3のBA.1/BA.2初期）では先頭バッチだけだと同じ予測・帰属が繰り返されるため。
見つからなかった分は従来通り先頭バッチの順送りサンプルで埋める。--no_voc_searchで旧動作に戻せる。

Usage:
  python -m transformer_260707.scripts.analysis.xai.local_explain \\
      --checkpoint outputs/.../models/best_model.pth --split test --n_samples 8 --ig_steps 32

  # walk-forward: 特定のVOC期を担当するフォールドを指定して、その期のtestサンプルで
  # ケーススタディを行う（例: fold 3 = Omicron BA.1/BA.2 初期）
  python -m transformer_260707.scripts.analysis.xai.local_explain \\
      --walk_forward_dir outputs/transformer_260707/results/walk_forward/<timestamp> --fold 3 --n_samples 8
"""

import argparse
import json
import os

import numpy as np
import torch

from transformer_260707 import config
from transformer_260707.utils.logging import force_print
from transformer_260707.scripts.analysis.xai import _xai_common as X
from transformer_260707.scripts.analysis.feature_importance import _NUM_FEATURE_NAMES

# 著名なVOC定義変異（Spike, nucleotide position -> 変異名）。
# reference/codon/codon_mutation4.csv の該当行(base_pos, protein=S, protein_pos)と
# 突合して塩基位置を確認済み（記憶からの引用ではなく本プロジェクトの参照データに基づく）。
KNOWN_VOC_POSITIONS = {
    23403: 'D614G (early pandemic, near-fixation)',
    23063: 'N501Y (Alpha/Beta/Gamma/Omicron)',
    22813: 'K417N/T (Beta/Gamma/Omicron)',
    22917: 'L452R (Delta/Epsilon)',
    22995: 'T478K (Delta)',
    23012: 'E484K (Beta/Gamma)',
    23604: 'P681R/H (Delta/Alpha)',
}


def _find_voc_samples(model, loader, device, voc_positions, n_needed, max_scan_batches):
    """複数バッチをスキャンし、既知VOC変異位置がtop-1予測になっているサンプルを優先収集する。

    fold3等、単一系統に偏ったデータでは先頭バッチだけだと同じ予測・帰属が繰り返され
    ケーススタディとして代わり映えしないため、既知VOC変異を狙って探す。
    見つかった分だけ返し、残りは呼び出し側で先頭バッチの順送りサンプルで埋める。
    """
    collected = []
    fallback_batch = None
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            x_cat, x_num, mask = batch[0]
            x_cat = x_cat.to(device); x_num = x_num.to(device); mask = mask.to(device)
            if fallback_batch is None:
                fallback_batch = (x_cat, x_num, mask)
            out = model(x_cat, x_num, src_key_padding_mask=mask)
            pred_pos = torch.argmax(out[1], dim=1)
            for i in range(x_cat.shape[0]):
                p = int(pred_pos[i].item())
                if p in voc_positions and len(collected) < n_needed:
                    collected.append((x_cat[i].clone(), x_num[i].clone(), mask[i].clone(), p))
            if len(collected) >= n_needed or batch_idx + 1 >= max_scan_batches:
                break
    return collected, fallback_batch


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
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--walk_forward_dir', type=str, default=None,
                         help='walk_forward 結果ディレクトリ（--fold と併用。特定VOC期のフォールドで説明する）')
    parser.add_argument('--fold', type=int, default=None, help='walk_forward_dir 使用時に対象とするフォールドID')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--n_samples', type=int, default=8, help='説明するサンプル数（先頭から）')
    parser.add_argument('--ig_steps', type=int, default=32)
    parser.add_argument('--top', type=int, default=5, help='上位いくつを表示するか')
    parser.add_argument('--force_cpu', action='store_true')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--voc_search', action='store_true', default=True,
                         help='既知VOC変異(D614G, N501Y等)がtop-1予測のサンプルを優先抽出する（既定で有効）')
    parser.add_argument('--no_voc_search', dest='voc_search', action='store_false')
    parser.add_argument('--scan_batches', type=int, default=30,
                         help='VOC変異サンプルを探すために走査する最大バッチ数')
    args = parser.parse_args()

    if args.walk_forward_dir:
        if args.fold is None:
            raise SystemExit("--walk_forward_dir 使用時は --fold の指定が必要です")
        fold_ckpts = X.discover_fold_checkpoints(args.walk_forward_dir, [args.fold])
        if args.fold not in fold_ckpts:
            raise SystemExit(f"[ERROR] fold {args.fold} のチェックポイントが見つかりません: {args.walk_forward_dir}")
        train_start, split_date, split_end = X.get_fold_windows()[args.fold]
        from transformer_260707.db.connection import get_db_path
        X.assign_fold_test_window(get_db_path(), train_start, split_date, split_end)
        args.checkpoint = fold_ckpts[args.fold]
        force_print(f"[INFO] Fold {args.fold} (test window [{split_date}, {split_end})) の checkpoint を使用: {args.checkpoint}")
    elif not args.checkpoint:
        raise SystemExit("--checkpoint または --walk_forward_dir+--fold のいずれかが必要です")

    out_dir = X.make_output_dir('local_explain', args.output_dir)
    force_print(f"Output dir: {out_dir}")

    model, loader, device = X.load_model_and_loader(
        args.checkpoint, split=args.split, batch_size=max(8, args.n_samples), force_cpu=args.force_cpu)
    pos2gene = X.load_position_gene_map()
    feat_names = _NUM_FEATURE_NAMES[:config.NUM_CHEM_FEATURES]

    voc_samples = []
    if args.voc_search:
        voc_samples, fallback_batch = _find_voc_samples(
            model, loader, device, KNOWN_VOC_POSITIONS, args.n_samples, args.scan_batches)
        force_print(f"[INFO] 既知VOC変異が予測されたサンプル: {len(voc_samples)}件"
                     f"（最大{args.scan_batches}バッチ走査、対象: {list(KNOWN_VOC_POSITIONS.values())}）")
    else:
        fallback_batch = next(iter(loader))[0]
        fallback_batch = tuple(t.to(device) for t in fallback_batch)

    n_fallback = max(0, args.n_samples - len(voc_samples))
    fb_x_cat, fb_x_num, fb_mask = fallback_batch
    fb_x_cat, fb_x_num, fb_mask = fb_x_cat[:n_fallback], fb_x_num[:n_fallback], fb_mask[:n_fallback]

    if voc_samples:
        voc_x_cat = torch.stack([s[0] for s in voc_samples])
        voc_x_num = torch.stack([s[1] for s in voc_samples])
        voc_mask = torch.stack([s[2] for s in voc_samples])
        x_cat = torch.cat([voc_x_cat, fb_x_cat], dim=0) if n_fallback > 0 else voc_x_cat
        x_num = torch.cat([voc_x_num, fb_x_num], dim=0) if n_fallback > 0 else voc_x_num
        mask = torch.cat([voc_mask, fb_mask], dim=0) if n_fallback > 0 else voc_mask
    else:
        x_cat, x_num, mask = fb_x_cat, fb_x_num, fb_mask

    B = x_cat.shape[0]
    is_voc_match = [True] * len(voc_samples) + [False] * (B - len(voc_samples))

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
            'is_known_voc_position': is_voc_match[i],
            'voc_mutation_name': KNOWN_VOC_POSITIONS.get(p) if is_voc_match[i] else None,
            'top_features': [
                {'feature': feat_names[j] if j < len(feat_names) else str(j),
                 'attribution': float(feat_attr[j])} for j in top_feats],
            'top_input_mutations': [
                {'timestep': t, 'position': pos, 'gene': pos2gene.get(pos, ('unknown', '0'))[0],
                 'attribution': a} for a, t, pos in slot_list[:args.top]],
        })

    X.save_json(explanations, out_dir, 'local_explanations.json')

    for e in explanations:
        voc_tag = f" [VOC: {e['voc_mutation_name']}]" if e['is_known_voc_position'] else ""
        force_print(f"\n[sample {e['sample_index']}] 予測次変異位置={e['predicted_next_position']} ({e['predicted_gene']}){voc_tag}")
        force_print("  寄与の大きい特徴: " + ", ".join(f"{f['feature']}({f['attribution']:.3g})" for f in e['top_features']))
        force_print("  寄与の大きい入力変異: " + ", ".join(
            f"{m['position']}({m['gene']})" for m in e['top_input_mutations']))


if __name__ == '__main__':
    main()
