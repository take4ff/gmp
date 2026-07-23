# --- transformer_260723/scripts/analysis/probe_representation.py ---
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
  python -m transformer_260723.scripts.analysis.xai.probe_representation \\
      --checkpoint outputs/.../models/best_model.pth --split test --n_batches 150

  # walk-forward: 各フォールド自身のモデル・test期間で個別にプローブし(fold個別結果は
  # out_dir/fold_N/)、n_samples重み付き平均を out_dir 直下に保存する。fold毎に別モデル
  # （別表現空間）のため、生の表現ベクトルはプールせず精度を後から平均する。
  python -m transformer_260723.scripts.analysis.xai.probe_representation \\
      --walk_forward_dir outputs/transformer_260723/results/walk_forward/<timestamp> --n_batches 150
"""

import argparse
import json
import os

import numpy as np
import torch

from transformer_260723 import config
from transformer_260723.utils.logging import force_print
from transformer_260723.scripts.analysis.xai import _xai_common as X


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


def _probe_all(feats, gene_lbl, spike_lbl, syn_lbl, meta_extra=None):
    results = {
        'n_samples': int(len(feats)),
        'rep_dim': int(feats.shape[1]) if len(feats) else 0,
        'target_gene': _probe(feats, gene_lbl, 'target_gene'),
        'in_spike': _probe(feats, spike_lbl, 'in_spike'),
        'is_synonymous': _probe(feats, syn_lbl, 'is_synonymous'),
        'note': 'probe_accuracy >> majority_baseline なら、その生物概念が表現に符号化されている。',
    }
    if meta_extra:
        results.update(meta_extra)
    return results


def _print_probe_results(results):
    force_print("\n===== プローブ結果（精度 vs 多数派ベースライン）=====")
    for key in ('target_gene', 'in_spike', 'is_synonymous'):
        r = results[key]
        if r.get('probe_accuracy') is not None:
            force_print(f"  {key:14s}: acc={r['probe_accuracy']:.3f}  baseline={r['majority_baseline']:.3f}  "
                        f"lift={r['lift']:+.3f}")
        else:
            force_print(f"  {key:14s}: {r.get('note')}")


def _weighted_mean_probe(fold_results, key):
    """fold別のprobe結果（probe_accuracy/majority_baseline/lift が None のfoldは除外）を
    n_samples 重み付き平均で1つに集約する（各foldは別モデル・別表現空間のため、生特徴を
    プールせず、fold毎に学習したプローブの精度を後から重み付き平均する）。
    """
    valid = [(r[key]['n_samples'], r[key]) for r in fold_results
             if r[key].get('probe_accuracy') is not None]
    if not valid:
        return {'probe_accuracy': None, 'note': '有効なfoldが無くプローブ不能',
                'n_folds_used': 0}
    total_n = sum(n for n, _ in valid)
    acc = sum(n * r['probe_accuracy'] for n, r in valid) / total_n
    baseline = sum(n * r['majority_baseline'] for n, r in valid) / total_n
    return {
        'probe_accuracy': float(acc), 'majority_baseline': float(baseline),
        'lift': float(acc - baseline), 'n_samples': int(total_n),
        'n_folds_used': len(valid),
    }


def run_walk_forward(args):
    """各フォールド自身のモデル・test期間で個別にプローブし、fold個別結果を
    out_dir/fold_N/ に保存したうえで、n_samples重み付き平均を out_dir 直下に保存する。

    fold毎に重みの異なるモデル（＝異なる表現空間）のため、生の表現ベクトルは
    プールせず「fold毎にプローブ精度を測ってから精度を平均する」方式を取る。
    """
    from transformer_260723.db.connection import get_db_path
    from transformer_260723.db.dataset import create_db_dataloader

    device = 'cpu' if args.force_cpu else config.DEVICE
    db_path = get_db_path()
    fold_windows = X.get_fold_windows()

    fold_ckpts = X.discover_fold_checkpoints(args.walk_forward_dir, args.folds)
    if not fold_ckpts:
        raise SystemExit(f"[ERROR] fold チェックポイントが見つかりません: {args.walk_forward_dir}")
    force_print(f"[INFO] 発見したフォールド: {sorted(fold_ckpts)}")

    out_dir = X.make_output_dir('probe_walk_forward', args.output_dir)
    force_print(f"Output dir: {out_dir}")
    pos2gene = X.load_position_gene_map()

    fold_results = []
    used_folds = []

    for fold_id in sorted(fold_ckpts):
        if fold_id not in fold_windows:
            force_print(f"[WARN] fold {fold_id} は FOLDS 定義に無いためスキップ")
            continue
        train_start, split_date, split_end = fold_windows[fold_id]
        force_print(f"\n===== Fold {fold_id}: test window [{split_date}, {split_end}) =====")

        model = X.load_fold_model(fold_ckpts[fold_id], device)
        if getattr(model, 'use_flat_coattn', False):
            force_print(f"[WARN] fold {fold_id} は USE_FLAT_COATTN のためスキップ")
            continue
        X.assign_fold_test_window(db_path, train_start, split_date, split_end)

        loader = create_db_dataloader(
            db_path=db_path, split_type=2,
            batch_size=config.BATCH_SIZE, shuffle=False,
            max_cooccurrence=config.MAX_CO_OCCURRENCE,
        )
        feats, gene_lbl, spike_lbl, syn_lbl = _collect(model, loader, args.n_batches, device, pos2gene)
        force_print(f"[INFO] fold {fold_id}: Collected {len(feats)} samples, "
                    f"rep dim={feats.shape[1] if len(feats) else 0}")
        result = _probe_all(feats, gene_lbl, spike_lbl, syn_lbl,
                            meta_extra={'mode': 'single_fold', 'fold': fold_id})
        fold_dir = os.path.join(out_dir, f'fold_{fold_id}')
        os.makedirs(fold_dir, exist_ok=True)
        X.save_json(result, fold_dir, 'probe_results.json')
        _print_probe_results(result)
        fold_results.append(result)
        used_folds.append(fold_id)
        del model

    force_print(f"[INFO] folds={used_folds}")
    force_print("[INFO] 全fold n_samples重み付き平均（pooled）を集計します...")
    pooled = {
        'mode': 'walk_forward', 'folds': used_folds, 'walk_forward_dir': args.walk_forward_dir,
        'target_gene': _weighted_mean_probe(fold_results, 'target_gene'),
        'in_spike': _weighted_mean_probe(fold_results, 'in_spike'),
        'is_synonymous': _weighted_mean_probe(fold_results, 'is_synonymous'),
        'note': 'fold毎に別モデルで学習したプローブの精度を n_samples 重み付き平均で集約したもの'
                '（生の表現ベクトルはプールしていない）。',
    }
    X.save_json(pooled, out_dir, 'probe_results.json')
    _print_probe_results(pooled)


def main():
    parser = argparse.ArgumentParser(description='Linear probing of hidden representation')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--walk_forward_dir', type=str, default=None,
                         help='walk_forward 結果ディレクトリ（指定時は --checkpoint の代わりに全フォールドを個別プローブし平均する）')
    parser.add_argument('--folds', type=int, nargs='+', default=None, help='walk_forward_dir 使用時に対象を絞る')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--n_batches', type=int, default=150)
    parser.add_argument('--force_cpu', action='store_true')
    parser.add_argument('--output_dir', type=str, default=None)
    args = parser.parse_args()

    if args.walk_forward_dir:
        run_walk_forward(args)
        return
    if not args.checkpoint:
        raise SystemExit("--checkpoint または --walk_forward_dir のいずれかが必要です")

    out_dir = X.make_output_dir('probe', args.output_dir)
    force_print(f"Output dir: {out_dir}")

    model, loader, device = X.load_model_and_loader(
        args.checkpoint, split=args.split, force_cpu=args.force_cpu)
    pos2gene = X.load_position_gene_map()

    force_print(f"[INFO] Collecting representations over {args.n_batches} batches...")
    feats, gene_lbl, spike_lbl, syn_lbl = _collect(model, loader, args.n_batches, device, pos2gene)
    force_print(f"[INFO] Collected {len(feats)} samples, rep dim={feats.shape[1] if len(feats) else 0}")

    results = _probe_all(feats, gene_lbl, spike_lbl, syn_lbl)
    X.save_json(results, out_dir, 'probe_results.json')
    _print_probe_results(results)


if __name__ == '__main__':
    main()
