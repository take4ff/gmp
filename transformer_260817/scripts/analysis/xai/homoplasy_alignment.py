# --- transformer_260817/scripts/analysis/homoplasy_alignment.py ---
"""② ホモプラシー（収斂進化）との整合検証。

モデルの位置別「予測質量」を、系統樹上で繰り返し出現した部位（正の選択下）の
再発回数 reference/homoplasy/site_recurrence.csv と相関させ、
「モデルが進化的に重要な部位を学習したか」を定量化する。

負のコントロール: reference/genome/problematic_sites_sarsCov2.vcf（配列アーチファクト部位）
での予測質量が背景より低いこと＝ノイズ部位に依存していないことを確認する。

出力:
  {out}/homoplasy_alignment.json   … Spearman/Pearson 相関、負コントロールの統計
  {out}/homoplasy_scatter.png      … 再発回数 vs 予測質量の散布図

Usage:
  python -m transformer_260817.scripts.analysis.xai.homoplasy_alignment \\
      --checkpoint outputs/.../models/best_model.pth --split test --n_batches 100

  # walk-forward: 各フォールドを自分の test 期間で評価し、位置重要度を全期間分積算してから
  # 相関を計算する（fold個別結果は out_dir/fold_N/、全fold合算は out_dir 直下）
  python -m transformer_260817.scripts.analysis.xai.homoplasy_alignment \\
      --walk_forward_dir outputs/transformer_260817/results/walk_forward/<timestamp> --n_batches 100
"""

import argparse
import json
import os

import numpy as np

from transformer_260817 import config
from transformer_260817.utils.logging import force_print
from transformer_260817.scripts.analysis.xai import _xai_common as X


def _load_recurrence(path):
    import csv as _csv
    rec = {}
    with open(path, newline='', encoding='utf-8-sig') as f:
        for row in _csv.DictReader(f):
            try:
                rec[int(row['position_id'])] = float(row['recurrence_count'])
            except (KeyError, ValueError, TypeError):
                continue
    return rec


def _load_problematic_positions(vcf_path):
    masked = set()
    with open(vcf_path) as f:
        for line in f:
            if line.startswith('#'):
                continue
            cols = line.rstrip('\n').split('\t')
            if len(cols) >= 7 and 'mask' in cols[6].lower():
                try:
                    masked.add(int(cols[1]))
                except ValueError:
                    continue
    return masked


def _corr(x, y):
    """Pearson / Spearman を返す（scipy があれば使用、無ければ numpy でフォールバック）。"""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    try:
        from scipy.stats import pearsonr, spearmanr
        pr = float(pearsonr(x, y)[0])
        sr = float(spearmanr(x, y)[0])
        return pr, sr
    except Exception:
        def _pear(a, b):
            if a.std() == 0 or b.std() == 0:
                return float('nan')
            return float(np.corrcoef(a, b)[0, 1])
        def _rank(a):
            order = a.argsort()
            r = np.empty_like(order, dtype=float)
            r[order] = np.arange(len(a))
            return r
        return _pear(x, y), _pear(_rank(x), _rank(y))


def _compute_and_report(pred_mass, target_count, n_samples, rec, masked, out_dir, meta_extra=None):
    """正規化・相関・負コントロールを計算し、json+png を out_dir に保存する。
    genome_track._aggregate_and_report と同じ役割（fold個別呼び出し・pooled呼び出し両対応）。
    """
    if n_samples > 0:
        pred_mass = pred_mass / n_samples
        target_rate = target_count / n_samples
    else:
        target_rate = target_count

    positions = sorted(rec.keys())
    V = len(pred_mass)
    positions = [p for p in positions if 0 <= p < V]
    recur = np.array([rec[p] for p in positions])
    pm = np.array([pred_mass[p] for p in positions])
    tr = np.array([target_rate[p] for p in positions])

    pr_model, sr_model = _corr(recur, pm)
    pr_data, sr_data = _corr(recur, tr)

    all_pos = np.arange(V)
    is_masked = np.array([p in masked for p in all_pos])
    pm_masked = pred_mass[is_masked]
    pm_unmasked = pred_mass[~is_masked]
    neg_control = {
        'mean_pred_mass_problematic': float(pm_masked.mean()) if pm_masked.size else 0.0,
        'mean_pred_mass_normal': float(pm_unmasked.mean()) if pm_unmasked.size else 0.0,
        'n_problematic': int(is_masked.sum()),
    }
    ratio = (neg_control['mean_pred_mass_problematic'] /
             neg_control['mean_pred_mass_normal']) if neg_control['mean_pred_mass_normal'] > 0 else float('nan')
    neg_control['problematic_over_normal_ratio'] = ratio

    result = {
        'n_samples': n_samples,
        'n_homoplasy_positions': len(positions),
        'correlation_model_vs_homoplasy': {'pearson': pr_model, 'spearman': sr_model},
        'correlation_data_vs_homoplasy': {'pearson': pr_data, 'spearman': sr_data},
        'negative_control_problematic_sites': neg_control,
        'note': 'model が data(実出現) 以上に homoplasy と相関するなら、頻度以上の進化的シグナルを捉えている。'
                'ratio<1 なら problematic 部位を回避できている。',
    }
    if meta_extra:
        result.update(meta_extra)
    X.save_json(result, out_dir, 'homoplasy_alignment.json')

    _plot_scatter(recur, pm, sr_model, out_dir)

    force_print("\n===== 結果 =====")
    force_print(f"  homoplasy 相関  model : Spearman={sr_model:.3f}, Pearson={pr_model:.3f}")
    force_print(f"  homoplasy 相関  data  : Spearman={sr_data:.3f}, Pearson={pr_data:.3f}")
    force_print(f"  負コントロール problematic/normal 予測質量比 = {ratio:.3f} (1未満が望ましい)")
    return result


def run_walk_forward(args):
    """各フォールドを自分の test ウィンドウで評価し、位置重要度を全期間分積算してから相関を計算する。
    fold個別の結果は out_dir/fold_N/ に、全fold合算(pooled)は out_dir 直下に保存する。
    """
    from transformer_260817.db.connection import get_db_path
    from transformer_260817.db.dataset import create_db_dataloader

    device = 'cpu' if args.force_cpu else config.DEVICE
    db_path = get_db_path()
    fold_windows = X.get_fold_windows()

    fold_ckpts = X.discover_fold_checkpoints(args.walk_forward_dir, args.folds)
    if not fold_ckpts:
        raise SystemExit(f"[ERROR] fold チェックポイントが見つかりません: {args.walk_forward_dir}")
    force_print(f"[INFO] 発見したフォールド: {sorted(fold_ckpts)}")

    out_dir = X.make_output_dir('homoplasy_alignment_walk_forward', args.output_dir)
    force_print(f"Output dir: {out_dir}")

    rec = _load_recurrence(os.path.join('reference', 'homoplasy', 'site_recurrence.csv'))
    masked = _load_problematic_positions(
        os.path.join('reference', 'genome', 'problematic_sites_sarsCov2.vcf'))
    force_print(f"[INFO] homoplasy positions={len(rec)}, problematic(masked)={len(masked)}")

    V = config.VOCAB_SIZE_POSITION
    pred_mass_total = np.zeros(V, dtype=np.float64)
    target_count_total = np.zeros(V, dtype=np.float64)
    n_samples_total = 0
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
        pred_mass, target_count, n = X.compute_position_importance(model, loader, args.n_batches, device)
        fold_dir = os.path.join(out_dir, f'fold_{fold_id}')
        os.makedirs(fold_dir, exist_ok=True)
        _compute_and_report(pred_mass, target_count, n, rec, masked, fold_dir,
                            meta_extra={'mode': 'single_fold', 'fold': fold_id})
        pred_mass_total += pred_mass
        target_count_total += target_count
        n_samples_total += n
        used_folds.append(fold_id)
        del model

    force_print(f"[INFO] n_samples_total={n_samples_total} folds={used_folds}")
    force_print("[INFO] 全fold合算（pooled）を集計します...")
    _compute_and_report(pred_mass_total, target_count_total, n_samples_total, rec, masked, out_dir,
                        meta_extra={'mode': 'walk_forward', 'folds': used_folds,
                                    'n_batches_per_fold': args.n_batches,
                                    'walk_forward_dir': args.walk_forward_dir})


def main():
    parser = argparse.ArgumentParser(description='Homoplasy alignment of position importance')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--walk_forward_dir', type=str, default=None,
                         help='walk_forward 結果ディレクトリ（指定時は --checkpoint の代わりに全フォールドを積算）')
    parser.add_argument('--folds', type=int, nargs='+', default=None, help='walk_forward_dir 使用時に対象を絞る')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--n_batches', type=int, default=100)
    parser.add_argument('--force_cpu', action='store_true')
    parser.add_argument('--output_dir', type=str, default=None)
    args = parser.parse_args()

    if args.walk_forward_dir:
        run_walk_forward(args)
        return
    if not args.checkpoint:
        raise SystemExit("--checkpoint または --walk_forward_dir のいずれかが必要です")

    out_dir = X.make_output_dir('homoplasy_alignment', args.output_dir)
    force_print(f"Output dir: {out_dir}")

    model, loader, device = X.load_model_and_loader(
        args.checkpoint, split=args.split, force_cpu=args.force_cpu)

    force_print(f"[INFO] Aggregating position importance over {args.n_batches} batches...")
    pred_mass, target_count, n_samples = X.compute_position_importance(
        model, loader, args.n_batches, device)

    rec = _load_recurrence(os.path.join('reference', 'homoplasy', 'site_recurrence.csv'))
    masked = _load_problematic_positions(
        os.path.join('reference', 'genome', 'problematic_sites_sarsCov2.vcf'))
    force_print(f"[INFO] homoplasy positions={len(rec)}, problematic(masked)={len(masked)}")

    _compute_and_report(pred_mass, target_count, n_samples, rec, masked, out_dir)


def _plot_scatter(recur, pm, spearman, out_dir):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as e:
        force_print(f"[WARNING] plot skipped: {e}")
        return
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(recur + 1, pm + 1e-9, s=8, alpha=0.4, color='#4C72B0')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Homoplasy recurrence count (+1, log)')
    ax.set_ylabel('Model predicted mass / sample (log)')
    ax.set_title(f'Position importance vs homoplasy  (Spearman={spearman:.3f})')
    fig.tight_layout()
    X.save_fig(fig, out_dir, 'homoplasy_scatter.png')


if __name__ == '__main__':
    main()
