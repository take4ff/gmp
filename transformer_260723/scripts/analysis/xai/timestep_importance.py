# --- transformer_260723/scripts/analysis/timestep_importance.py ---
"""⑤ タイムステップ（recency）別の予測寄与度アトリビューション。

local_explain.py と同じ Integrated Gradients 機構を再利用し、集約軸を変える:
  local_explain.py … 特定サンプルについて「どの特徴量/どの入力変異」が効いたか（個別ケース）
  本スクリプト     … 多数サンプルを集計し「入力の何ステップ前（recency）が効くか」の減衰カーブ

生物学的/アーキテクチャ的な問い: モデルは直近の変異にのみ依存しているか、
それとも39ステップ窓の奥（古い履歴）まで実際に活用しているか。
config.MAX_SEQ_LEN=39 という入力窓の設計が妥当かどうかの直接的な傍証になる。

recency の定義: recency=0 が「最も直近の変異（入力の末尾）」、recency=R-1 が
「最も古い変異（39ステップ窓の先頭）」。padding_mask は前方パディング
（古い方が削られる）なので、recency は常にサンプルの path_length に依らず
「末尾から何番目か」で一貫する。

出力:
  {out}/timestep_importance.csv   … recency, mean_importance, n_valid
  {out}/timestep_importance.png   … recency vs 平均寄与度（サンプル数を併記）

Usage:
  python -m transformer_260723.scripts.analysis.xai.timestep_importance \\
      --checkpoint outputs/.../models/best_model.pth --split test --n_batches 20 --ig_steps 32

  # walk-forward: 各フォールドを自分の test 期間で評価し、全期間分を積算する
  python -m transformer_260723.scripts.analysis.xai.timestep_importance \\
      --walk_forward_dir outputs/transformer_260702/results/walk_forward/<timestamp> --n_batches 20
"""

import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from transformer_260723 import config
from transformer_260723.utils.logging import force_print
from transformer_260723.scripts.analysis.xai import _xai_common as X
from transformer_260723.scripts.analysis.xai.local_explain import _integrated_gradients


def compute_timestep_importance(model, loader, n_batches, device, ig_steps,
                                importance_sum=None, count=None):
    """loader を流し、recency別（末尾からの距離）の |IG| 寄与度合計と有効サンプル数を集計する。

    importance_sum/count を渡すと追記する（フォールド跨ぎのマージ用）。
    Returns: importance_sum[R], count[R], n_samples_processed
    """
    R = config.TRAIN_MAX
    if importance_sum is None:
        importance_sum = np.zeros(R, dtype=np.float64)
    if count is None:
        count = np.zeros(R, dtype=np.int64)

    model.eval()
    n_samples = 0
    bi = -1
    for bi, batch in enumerate(loader):
        if bi >= n_batches:
            break
        x_cat, x_num, mask = batch[0]
        x_cat = x_cat.to(device); x_num = x_num.to(device); mask = mask.to(device)

        import torch
        with torch.no_grad():
            out = model(x_cat, x_num, src_key_padding_mask=mask)
            pred_pos = out[1].argmax(dim=1)

        ig = _integrated_gradients(model, x_cat, x_num, mask, pred_pos, ig_steps)  # [B,T,C,F]
        ts_importance = ig.detach().abs().sum(dim=(2, 3)).cpu().numpy()            # [B,T]
        valid = (~mask).cpu().numpy()                                             # [B,T] True=有効

        B, T = ts_importance.shape
        for b in range(B):
            valid_t = np.nonzero(valid[b])[0]
            for t in valid_t:
                recency = (T - 1) - int(t)
                importance_sum[recency] += float(ts_importance[b, t])
                count[recency] += 1
        n_samples += B
        if (bi + 1) % 10 == 0:
            force_print(f"  [INFO] {bi + 1}/{n_batches} batches, {n_samples:,} samples processed")

    force_print(f"[INFO] Processed {bi + 1} batches, {n_samples:,} samples total.")
    return importance_sum, count, n_samples


def _save_and_plot(importance_sum, count, out_dir, title_suffix=''):
    R = len(importance_sum)
    mean_importance = np.divide(importance_sum, count, out=np.zeros(R), where=count > 0)
    recency = np.arange(R)

    df = pd.DataFrame({'recency': recency, 'mean_importance': mean_importance, 'n_valid': count})
    X.save_csv(df, out_dir, 'timestep_importance.csv')

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    ax1.plot(recency, mean_importance, marker='o', color='#c0392b')
    ax1.set_ylabel('Mean |IG| timestep importance')
    ax1.set_title(f'Timestep (recency) importance for next-mutation prediction{title_suffix}')
    ax1.grid(alpha=0.3)

    ax2.bar(recency, count, color='gray')
    ax2.set_yscale('log')
    ax2.set_xlabel('Recency (0 = most recent input mutation, larger = further back in history)')
    ax2.set_ylabel('n_valid\n(log)')
    plt.tight_layout()
    X.save_fig(fig, out_dir, 'timestep_importance.png')


def run_walk_forward(args):
    from transformer_260723.db.connection import get_db_path
    from transformer_260723.db.dataset import create_db_dataloader

    device = 'cpu' if args.force_cpu else config.DEVICE
    db_path = get_db_path()
    fold_windows = X.get_fold_windows()

    fold_ckpts = X.discover_fold_checkpoints(args.walk_forward_dir, args.folds)
    if not fold_ckpts:
        raise SystemExit(f"[ERROR] fold チェックポイントが見つかりません: {args.walk_forward_dir}")
    force_print(f"[INFO] 発見したフォールド: {sorted(fold_ckpts)}")

    out_dir = X.make_output_dir('timestep_importance_walk_forward', args.output_dir)
    force_print(f"Output dir: {out_dir}")

    R = config.TRAIN_MAX
    importance_sum = np.zeros(R, dtype=np.float64)
    count = np.zeros(R, dtype=np.int64)
    n_samples_total = 0

    for fold_id in sorted(fold_ckpts):
        if fold_id not in fold_windows:
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
        # foldごとにTRAIN_MAXが変わりうるため配列サイズを都度チェック
        this_R = config.TRAIN_MAX
        if this_R != R:
            force_print(f"[WARN] fold {fold_id} の TRAIN_MAX={this_R} が他フォールドと異なるためスキップ")
            continue

        importance_sum, count, n = compute_timestep_importance(
            model, loader, args.n_batches, device, args.ig_steps,
            importance_sum=importance_sum, count=count)
        n_samples_total += n

    force_print(f"[INFO] n_samples_total={n_samples_total:,}")
    _save_and_plot(importance_sum, count, out_dir, title_suffix=' (walk-forward, all folds)')


def main():
    parser = argparse.ArgumentParser(description='Timestep (recency) attribution for next-mutation prediction')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--walk_forward_dir', type=str, default=None,
                         help='walk_forward 結果ディレクトリ（指定時は --checkpoint の代わりに全フォールドを積算）')
    parser.add_argument('--folds', type=int, nargs='+', default=None, help='walk_forward_dir 使用時に対象を絞る')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--n_batches', type=int, default=20,
                         help='IGは1バッチあたりig_steps回のforward+backwardが必要なため控えめが既定')
    parser.add_argument('--ig_steps', type=int, default=32)
    parser.add_argument('--force_cpu', action='store_true')
    parser.add_argument('--output_dir', type=str, default=None)
    args = parser.parse_args()

    if args.walk_forward_dir:
        run_walk_forward(args)
        return
    if not args.checkpoint:
        raise SystemExit("--checkpoint または --walk_forward_dir のいずれかが必要です")

    out_dir = X.make_output_dir('timestep_importance', args.output_dir)
    force_print(f"Output dir: {out_dir}")

    model, loader, device = X.load_model_and_loader(
        args.checkpoint, split=args.split, force_cpu=args.force_cpu)

    force_print(f"[INFO] Aggregating timestep importance over {args.n_batches} batches "
                f"(ig_steps={args.ig_steps}, {args.split})...")
    importance_sum, count, n_samples = compute_timestep_importance(
        model, loader, args.n_batches, device, args.ig_steps)
    force_print(f"[INFO] n_samples={n_samples:,}")
    _save_and_plot(importance_sum, count, out_dir)


if __name__ == '__main__':
    main()
