# --- transformer_260723/scripts/analysis/timestep_attention.py ---
"""⑥ timestep方向（recency）の本物のAttention重み分析。

`timestep_importance.py`はIntegrated Gradients（勾配ベース）でrecency重要度を測るが、
本スクリプトは主Transformer Encoder（self.transformer_encoder, nn.TransformerEncoder,
N_LAYERS層）の**実際のSelf-Attention重み**を直接抽出する。Co-occurrence Attention
（共起集約方向、1層のCross-Attention）とは別の、timestep間の情報混合を担う機構。

TEMPORAL_POOLING='last'（既定）を前提に、各層で「最後のtimestep（予測に使う位置、
front-paddingのため常にインデックス T-1）」が他のどのtimestepにどれだけ注目しているか
を、layer別に集計する（4層あれば4パネル、層ごとに異なるパターンが出うる）。

出力:
  {out}/timestep_attention.csv   … layer, recency, mean_attn, n_valid
  {out}/timestep_attention.png   … layer別 recency vs 平均attention重み

Usage:
  python -m transformer_260723.scripts.analysis.xai.timestep_attention \\
      --checkpoint outputs/.../models/best_model.pth --split test --n_batches 50

  python -m transformer_260723.scripts.analysis.xai.timestep_attention \\
      --walk_forward_dir outputs/transformer_260702/results/walk_forward/<timestamp> --n_batches 50
"""

import argparse

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from transformer_260723 import config
from transformer_260723.utils.logging import force_print
from transformer_260723.scripts.analysis.xai import _xai_common as X


def _attach_encoder_attention_capture(model):
    """model.transformer_encoder の各層 self_attn.forward をラップし、
    直近 forward の [B, T, T] attention重み（head平均済み）を captured[layer_idx] に格納する。
    """
    captured = {}
    layers = model.transformer_encoder.layers
    originals = []
    for i, layer in enumerate(layers):
        orig_forward = layer.self_attn.forward
        originals.append(orig_forward)

        def make_wrapped(idx, orig):
            def wrapped(query, key, value, key_padding_mask=None, need_weights=True,
                       attn_mask=None, average_attn_weights=True, is_causal=False):
                out, w = orig(query, key, value, key_padding_mask=key_padding_mask,
                             need_weights=True, attn_mask=attn_mask,
                             average_attn_weights=True, is_causal=is_causal)
                captured[idx] = w.detach()  # [B, T, T]
                return out, w
            return wrapped

        layer.self_attn.forward = make_wrapped(i, orig_forward)

    def restore():
        for layer, orig in zip(layers, originals):
            layer.self_attn.forward = orig

    return captured, restore


def aggregate_encoder_attention(model, loader, n_batches, device,
                                recency_sum=None, recency_count=None, n_layers=None):
    """loader を流し、各層で「最後の有効queryから各keyへのattention」をrecency別に集計する。"""
    n_layers = n_layers or config.N_LAYERS
    R = config.TRAIN_MAX
    if recency_sum is None:
        recency_sum = np.zeros((n_layers, R), dtype=np.float64)
    if recency_count is None:
        recency_count = np.zeros((n_layers, R), dtype=np.int64)

    captured, restore = _attach_encoder_attention_capture(model)
    model.eval()
    n_samples = 0
    bi = -1
    try:
        # 注意: torch.no_grad() 下では nn.TransformerEncoderLayer (norm_first=True) が
        # self_attn.forward を経由しない融合カーネル(fast path)に切り替わり、monkey-patch
        # したhookが一切呼ばれず captured が空になる。勾配は使わないが no_grad() は
        # 意図的に外し、代わりに各テンソルを都度 detach() してメモリを解放する。
        for bi, batch in enumerate(loader):
            if bi >= n_batches:
                break
            x_cat, x_num, mask = batch[0]
            x_cat = x_cat.to(device); x_num = x_num.to(device); mask = mask.to(device)
            model(x_cat, x_num, src_key_padding_mask=mask)

            valid = (~mask).cpu().numpy()  # [B, T] True=有効
            B, T = valid.shape
            for layer_idx in range(n_layers):
                w = captured[layer_idx].detach().cpu().numpy()  # [B, T, T]
                for b in range(B):
                    if not valid[b, T - 1]:
                        continue  # 最終位置がPAD＝有効サンプルなし
                    for key_t in range(T):
                        if not valid[b, key_t]:
                            continue
                        recency = (T - 1) - key_t
                        recency_sum[layer_idx, recency] += float(w[b, T - 1, key_t])
                        recency_count[layer_idx, recency] += 1
            captured.clear()
            n_samples += B
            if (bi + 1) % 10 == 0:
                force_print(f"  [INFO] {bi + 1}/{n_batches} batches, {n_samples:,} samples")
    finally:
        restore()

    force_print(f"[INFO] Processed {bi + 1} batches, {n_samples:,} samples total.")
    return recency_sum, recency_count, n_samples


def _save_and_plot(recency_sum, recency_count, out_dir, n_layers, title_suffix=''):
    R = recency_sum.shape[1]
    rows = []
    for layer_idx in range(n_layers):
        for r in range(R):
            n = recency_count[layer_idx, r]
            mean_attn = recency_sum[layer_idx, r] / n if n > 0 else 0.0
            rows.append({'layer': layer_idx, 'recency': r, 'mean_attn': mean_attn, 'n_valid': n})
    df = pd.DataFrame(rows)
    X.save_csv(df, out_dir, 'timestep_attention.csv')

    fig, axes = plt.subplots(1, n_layers, figsize=(5.5 * n_layers, 5), sharey=True)
    if n_layers == 1:
        axes = [axes]
    recency = np.arange(R)
    for layer_idx in range(n_layers):
        ax = axes[layer_idx]
        mean_attn = [recency_sum[layer_idx, r] / recency_count[layer_idx, r]
                    if recency_count[layer_idx, r] > 0 else np.nan for r in range(R)]
        ax.plot(recency, mean_attn, marker='o', color='#c0392b', markersize=4)
        ax.axhline(1.0 / R, color='gray', linestyle='--', linewidth=1, label=f'uniform (1/{R})')
        ax.set_title(f'Layer {layer_idx}')
        ax.set_xlabel('Recency (0=most recent)')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel('Mean attention weight (from last query position)')
    fig.suptitle(f'Self-Attention weight by recency, per layer{title_suffix}')
    X.save_fig(fig, out_dir, 'timestep_attention.png')


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

    out_dir = X.make_output_dir('timestep_attention_walk_forward', args.output_dir)
    force_print(f"Output dir: {out_dir}")

    n_layers = None
    R = config.TRAIN_MAX
    recency_sum = None
    recency_count = None
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

        this_n_layers = config.N_LAYERS
        this_R = config.TRAIN_MAX
        if n_layers is None:
            n_layers = this_n_layers
            recency_sum = np.zeros((n_layers, this_R), dtype=np.float64)
            recency_count = np.zeros((n_layers, this_R), dtype=np.int64)
        elif this_n_layers != n_layers or this_R != R:
            force_print(f"[WARN] fold {fold_id} の N_LAYERS/TRAIN_MAX が他フォールドと異なるためスキップ")
            continue

        loader = create_db_dataloader(
            db_path=db_path, split_type=2,
            batch_size=config.BATCH_SIZE, shuffle=False,
            max_cooccurrence=config.MAX_CO_OCCURRENCE,
        )
        recency_sum, recency_count, n = aggregate_encoder_attention(
            model, loader, args.n_batches, device,
            recency_sum=recency_sum, recency_count=recency_count, n_layers=n_layers)
        n_samples_total += n

    force_print(f"[INFO] n_samples_total={n_samples_total:,}")
    _save_and_plot(recency_sum, recency_count, out_dir, n_layers, title_suffix=' (walk-forward, all folds)')


def main():
    parser = argparse.ArgumentParser(description='Timestep-direction Self-Attention weight analysis')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--walk_forward_dir', type=str, default=None)
    parser.add_argument('--folds', type=int, nargs='+', default=None)
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--n_batches', type=int, default=50)
    parser.add_argument('--force_cpu', action='store_true')
    parser.add_argument('--output_dir', type=str, default=None)
    args = parser.parse_args()

    if args.walk_forward_dir:
        run_walk_forward(args)
        return
    if not args.checkpoint:
        raise SystemExit("--checkpoint または --walk_forward_dir のいずれかが必要です")

    out_dir = X.make_output_dir('timestep_attention', args.output_dir)
    force_print(f"Output dir: {out_dir}")

    model, loader, device = X.load_model_and_loader(
        args.checkpoint, split=args.split, force_cpu=args.force_cpu)

    if getattr(model, 'use_flat_coattn', False):
        raise SystemExit("[ERROR] USE_FLAT_COATTN=True のモデルは非対応です。")

    force_print(f"[INFO] Aggregating encoder self-attention over {args.n_batches} batches ({args.split})...")
    recency_sum, recency_count, n_samples = aggregate_encoder_attention(
        model, loader, args.n_batches, device)
    force_print(f"[INFO] n_samples={n_samples:,}")
    _save_and_plot(recency_sum, recency_count, out_dir, config.N_LAYERS)


if __name__ == '__main__':
    main()
