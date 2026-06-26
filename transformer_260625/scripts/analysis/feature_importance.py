# --- transformer_260625/scripts/feature_importance.py ---
"""学習済みモデルへの Gradient-based Feature Importance 分析

手法:
  1. x_num (32次元連続特徴量): Gradient × Input
       各サンプルで position ヘッドの Top-1 ロジットに対する勾配 × 入力値の絶対値を算出し、
       B / T / C 次元で平均して特徴量ごとの重要度スコアを得る。
  2. x_cat 埋め込み層の重みノルム:
       各 Embedding 行列の行ベクトル L2 ノルムの平均を「埋め込みの活用強度」として報告する。
  3. Co-occurrence Attention weight 分布:
       Cross-Attention の attn_weights [B*T, 1, C] を集計し、
       共起変異スロットごとの注目度分布を可視化する。

Usage:
    python -m transformer_260625.scripts.feature_importance \\
        --checkpoint outputs/transformer_260625/results/<timestamp>/models/best_model.pth \\
        [--n_batches 200] [--split test] [--target position]
"""

import os
import argparse
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from transformer_260625 import config
from transformer_260625.model import HierarchicalTransformer
from transformer_260625.utils.logging import force_print


# ─────────────────────────────────────────────────────────────
# 特徴量名定義
# ─────────────────────────────────────────────────────────────

# x_num の各次元名（feature.py の num_feat 構造と一致）
_NUM_FEATURE_NAMES = [
    'codon_freq',            # 0  コドン使用頻度
    'hydrophobicity',        # 1  疎水性差 (Eisenberg-Weiss)
    'charge',                # 2  電荷差
    'aa_size',               # 3  アミノ酸サイズ差
    'blosum',                # 4  BLOSUM スコア
    'pam250',                # 5  PAM250 スコア
    # ── ホスト適応特徴量 (_ADAPT_COLS 順) ─────────────────────
    'host_log_ratio_diff',         # 6
    'host_log_ratio_before',       # 7
    'human_RSCU_diff',             # 8
    'SCV2_RSCU_diff',              # 9
    'opt_to_opt',                  # 10
    'nonopt_to_opt',               # 11
    'opt_to_nonopt',               # 12
    'is_transition',               # 13
    'trans_human_RSCU_diff',       # 14
    'trans_SCV2_RSCU_diff',        # 15
    'CpG_diff',                    # 16
    'UpA_diff',                    # 17
    'human_RSCU_before',           # 18
    'SCV2_RSCU_before',            # 19
    'human_freq_before',           # 20
    'human_freq_diff',             # 21
    'SCV2_freq_before',            # 22
    'SCV2_freq_diff',              # 23
    'human_CAI_before',            # 24
    'human_CAI_diff',              # 25
    'SCV2_CAI_before',             # 26
    'SCV2_CAI_diff',               # 27
    'host_RSCU_ratio_before',      # 28
    'host_RSCU_ratio_diff',        # 29
    # ──────────────────────────────────────────────────────────
    'cum_syn',                     # 30  累積同義変異カウント
    'cum_nonsyn',                  # 31  累積非同義変異カウント
]

# x_cat の各スロットと対応する埋め込み層名
_CAT_SLOT_NAMES = [
    ('base_before',  'base_embed'),
    ('nuc_position', 'pos_embed'),
    ('base_after',   'base_embed'),
    ('codon_pos',    'codon_pos_embed'),
    ('aa_before',    'aa_embed'),
    ('aa_position',  'prot_pos_embed'),
    ('aa_after',     'aa_embed'),
    ('region',       'region_embed'),
    ('synonymous',   'synonymous_embed'),
]
# CONTEXT_WINDOW 分のコンテキスト塩基スロット（実行時に追加）


# ─────────────────────────────────────────────────────────────
# 1. Gradient × Input (x_num)
# ─────────────────────────────────────────────────────────────

def compute_grad_x_input(model, loader, target_head_idx, n_batches, device):
    """position (or region) ロジット Top-1 に対する Gradient × Input を集計する。

    Returns:
        importance: ndarray [NUM_CHEM_FEATURES]  — 平均絶対値スコア
    """
    model.eval()
    accum = None
    n_total = 0

    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        (x_cat, x_num, mask), *_ = batch

        x_cat = x_cat.to(device)
        x_num = x_num.to(device, dtype=torch.float32)
        mask  = mask.to(device)

        x_num = x_num.detach().requires_grad_(True)

        with torch.enable_grad():
            out = model(x_cat, x_num, src_key_padding_mask=mask)
            logits = out[target_head_idx]          # [B, V]
            score  = logits.max(dim=-1).values     # [B]  Top-1 ロジット
            score.sum().backward()

        grad = x_num.grad                          # [B, T, C, F]
        gi   = (grad * x_num).abs()               # [B, T, C, F]
        # B / T / C を平均し [F] に集約
        imp  = gi.detach().cpu().mean(dim=(0, 1, 2)).numpy()

        if accum is None:
            accum = imp
        else:
            accum += imp
        n_total += 1

        if (i + 1) % 50 == 0:
            force_print(f"  GradxInput: {i+1}/{n_batches} batches processed")

    return accum / max(n_total, 1)


# ─────────────────────────────────────────────────────────────
# 2. 埋め込み重みノルム
# ─────────────────────────────────────────────────────────────

def compute_embedding_norms(model):
    """各 Embedding 層の行ベクトル L2 ノルムの平均・最大を返す。

    Returns:
        records: list of dict {name, mean_norm, max_norm, n_tokens}
    """
    embed_module = model.input_embed
    ctx_w = getattr(config, 'CONTEXT_WINDOW', 3)

    layers = {
        'base_embed':       embed_module.base_embed,
        'pos_embed':        embed_module.pos_embed,
        'aa_embed':         embed_module.aa_embed,
        'codon_pos_embed':  embed_module.codon_pos_embed,
        'prot_pos_embed':   embed_module.prot_pos_embed,
        'region_embed':     embed_module.region_embed,
        'synonymous_embed': embed_module.synonymous_embed,
    }

    records = []
    for name, layer in layers.items():
        w = layer.weight.detach().cpu()  # [vocab, dim]
        norms = w.norm(dim=1)            # [vocab]
        records.append({
            'embedding': name,
            'n_tokens':  w.shape[0],
            'mean_norm': float(norms.mean()),
            'max_norm':  float(norms.max()),
        })

    return records


# ─────────────────────────────────────────────────────────────
# 3. Co-occurrence Attention weight 分布
# ─────────────────────────────────────────────────────────────

def collect_coattn_weights(model, loader, n_batches, device):
    """Co-occurrence Cross-Attention の重みを収集して返す。

    Returns:
        all_weights: ndarray [N, C]  N=サンプル×タイムステップ数, C=max_co_occurrence
    """
    model.eval()
    collected = []
    original_forward = model.co_attn.forward

    # monkey-patch: need_weights を常に True にして重みを横取り
    captured = []

    def _patched(x, need_weights=False):
        out, w = original_forward(x, need_weights=True)
        # w: [B*T, 1, C]
        captured.append(w.detach().cpu())
        return out

    model.co_attn.forward = _patched

    try:
        with torch.no_grad():
            for i, batch in enumerate(loader):
                if i >= n_batches:
                    break
                (x_cat, x_num, mask), *_ = batch
                x_cat = x_cat.to(device)
                x_num = x_num.to(device, dtype=torch.float32)
                mask  = mask.to(device)
                model(x_cat, x_num, src_key_padding_mask=mask)

                if (i + 1) % 50 == 0:
                    force_print(f"  CoAttn weights: {i+1}/{n_batches} batches processed")
    finally:
        model.co_attn.forward = original_forward  # 必ず元に戻す

    if not captured:
        return None

    # [N_total, 1, C] → [N_total, C]
    all_w = torch.cat(captured, dim=0).squeeze(1).numpy()
    return all_w


# ─────────────────────────────────────────────────────────────
# プロット
# ─────────────────────────────────────────────────────────────

def _save_fig(fig, path):
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    force_print(f"  Saved: {path}")


def plot_num_importance(importance, feature_names, out_dir, top_n=32):
    idx_sorted = np.argsort(importance)[::-1][:top_n]
    names = [feature_names[i] for i in idx_sorted]
    vals  = importance[idx_sorted]

    # グループ色（物理化学 / ホスト適応 / 累積カウント）
    colors = []
    for i in idx_sorted:
        if i < 6:
            colors.append('#4C72B0')    # 物理化学
        elif i < 30:
            colors.append('#DD8452')    # ホスト適応
        else:
            colors.append('#55A868')    # 累積カウント

    fig, ax = plt.subplots(figsize=(10, max(4, top_n * 0.32)))
    bars = ax.barh(range(len(names)), vals[::-1], color=colors[::-1])
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names[::-1], fontsize=9)
    ax.set_xlabel('Mean |Gradient × Input|')
    ax.set_title('Numerical Feature Importance (Gradient × Input)')

    # 凡例
    from matplotlib.patches import Patch
    legend = [
        Patch(color='#4C72B0', label='Physicochemical (0-5)'),
        Patch(color='#DD8452', label='Host adaptation (6-29)'),
        Patch(color='#55A868', label='Cumulative count (30-31)'),
    ]
    ax.legend(handles=legend, loc='lower right', fontsize=8)
    fig.tight_layout()
    _save_fig(fig, os.path.join(out_dir, 'num_feature_importance.png'))


def plot_embedding_norms(records, out_dir):
    df = pd.DataFrame(records).sort_values('mean_norm', ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(df['embedding'], df['mean_norm'], color='#4C72B0', label='mean L2 norm')
    ax.bar(df['embedding'], df['max_norm'],  color='#4C72B0', alpha=0.3, label='max L2 norm')
    ax.set_ylabel('L2 norm of embedding vectors')
    ax.set_title('Embedding Weight Norms (mean / max)')
    ax.legend()
    plt.xticks(rotation=25, ha='right', fontsize=9)
    fig.tight_layout()
    _save_fig(fig, os.path.join(out_dir, 'embedding_norms.png'))


def plot_coattn_distribution(all_weights, out_dir):
    # all_weights: [N, C]
    C = all_weights.shape[1]
    mean_w = all_weights.mean(axis=0)   # [C]
    std_w  = all_weights.std(axis=0)    # [C]
    slots  = np.arange(C)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # 左: 平均注目重み per スロット
    ax = axes[0]
    ax.bar(slots, mean_w, yerr=std_w, capsize=3, color='#4C72B0', alpha=0.8)
    ax.axhline(1.0 / C, color='red', linestyle='--', label=f'uniform (1/{C}={1/C:.3f})')
    ax.set_xlabel('Co-occurrence slot index')
    ax.set_ylabel('Mean attention weight')
    ax.set_title('Co-occurrence Attention: Mean weight per slot')
    ax.legend(fontsize=9)

    # 右: 最大注目スロットの分布（どのスロットが最も注目されるか）
    ax = axes[1]
    argmax_slots = all_weights.argmax(axis=1)   # [N]
    counts = np.bincount(argmax_slots, minlength=C)
    ax.bar(slots, counts / counts.sum(), color='#DD8452', alpha=0.8)
    ax.axhline(1.0 / C, color='red', linestyle='--', label=f'uniform')
    ax.set_xlabel('Co-occurrence slot index')
    ax.set_ylabel('Fraction of timesteps where slot is max-attended')
    ax.set_title('Co-occurrence Attention: Argmax slot distribution')
    ax.legend(fontsize=9)

    fig.tight_layout()
    _save_fig(fig, os.path.join(out_dir, 'coattn_weight_distribution.png'))


# ─────────────────────────────────────────────────────────────
# メイン
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--n_batches', type=int, default=200,
                        help='解析に使うバッチ数（デフォルト 200）')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--target', type=str, default='position',
                        choices=['position', 'region'],
                        help='勾配を計算するヘッド（position=index1, region=index0）')
    parser.add_argument('--output_dir', type=str, default=None)
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    # config_snapshot.py で config を上書き（evaluate_only.py と同じ手順）
    checkpoint_dir = os.path.dirname(args.checkpoint)
    snapshot_path  = os.path.join(checkpoint_dir, 'config_snapshot.py')
    if os.path.exists(snapshot_path):
        import importlib.util
        spec = importlib.util.spec_from_file_location('config_snapshot', snapshot_path)
        snap = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(snap)
        overridden = [a for a in dir(snap) if not a.startswith('_') and hasattr(config, a)]
        for a in overridden:
            setattr(config, a, getattr(snap, a))
        force_print(f"[INFO] Loaded config_snapshot.py ({len(overridden)} attrs overridden)")

    # 出力ディレクトリ
    if args.output_dir:
        out_dir = args.output_dir
    else:
        scripts_out = os.path.join('outputs', 'transformer_260625', 'scripts', 'feature_importance')
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_dir = os.path.join(scripts_out, ts)
    os.makedirs(out_dir, exist_ok=True)
    force_print(f"Output dir: {out_dir}")

    # モデルロード
    device = config.DEVICE
    model  = HierarchicalTransformer().to(device)
    ckpt   = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    force_print(f"Loaded checkpoint: epoch={ckpt.get('epoch', -1) + 1}")

    # データローダー
    from transformer_260625.db.connection import get_db_path
    from transformer_260625.db.dataset import create_db_dataloader
    split_map = {'train': 0, 'val': 1, 'test': 2}
    loader = create_db_dataloader(
        db_path=get_db_path(),
        split_type=split_map[args.split],
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        max_cooccurrence=config.MAX_CO_OCCURRENCE,
    )

    target_idx = 1 if args.target == 'position' else 0
    n_feats    = config.NUM_CHEM_FEATURES
    feat_names = _NUM_FEATURE_NAMES[:n_feats]

    # ── 1. Gradient × Input ───────────────────────────────────
    force_print(f"\n[1/3] Gradient × Input on x_num ({args.target} head, {args.n_batches} batches)...")
    importance = compute_grad_x_input(model, loader, target_idx, args.n_batches, device)

    df_num = pd.DataFrame({
        'feature':    feat_names,
        'importance': importance,
    }).sort_values('importance', ascending=False)
    df_num.to_csv(os.path.join(out_dir, 'num_feature_importance.csv'), index=False)
    plot_num_importance(importance, feat_names, out_dir)

    force_print("  Top-10 features:")
    for _, row in df_num.head(10).iterrows():
        force_print(f"    {row['feature']:35s}  {row['importance']:.6f}")

    # ── 2. 埋め込みノルム ─────────────────────────────────────
    force_print("\n[2/3] Embedding weight norms...")
    records = compute_embedding_norms(model)
    df_emb  = pd.DataFrame(records).sort_values('mean_norm', ascending=False)
    df_emb.to_csv(os.path.join(out_dir, 'embedding_norms.csv'), index=False)
    plot_embedding_norms(records, out_dir)

    force_print("  Embedding norms:")
    for _, row in df_emb.iterrows():
        force_print(f"    {row['embedding']:25s}  mean={row['mean_norm']:.4f}  max={row['max_norm']:.4f}")

    # ── 3. Co-occurrence Attention 重み ───────────────────────
    force_print(f"\n[3/3] Co-occurrence Attention weights ({args.n_batches} batches)...")
    all_w = collect_coattn_weights(model, loader, args.n_batches, device)
    if all_w is not None:
        np.save(os.path.join(out_dir, 'coattn_weights.npy'), all_w)
        plot_coattn_distribution(all_w, out_dir)
        mean_w = all_w.mean(axis=0)
        C      = len(mean_w)
        uniform = 1.0 / C
        entropy = -np.sum(mean_w * np.log(mean_w + 1e-10)) / np.log(C)  # 正規化エントロピー
        force_print(f"  Slots: {C}, Uniform baseline: {uniform:.4f}")
        force_print(f"  Normalized entropy of mean attention: {entropy:.4f} (1.0=uniform, 0.0=peaked)")
    else:
        force_print("  No Co-attention weights collected (use_flat_coattn may be active).")

    force_print(f"\nDone. Results saved to {out_dir}")


if __name__ == '__main__':
    main()
