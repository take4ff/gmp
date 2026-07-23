# --- transformer_260723/scripts/analysis/verify/verify_position_embedding_norm_confound.py ---
"""pos_embed（ゲノム位置embedding）のL2ノルムが、頻度・ホモプラシー再発回数と
相関しているかを検証する。

背景: verify_coattn_frequency_confound.py で、大きな共起グループを持つ系統
(BA.1/BA.2 等)ではCo-occurrence Attentionの「勝者」が nsp12:P323L・E:T9I・
N:204・S:417 等の、ほぼ全系統に共通する創始(founder)変異に収束することが
分かった。Cross-AttentionのスコアはQK^T内積で決まるため、これらpositionの
embeddingが「頻繁に見た結果ノルムが単に大きく育っている」だけで、内容に関わらず
機械的に勝っている可能性がある（[[ablation-results]]のhydrophobicity/charge
重要度低下調査で確認した「頻度→embedding成長→相対重要度の歪み」と同じ機構の疑い）。

手法:
  1. pos_embed.weight の各行(=各ゲノム位置)のL2ノルムを計算。
  2. reference/homoplasy/site_recurrence.csv の再発回数、および実データでの
     出現回数（attention_by_month.py と同じ方法でテストバッチから集計）と
     突き合わせ、embed_norm との Spearman相関を全ゲノム位置で見る。
  3. verify_coattn_frequency_confound.py で見つかった「偏り大グループの
     argmax常連ポジション」と「偏り小グループのそれ」について、ノルムの
     百分位順位を個別に確認する。

DB読み取り専用。

Usage:
  python -m transformer_260723.scripts.analysis.verify.verify_position_embedding_norm_confound \\
      --checkpoint outputs/transformer_260723/results/walk_forward/20260719_001947/fold_3/20260719_031121/models/best_model.pth \\
      [--split test] [--n_batches 100] [--force_cpu]
"""
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from transformer_260723 import config
from transformer_260723.utils.logging import force_print
from transformer_260723.scripts.analysis.xai import _xai_common as X
from transformer_260723.scripts.analysis.xai.attention_by_month import POS_COL, BASE_BEFORE_COL

# verify_coattn_frequency_confound.py (20260723_130628) の argmax 常連ポジション
CONCENTRATED_HOTSPOTS = {
    21846: 'S:95', 22813: 'S:417', 28883: 'N:204', 10029: 'nsp4:492',
    26270: 'E:9', 28271: 'non_coding9:0', 14408: 'nsp12:323', 12880: 'nsp9:65',
    13195: '?', 21618: '?',
}
UNIFORM_WINNERS = {28461: '?', 29402: '?'}


def _load_recurrence_map():
    df = pd.read_csv(config.HOMOPLASY_CSV)
    return dict(zip(df['position_id'].astype(int), df['recurrence_count'].astype(float)))


def _count_position_occurrences(loader, n_batches, device):
    """attention_by_month.py と同じ方法で、テストバッチ中の各ゲノム位置の出現回数を数える
    （PADスロットを除く、x_cat由来の入力側の出現＝共起集合を含む全変異）。
    """
    occ = defaultdict(int)
    n_processed = 0
    bi = -1
    for bi, batch in enumerate(loader):
        if bi >= n_batches:
            break
        x_cat, _x_num, _mask = batch[0]
        x_cat = x_cat.to(device)
        pos = x_cat[..., POS_COL].cpu().numpy()
        is_pad = (x_cat[..., BASE_BEFORE_COL] == 0).cpu().numpy()
        for p in pos[~is_pad]:
            occ[int(p)] += 1
        n_processed += x_cat.shape[0]
    force_print(f"[INFO] Counted occurrences over {n_processed:,} samples, {bi + 1} batches")
    return occ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--split', default='test', choices=['train', 'val', 'test'])
    ap.add_argument('--n_batches', type=int, default=100)
    ap.add_argument('--force_cpu', action='store_true')
    ap.add_argument('--output_dir', default=None)
    args = ap.parse_args()

    model, loader, device = X.load_model_and_loader(
        args.checkpoint, split=args.split, force_cpu=args.force_cpu)

    pos_embed = model.input_embed.pos_embed.weight.detach().cpu().numpy()  # [VOCAB, DIM]
    norms = np.linalg.norm(pos_embed, axis=1)
    V = norms.shape[0]
    force_print(f"[INFO] pos_embed: vocab_size={V}, dim={pos_embed.shape[1]}")

    recurrence = _load_recurrence_map()
    occ = _count_position_occurrences(loader, args.n_batches, device)

    rows = [{'position': pid, 'embed_norm': float(norms[pid]),
            'recurrence': recurrence.get(pid, 0.0), 'occurrence_count': occ.get(pid, 0)}
           for pid in range(1, V)]  # 0 = padding_idx を除外
    df = pd.DataFrame(rows)
    df = df[(df['recurrence'] > 0) | (df['occurrence_count'] > 0)]  # 実在する位置のみ
    df['norm_percentile'] = df['embed_norm'].rank(pct=True) * 100

    rho_r, p_r = spearmanr(df['embed_norm'], df['recurrence'])
    rho_o, p_o = spearmanr(df['embed_norm'], df['occurrence_count'])
    force_print(f"\n[RESULT] Spearman(embed_norm, homoplasy_recurrence): r={rho_r:.4f}, p={p_r:.2e}  (n={len(df)})")
    force_print(f"[RESULT] Spearman(embed_norm, occurrence_count@{args.n_batches}batches): r={rho_o:.4f}, p={p_o:.2e}")

    def _report(label, mapping):
        force_print(f"\n[INFO] {label}（ノルム百分位）:")
        for pid, gene in mapping.items():
            row = df[df['position'] == pid]
            if len(row):
                r = row.iloc[0]
                force_print(f"  pos={pid:6d} ({gene:16s}) norm={r['embed_norm']:.4f}  "
                            f"percentile={r['norm_percentile']:.1f}%  recurrence={r['recurrence']:.0f}  "
                            f"occurrence={r['occurrence_count']:.0f}")
            else:
                force_print(f"  pos={pid:6d} ({gene:16s}) データなし")

    _report('偏り大グループのargmax常連ポジション', CONCENTRATED_HOTSPOTS)
    _report('偏り小グループのargmax常連ポジション', UNIFORM_WINNERS)

    out_dir = X.make_output_dir('verify_position_embedding_norm_confound', args.output_dir)
    X.save_csv(df.sort_values('embed_norm', ascending=False), out_dir, 'position_embedding_norms.csv')
    X.save_json({
        'checkpoint': args.checkpoint, 'split': args.split, 'n_batches': args.n_batches,
        'spearman_norm_vs_recurrence': {'r': rho_r, 'p': p_r},
        'spearman_norm_vs_occurrence': {'r': rho_o, 'p': p_o},
    }, out_dir, 'meta.json')
    force_print("\n[INFO] Done.")


if __name__ == '__main__':
    main()
