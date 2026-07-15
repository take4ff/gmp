# petra/eval/plot_monthly_hitrate.py — PETRA walk-forwardの月別Hit Rate（末尾のみ）をプロットする。
#
# 本体の monthly_hitrate_all_folds.png と同じレイアウト（月別折れ線 + サンプル数logバー）で、
# PETRAの「変異パス末尾の変異1件」予測精度（提案モデルのposition_hit_rateと対応するタスク）を
# 全フォールド分stitchして可視化する。
#
# split_type_wf 列には一切触れず、日付範囲を直接指定して各フォールドのテストデータを解決する
# （他フォールドの学習が同時に走っていても安全）。
#
# Usage:
#   python -m petra.eval.plot_monthly_hitrate --walk_forward_dir outputs/petra/walk_forward/<timestamp>
import argparse
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from transformer_260707.scripts.eval.walk_forward import FOLDS
from petra import config as petra_config
from petra.dataset import PetraDataset
from petra.eval.eval_tail_by_daterange import (resolve_test_ids_by_daterange, _FixedIdsPetraDataset,
                                          get_raw_path_set)
from petra.main import resolve_vocab_cache_path
from petra.model import PetraDecoder
from petra.tokenizer import MutationTokenizer


@torch.no_grad()
def eval_tail_by_month(model, loader, tokenizer, device, k=1):
    """配列末尾のみの hit(0/1) を月ごとにリスト集計する。

    is_leaked=True（raw_pathが同一foldのtrain窓に既出）のサンプルは month_hits とは別に
    month_hits_nonleaked から除外する。PETRAのCLM設計ではraw_path一致でinput/targetの
    対応も機械的に一致するため、真の汎化性能を見るには non_leaked 側を見る必要がある。
    """
    model.eval()
    context_ids = set(i for i in range(tokenizer.vocab_size) if tokenizer.is_context_token(i))
    context_tensor = torch.tensor(sorted(context_ids), dtype=torch.long, device=device)

    month_hits = defaultdict(list)
    month_hits_nonleaked = defaultdict(list)
    for input_ids, target_ids, _mask, _countries, dates, is_leaked_flags in loader:
        input_ids = input_ids.to(device)
        target_ids_dev = target_ids.to(device)
        logits = model(input_ids)
        topk_preds = logits.topk(k, dim=-1).indices

        is_context = torch.isin(target_ids_dev, context_tensor)
        is_valid = (~is_context) & (target_ids_dev != -100)

        B = target_ids_dev.shape[0]
        for b in range(B):
            valid_positions = torch.nonzero(is_valid[b], as_tuple=True)[0]
            if len(valid_positions) == 0:
                continue
            last_pos = int(valid_positions[-1].item())
            target = target_ids_dev[b, last_pos]
            hit = bool((topk_preds[b, last_pos] == target).any().item())
            date = dates[b]
            if date and len(date) >= 7:
                month_hits[date[:7]].append(1.0 if hit else 0.0)
                if not is_leaked_flags[b]:
                    month_hits_nonleaked[date[:7]].append(1.0 if hit else 0.0)
    return month_hits, month_hits_nonleaked


def find_fold_checkpoints(wf_dir):
    found = {}
    for root, _dirs, files in os.walk(wf_dir):
        if 'best_model.pth' in files:
            m = re.search(r'fold_(\d+)', root)
            if m:
                found[int(m.group(1))] = os.path.join(root, 'best_model.pth')
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--walk_forward_dir', required=True)
    ap.add_argument('--force_cpu', action='store_true')
    ap.add_argument('--output_dir', default=None)
    args = ap.parse_args()

    device = torch.device('cpu' if args.force_cpu or not torch.cuda.is_available()
                          else petra_config.DEVICE)

    from transformer_260707.db.connection import get_db_path
    db_path = get_db_path()
    tokenizer = MutationTokenizer.load(resolve_vocab_cache_path())
    print(f'Vocab size: {tokenizer.vocab_size}, device={device}')

    fold_ckpts = find_fold_checkpoints(args.walk_forward_dir)
    print(f"発見したフォールド: {sorted(fold_ckpts)}"
          f"（未完了フォールドは今回スキップ、後で再実行すれば追加される）")

    all_month_hits = defaultdict(list)
    all_month_hits_nonleaked = defaultdict(list)
    per_fold_summary = []

    for fold_id, train_start, split_date, split_end, desc in FOLDS:
        if fold_id not in fold_ckpts:
            print(f"[SKIP] Fold {fold_id} ({desc}): チェックポイント未生成")
            continue
        print(f"\n=== Fold {fold_id}: {desc} ===")
        ids = resolve_test_ids_by_daterange(db_path, split_date, split_end)
        if not ids:
            continue

        leaked_raw_paths = get_raw_path_set(db_path, train_start, split_date)
        print(f"  [INFO] train窓のユニークraw_path数: {len(leaked_raw_paths):,}（is_leaked判定に使用）")

        ds = _FixedIdsPetraDataset(db_path, tokenizer, ids, max_seq_len=petra_config.MAX_SEQ_LEN,
                                   chunk_size=petra_config.CHUNK_SIZE,
                                   max_history_steps=petra_config.MAX_HISTORY_STEPS,
                                   leaked_raw_paths=leaked_raw_paths)
        loader = DataLoader(ds, batch_size=petra_config.BATCH_SIZE,
                            collate_fn=PetraDataset.collate_fn,
                            num_workers=0, pin_memory=(device.type == 'cuda'))

        model = PetraDecoder(
            vocab_size=tokenizer.vocab_size, d_model=petra_config.D_MODEL,
            n_heads=petra_config.N_HEADS, n_layers=petra_config.N_LAYERS,
            ffn_dim=petra_config.FFN_DIM, max_seq_len=petra_config.MAX_SEQ_LEN,
            dropout=petra_config.DROPOUT,
        ).to(device)
        ckpt = torch.load(fold_ckpts[fold_id], map_location=device, weights_only=True)
        model.load_state_dict(ckpt['model_state_dict'])

        month_hits, month_hits_nonleaked = eval_tail_by_month(model, loader, tokenizer, device, k=1)
        for month, hits in sorted(month_hits.items()):
            all_month_hits[month].extend(hits)
            nonleaked_hits = month_hits_nonleaked.get(month, [])
            all_month_hits_nonleaked[month].extend(nonleaked_hits)
            rate = float(np.mean(hits) * 100)
            nonleaked_rate = float(np.mean(nonleaked_hits) * 100) if nonleaked_hits else float('nan')
            print(f"  {month}: n={len(hits):,}, hit_rate@1={rate:.2f}%  "
                  f"(non_leaked: n={len(nonleaked_hits):,}, hit_rate@1={nonleaked_rate:.2f}%)")
            per_fold_summary.append({'fold': fold_id, 'month': month,
                                     'n_sequences': len(hits), 'hit_rate_pct': rate,
                                     'n_sequences_nonleaked': len(nonleaked_hits),
                                     'hit_rate_pct_nonleaked': nonleaked_rate})
        del model

    if not all_month_hits:
        raise SystemExit("[ERROR] 集計できたフォールドがありません。")

    months = sorted(all_month_hits.keys())
    rates = [float(np.mean(all_month_hits[m]) * 100) for m in months]
    ns = [len(all_month_hits[m]) for m in months]
    rates_nonleaked = [float(np.mean(all_month_hits_nonleaked[m]) * 100)
                       if all_month_hits_nonleaked.get(m) else float('nan') for m in months]
    ns_nonleaked = [len(all_month_hits_nonleaked.get(m, [])) for m in months]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(max(14, len(months) * 0.35), 8),
        gridspec_kw={'height_ratios': [3, 1]}, sharex=True)

    ax1.plot(months, rates, marker='o', color='#c0392b',
             label='PETRA: next-mutation-token top-1 (tail only, all)')
    ax1.plot(months, rates_nonleaked, marker='s', color='#2980b9', linestyle='--',
             label='PETRA: same, non_leaked only (raw_path not seen in train window)')
    ax1.set_ylabel('Hit Rate (%)')
    ax1.set_title('PETRA Walk-forward: Monthly Test Hit Rate (tail-only, next mutation)')
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.bar(months, ns, color='gray', alpha=0.5, label='all')
    ax2.bar(months, ns_nonleaked, color='#2980b9', alpha=0.7, label='non_leaked')
    ax2.set_yscale('log')
    ax2.set_ylabel('num_sequences\n(log)')
    ax2.legend(fontsize=8)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    out_dir = args.output_dir or os.path.join(args.walk_forward_dir, 'monthly_plot')
    os.makedirs(out_dir, exist_ok=True)
    fig_path = os.path.join(out_dir, 'petra_monthly_hitrate.png')
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    df = pd.DataFrame({'month': months, 'hit_rate_pct': rates, 'n_sequences': ns,
                       'hit_rate_pct_nonleaked': rates_nonleaked, 'n_sequences_nonleaked': ns_nonleaked})
    csv_path = os.path.join(out_dir, 'petra_monthly_hitrate.csv')
    df.to_csv(csv_path, index=False)
    pd.DataFrame(per_fold_summary).to_csv(
        os.path.join(out_dir, 'petra_monthly_hitrate_by_fold.csv'), index=False)

    print(f"\n[INFO] Saved: {fig_path}")
    print(f"[INFO] Saved: {csv_path}")


if __name__ == '__main__':
    main()
