# petra/eval/plot_monthly_position_tolerance.py — PETRA walk-forwardの月別「位置のみ許容誤差付き
# Top-1正解率」（塩基不一致でも位置が近ければ正解）を集計する。
#
# 既存の plot_monthly_hitrate.py（変異トークン完全一致＝位置＋塩基一致）とは判定基準が異なる
# 別スクリプト（動作実績のある既存コードに触れないため）。tolerance=0 は「予測位置==正解位置」
# （塩基不一致でも正解扱い）で、提案モデルの position_hit_rate と判定基準を揃えてある。
#
# transformer_260707/scripts/analysis/position_tolerance_monthly.py と対になる月別粒度の
# CSVを出力し、同パッケージの plot_position_tolerance_comparison.py で両モデルを重ねてプロットする。
#
# split_type_wf 列には一切触れず、日付範囲を直接指定して各フォールドのテストデータを解決する
# （他フォールドの学習が同時に走っていても安全）。
#
# Usage:
#   python -m petra.eval.plot_monthly_position_tolerance \
#       --walk_forward_dir outputs/petra/walk_forward/<timestamp> --tolerances 0 5 10 50
import argparse
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from transformer_260707.scripts.eval.walk_forward import FOLDS
from petra import config as petra_config
from petra.dataset import PetraDataset
from petra.eval.eval_tail_by_daterange import (resolve_test_ids_by_daterange, _FixedIdsPetraDataset,
                                          get_raw_path_set)
from petra.eval.evaluate import build_position_lookup
from petra.model import PetraDecoder
from petra.tokenizer import MutationTokenizer
from petra.eval.plot_monthly_hitrate import find_fold_checkpoints


@torch.no_grad()
def eval_position_tolerance_by_month(model, loader, tokenizer, device, position_lookup, tolerances):
    """配列末尾のみ（提案モデルのposition_hit_rateと対応する「次の変異1件」予測タスク）を対象に、
    位置のみの許容誤差付きTop-1正解率(0/1)を月ごとにリスト集計する。is_leaked=Trueのサンプルは
    別途 non_leaked から除外する。
    """
    model.eval()
    context_ids = set(i for i in range(tokenizer.vocab_size) if tokenizer.is_context_token(i))
    context_tensor = torch.tensor(sorted(context_ids), dtype=torch.long, device=device)

    month_hits = {tol: defaultdict(list) for tol in tolerances}
    month_hits_nonleaked = {tol: defaultdict(list) for tol in tolerances}

    for input_ids, target_ids, _mask, _countries, dates, is_leaked_flags in loader:
        input_ids = input_ids.to(device)
        target_ids_dev = target_ids.to(device)
        logits = model(input_ids)
        top1_preds = logits.argmax(dim=-1)          # B, T

        is_context = torch.isin(target_ids_dev, context_tensor)
        is_valid = (~is_context) & (target_ids_dev != -100)

        B = target_ids_dev.shape[0]
        for b in range(B):
            valid_positions = torch.nonzero(is_valid[b], as_tuple=True)[0]
            if len(valid_positions) == 0:
                continue
            last_pos = int(valid_positions[-1].item())
            target = int(target_ids_dev[b, last_pos].item())
            target_pos = position_lookup.get(target)
            if target_pos is None:
                continue  # ターゲットが変異トークンでない（起こらないはずだが念のため）

            pred_tok = int(top1_preds[b, last_pos].item())
            pred_pos = position_lookup.get(pred_tok)
            dist = abs(pred_pos - target_pos) if pred_pos is not None else None

            date = dates[b]
            if not date or len(date) < 7:
                continue
            ym = date[:7]
            for tol in tolerances:
                hit = 1.0 if (dist is not None and dist <= tol) else 0.0
                month_hits[tol][ym].append(hit)
                if not is_leaked_flags[b]:
                    month_hits_nonleaked[tol][ym].append(hit)
    return month_hits, month_hits_nonleaked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--walk_forward_dir', required=True)
    ap.add_argument('--tolerances', type=int, nargs='+', default=[0, 5, 10, 50])
    ap.add_argument('--force_cpu', action='store_true')
    ap.add_argument('--output_dir', default=None)
    args = ap.parse_args()

    device = torch.device('cpu' if args.force_cpu or not torch.cuda.is_available()
                          else petra_config.DEVICE)

    from transformer_260707.db.connection import get_db_path
    db_path = get_db_path()
    tokenizer = MutationTokenizer.load(petra_config.VOCAB_CACHE)
    position_lookup = build_position_lookup(tokenizer)
    print(f'Vocab size: {tokenizer.vocab_size}, device={device}, '
          f'tolerances={args.tolerances}, position_lookup size={len(position_lookup):,}')

    fold_ckpts = find_fold_checkpoints(args.walk_forward_dir)
    print(f"発見したフォールド: {sorted(fold_ckpts)}")

    all_month_hits = {tol: defaultdict(list) for tol in args.tolerances}
    all_month_hits_nonleaked = {tol: defaultdict(list) for tol in args.tolerances}
    per_fold_rows = []

    for fold_id, train_start, split_date, split_end, desc in FOLDS:
        if fold_id not in fold_ckpts:
            print(f"[SKIP] Fold {fold_id} ({desc}): チェックポイント未生成")
            continue
        print(f"\n=== Fold {fold_id}: {desc} ===")
        ids = resolve_test_ids_by_daterange(db_path, split_date, split_end)
        if not ids:
            continue
        leaked_raw_paths = get_raw_path_set(db_path, train_start, split_date)

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

        month_hits, month_hits_nonleaked = eval_position_tolerance_by_month(
            model, loader, tokenizer, device, position_lookup, args.tolerances)

        base_tol = args.tolerances[0]
        for ym, hits in sorted(month_hits[base_tol].items()):
            for tol in args.tolerances:
                all_month_hits[tol][ym].extend(month_hits[tol][ym])
                all_month_hits_nonleaked[tol][ym].extend(month_hits_nonleaked[tol].get(ym, []))
            nl_hits = month_hits_nonleaked[base_tol].get(ym, [])
            row = {'fold': fold_id, 'month': ym, 'n_sequences': len(hits),
                   'n_sequences_nonleaked': len(nl_hits)}
            for tol in args.tolerances:
                row[f'hit_rate_tol{tol}'] = float(np.mean(month_hits[tol][ym]) * 100)
                nl = month_hits_nonleaked[tol].get(ym, [])
                row[f'hit_rate_tol{tol}_nonleaked'] = float(np.mean(nl) * 100) if nl else float('nan')
            per_fold_rows.append(row)
            print(f"  {ym}: n={len(hits):,} (non_leaked n={len(nl_hits):,})  " +
                  "  ".join(f"tol{tol}={row[f'hit_rate_tol{tol}']:.2f}%/"
                            f"{row[f'hit_rate_tol{tol}_nonleaked']:.2f}%" for tol in args.tolerances))
        del model

    if not any(all_month_hits[args.tolerances[0]]):
        raise SystemExit("[ERROR] 集計できたフォールドがありません。")

    months = sorted(all_month_hits[args.tolerances[0]].keys())
    rows = []
    for ym in months:
        row = {'month': ym, 'n_sequences': len(all_month_hits[args.tolerances[0]][ym]),
               'n_sequences_nonleaked': len(all_month_hits_nonleaked[args.tolerances[0]].get(ym, []))}
        for tol in args.tolerances:
            row[f'hit_rate_tol{tol}'] = float(np.mean(all_month_hits[tol][ym]) * 100)
            nl = all_month_hits_nonleaked[tol].get(ym, [])
            row[f'hit_rate_tol{tol}_nonleaked'] = float(np.mean(nl) * 100) if nl else float('nan')
        rows.append(row)
    df = pd.DataFrame(rows)

    out_dir = args.output_dir or os.path.join(args.walk_forward_dir, 'monthly_plot')
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, 'petra_position_tolerance_monthly.csv')
    df.to_csv(csv_path, index=False)
    pd.DataFrame(per_fold_rows).to_csv(
        os.path.join(out_dir, 'petra_position_tolerance_monthly_by_fold.csv'), index=False)
    print(f"\n[INFO] Saved: {csv_path}")
    print(df.to_string(index=False))


if __name__ == '__main__':
    main()
