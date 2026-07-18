# petra/eval/plot_monthly_position_tolerance.py — PETRA walk-forwardの月別「位置のみ許容誤差付き
# Top-1正解率」（塩基不一致でも位置が近ければ正解）を集計する。
#
# 既存の plot_monthly_hitrate.py（変異トークン完全一致＝位置＋塩基一致）とは判定基準が異なる
# 別スクリプト（動作実績のある既存コードに触れないため）。tolerance=0 は「予測位置==正解位置」
# （塩基不一致でも正解扱い）で、提案モデルの position_hit_rate と判定基準を揃えてある。
#
# transformer_260707/scripts/analysis/walk_forward/position_tolerance_monthly.py と対になる月別粒度の
# CSVを出力し、同パッケージの plot_position_tolerance_comparison.py で両モデルを重ねてプロットする。
#
# split_type_wf 列には一切触れず、日付範囲を直接指定して各フォールドのテストデータを解決する
# （他フォールドの学習が同時に走っていても安全）。
#
# --- 追加: 「親グループ単位・any-of-set」版（本体の評価方式に揃えた比較用) ---
# 本体(transformer_260707)は分岐(同一 input_path_str から複数の子ノードへ分岐)と共起(同一
# ステップ内の複数同時変異)を、db/dataset.py の collate_fn で1つの評価単位にまとめ、
# 「予測1個がその評価単位の正解集合(分岐K個×共起M個)のいずれかに当たればヒット」という
# any-of-set 方式で評価している(evaluate.py)。実測ではこの分岐はサンプルの約87%に該当する
# ため(2026-07-18 branching_cooccurrence_frequency.py調査)、無視できない構造差になっている。
# 一方、本スクリプトの既存版(eval_position_tolerance_by_month)は枝(=raw_path)ごとに独立した
# 別サンプルとして評価しており、単一ターゲットとの一致のみを見る。
#
# 以下の resolve_test_groups_by_daterange / _GroupedRepDataset / eval_position_tolerance_by_month_grouped
# は、本体と同じ input_path_str グルーピング + any-of-set 判定を PETRA 側にも適用したもの。
# 既存の（枝ごと単一ターゲット）評価は変更せず残し、本関数群はそれと並行して追加の
# petra_position_tolerance_monthly_grouped.csv を出力する（本体との厳密な比較に使う）。
# グループの代表サンプル1件だけをモデルに通す（本体の collate_fn が代表サンプルの
# x_cat/x_num のみを使うのと同じ設計）ため、推論コストはむしろ既存版より軽い。
#
# Usage:
#   python -m petra.eval.plot_monthly_position_tolerance \
#       --walk_forward_dir outputs/petra/walk_forward/<timestamp> --tolerances 0 5 10 50
import argparse
import os
import pickle
import re
from collections import defaultdict

import duckdb
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, IterableDataset

from transformer_260707 import config as main_config
from transformer_260707.scripts.eval.walk_forward import FOLDS
from petra import config as petra_config
from petra.dataset import PetraDataset
from petra.eval.eval_tail_by_daterange import (resolve_test_ids_by_daterange, _FixedIdsPetraDataset,
                                          get_raw_path_set)
from petra.eval.evaluate import build_position_lookup
from petra.main import resolve_vocab_cache_path
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
            if not _YEARMONTH_RE.match(ym):
                continue  # 既知issue: collection_date不正値（'2022/2024'等）由来の異常な月ラベルを除外
            for tol in tolerances:
                hit = 1.0 if (dist is not None and dist <= tol) else 0.0
                month_hits[tol][ym].append(hit)
                if not is_leaked_flags[b]:
                    month_hits_nonleaked[tol][ym].append(hit)
    return month_hits, month_hits_nonleaked


# ──────────────────────────────────────────────────────────────────────────
# 「親グループ単位・any-of-set」版（本体の評価方式に揃えた比較用）
# ──────────────────────────────────────────────────────────────────────────

_MUT_POS_RE = re.compile(r'[A-Za-z](\d+)[A-Za-z]')

# 既知issue: reference/sequences-241017_2.csv の一部レコード（Zimbabwe/Harare 217件）が
# Collection_Date='2022/2024' という不正値を持ち、date[:7] が '2022/20' という異常な月ラベルに
# なる（本体main.py・entropy_fano_by_month.py 等と同じ対応。data_quality_collection_date_bug 参照）。
_YEARMONTH_RE = re.compile(r'^\d{4}-\d{2}$')


def _target_positions_from_raw_path(raw_path):
    """raw_path 末尾ステップ（','区切りの共起変異全て）から塩基位置集合を得る。
    本体側の get_db_lineage_target_entropy 等と同じ正規表現でregex抽出する
    （トークン化を経由せず raw_path 文字列から直接。共起M個を全て含む）。
    """
    last_step = raw_path.split('>')[-1]
    positions = set()
    for mut in last_step.split(','):
        m = _MUT_POS_RE.search(mut.strip())
        if m:
            positions.add(int(m.group(1)))
    return positions


def resolve_test_groups_by_daterange(db_path, split_date, split_end):
    """本体の分岐グルーピング（同一 input_path_str = raw_path から末尾ステップを除いた履歴）に
    揃えて、フォールドのtestサンプルを親グループ単位にまとめる（読み取り専用）。

    Returns: dict input_path_str -> list of (sample_id, raw_path, collection_date)
    """
    con = duckdb.connect(db_path, read_only=True)
    where = "collection_date IS NOT NULL AND collection_date != '' "
    where += "AND RPAD(collection_date, 10, '-01-01') >= ? "
    params = [split_date]
    if split_end:
        where += "AND RPAD(collection_date, 10, '-01-01') < ? "
        params.append(split_end)
    where += "AND max_cooccurrence <= ?"
    params.append(main_config.MAX_CO_OCCURRENCE)

    rows = con.execute(
        f"SELECT s.sample_id, s.raw_path, s.collection_date, l.targets FROM samples s "
        f"JOIN labels l ON s.sample_id = l.sample_id WHERE {where}",
        params,
    ).fetchall()
    con.close()

    eval_max_y_co = main_config.EVAL_MAX_Y_CO_OCCURRENCE
    excluded_y = 0
    groups = defaultdict(list)
    for sample_id, raw_path, cdate, targets_blob in rows:
        if eval_max_y_co is not None:
            targets = pickle.loads(targets_blob)
            if len(targets) > eval_max_y_co:
                excluded_y += 1
                continue
        parts = raw_path.split('>')
        input_path_str = '>'.join(parts[:-1]) if len(parts) > 1 else ''
        groups[input_path_str].append((sample_id, raw_path, cdate))

    n_branch_groups = sum(1 for members in groups.values() if len(members) >= 2)
    print(f"[INFO] 日付範囲 [{split_date}, {split_end}) : 生{len(rows):,}件 -> "
          f"Y共起除外{excluded_y:,}件 -> 親グループ数{len(groups):,}件 "
          f"（うち分岐(K>=2)グループ{n_branch_groups:,}件）")
    return groups


class _GroupedRepDataset(IterableDataset):
    """親グループの代表サンプル1件のみをモデルに通すためのデータセット。

    本体の collate_fn がグループ代表の x_cat/x_num のみを使い、正解はグループ全体の
    フラット集合を使うのと同じ設計。target_positions/is_leaked は事前計算済みの値を
    そのままitemに載せて渡す（target_ids から導出しない）。
    """

    def __init__(self, db_path, tokenizer, rep_items, max_seq_len=512, chunk_size=2000,
                 max_history_steps=None):
        # rep_items: list of (sample_id, target_positions: set[int], is_leaked: bool)
        self.db_path = db_path
        self.tokenizer = tokenizer
        self.rep_items = rep_items
        self.max_seq_len = max_seq_len
        self.chunk_size = chunk_size
        self.max_history_steps = max_history_steps

    def __len__(self):
        return len(self.rep_items)

    def __iter__(self):
        con = duckdb.connect(self.db_path, read_only=True)
        has_country = PetraDataset._has_country_col(con)
        country_sel = ', s.country' if has_country else ''

        meta_by_id = {sid: (tp, leaked) for sid, tp, leaked in self.rep_items}
        ids = [sid for sid, _tp, _leaked in self.rep_items]

        for i in range(0, len(ids), self.chunk_size):
            chunk = ids[i:i + self.chunk_size]
            placeholders = ','.join(['?'] * len(chunk))
            rows = con.execute(f"""
                SELECT s.sample_id, s.raw_path, s.collection_date, st.clade{country_sel}
                FROM samples s
                JOIN strains st ON s.strain_id = st.strain_id
                WHERE s.sample_id IN ({placeholders})
            """, chunk).fetchall()

            for row in rows:
                sample_id, raw_path, cdate, clade = row[0], row[1], row[2], row[3]
                country = row[4] if has_country else None
                target_positions, is_leaked = meta_by_id[sample_id]
                # 代表サンプル自身の末尾ステップ（','区切りの共起変異数）のトークン数。
                # これを除いた位置が「親グループの共有コンテキスト直後」の境界になる。
                n_tail_tokens = len(raw_path.split('>')[-1].split(','))
                token_ids = self.tokenizer.encode(raw_path, cdate, clade, country,
                                                  max_history_steps=self.max_history_steps)
                if len(token_ids) > self.max_seq_len:
                    token_ids = token_ids[:self.max_seq_len - 1] + [self.tokenizer.eos_id]
                input_ids = token_ids[:-1]
                target_ids = token_ids[1:]
                yield {
                    'input_ids': torch.tensor(input_ids, dtype=torch.long),
                    'target_ids': torch.tensor(target_ids, dtype=torch.long),
                    'length': len(input_ids),
                    'collection_date': cdate,
                    'is_leaked': is_leaked,
                    'target_positions': target_positions,
                    'n_tail_tokens': n_tail_tokens,
                }
        con.close()


def _grouped_collate_fn(batch):
    max_len = max(item['length'] for item in batch)
    B = len(batch)
    input_ids = torch.zeros(B, max_len, dtype=torch.long)
    target_ids = torch.full((B, max_len), -100, dtype=torch.long)
    for i, item in enumerate(batch):
        n = item['length']
        input_ids[i, :n] = item['input_ids']
        target_ids[i, :n] = item['target_ids']
    dates = [item['collection_date'] for item in batch]
    is_leaked = [item['is_leaked'] for item in batch]
    target_positions = [item['target_positions'] for item in batch]
    n_tail_tokens = [item['n_tail_tokens'] for item in batch]
    return input_ids, target_ids, dates, is_leaked, target_positions, n_tail_tokens


@torch.no_grad()
def eval_position_tolerance_by_month_grouped(model, loader, tokenizer, device, position_lookup, tolerances):
    """親グループ単位・any-of-set版の月別集計。各グループの代表サンプルについて、
    「親グループの共有コンテキスト直後」＝代表サンプル自身の末尾ステップ（共起変異）を
    除いた最後の有効（非コンテキスト）トークン位置での top1予測が、グループ全体
    （分岐K個×共起M個）の正解位置集合のいずれかに入っていれば tolerance範囲内でヒットとする
    （本体 evaluate.py の hit_position 判定と同じ基準）。

    既存の eval_position_tolerance_by_month と同様、EOS・日付・国・クレード等の
    コンテキストトークンは tokenizer.is_context_token で除外してから有効位置を数える
    （これをしないと末尾トークンがEOS等になり、無意味な予測を評価してしまう）。
    """
    model.eval()
    context_ids = set(i for i in range(tokenizer.vocab_size) if tokenizer.is_context_token(i))
    context_tensor = torch.tensor(sorted(context_ids), dtype=torch.long, device=device)

    month_hits = {tol: defaultdict(list) for tol in tolerances}
    month_hits_nonleaked = {tol: defaultdict(list) for tol in tolerances}

    for input_ids, target_ids, dates, is_leaked_flags, target_positions_batch, n_tail_tokens_batch in loader:
        input_ids = input_ids.to(device)
        target_ids_dev = target_ids.to(device)
        logits = model(input_ids)
        top1_preds = logits.argmax(dim=-1)          # B, T

        is_context = torch.isin(target_ids_dev, context_tensor)
        is_valid = (~is_context) & (target_ids_dev != -100)

        B = input_ids.shape[0]
        for b in range(B):
            target_positions = target_positions_batch[b]
            if not target_positions:
                continue
            valid_positions = torch.nonzero(is_valid[b], as_tuple=True)[0]
            n_tail = n_tail_tokens_batch[b]
            if n_tail < 1 or len(valid_positions) < n_tail:
                continue  # 想定外（代表サンプルの末尾ステップに変異トークンが無い）
            # 末尾ステップ(共起含むn_tail個)の直前ではなく、その最初のトークン位置を予測する
            # 位置を境界とする（valid_positionsの末尾n_tail個が代表サンプル自身の末尾ステップ）。
            last_pos = int(valid_positions[-n_tail].item())
            pred_tok = int(top1_preds[b, last_pos].item())
            pred_pos = position_lookup.get(pred_tok)

            date = dates[b]
            if not date or len(date) < 7:
                continue
            ym = date[:7]
            if not _YEARMONTH_RE.match(ym):
                continue  # 既知issue: collection_date不正値（'2022/2024'等）由来の異常な月ラベルを除外
            for tol in tolerances:
                hit = 1.0 if (pred_pos is not None
                              and any(abs(pred_pos - tp) <= tol for tp in target_positions)) else 0.0
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
    tokenizer = MutationTokenizer.load(resolve_vocab_cache_path())
    position_lookup = build_position_lookup(tokenizer)
    print(f'Vocab size: {tokenizer.vocab_size}, device={device}, '
          f'tolerances={args.tolerances}, position_lookup size={len(position_lookup):,}')

    fold_ckpts = find_fold_checkpoints(args.walk_forward_dir)
    print(f"発見したフォールド: {sorted(fold_ckpts)}")

    all_month_hits = {tol: defaultdict(list) for tol in args.tolerances}
    all_month_hits_nonleaked = {tol: defaultdict(list) for tol in args.tolerances}
    per_fold_rows = []

    # 親グループ単位・any-of-set版（本体の評価方式に揃えた比較用）の集計先
    all_g_month_hits = {tol: defaultdict(list) for tol in args.tolerances}
    all_g_month_hits_nonleaked = {tol: defaultdict(list) for tol in args.tolerances}
    per_fold_rows_grouped = []

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

        # --- 親グループ単位・any-of-set版（本体の評価方式に揃えた比較用） ---
        groups = resolve_test_groups_by_daterange(db_path, split_date, split_end)
        rep_items = []
        for input_path_str, members in groups.items():
            rep_id, rep_raw_path, _rep_cdate = members[0]
            target_positions = set()
            for _sid, member_raw_path, _cdate in members:
                target_positions |= _target_positions_from_raw_path(member_raw_path)
            is_leaked = bool(leaked_raw_paths) and rep_raw_path in leaked_raw_paths
            rep_items.append((rep_id, target_positions, is_leaked))

        grouped_ds = _GroupedRepDataset(db_path, tokenizer, rep_items, max_seq_len=petra_config.MAX_SEQ_LEN,
                                        chunk_size=petra_config.CHUNK_SIZE,
                                        max_history_steps=petra_config.MAX_HISTORY_STEPS)
        grouped_loader = DataLoader(grouped_ds, batch_size=petra_config.BATCH_SIZE,
                                    collate_fn=_grouped_collate_fn,
                                    num_workers=0, pin_memory=(device.type == 'cuda'))
        g_month_hits, g_month_hits_nonleaked = eval_position_tolerance_by_month_grouped(
            model, grouped_loader, tokenizer, device, position_lookup, args.tolerances)

        for ym, hits in sorted(g_month_hits[base_tol].items()):
            for tol in args.tolerances:
                all_g_month_hits[tol][ym].extend(g_month_hits[tol][ym])
                all_g_month_hits_nonleaked[tol][ym].extend(g_month_hits_nonleaked[tol].get(ym, []))
            nl_hits_g = g_month_hits_nonleaked[base_tol].get(ym, [])
            row_g = {'fold': fold_id, 'month': ym, 'n_groups': len(hits),
                     'n_groups_nonleaked': len(nl_hits_g)}
            for tol in args.tolerances:
                row_g[f'hit_rate_tol{tol}'] = float(np.mean(g_month_hits[tol][ym]) * 100)
                nl = g_month_hits_nonleaked[tol].get(ym, [])
                row_g[f'hit_rate_tol{tol}_nonleaked'] = float(np.mean(nl) * 100) if nl else float('nan')
            per_fold_rows_grouped.append(row_g)
            print(f"  [grouped any-of-set] {ym}: n_groups={len(hits):,} (non_leaked n={len(nl_hits_g):,})  " +
                  "  ".join(f"tol{tol}={row_g[f'hit_rate_tol{tol}']:.2f}%/"
                            f"{row_g[f'hit_rate_tol{tol}_nonleaked']:.2f}%" for tol in args.tolerances))
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

    # --- 親グループ単位・any-of-set版（本体の評価方式に揃えた比較用） ---
    if any(all_g_month_hits[args.tolerances[0]]):
        months_g = sorted(all_g_month_hits[args.tolerances[0]].keys())
        rows_g = []
        for ym in months_g:
            row_g = {'month': ym, 'n_groups': len(all_g_month_hits[args.tolerances[0]][ym]),
                     'n_groups_nonleaked': len(all_g_month_hits_nonleaked[args.tolerances[0]].get(ym, []))}
            for tol in args.tolerances:
                row_g[f'hit_rate_tol{tol}'] = float(np.mean(all_g_month_hits[tol][ym]) * 100)
                nl = all_g_month_hits_nonleaked[tol].get(ym, [])
                row_g[f'hit_rate_tol{tol}_nonleaked'] = float(np.mean(nl) * 100) if nl else float('nan')
            rows_g.append(row_g)
        df_grouped = pd.DataFrame(rows_g)

        csv_path_grouped = os.path.join(out_dir, 'petra_position_tolerance_monthly_grouped.csv')
        df_grouped.to_csv(csv_path_grouped, index=False)
        pd.DataFrame(per_fold_rows_grouped).to_csv(
            os.path.join(out_dir, 'petra_position_tolerance_monthly_by_fold_grouped.csv'), index=False)
        print(f"\n[INFO] Saved (grouped any-of-set, 本体と同じ評価方式): {csv_path_grouped}")
        print(df_grouped.to_string(index=False))


if __name__ == '__main__':
    main()
