# petra/eval_tail_by_daterange.py — 変異パス末尾のみでの Recall@K を、日付範囲で
# 直接指定したフォールドのテストデータに対して評価する。
#
# 通常の walk_forward 評価は共有DB列 split_type_wf を使うが、他フォールドの学習が
# 進行中のときにこれを触ると競合するため、本スクリプトは split_type_wf に一切触れず
# raw collection_date のみで対象サンプルを直接クエリする（読み取り専用、他プロセスの
# 学習と安全に並行実行できる）。
#
# 本体 DBIterableDataset._get_sample_ids() と同じフィルタ（USE_UNIQUE_FILTER・
# MAX_CO_OCCURRENCE・EVAL_MAX_Y_CO_OCCURRENCE）を、split列ではなく日付範囲を条件に
# 手動で再現する。
#
# Usage:
#   python -m petra.eval_tail_by_daterange --checkpoint outputs/petra/walk_forward/.../fold_1/best_model.pth \
#       --train_start "" --split_date 2021-01-01 --split_end 2021-07-01
import argparse
import pickle

import duckdb
import torch
from torch.utils.data import DataLoader, IterableDataset

from transformer_260707 import config as main_config
from petra import config as petra_config
from petra.dataset import PetraDataset
from petra.evaluate import evaluate_recall_topk, evaluate_recall_topk_tail_only, PetraRepresentativenessWeights
from petra.model import PetraDecoder
from petra.tokenizer import MutationTokenizer


def resolve_test_ids_by_daterange(db_path, split_date, split_end):
    """[split_date, split_end) の collection_date を持つサンプルIDを、本体と同じフィルタ
    （dedup・MAX_CO_OCCURRENCE・EVAL_MAX_Y_CO_OCCURRENCE）込みで解決する（読み取り専用）。
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
        f"SELECT s.sample_id, s.raw_path, l.targets FROM samples s "
        f"JOIN labels l ON s.sample_id = l.sample_id WHERE {where}",
        params,
    ).fetchall()
    con.close()

    eval_max_y_co = main_config.EVAL_MAX_Y_CO_OCCURRENCE
    excluded_y = 0
    seen_paths = set()
    ids = []
    for sample_id, raw_path, targets_blob in rows:
        if eval_max_y_co is not None:
            targets = pickle.loads(targets_blob)
            if len(targets) > eval_max_y_co:
                excluded_y += 1
                continue
        if raw_path in seen_paths:
            continue
        seen_paths.add(raw_path)
        ids.append(sample_id)

    print(f"[INFO] 日付範囲 [{split_date}, {split_end}) : 生{len(rows):,}件 -> "
          f"Y共起除外{excluded_y:,}件 -> 重複除去後{len(ids):,}件")
    return ids


class _FixedIdsPetraDataset(IterableDataset):
    """PetraDataset とほぼ同じだが、あらかじめ解決済みの sample_id リストをそのまま使う版。"""

    def __init__(self, db_path, tokenizer, ids, max_seq_len=512, chunk_size=2000,
                 max_history_steps=None):
        self.db_path = db_path
        self.tokenizer = tokenizer
        self.ids = ids
        self.max_seq_len = max_seq_len
        self.chunk_size = chunk_size
        self.max_history_steps = max_history_steps

    def __len__(self):
        return len(self.ids)

    def __iter__(self):
        con = duckdb.connect(self.db_path, read_only=True)
        has_country = PetraDataset._has_country_col(con)
        country_sel = ', s.country' if has_country else ''

        for i in range(0, len(self.ids), self.chunk_size):
            chunk = self.ids[i:i + self.chunk_size]
            placeholders = ','.join(['?'] * len(chunk))
            rows = con.execute(f"""
                SELECT s.raw_path, s.collection_date, st.clade{country_sel}
                FROM samples s
                JOIN strains st ON s.strain_id = st.strain_id
                WHERE s.sample_id IN ({placeholders})
            """, chunk).fetchall()

            for row in rows:
                raw_path, cdate, clade = row[0], row[1], row[2]
                country = row[3] if has_country else None
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
                    'country': country,
                    'collection_date': cdate,
                }
        con.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--split_date', required=True)
    ap.add_argument('--split_end', default=None)
    ap.add_argument('--force_cpu', action='store_true')
    args = ap.parse_args()

    if args.force_cpu or not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device(petra_config.DEVICE)

    from transformer_260707.db.connection import get_db_path
    db_path = get_db_path()

    tokenizer = MutationTokenizer.load(petra_config.VOCAB_CACHE)
    print(f'Vocab size: {tokenizer.vocab_size}')

    ids = resolve_test_ids_by_daterange(db_path, args.split_date, args.split_end)
    test_ds = _FixedIdsPetraDataset(db_path, tokenizer, ids, max_seq_len=petra_config.MAX_SEQ_LEN,
                                    chunk_size=petra_config.CHUNK_SIZE,
                                    max_history_steps=petra_config.MAX_HISTORY_STEPS)
    test_loader = DataLoader(test_ds, batch_size=petra_config.BATCH_SIZE,
                             collate_fn=PetraDataset.collate_fn,
                             num_workers=0, pin_memory=(device.type == 'cuda'))

    model = PetraDecoder(
        vocab_size=tokenizer.vocab_size, d_model=petra_config.D_MODEL,
        n_heads=petra_config.N_HEADS, n_layers=petra_config.N_LAYERS,
        ffn_dim=petra_config.FFN_DIM, max_seq_len=petra_config.MAX_SEQ_LEN,
        dropout=petra_config.DROPOUT,
    ).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f'Loaded checkpoint: {args.checkpoint} (epoch={ckpt.get("epoch")}, val_loss={ckpt.get("val_loss"):.4f})')

    weights = PetraRepresentativenessWeights(db_path, split_type=2)
    from petra.evaluate import build_spike_token_ids, build_region_token_ids
    spike_ids = build_spike_token_ids(tokenizer)
    region_ids = build_region_token_ids(tokenizer)

    print('\n--- 通常評価（配列全体の平均、参考） ---')
    full = evaluate_recall_topk(model, test_loader, tokenizer, device, petra_config.TOP_K_LIST,
                                weights=weights, spike_token_ids=spike_ids, region_token_ids=region_ids)
    for k in petra_config.TOP_K_LIST:
        print(f'  @{k:3d}: Average={full["average"][f"recall@{k}"]:.4f}  Weighted={full["weighted"][f"recall@{k}"]:.4f}')
    if 'spike' in full:
        print(f'  [スパイク限定] n={full["spike"]["n_sequences"]:,}')
        for k in petra_config.TOP_K_LIST:
            print(f'    @{k:3d}: Average={full["spike"]["average"][f"recall@{k}"]:.4f}  '
                  f'Weighted={full["spike"]["weighted"][f"recall@{k}"]:.4f}')
    if 'region' in full:
        print(f'  [region限定] n={full["region"]["n_sequences"]:,}')
        for k in petra_config.TOP_K_LIST:
            print(f'    @{k:3d}: Average={full["region"]["average"][f"recall@{k}"]:.4f}  '
                  f'Weighted={full["region"]["weighted"][f"recall@{k}"]:.4f}')

    print('\n--- 末尾のみ評価（提案モデルの position_hit_rate と対応） ---')
    tail = evaluate_recall_topk_tail_only(model, test_loader, tokenizer, device, petra_config.TOP_K_LIST,
                                          weights=weights, spike_token_ids=spike_ids, region_token_ids=region_ids)
    print(f'  n_sequences={tail["n_sequences"]:,}')
    for k in petra_config.TOP_K_LIST:
        print(f'  @{k:3d}: Average={tail["average"][f"recall@{k}"]:.4f}  Weighted={tail["weighted"][f"recall@{k}"]:.4f}')
    if 'spike' in tail:
        print(f'  [スパイク限定] n={tail["spike"]["n_sequences"]:,}')
        for k in petra_config.TOP_K_LIST:
            print(f'    @{k:3d}: Average={tail["spike"]["average"][f"recall@{k}"]:.4f}  '
                  f'Weighted={tail["spike"]["weighted"][f"recall@{k}"]:.4f}')
    if 'region' in tail:
        print(f'  [region限定] n={tail["region"]["n_sequences"]:,}（本体の region_hit_rate と対応）')
        for k in petra_config.TOP_K_LIST:
            print(f'    @{k:3d}: Average={tail["region"]["average"][f"recall@{k}"]:.4f}  '
                  f'Weighted={tail["region"]["weighted"][f"recall@{k}"]:.4f}')


if __name__ == '__main__':
    main()
