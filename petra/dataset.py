# petra/dataset.py — DuckDB からトークン系列をストリーム読み込み

import random
import torch
from torch.utils.data import IterableDataset, get_worker_info
import duckdb


class PetraDataset(IterableDataset):
    """DuckDB から変異パスをトークン列として読み込む IterableDataset。

    各サンプルの形式:
        input_ids  : [BOS] [YEAR] [MONTH] [CLADE] mut1 mut2 ...       (T-1 tokens)
        target_ids : [YEAR] [MONTH] [CLADE] mut1 mut2 ... [EOS]       (T-1 tokens, CLM target)
        パディング位置の target_ids = -100 (CrossEntropy で ignore)
    """

    def __init__(self, db_path: str, tokenizer, split_type: int,
                 max_seq_len: int = 512, shuffle: bool = False,
                 chunk_size: int = 2000, split_col: str = 'split_type',
                 max_history_steps: int = None):
        self.db_path = db_path
        self.tokenizer = tokenizer
        self.split_type = split_type
        self.max_seq_len = max_seq_len
        self.shuffle = shuffle
        self.chunk_size = chunk_size
        self.split_col = split_col
        self.max_history_steps = max_history_steps
        # 「同一データでの厳密比較」を保証するため、独自にフィルタを再実装するのではなく
        # 本体(transformer_260707/db/dataset.py)の DBIterableDataset が実際に解決する
        # sample_id 集合をそのまま再利用する。これにより USE_UNIQUE_FILTER（重複排除）だけでなく
        # MAX_CO_OCCURRENCE（共起数上限）・EVAL_MAX_Y_CO_OCCURRENCE（評価時ターゲット共起上限）・
        # USE_TRAIN_ENTROPY_FILTER 等、本体の全フィルタリングロジックと完全に一致する。
        self._ids = self._resolve_ids_from_main_model()
        self._length = len(self._ids)

    def _resolve_ids_from_main_model(self) -> list[int]:
        from transformer_260707 import config as main_config
        from transformer_260707.db.dataset import DBIterableDataset

        main_ds = DBIterableDataset(
            db_path=self.db_path,
            split_type=self.split_type,
            max_cooccurrence=main_config.MAX_CO_OCCURRENCE,
            split_col_override=self.split_col,
        )
        ids = main_ds.sample_ids
        print(f"[INFO] PetraDataset(split={self.split_type}): 本体 DBIterableDataset と同一の "
              f"{len(ids):,}件を使用（USE_UNIQUE_FILTER/MAX_CO_OCCURRENCE/"
              f"EVAL_MAX_Y_CO_OCCURRENCE等、本体の全フィルタ込み）")
        return ids

    @staticmethod
    def _has_country_col(con) -> bool:
        cols = [r[1] for r in con.execute('PRAGMA table_info(samples)').fetchall()]
        return 'country' in cols

    def __len__(self) -> int:
        return self._length

    def __iter__(self):
        con = duckdb.connect(self.db_path, read_only=True)

        ids = list(self._ids)

        # DataLoader(num_workers>0) 使用時、各ワーカーがこの __iter__ を独立に呼ぶため、
        # sharding しないと全ワーカーが同一データを重複して処理し実質 num_workers 倍の
        # 重複学習になってしまう（本体 db/dataset.py で修正済みと同じ落とし穴）。
        # shuffle より先に固定順序でシャーディングすることで、各ワーカーの担当分を合わせると
        # ちょうど全体を1回ずつ覆う（shuffle後にシャーディングすると各ワーカーが独立乱数で
        # 部分集合を選ぶことになり、重複・欠落が生じるため順序が重要）。
        worker = get_worker_info()
        if worker is not None:
            ids = ids[worker.id::worker.num_workers]

        if self.shuffle:
            random.shuffle(ids)

        for i in range(0, len(ids), self.chunk_size):
            chunk = ids[i:i + self.chunk_size]
            placeholders = ','.join(['?'] * len(chunk))
            # country カラムが存在する場合はJOINして取得
            has_country = self._has_country_col(con)
            country_sel = ', s.country' if has_country else ''
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

                # 長すぎる場合は末尾を切り捨て（EOS を必ず末尾に）
                if len(token_ids) > self.max_seq_len:
                    token_ids = token_ids[:self.max_seq_len - 1] + [self.tokenizer.eos_id]

                input_ids = token_ids[:-1]   # BOS ... last_mut
                target_ids = token_ids[1:]   # first_ctx ... EOS

                yield {
                    'input_ids': torch.tensor(input_ids, dtype=torch.long),
                    'target_ids': torch.tensor(target_ids, dtype=torch.long),
                    'length': len(input_ids),
                    # PETRA 準拠の representativeness 重み計算用（評価時のみ使用）
                    'country': country,
                    'collection_date': cdate,
                }

        con.close()

    @staticmethod
    def collate_fn(batch: list[dict]) -> tuple:
        max_len = max(item['length'] for item in batch)
        B = len(batch)

        input_ids = torch.zeros(B, max_len, dtype=torch.long)       # PAD=0
        target_ids = torch.full((B, max_len), -100, dtype=torch.long)  # ignore_index
        attention_mask = torch.zeros(B, max_len, dtype=torch.bool)

        for i, item in enumerate(batch):
            n = item['length']
            input_ids[i, :n] = item['input_ids']
            target_ids[i, :n] = item['target_ids']
            attention_mask[i, :n] = True

        countries = [item.get('country') for item in batch]
        dates = [item.get('collection_date') for item in batch]
        # raw_path が同一foldのtrain窓に既出か（PETRAのCLM設計では raw_path が一致すると
        # input/targetの対応も機械的に一致するため、リーク判定に使う）。
        # 未設定（通常学習時など）は常にFalse。
        is_leaked = [item.get('is_leaked', False) for item in batch]

        return input_ids, target_ids, attention_mask, countries, dates, is_leaked
