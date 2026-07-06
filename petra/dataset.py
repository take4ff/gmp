# petra/dataset.py — DuckDB からトークン系列をストリーム読み込み

import random
import torch
from torch.utils.data import IterableDataset
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
                 chunk_size: int = 2000, split_col: str = 'split_type'):
        self.db_path = db_path
        self.tokenizer = tokenizer
        self.split_type = split_type
        self.max_seq_len = max_seq_len
        self.shuffle = shuffle
        self.chunk_size = chunk_size
        self.split_col = split_col
        self._length = self._count()

    def _count(self) -> int:
        con = duckdb.connect(self.db_path, read_only=True)
        n = con.execute(
            f'SELECT COUNT(*) FROM samples WHERE {self.split_col} = ?',
            [self.split_type]
        ).fetchone()[0]
        con.close()
        return n

    @staticmethod
    def _has_country_col(con) -> bool:
        cols = [r[1] for r in con.execute('PRAGMA table_info(samples)').fetchall()]
        return 'country' in cols

    def __len__(self) -> int:
        return self._length

    def __iter__(self):
        con = duckdb.connect(self.db_path, read_only=True)

        ids = [r[0] for r in con.execute(
            f'SELECT sample_id FROM samples WHERE {self.split_col} = ?',
            [self.split_type]
        ).fetchall()]

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
                token_ids = self.tokenizer.encode(raw_path, cdate, clade, country)

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

        return input_ids, target_ids, attention_mask, countries, dates
