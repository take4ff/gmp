# --- db/dataset.py ---
# DBIterableDataset: DuckDBからバッチ単位でデータを読み込むDataset

import pickle
import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader

from .connection import connect_db
from .. import config


class DBIterableDataset(IterableDataset):
    """DuckDBからバッチ単位でデータを読み込むIterableDataset。

    全データをメモリに保持せず、chunk_size件ずつDBから読み込むことで
    メモリ効率的な学習を実現する。
    """

    def __init__(self, db_path, split_type=None, max_cooccurrence=None,
                 min_length=None, max_length=None, shuffle=False, chunk_size=1000):
        """
        Args:
            db_path: DuckDBファイルパス
            split_type: 0=train, 1=valid, 2=test
            max_cooccurrence: 最大共起数フィルタ
            min_length, max_length: パス長フィルタ
            shuffle: シャッフルするか
            chunk_size: 一度に読み込むサンプル数
        """
        self.db_path = db_path
        self.split_type = split_type
        self.max_cooccurrence = max_cooccurrence
        self.min_length = min_length
        self.max_length = max_length
        self.shuffle = shuffle
        self.chunk_size = chunk_size

        # サンプルIDリストを取得
        self.sample_ids = self._get_sample_ids()
        self._length = len(self.sample_ids)

    def _get_sample_ids(self):
        """条件に合うサンプルIDのリストを取得する。"""
        con = connect_db(self.db_path, read_only=True)

        query = "SELECT sample_id FROM samples WHERE 1=1"
        params = []

        if self.split_type is not None:
            query += " AND split_type = ?"
            params.append(self.split_type)

        if self.max_cooccurrence is not None:
            query += " AND max_cooccurrence <= ?"
            params.append(self.max_cooccurrence)

        if self.min_length is not None:
            query += " AND path_length >= ?"
            params.append(self.min_length)

        if self.max_length is not None:
            query += " AND path_length <= ?"
            params.append(self.max_length)

        # raw_path, strain_nameも取得してPython側でサンプリング・ユニーク化判定を行う
        query = query.replace("SELECT sample_id", "SELECT s.sample_id, s.raw_path, st.strain_name")
        query = query.replace("FROM samples", "FROM samples s JOIN strains st ON s.strain_id = st.strain_id")
        query = query.replace(" AND split_type", " AND s.split_type")
        query = query.replace(" AND max_cooccurrence", " AND s.max_cooccurrence")
        query = query.replace(" AND path_length", " AND s.path_length")
        
        query += " ORDER BY s.sample_id"

        result = con.execute(query, params).fetchall()

        # データ全体のサンプル数を取得 (現在適用されている全体フィルタと同条件だが、split_typeは問わない)
        query_all = "SELECT COUNT(*) FROM samples WHERE 1=1"
        params_all = []
        if self.max_cooccurrence is not None:
            query_all += " AND max_cooccurrence <= ?"
            params_all.append(self.max_cooccurrence)
        if self.min_length is not None:
            query_all += " AND path_length >= ?"
            params_all.append(self.min_length)
        if self.max_length is not None:
            query_all += " AND path_length <= ?"
            params_all.append(self.max_length)
        
        global_total_count = con.execute(query_all, params_all).fetchone()[0]
        con.close()

        total_count = len(result)
        split_name = {0: 'Train', 1: 'Valid', 2: 'Test'}.get(self.split_type, 'Unknown')
        print(f"[INFO] {split_name} data - Initial sample count: {total_count:,} (Global Total: {global_total_count:,})")

        if total_count == 0:
            return []

        import pandas as pd
        df = pd.DataFrame(result, columns=['sample_id', 'raw_path', 'strain'])

        if getattr(config, 'USE_UNIQUE_FILTER', False):
            df['path_tuple'] = df['raw_path'].apply(lambda x: tuple(x.split('>')))
            df = df.drop_duplicates(subset='path_tuple', keep='first')
            df = df.drop(columns=['path_tuple'])
            unique_count = len(df)
            print(f"[INFO] {split_name} data - After unique filter: {unique_count:,} (Removed {total_count - unique_count:,} duplicates)")

        # サンプリングの適用
        if getattr(config, 'MAX_STRAIN_NUM', None) is not None:
            strain_counts = df['strain'].value_counts()
            top_strains = strain_counts.nlargest(config.MAX_STRAIN_NUM).index.tolist()
            df = df[df['strain'].isin(top_strains)]
            print(f"[INFO] {split_name} data - After top {config.MAX_STRAIN_NUM} strain filter: {len(df):,}")

        sampling_mode = getattr(config, 'SAMPLING_MODE', 'proportional')
        
        if sampling_mode == 'proportional' and getattr(config, 'MAX_NUM', None) is not None:
            # 合計が MAX_NUM になるように、全体に占める現在のsplitの割合で target_num を割り当てる
            split_ratio = total_count / max(1, global_total_count)
            target_num = max(1, int(config.MAX_NUM * split_ratio))
            
            if len(df) > target_num:
                print(f"[INFO] {split_name} data - 比率サンプリング: {len(df):,}件から{target_num:,}件を抽出 (全体割当: {config.MAX_NUM:,} * {split_ratio*100:.1f}%)")
                strain_counts = df['strain'].value_counts()
                strain_samples = {}
                remaining = target_num
                sorted_strains = strain_counts.sort_values().index.tolist()
                
                for i, strain in enumerate(sorted_strains):
                    count = strain_counts[strain]
                    ratio = count / strain_counts[sorted_strains[i:]].sum()
                    samples_from_this = min(count, max(1, int(remaining * ratio)))
                    if i == len(sorted_strains) - 1:
                        samples_from_this = min(count, remaining)
                    strain_samples[strain] = samples_from_this
                    remaining -= samples_from_this
                
                sampled_dfs = []
                for strain, n_samples in strain_samples.items():
                    strain_df = df[df['strain'] == strain]
                    if len(strain_df) > n_samples:
                        sampled_dfs.append(strain_df.sample(n=n_samples, random_state=config.SEED))
                    else:
                        sampled_dfs.append(strain_df)
                df = pd.concat(sampled_dfs)
                print(f"[INFO] {split_name} data - 比率サンプリング完了: {len(df):,}件")

        elif sampling_mode == 'fixed_per_strain' and getattr(config, 'MAX_NUM_PER_STRAIN', None) is not None:
            max_num_per_strain = config.MAX_NUM_PER_STRAIN
            print(f"[INFO] {split_name} data - Filtering to max {max_num_per_strain} per strain...")
            sampled_dfs = []
            for _, group in df.groupby('strain'):
                if len(group) > max_num_per_strain:
                    sampled_dfs.append(group.sample(n=max_num_per_strain, random_state=config.SEED))
                else:
                    sampled_dfs.append(group)
            df = pd.concat(sampled_dfs)
            print(f"[INFO] {split_name} data - After filtering: {len(df):,}件")

        # ランダムソートによる順序シャッフル（後でDataLoader側でも必要に応じてやるが、行を保持しておく）
        df = df.sort_values('sample_id')

        return df['sample_id'].tolist()

    def __len__(self):
        return self._length

    def __iter__(self):
        import random

        sample_ids = self.sample_ids.copy()
        if self.shuffle:
            random.shuffle(sample_ids)

        # チャンク単位で処理
        for i in range(0, len(sample_ids), self.chunk_size):
            chunk_ids = sample_ids[i:i + self.chunk_size]
            samples = self._load_chunk(chunk_ids)

            for sample in samples:
                yield self._process_sample(sample)

    def _load_chunk(self, sample_ids):
        """指定されたサンプルIDのデータをDBからバッチ読み込みする（IN句使用で高速化）。"""
        if not sample_ids:
            return []

        con = connect_db(self.db_path, read_only=True)

        placeholders = ','.join(['?' for _ in sample_ids])

        sample_rows = con.execute(f"""
            SELECT s.sample_id, s.raw_path, s.path_length, s.strength_score, st.strain_name
            FROM samples s
            JOIN strains st ON s.strain_id = st.strain_id
            WHERE s.sample_id IN ({placeholders})
            ORDER BY s.sample_id
        """, sample_ids).fetchall()

        features_rows = con.execute(f"""
            SELECT sample_id, timestep, cat_features, num_features
            FROM features
            WHERE sample_id IN ({placeholders})
            ORDER BY sample_id, timestep
        """, sample_ids).fetchall()

        label_rows = con.execute(f"""
            SELECT sample_id, targets
            FROM labels
            WHERE sample_id IN ({placeholders})
        """, sample_ids).fetchall()

        con.close()

        # データを辞書に整理
        sample_dict = {row[0]: row for row in sample_rows}

        # 特徴量をsample_id別にグループ化
        features_dict = {}
        for sample_id, timestep, cat_blob, num_blob in features_rows:
            if sample_id not in features_dict:
                features_dict[sample_id] = []
            cat_feat = pickle.loads(cat_blob)
            num_feat = pickle.loads(num_blob)
            features_dict[sample_id].append((timestep, cat_feat, num_feat))

        # ラベルを辞書化
        label_dict = {row[0]: pickle.loads(row[1]) for row in label_rows}

        # 結果を構築
        results = []
        for sample_id in sample_ids:
            if sample_id not in sample_dict:
                continue

            _, raw_path, path_length, strength_score, strain_name = sample_dict[sample_id]

            x = []
            if sample_id in features_dict:
                for ts, cat_feat, num_feat in sorted(features_dict[sample_id], key=lambda x: x[0]):
                    x.append([(cat_feat, num_feat)])

            y_targets = label_dict.get(sample_id, [])

            results.append((x, y_targets, path_length, raw_path, strain_name, strength_score))

        return results

    def _process_sample(self, sample):
        """サンプルをモデル入力形式（パディング済みテンソル）に変換する。"""
        sequence, targets, original_len, raw_path, strain, strength_score = sample

        target_len = config.TARGET_LEN

        if len(raw_path) > target_len:
            input_path_str = raw_path[:-target_len]
            target_path_str = raw_path[-target_len:]
        else:
            input_path_str = []
            target_path_str = raw_path

        if len(input_path_str) > config.MAX_SEQ_LEN:
            start_index_str = len(input_path_str) - config.MAX_SEQ_LEN
            input_path_str = input_path_str[start_index_str:]

        if len(sequence) > config.TRAIN_MAX:
            start_index = len(sequence) - config.TRAIN_MAX
            sequence_to_pad = sequence[start_index:]
            active_seq_len = config.TRAIN_MAX
        else:
            sequence_to_pad = sequence
            active_seq_len = len(sequence)

        padded_cat = np.zeros(
            (config.TRAIN_MAX, config.MAX_CO_OCCURRENCE, config.NUM_FEATURE_STRING), dtype=np.int64
        )
        padded_num = np.zeros(
            (config.TRAIN_MAX, config.MAX_CO_OCCURRENCE, config.NUM_CHEM_FEATURES), dtype=np.float32
        )

        padding_mask = np.ones(config.TRAIN_MAX, dtype=bool)
        pad_len = config.TRAIN_MAX - active_seq_len
        padding_mask[pad_len:] = False

        for t, timestep_events in enumerate(sequence_to_pad):
            target_idx = pad_len + t
            for c, (cat_origin, num_origin) in enumerate(timestep_events):
                if c >= config.MAX_CO_OCCURRENCE:
                    break

                cat = list(cat_origin)
                num = list(num_origin)

                # 数値特徴量の個別マスク
                # num構造: [freq(0), hydro(1), charge(2), size(3), blsm(4), pam250(5)]
                mask_keys = ['FREQ', 'HYDRO', 'CHARGE', 'SIZE', 'BLSM', 'PAM250']
                for i, key in enumerate(mask_keys):
                    if config.ABLATION_MASKS[key]:
                        num[i] = 0.0

                padded_cat[target_idx, c] = cat
                padded_num[target_idx, c] = num

        return {
            "x_cat": torch.tensor(padded_cat, dtype=torch.long),
            "x_num": torch.tensor(padded_num, dtype=torch.float),
            "mask": torch.tensor(padding_mask, dtype=torch.bool),
            "y": targets,
            "original_len": original_len,
            "raw_path": raw_path,
            "strain": strain,
            "strength_score": strength_score,
            "input_path_str": input_path_str,
            "target_path_str": target_path_str,
            "full_raw_path": raw_path
        }


def create_db_dataloader(db_path, split_type, batch_size, shuffle=False,
                         max_cooccurrence=None, min_length=None, max_length=None,
                         chunk_size=1000):
    """DBからデータを読み込むDataLoaderを作成する。

    Args:
        db_path: DuckDBファイルパス
        split_type: 0=train, 1=valid, 2=test
        batch_size: バッチサイズ
        shuffle: シャッフルするか
        max_cooccurrence: 最大共起数フィルタ
        min_length, max_length: パス長フィルタ
        chunk_size: DBから一度に読み込むサンプル数

    Returns:
        DataLoader
    """
    dataset = DBIterableDataset(
        db_path=db_path,
        split_type=split_type,
        max_cooccurrence=max_cooccurrence,
        min_length=min_length,
        max_length=max_length,
        shuffle=shuffle,
        chunk_size=chunk_size
    )

    def collate_fn(batch):
        batch_x_cat = torch.stack([item['x_cat'] for item in batch])
        batch_x_num = torch.stack([item['x_num'] for item in batch])
        batch_mask = torch.stack([item['mask'] for item in batch])
        batch_y = [item['y'] for item in batch]
        batch_lens = [item['original_len'] for item in batch]
        batch_strains = [item['strain'] for item in batch]
        batch_strength_scores = [item['strength_score'] for item in batch]
        batch_input_strs = [item['input_path_str'] for item in batch]
        batch_target_strs = [item['target_path_str'] for item in batch]
        batch_full_paths = [item['full_raw_path'] for item in batch]

        return (batch_x_cat, batch_x_num, batch_mask), batch_y, batch_lens, batch_strains, batch_strength_scores, batch_input_strs, batch_target_strs, batch_full_paths

    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
