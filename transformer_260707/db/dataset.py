# --- db/dataset.py ---
# DBIterableDataset: DuckDBからバッチ単位でデータを読み込むDataset
# collate_fn内でSoft Target（確率分配ベクトル）を生成する機能を追加

import pickle
from typing import Any, List, NamedTuple
import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader

from .connection import connect_db
from .. import config


class DBBatch(NamedTuple):
    """collate_fn が返すバッチ。位置アンパック（従来コード）と名前アクセスの両対応。

    inputs = (x_cat, x_num, mask)。従来の
        `(x_cat, x_num, mask), y, lens, ... = batch`
    がそのまま動く（NamedTuple はフィールド順のタプル）。
    """
    inputs: Any            # (x_cat, x_num, mask)
    y: Any                 # Soft: list[dict] / Hard: list[list[tuple]]
    lens: List
    strains: List
    strength_scores: List
    input_strs: List
    target_strs: List
    full_paths: List
    raw_y: List            # 評価用 Hard Target（Soft 時は全ルートをフラット化）
    collection_dates: List


# --- 数値特徴量マスクのキー順（num[] のインデックス順に一致）---
# 旧実装は (サンプル×タイムステップ×共起変異) ごとにこの 32 要素リストを再構築していた。
# 静的なのでモジュール定数化してホットパスの割り当てを排除する。成長特徴の 4 キーは
# USE_LINEAGE_GROWTH_FEATURES=True のとき末尾（num[32..35]）に追加される。
_NUM_MASK_KEYS_BASE = [
    'FREQ', 'HYDRO', 'CHARGE', 'SIZE', 'BLSM', 'PAM250',
    'HOST_LOG_RATIO_DIFF', 'HOST_LOG_RATIO_BEFORE',
    'HUMAN_RSCU_DIFF', 'SCV2_RSCU_DIFF',
    'OPTIMAL_TO_OPTIMAL', 'NON_OPTIMAL_TO_OPTIMAL', 'OPTIMAL_TO_NON_OPTIMAL',
    'IS_TRANSITION', 'TRANSITION_HUMAN_RSCU_DIFF', 'TRANSITION_SCV2_RSCU_DIFF',
    'CPG_DIFF', 'UPA_DIFF',
    'HUMAN_RSCU_BEFORE', 'SCV2_RSCU_BEFORE',
    'HUMAN_FREQ_BEFORE', 'HUMAN_FREQ_DIFF', 'SCV2_FREQ_BEFORE', 'SCV2_FREQ_DIFF',
    'HUMAN_CAI_BEFORE', 'HUMAN_CAI_DIFF', 'SCV2_CAI_BEFORE', 'SCV2_CAI_DIFF',
    'HOST_RSCU_RATIO_BEFORE', 'HOST_RSCU_RATIO_DIFF',
    'CUM_SYN', 'CUM_NONSYN',
]
_NUM_MASK_KEYS_GROWTH = [
    'LINEAGE_LOG_COUNT_RECENT', 'LINEAGE_GROWTH_RATE',
    'LINEAGE_REL_GROWTH_ADV', 'LINEAGE_GROWTH_ACCEL',
]


# --- aa_after ラベル用ルックアップ表 ---
# codon_mutation4.csv は各塩基位置について置換後コドン（>A/>T/>G/>C 列）を持つ。
# これを DNA2Protein で AA に翻訳し AA_VOCABS の token へ変換した表を
# {(base_pos, new_base): aa_token} として1プロセスにつき1回だけ構築・キャッシュする。
_AA_AFTER_LUT = None


def _get_aa_after_lut():
    """(base_pos:int, new_base:'A'|'T'|'G'|'C') -> aa_after token(int) の辞書を返す（キャッシュ）。"""
    global _AA_AFTER_LUT
    if _AA_AFTER_LUT is not None:
        return _AA_AFTER_LUT
    import csv as _csv
    from .feature import DNA2Protein
    n_token = config.AA_VOCABS.get('n', 0)
    lut = {}
    try:
        with open(config.CODON_CSV, newline='', encoding='utf-8-sig') as f:
            for row in _csv.DictReader(f):
                try:
                    pos = int(row['base_pos'])
                except (KeyError, TypeError, ValueError):
                    continue
                for base in ('A', 'T', 'G', 'C'):
                    new_codon = (row.get('>' + base) or '').strip()
                    aa = DNA2Protein.get(new_codon.upper(), 'n') if new_codon and new_codon != 'none' else 'n'
                    lut[(pos, base)] = config.AA_VOCABS.get(aa, n_token)
    except FileNotFoundError:
        lut = {}
    _AA_AFTER_LUT = lut
    return lut


# --- point-in-time 変異頻度ルックアップ（config.USE_POINT_IN_TIME_FREQ=True のときのみ）---
# {(base_before:'A'|'T'|'G'|'C', position:int, base_after:'A'|'T'|'G'|'C') -> count} を
# config.TEMPORAL_SPLIT_DATE（cutoff日付）ごとにキャッシュする。
_REV_BASE_VOCABS = None
_POINT_IN_TIME_FREQ_CACHE = {}


def _get_rev_base_vocabs():
    global _REV_BASE_VOCABS
    if _REV_BASE_VOCABS is None:
        _REV_BASE_VOCABS = {v: k for k, v in config.BASE_VOCABS.items()}
    return _REV_BASE_VOCABS


def _get_point_in_time_freq_dict():
    """現在の config.TEMPORAL_SPLIT_DATE に対応する point-in-time 頻度表を返す（キャッシュ）。

    ファイルが見つからない場合は空dictを返し警告を1回だけ出す（呼び出し側は0.0にフォールバック）。
    """
    import os
    cutoff = getattr(config, 'TEMPORAL_SPLIT_DATE', None)
    if cutoff in _POINT_IN_TIME_FREQ_CACHE:
        return _POINT_IN_TIME_FREQ_CACHE[cutoff]

    freq_dir = getattr(config, 'POINT_IN_TIME_FREQ_DIR', 'reference/point_in_time_freq')
    path = os.path.join(freq_dir, f'{cutoff}.csv')
    d = {}
    if cutoff is not None and os.path.exists(path):
        import csv as _csv
        with open(path, newline='') as f:
            reader = _csv.DictReader(f)
            transitions = [c for c in reader.fieldnames if c != 'position']
            for row in reader:
                pos = int(row['position'])
                for t in transitions:
                    count = float(row[t])
                    if count > 0:
                        bef, aft = t.split('->')
                        d[(bef, pos, aft)] = count
        print(f"[INFO] Loaded point-in-time freq table: {path} ({len(d):,} 非ゼロ変異)")
    else:
        print(f"[WARNING] point-in-time freq table not found for cutoff={cutoff} "
              f"(expected {path})。FREQ_CSV由来の値のままフォールバックします。")
    _POINT_IN_TIME_FREQ_CACHE[cutoff] = d
    return d


class DBIterableDataset(IterableDataset):
    """DuckDBからバッチ単位でデータを読み込むIterableDataset。

    全データをメモリに保持せず、chunk_size件ずつDBから読み込むことで
    メモリ効率的な学習を実現する。
    """

    def __init__(self, db_path, split_type=None, max_cooccurrence=None,
                 min_length=None, max_length=None, shuffle=False, chunk_size=1000,
                 strain_to_strength=None, split_col_override=None):
        """
        Args:
            db_path: DuckDBファイルパス
            split_type: 0=train, 1=valid, 2=test
            max_cooccurrence: 最大共起数フィルタ
            min_length, max_length: パス長フィルタ
            shuffle: シャッフルするか
            chunk_size: 一度に読み込むサンプル数
            strain_to_strength: サンプリング後株出現数の動的流行度辞書 (NoneならDB値を使用)
            split_col_override: 参照するカラム名を強制指定（Noneなら get_split_col() に従う）
        """
        self.db_path = db_path
        self.split_type = split_type
        self.max_cooccurrence = max_cooccurrence
        self.min_length = min_length
        self.max_length = max_length
        self.shuffle = shuffle
        self.chunk_size = chunk_size
        self.strain_to_strength = strain_to_strength
        self.split_col_override = split_col_override

        # 数値特徴量マスクのキー順を一度だけ確定（ホットパスでの再構築を回避）
        self._num_mask_keys = list(_NUM_MASK_KEYS_BASE)
        if getattr(config, 'USE_LINEAGE_GROWTH_FEATURES', False):
            self._num_mask_keys = self._num_mask_keys + _NUM_MASK_KEYS_GROWTH

        # サンプルIDリストを取得し、同一 input_path_str（分岐親履歴）を持つサンプルを
        # split全体でグローバルにグルーピングする（バッチ局所グルーピングだと、大半の
        # 分岐兄弟サンプルがバッチ境界を跨いで分断され、any-of-set評価が事実上機能しない
        # ことが実測で判明したため。詳細は verify_batch_local_grouping.py 参照）。
        id_path_pairs = self._get_sample_ids()
        self.sample_ids, self.group_members = self._build_groups(id_path_pairs)
        self._length = len(self.sample_ids)

    def _build_groups(self, id_path_pairs):
        """(sample_id, raw_path) のリストから、同一input_path_str（raw_pathの最終ステップを
        除いた履歴）を持つサンプルをグループ化する。

        代表サンプルは各グループの min(sample_id) を決定的に選ぶ（shuffle順に依存しないため）。

        Returns:
            (representative_ids: List[int]（昇順）, group_members: Dict[int, List[int]])
        """
        groups = {}
        for sid, raw_path in id_path_pairs:
            parts = raw_path.split('>')
            input_path_str = '>'.join(parts[:-1]) if len(parts) > 1 else ''
            groups.setdefault(input_path_str, []).append(sid)

        group_members = {}
        for members in groups.values():
            repr_id = min(members)
            group_members[repr_id] = members

        representative_ids = sorted(group_members.keys())
        n_raw = len(id_path_pairs)
        n_groups = len(representative_ids)
        if n_raw > 0:
            print(f"[INFO] Grouping by input_path_str: {n_raw:,} raw samples -> "
                  f"{n_groups:,} groups ({100*n_groups/n_raw:.1f}%)")
        return representative_ids, group_members

    def _get_sample_ids(self):
        """条件に合う (sample_id, raw_path) のリストを取得する（sample_id昇順）。"""
        con = connect_db(self.db_path, read_only=True)

        query = "SELECT sample_id FROM samples WHERE 1=1"
        params = []

        from .queries import get_split_col
        split_col = self.split_col_override if self.split_col_override else get_split_col()

        if self.split_type is not None:
            query += f" AND {split_col} = ?"
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

        # raw_path, strain_name, strength_score も取得してPython側でサンプリング・ユニーク化判定を行う
        # 評価（Valid/Test）時には、多共起のラベルを持つサンプルを除外するために labels テーブルもジョインする
        eval_max_y_co = getattr(config, 'EVAL_MAX_Y_CO_OCCURRENCE', None)
        is_eval_split = (self.split_type in [1, 2])
        
        strength_col = 'strength_score_usher' if getattr(config, 'STRENGTH_SOURCE', 'ncbi') == 'usher' else 'strength_score_ncbi'
        strain_name_col = 'strain_name_usher' if getattr(config, 'STRENGTH_SOURCE', 'ncbi') == 'usher' else 'strain_name_ncbi'
        
        if is_eval_split and eval_max_y_co is not None:
            # labelsテーブルをジョインして targets (blob) を取得
            query = query.replace("SELECT sample_id", f"SELECT s.sample_id, s.raw_path, st.{strain_name_col} as strain_name, st.{strength_col} as strength_score, l.targets")
            query = query.replace("FROM samples", "FROM samples s JOIN strains st ON s.strain_id = st.strain_id JOIN labels l ON s.sample_id = l.sample_id")
        else:
            query = query.replace("SELECT sample_id", f"SELECT s.sample_id, s.raw_path, st.{strain_name_col} as strain_name, st.{strength_col} as strength_score")
            query = query.replace("FROM samples", "FROM samples s JOIN strains st ON s.strain_id = st.strain_id")
            
        query = query.replace(f" AND {split_col}", f" AND s.{split_col}")
        query = query.replace(" AND max_cooccurrence", " AND s.max_cooccurrence")
        query = query.replace(" AND path_length", " AND s.path_length")

        # 学習時 感染規模フィルタ: Train (split_type=0) のみ適用
        # valid/test は除外せず評価時にカテゴリ分析を行う
        use_train_sf = getattr(config, 'USE_TRAIN_STRENGTH_FILTER', False)
        train_sf_min = getattr(config, 'TRAIN_STRENGTH_MIN', 0.0)
        if use_train_sf and self.split_type == 0 and train_sf_min > 0:
            query += f" AND st.{strength_col} >= {train_sf_min}"
        query += " ORDER BY s.sample_id"

        raw_result = con.execute(query, params).fetchall()
        
        # Python側で結果を処理し、評価用多共起ラベルフィルタを適用
        result = []
        excluded_y_co_count = 0
        
        for row in raw_result:
            if is_eval_split and eval_max_y_co is not None:
                # row: (sample_id, raw_path, strain, strength_score, targets_blob)
                targets = pickle.loads(row[4])
                if len(targets) > eval_max_y_co:
                    excluded_y_co_count += 1
                    continue
                result.append(row[:4])
            else:
                result.append(row)
                
        if excluded_y_co_count > 0:
            split_name = {0: 'Train', 1: 'Valid', 2: 'Test'}.get(self.split_type, 'Unknown')
            print(f"[INFO] {split_name} data - Excluded {excluded_y_co_count:,} samples due to target Y co-occurrence > {eval_max_y_co}")

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
        df = pd.DataFrame(result, columns=['sample_id', 'raw_path', 'strain', 'strength_score'])

        # --- 提案2: ターゲットエントロピーによる学習サンプルフィルタリング（Train のみ）---
        # 系統別の正規化ターゲット位置エントロピーが TRAIN_ENTROPY_MAX を超える系統の
        # サンプルを除外する。valid/test には適用しない（層別評価のため）。
        if getattr(config, 'USE_TRAIN_ENTROPY_FILTER', False) and self.split_type == 0:
            from .queries import compute_lineage_entropy_map
            ent_max = getattr(config, 'TRAIN_ENTROPY_MAX', 0.7)
            entropy_map = compute_lineage_entropy_map(self.db_path, split_type=0)
            before_n = len(df)
            # マップに無い系統（想定外）は保守的に保持する（除外しない）
            keep_mask = df['strain'].map(lambda s: entropy_map.get(s, 0.0) <= ent_max)
            df = df[keep_mask]
            removed = before_n - len(df)
            print(f"[INFO] {split_name} data - Entropy filter (norm<= {ent_max}): "
                  f"kept {len(df):,} / removed {removed:,}")

        # 学習時 感染規模フィルタのログ出力
        if use_train_sf and self.split_type == 0 and train_sf_min > 0:
            print(f"[INFO] {split_name} data - Train strength filter applied: strength >= {train_sf_min} "
                  f"(log(1+n) scale, 小={getattr(config,'STRENGTH_CATEGORY_LOW_MAX',3.0)}, "
                  f"中上限={getattr(config,'STRENGTH_CATEGORY_MED_MAX',5.0)})")

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

        return list(zip(df['sample_id'].tolist(), df['raw_path'].tolist()))

    def __len__(self):
        return self._length

    def __iter__(self):
        import random
        from torch.utils.data import get_worker_info

        # num_workers>0 のとき、各ワーカーが disjoint な部分集合のみを処理する。
        # sharding しないと全ワーカーが同じ sample_ids を流し、データが num_workers 倍
        # 重複してしまう（学習が壊れる）。ストライド分割で重複なく全体を被覆する。
        worker = get_worker_info()
        if worker is not None and worker.num_workers > 1:
            sample_ids = self.sample_ids[worker.id::worker.num_workers]
        else:
            sample_ids = list(self.sample_ids)

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

        # ワーカー並列時、各接続が全コアのスレッドを張るとオーバーサブスクライブして
        # かえって遅くなるため、ワーカー内では DuckDB のスレッド数を抑える。
        from torch.utils.data import get_worker_info
        _db_threads = getattr(config, 'DATALOADER_DUCKDB_THREADS', None)
        if _db_threads is None and get_worker_info() is not None:
            _db_threads = 2
        if _db_threads is not None:
            try:
                con.execute(f"PRAGMA threads={int(_db_threads)}")
            except Exception:
                pass

        placeholders = ','.join(['?' for _ in sample_ids])

        strength_col = 'strength_score_usher' if getattr(config, 'STRENGTH_SOURCE', 'ncbi') == 'usher' else 'strength_score_ncbi'
        strain_name_col = 'strain_name_usher' if getattr(config, 'STRENGTH_SOURCE', 'ncbi') == 'usher' else 'strain_name_ncbi'
        sample_rows = con.execute(f"""
            SELECT s.sample_id, s.raw_path, s.path_length, st.{strength_col} as strength_score, st.{strain_name_col} as strain_name, s.collection_date
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

        # sample_ids は代表IDのリスト。any-of-set正解集合はグループ全メンバー
        # （分岐兄弟サンプル）のラベルを合算する必要があるため、代表だけでなく
        # 全メンバーのラベル（と USE_SUBSTITUTION_HEAD 用の raw_path）を取得する。
        # getattr: object.__new__(DBIterableDataset) 経由で __init__ を経ずに手動構築された
        # インスタンス（lineage_attention_compare.py 等）は group_members を持たないため、
        # 単独サンプル（グループ化なし）として安全にフォールバックする。
        group_members = getattr(self, 'group_members', {})
        all_member_ids = []
        for repr_id in sample_ids:
            all_member_ids.extend(group_members.get(repr_id, [repr_id]))
        member_placeholders = ','.join(['?' for _ in all_member_ids])

        member_raw_path_rows = con.execute(f"""
            SELECT sample_id, raw_path FROM samples WHERE sample_id IN ({member_placeholders})
        """, all_member_ids).fetchall()
        member_raw_path_dict = dict(member_raw_path_rows)

        label_rows = con.execute(f"""
            SELECT sample_id, targets
            FROM labels
            WHERE sample_id IN ({member_placeholders})
        """, all_member_ids).fetchall()

        con.close()

        # データを辞書に整理（代表サンプルの行。x_cat/x_numは代表のみで構築する）
        sample_dict = {row[0]: row for row in sample_rows}

        # 特徴量をsample_id → timestep別にグループ化（同一タイムステップの共起変異を束ねる）
        features_dict = {}
        for sample_id, timestep, cat_blob, num_blob in features_rows:
            if sample_id not in features_dict:
                features_dict[sample_id] = {}
            if timestep not in features_dict[sample_id]:
                features_dict[sample_id][timestep] = []
            cat_feat = pickle.loads(cat_blob)
            num_feat = pickle.loads(num_blob)
            features_dict[sample_id][timestep].append((cat_feat, num_feat))

        # ラベルを辞書化
        # USE_SUBSTITUTION_HEAD=True の場合、ラベルタプルを 5 要素から 7 要素に拡張する
        # 拡張分: (base_after_token, aa_after_token)
        # raw_path の最終セグメント（T+1 の各変異）を nucpos でマッチングして付与する
        # aa_after は codon_mutation4.csv の置換後コドン → DNA2Protein → AA_VOCABS で算出する
        _use_sub_head = getattr(config, 'USE_SUBSTITUTION_HEAD', False)

        if _use_sub_head:
            import re as _re
            _MUT_NUC_RE = _re.compile(r'[A-Za-z](\d+)([A-Za-z])')
            _aa_after_lut = _get_aa_after_lut()
            _aa_n_token = config.AA_VOCABS.get('n', 0)

            def _parse_target_base_map(raw_path):
                """raw_path の最終セグメントから {nucpos: (base_after_token, aa_after_token)} を返す。"""
                last_step = raw_path.split('>')[-1]
                pos_to_token = {}
                for mut in last_step.split(','):
                    m = _MUT_NUC_RE.search(mut.strip())
                    if m:
                        nucpos = int(m.group(1))
                        new_base = m.group(2)
                        base_token = config.BASE_VOCABS.get(new_base, config.BASE_VOCABS.get('n', 0))
                        aa_token = _aa_after_lut.get((nucpos, new_base.upper()), _aa_n_token)
                        pos_to_token[nucpos] = (base_token, aa_token)
                return pos_to_token

        def _maybe_extend_labels(sample_id, base_labels):
            """USE_SUBSTITUTION_HEAD=True の場合に 5-tuple → 7-tuple へ拡張する。

            グループ全メンバー（代表以外も含む）に対して呼ぶため、代表限定の sample_dict
            ではなく全メンバーの raw_path を持つ member_raw_path_dict を参照する。
            """
            if not _use_sub_head or not base_labels:
                return base_labels
            raw_path = member_raw_path_dict.get(sample_id)
            if raw_path is None:
                return [(t[0], t[1], t[2], t[3], t[4], 0, 0) for t in base_labels]
            base_map = _parse_target_base_map(raw_path)
            # 各ターゲット変異を nucpos（t[1]）でマッチングして base_after / aa_after を個別付与
            return [(t[0], t[1], t[2], t[3], t[4],
                     base_map.get(t[1], (0, 0))[0],   # base_after
                     base_map.get(t[1], (0, 0))[1])   # aa_after
                    for t in base_labels]

        label_dict = {
            row[0]: _maybe_extend_labels(row[0], pickle.loads(row[1]))
            for row in label_rows
        }

        # 結果を構築（sample_id は代表IDのリスト。x_cat/x_numは代表のみで構築し、
        # 正解集合はグループ全メンバーのラベルリスト group_targets_list として渡す）
        results = []
        for sample_id in sample_ids:
            if sample_id not in sample_dict:
                continue

            _, raw_path, path_length, strength_score, strain_name, collection_date = sample_dict[sample_id]

            x = []
            if sample_id in features_dict:
                for ts in sorted(features_dict[sample_id].keys()):
                    x.append(features_dict[sample_id][ts])

            y_targets = label_dict.get(sample_id, [])
            member_ids = group_members.get(sample_id, [sample_id])
            group_targets_list = [label_dict.get(mid, []) for mid in member_ids]

            if self.strain_to_strength is not None:
                strength_score = self.strain_to_strength.get(strain_name, 0.0)

            results.append((x, y_targets, path_length, raw_path, strain_name, strength_score,
                            collection_date, group_targets_list))

        return results

    def _process_sample(self, sample):
        """サンプルをモデル入力形式（パディング済みテンソル）に変換する。"""
        (sequence, targets, original_len, raw_path, strain, strength_score, collection_date,
         group_targets_list) = sample

        target_len = config.TARGET_LEN

        # raw_path は '>' 区切りの変異ステップ文字列（例: "A1G>C2T>G3A"）
        # TARGET_LEN ステップ分を末尾から Target として分離する
        path_steps = raw_path.split('>')
        if len(path_steps) > target_len:
            input_path_str = '>'.join(path_steps[:-target_len])
            target_path_str = '>'.join(path_steps[-target_len:])
        else:
            input_path_str = ''
            target_path_str = raw_path

        if len(input_path_str.split('>')) > config.MAX_SEQ_LEN:
            trimmed_steps = input_path_str.split('>')[-config.MAX_SEQ_LEN:]
            input_path_str = '>'.join(trimmed_steps)

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

                # point-in-time 変異頻度でcodon_freq(num[0])を上書き（時系列リーク修正）。
                # cat[0]=base_before token, cat[1]=position, cat[2]=base_after token
                # （db/feature.py Mutation_features_fast の cat_feat 構成に一致）。
                if getattr(config, 'USE_POINT_IN_TIME_FREQ', False) and len(num) > 0:
                    rev_base = _get_rev_base_vocabs()
                    bef = rev_base.get(cat[0])
                    aft = rev_base.get(cat[2])
                    if bef in ('A', 'T', 'G', 'C') and aft in ('A', 'T', 'G', 'C'):
                        freq_dict = _get_point_in_time_freq_dict()
                        num[0] = freq_dict.get((bef, cat[1], aft), 0.0)

                # 数値特徴量の個別マスク（キー順はモジュール定数 _NUM_MASK_KEYS_* を参照）。
                # num構造の並びは _NUM_MASK_KEYS_BASE のコメント/順序に一致する。
                num_mask_keys = self._num_mask_keys
                for i, key in enumerate(num_mask_keys):
                    if i < len(num) and config.ABLATION_MASKS.get(key, False):
                        num[i] = 0.0

                # カテゴリ特徴量: コンテキスト塩基の個別マスク (PAD index=0 に置換)
                # cat構造: [9..9+W-1]=left{W}..left1, [9+W..9+2W-1]=right1..right{W}  (W=CONTEXT_WINDOW)
                _ctx_w = getattr(config, 'CONTEXT_WINDOW', 3)
                if config.ABLATION_MASKS.get('MUTATION_CONTEXT', False):
                    for ci in range(9, 9 + 2 * _ctx_w):
                        cat[ci] = 0
                else:
                    for j in range(1, _ctx_w + 1):
                        if config.ABLATION_MASKS.get(f'MUTATION_CONTEXT_L{j}', False):
                            cat[9 + _ctx_w - j] = 0      # left_j: 遠い方が小インデックス
                        if config.ABLATION_MASKS.get(f'MUTATION_CONTEXT_R{j}', False):
                            cat[9 + _ctx_w + j - 1] = 0  # right_j: 近い方が小インデックス

                padded_cat[target_idx, c] = cat
                padded_num[target_idx, c] = num

        return {
            "x_cat": torch.tensor(padded_cat, dtype=torch.long),
            "x_num": torch.tensor(padded_num, dtype=torch.float),
            "mask": torch.tensor(padding_mask, dtype=torch.bool),
            "y": targets,
            "group_targets_list": group_targets_list,
            "original_len": original_len,
            "raw_path": raw_path,
            "strain": strain,
            "strength_score": strength_score,
            "collection_date": collection_date,
            "input_path_str": input_path_str,
            "target_path_str": target_path_str,
            "full_raw_path": raw_path
        }


def _build_soft_target(group_items, temperature=None, route_weights=None):
    """バッチ内で同一 input_path_str を持つサンプルグループから
    タスクごとの Soft Target 確率ベクトルを生成する。

    確率分配アルゴリズム:
        - グループ内のサンプル数 = K（分岐ルート数）
        - 各ルート k のターゲット変異数 = M_k（共起変異数）
        - 各変異への確率 = (1/K) × (1/M_k)
        - 同一変異IDが複数ルートに出現する場合は確率を加算

    Temperature Scaling (SOFT_TARGET_TEMPERATURE):
        生成した確率ベクトル p に対し p^(1/T) を計算して再正規化する。
        T < 1.0: よりシャープ（頻出変異への集中を強化）
        T > 1.0: より均一（ラベルスムージング的効果、正則化）
        T = 1.0: 変化なし（デフォルト）

    Args:
        group_items: list of list of tuples
            各要素は 1サンプルのターゲットタプルリスト
            [(region_id, pos_id, aa_pos_id, codon_pos_id, syn_id), ...]
        temperature: float または None。None の場合 config.SOFT_TARGET_TEMPERATURE を使用。

    Returns:
        dict: タスクごとの確率ベクトル (FloatTensor)
            {
                'region':    FloatTensor [NUM_REGIONS],
                'position':  FloatTensor [VOCAB_SIZE_POSITION],
                'aa_pos':    FloatTensor [VOCAB_SIZE_AA_POS],
                'codon_pos': FloatTensor [VOCAB_SIZE_CODON_POS],
                'synonymous':FloatTensor [VOCAB_SIZE_SYNONYMOUS],
            }
            ※ 各ベクトルの合計は 1.0（ターゲットが存在する場合）
    """
    if temperature is None:
        temperature = getattr(config, 'SOFT_TARGET_TEMPERATURE', 1.0)

    # ボキャブラリサイズ
    vocab_sizes = {
        'region':    config.NUM_REGIONS,
        'position':  config.VOCAB_SIZE_POSITION,
        'aa_pos':    config.VOCAB_SIZE_AA_POS,
        'codon_pos': config.VOCAB_SIZE_CODON_POS,
        'synonymous':config.VOCAB_SIZE_SYNONYMOUS,
    }
    task_indices = ['region', 'position', 'aa_pos', 'codon_pos', 'synonymous']

    # 各タスクのスコアベクトルを初期化
    scores = {task: torch.zeros(vs, dtype=torch.float) for task, vs in vocab_sizes.items()}

    K = len(group_items)  # 分岐ルート数
    if K == 0:
        return scores

    # --- 提案6: 子孫頻度で重み付けした確率的ターゲット ---
    # route_weights（各ルートの子孫リーフ数）が与えられ USE_FREQ_WEIGHTED_TARGETS=True の場合、
    # ルート寄与を一様 1/K ではなく頻度比 n_k^gamma / Σ n^gamma に置き換える。
    # 与えられない（＝preprocess 未対応の DB）場合は一様（現行と同一）にフォールバック。
    use_freq = (
        getattr(config, 'USE_FREQ_WEIGHTED_TARGETS', False)
        and route_weights is not None and len(route_weights) == K
    )
    if use_freq:
        gamma = getattr(config, 'FREQ_WEIGHT_GAMMA', 1.0)
        rw = [max(float(w), 0.0) ** gamma for w in route_weights]
        rw_sum = sum(rw)
        route_frac = [w / rw_sum for w in rw] if rw_sum > 0 else [1.0 / K] * K
    else:
        route_frac = [1.0 / K] * K

    for k_idx, route_targets in enumerate(group_items):
        M = len(route_targets)  # このルートの共起変異数
        if M == 0:
            continue
        weight = route_frac[k_idx] / M  # ルート寄与 × (1/M)
        for t in route_targets:
            # t: (region_id, pos_id, aa_pos_id, codon_pos_id, syn_id)
            for j, task in enumerate(task_indices):
                idx = t[j]
                vs = vocab_sizes[task]
                if 0 <= idx < vs:
                    scores[task][idx] += weight

    # --- Temperature Scaling ---
    # T != 1.0 の場合のみ適用（T=1.0 は数値的に同一なのでスキップ）
    if abs(temperature - 1.0) > 1e-6:
        for task in task_indices:
            s = scores[task]
            if s.sum() > 0:
                # scores^(1/T) を計算して再正規化
                # ゼロエントリは 0 のまま維持するため clamp は行わない
                # （0^(1/T) = 0 なので数学的に正しい）
                s = torch.pow(s.clamp(min=0.0), 1.0 / temperature)
                s_sum = s.sum()
                if s_sum > 0:
                    scores[task] = s / s_sum

    return scores


def create_db_dataloader(db_path, split_type, batch_size, shuffle=False,
                         max_cooccurrence=None, min_length=None, max_length=None,
                         chunk_size=1000, strain_to_strength=None, split_col_override=None):
    """DBからデータを読み込むDataLoaderを作成する。

    Args:
        db_path: DuckDBファイルパス
        split_type: 0=train, 1=valid, 2=test
        batch_size: バッチサイズ
        shuffle: シャッフルするか
        max_cooccurrence: 最大共起数フィルタ
        min_length, max_length: パス長フィルタ
        chunk_size: DBから一度に読み込むサンプル数
        strain_to_strength: サンプリング後株出現数の動的流行度辞書
        split_col_override: 参照カラムを強制指定（Noneなら get_split_col() に従う）

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
        chunk_size=chunk_size,
        strain_to_strength=strain_to_strength,
        split_col_override=split_col_override,
    )

    hybrid_alpha = getattr(config, 'HYBRID_ALPHA', 1.0)
    use_soft_target = hybrid_alpha > 0  # 0より大きければSoftラベルを生成（Soft/Hybrid共通）

    def collate_fn(batch):
        # データセット側（DBIterableDataset._build_groups）で既に同一input_path_str
        # （分岐親履歴）を持つサンプルがsplit全体でグローバルにグループ化され、各itemは
        # 1グループの代表サンプル + group_targets_list（グループ全メンバーのターゲット
        # タプルリストのリスト）を保持している。そのためバッチ内での再グルーピングは不要で、
        # batch の各要素をそのままスタック/整形するだけでよい。
        batch_x_cat = torch.stack([item['x_cat'] for item in batch])
        batch_x_num = torch.stack([item['x_num'] for item in batch])
        batch_mask = torch.stack([item['mask'] for item in batch])
        batch_lens = [item['original_len'] for item in batch]
        batch_strains = [item['strain'] for item in batch]
        batch_strength_scores = [item['strength_score'] for item in batch]
        batch_input_strs = [item['input_path_str'] for item in batch]
        batch_target_strs = [item['target_path_str'] for item in batch]
        batch_full_paths = [item['full_raw_path'] for item in batch]
        batch_collection_dates = [item.get('collection_date', '') for item in batch]

        # 評価用 Hard Target: グループ全メンバーのターゲットをフラット化（重複除去なし）。
        # Soft/Hard 両モードで共通（Hard モードも any-of-set 評価が正しく機能するように統一）。
        raw_y = [
            [t for member_targets in item['group_targets_list'] for t in member_targets]
            for item in batch
        ]

        if use_soft_target:
            y_out = [_build_soft_target(item['group_targets_list']) for item in batch]
        else:
            # ── Hard Target モード（従来互換）: 学習ターゲットは代表サンプル自身の値 ──
            y_out = [item['y'] for item in batch]

        return DBBatch(
            inputs=(batch_x_cat, batch_x_num, batch_mask),
            y=y_out,
            lens=batch_lens,
            strains=batch_strains,
            strength_scores=batch_strength_scores,
            input_strs=batch_input_strs,
            target_strs=batch_target_strs,
            full_paths=batch_full_paths,
            raw_y=raw_y,
            collection_dates=batch_collection_dates,
        )

    # DataLoader 並列化（データ律速の学習を GPU 計算とオーバーラップさせて時短する）
    loader_kwargs = dict(batch_size=batch_size, collate_fn=collate_fn)
    num_workers = int(getattr(config, 'NUM_DATALOADER_WORKERS', 0) or 0)
    if num_workers > 0:
        loader_kwargs['num_workers'] = num_workers
        # prefetch_factor / persistent_workers は num_workers>0 のときのみ指定可
        loader_kwargs['prefetch_factor'] = getattr(config, 'DATALOADER_PREFETCH_FACTOR', 2)
        loader_kwargs['persistent_workers'] = getattr(config, 'DATALOADER_PERSISTENT_WORKERS', True)
    if getattr(config, 'DATALOADER_PIN_MEMORY', False) and str(getattr(config, 'DEVICE', 'cpu')).startswith('cuda'):
        loader_kwargs['pin_memory'] = True

    return DataLoader(dataset, **loader_kwargs)

