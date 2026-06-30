# --- db/dataset.py ---
# DBIterableDataset: DuckDBからバッチ単位でデータを読み込むDataset
# collate_fn内でSoft Target（確率分配ベクトル）を生成する機能を追加

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

        # サンプルIDリストを取得
        self.sample_ids = self._get_sample_ids()
        self._length = len(self.sample_ids)

    def _get_sample_ids(self):
        """条件に合うサンプルIDのリストを取得する。"""
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

        label_rows = con.execute(f"""
            SELECT sample_id, targets
            FROM labels
            WHERE sample_id IN ({placeholders})
        """, sample_ids).fetchall()

        con.close()

        # データを辞書に整理
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
        # aa_after は参照ゲノムのコドン計算が必要なため暫定 PAD(0)
        _use_sub_head = getattr(config, 'USE_SUBSTITUTION_HEAD', False)

        if _use_sub_head:
            import re as _re
            _MUT_NUC_RE = _re.compile(r'[A-Za-z](\d+)([A-Za-z])')

            def _parse_target_base_map(raw_path):
                """raw_path の最終セグメントから {nucpos: base_after_token} を返す。"""
                last_step = raw_path.split('>')[-1]
                pos_to_token = {}
                for mut in last_step.split(','):
                    m = _MUT_NUC_RE.search(mut.strip())
                    if m:
                        nucpos = int(m.group(1))
                        pos_to_token[nucpos] = config.BASE_VOCABS.get(
                            m.group(2), config.BASE_VOCABS.get('n', 0))
                return pos_to_token

        def _maybe_extend_labels(sample_id, base_labels):
            """USE_SUBSTITUTION_HEAD=True の場合に 5-tuple → 7-tuple へ拡張する。"""
            if not _use_sub_head or not base_labels:
                return base_labels
            raw = sample_dict.get(sample_id)
            if raw is None:
                return [(t[0], t[1], t[2], t[3], t[4], 0, 0) for t in base_labels]
            base_map = _parse_target_base_map(raw[1])  # raw[1] = raw_path
            # 各ターゲット変異を nucpos（t[1]）でマッチングして個別に base_after を付与
            return [(t[0], t[1], t[2], t[3], t[4],
                     base_map.get(t[1], 0),  # nucpos でマッチング
                     0)                       # aa_after: 暫定PAD
                    for t in base_labels]

        label_dict = {
            row[0]: _maybe_extend_labels(row[0], pickle.loads(row[1]))
            for row in label_rows
        }

        # 結果を構築
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

            if self.strain_to_strength is not None:
                strength_score = self.strain_to_strength.get(strain_name, 0.0)

            results.append((x, y_targets, path_length, raw_path, strain_name, strength_score, collection_date))

        return results

    def _process_sample(self, sample):
        """サンプルをモデル入力形式（パディング済みテンソル）に変換する。"""
        sequence, targets, original_len, raw_path, strain, strength_score, collection_date = sample

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

                # 数値特徴量の個別マスク
                # num構造: [freq(0), hydro(1), charge(2), size(3), blsm(4), pam250(5),
                #           host_distance_log_ratio_diff(6), host_distance_log_ratio_before(7),
                #           human_RSCU_diff(8), SCV2_RSCU_diff(9),
                #           optimal_to_optimal(10), non_optimal_to_optimal(11), optimal_to_non_optimal(12),
                #           is_transition(13), transition_human_RSCU_diff(14), transition_SCV2_RSCU_diff(15),
                #           CpG_diff(16), UpA_diff(17),
                #           human_RSCU_before(18), SCV2_RSCU_before(19),
                #           human_freq_before(20), human_freq_diff(21), SCV2_freq_before(22), SCV2_freq_diff(23),
                #           human_CAI_before(24), human_CAI_diff(25), SCV2_CAI_before(26), SCV2_CAI_diff(27),
                #           host_distance_RSCU_ratio_before(28), host_distance_RSCU_ratio_diff(29),
                #           cum_syn(30), cum_nonsyn(31)]
                num_mask_keys = [
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
                for i, key in enumerate(num_mask_keys):
                    if config.ABLATION_MASKS[key]:
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
            "original_len": original_len,
            "raw_path": raw_path,
            "strain": strain,
            "strength_score": strength_score,
            "collection_date": collection_date,
            "input_path_str": input_path_str,
            "target_path_str": target_path_str,
            "full_raw_path": raw_path
        }


def _build_soft_target(group_items, temperature=None):
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

    for route_targets in group_items:
        M = len(route_targets)  # このルートの共起変異数
        if M == 0:
            continue
        weight = 1.0 / (K * M)  # (1/K) × (1/M)
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

        if use_soft_target:
            # ── Soft Target モード ─────────────────────────────────────────
            # input_path_str をキーとしてバッチ内サンプルをグループ化し、
            # グループ単位で確率分配ベクトルを計算する。
            #
            # グループ化の結果:
            #   - 同一 input_path_str を持つ複数サンプル → 1つの Soft Target に統合
            #   - グループ内の位置がバッチ内で先頭のサンプルの x_cat/x_num/mask を代表として使用
            #
            # 出力は "統合後のバッチ" として返すため、バッチサイズが元より小さくなる場合がある。

            # input_path_str → バッチインデックス of グループマップを構築
            # input_path_str は list[str] or str の場合があるので tuple 化してハッシュ可能にする
            group_map = {}  # key: tuple(input_path_str) → list of batch indices
            for idx, item in enumerate(batch):
                key = tuple(item['input_path_str']) if isinstance(item['input_path_str'], list) \
                    else (item['input_path_str'],)
                if key not in group_map:
                    group_map[key] = []
                group_map[key].append(idx)

            # グループ順序を保持（最初に出現した順）
            group_keys = list(group_map.keys())

            # 統合後バッチの各要素を構築
            merged_x_cat = []
            merged_x_num = []
            merged_mask = []
            merged_soft_targets = []      # list of dict (タスク→FloatTensor)
            merged_raw_y = []             # 評価用: 代表サンプルの元ターゲット（全ルートをフラット化）
            merged_lens = []
            merged_strains = []
            merged_strength = []
            merged_input_strs = []
            merged_target_strs = []
            merged_full_paths = []
            merged_collection_dates = []

            for key in group_keys:
                indices = group_map[key]
                # 代表サンプル（先頭）の入力特徴量を使用
                rep_idx = indices[0]

                merged_x_cat.append(batch[rep_idx]['x_cat'])
                merged_x_num.append(batch[rep_idx]['x_num'])
                merged_mask.append(batch[rep_idx]['mask'])
                merged_lens.append(batch[rep_idx]['original_len'])
                merged_strains.append(batch[rep_idx]['strain'])
                # strength_score: 代表サンプル（先頭）の値を使用
                # 入力パスが同じでも strain が異なる場合、max で上方バイアスが生じるため
                merged_strength.append(batch[rep_idx]['strength_score'])
                merged_input_strs.append(batch[rep_idx]['input_path_str'])
                merged_target_strs.append(batch[rep_idx]['target_path_str'])
                merged_full_paths.append(batch[rep_idx]['full_raw_path'])
                merged_collection_dates.append(batch[rep_idx].get('collection_date', ''))

                # グループ内の全ルートのターゲットを収集し Soft Target を生成
                group_y_list = [batch[i]['y'] for i in indices]
                soft_target = _build_soft_target(group_y_list)
                merged_soft_targets.append(soft_target)

                # 評価用: グループ内の全ルートのターゲットをフラット化（重複除去なし）
                all_targets_flat = [t for y in group_y_list for t in y]
                merged_raw_y.append(all_targets_flat)

            stacked_x_cat = torch.stack(merged_x_cat)
            stacked_x_num = torch.stack(merged_x_num)
            stacked_mask = torch.stack(merged_mask)

            return (
                (stacked_x_cat, stacked_x_num, stacked_mask),
                merged_soft_targets,   # list of dict (Soft Target ベクトル)
                merged_lens,
                merged_strains,
                merged_strength,
                merged_input_strs,
                merged_target_strs,
                merged_full_paths,
                merged_raw_y,          # 評価用 Hard Target（フラット）
                merged_collection_dates,
            )

        else:
            # ── Hard Target モード（従来互換） ──────────────────────────────
            batch_y = [item['y'] for item in batch]
            return (
                (batch_x_cat, batch_x_num, batch_mask),
                batch_y,
                batch_lens,
                batch_strains,
                batch_strength_scores,
                batch_input_strs,
                batch_target_strs,
                batch_full_paths,
                batch_y,   # raw_y = y（Hard Target モードでは同一）
                batch_collection_dates,
            )

    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

