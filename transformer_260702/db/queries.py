# --- db/queries.py ---
# SQL関連クエリ（サンプル取得・統計・分割割り当て）

import random
import pickle
from .connection import connect_db, get_db_path
from .. import config


def get_split_col():
    """現在設定されている SPLIT_MODE に応じて参照すべき DB のカラム名を返す。"""
    mode = getattr(config, 'SPLIT_MODE', 'timestep').lower()
    if mode == 'walk_forward':
        return 'split_type_wf'
    if mode == 'date':
        return 'split_type_date'
    return 'split_type'


def get_db_data_info(db_path=None, split_type=None, max_cooccurrence=None):
    """DBからデータ情報を取得（data_info相当）。

    Returns:
        dict: train/val/test件数・長さ範囲・流行度閾値を含む辞書
    """
    if db_path is None:
        db_path = get_db_path()

    con = connect_db(db_path, read_only=True)

    split_col = get_split_col()
    query_base = f"SELECT MIN(path_length), MAX(path_length), COUNT(*) FROM samples WHERE 1=1"

    def get_stats(st):
        q = query_base + f" AND {split_col} = {st}"
        if max_cooccurrence:
            q += f" AND max_cooccurrence <= {max_cooccurrence}"
        return con.execute(q).fetchone()

    train_stats = get_stats(0)
    valid_stats = get_stats(1)
    test_stats = get_stats(2)

    # 流行度閾値
    strength_col = 'strength_score_usher' if getattr(config, 'STRENGTH_SOURCE', 'ncbi') == 'usher' else 'strength_score_ncbi'
    strength_stats = con.execute(f"SELECT MIN({strength_col}), MAX({strength_col}) FROM strains").fetchone()
    min_s, max_s = strength_stats
    score_range = (max_s - min_s) if (max_s is not None and min_s is not None) else 10
    low_max = int(min_s + score_range / 3) + 1 if min_s is not None else 6
    med_max = int(min_s + 2 * score_range / 3) + 1 if min_s is not None else 10

    con.close()

    return {
        'train_min_len': train_stats[0] or 0,
        'train_max_len': train_stats[1] or 0,
        'train_count': train_stats[2] or 0,
        'val_min_len': valid_stats[0] or 0,
        'val_max_len': valid_stats[1] or 0,
        'val_count': valid_stats[2] or 0,
        'test_min_len': test_stats[0] or 0,
        'test_max_len': test_stats[1] or 0,
        'test_count': test_stats[2] or 0,
        'strength_low_max': low_max,
        'strength_med_max': med_max,
    }


def load_samples_from_db(db_path=None, split_type=None, max_cooccurrence=None,
                         min_length=None, max_length=None, limit=None):
    """DBからサンプルを読み込み、学習用データ形式に変換する。

    Args:
        db_path: DuckDBファイルパス
        split_type: 0=train, 1=valid, 2=test
        max_cooccurrence: 最大共起数フィルタ
        min_length, max_length: パス長フィルタ
        limit: 取得件数制限

    Returns:
        list of (x, y_targets, path_length, raw_path, strain_name, strength_score)
    """
    if db_path is None:
        db_path = get_db_path()

    con = connect_db(db_path, read_only=True)

    strength_col = 'strength_score_usher' if getattr(config, 'STRENGTH_SOURCE', 'ncbi') == 'usher' else 'strength_score_ncbi'
    strain_name_col = 'strain_name_usher' if getattr(config, 'STRENGTH_SOURCE', 'ncbi') == 'usher' else 'strain_name_ncbi'
    query = f"""
        SELECT s.sample_id, s.raw_path, s.path_length, st.{strength_col} as strength_score,
               st.{strain_name_col} as strain_name
        FROM samples s
        JOIN strains st ON s.strain_id = st.strain_id
        WHERE 1=1
    """
    params = []

    if split_type is not None:
        split_col = get_split_col()
        query += f" AND s.{split_col} = ?"
        params.append(split_type)

    if max_cooccurrence is not None:
        query += " AND s.max_cooccurrence <= ?"
        params.append(max_cooccurrence)

    if min_length is not None:
        query += " AND s.path_length >= ?"
        params.append(min_length)

    if max_length is not None:
        query += " AND s.path_length <= ?"
        params.append(max_length)

    query += " ORDER BY s.sample_id"

    if limit is not None:
        query += f" LIMIT {limit}"

    samples = con.execute(query, params).fetchall()

    result = []
    for sample_id, raw_path, path_length, strength_score, strain_name in samples:
        features = con.execute("""
            SELECT timestep, cat_features, num_features
            FROM features
            WHERE sample_id = ?
            ORDER BY timestep
        """, [sample_id]).fetchall()

        x = []
        for ts, cat_blob, num_blob in features:
            cat_feat = pickle.loads(cat_blob)
            num_feat = pickle.loads(num_blob)
            x.append([(cat_feat, num_feat)])

        label_row = con.execute("""
            SELECT targets FROM labels WHERE sample_id = ?
        """, [sample_id]).fetchone()

        y_targets = pickle.loads(label_row[0]) if label_row else []
        result.append((x, y_targets, path_length, raw_path, strain_name, strength_score))

    con.close()
    return result


def assign_splits(con, train_ratio=0.8, valid_ratio=None):
    """パス長に基づいてtrain/valid/testに分割し、DBを更新する。"""
    if valid_ratio is None:
        valid_ratio = getattr(config, 'VALID_RATIO', 0.2)
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [INFO] Assigning train/valid/test splits based on path length...")

    train_max = config.TRAIN_MAX
    valid_num = config.VALID_NUM
    overlap_start = train_max - valid_num + 1

    # Train (オーバーラップ外)
    con.execute(f"""
        UPDATE samples SET split_type = 0
        WHERE path_length < {overlap_start}
    """)

    # Test
    con.execute(f"""
        UPDATE samples SET split_type = 2
        WHERE path_length > {train_max}
    """)

    # オーバーラップ領域 (valid_ratio で分割)
    overlap_samples = con.execute(f"""
        SELECT sample_id FROM samples
        WHERE path_length >= {overlap_start} AND path_length <= {train_max}
        ORDER BY strain_id, raw_path
    """).fetchall()

    if overlap_samples:
        sample_ids = [s[0] for s in overlap_samples]
        random.seed(config.SEED)
        random.shuffle(sample_ids)

        n_valid = int(len(sample_ids) * valid_ratio)
        valid_ids = sample_ids[:n_valid]
        train_ids = sample_ids[n_valid:]

        if valid_ids:
            con.execute(f"UPDATE samples SET split_type = 1 WHERE sample_id IN ({','.join(map(str, valid_ids))})")
        if train_ids:
            con.execute(f"UPDATE samples SET split_type = 0 WHERE sample_id IN ({','.join(map(str, train_ids))})")

    # 統計を表示
    stats = con.execute("SELECT * FROM split_stats").fetchall()
    for split_type, split_name, count in stats:
        print(f"[{timestamp}]   {split_name}: {count:,} samples")


def assign_date_splits(con):
    """collection_date の基準日を用いて train/valid/test を分割し、DB を更新する。

    - collection_date < TEMPORAL_SPLIT_DATE                                       → train / valid
    - TEMPORAL_SPLIT_DATE <= collection_date < TEMPORAL_SPLIT_TEST_END (or None) → test (2)
    - collection_date >= TEMPORAL_SPLIT_TEST_END (TEMPORAL_SPLIT_TEST_END != None) → 除外 (-1)
    - collection_date が NULL または空文字                                         → train に割り当て

    config キー:
        TEMPORAL_SPLIT_DATE     : str        基準日（ISO 形式: YYYY-MM-DD）
        TEMPORAL_SPLIT_TEST_END : str | None テスト期間終端日（None = 上限なし）
        DATE_VALID_RATIO        : float      基準日前データのうち Valid に充てる比率
        SEED                    : int        再現性のためのランダムシード
    """
    from datetime import datetime

    split_date = getattr(config, 'TEMPORAL_SPLIT_DATE', '2023-12-31')
    split_end  = getattr(config, 'TEMPORAL_SPLIT_TEST_END', None)
    valid_ratio = getattr(config, 'DATE_VALID_RATIO', 0.1)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"[{timestamp}] [INFO] Assigning splits by date (split_date={split_date}, "
          f"split_end={split_end}, valid_ratio={valid_ratio})...")

    # まず全サンプルを train (0) にリセット
    con.execute("UPDATE samples SET split_type_date = 0")

    # テスト窓を設定。RPAD で YYYY や YYYY-MM 形式を補完して文字列比較
    if split_end is None:
        # 従来動作: 基準日以降がすべて test
        con.execute(f"""
            UPDATE samples
            SET split_type_date = 2
            WHERE RPAD(collection_date, 10, '-01-01') >= '{split_date}'
              AND collection_date IS NOT NULL
              AND collection_date != ''
        """)
    else:
        # Walk-forward: [split_date, split_end) → test、split_end 以降 → 除外
        con.execute(f"""
            UPDATE samples
            SET split_type_date = 2
            WHERE RPAD(collection_date, 10, '-01-01') >= '{split_date}'
              AND RPAD(collection_date, 10, '-01-01') <  '{split_end}'
              AND collection_date IS NOT NULL
              AND collection_date != ''
        """)
        con.execute(f"""
            UPDATE samples
            SET split_type_date = -1
            WHERE RPAD(collection_date, 10, '-01-01') >= '{split_end}'
              AND collection_date IS NOT NULL
              AND collection_date != ''
        """)

    # 基準日前 (NULL / 空文字も含む、split_type_date = 0) の sample_id を取得して valid を確率的に割り当て
    before_rows = con.execute("""
        SELECT sample_id FROM samples
        WHERE split_type_date = 0
        ORDER BY sample_id
    """).fetchall()

    if before_rows:
        before_ids = [r[0] for r in before_rows]
        random.seed(config.SEED)
        random.shuffle(before_ids)
        n_valid = int(len(before_ids) * valid_ratio)
        valid_ids = before_ids[:n_valid]

        if valid_ids:
            con.execute(
                f"UPDATE samples SET split_type_date = 1 "
                f"WHERE sample_id IN ({','.join(map(str, valid_ids))})"
            )

    # 統計を表示（-1 = 除外 も含む）
    stats = con.execute("""
        SELECT split_type_date,
               CASE split_type_date
                   WHEN -1 THEN 'excluded'
                   WHEN  0 THEN 'train'
                   WHEN  1 THEN 'valid'
                   WHEN  2 THEN 'test'
                   ELSE 'unknown'
               END as split_name,
               COUNT(*) as sample_count
        FROM samples
        GROUP BY split_type_date
        ORDER BY split_type_date
    """).fetchall()
    for split_type, split_name, count in stats:
        print(f"[{timestamp}]   {split_name}: {count:,} samples")


def assign_wf_splits(con):
    """walk_forward モード専用の split_type_wf カラムを更新する。

    split_type_date を汚染しないよう専用カラム split_type_wf に書き込む。
    フォールドごとに呼ばれ上書きされる。

    訓練ウィンドウ:
      Fold 1 (WALK_FORWARD_TRAIN_START=None): collection_date < split_date OR NULL/空
      Fold N (WALK_FORWARD_TRAIN_START 設定): [train_start, split_date) のみ
    テストウィンドウ: [split_date, split_end)
    それ以外: -1 (excluded)
    """
    from datetime import datetime

    train_start = getattr(config, 'WALK_FORWARD_TRAIN_START', None)
    split_date  = getattr(config, 'TEMPORAL_SPLIT_DATE', '2023-12-31')
    split_end   = getattr(config, 'TEMPORAL_SPLIT_TEST_END', None)
    valid_ratio = getattr(config, 'DATE_VALID_RATIO', 0.1)
    timestamp   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"[{timestamp}] [INFO] Assigning walk_forward splits to split_type_wf "
          f"(train_start={train_start}, split_date={split_date}, "
          f"split_end={split_end}, valid_ratio={valid_ratio})...")

    # まず全件を除外扱いにリセット
    con.execute("UPDATE samples SET split_type_wf = -1")

    # 訓練ウィンドウを 0 (train) に設定
    if train_start is None:
        # Fold 1: 上限なし（全歴史データ）。NULL/空文字も train に含める
        con.execute(f"""
            UPDATE samples SET split_type_wf = 0
            WHERE collection_date IS NULL OR collection_date = ''
               OR RPAD(collection_date, 10, '-01-01') < '{split_date}'
        """)
    else:
        # Fold N: [train_start, split_date) のみ
        con.execute(f"""
            UPDATE samples SET split_type_wf = 0
            WHERE collection_date IS NOT NULL AND collection_date != ''
              AND RPAD(collection_date, 10, '-01-01') >= '{train_start}'
              AND RPAD(collection_date, 10, '-01-01') <  '{split_date}'
        """)

    # テストウィンドウを 2 (test) に設定（訓練と重複しない前提）
    if split_end is None:
        con.execute(f"""
            UPDATE samples SET split_type_wf = 2
            WHERE collection_date IS NOT NULL AND collection_date != ''
              AND RPAD(collection_date, 10, '-01-01') >= '{split_date}'
        """)
    else:
        con.execute(f"""
            UPDATE samples SET split_type_wf = 2
            WHERE collection_date IS NOT NULL AND collection_date != ''
              AND RPAD(collection_date, 10, '-01-01') >= '{split_date}'
              AND RPAD(collection_date, 10, '-01-01') <  '{split_end}'
        """)

    # 訓練データを train/valid に分割（DATE_VALID_RATIO）
    before_rows = con.execute(
        "SELECT sample_id FROM samples WHERE split_type_wf = 0 ORDER BY sample_id"
    ).fetchall()

    if before_rows:
        before_ids = [r[0] for r in before_rows]
        random.seed(config.SEED)
        random.shuffle(before_ids)
        n_valid = int(len(before_ids) * valid_ratio)
        valid_ids = before_ids[:n_valid]
        if valid_ids:
            con.execute(
                f"UPDATE samples SET split_type_wf = 1 "
                f"WHERE sample_id IN ({','.join(map(str, valid_ids))})"
            )

    stats = con.execute("""
        SELECT split_type_wf,
               CASE split_type_wf
                   WHEN -1 THEN 'excluded'
                   WHEN  0 THEN 'train'
                   WHEN  1 THEN 'valid'
                   WHEN  2 THEN 'test'
                   ELSE 'unknown'
               END, COUNT(*)
        FROM samples
        GROUP BY split_type_wf ORDER BY split_type_wf
    """).fetchall()
    for val, label, count in stats:
        print(f"[{timestamp}]   {label}: {count:,} samples")


def assign_splits_auto(con):
    """全3カラム（timestep / date / walk_forward）を初期化する。"""
    assign_splits(con)
    assign_date_splits(con)
    assign_wf_splits(con)


def get_processed_strains(con):
    """DBから処理済みの株一覧を取得する（再開機能用）。"""
    try:
        result = con.execute("""
            SELECT DISTINCT st.strain_name
            FROM samples s
            JOIN strains st ON s.strain_id = st.strain_id
        """).fetchall()
        return set(r[0] for r in result)
    except Exception:
        return set()


def get_next_ids(con):
    """DBから次のID値を取得する（再開機能用）。"""
    try:
        sample_id = con.execute("SELECT COALESCE(MAX(sample_id), -1) + 1 FROM samples").fetchone()[0]
        feature_id = con.execute("SELECT COALESCE(MAX(feature_id), -1) + 1 FROM features").fetchone()[0]
        label_id = con.execute("SELECT COALESCE(MAX(label_id), -1) + 1 FROM labels").fetchone()[0]
        return sample_id, feature_id, label_id
    except Exception:
        return 0, 0, 0


def get_train_class_counts(db_path=None):
    """学習データの各タスクごとのクラス頻度(n_t)を計算・キャッシュする。

    Returns:
        dict: {'region': Tensor, 'position': Tensor, 'aa_pos': Tensor, 'codon_pos': Tensor, 'synonymous': Tensor}
    """
    import os
    import torch
    import pickle
    from .. import config

    if db_path is None:
        db_path = get_db_path()

    os.makedirs(config.CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(config.CACHE_DIR, 'class_counts.pkl')

    if getattr(config, 'FORCE_REPROCESS', False) == False and os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                counts = pickle.load(f)
            print(f"[INFO] Loaded class counts from {cache_file}")
            return counts
        except Exception as e:
            print(f"[WARNING] Failed to load class counts cache: {e}. Recomputing...")

    print("[INFO] Computing training class counts (n_t)... This may take a moment.")
    con = connect_db(db_path, read_only=True)

    # 必要なすべてのサンプルのラベルを取得 (Trainのみ)
    split_col = get_split_col()
    query = f"""
        SELECT l.targets 
        FROM labels l
        JOIN samples s ON l.sample_id = s.sample_id
        WHERE s.{split_col} = 0
    """
    rows = con.execute(query).fetchall()
    con.close()

    counts = {
        'region': torch.zeros(config.NUM_REGIONS, dtype=torch.long),
        'position': torch.zeros(config.VOCAB_SIZE_POSITION, dtype=torch.long),
        'aa_pos': torch.zeros(config.VOCAB_SIZE_AA_POS, dtype=torch.long),
        'codon_pos': torch.zeros(config.VOCAB_SIZE_CODON_POS, dtype=torch.long),
        'synonymous': torch.zeros(config.VOCAB_SIZE_SYNONYMOUS, dtype=torch.long),
    }

    from tqdm import tqdm
    for row in tqdm(rows, desc="Accumulating class counts"):
        targets = pickle.loads(row[0])
        for t in targets:
            # t: (region, position, aa_pos, codon_pos, synonymous)
            if t[0] < config.NUM_REGIONS: counts['region'][t[0]] += 1
            if t[1] < config.VOCAB_SIZE_POSITION: counts['position'][t[1]] += 1
            if t[2] < config.VOCAB_SIZE_AA_POS: counts['aa_pos'][t[2]] += 1
            if t[3] < config.VOCAB_SIZE_CODON_POS: counts['codon_pos'][t[3]] += 1
            if t[4] < config.VOCAB_SIZE_SYNONYMOUS: counts['synonymous'][t[4]] += 1

    with open(cache_file, 'wb') as f:
        pickle.dump(counts, f)
    
    print(f"[INFO] Saved class counts to {cache_file}")
    return counts


def get_combined_sampled_strength(db_path=None):
    """サンプリング適用後の全スプリットの合計株出現数から流行度辞書を一括で取得する (v260224 互換)。"""
    import math
    from collections import Counter
    import pandas as pd
    
    if db_path is None:
        db_path = get_db_path()
        
    # すべてのサンプルをDBから取得
    con = connect_db(db_path, read_only=True)
    split_col = get_split_col()
    strength_col = 'strength_score_usher' if getattr(config, 'STRENGTH_SOURCE', 'ncbi') == 'usher' else 'strength_score_ncbi'
    strain_name_col = 'strain_name_usher' if getattr(config, 'STRENGTH_SOURCE', 'ncbi') == 'usher' else 'strain_name_ncbi'
    query = f"""
        SELECT s.sample_id, s.raw_path, st.{strain_name_col} as strain_name, s.{split_col} as split_type, st.{strength_col} as strength_score
        FROM samples s
        JOIN strains st ON s.strain_id = st.strain_id
    """
    result = con.execute(query).fetchall()
    
    # 統計用の全体件数
    # dataset.py と同様に、各スプリットの比率計算に使う global_total_count を取得するが、
    # dataset.py の global_total_count は「現在適用されている全体フィルタと同条件だが、split_typeは問わない」件数。
    # ここでは、データ全体に対して特別な長さ制限などが適用されているかを考慮して、
    # サンプリングで抽出されたものの割合計算に合わせる。
    # 通常 samples に入っている全件数で問題ない。
    query_all = "SELECT COUNT(*) FROM samples"
    global_total_count = con.execute(query_all).fetchone()[0]
    con.close()
    
    if not result:
        return {}
        
    df = pd.DataFrame(result, columns=['sample_id', 'raw_path', 'strain', 'split_type', 'strength_score'])
    
    # 重複パスの除外 (Unique Filter)
    if getattr(config, 'USE_UNIQUE_FILTER', False):
        df['path_tuple'] = df['raw_path'].apply(lambda x: tuple(x.split('>')))
        df = df.drop_duplicates(subset='path_tuple', keep='first')
        df = df.drop(columns=['path_tuple'])
        
    # 上位株フィルタ
    if getattr(config, 'MAX_STRAIN_NUM', None) is not None:
        strain_counts = df['strain'].value_counts()
        top_strains = strain_counts.nlargest(config.MAX_STRAIN_NUM).index.tolist()
        df = df[df['strain'].isin(top_strains)]
        
    # スプリットごとにサンプリングを適用して最終的な株リストを取得
    sampling_mode = getattr(config, 'SAMPLING_MODE', 'proportional')
    sampled_dfs = []
    
    # split_typeごとに処理する。dataset.py の DBIterableDataset._get_sample_ids と同じロジック
    for split_type in [0, 1, 2]:
        split_df = df[df['split_type'] == split_type].copy()
        if split_df.empty:
            continue
            
        # [260417] 学習時 感染規模フィルタ (Train のみ)
        use_train_sf = getattr(config, 'USE_TRAIN_STRENGTH_FILTER', False)
        train_sf_min = getattr(config, 'TRAIN_STRENGTH_MIN', 0.0)
        if use_train_sf and split_type == 0 and train_sf_min > 0:
            split_df = split_df[split_df['strength_score'] >= train_sf_min]
            
        if sampling_mode == 'proportional' and getattr(config, 'MAX_NUM', None) is not None:
            # dataset.py と同様に split の比率で target_num を計算
            # ここでの比率計算における分母は、DBIterableDataset 内と同様に、フィルタ前の該当 split 初期件数の合計（= global_total_count）とする
            # （dataset.py: split_ratio = total_count / max(1, global_total_count)）
            # ここで、total_count は unique フィルタ適用前の DB から引いた件数。
            # したがって、一括シミュレーションでも、unique フィルタ適用前の該当 split の件数を total_count_pre_unique とする。
            total_count_split = sum(1 for r in result if r[3] == split_type)
            split_ratio = total_count_split / max(1, global_total_count)
            target_num = max(1, int(config.MAX_NUM * split_ratio))
            
            if len(split_df) > target_num:
                strain_counts = split_df['strain'].value_counts()
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
                
                split_sampled_dfs = []
                for strain, n_samples in strain_samples.items():
                    strain_df = split_df[split_df['strain'] == strain]
                    if len(strain_df) > n_samples:
                        split_sampled_dfs.append(strain_df.sample(n=n_samples, random_state=config.SEED))
                    else:
                        split_sampled_dfs.append(strain_df)
                split_df = pd.concat(split_sampled_dfs)
                
        elif sampling_mode == 'fixed_per_strain' and getattr(config, 'MAX_NUM_PER_STRAIN', None) is not None:
            max_num_per_strain = config.MAX_NUM_PER_STRAIN
            split_sampled_dfs = []
            for _, group in split_df.groupby('strain'):
                if len(group) > max_num_per_strain:
                    split_sampled_dfs.append(group.sample(n=max_num_per_strain, random_state=config.SEED))
                else:
                    split_sampled_dfs.append(group)
            split_df = pd.concat(split_sampled_dfs)
            
        sampled_dfs.append(split_df)
        
    if not sampled_dfs:
        return {}
        
    final_df = pd.concat(sampled_dfs)
    
    # サンプリング後の全スプリット合計の株出現数をカウント
    strain_counts = Counter(final_df['strain'].tolist())
    strain_to_strength = {
        strain: math.log(1 + count) for strain, count in strain_counts.items()
    }
    
    print(f"[INFO] Computed combined sample-based strength for {len(strain_to_strength)} strains")
    return strain_to_strength


def get_position_region_map(db_path=None):
    """提案3（階層的予測）用: position_id → region_id の写像を返す。

    各ゲノム位置は一意の遺伝子領域に属するため、labels テーブルの
    ターゲットタプル (region, position, ...) から position→region を集計する。
    未出現の position は 0（PAD 領域）とする。多数決で最頻 region を採用する
    （ノイズや領域境界の曖昧さに対して頑健にするため）。

    Returns:
        LongTensor [VOCAB_SIZE_POSITION]  各 position の region_id
    """
    import os
    import torch
    from collections import defaultdict, Counter

    if db_path is None:
        db_path = get_db_path()

    os.makedirs(config.CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(config.CACHE_DIR, 'position_region_map.pkl')

    if getattr(config, 'FORCE_REPROCESS', False) is False and os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                arr = pickle.load(f)
            print(f"[INFO] Loaded position→region map from {cache_file}")
            return torch.as_tensor(arr, dtype=torch.long)
        except Exception as e:
            print(f"[WARNING] Failed to load position→region map cache: {e}. Recomputing...")

    print("[INFO] Building position→region map (提案3)...")
    con = connect_db(db_path, read_only=True)
    split_col = get_split_col()
    # Train のみで十分（未出現位置は評価時にマスクされないだけで無害）
    rows = con.execute(f"""
        SELECT l.targets
        FROM labels l
        JOIN samples s ON l.sample_id = s.sample_id
        WHERE s.{split_col} = 0
    """).fetchall()
    con.close()

    votes = defaultdict(Counter)
    for row in rows:
        for t in pickle.loads(row[0]):
            region, position = t[0], t[1]
            if 0 <= position < config.VOCAB_SIZE_POSITION and 0 <= region < config.NUM_REGIONS:
                votes[position][region] += 1

    pos_region = torch.zeros(config.VOCAB_SIZE_POSITION, dtype=torch.long)
    for position, counter in votes.items():
        pos_region[position] = counter.most_common(1)[0][0]

    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(pos_region.numpy(), f)
        print(f"[INFO] Saved position→region map to {cache_file} "
              f"({len(votes):,} positions mapped)")
    except Exception as e:
        print(f"[WARNING] Failed to cache position→region map: {e}")
    return pos_region


def compute_lineage_entropy_map(db_path=None, split_type=0):
    """提案2（学習サンプルフィルタ）用: 系統別の正規化ターゲット位置エントロピーを返す。

    指定スプリットの各サンプルのターゲット位置を系統(strain)ごとに集計し、
    位置分布の Shannon エントロピー（bit）を distinct 位置数の log2 で正規化する
    （utils/io.py の save_lineage_metrics_csv と同一の定義）。

    Returns:
        dict: {strain_name: entropy_norm(float, 0〜1)}
    """
    import os
    import math
    from collections import defaultdict, Counter

    if db_path is None:
        db_path = get_db_path()

    strain_name_col = 'strain_name_usher' if getattr(config, 'STRENGTH_SOURCE', 'ncbi') == 'usher' else 'strain_name_ncbi'
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(
        config.CACHE_DIR, f'lineage_entropy_{strain_name_col}_split{split_type}.pkl')

    if getattr(config, 'FORCE_REPROCESS', False) is False and os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                emap = pickle.load(f)
            print(f"[INFO] Loaded lineage entropy map from {cache_file} ({len(emap)} lineages)")
            return emap
        except Exception as e:
            print(f"[WARNING] Failed to load lineage entropy cache: {e}. Recomputing...")

    print(f"[INFO] Computing lineage target-position entropy (提案2, split={split_type})...")
    con = connect_db(db_path, read_only=True)
    split_col = get_split_col()
    rows = con.execute(f"""
        SELECT st.{strain_name_col} as strain_name, l.targets
        FROM labels l
        JOIN samples s ON l.sample_id = s.sample_id
        JOIN strains st ON s.strain_id = st.strain_id
        WHERE s.{split_col} = ?
    """, [split_type]).fetchall()
    con.close()

    positions_by_lineage = defaultdict(list)
    for strain_name, targets_blob in rows:
        lineage = strain_name or 'unknown'
        for t in pickle.loads(targets_blob):
            positions_by_lineage[lineage].append(t[1])

    entropy_map = {}
    for lineage, pos_list in positions_by_lineage.items():
        if not pos_list:
            entropy_map[lineage] = 0.0
            continue
        counts = Counter(pos_list)
        total = len(pos_list)
        entropy = -sum((c / total) * math.log2(c / total) for c in counts.values())
        max_entropy = math.log2(len(counts)) if len(counts) > 1 else 1.0
        entropy_map[lineage] = entropy / max_entropy if max_entropy > 0 else 0.0

    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(entropy_map, f)
        print(f"[INFO] Saved lineage entropy map to {cache_file} ({len(entropy_map)} lineages)")
    except Exception as e:
        print(f"[WARNING] Failed to cache lineage entropy map: {e}")
    return entropy_map
