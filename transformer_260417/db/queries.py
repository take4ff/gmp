# --- db/queries.py ---
# SQL関連クエリ（サンプル取得・統計・分割割り当て）

import random
import pickle
from .connection import connect_db, get_db_path
from .. import config


def get_db_data_info(db_path=None, split_type=None, max_cooccurrence=None):
    """DBからデータ情報を取得（data_info相当）。

    Returns:
        dict: train/val/test件数・長さ範囲・流行度閾値を含む辞書
    """
    if db_path is None:
        db_path = get_db_path()

    con = connect_db(db_path, read_only=True)

    query_base = "SELECT MIN(path_length), MAX(path_length), COUNT(*) FROM samples WHERE 1=1"

    def get_stats(st):
        q = query_base + f" AND split_type = {st}"
        if max_cooccurrence:
            q += f" AND max_cooccurrence <= {max_cooccurrence}"
        return con.execute(q).fetchone()

    train_stats = get_stats(0)
    valid_stats = get_stats(1)
    test_stats = get_stats(2)

    # 流行度閾値
    strength_stats = con.execute("SELECT MIN(strength_score), MAX(strength_score) FROM samples").fetchone()
    min_s, max_s = strength_stats
    score_range = (max_s - min_s) if max_s and min_s else 10
    low_max = int(min_s + score_range / 3) + 1 if min_s else 6
    med_max = int(min_s + 2 * score_range / 3) + 1 if min_s else 10

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

    query = """
        SELECT s.sample_id, s.raw_path, s.path_length, s.strength_score,
               st.strain_name
        FROM samples s
        JOIN strains st ON s.strain_id = st.strain_id
        WHERE 1=1
    """
    params = []

    if split_type is not None:
        query += " AND s.split_type = ?"
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


def assign_splits(con, train_ratio=0.8, valid_ratio=0.1):
    """パス長に基づいてtrain/valid/testに分割し、DBを更新する。"""
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
        ORDER BY sample_id
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
    query = """
        SELECT l.targets 
        FROM labels l
        JOIN samples s ON l.sample_id = s.sample_id
        WHERE s.split_type = 0
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
    query = """
        SELECT s.sample_id, s.raw_path, st.strain_name, s.split_type, s.strength_score
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
            total_count_pre_unique = len(result) # この簡略化で問題ない（全体割合を維持するため）
            # もしくは、正確に元の result から split_type の件数を数える
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
