# --- utils/io.py ---
# ファイル操作・キャッシュ・結果保存

import os
import pickle
import hashlib
import shutil
import pandas as pd
from . import logging as _log
from .. import config


# ==========================================
# 1. キャッシュ関連
# ==========================================

def get_config_hash():
    """config設定からハッシュを生成（設定変更時にキャッシュを無効化するため）。"""
    try:
        sampling_mode = getattr(config, 'SAMPLING_MODE', 'proportional')

        relevant_configs = [
            str(config.MAX_SEQ_LEN),
            str(config.TRAIN_MAX),
            str(config.VALID_NUM),
            str(config.DATA_BASE_DIR),
            str(config.TARGET_LEN),
            str(config.MAX_CO_OCCURRENCE),
            str(config.VALID_RATIO),
            str(config.SEED),
            str(config.NUM_FEATURE_STRING),
            str(config.NUM_CHEM_FEATURES),
            str(config.ABLATION_MASKS),
            str(sampling_mode),
        ]

        if sampling_mode == 'proportional':
            relevant_configs.append(str(config.MAX_NUM))
        elif sampling_mode == 'fixed_per_strain':
            relevant_configs.append(str(config.MAX_STRAIN_NUM))
            relevant_configs.append(str(config.MAX_NUM_PER_STRAIN))

        config_string = "_".join(relevant_configs)
        return hashlib.md5(config_string.encode('utf-8')).hexdigest()
    except Exception as e:
        print(f"[WARNING] config.py からハッシュを生成できませんでした: {e}")
        return "default_cache"


def load_cache(cache_path):
    """メインキャッシュをロードする。"""
    if os.path.exists(cache_path):
        print(f"[INFO] Loading data from cache: {cache_path}")
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"[WARNING] Failed to load cache: {e}. Re-processing data.")
            return None
    print("[INFO] Cache not found. Processing data...")
    return None


def save_cache(data, cache_path):
    """メインキャッシュを保存する。"""
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        _log.force_print(f"[INFO] Saving data to cache: {cache_path}")
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        _log.force_print(f"[WARNING] Failed to save cache: {e}")


def get_batch_cache_path(batch_idx, total_batches, config_hash):
    """インクリメンタルキャッシュ用のパスを生成する。"""
    filename = f"feat_batch_{config_hash}_{batch_idx}_of_{total_batches}.pkl"
    return os.path.join(config.INCREMENTAL_CACHE_DIR, filename)


def load_batch_cache(path):
    """バッチキャッシュをロードする。"""
    if os.path.exists(path):
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except Exception:
            return None
    return None


def save_batch_cache(data, path):
    """バッチキャッシュを保存する。"""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        _log.force_print(f"[WARNING] Failed to save batch cache: {e}")


def _get_save_path(output_dir, filename):
    """出力先をファイルの種類に応じて csv/, plots/, models/ に振り分ける"""
    if filename.endswith('.csv') or filename.endswith('.txt') and 'strains.txt' in filename:
        target_dir = os.path.join(output_dir, 'csv')
    elif filename.endswith('.png'):
        target_dir = os.path.join(output_dir, 'plots')
    elif filename.endswith('.pth') or filename.endswith('summary.txt') or 'config' in filename or filename.endswith('.json'):
        target_dir = os.path.join(output_dir, 'models')
    else:
        target_dir = output_dir

    os.makedirs(target_dir, exist_ok=True)
    return os.path.join(target_dir, filename)

# ==========================================
# 2. 結果保存関連
# ==========================================

def save_training_log(log_data, output_dir):
    """学習経過をCSVに保存する。"""
    df = pd.DataFrame(log_data)
    path = _get_save_path(output_dir, 'training_log.csv')
    df.to_csv(path, index=False)
    _log.force_print(f"[INFO] Training log saved to {path}")


def save_prediction_results(results, output_dir, prefix="test"):
    """予測結果の詳細をCSVに保存する。

    Args:
        results: evaluate()から返されるdetailed_resultsリスト
        output_dir: 保存先ディレクトリ
        prefix: ファイル名プレフィックス ("valid" or "test")
    """
    rows = []
    for r in results:
        row = {
            'original_len': r['len'],
            'raw_path': r['raw_path'],
            'strain': r['strain'],
            'strength_score': r.get('strength_score', 0.0),
            'pred_strength': r.get('pred_strength', 0.0),
            'collection_date': r.get('collection_date', ''),
            'target_region': str(list(r['targets_region'])),
            'pred_region': str(list(r['preds_region'])),
            'pred_prob_region': r.get('pred_prob_region', 0.0),
            'hit_region': r['hit_region'],
            'target_position': str(list(r['targets_position'])),
            'pred_position': str(list(r['preds_position'])),
            'pred_prob_position': r.get('pred_prob_position', 0.0),
            'hit_position': r['hit_position'],
        }
        if 'targets_aa_pos' in r:
            row['target_aa_pos'] = str(list(r['targets_aa_pos']))
            row['pred_aa_pos'] = str(list(r['preds_aa_pos']))
            row['hit_aa_pos'] = r['hit_aa_pos']
        if 'targets_codon_pos' in r:
            row['target_codon_pos'] = str(list(r['targets_codon_pos']))
            row['pred_codon_pos'] = str(list(r['preds_codon_pos']))
            row['hit_codon_pos'] = r['hit_codon_pos']
        if 'targets_synonymous' in r:
            row['target_synonymous'] = str(list(r['targets_synonymous']))
            row['pred_synonymous'] = str(list(r['preds_synonymous']))
            row['hit_synonymous'] = r['hit_synonymous']
        rows.append(row)

    df = pd.DataFrame(rows)
    path = _get_save_path(output_dir, f'{prefix}_predictions.csv')
    df.to_csv(path, index=False)
    _log.force_print(f"[INFO] Prediction results saved to {path}")


def save_strain_info(strains, output_dir, prefix="train"):
    """使用した株のリストをテキストファイルに保存する。"""
    path = _get_save_path(output_dir, f'{prefix}_strains.txt')
    with open(path, 'w') as f:
        for s in sorted(list(set(strains))):
            f.write(f"{s}\n")
    _log.force_print(f"[INFO] Strain info saved to {path}")


def save_config_copy(output_dir):
    """config.pyのコピーを保存する（再現性のため）。"""
    config_src = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.py')
    config_dst = _get_save_path(output_dir, 'config_snapshot.py')
    try:
        shutil.copy2(config_src, config_dst)
        _log.force_print(f"[INFO] Config snapshot saved to {config_dst}")
    except Exception as e:
        _log.force_print(f"[WARNING] Failed to save config snapshot: {e}")


def save_synonymous_distribution_csv(train_data, valid_data, test_data, output_dir):
    """シノニマス/ノンシノニマスの分布をCSVに保存する。"""
    def count_synonymous(data_list):
        syn_count, non_syn_count = 0, 0
        for item in data_list:
            y_targets = item[1]
            for target in y_targets:
                is_synonymous = target[4] if len(target) > 4 else 0
                if is_synonymous == 1:
                    syn_count += 1
                else:
                    non_syn_count += 1
        return syn_count, non_syn_count

    rows = []
    for name, data in [('Train', train_data), ('Validation', valid_data), ('Test', test_data)]:
        if data:
            syn, non_syn = count_synonymous(data)
            total = syn + non_syn
            rows.append({
                'dataset': name,
                'synonymous_count': syn,
                'non_synonymous_count': non_syn,
                'total': total,
                'synonymous_ratio_pct': syn / total * 100 if total > 0 else 0,
                'non_synonymous_ratio_pct': non_syn / total * 100 if total > 0 else 0
            })

    df = pd.DataFrame(rows)
    path = _get_save_path(output_dir, 'synonymous_distribution.csv')
    df.to_csv(path, index=False)
    _log.force_print(f"[INFO] Synonymous distribution saved to {path}")


def save_metrics_csv(metrics_by_ts, output_dir, prefix="val", save=True):
    """タイムステップ別メトリクスをCSVに保存する。save=False のときはDataFrameのみ返す。"""
    rows = []
    for ts_len in sorted(metrics_by_ts.keys()):
        m = metrics_by_ts[ts_len]
        row = {'timestep': ts_len}
        row.update(m)
        rows.append(row)

    df = pd.DataFrame(rows)
    if save:
        path = _get_save_path(output_dir, f'{prefix}_metrics_by_timestep.csv')
        df.to_csv(path, index=False)
        _log.force_print(f"[INFO] Metrics saved to {path}")
    return df


def save_category_metrics_csv(cat_metrics, output_dir, prefix="val", save=True):
    """流行度カテゴリ別メトリクスをCSVに保存する。save=False のときはDataFrameのみ返す。"""
    rows = []
    for ts_len in sorted(cat_metrics.keys()):
        for category in ['low', 'medium', 'high']:
            m = cat_metrics[ts_len][category]
            row = {'timestep': ts_len, 'category': category}
            row.update(m)
            rows.append(row)

    df = pd.DataFrame(rows)
    if save:
        path = _get_save_path(output_dir, f'{prefix}_metrics_by_category.csv')
        df.to_csv(path, index=False)
        _log.force_print(f"[INFO] Category metrics saved to {path}")

        for category in ['low', 'medium', 'high']:
            cat_rows = [r for r in rows if r['category'] == category]
            if cat_rows:
                cat_path = _get_save_path(output_dir, f'{prefix}_metrics_by_category_{category}.csv')
                pd.DataFrame(cat_rows).to_csv(cat_path, index=False)

    return df


def save_strength_fine_csv(details, output_dir, prefix="test"):
    """対数スケール4区分の流行度別メトリクスをCSVに保存する。"""
    import math
    if not details:
        return None

    bins = [
        ('~100',   math.log1p(100)),
        ('~500',   math.log1p(500)),
        ('~1K',    math.log1p(1_000)),
        ('~5K',    math.log1p(5_000)),
        ('~10K',   math.log1p(10_000)),
        ('~50K',   math.log1p(50_000)),
        ('~100K',  math.log1p(100_000)),
        ('~500K',  math.log1p(500_000)),
        ('>500K',  float('inf')),
    ]
    tasks = ['region', 'position', 'aa_pos', 'codon_pos', 'synonymous']
    fine_stats = {}

    for r in details:
        score = r.get('strength_score', 0.0)
        cat = next((label for label, thr in bins if score < thr), bins[-1][0])
        if cat not in fine_stats:
            fine_stats[cat] = {'n': 0, 'hits': {t: 0 for t in tasks}, 'strains': set()}
        fine_stats[cat]['n'] += 1
        fine_stats[cat]['strains'].add(r.get('strain', ''))
        for t in tasks:
            fine_stats[cat]['hits'][t] += int(r.get(f'hit_{t}', False))

    rows = []
    for label, _ in bins:
        s = fine_stats.get(label, {'n': 0, 'hits': {t: 0 for t in tasks}, 'strains': set()})
        n = s['n']
        row = {'strength_category': label, 'num_samples': n, 'num_strains': len(s['strains'])}
        for t in tasks:
            row[f'{t}_hit_rate_pct'] = round(s['hits'][t] / n * 100, 2) if n > 0 else 0.0
        rows.append(row)

    df = pd.DataFrame(rows)
    path = _get_save_path(output_dir, f'{prefix}_strength_fine.csv')
    df.to_csv(path, index=False)
    _log.force_print(f"[INFO] Fine-grained strength metrics saved to {path}")
    return df


def save_random_baseline_csv(details, output_dir, prefix="test"):
    """ランダムベースラインとの比較をCSVに保存する。"""
    df = pd.DataFrame(details) if details else pd.DataFrame()
    if len(df) == 0:
        return

    random_baselines = {
        'タンパク質領域': 1 / config.NUM_REGIONS * 100,
        '塩基配列位置': 1 / config.VOCAB_SIZE_POSITION * 100,
        'アミノ酸配列位置': 1 / config.VOCAB_SIZE_AA_POS * 100,
        'コドン位置': 1 / 3 * 100,
        'シノニマス': 50.0,
    }

    label_mappings = [
        ('タンパク質領域', 'hit_region'),
        ('塩基配列位置', 'hit_position'),
        ('アミノ酸配列位置', 'hit_aa_pos'),
        ('コドン位置', 'hit_codon_pos'),
        ('シノニマス', 'hit_synonymous'),
    ]

    rows = []
    for label_name, hit_col in label_mappings:
        random_rate = random_baselines.get(label_name, 0)
        hit_rate = df[hit_col].mean() * 100 if hit_col in df.columns else 0
        improvement = hit_rate / random_rate if random_rate > 0 else 0

        rows.append({
            'prediction_target': label_name,
            'random_baseline_pct': random_rate,
            'model_hit_rate_pct': hit_rate,
            'improvement_ratio': improvement
        })

    result_df = pd.DataFrame(rows)
    path = _get_save_path(output_dir, f'{prefix}_random_baseline.csv')
    result_df.to_csv(path, index=False)
    _log.force_print(f"[INFO] Random baseline comparison saved to {path}")


def save_region_metrics_csv(details, output_dir, prefix="test"):
    """タンパク質領域別メトリクスをCSVに保存する。"""
    df = pd.DataFrame(details) if details else pd.DataFrame()
    if len(df) == 0:
        return

    region_names = {v: k for k, v in config.PROTEIN_VOCABS.items()}
    region_stats = {}

    for _, row in df.iterrows():
        for reg_id in row['targets_region']:
            if reg_id not in region_stats:
                region_stats[reg_id] = {'tp': 0, 'fp': 0, 'fn': 0, 'count': 0}
            region_stats[reg_id]['count'] += 1
            if reg_id in row['preds_region']:
                region_stats[reg_id]['tp'] += 1
            else:
                region_stats[reg_id]['fn'] += 1
        for pred_id in row['preds_region']:
            if pred_id not in row['targets_region']:
                if pred_id not in region_stats:
                    region_stats[pred_id] = {'tp': 0, 'fp': 0, 'fn': 0, 'count': 0}
                region_stats[pred_id]['fp'] += 1

    rows = []
    for reg_id in sorted(region_stats.keys(), key=lambda x: region_stats[x]['count'], reverse=True):
        s = region_stats[reg_id]
        prec = s['tp'] / (s['tp'] + s['fp']) * 100 if (s['tp'] + s['fp']) > 0 else 0
        rec = s['tp'] / (s['tp'] + s['fn']) * 100 if (s['tp'] + s['fn']) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

        rows.append({
            'region_id': reg_id,
            'region_name': region_names.get(reg_id, f"ID:{reg_id}"),
            'sample_count': s['count'],
            'true_positives': s['tp'],
            'false_positives': s['fp'],
            'false_negatives': s['fn'],
            'precision_pct': prec,
            'recall_pct': rec,
            'f1_pct': f1
        })

    result_df = pd.DataFrame(rows)
    path = _get_save_path(output_dir, f'{prefix}_region_metrics.csv')
    result_df.to_csv(path, index=False)
    _log.force_print(f"[INFO] Region metrics saved to {path}")


def save_prediction_distribution_csv(details, output_dir, prefix="test"):
    """予測分布（どの領域に何回予測したか）をCSVに保存する。"""
    df = pd.DataFrame(details) if details else pd.DataFrame()
    if len(df) == 0:
        return

    region_names = {v: k for k, v in config.PROTEIN_VOCABS.items()}

    pred_counts = {}
    target_counts = {}

    for _, row in df.iterrows():
        for pred_id in row['preds_region']:
            pred_counts[pred_id] = pred_counts.get(pred_id, 0) + 1
        for target_id in row['targets_region']:
            target_counts[target_id] = target_counts.get(target_id, 0) + 1

    all_ids = set(pred_counts.keys()) | set(target_counts.keys())

    rows = []
    for reg_id in sorted(all_ids, key=lambda x: pred_counts.get(x, 0) + target_counts.get(x, 0), reverse=True):
        rows.append({
            'region_id': reg_id,
            'region_name': region_names.get(reg_id, f"ID:{reg_id}"),
            'prediction_count': pred_counts.get(reg_id, 0),
            'target_count': target_counts.get(reg_id, 0)
        })

    result_df = pd.DataFrame(rows)
    path = _get_save_path(output_dir, f'{prefix}_prediction_distribution.csv')
    result_df.to_csv(path, index=False)
    _log.force_print(f"[INFO] Prediction distribution saved to {path}")


def save_confusion_matrix_csv(details, output_dir, prefix="test"):
    """タンパク質領域の混同行列をCSVに保存する。"""
    df = pd.DataFrame(details) if details else pd.DataFrame()
    if len(df) == 0:
        return

    region_names = {v: k for k, v in config.PROTEIN_VOCABS.items()}
    confusion = {}

    for _, row in df.iterrows():
        targets = list(row['targets_region'])
        preds = list(row['preds_region'])

        for t in targets:
            for p in preds:
                key = (t, p)
                confusion[key] = confusion.get(key, 0) + 1

    rows = []
    for (true_id, pred_id), count in sorted(confusion.items(), key=lambda x: x[1], reverse=True):
        rows.append({
            'true_region_id': true_id,
            'true_region_name': region_names.get(true_id, f"ID:{true_id}"),
            'pred_region_id': pred_id,
            'pred_region_name': region_names.get(pred_id, f"ID:{pred_id}"),
            'count': count
        })

    result_df = pd.DataFrame(rows)
    path = _get_save_path(output_dir, f'{prefix}_confusion_matrix.csv')
    result_df.to_csv(path, index=False)
    _log.force_print(f"[INFO] Confusion matrix saved to {path}")


def save_recall_summary_csv(metrics_by_ts, output_dir, prefix="test"):
    """Weighted Recall@1 と Macro Recall@1 のサマリーをCSVに保存する。

    全タイムステップを num_samples で加重平均した 1行サマリーと、
    タイムステップ別の詳細行を含む CSV を出力する。

    Args:
        metrics_by_ts: evaluate() が返す final_metrics_by_ts
        output_dir: 保存先ディレクトリ
        prefix: ファイル名プレフィックス
    """
    tasks = [
        ('region',      'タンパク質領域'),
        ('position',    '塩基配列位置'),
        ('aa_pos',      'アミノ酸配列位置'),
        ('codon_pos',   'コドン位置'),
        ('synonymous',  'シノニマス'),
    ]

    # タイムステップ別の行
    rows = []
    for ts_len in sorted(metrics_by_ts.keys()):
        m = metrics_by_ts[ts_len]
        row = {'timestep': ts_len, 'num_samples': m['num_samples']}
        for task_key, _ in tasks:
            row[f'{task_key}_weighted_recall'] = m.get(f'{task_key}_weighted_recall', 0.0)
            row[f'{task_key}_macro_recall'] = m.get(f'{task_key}_macro_recall', 0.0)
        rows.append(row)

    ts_df = pd.DataFrame(rows)

    # 全体集約行（num_samples で加重平均）
    total_samples = sum(m['num_samples'] for m in metrics_by_ts.values())
    if total_samples > 0:
        summary_row = {'timestep': 'ALL', 'num_samples': total_samples}
        for task_key, _ in tasks:
            w_avg = sum(
                m.get(f'{task_key}_weighted_recall', 0.0) * m['num_samples']
                for m in metrics_by_ts.values()
            ) / total_samples
            m_avg = sum(
                m.get(f'{task_key}_macro_recall', 0.0) * m['num_samples']
                for m in metrics_by_ts.values()
            ) / total_samples
            summary_row[f'{task_key}_weighted_recall'] = w_avg
            summary_row[f'{task_key}_macro_recall'] = m_avg
        ts_df = pd.concat([ts_df, pd.DataFrame([summary_row])], ignore_index=True)

    path = _get_save_path(output_dir, f'{prefix}_recall_summary.csv')
    ts_df.to_csv(path, index=False)
    _log.force_print(f"[INFO] Recall summary saved to {path}")

    # コンソール表示（ALL行のみ）
    _log.force_print(f"\n[INFO] === Recall@1 Summary ({prefix.upper()}) ===")
    _log.force_print(f"  {'Task':<22} {'WeightedRecall@1':>18} {'MacroRecall@1':>15}")
    _log.force_print(f"  {'-'*57}")
    if total_samples > 0:
        for task_key, task_label in tasks:
            w = summary_row[f'{task_key}_weighted_recall']
            m = summary_row[f'{task_key}_macro_recall']
            _log.force_print(f"  {task_label:<22} {w:>17.2f}%  {m:>13.2f}%")

    return ts_df


def save_per_position_recall_csv(details, output_dir, prefix="test"):
    """塩基位置ごとの Recall@1（TP/出現数）をCSVに保存する。"""
    pos_tp = {}
    pos_cnt = {}
    for r in details:
        targets = r.get('targets_position', set())
        preds   = r.get('preds_position', set())
        for p in targets:
            pos_cnt[p] = pos_cnt.get(p, 0) + 1
            if p in preds:
                pos_tp[p] = pos_tp.get(p, 0) + 1

    rows = [
        {'position_id': pos_id, 'total_targets': cnt,
         'true_positives': pos_tp.get(pos_id, 0),
         'recall_pct': pos_tp.get(pos_id, 0) / cnt * 100 if cnt > 0 else 0.0}
        for pos_id, cnt in sorted(pos_cnt.items())
    ]
    df = pd.DataFrame(rows).sort_values('total_targets', ascending=False)
    path = _get_save_path(output_dir, f'{prefix}_per_position_recall.csv')
    df.to_csv(path, index=False)
    _log.force_print(f"[INFO] Per-position recall saved to {path}")
    return df


def save_val_test_gap_csv(val_metrics, test_metrics, output_dir):
    """Validation と Test の指標差（Generalization Gap）をCSVに保存する。"""
    if not val_metrics or not test_metrics:
        return

    metric_keys = [
        'region_hit_rate', 'position_hit_rate', 'aa_pos_hit_rate',
        'codon_pos_hit_rate', 'synonymous_hit_rate',
        'region_weighted_recall', 'position_weighted_recall',
        'aa_pos_weighted_recall', 'region_macro_recall', 'position_macro_recall',
        'strength_mae',
    ]

    def _wavg(metrics_by_ts, key):
        total = sum(m['num_samples'] for m in metrics_by_ts.values())
        if total == 0:
            return 0.0
        return sum(m.get(key, 0) * m['num_samples'] for m in metrics_by_ts.values()) / total

    rows = []
    for key in metric_keys:
        v = _wavg(val_metrics, key)
        t = _wavg(test_metrics, key)
        rows.append({
            'metric': key,
            'validation': round(v, 4),
            'test': round(t, 4),
            'gap_val_minus_test': round(v - t, 4),
            'gap_pct': round((v - t) / v * 100, 2) if v != 0 else 0.0,
        })

    df = pd.DataFrame(rows)
    path = _get_save_path(output_dir, 'val_test_gap.csv')
    df.to_csv(path, index=False)
    _log.force_print(f"[INFO] Val-Test generalization gap saved to {path}")
    return df


def save_error_analysis_csv(details, output_dir, prefix="test"):
    """全タスクで予測を外したサンプル（完全失敗例）をCSVに保存する。"""
    hard_cases = [
        r for r in details
        if not r.get('hit_region') and not r.get('hit_position') and not r.get('hit_aa_pos')
    ]
    if not hard_cases:
        _log.force_print(f"[INFO] Error analysis: no complete miss samples ({prefix})")
        return

    rows = [{
        'strain': r.get('strain'),
        'len': r.get('len'),
        'strength_score': r.get('strength_score', 0.0),
        'pred_strength': r.get('pred_strength', 0.0),
        'raw_path': r.get('raw_path', ''),
        'targets_region': str(list(r.get('targets_region', set()))),
        'preds_region': str(list(r.get('preds_region', set()))),
        'targets_position': str(list(r.get('targets_position', set()))),
        'preds_position': str(list(r.get('preds_position', set()))),
    } for r in hard_cases]

    df = pd.DataFrame(rows)
    path = _get_save_path(output_dir, f'{prefix}_error_analysis.csv')
    df.to_csv(path, index=False)
    _log.force_print(
        f"[INFO] Error analysis: {len(hard_cases)}/{len(details)} complete miss "
        f"({len(hard_cases)/len(details)*100:.1f}%) saved to {path}"
    )
    return df


def save_strain_metrics_csv(details, output_dir, prefix="test"):
    """株（Pango lineage）ごとの各タスク Hit Rate をCSVに保存する。"""
    tasks = ['region', 'position', 'aa_pos', 'codon_pos', 'synonymous']
    strain_stats = {}

    for r in details:
        strain = r.get('strain', 'unknown')
        if strain not in strain_stats:
            strain_stats[strain] = {t: {'hits': 0, 'n': 0} for t in tasks}
            strain_stats[strain].update({'strength_sum': 0.0, 'count': 0})
        strain_stats[strain]['count'] += 1
        strain_stats[strain]['strength_sum'] += r.get('strength_score', 0.0)
        for task in tasks:
            hit_key = f'hit_{task}'
            if hit_key in r:
                strain_stats[strain][task]['n']    += 1
                strain_stats[strain][task]['hits'] += int(r[hit_key])

    rows = []
    for strain, s in sorted(strain_stats.items(), key=lambda x: x[1]['count'], reverse=True):
        row = {
            'strain': strain,
            'num_samples': s['count'],
            'avg_strength': s['strength_sum'] / s['count'] if s['count'] > 0 else 0.0,
        }
        for task in tasks:
            n = s[task]['n']
            row[f'{task}_hit_rate_pct'] = s[task]['hits'] / n * 100 if n > 0 else 0.0
        rows.append(row)

    df = pd.DataFrame(rows)
    path = _get_save_path(output_dir, f'{prefix}_strain_metrics.csv')
    df.to_csv(path, index=False)
    _log.force_print(f"[INFO] Strain metrics saved to {path} ({len(rows)} strains)")
    return df


def save_lineage_metrics_csv(details, output_dir, prefix="test"):
    """系統（フルPango名）単位で精度・予測難易度（エントロピー）を集計したCSVを保存する。"""
    import math
    from collections import Counter

    if not details:
        return None

    # DB 基準（フィルタ前・重複込み）のエントロピーを併記する。モデル評価済み(eval)基準と
    # 対で見られるよう、同じフル Pango 粒度で系統別に DB から計算する。
    db_entropy_map = {}
    try:
        from ..db.queries import get_db_lineage_target_entropy
        split_type = {'train': 0, 'valid': 1, 'test': 2}.get(prefix, 2)
        db_entropy_map = get_db_lineage_target_entropy(split_type)
    except Exception as e:
        _log.force_print(f"[WARNING] DB基準エントロピーの計算をスキップ（eval基準のみ出力）: {e}")

    tasks = ['region', 'position', 'aa_pos', 'codon_pos', 'synonymous']
    lineage_stats = {}

    for r in details:
        strain = r.get('strain', 'unknown') or 'unknown'
        lineage = strain

        if lineage not in lineage_stats:
            lineage_stats[lineage] = {
                'n': 0, 'hits': {t: 0 for t in tasks},
                'strength_sum': 0.0, 'target_positions': [],
            }
        s = lineage_stats[lineage]
        s['n'] += 1
        s['strength_sum'] += r.get('strength_score', 0.0)
        for t in tasks:
            s['hits'][t] += int(r.get(f'hit_{t}', False))
        s['target_positions'].extend(list(r.get('targets_position', set())))

    rows = []
    for lineage, s in sorted(lineage_stats.items(), key=lambda x: x[1]['n'], reverse=True):
        n = s['n']
        pos_list = s['target_positions']
        entropy = 0.0
        normalized_entropy = 0.0
        if pos_list:
            counts = Counter(pos_list)
            total = len(pos_list)
            entropy = -sum((c / total) * math.log2(c / total) for c in counts.values())
            max_entropy = math.log2(len(counts)) if len(counts) > 1 else 1.0
            normalized_entropy = entropy / max_entropy

        db_ent = db_entropy_map.get(lineage, (0.0, 0.0))
        row = {
            'lineage': lineage,
            'num_samples': n,
            'avg_strength': round(s['strength_sum'] / n, 4) if n > 0 else 0.0,
            # eval 基準: モデルが実際に評価した(フィルタ後)サンプルの多様性
            'eval_target_entropy': round(entropy, 4),
            'eval_target_entropy_norm': round(normalized_entropy, 4),
            # db 基準: DB 生サンプル(フィルタ前・重複込み)の多様性
            'db_target_entropy': db_ent[0],
            'db_target_entropy_norm': db_ent[1],
        }
        for t in tasks:
            row[f'{t}_hit_rate_pct'] = round(s['hits'][t] / n * 100, 2) if n > 0 else 0.0
        rows.append(row)

    df = pd.DataFrame(rows)
    path = _get_save_path(output_dir, f'{prefix}_lineage_metrics.csv')
    df.to_csv(path, index=False)
    _log.force_print(f"[INFO] Lineage metrics saved to {path} ({len(rows)} lineages)")
    return df


def save_entropy_bin_csv(details, output_dir, prefix="test"):
    """エントロピー(0.1刻み)ビン別の精度集計CSVを保存する。"""
    import math
    from collections import Counter

    if not details:
        return None

    tasks = ['region', 'position', 'aa_pos', 'codon_pos', 'synonymous']

    # 系統単位でentropy_normとhit数を集計
    lineage_stats = {}
    for r in details:
        lin = r.get('strain', 'unknown') or 'unknown'
        if lin not in lineage_stats:
            lineage_stats[lin] = {'n': 0, 'hits': {t: 0 for t in tasks}, 'positions': []}
        s = lineage_stats[lin]
        s['n'] += 1
        for t in tasks:
            s['hits'][t] += int(r.get(f'hit_{t}', False))
        s['positions'].extend(list(r.get('targets_position', set())))

    # 系統ごとのentropy_norm計算
    lineage_entropy = {}
    for lin, s in lineage_stats.items():
        pos_list = s['positions']
        if pos_list:
            counts = Counter(pos_list)
            total = len(pos_list)
            entr = -sum((c / total) * math.log2(c / total) for c in counts.values())
            max_e = math.log2(len(counts)) if len(counts) > 1 else 1.0
            lineage_entropy[lin] = entr / max_e
        else:
            lineage_entropy[lin] = 0.0

    # サンプル単位でビンに振り分け
    bin_edges = [i / 10 for i in range(11)]  # 0.0, 0.1, ..., 1.0
    bin_labels = [f'{bin_edges[i]:.1f}~{bin_edges[i+1]:.1f}' for i in range(len(bin_edges) - 1)]
    bin_data = {lbl: {'n': 0, 'strains': set(), 'hits': {t: 0 for t in tasks}} for lbl in bin_labels}

    for lin, s in lineage_stats.items():
        e = lineage_entropy[lin]
        idx = min(int(e * 10), 9)  # 1.0 は最終ビンへ
        lbl = bin_labels[idx]
        bin_data[lbl]['n'] += s['n']
        bin_data[lbl]['strains'].add(lin)
        for t in tasks:
            bin_data[lbl]['hits'][t] += s['hits'][t]

    rows = []
    for lbl in bin_labels:
        b = bin_data[lbl]
        n = b['n']
        row = {
            'entropy_bin': lbl,
            'num_samples': n,
            'num_strains': len(b['strains']),
        }
        for t in tasks:
            row[f'{t}_hit_rate_pct'] = round(b['hits'][t] / n * 100, 2) if n > 0 else 0.0
        rows.append(row)

    df = pd.DataFrame(rows)
    path = _get_save_path(output_dir, f'{prefix}_entropy_bin.csv')
    df.to_csv(path, index=False)
    _log.force_print(f"[INFO] Entropy bin CSV saved to {path}")
    return df


def save_confident_recall_csv(details, output_dir, prefix="test"):
    """高確信サブセットの Top-1 hit-rate（Recall@1 相当）をカバレッジ別に集計する（C-4）。

    各サンプルは evaluate.py で Top-1 予測確率（pred_prob_<task>）と hit フラグ
    （hit_<task>）を持つ。確信度の高い順にソートし、上位カバレッジ割合ごとに
    hit-rate と平均確信度を出す。coverage=1.0 は全体（既存の hit-rate と一致）。
    多義な系統を確信度で切り落としたときにどこまで精度が上がるかを示す。

    出力: {prefix}_confident_subset.csv（列: task, coverage, n, avg_confidence, hit_rate_pct）
    """
    if not details:
        return None
    levels = getattr(config, 'CONFIDENT_COVERAGE_LEVELS', [1.0, 0.75, 0.5, 0.25])
    tasks = [
        ('region',   'pred_prob_region',   'hit_region'),
        ('position', 'pred_prob_position', 'hit_position'),
        ('aa_pos',   'pred_prob_aa_pos',   'hit_aa_pos'),
    ]
    rows = []
    for task, pcol, hcol in tasks:
        pairs = [(r[pcol], bool(r[hcol])) for r in details
                 if r.get(pcol) is not None and r.get(hcol) is not None]
        if not pairs:
            continue
        pairs.sort(key=lambda x: x[0], reverse=True)   # 確信度降順
        n_total = len(pairs)
        for cov in levels:
            n = max(1, int(round(n_total * cov)))
            subset = pairs[:n]
            hit_rate = 100.0 * sum(1 for _, h in subset if h) / len(subset)
            avg_conf = sum(p for p, _ in subset) / len(subset)
            rows.append({
                'task': task,
                'coverage': cov,
                'n': len(subset),
                'avg_confidence': round(avg_conf, 4),
                'hit_rate_pct': round(hit_rate, 3),
            })
    if not rows:
        return None
    df = pd.DataFrame(rows)
    path = _get_save_path(output_dir, f'{prefix}_confident_subset.csv')
    df.to_csv(path, index=False)
    _log.force_print(f"[INFO] Confident-subset recall saved to {path}")
    # position（最も伸びしろの大きいタスク）のカバレッジ別 hit-rate をログに要約
    for r in rows:
        if r['task'] == 'position':
            _log.force_print(f"        Position cov={r['coverage']:.2f}: "
                             f"hit={r['hit_rate_pct']:.2f}% (n={r['n']:,}, conf={r['avg_confidence']:.3f})")
    return df


def save_nll_csv(details, output_dir, prefix="test"):
    """ホールドアウト NLL（負の対数尤度）をタスク別に集計して保存する。

    NLL = -(1/|T|) Σ_{t∈T} log p(t)。参照ターゲット T は真の変異集合上の一様分布
    （config 不変）、p はモデルの全クラス予測分布（evaluate.py で per-sample 算出済み）。
    値が小さいほど予測分布が良い。perplexity = exp(NLL)。AIC の代替として、モデル構成間の
    「複雑さを検証データで暗黙に罰した」公平な予測性能比較に使える（＋ early stopping と併用）。

    出力:
      - {prefix}_nll.csv          : タスク別の overall NLL / bits / perplexity / n
      - {prefix}_nll_by_entropy.csv: 系統エントロピー 0.1 刻みビン別の Position/AAPos NLL
    Returns:
      pd.DataFrame (overall)
    """
    import math
    from collections import Counter

    if not details:
        return None

    tasks = ['region', 'position', 'aa_pos', 'codon_pos', 'synonymous']

    # --- overall（タスク別）---
    rows = []
    for t in tasks:
        vals = [r[f'nll_{t}'] for r in details if r.get(f'nll_{t}') is not None]
        if not vals:
            continue
        mean_nll = sum(vals) / len(vals)
        rows.append({
            'task': t,
            'n_samples': len(vals),
            'nll_nats': round(mean_nll, 4),
            'nll_bits': round(mean_nll / math.log(2), 4),
            'perplexity': round(math.exp(mean_nll), 2),
        })
    df = pd.DataFrame(rows)
    path = _get_save_path(output_dir, f'{prefix}_nll.csv')
    df.to_csv(path, index=False)
    _log.force_print(f"[INFO] Held-out NLL saved to {path}")
    for r in rows:
        _log.force_print(f"        NLL[{r['task']:>10}] = {r['nll_nats']:.4f} nats "
                         f"(ppl {r['perplexity']:.1f}, n={r['n_samples']:,})")

    # --- entropy-stratified（save_entropy_bin_csv と同じビニング）---
    lineage_pos = {}
    for r in details:
        lin = r.get('strain', 'unknown') or 'unknown'
        lineage_pos.setdefault(lin, []).extend(list(r.get('targets_position', set())))
    lineage_entropy = {}
    for lin, pos_list in lineage_pos.items():
        if pos_list:
            counts = Counter(pos_list)
            total = len(pos_list)
            entr = -sum((c / total) * math.log2(c / total) for c in counts.values())
            max_e = math.log2(len(counts)) if len(counts) > 1 else 1.0
            lineage_entropy[lin] = entr / max_e if max_e > 0 else 0.0
        else:
            lineage_entropy[lin] = 0.0

    bin_edges = [i / 10 for i in range(11)]
    bin_labels = [f'{bin_edges[i]:.1f}~{bin_edges[i+1]:.1f}' for i in range(len(bin_edges) - 1)]
    bin_acc = {lbl: {t: [] for t in tasks} | {'n': 0} for lbl in bin_labels}
    for r in details:
        lin = r.get('strain', 'unknown') or 'unknown'
        idx = min(int(lineage_entropy.get(lin, 0.0) * 10), 9)
        lbl = bin_labels[idx]
        bin_acc[lbl]['n'] += 1
        for t in tasks:
            v = r.get(f'nll_{t}')
            if v is not None:
                bin_acc[lbl][t].append(v)

    ent_rows = []
    for lbl in bin_labels:
        b = bin_acc[lbl]
        row = {'entropy_bin': lbl, 'num_samples': b['n']}
        for t in tasks:
            row[f'{t}_nll_nats'] = round(sum(b[t]) / len(b[t]), 4) if b[t] else None
        ent_rows.append(row)
    ent_df = pd.DataFrame(ent_rows)
    ent_path = _get_save_path(output_dir, f'{prefix}_nll_by_entropy.csv')
    ent_df.to_csv(ent_path, index=False)
    _log.force_print(f"[INFO] Held-out NLL by entropy bin saved to {ent_path}")

    return df


def save_early_stopping_json(training_log, best_model_path, output_dir, max_epochs):
    """Early Stopping 情報を JSON に保存する。"""
    import json
    if not training_log:
        return

    # nan \u30ac\u30fc\u30c9: val_loss \u304c nan \u306e\u884c\u3092\u9664\u5916\u3057\u3066 best_epoch \u3092\u78ba\u5b9a\u3059\u308b
    # (Hard Target + cbce \u7b49\u3067 nan \u304c\u767a\u751f\u3057\u305f\u5834\u5408\u306b min() \u304c\u8ab0\u686f\u306b\u306a\u308b\u554f\u984c\u3092\u9632\u3050)
    valid_rows = [r for r in training_log if r['val_loss'] == r['val_loss']]  # nan != nan \u3092\u5229\u7528
    rows_for_best = valid_rows if valid_rows else training_log  # \u5168\u884c nan \u306a\u3089\u5148\u982d\u884c\u3092\u4ee3\u7528
    best_row = min(rows_for_best, key=lambda x: x['val_loss'])

    last_row = training_log[-1]

    info = {
        'total_epochs_run': len(training_log),
        'max_epochs': max_epochs,
        'early_stopped': len(training_log) < max_epochs,
        'best_epoch': best_row['epoch'],
        'best_val_loss': round(best_row['val_loss'], 6),
        'best_train_loss': round(best_row['train_loss'], 6),
        'final_epoch': last_row['epoch'],
        'final_val_loss': round(last_row['val_loss'], 6),
        'best_model_path': str(best_model_path),
        'total_training_time_s': round(sum(r['time_seconds'] for r in training_log), 1),
    }

    path = _get_save_path(output_dir, 'early_stopping_info.json')
    with open(path, 'w') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    _log.force_print(
        f"[INFO] Early stopping info saved to {path} "
        f"(best epoch={info['best_epoch']}, val_loss={info['best_val_loss']})"
    )
    return info


def save_run_summary_json(val_metrics, test_metrics, val_topk, test_topk,
                          val_loss, test_loss, output_dir):
    """主要指標だけをまとめた軽量 JSON を保存する。"""
    import json

    def _weighted_avg(metrics, key):
        if not metrics:
            return None
        total_n = sum(m.get('num_samples', 0) for m in metrics.values())
        if total_n == 0:
            return None
        return round(sum(m.get(key, 0.0) * m.get('num_samples', 0) for m in metrics.values()) / total_n, 4)

    def _topk_entry(topk, task, k):
        if topk is None:
            return None
        m = topk.get(task, {}).get(k)
        return round(m['hit_rate'], 4) if m else None

    def _r_prec_entry(topk, task):
        if topk is None:
            return None
        rp = topk.get(task, {}).get('r_precision')
        return round(rp['hit_rate'], 4) if rp else None

    eval_ks = getattr(config, 'EVAL_TOP_KS', (1, 3, 5))
    summary = {
        'val_loss': round(val_loss, 6) if val_loss is not None else None,
        'test_loss': round(test_loss, 6) if test_loss is not None else None,
        'val': {
            'macro_recall_region': _weighted_avg(val_metrics, 'region_macro_recall'),
            'macro_recall_position': _weighted_avg(val_metrics, 'position_macro_recall'),
            'base_after_accuracy': _weighted_avg(val_metrics, 'base_after_accuracy'),
            'aa_after_accuracy': _weighted_avg(val_metrics, 'aa_after_accuracy'),
            **{f'top{k}_region_hit_rate_pct': _topk_entry(val_topk, 'region', k) for k in eval_ks},
            **{f'top{k}_position_hit_rate_pct': _topk_entry(val_topk, 'position', k) for k in eval_ks},
            **{f'top{k}_aa_pos_hit_rate_pct': _topk_entry(val_topk, 'aa_pos', k) for k in eval_ks},
            'r_precision_region': _r_prec_entry(val_topk, 'region'),
            'r_precision_position': _r_prec_entry(val_topk, 'position'),
            'r_precision_aa_pos': _r_prec_entry(val_topk, 'aa_pos'),
        },
        'test': {
            'macro_recall_region': _weighted_avg(test_metrics, 'region_macro_recall'),
            'macro_recall_position': _weighted_avg(test_metrics, 'position_macro_recall'),
            'base_after_accuracy': _weighted_avg(test_metrics, 'base_after_accuracy'),
            'aa_after_accuracy': _weighted_avg(test_metrics, 'aa_after_accuracy'),
            **{f'top{k}_region_hit_rate_pct': _topk_entry(test_topk, 'region', k) for k in eval_ks},
            **{f'top{k}_position_hit_rate_pct': _topk_entry(test_topk, 'position', k) for k in eval_ks},
            **{f'top{k}_aa_pos_hit_rate_pct': _topk_entry(test_topk, 'aa_pos', k) for k in eval_ks},
            'r_precision_region': _r_prec_entry(test_topk, 'region'),
            'r_precision_position': _r_prec_entry(test_topk, 'position'),
            'r_precision_aa_pos': _r_prec_entry(test_topk, 'aa_pos'),
        },
    }

    path = _get_save_path(output_dir, 'run_summary.json')
    with open(path, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    _log.force_print(f"[INFO] Run summary saved to {path}")
    return summary


def save_model_summary_txt(model, output_dir):
    """モデルのアーキテクチャ・パラメータ数をテキストファイルに保存する。"""
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    lines = [
        "=" * 70,
        "Model Summary",
        "=" * 70,
        str(model),
        "",
        f"Total parameters:     {total_params:,}",
        f"Trainable parameters: {trainable_params:,}",
        f"Non-trainable:        {total_params - trainable_params:,}",
        "=" * 70,
    ]

    path = _get_save_path(output_dir, 'model_summary.txt')
    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    _log.force_print(f"[INFO] Model summary saved to {path} (total params: {total_params:,})")


def save_topk_precision_csv(topk_results, output_dir, prefix="test"):
    """Top-K Precision / Recall / Hit Rate の比較表をCSVに保存する。"""
    task_labels = {
        'region': 'region', 'position': 'position',
        'aa_pos': 'aa_pos', 'codon_pos': 'codon_pos', 'synonymous': 'synonymous',
    }
    rows = []
    for task, k_dict in topk_results.items():
        for k, metrics in k_dict.items():
            if not isinstance(k, int):
                continue  # 'r_precision' 等の非整数キーは別ファイルに出力
            rows.append({
                'task': task,
                'top_k': k,
                'precision_pct': round(metrics['precision'], 2),
                'recall_pct': round(metrics['recall'], 2),
                'hit_rate_pct': round(metrics['hit_rate'], 2),
            })

    df = pd.DataFrame(rows)
    path = _get_save_path(output_dir, f'{prefix}_topk_precision.csv')
    df.to_csv(path, index=False)
    _log.force_print(f"[INFO] Top-K precision summary saved to {path}")

    # コンソール表示
    _log.force_print(f"\n[INFO] === Top-K Precision ({prefix.upper()}) ===")
    _log.force_print(f"  {'Task':<20} {'K':>4} {'Precision':>11} {'Recall':>9} {'HitRate':>9}")
    _log.force_print(f"  {'-'*58}")
    for task in task_labels:
        if task not in topk_results:
            continue
        for k in sorted(k for k in topk_results[task] if isinstance(k, int)):
            m = topk_results[task][k]
            _log.force_print(
                f"  {task:<20} {k:>4}  {m['precision']:>9.2f}%  {m['recall']:>7.2f}%  {m['hit_rate']:>7.2f}%"
            )
    return df


def save_r_precision_csv(topk_results, output_dir, prefix="test"):
    """R-Precision（動的K=|target|）の結果を独立ファイルに保存する。"""
    if not topk_results:
        return None

    rows = []
    for task in ['region', 'position', 'aa_pos', 'codon_pos', 'synonymous']:
        rp = topk_results.get(task, {}).get('r_precision')
        if rp is not None:
            rows.append({
                'task': task,
                'precision_pct': round(rp['precision'], 4),
                'recall_pct': round(rp['recall'], 4),
                'hit_rate_pct': round(rp['hit_rate'], 4),
            })

    if not rows:
        return None

    df = pd.DataFrame(rows)
    path = _get_save_path(output_dir, f'{prefix}_r_precision.csv')
    df.to_csv(path, index=False)
    _log.force_print(f"[INFO] R-Precision results saved to {path}")
    return df


def save_combined_metrics_csv(val_df, test_df, output_dir, filename):
    """Validation と Test のメトリクス DataFrame を結合してCSVに保存する。"""
    if val_df is None or test_df is None:
        return None

    val_copy = val_df.copy()
    test_copy = test_df.copy()

    # split カラムを先頭に追加
    val_copy.insert(0, 'split', 'valid')
    test_copy.insert(0, 'split', 'test')

    combined_df = pd.concat([val_copy, test_copy], ignore_index=True)

    # timestep カラムが存在する場合はソートする
    if 'timestep' in combined_df.columns:
        sort_cols = ['timestep', 'split']
        if 'category' in combined_df.columns:
            sort_cols.append('category')
        combined_df = combined_df.sort_values(by=sort_cols).reset_index(drop=True)

    path = _get_save_path(output_dir, filename)
    combined_df.to_csv(path, index=False)
    _log.force_print(f"[INFO] Combined metrics saved to {path}")
    return combined_df


def save_date_metrics_csv(metrics_by_ym, output_dir, prefix="test"):
    """月別（Year-Month）メトリクスを CSV に保存する。

    Args:
        metrics_by_ym: evaluate() が返す final_metrics_by_ym
                       キーが 'YYYY-MM' 文字列、値が metrics dict
        output_dir:    保存先ディレクトリ
        prefix:        ファイル名プレフィックス ('valid' or 'test')

    Returns:
        pd.DataFrame または None（データが空の場合）
    """
    if not metrics_by_ym:
        return None

    rows = []
    for ym in sorted(metrics_by_ym.keys()):
        m = metrics_by_ym[ym]
        row = {'yearmonth': ym}
        row.update(m)
        rows.append(row)

    df = pd.DataFrame(rows)
    path = _get_save_path(output_dir, f'{prefix}_metrics_by_date.csv')
    df.to_csv(path, index=False)
    _log.force_print(f"[INFO] Date metrics saved to {path}")
    return df


def save_calibration_csv(calib_bins, output_dir, prefix="test"):
    """キャリブレーション bin データを CSV に保存する（Reliability Diagram 用）。

    columns: task, bin, lower, upper, avg_confidence, avg_accuracy, count
    SAVE_ECE=True のときのみ calib_bins にデータが入る。
    """
    rows = []
    for task, bins in calib_bins.items():
        for b in bins:
            rows.append({'task': task, **b})
    if not rows:
        return
    df = pd.DataFrame(rows)
    path = _get_save_path(output_dir, f'{prefix}_calibration_bins.csv')
    df.to_csv(path, index=False)
    _log.force_print(f"[INFO] Calibration bin data saved to {path}")
