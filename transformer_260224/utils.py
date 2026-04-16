# --- utils.py ---
import torch
import torch.nn as nn
import math
import pickle
import os
import hashlib
import time
import pandas as pd
import json
from tabulate import tabulate
from . import config

# ==========================================
# 1. ログ・表示関連
# ==========================================

def force_print(message):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"[{timestamp}] {message}", flush=True)

def print_config():
    """現在の設定値を表示"""
    force_print("Current Configurations:")
    for attr in dir(config):
        if not attr.startswith("__"):
            value = getattr(config, attr)
            print(f"  {attr}: {value}")

def print_sample_structure(sample, sample_idx=0):
    """
    1サンプルのデータ構造をCSV形式で表示して、モデルの入力形式を確認する
    
    Args:
        sample: (x, y_targets, original_len, raw_path, strain, strength_score) のタプル
        sample_idx: 表示用のサンプル番号
    """
    x, y_targets, original_len, raw_path, strain, strength_score = sample
    
    print("\n" + "=" * 60)
    print(f"【サンプル {sample_idx} のデータ構造 (CSV形式)】")
    print("=" * 60)
    
    # メタ情報
    print(f"\n# メタ情報")
    print(f"strain,{strain}")
    print(f"strength_score,{strength_score:.3f}")
    print(f"original_len,{original_len}")
    print(f"x_len,{len(x)}")
    print(f"raw_path,{raw_path}")
    
    # ターゲット
    print(f"\n# ターゲット (Y)")
    print("target_idx,region_id,position_id,aa_pos_id,codon_pos_id,is_synonymous")
    for i, target in enumerate(y_targets):
        region_id, position_id, aa_pos_id, codon_pos_id, is_synonymous = target
        print(f"{i},{region_id},{position_id},{aa_pos_id},{codon_pos_id},{is_synonymous}")
    
    # 入力特徴量 (CSV形式)
    print(f"\n# 入力特徴量 (X)")
    print("# cat: bef=塩基前, pos=塩基位置, aft=塩基後, c_pos=コドン位置, aa_b=アミノ酸前, p_pos=アミノ酸配列位置, aa_a=アミノ酸後, region=領域")
    print("# num: hydro=疎水性変化, size=サイズ変化, charge=電荷変化, dissim=類似度, pam=PAM250, freq=頻度")
    
    # ヘッダ
    print("timestep,co_occur,bef,pos,aft,c_pos,aa_b,p_pos,aa_a,region,freq,hydro,charge,size,dissim,pam")
    
    # 全タイムステップを出力
    for t_idx, timestep in enumerate(x):
        for c_idx, event in enumerate(timestep):
            cat_features, num_features = event
            
            # 全て0の場合はスキップ（パディング）
            if all(f == 0 for f in cat_features):
                continue
            
            cat_str = ",".join(str(f) for f in cat_features)
            num_str = ",".join(f"{f:.3f}" for f in num_features)
            print(f"{t_idx},{c_idx},{cat_str},{num_str}")
    
    print("=" * 60 + "\n")

# ==========================================
# 2. キャッシュ・保存関連 (旧 cache_utils 含む)
# ==========================================

def get_config_hash():
    """config設定からハッシュを生成（設定変更時にキャッシュを無効化するため）"""
    try:
        sampling_mode = getattr(config, 'SAMPLING_MODE', 'proportional')
        
        relevant_configs = [
            str(config.MAX_SEQ_LEN),        # 入力系列長
            str(config.TRAIN_MAX),          # 学習データの期間
            str(config.VALID_NUM),          # 評価期間
            str(config.DATA_BASE_DIR),      # データパス
            str(config.TARGET_LEN),         # 予測ターゲット長
            str(config.MAX_CO_OCCURRENCE),  # 共起上限
            str(config.VALID_RATIO),        # 分割比率
            str(config.SEED),               # 乱数シード (分割に影響)
            str(config.NUM_FEATURE_STRING), # 特徴量数
            str(config.NUM_CHEM_FEATURES),  # 特徴量数
            str(config.ABLATION_MASKS),     # マスク設定 (データの中身が変わるため)
            str(sampling_mode),             # サンプリングモード
        ]
        
        # サンプリングモードに応じたパラメータを追加
        if sampling_mode == 'proportional':
            relevant_configs.append(str(config.MAX_NUM))  # 比率サンプリング用
        elif sampling_mode == 'fixed_per_strain':
            relevant_configs.append(str(config.MAX_STRAIN_NUM))     # 株数上限
            relevant_configs.append(str(config.MAX_NUM_PER_STRAIN)) # 株ごとのサンプル上限
        
        config_string = "_".join(relevant_configs)
        return hashlib.md5(config_string.encode('utf-8')).hexdigest()
    except Exception as e:
        print(f"[WARNING] config.py からハッシュを生成できませんでした: {e}")
        return "default_cache"

def load_cache(cache_path):
    """メインキャッシュのロード"""
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
    """メインキャッシュの保存"""
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        force_print(f"[INFO] Saving data to cache: {cache_path}")
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        force_print(f"[WARNING] Failed to save cache: {e}")

def get_batch_cache_path(batch_idx, total_batches, config_hash):
    """インクリメンタルキャッシュ用のパス生成"""
    filename = f"feat_batch_{config_hash}_{batch_idx}_of_{total_batches}.pkl"
    return os.path.join(config.INCREMENTAL_CACHE_DIR, filename)

def load_batch_cache(path):
    """バッチキャッシュのロード"""
    if os.path.exists(path):
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except Exception:
            return None
    return None

def save_batch_cache(data, path):
    """バッチキャッシュの保存"""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        force_print(f"[WARNING] Failed to save batch cache: {e}")

# ==========================================
# 3. モデル評価関連
# ==========================================

def calculate_topk_hit_rate(pred_sets, target_sets):
    """
    議論したカスタム評価指標 (Top-Nヒット率)
    予測セット(Top-K)と正解セット(共起)の積集合が空でなければヒット
    """
    total_samples = len(pred_sets)
    if total_samples == 0:
        return 0.0

    hits = 0
    for pred_s, target_s in zip(pred_sets, target_sets):
        if not target_s: # ターゲットがない場合はスキップ
            total_samples -= 1
            continue
            
        # 積集合(intersection)が空でない(> 0)かチェック
        if len(pred_s.intersection(target_s)) > 0:
            hits += 1
            
    return (hits / total_samples) * 100 if total_samples > 0 else 0.0

def calculate_metrics(pred_sets, target_sets):
    """
    ★拡張: Recall(Hit Rate), Precision, F1スコアを計算
    """
    total_samples = len(pred_sets)
    if total_samples == 0:
        return 0.0, 0.0, 0.0

    hits = 0
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    valid_samples = 0

    for pred_s, target_s in zip(pred_sets, target_sets):
        if not target_s: 
            continue
        
        valid_samples += 1
        
        # Intersection (TP)
        tp = len(pred_s.intersection(target_s))
        
        # Hit Rate (Recall@K > 0)
        if tp > 0:
            hits += 1
            
        # Precision: TP / Predicted
        precision = tp / len(pred_s) if len(pred_s) > 0 else 0
        
        # Recall: TP / Actual
        recall = tp / len(target_s) if len(target_s) > 0 else 0
        
        # F1
        if (precision + recall) > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0
            
        total_precision += precision
        total_recall += recall
        total_f1 += f1

    if valid_samples == 0:
        return 0.0, 0.0, 0.0

    avg_hit_rate = (hits / valid_samples) * 100
    avg_precision = (total_precision / valid_samples) * 100
    avg_recall = (total_recall / valid_samples) * 100 # 本来のRecall (網羅率)
    avg_f1 = (total_f1 / valid_samples) * 100
    
    return avg_hit_rate, avg_precision, avg_recall, avg_f1

# --- ★追加: 結果保存用関数 ---
def save_training_log(log_data, output_dir):
    """学習経過(csv)を保存"""
    df = pd.DataFrame(log_data)
    path = os.path.join(output_dir, 'training_log.csv')
    df.to_csv(path, index=False)
    force_print(f"[INFO] Training log saved to {path}")

def save_prediction_results(results, output_dir, prefix="test"):
    """
    予測結果の詳細を保存 (csv)
    columns: [original_len, strain, strength_score, targets, predictions, hit_*]
    """
    # 辞書リストに変換して保存
    rows = []
    for r in results:
        row = {
            'original_len': r['len'],
            'raw_path': r['raw_path'], # 生のパス文字列
            'strain': r['strain'],     # 株名
            'strength_score': r.get('strength_score', 0.0),  # 流行度 (正解)
            'pred_strength': r.get('pred_strength', 0.0),    # 流行度 (予測)
            'target_region': str(list(r['targets_region'])),
            'pred_region': str(list(r['preds_region'])),
            'hit_region': r['hit_region'],
            'target_position': str(list(r['targets_position'])),
            'pred_position': str(list(r['preds_position'])),
            'hit_position': r['hit_position'],
        }
        # AA Pos がある場合のみ追加
        if 'targets_aa_pos' in r:
            row['target_aa_pos'] = str(list(r['targets_aa_pos']))
            row['pred_aa_pos'] = str(list(r['preds_aa_pos']))
            row['hit_aa_pos'] = r['hit_aa_pos']
        
        # Codon Pos がある場合のみ追加
        if 'targets_codon_pos' in r:
            row['target_codon_pos'] = str(list(r['targets_codon_pos']))
            row['pred_codon_pos'] = str(list(r['preds_codon_pos']))
            row['hit_codon_pos'] = r['hit_codon_pos']
        
        # Synonymous がある場合のみ追加
        if 'targets_synonymous' in r:
            row['target_synonymous'] = str(list(r['targets_synonymous']))
            row['pred_synonymous'] = str(list(r['preds_synonymous']))
            row['hit_synonymous'] = r['hit_synonymous']
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    path = os.path.join(output_dir, f'{prefix}_predictions.csv')
    df.to_csv(path, index=False)
    force_print(f"[INFO] Prediction results saved to {path}")

def save_strain_info(strains, output_dir, prefix="train"):
    """使用した株のリストを保存"""
    path = os.path.join(output_dir, f'{prefix}_strains.txt')
    with open(path, 'w') as f:
        for s in sorted(list(set(strains))):
            f.write(f"{s}\n")
    force_print(f"[INFO] Strain info saved to {path}")


def save_config_copy(output_dir):
    """
    config.pyのコピーを保存（再現性のため）
    """
    import shutil
    config_src = os.path.join(os.path.dirname(__file__), 'config.py')
    config_dst = os.path.join(output_dir, 'config_snapshot.py')
    
    try:
        shutil.copy2(config_src, config_dst)
        force_print(f"[INFO] Config snapshot saved to {config_dst}")
    except Exception as e:
        force_print(f"[WARNING] Failed to save config snapshot: {e}")


def save_synonymous_distribution_csv(train_data, valid_data, test_data, output_dir):
    """
    シノニマス/ノンシノニマスの分布をCSVに保存
    
    Args:
        train_data, valid_data, test_data: list of (x, y_targets, ...)
    """
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
    path = os.path.join(output_dir, 'synonymous_distribution.csv')
    df.to_csv(path, index=False)
    force_print(f"[INFO] Synonymous distribution saved to {path}")


def plot_training_curve(training_log, output_dir):
    """
    学習曲線（train_loss / val_loss）をプロット
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        if not training_log:
            return
        
        epochs = [log['epoch'] for log in training_log]
        train_losses = [log['train_loss'] for log in training_log]
        val_losses = [log['val_loss'] for log in training_log]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(epochs, train_losses, 'b-o', label='Train Loss', linewidth=2, markersize=4)
        ax.plot(epochs, val_losses, 'r-s', label='Validation Loss', linewidth=2, markersize=4)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Training Curve', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # 最小値をマーク
        min_val_idx = val_losses.index(min(val_losses))
        ax.axvline(x=epochs[min_val_idx], color='green', linestyle='--', alpha=0.5, label=f'Best epoch: {epochs[min_val_idx]}')
        ax.scatter([epochs[min_val_idx]], [val_losses[min_val_idx]], color='green', s=100, zorder=5)
        
        plt.tight_layout()
        path = os.path.join(output_dir, 'training_curve.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        force_print(f"[INFO] Training curve saved to {path}")
        
    except ImportError:
        force_print("[WARNING] matplotlib not available, skipping training curve plot")


def save_confusion_matrix_csv(details, output_dir, prefix="test"):
    """
    タンパク質領域の混同行列をCSVに保存
    """
    df = pd.DataFrame(details) if details else pd.DataFrame()
    if len(df) == 0:
        return
    
    region_names = {v: k for k, v in config.PROTEIN_VOCABS.items()}
    
    # 混同行列を集計（どの真値がどの予測に変換されたか）
    confusion = {}  # {(true_id, pred_id): count}
    
    for _, row in df.iterrows():
        targets = list(row['targets_region'])
        preds = list(row['preds_region'])
        
        # 真値と予測値のペアをカウント
        for t in targets:
            for p in preds:
                key = (t, p)
                confusion[key] = confusion.get(key, 0) + 1
    
    # CSVに変換
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
    path = os.path.join(output_dir, f'{prefix}_confusion_matrix.csv')
    result_df.to_csv(path, index=False)
    force_print(f"[INFO] Confusion matrix saved to {path}")


# ==========================================
# 4. メトリクス保存・可視化関連
# ==========================================

def save_metrics_csv(metrics_by_ts, output_dir, prefix="val"):
    """
    タイムステップ別メトリクスをCSVに保存
    """
    rows = []
    for ts_len in sorted(metrics_by_ts.keys()):
        m = metrics_by_ts[ts_len]
        row = {'timestep': ts_len}
        row.update(m)
        rows.append(row)
    
    df = pd.DataFrame(rows)
    path = os.path.join(output_dir, f'{prefix}_metrics_by_timestep.csv')
    df.to_csv(path, index=False)
    force_print(f"[INFO] Metrics saved to {path}")
    return df

def save_category_metrics_csv(cat_metrics, output_dir, prefix="val"):
    """
    流行度カテゴリ別メトリクスをCSVに保存
    """
    rows = []
    for ts_len in sorted(cat_metrics.keys()):
        for category in ['low', 'medium', 'high']:
            m = cat_metrics[ts_len][category]
            row = {
                'timestep': ts_len,
                'category': category,
            }
            row.update(m)
            rows.append(row)
    
    df = pd.DataFrame(rows)
    path = os.path.join(output_dir, f'{prefix}_metrics_by_category.csv')
    df.to_csv(path, index=False)
    force_print(f"[INFO] Category metrics saved to {path}")
    return df


def save_random_baseline_csv(details, output_dir, prefix="test"):
    """
    ランダムベースラインとの比較をCSVに保存
    """
    df = pd.DataFrame(details) if details else pd.DataFrame()
    if len(df) == 0:
        return
    
    # ランダム予測の期待値（1/クラス数）
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
    path = os.path.join(output_dir, f'{prefix}_random_baseline.csv')
    result_df.to_csv(path, index=False)
    force_print(f"[INFO] Random baseline comparison saved to {path}")


def save_region_metrics_csv(details, output_dir, prefix="test"):
    """
    タンパク質領域別メトリクスをCSVに保存
    """
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
    path = os.path.join(output_dir, f'{prefix}_region_metrics.csv')
    result_df.to_csv(path, index=False)
    force_print(f"[INFO] Region metrics saved to {path}")


def save_prediction_distribution_csv(details, output_dir, prefix="test"):
    """
    予測分布（どの領域に何回予測したか）をCSVに保存
    """
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
    path = os.path.join(output_dir, f'{prefix}_prediction_distribution.csv')
    result_df.to_csv(path, index=False)
    force_print(f"[INFO] Prediction distribution saved to {path}")

def plot_metrics_by_timestep(metrics_by_ts, output_dir, prefix="val"):
    """
    タイムステップ別のメトリクスをグラフ化 (Hit Rate, Precision, Recall, F1, MAE)
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # GUIなし環境対応
        import matplotlib.pyplot as plt
        
        timesteps = sorted(metrics_by_ts.keys())
        
        fig, axes = plt.subplots(3, 2, figsize=(14, 15))
        fig.suptitle(f'{prefix.upper()} - Metrics by Timestep', fontsize=14)
        
        colors = {'region': '#2ecc71', 'position': '#3498db', 'aa_pos': '#9b59b6', 'codon': '#e67e22', 'synonymous': '#e74c3c'}
        
        # 1. Hit Rate比較
        ax = axes[0, 0]
        ax.plot(timesteps, [metrics_by_ts[t]['region_hit_rate'] for t in timesteps], 'o-', label='Region', color=colors['region'])
        ax.plot(timesteps, [metrics_by_ts[t]['position_hit_rate'] for t in timesteps], 's-', label='Base Pos', color=colors['position'])
        ax.plot(timesteps, [metrics_by_ts[t]['aa_pos_hit_rate'] for t in timesteps], '^-', label='AA Pos', color=colors['aa_pos'])
        ax.plot(timesteps, [metrics_by_ts[t].get('codon_pos_hit_rate', 0) for t in timesteps], 'D-', label='Codon Pos', color=colors['codon'])
        ax.plot(timesteps, [metrics_by_ts[t].get('synonymous_hit_rate', 0) for t in timesteps], 'x-', label='Synonymous', color=colors['synonymous'])
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Hit Rate (%)')
        ax.set_title('Hit Rate')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Precision (適合率)
        ax = axes[0, 1]
        ax.plot(timesteps, [metrics_by_ts[t]['region_precision'] for t in timesteps], 'o-', label='Region', color=colors['region'])
        ax.plot(timesteps, [metrics_by_ts[t]['position_precision'] for t in timesteps], 's-', label='Base Pos', color=colors['position'])
        ax.plot(timesteps, [metrics_by_ts[t]['aa_pos_precision'] for t in timesteps], '^-', label='AA Pos', color=colors['aa_pos'])
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Precision (%)')
        ax.set_title('Precision')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Recall (再現率)
        ax = axes[1, 0]
        ax.plot(timesteps, [metrics_by_ts[t]['region_recall'] for t in timesteps], 'o-', label='Region', color=colors['region'])
        ax.plot(timesteps, [metrics_by_ts[t]['position_recall'] for t in timesteps], 's-', label='Base Pos', color=colors['position'])
        ax.plot(timesteps, [metrics_by_ts[t]['aa_pos_recall'] for t in timesteps], '^-', label='AA Pos', color=colors['aa_pos'])
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Recall (%)')
        ax.set_title('Recall')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. F1 Score
        ax = axes[1, 1]
        ax.plot(timesteps, [metrics_by_ts[t]['region_f1'] for t in timesteps], 'o-', label='Region', color=colors['region'])
        ax.plot(timesteps, [metrics_by_ts[t]['position_f1'] for t in timesteps], 's-', label='Base Pos', color=colors['position'])
        ax.plot(timesteps, [metrics_by_ts[t]['aa_pos_f1'] for t in timesteps], '^-', label='AA Pos', color=colors['aa_pos'])
        ax.set_xlabel('Timestep')
        ax.set_ylabel('F1 Score (%)')
        ax.set_title('F1 Score')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 5. Strength MAE
        ax = axes[2, 0]
        ax.plot(timesteps, [metrics_by_ts[t]['strength_mae'] for t in timesteps], 'D-', color='#e74c3c', linewidth=2)
        ax.set_xlabel('Timestep')
        ax.set_ylabel('MAE')
        ax.set_title('Strength Prediction MAE')
        ax.grid(True, alpha=0.3)
        
        # 6. サンプル数
        ax = axes[2, 1]
        ax.bar(timesteps, [metrics_by_ts[t]['num_samples'] for t in timesteps], color='#34495e', alpha=0.7)
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Number of Samples')
        ax.set_title('Sample Count')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        path = os.path.join(output_dir, f'{prefix}_metrics_plot.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        force_print(f"[INFO] Metrics plot saved to {path}")
        
    except ImportError:
        force_print("[WARNING] matplotlib not available, skipping plots")

def plot_category_metrics(cat_metrics, output_dir, prefix="val"):
    """
    流行度カテゴリ別のメトリクスをグラフ化
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        
        timesteps = sorted(cat_metrics.keys())
        categories = ['low', 'medium', 'high']
        colors = {'low': '#e74c3c', 'medium': '#f39c12', 'high': '#27ae60'}
        
        fig, axes = plt.subplots(3, 2, figsize=(14, 15))
        fig.suptitle(f'{prefix.upper()} - Metrics by Strength Category', fontsize=14)
        
        # 1. Region Hit Rate by Category
        ax = axes[0, 0]
        for cat in categories:
            values = [cat_metrics[t][cat]['region_hit_rate'] for t in timesteps]
            ax.plot(timesteps, values, 'o-', label=cat.capitalize(), color=colors[cat])
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Hit Rate (%)')
        ax.set_title('Region Hit Rate by Category')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Base Pos Hit Rate by Category
        ax = axes[0, 1]
        for cat in categories:
            values = [cat_metrics[t][cat]['position_hit_rate'] for t in timesteps]
            ax.plot(timesteps, values, 's-', label=cat.capitalize(), color=colors[cat])
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Hit Rate (%)')
        ax.set_title('Base Pos Hit Rate by Category')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. AA Pos Hit Rate by Category
        ax = axes[1, 0]
        for cat in categories:
            values = [cat_metrics[t][cat]['aa_pos_hit_rate'] for t in timesteps]
            ax.plot(timesteps, values, '^-', label=cat.capitalize(), color=colors[cat])
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Hit Rate (%)')
        ax.set_title('AA Pos Hit Rate by Category')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Strength MAE by Category
        ax = axes[1, 1]
        for cat in categories:
            values = [cat_metrics[t][cat]['strength_mae'] for t in timesteps]
            ax.plot(timesteps, values, 'D-', label=cat.capitalize(), color=colors[cat])
        ax.set_xlabel('Timestep')
        ax.set_ylabel('MAE')
        ax.set_title('Strength MAE by Category')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 5. Sample Distribution (Stacked Bar)
        ax = axes[2, 0]
        width = 0.6
        bottom = np.zeros(len(timesteps))
        for cat in categories:
            values = [cat_metrics[t][cat]['num_samples'] for t in timesteps]
            ax.bar(timesteps, values, width, label=cat.capitalize(), bottom=bottom, color=colors[cat], alpha=0.8)
            bottom += np.array(values)
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Number of Samples')
        ax.set_title('Sample Distribution by Category')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # 6. Codon Pos Hit Rate by Category
        ax = axes[2, 1]
        for cat in categories:
            values = [cat_metrics[t][cat].get('codon_pos_hit_rate', 0) for t in timesteps]
            ax.plot(timesteps, values, 'x-', label=cat.capitalize(), color=colors[cat])
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Hit Rate (%)')
        ax.set_title('Codon Pos Hit Rate by Category')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        path = os.path.join(output_dir, f'{prefix}_category_plot.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        force_print(f"[INFO] Category plot saved to {path}")
        
    except ImportError:
        force_print("[WARNING] matplotlib not available, skipping plots")


def plot_combined_val_test_metrics(val_metrics, test_metrics, output_dir):
    """
    ValidationとTestのメトリクスを1つのグラフで比較
    Val=実線、Test=点線で区別
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        if not val_metrics and not test_metrics:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Validation vs Test - Metrics Comparison', fontsize=14)
        
        colors = {
            'region': '#2ecc71',      # 緑
            'position': '#3498db',    # 青
            'aa_pos': '#9b59b6',      # 紫
            'codon': '#e67e22',       # オレンジ
            'synonymous': '#e74c3c'   # 赤
        }
        
        metric_configs = [
            ('region_hit_rate', 'Region Hit Rate (%)', (0, 0)),
            ('position_hit_rate', 'Base Position Hit Rate (%)', (0, 1)),
            ('aa_pos_hit_rate', 'AA Position Hit Rate (%)', (0, 2)),
            ('codon_pos_hit_rate', 'Codon Position Hit Rate (%)', (1, 0)),
            ('synonymous_hit_rate', 'Synonymous Hit Rate (%)', (1, 1)),
            ('strength_mae', 'Strength MAE', (1, 2)),
        ]
        
        val_timesteps = sorted(val_metrics.keys()) if val_metrics else []
        test_timesteps = sorted(test_metrics.keys()) if test_metrics else []
        
        for metric_key, title, (row, col) in metric_configs:
            ax = axes[row, col]
            
            # Validation (実線、丸マーカー)
            if val_metrics and val_timesteps:
                val_values = [val_metrics[t].get(metric_key, 0) for t in val_timesteps]
                ax.plot(val_timesteps, val_values, 'o-', 
                        label='Validation', color='#3498db', linewidth=2, markersize=6)
            
            # Test (点線、三角マーカー)
            if test_metrics and test_timesteps:
                test_values = [test_metrics[t].get(metric_key, 0) for t in test_timesteps]
                ax.plot(test_timesteps, test_values, '^--', 
                        label='Test', color='#e74c3c', linewidth=2, markersize=6)
            
            ax.set_xlabel('Timestep (Path Length)')
            ax.set_ylabel(title)
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        path = os.path.join(output_dir, 'combined_val_test_metrics.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        force_print(f"[INFO] Combined Val/Test metrics plot saved to {path}")
        
    except ImportError:
        force_print("[WARNING] matplotlib not available, skipping combined plot")


def plot_combined_category_comparison(val_details, test_details, strength_thresholds, output_dir):
    """
    流行度カテゴリ別のVal/Test比較グラフ
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        
        if not val_details and not test_details:
            return
        
        low_max, med_max = strength_thresholds
        
        def get_strength_category(score):
            if score < low_max:
                return 'low'
            elif score < med_max:
                return 'medium'
            else:
                return 'high'
        
        def calc_category_metrics(details):
            category_stats = {'low': [], 'medium': [], 'high': []}
            for d in details:
                cat = get_strength_category(d['strength_score'])
                category_stats[cat].append({
                    'region': d['hit_region'],
                    'position': d['hit_position'],
                    'aa_pos': d['hit_aa_pos'],
                    'codon': d.get('hit_codon_pos', 0),
                    'synonymous': d.get('hit_synonymous', 0)
                })
            
            result = {}
            for cat in ['low', 'medium', 'high']:
                if category_stats[cat]:
                    result[cat] = {
                        'count': len(category_stats[cat]),
                        'region': np.mean([x['region'] for x in category_stats[cat]]) * 100,
                        'position': np.mean([x['position'] for x in category_stats[cat]]) * 100,
                        'synonymous': np.mean([x['synonymous'] for x in category_stats[cat]]) * 100,
                    }
                else:
                    result[cat] = {'count': 0, 'region': 0, 'position': 0, 'synonymous': 0}
            return result
        
        val_cat = calc_category_metrics(val_details) if val_details else None
        test_cat = calc_category_metrics(test_details) if test_details else None
        
        categories = ['low', 'medium', 'high']
        x = np.arange(len(categories))
        width = 0.35
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'Val vs Test by Strength Category (Low<{low_max}, Medium<{med_max}, High≥{med_max})', fontsize=12)
        
        metrics = [
            ('region', 'Region Hit Rate (%)', 0),
            ('position', 'Base Position Hit Rate (%)', 1),
            ('synonymous', 'Synonymous Hit Rate (%)', 2),
        ]
        
        for metric_key, title, idx in metrics:
            ax = axes[idx]
            
            val_values = [val_cat[cat][metric_key] if val_cat else 0 for cat in categories]
            test_values = [test_cat[cat][metric_key] if test_cat else 0 for cat in categories]
            
            bars1 = ax.bar(x - width/2, val_values, width, label='Validation', color='#3498db', alpha=0.8)
            bars2 = ax.bar(x + width/2, test_values, width, label='Test', color='#e74c3c', alpha=0.8)
            
            ax.set_xlabel('Strength Category')
            ax.set_ylabel(title)
            ax.set_title(title)
            ax.set_xticks(x)
            ax.set_xticklabels(categories)
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            
            # 値を表示
            for bar in bars1:
                height = bar.get_height()
                ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
            for bar in bars2:
                height = bar.get_height()
                ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        path = os.path.join(output_dir, 'combined_category_comparison.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        force_print(f"[INFO] Combined category comparison plot saved to {path}")
        
    except ImportError:
        force_print("[WARNING] matplotlib not available, skipping combined category plot")

# ==========================================
# 5. 結果表示・WandB関連
# ==========================================

def init_wandb():
    """WandB初期化"""
    import os
    import types
    from datetime import datetime
    try:
        import wandb
        if config.USE_WANDB:
            # オフラインモード設定
            if getattr(config, 'WANDB_OFFLINE', False):
                os.environ["WANDB_MODE"] = "offline"
                force_print("[INFO] WandB: オフラインモードで起動")
            
            config_dict = {
                k: v for k, v in vars(config).items() 
                if not k.startswith('__') and not isinstance(v, types.ModuleType)
            }
            wandb.init(
                project=config.WANDB_PROJECT_NAME, 
                name=config.WANDB_RUN_NAME if config.WANDB_RUN_NAME else f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config=config_dict
            )
            return wandb
    except ImportError:
        force_print("[WARNING] wandb not available")
    return None


def finish_wandb(wandb_module):
    """
    WandBセッションを終了し、オフラインモードの場合は自動でsyncを実行
    
    Args:
        wandb_module: init_wandb()で返されたwandbモジュール
    """
    if wandb_module is None or not config.USE_WANDB:
        return
    
    try:
        import subprocess
        import os
        
        # 現在のランのディレクトリを取得
        run_dir = None
        if wandb_module.run is not None:
            run_dir = wandb_module.run.dir
        
        # セッションを終了
        wandb_module.finish()
        force_print("[INFO] WandB: セッション終了")
        
        # オフラインモードの場合は自動sync
        if getattr(config, 'WANDB_OFFLINE', False) and run_dir:
            # run_dirは .../wandb/run-xxx/files なので、親ディレクトリを取得
            run_path = os.path.dirname(run_dir)  # wandb/run-xxx
            
            force_print(f"[INFO] WandB: オフラインデータを同期中... ({run_path})")
            
            try:
                result = subprocess.run(
                    ['wandb', 'sync', run_path],
                    capture_output=True,
                    text=True,
                    timeout=300  # 5分タイムアウト
                )
                if result.returncode == 0:
                    force_print("[INFO] WandB: 同期完了")
                else:
                    force_print(f"[WARNING] WandB sync failed: {result.stderr}")
            except subprocess.TimeoutExpired:
                force_print("[WARNING] WandB sync timed out (5 min). Run manually: wandb sync ./wandb/run-*")
            except Exception as e:
                force_print(f"[WARNING] WandB sync error: {e}")
                
    except Exception as e:
        force_print(f"[WARNING] WandB finish error: {e}")

# ==========================================
# 6. 階層的評価レポート (4部構成)
# ==========================================


def print_combined_report(val_details, test_details, strength_thresholds=None):
    """
    ValidationとTestを統合した階層的評価レポート (pandas DataFrame使用)
    
    Args:
        val_details: Validation評価の詳細結果リスト
        test_details: Test評価の詳細結果リスト
        strength_thresholds: (low_max, med_max) の動的閾値タプル。Noneの場合はconfigの値を使用
    """
    import numpy as np
    
    if not val_details and not test_details:
        print("[WARNING] No results to display")
        return
    
    df_val = pd.DataFrame(val_details) if val_details else pd.DataFrame()
    df_test = pd.DataFrame(test_details) if test_details else pd.DataFrame()
    
    print("\n" + "=" * 100)
    print("【統合評価レポート - Validation vs Test】")
    print("=" * 100)
    
    # =================================================================
    # 1. Executive Summary (流行度別)
    # =================================================================
    print("\n" + "-" * 100)
    print("=== 1. Executive Summary (流行度別: High/Medium/Low) ===")
    print("-" * 100)
    
    def calc_summary(df):
        if len(df) == 0:
            return {'サンプル': '-', 'タンパク質領域': '-', '塩基配列位置': '-', 'アミノ酸配列位置': '-', 'コドン位置': '-', 'シノニマス': '-'}
        return {
            'サンプル': len(df),
            'タンパク質領域': f"{df['hit_region'].mean() * 100:.1f}%",
            '塩基配列位置': f"{df['hit_position'].mean() * 100:.1f}%",
            'アミノ酸配列位置': f"{df['hit_aa_pos'].mean() * 100:.1f}%",
            'コドン位置': f"{df['hit_codon_pos'].mean() * 100:.1f}%" if 'hit_codon_pos' in df.columns else '-',
            'シノニマス': f"{df['hit_synonymous'].mean() * 100:.1f}%" if 'hit_synonymous' in df.columns else '-'
        }
    
    def get_strength_category(score, low_max, med_max):
        if score < low_max:
            return 'low'
        elif score < med_max:
            return 'medium'
        else:
            return 'high'
    
    # 動的閾値の設定
    if strength_thresholds is not None:
        low_max, med_max = strength_thresholds
    else:
        low_max = config.STRENGTH_CATEGORY_LOW_MAX
        med_max = config.STRENGTH_CATEGORY_MED_MAX
    
    print(f"\n  閾値: Low(<{low_max}), Medium(<{med_max}), High(≥{med_max})\n")
    
    # カテゴリ毎のサマリーを作成
    summary_rows = []
    for category in ['high', 'medium', 'low']:
        row = {'カテゴリ': category}
        
        # Validation
        if len(df_val) > 0:
            df_val['strength_cat'] = df_val['strength_score'].apply(lambda x: get_strength_category(x, low_max, med_max))
            v_cat = df_val[df_val['strength_cat'] == category]
        else:
            v_cat = pd.DataFrame()
        v_s = calc_summary(v_cat)
        row['Val_サンプル'] = v_s['サンプル']
        row['Val_タンパク質領域'] = v_s['タンパク質領域']
        row['Val_塩基配列位置'] = v_s['塩基配列位置']
        row['Val_アミノ酸配列位置'] = v_s['アミノ酸配列位置']
        row['Val_コドン位置'] = v_s['コドン位置']
        row['Val_シノニマス'] = v_s['シノニマス']
        
        # Test
        if len(df_test) > 0:
            df_test['strength_cat'] = df_test['strength_score'].apply(lambda x: get_strength_category(x, low_max, med_max))
            t_cat = df_test[df_test['strength_cat'] == category]
        else:
            t_cat = pd.DataFrame()
        t_s = calc_summary(t_cat)
        row['Test_サンプル'] = t_s['サンプル']
        row['Test_タンパク質領域'] = t_s['タンパク質領域']
        row['Test_塩基配列位置'] = t_s['塩基配列位置']
        row['Test_アミノ酸配列位置'] = t_s['アミノ酸配列位置']
        row['Test_コドン位置'] = t_s['コドン位置']
        row['Test_シノニマス'] = t_s['シノニマス']
        
        summary_rows.append(row)
    
    df_summary = pd.DataFrame(summary_rows)
    print(tabulate(df_summary, headers='keys', tablefmt='simple', showindex=False))
    
    # =================================================================
    # 2. 予測対象別メトリクスサマリー (適合率・再現率・F1)
    # =================================================================
    print("\n" + "-" * 100)
    print("=== 2. Prediction Metrics Summary (予測対象別: 適合率/再現率/F1) ===")
    print("-" * 100 + "\n")
    
    def calc_label_metrics(df, targets_key, preds_key):
        """ラベル種別ごとに全体の適合率・再現率・F1を計算"""
        if len(df) == 0:
            return {'適合率': '-', '再現率': '-', 'F1': '-', 'サンプル': '-'}
        
        total_tp = 0
        total_fp = 0
        total_fn = 0
        
        for _, row in df.iterrows():
            targets = set(row[targets_key])
            preds = set(row[preds_key])
            
            tp = len(targets & preds)
            fp = len(preds - targets)
            fn = len(targets - preds)
            
            total_tp += tp
            total_fp += fp
            total_fn += fn
        
        precision = total_tp / (total_tp + total_fp) * 100 if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) * 100 if (total_tp + total_fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'サンプル': len(df),
            '適合率': f"{precision:.1f}%",
            '再現率': f"{recall:.1f}%",
            'F1': f"{f1:.1f}%"
        }
    
    label_types = [
        ('タンパク質領域', 'targets_region', 'preds_region'),
        ('塩基配列位置', 'targets_position', 'preds_position'),
        ('アミノ酸配列位置', 'targets_aa_pos', 'preds_aa_pos'),
        ('コドン位置', 'targets_codon_pos', 'preds_codon_pos'),
        ('シノニマス', 'targets_synonymous', 'preds_synonymous'),
    ]
    
    metrics_rows = []
    for label_name, targets_key, preds_key in label_types:
        row = {'予測対象': label_name}
        
        # Validation
        v_m = calc_label_metrics(df_val, targets_key, preds_key)
        row['Val_サンプル'] = v_m['サンプル']
        row['Val_適合率'] = v_m['適合率']
        row['Val_再現率'] = v_m['再現率']
        row['Val_F1'] = v_m['F1']
        
        # Test
        t_m = calc_label_metrics(df_test, targets_key, preds_key)
        row['Test_サンプル'] = t_m['サンプル']
        row['Test_適合率'] = t_m['適合率']
        row['Test_再現率'] = t_m['再現率']
        row['Test_F1'] = t_m['F1']
        
        metrics_rows.append(row)
    
    df_metrics = pd.DataFrame(metrics_rows)
    print(tabulate(df_metrics, headers='keys', tablefmt='simple', showindex=False))
    
    # --- 流行度予測 MAE ---
    def calc_strength_mae(df):
        if len(df) == 0 or 'strength_score' not in df.columns or 'pred_strength' not in df.columns:
            return '-'
        mae = (df['strength_score'] - df['pred_strength']).abs().mean()
        return f"{mae:.3f}"
    
    print(f"\n  流行度予測 MAE: Val={calc_strength_mae(df_val)}, Test={calc_strength_mae(df_test)}")
    
    # --- ランダムベースラインとの比較 ---
    print("\n" + "-" * 100)
    print("=== 2.5. Random Baseline Comparison (ランダム予測との比較) ===")
    print("-" * 100 + "\n")
    
    # ランダム予測の期待値（1/クラス数）
    random_baselines = {
        'タンパク質領域': 1 / config.NUM_REGIONS * 100,
        '塩基配列位置': 1 / config.VOCAB_SIZE_POSITION * 100,
        'アミノ酸配列位置': 1 / config.VOCAB_SIZE_AA_POS * 100,
        'コドン位置': 1 / 3 * 100,  # 1, 2, 3 の3クラス
        'シノニマス': 50.0,  # 2クラス
    }
    
    comparison_rows = []
    for label_name, targets_key, preds_key in label_types:
        random_rate = random_baselines.get(label_name, 0)
        
        # Validation の Hit Rate
        if len(df_val) > 0 and f'hit_{targets_key.replace("targets_", "")}' in df_val.columns:
            hit_col = f'hit_{targets_key.replace("targets_", "")}'
            val_hit = df_val[hit_col].mean() * 100
        else:
            val_hit = 0
        
        # Test の Hit Rate
        if len(df_test) > 0 and f'hit_{targets_key.replace("targets_", "")}' in df_test.columns:
            hit_col = f'hit_{targets_key.replace("targets_", "")}'
            test_hit = df_test[hit_col].mean() * 100
        else:
            test_hit = 0
        
        # 向上率 (モデル / ランダム)
        val_improvement = val_hit / random_rate if random_rate > 0 else 0
        test_improvement = test_hit / random_rate if random_rate > 0 else 0
        
        comparison_rows.append({
            '予測対象': label_name,
            'ランダム': f"{random_rate:.4f}%",
            'Val_HitRate': f"{val_hit:.1f}%",
            'Val_向上率': f"{val_improvement:.1f}x",
            'Test_HitRate': f"{test_hit:.1f}%",
            'Test_向上率': f"{test_improvement:.1f}x",
        })
    
    df_comparison = pd.DataFrame(comparison_rows)
    print(tabulate(df_comparison, headers='keys', tablefmt='simple', showindex=False))
    
    # =================================================================
    # 3. Biological Analysis (領域別分析)
    # =================================================================
    print("\n" + "-" * 100)
    print("=== 3. Biological Analysis (タンパク質領域別) ===")
    print("-" * 100 + "\n")
    
    def calc_region_metrics(df):
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
        return region_stats
    
    val_regions = calc_region_metrics(df_val) if len(df_val) > 0 else {}
    test_regions = calc_region_metrics(df_test) if len(df_test) > 0 else {}
    all_regions = set(val_regions.keys()) | set(test_regions.keys())
    region_names = {v: k for k, v in config.PROTEIN_VOCABS.items()}
    
    def total_count(reg_id):
        return val_regions.get(reg_id, {}).get('count', 0) + test_regions.get(reg_id, {}).get('count', 0)
    
    sorted_regs = sorted(all_regions, key=total_count, reverse=True)[:10]
    
    region_rows = []
    for reg_id in sorted_regs:
        name = region_names.get(reg_id, f"ID:{reg_id}")[:10]
        row = {'領域': name}
        
        # Validation
        vs = val_regions.get(reg_id, {'tp': 0, 'fp': 0, 'fn': 0, 'count': 0})
        v_prec = vs['tp'] / (vs['tp'] + vs['fp']) * 100 if (vs['tp'] + vs['fp']) > 0 else 0
        v_rec = vs['tp'] / (vs['tp'] + vs['fn']) * 100 if (vs['tp'] + vs['fn']) > 0 else 0
        v_f1 = 2 * v_prec * v_rec / (v_prec + v_rec) if (v_prec + v_rec) > 0 else 0
        row['Val_サンプル'] = vs['count']
        row['Val_適合率'] = f"{v_prec:.1f}%"
        row['Val_再現率'] = f"{v_rec:.1f}%"
        row['Val_F1'] = f"{v_f1:.1f}%"
        
        # Test
        ts = test_regions.get(reg_id, {'tp': 0, 'fp': 0, 'fn': 0, 'count': 0})
        t_prec = ts['tp'] / (ts['tp'] + ts['fp']) * 100 if (ts['tp'] + ts['fp']) > 0 else 0
        t_rec = ts['tp'] / (ts['tp'] + ts['fn']) * 100 if (ts['tp'] + ts['fn']) > 0 else 0
        t_f1 = 2 * t_prec * t_rec / (t_prec + t_rec) if (t_prec + t_rec) > 0 else 0
        row['Test_サンプル'] = ts['count']
        row['Test_適合率'] = f"{t_prec:.1f}%"
        row['Test_再現率'] = f"{t_rec:.1f}%"
        row['Test_F1'] = f"{t_f1:.1f}%"
        
        region_rows.append(row)
    
    df_regions = pd.DataFrame(region_rows)
    print(tabulate(df_regions, headers='keys', tablefmt='simple', showindex=False))
    
    # --- 予測分布 (どの領域に何回予測したか) ---
    print("\n  [予測先の分布 (Top 5)]")
    
    def count_predictions(df, region_names):
        pred_counts = {}
        for _, row in df.iterrows():
            for pred_id in row['preds_region']:
                name = region_names.get(pred_id, f"ID:{pred_id}")
                pred_counts[name] = pred_counts.get(name, 0) + 1
        # ソートしてTop 5を返す
        sorted_counts = sorted(pred_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        return sorted_counts
    
    val_pred_dist = count_predictions(df_val, region_names) if len(df_val) > 0 else []
    test_pred_dist = count_predictions(df_test, region_names) if len(df_test) > 0 else []
    
    val_dist_str = ", ".join([f"{name}:{cnt}" for name, cnt in val_pred_dist]) if val_pred_dist else "-"
    test_dist_str = ", ".join([f"{name}:{cnt}" for name, cnt in test_pred_dist]) if test_pred_dist else "-"
    
    print(f"  Val:  {val_dist_str}")
    print(f"  Test: {test_dist_str}")
    
    # =================================================================
    # 4. Temporal Dynamics (パス長別)
    # =================================================================
    print("\n" + "-" * 100)
    print("=== 4. Temporal Dynamics (パス長別) ===")
    print("-" * 100)
    
    # ヘルパー関数: 各指標をタイムステップ別に計算
    def calc_metrics_by_length(df, targets_col, preds_col):
        """DataFrameから適合率・再現率・F1を計算"""
        if len(df) == 0:
            return 0, 0, 0
        
        total_tp, total_fp, total_fn = 0, 0, 0
        for _, row in df.iterrows():
            targets = set(row[targets_col])
            preds = set(row[preds_col])
            tp = len(targets & preds)
            fp = len(preds - targets)
            fn = len(targets - preds)
            total_tp += tp
            total_fp += fp
            total_fn += fn
        
        precision = total_tp / (total_tp + total_fp) * 100 if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) * 100 if (total_tp + total_fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        return precision, recall, f1
    
    # 予測対象の定義
    metric_targets = [
        ('タンパク質領域', 'targets_region', 'preds_region', 'hit_region'),
        ('塩基配列位置', 'targets_position', 'preds_position', 'hit_position'),
        ('アミノ酸配列位置', 'targets_aa_pos', 'preds_aa_pos', 'hit_aa_pos'),
        ('コドン位置', 'targets_codon_pos', 'preds_codon_pos', 'hit_codon_pos'),
        ('シノニマス', 'targets_synonymous', 'preds_synonymous', 'hit_synonymous'),
    ]
    
    # 全データを結合してループ用リストを作成
    all_dfs = []
    if len(df_val) > 0:
        for length in sorted(df_val['len'].unique()):
            all_dfs.append(('Val', length, df_val[df_val['len'] == length]))
    if len(df_test) > 0:
        for length in sorted(df_test['len'].unique()):
            all_dfs.append(('Test', length, df_test[df_test['len'] == length]))
    
    # --- 4.1 ヒット率 ---
    print("\n  [4.1 Hit Rate (ヒット率)]")
    
    hit_rows = []
    for kind, length, sub_df in all_dfs:
        row = {'種別': kind, 'パス長': length, 'サンプル': len(sub_df)}
        for label_name, targets_col, preds_col, hit_col in metric_targets:
            if hit_col in sub_df.columns:
                row[label_name] = f"{sub_df[hit_col].mean() * 100:.1f}%"
            else:
                row[label_name] = '-'
        hit_rows.append(row)
    
    print(tabulate(pd.DataFrame(hit_rows), headers='keys', tablefmt='simple', showindex=False))
    
    # --- 4.2 適合率 ---
    print("\n  [4.2 Precision (適合率)]")
    
    prec_rows = []
    for kind, length, sub_df in all_dfs:
        row = {'種別': kind, 'パス長': length, 'サンプル': len(sub_df)}
        for label_name, targets_col, preds_col, hit_col in metric_targets:
            if targets_col in sub_df.columns and preds_col in sub_df.columns:
                prec, _, _ = calc_metrics_by_length(sub_df, targets_col, preds_col)
                row[label_name] = f"{prec:.1f}%"
            else:
                row[label_name] = '-'
        prec_rows.append(row)
    
    print(tabulate(pd.DataFrame(prec_rows), headers='keys', tablefmt='simple', showindex=False))
    
    # --- 4.3 再現率 ---
    print("\n  [4.3 Recall (再現率)]")
    
    rec_rows = []
    for kind, length, sub_df in all_dfs:
        row = {'種別': kind, 'パス長': length, 'サンプル': len(sub_df)}
        for label_name, targets_col, preds_col, hit_col in metric_targets:
            if targets_col in sub_df.columns and preds_col in sub_df.columns:
                _, rec, _ = calc_metrics_by_length(sub_df, targets_col, preds_col)
                row[label_name] = f"{rec:.1f}%"
            else:
                row[label_name] = '-'
        rec_rows.append(row)
    
    print(tabulate(pd.DataFrame(rec_rows), headers='keys', tablefmt='simple', showindex=False))
    
    # --- 4.4 F1値 ---
    print("\n  [4.4 F1 Score (F1値)]")
    
    f1_rows = []
    for kind, length, sub_df in all_dfs:
        row = {'種別': kind, 'パス長': length, 'サンプル': len(sub_df)}
        for label_name, targets_col, preds_col, hit_col in metric_targets:
            if targets_col in sub_df.columns and preds_col in sub_df.columns:
                _, _, f1 = calc_metrics_by_length(sub_df, targets_col, preds_col)
                row[label_name] = f"{f1:.1f}%"
            else:
                row[label_name] = '-'
        f1_rows.append(row)
    
    print(tabulate(pd.DataFrame(f1_rows), headers='keys', tablefmt='simple', showindex=False))
    
    print("\n" + "=" * 100 + "\n")