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
from . import config

# ==========================================
# 1. ログ・表示関連
# ==========================================

def force_print(message):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"[{timestamp}] {message}", flush=True)

def print_config():
    """現在の設定値を表示"""
    force_print("Current Configuration:")
    for attr in dir(config):
        if not attr.startswith("__"):
            value = getattr(config, attr)
            force_print(f"  {attr}: {value}")

def print_path_length_distribution(data, name):
    """パス長の分布を表示（デバッグ用）"""
    force_print(f"--- {name} Path Length Distribution ---")
    if hasattr(data, '__len__') and len(data) > 0:
        if isinstance(data, dict) and 'original_len' in data:
            lengths = data['original_len']
        elif hasattr(data[0], 'original_len'):
            lengths = [item.original_len for item in data]
        # 以前の形式(tupleの3番目)の場合
        elif isinstance(data[0], tuple) and len(data[0]) >= 3:
            lengths = [item[2] for item in data]
        else:
            lengths = [len(item) for item in data]
        
        path_length_counts = pd.Series(lengths).value_counts().sort_index()
        with pd.option_context('display.max_rows', None):
            print(path_length_counts)
    else:
        force_print("No data to display.")
    force_print("--------------------------------------\n")

# ==========================================
# 2. キャッシュ・保存関連 (旧 cache_utils 含む)
# ==========================================

def get_config_hash():
    """config設定からハッシュを生成（設定変更時にキャッシュを無効化するため）"""
    try:
        relevant_configs = [
            str(config.MAX_SEQ_LEN),        # 入力系列長
            str(config.TRAIN_MAX),          # 学習データの期間
            str(config.VALID_NUM),          # 評価期間
            str(config.DATA_BASE_DIR),      # データパス
            str(config.MAX_NUM),            # 読み込み上限
            str(config.MAX_STRAIN_NUM),     # 株数上限
            str(config.MAX_NUM_PER_STRAIN), # 株ごとのサンプル上限
            str(config.TARGET_LEN),         # 予測ターゲット長
            str(config.MAX_CO_OCCURRENCE),  # 共起上限
            str(config.VALID_RATIO),        # 分割比率
            str(config.SEED),               # 乱数シード (分割に影響)
            str(config.NUM_FEATURE_STRING), # 特徴量数
            str(config.NUM_CHEM_FEATURES),  # 特徴量数
            str(config.ABLATION_MASKS),     # マスク設定 (データの中身が変わるため)
        ]
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

def save_model(model, dir_path):
    """簡易モデル保存"""
    os.makedirs(dir_path, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    model_path = os.path.join(dir_path, f'{timestamp}_model_state.pth')
    torch.save(model.state_dict(), model_path)
    print(f"[INFO] Model saved to {model_path}")

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
    columns: [original_len, targets, predictions_region, predictions_position, hit_region, hit_position]
    """
    # 辞書リストに変換して保存
    rows = []
    for r in results:
        rows.append({
            'original_len': r['len'],
            'raw_path': r['raw_path'], # 生のパス文字列
            'strain': r['strain'],     # 株名
            'target_region': str(list(r['targets_region'])),
            'pred_region': str(list(r['preds_region'])),
            'hit_region': r['hit_region'],
            'target_position': str(list(r['targets_position'])),
            'pred_position': str(list(r['preds_position'])),
            'hit_position': r['hit_position']
        })
    
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