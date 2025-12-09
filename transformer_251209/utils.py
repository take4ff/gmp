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
    print("target_idx,region_id,position_id,protein_pos_id")
    for i, target in enumerate(y_targets):
        region_id, position_id, protein_pos_id = target
        print(f"{i},{region_id},{position_id},{protein_pos_id}")
    
    # 入力特徴量 (CSV形式)
    print(f"\n# 入力特徴量 (X)")
    print("# cat: bef=塩基前, pos=塩基位置, aft=塩基後, c_pos=コドン位置, aa_b=AA前, p_pos=タンパク質位置, aa_a=AA後, region=領域")
    print("# num: hydro=疎水性変化, size=サイズ変化, charge=電荷変化, dissim=類似度, pam=PAM250, freq=頻度")
    
    # ヘッダ
    print("timestep,co_occur,bef,pos,aft,c_pos,aa_b,p_pos,aa_a,region,hydro,size,charge,dissim,pam,freq")
    
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
            'strength_score': r.get('strength_score', 0.0),  # 強度スコア (正解)
            'pred_strength': r.get('pred_strength', 0.0),    # 強度スコア (予測)
            'target_region': str(list(r['targets_region'])),
            'pred_region': str(list(r['preds_region'])),
            'hit_region': r['hit_region'],
            'target_position': str(list(r['targets_position'])),
            'pred_position': str(list(r['preds_position'])),
            'hit_position': r['hit_position'],
        }
        # Protein Position がある場合のみ追加
        if 'targets_protein_pos' in r:
            row['target_protein_pos'] = str(list(r['targets_protein_pos']))
            row['pred_protein_pos'] = str(list(r['preds_protein_pos']))
            row['hit_protein_pos'] = r['hit_protein_pos']
        
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
    強度カテゴリ別メトリクスをCSVに保存
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
        
        colors = {'region': '#2ecc71', 'position': '#3498db', 'protein': '#9b59b6'}
        
        # 1. Hit Rate比較
        ax = axes[0, 0]
        ax.plot(timesteps, [metrics_by_ts[t]['region_hit_rate'] for t in timesteps], 'o-', label='Region', color=colors['region'])
        ax.plot(timesteps, [metrics_by_ts[t]['position_hit_rate'] for t in timesteps], 's-', label='Position', color=colors['position'])
        ax.plot(timesteps, [metrics_by_ts[t]['protein_pos_hit_rate'] for t in timesteps], '^-', label='Protein Pos', color=colors['protein'])
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Hit Rate (%)')
        ax.set_title('Hit Rate')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Precision (適合率)
        ax = axes[0, 1]
        ax.plot(timesteps, [metrics_by_ts[t]['region_precision'] for t in timesteps], 'o-', label='Region', color=colors['region'])
        ax.plot(timesteps, [metrics_by_ts[t]['position_precision'] for t in timesteps], 's-', label='Position', color=colors['position'])
        ax.plot(timesteps, [metrics_by_ts[t]['protein_pos_precision'] for t in timesteps], '^-', label='Protein Pos', color=colors['protein'])
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Precision (%)')
        ax.set_title('Precision (適合率)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Recall (再現率)
        ax = axes[1, 0]
        ax.plot(timesteps, [metrics_by_ts[t]['region_recall'] for t in timesteps], 'o-', label='Region', color=colors['region'])
        ax.plot(timesteps, [metrics_by_ts[t]['position_recall'] for t in timesteps], 's-', label='Position', color=colors['position'])
        ax.plot(timesteps, [metrics_by_ts[t]['protein_pos_recall'] for t in timesteps], '^-', label='Protein Pos', color=colors['protein'])
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Recall (%)')
        ax.set_title('Recall (再現率)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. F1 Score
        ax = axes[1, 1]
        ax.plot(timesteps, [metrics_by_ts[t]['region_f1'] for t in timesteps], 'o-', label='Region', color=colors['region'])
        ax.plot(timesteps, [metrics_by_ts[t]['position_f1'] for t in timesteps], 's-', label='Position', color=colors['position'])
        ax.plot(timesteps, [metrics_by_ts[t]['protein_pos_f1'] for t in timesteps], '^-', label='Protein Pos', color=colors['protein'])
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
    強度カテゴリ別のメトリクスをグラフ化
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        
        timesteps = sorted(cat_metrics.keys())
        categories = ['low', 'medium', 'high']
        colors = {'low': '#e74c3c', 'medium': '#f39c12', 'high': '#27ae60'}
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
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
        
        # 2. Position Hit Rate by Category
        ax = axes[0, 1]
        for cat in categories:
            values = [cat_metrics[t][cat]['position_hit_rate'] for t in timesteps]
            ax.plot(timesteps, values, 's-', label=cat.capitalize(), color=colors[cat])
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Hit Rate (%)')
        ax.set_title('Position Hit Rate by Category')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Strength MAE by Category
        ax = axes[1, 0]
        for cat in categories:
            values = [cat_metrics[t][cat]['strength_mae'] for t in timesteps]
            ax.plot(timesteps, values, 'D-', label=cat.capitalize(), color=colors[cat])
        ax.set_xlabel('Timestep')
        ax.set_ylabel('MAE')
        ax.set_title('Strength MAE by Category')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Sample Distribution (Stacked Bar)
        ax = axes[1, 1]
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
        
        plt.tight_layout()
        path = os.path.join(output_dir, f'{prefix}_category_plot.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        force_print(f"[INFO] Category plot saved to {path}")
        
    except ImportError:
        force_print("[WARNING] matplotlib not available, skipping plots")

# ==========================================
# 5. 結果表示・WandB関連
# ==========================================

def init_wandb():
    """WandB初期化"""
    import types
    from datetime import datetime
    try:
        import wandb
        if config.USE_WANDB:
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

# ==========================================
# 6. 階層的評価レポート (4部構成)
# ==========================================


def print_combined_report(val_details, test_details):
    """
    ValidationとTestを統合した階層的評価レポート
    
    Args:
        val_details: Validation評価の詳細結果リスト
        test_details: Test評価の詳細結果リスト
    """
    import numpy as np
    
    if not val_details and not test_details:
        print("[WARNING] No results to display")
        return
    
    df_val = pd.DataFrame(val_details) if val_details else pd.DataFrame()
    df_test = pd.DataFrame(test_details) if test_details else pd.DataFrame()
    
    # 全データを結合して強度閾値を計算
    df_all = pd.concat([df_val, df_test], ignore_index=True) if len(df_val) > 0 or len(df_test) > 0 else pd.DataFrame()
    
    print("\n" + "=" * 100)
    print("【統合評価レポート - Validation vs Test】")
    print("=" * 100)
    
    # =================================================================
    # 1. Executive Summary (強度スコア別)
    # =================================================================
    print("\n" + "-" * 100)
    print("=== 1. Executive Summary (強度スコア別: High/Medium/Low) ===")
    print("-" * 100)
    
    def calc_summary(df):
        if len(df) == 0:
            return {'count': 0, 'region': 0, 'pos': 0, 'prot': 0}
        return {
            'count': len(df),
            'region': df['hit_region'].mean() * 100,
            'pos': df['hit_position'].mean() * 100,
            'prot': df['hit_protein_pos'].mean() * 100
        }
    
    def get_strength_category(score, low_max, med_max):
        if score < low_max:
            return 'low'
        elif score < med_max:
            return 'medium'
        else:
            return 'high'
    
    # 閾値を使用してカテゴリ分け
    low_max = config.STRENGTH_CATEGORY_LOW_MAX
    med_max = config.STRENGTH_CATEGORY_MED_MAX
    
    print(f"\n  閾値: Low(<{low_max}), Medium(<{med_max}), High(≥{med_max})")
    print("\n  Category |  Val Count | Val Reg | Val Pos | Val Prot ||  Test Count | Test Reg | Test Pos | Test Prot")
    print("  " + "-" * 105)
    
    for category in ['high', 'medium', 'low']:
        # Validation
        if len(df_val) > 0:
            df_val['strength_cat'] = df_val['strength_score'].apply(lambda x: get_strength_category(x, low_max, med_max))
            v_cat = df_val[df_val['strength_cat'] == category]
        else:
            v_cat = pd.DataFrame()
        
        # Test
        if len(df_test) > 0:
            df_test['strength_cat'] = df_test['strength_score'].apply(lambda x: get_strength_category(x, low_max, med_max))
            t_cat = df_test[df_test['strength_cat'] == category]
        else:
            t_cat = pd.DataFrame()
        
        v_s = calc_summary(v_cat)
        t_s = calc_summary(t_cat)
        
        v_str = f"{v_s['count']:>10} | {v_s['region']:>5.1f}%  | {v_s['pos']:>5.1f}%  | {v_s['prot']:>6.1f}%" if v_s['count'] > 0 else "         - |      -  |      -  |       -"
        t_str = f"{t_s['count']:>11} | {t_s['region']:>6.1f}%  | {t_s['pos']:>6.1f}%  | {t_s['prot']:>7.1f}%" if t_s['count'] > 0 else "          - |       -  |       -  |        -"
        
        marker = " ★" if category == 'high' else ""
        print(f"  {category:8s} | {v_str} || {t_str}{marker}")
    
    # 全体サマリー（Generalizationセクション用）
    val_s = calc_summary(df_val)
    test_s = calc_summary(df_test)
    
    # =================================================================
    # 2. Biological Analysis (領域別分析) - 適合率・再現率・F1追加
    # =================================================================
    print("\n" + "-" * 85)
    print("=== 2. Biological Analysis (タンパク質領域別) ===")
    print("-" * 85)
    
    def calc_region_metrics(df):
        """領域別の詳細メトリクスを計算"""
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
            # FP: 予測したが正解でない
            for pred_id in row['preds_region']:
                if pred_id not in row['targets_region']:
                    if pred_id not in region_stats:
                        region_stats[pred_id] = {'tp': 0, 'fp': 0, 'fn': 0, 'count': 0}
                    region_stats[pred_id]['fp'] += 1
        return region_stats
    
    val_regions = calc_region_metrics(df_val) if len(df_val) > 0 else {}
    test_regions = calc_region_metrics(df_test) if len(df_test) > 0 else {}
    
    # 全領域をマージ
    all_regions = set(val_regions.keys()) | set(test_regions.keys())
    region_names = {v: k for k, v in config.PROTEIN_VOCABS.items()}
    
    print("\n  Region     |  Val Count |  Val Prec |  Val Rec  |  Val F1   || Test Count | Test Prec | Test Rec  | Test F1")
    print("  " + "-" * 110)
    
    # 合計Count順でソート
    def total_count(reg_id):
        return val_regions.get(reg_id, {}).get('count', 0) + test_regions.get(reg_id, {}).get('count', 0)
    
    sorted_regs = sorted(all_regions, key=total_count, reverse=True)[:10]
    
    for reg_id in sorted_regs:
        name = region_names.get(reg_id, f"ID:{reg_id}")[:10]
        
        # Validation
        vs = val_regions.get(reg_id, {'tp': 0, 'fp': 0, 'fn': 0, 'count': 0})
        v_prec = vs['tp'] / (vs['tp'] + vs['fp']) * 100 if (vs['tp'] + vs['fp']) > 0 else 0
        v_rec = vs['tp'] / (vs['tp'] + vs['fn']) * 100 if (vs['tp'] + vs['fn']) > 0 else 0
        v_f1 = 2 * v_prec * v_rec / (v_prec + v_rec) if (v_prec + v_rec) > 0 else 0
        
        # Test
        ts = test_regions.get(reg_id, {'tp': 0, 'fp': 0, 'fn': 0, 'count': 0})
        t_prec = ts['tp'] / (ts['tp'] + ts['fp']) * 100 if (ts['tp'] + ts['fp']) > 0 else 0
        t_rec = ts['tp'] / (ts['tp'] + ts['fn']) * 100 if (ts['tp'] + ts['fn']) > 0 else 0
        t_f1 = 2 * t_prec * t_rec / (t_prec + t_rec) if (t_prec + t_rec) > 0 else 0
        
        marker = " ★" if name == 'S' else ""
        print(f"  {name:10s} | {vs['count']:>10} | {v_prec:>7.1f}% | {v_rec:>7.1f}% | {v_f1:>7.1f}% || {ts['count']:>10} | {t_prec:>7.1f}% | {t_rec:>7.1f}% | {t_f1:>7.1f}%{marker}")
    
    # =================================================================
    # 3. Temporal Dynamics (パス長別) - 長さ1刻み
    # =================================================================
    print("\n" + "-" * 85)
    print("=== 3. Temporal Dynamics (パス長別 - 長さ1刻み) ===")
    print("-" * 85)
    
    all_lengths = set()
    if len(df_val) > 0:
        all_lengths |= set(df_val['len'].unique())
    if len(df_test) > 0:
        all_lengths |= set(df_test['len'].unique())
    
    print("\n  Length |  Val n | Val Reg | Val Pos | Val Prot ||  Test n | Test Reg | Test Pos | Test Prot")
    print("  " + "-" * 100)
    
    for length in sorted(all_lengths):
        v_df = df_val[df_val['len'] == length] if len(df_val) > 0 else pd.DataFrame()
        t_df = df_test[df_test['len'] == length] if len(df_test) > 0 else pd.DataFrame()
        
        v_n = len(v_df)
        t_n = len(t_df)
        
        v_reg = v_df['hit_region'].mean() * 100 if v_n > 0 else 0
        v_pos = v_df['hit_position'].mean() * 100 if v_n > 0 else 0
        v_prot = v_df['hit_protein_pos'].mean() * 100 if v_n > 0 else 0
        
        t_reg = t_df['hit_region'].mean() * 100 if t_n > 0 else 0
        t_pos = t_df['hit_position'].mean() * 100 if t_n > 0 else 0
        t_prot = t_df['hit_protein_pos'].mean() * 100 if t_n > 0 else 0
        
        v_str = f"{v_n:>6} | {v_reg:>5.1f}%  | {v_pos:>5.1f}%  | {v_prot:>6.1f}%" if v_n > 0 else "     - |      -  |      -  |       -"
        t_str = f"{t_n:>7} | {t_reg:>6.1f}%  | {t_pos:>6.1f}%  | {t_prot:>7.1f}%" if t_n > 0 else "      - |       -  |       -  |        -"
        
        print(f"  {length:>6} | {v_str} || {t_str}")
    
    print("\n" + "=" * 85 + "\n")