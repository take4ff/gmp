import torch
import torch.nn as nn
import os
import pickle
import hashlib
import json
from . import config
import shutil
import matplotlib.pyplot as plt
import time
import pandas as pd

def print_path_length_distribution(data, name, utils):
    """Helper function to print path length distribution."""
    utils.force_print(f"--- {name} Path Length Distribution ---")
    if 'original_len' in data and len(data['original_len']) > 0:
        path_length_counts = pd.Series(data['original_len']).value_counts().sort_index()
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(path_length_counts)
    else:
        utils.force_print("No data to display.")
    utils.force_print("--------------------------------------\n")

def save_config(output_dir):
    """configモジュールの設定をJSONファイルとして保存する"""
    config_path = os.path.join(output_dir, 'config.json')
    try:
        config_dict = {
            attr: getattr(config, attr)
            for attr in dir(config)
            if not attr.startswith("__") and isinstance(getattr(config, attr), (int, float, str, bool, list, dict, tuple))
        }
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=4)
        force_print(f"[INFO] Configuration saved to {config_path}")
    except Exception as e:
        force_print(f"[WARNING] Failed to save configuration: {e}")

def force_print(message):
    """強制的に標準出力にメッセージを表示するユーティリティ関数"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"[{timestamp}] {message}", flush=True)

def get_config_hash():
    """現在のconfig設定から一意のハッシュを生成する"""
    try:
        config_dict = {
            attr: getattr(config, attr)
            for attr in dir(config)
            if not attr.startswith("__") and isinstance(getattr(config, attr), (int, float, str, bool, list, dict, tuple))
        }
        # 辞書のキーでソートして、常に同じ順序のJSON文字列を生成
        config_str = json.dumps(config_dict, sort_keys=True)
        return hashlib.md5(config_str.encode('utf-8')).hexdigest()
    except Exception as e:
        force_print(f"[WARNING] Failed to create config hash: {e}")
        return "default_hash"

def print_config():
    """configモジュールの設定を標準出力に表示する"""
    force_print("--- Current Configuration ---")
    for attr in dir(config):
        if not attr.startswith("__"):
            value = getattr(config, attr)
            # 関数やモジュールは表示しない
            if not callable(value) and not isinstance(value, type(torch)):
                force_print(f"{attr}: {value}")
    force_print("---------------------------\n")

def load_cache(cache_path):
    """キャッシュファイルからデータをロードする"""
    if os.path.exists(cache_path):
        force_print(f"[INFO] Loading data from cache: {cache_path}")
        try:
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            return data
        except Exception as e:
            force_print(f"[WARNING] Failed to load cache: {e}. Re-processing data.")
            return None
    force_print("[INFO] Cache not found. Processing data...")
    return None

def save_cache(data, cache_path):
    """データをキャッシュファイルに保存する"""
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        force_print(f"[INFO] Saving data to cache: {cache_path}")
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        force_print(f"[WARNING] Failed to save cache: {e}")

def save_model(model, optimizer, epoch, loss, file_path):
    """モデルの状態を保存する"""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, file_path)
        force_print(f"[INFO] Model saved to {file_path}")
    except Exception as e:
        force_print(f"[WARNING] Failed to save model: {e}")

def calculate_topk_hit_rate(pred_sets, target_sets):
    """
    予測セット(Top-K)と正解セット(共起)の積集合が空でなければヒットと見なす。
    Args:
        pred_sets (list of set): 各サンプルの予測IDのセットのリスト
        target_sets (list of set): 各サンプルの正解IDのセットのリスト
    Returns:
        float: ヒット率 (0.0-100.0)
    """
    if not pred_sets or not target_sets:
        return 0.0

    hits = 0
    total = len(pred_sets)

    for pred_s, target_s in zip(pred_sets, target_s):
        # 正解セットが空の場合は評価対象外
        if not target_s:
            total -= 1
            continue
        # 予測セットと正解セットの積集合（共通部分）があればヒット
        if len(pred_s.intersection(target_s)) > 0:
            hits += 1

    return (hits / total * 100.0) if total > 0 else 0.0

def save_plots(history, output_dir):
    """訓練過程の損失と精度をグラフにして保存する"""
    plt.style.use('ggplot')
    
    # 損失のグラフ
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(history['train_loss'], label='Train Loss')
    ax.plot(history['valid_loss'], label='Validation Loss')
    ax.set_title('Training and Validation Loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()
    loss_path = os.path.join(output_dir, 'loss_curve.png')
    fig.savefig(loss_path)
    plt.close(fig)

    # 精度のグラフ
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(history['valid_acc_protein'], label=f'Validation Protein Acc (Top-{config.TOP_K_EVAL})')
    ax.plot(history['valid_acc_pos'], label=f'Validation Position Acc (Top-{config.TOP_K_EVAL})')
    ax.set_title('Validation Accuracy')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy (%)')
    ax.legend()
    acc_path = os.path.join(output_dir, 'accuracy_curve.png')
    fig.savefig(acc_path)
    plt.close(fig)

    force_print(f"Plots saved to {output_dir}")

def save_timestep_evaluation(df, output_dir):
    """タイムステップごとの評価結果をCSVとグラフで保存する"""
    if df.empty:
        force_print("[WARNING] Timestep evaluation data is empty. Skipping saving.")
        return

    # CSVとして保存
    csv_path = os.path.join(output_dir, 'timestep_accuracy.csv')
    df.to_csv(csv_path, index=False)
    force_print(f"Timestep accuracy saved to {csv_path}")

    # グラフを作成して保存
    plt.style.use('ggplot')
    fig, ax = plt.subplots(1, 1, figsize=(15, 7))
    
    ax.plot(df['original_len'], df['protein_accuracy'], marker='o', linestyle='-', label='Protein Accuracy')
    ax.plot(df['original_len'], df['position_accuracy'], marker='x', linestyle='--', label='Position Accuracy')
    
    ax.set_title('Accuracy by Timestep (Validation + Test)')
    ax.set_xlabel('Timestep (Original Sequence Length)')
    ax.set_ylabel(f'Top-{config.TOP_K_EVAL} Accuracy (%)')
    ax.legend()
    ax.grid(True)
    
    # X軸の目盛りを整数にする
    max_len = df['original_len'].max()
    min_len = df['original_len'].min()
    ax.set_xticks(range(min_len, max_len + 1, max(1, (max_len - min_len) // 20)))

    plot_path = os.path.join(output_dir, 'timestep_accuracy.png')
    fig.savefig(plot_path)
    plt.close(fig)
    force_print(f"Timestep accuracy plot saved to {plot_path}")

def print_debug():
    from . import dataset
    df_codon, df_freq, df_dissimilarity = dataset.load_aux_data()

    file_path = "../usher_output/B.1.1.7/0/mutation_paths.tsv"
    df = pd.read_csv(file_path, sep='\t', header=0, names=['name', 'original_len', 'path'])
    df = df.head(1)
    print(df['path'][0])
    df['strain'] = "B.1.1.7"
    df['path'] = df['path'].str.split('>')
    lecode = dataset.process_feature_batch(df, df_codon, df_freq, df_dissimilarity)
    print(f"lecode: {lecode}")
