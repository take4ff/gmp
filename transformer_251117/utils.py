# --- utils.py ---
import torch
import torch.nn as nn
import math

import pickle
import os
import hashlib
from . import config
import time

def save_model(model, dir):
    """モデルの状態を保存するユーティリティ関数"""
    os.makedirs(dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    model_path = os.path.join(dir, f'{timestamp}_model_state.pth')
    torch.save(model.state_dict(), model_path)
    print(f"[INFO] Model saved to {model_path}")

def force_print(message):
    """強制的に標準出力にメッセージを表示するユーティリティ関数"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"[{timestamp}] {message}", flush=True)

def get_config_hash():
    """configモジュールの関連する設定からハッシュを生成する"""
    
    # キャッシュファイル名に影響する主要な設定を文字列として連結
    # これらの設定が変われば、キャッシュは自動的に再生成される
    try:
        relevant_configs = [
            str(config.SEQ_LEN),
            str(config.TRAIN_MAX),
            str(config.VALID_NUM),
            str(config.DATA_BASE_DIR),
            str(config.MAX_NUM),
            str(config.MAX_STRAIN_NUM),
            str(config.MAX_NUM_PER_STRAIN),
            str(config.TARGET_LEN),
            str(config.MAX_CO_OCCUR),
            str(config.VALID_RATIO),
            str(config.SEED),
        ]
        config_string = "_".join(relevant_configs)
        
        # ハッシュを生成して短いファイル名にする
        return hashlib.md5(config_string.encode('utf-8')).hexdigest()
        
    except Exception as e:
        print(f"[WARNING] config.py からハッシュを生成できませんでした: {e}")
        return "default_cache"

def print_config():
    """configモジュールの設定を標準出力に表示する"""
    force_print("Current Configuration:")
    for attr in dir(config):
        if not attr.startswith("__"):
            value = getattr(config, attr)
            force_print(f"  {attr}: {value}")

def load_cache(cache_path):
    """キャッシュファイルからデータをロードする"""
    if os.path.exists(cache_path):
        print(f"[INFO] Loading data from cache: {cache_path}")
        try:
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            return data
        except Exception as e:
            print(f"[WARNING] Failed to load cache: {e}. Re-processing data.")
            return None
    print("[INFO] Cache not found. Processing data...")
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

class PositionalEncoding(nn.Module):
    """ Transformer標準の位置エンコーディング """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = pe.permute(1, 0, 2) # [1, max_len, d_model] (batch_first=True用)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ x: [Batch, Seq, Feature] """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


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

# === 拡張ユーティリティ関数 (transformer_251117からの移植) ===
def save_config(output_dir):
    """configモジュールの設定をJSONファイルとして保存する"""
    import json
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

def print_path_length_distribution(data, name):
    """Helper function to print path length distribution."""
    import pandas as pd
    force_print(f"--- {name} Path Length Distribution ---")
    if hasattr(data, '__len__') and len(data) > 0:
        if isinstance(data, dict) and 'original_len' in data:
            lengths = data['original_len']
        elif hasattr(data[0], 'original_len'):
            lengths = [item.original_len for item in data]
        else:
            lengths = [len(item) for item in data]
        
        path_length_counts = pd.Series(lengths).value_counts().sort_index()
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(path_length_counts)
    else:
        force_print("No data to display.")
    force_print("--------------------------------------\n")

def save_enhanced_model(model, optimizer, epoch, loss, file_path):
    """拡張モデル保存（メタデータ付き）"""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        save_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'timestamp': time.time(),
            'config_hash': get_config_hash()
        }
        torch.save(save_data, file_path)
        force_print(f"[INFO] Enhanced model saved to {file_path}")
    except Exception as e:
        force_print(f"[WARNING] Failed to save enhanced model: {e}")

def load_enhanced_model(model, optimizer, file_path):
    """拡張モデル読み込み"""
    try:
        checkpoint = torch.load(file_path, map_location=config.DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        epoch = checkpoint.get('epoch', 0)
        loss = checkpoint.get('loss', 0.0)
        force_print(f"[INFO] Enhanced model loaded from {file_path} (epoch {epoch}, loss {loss:.4f})")
        return epoch, loss
    except Exception as e:
        force_print(f"[ERROR] Failed to load enhanced model: {e}")
        return 0, 0.0