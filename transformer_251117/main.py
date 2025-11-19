# 251028より特徴量生成を効率化（したが処理時間に変化なし？）
# 学習経過・予測結果の保存機能、ablation実験用のマスク機能、再現率etcの表示機能、使用株の表示機能は未実装
# --- main.py (Enhanced with Efficiency Features) ---
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import pandas as pd
import os
from datetime import datetime
import time

# 各モジュールをインポート
from . import config

from .model import HierarchicalTransformer
from .dataset import get_mock_data, create_dataloader, import_strains, filter_unique, split_data_by_length, get_mutation_data, separate_XY, filter_num_per_strain, sort_strain_by_num_and_filter_strain
from .train import train_one_epoch
from .evaluate import evaluate
from .utils import load_cache, save_cache, get_config_hash, force_print, save_model, print_config

def set_seed(seed):
    """乱数シードを固定する関数"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def main():
    print(f"PID: {os.getpid()}")
    set_seed(config.SEED)
    force_print(f"Using device: {config.DEVICE}")
    print_config()

    # 1. キャッシュのパスを決定
    config_hash = get_config_hash()
    cache_file_name = f"data_cache_{config_hash}.pkl"
    cache_path = os.path.join(config.CACHE_DIR, cache_file_name)

    # 2. キャッシュの読み込み試行
    cached_data = None
    if not config.FORCE_REPROCESS:
        cached_data = load_cache(cache_path)

    if cached_data:
        # キャッシュからロード
        train, valid, test = cached_data
        
        # DataFrameが空でないかチェックしつつ、最小・最大長を取得 (表示用)
        # (★注: この部分もキャッシュから取得するのが望ましいが、簡易実装としてdfから再計算)
        force_print("[INFO] Calculating split lengths from cached data...")
        # (この部分は main の print 文のために必要)
        names, lengths, mutation_paths, strains = import_strains(usher_dir=config.DATA_BASE_DIR, max_num=config.MAX_NUM, max_cooccur=config.MAX_CO_OCCUR)
        df_unique = filter_unique(names, lengths, mutation_paths, strains)

        if config.MAX_STRAIN_NUM is not None:
            df_unique = sort_strain_by_num_and_filter_strain(df_unique, config.MAX_STRAIN_NUM)

        if config.MAX_NUM_PER_STRAIN is not None:
            df_unique = filter_num_per_strain(df_unique, config.MAX_NUM_PER_STRAIN)

        train_df, valid_df, test_df = split_data_by_length(
            df_unique, config.TRAIN_MAX, config.VALID_NUM, config.VALID_RATIO, config.SEED
        )
        train_min_len = int(train_df['original_len'].min()) if not train_df.empty else 0
        train_max_len = int(train_df['original_len'].max()) if not train_df.empty else 0
        val_min_len = int(valid_df['original_len'].min()) if not valid_df.empty else 0
        val_max_len = int(valid_df['original_len'].max()) if not valid_df.empty else 0
        test_min_len = int(test_df['original_len'].min()) if not test_df.empty else 0
        test_max_len = int(test_df['original_len'].max()) if not test_df.empty else 0

    else:
        # 3. キャッシュがない場合、従来の処理を実行
        force_print("--- 1. Loading and Preprocessing Data (Cache not found or forced) ---")
        
        # 3a. 重いCSVを1回だけロード
        force_print("[INFO] Loading base dataframes (Codon, Freq, Dissimilarity)...")
        df_freq = pd.read_csv(config.Freq_csv)
        df_dissimilarity = pd.read_csv(config.Disimilarity_csv)
        df_codon = pd.read_csv(config.Codon_csv)

        # 3b. データの読み込みと前処理（共起、ユニークによるフィルタリング）
        names, lengths, mutation_paths, strains = import_strains(
            usher_dir=config.DATA_BASE_DIR, max_num=config.MAX_NUM, max_cooccur=config.MAX_CO_OCCUR
        )
        df_unique = filter_unique(names, lengths, mutation_paths, strains)

        if config.MAX_STRAIN_NUM is not None:
            df_unique = sort_strain_by_num_and_filter_strain(df_unique, config.MAX_STRAIN_NUM)

        if config.MAX_NUM_PER_STRAIN is not None:
            df_unique = filter_num_per_strain(df_unique, config.MAX_NUM_PER_STRAIN)

        # 3c. 訓練/評価/テストに分割
        train_df, valid_df, test_df = split_data_by_length(
            df_unique, config.TRAIN_MAX, config.VALID_NUM, config.VALID_RATIO, config.SEED
        )
        
        # (表示用の変数)
        train_min_len = int(train_df['original_len'].min()) if not train_df.empty else 0
        train_max_len = int(train_df['original_len'].max()) if not train_df.empty else 0
        val_min_len = int(valid_df['original_len'].min()) if not valid_df.empty else 0
        val_max_len = int(valid_df['original_len'].max()) if not valid_df.empty else 0
        test_min_len = int(test_df['original_len'].min()) if not test_df.empty else 0
        test_max_len = int(test_df['original_len'].max()) if not test_df.empty else 0

        # 3d. 特徴量付加 (引数でdf_* を渡す)
        force_print("[INFO] Processing Train features...")
        train_names, train_lengths, train_features_paths = get_mutation_data(
            train_df['name'].tolist(), train_df['original_len'].tolist(), train_df['path'].tolist(),
            df_codon, df_freq, df_dissimilarity
        )
        force_print("[INFO] Processing Validation features...")
        valid_names, valid_lengths, valid_features_paths = get_mutation_data(
            valid_df['name'].tolist(), valid_df['original_len'].tolist(), valid_df['path'].tolist(),
            df_codon, df_freq, df_dissimilarity
        )
        force_print("[INFO] Processing Test features...")
        test_names, test_lengths, test_features_paths = get_mutation_data(
            test_df['name'].tolist(), test_df['original_len'].tolist(), test_df['path'].tolist(),
            df_codon, df_freq, df_dissimilarity
        )

        # 3e. X, Y への分割
        force_print("[INFO] Separating X and Y...")
        train = separate_XY(train_features_paths, train_lengths, train_df['path'].tolist(), config.SEQ_LEN, config.TARGET_LEN)
        valid = separate_XY(valid_features_paths, valid_lengths, valid_df['path'].tolist(), config.SEQ_LEN, config.TARGET_LEN)
        test = separate_XY(test_features_paths, test_lengths, test_df['path'].tolist(), config.SEQ_LEN, config.TARGET_LEN)

        # 3f. キャッシュに保存
        save_cache([train, valid, test], cache_path)   

    train_loader = create_dataloader(train, config.BATCH_SIZE, shuffle=True)
    val_loader = create_dataloader(valid, config.BATCH_SIZE, shuffle=False)
    test_loader = create_dataloader(test, config.BATCH_SIZE, shuffle=False)
    
    force_print(f"Data loaded: {len(train)} train, {len(valid)} validation, {len(test)} test samples.")
    force_print(f"Training lengths: {train_min_len} to {train_max_len}")
    force_print(f"Validation lengths: {val_min_len} to {val_max_len}")
    force_print(f"Test lengths: {test_min_len} to {test_max_len}")
    
    # 2. モデル、損失関数、オプティマイザの初期化
    model = HierarchicalTransformer().to(config.DEVICE)
    loss_fn = nn.CrossEntropyLoss(reduction='none') 
    
    # Weight Decay追加
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config.LEARNING_RATE,
        weight_decay=getattr(config, 'WEIGHT_DECAY', 0.01)
    )
    
    # Early Stopping用の変数
    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience = getattr(config, 'EARLY_STOPPING_PATIENCE', 5)
    
    # タイムスタンプベースの出力ディレクトリ
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = os.path.join(config.MODEL_SAVE_DIR, timestamp)
    os.makedirs(run_output_dir, exist_ok=True)
    best_model_path = os.path.join(run_output_dir, f"best_model_{get_config_hash()[:8]}.pth")
    
    # 3. 訓練・評価ループ
    force_print(f"Starting training... Output dir: {run_output_dir}")
    force_print(f"Early stopping patience: {patience}")
    
    for epoch in range(config.EPOCHS):
        epoch_start_time = time.time()
        force_print(f"--- Epoch {epoch+1}/{config.EPOCHS} ---")
        
        # 訓練
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn)

        # 評価
        val_loss, val_metrics = evaluate(model, val_loader, loss_fn)
        
        epoch_time = time.time() - epoch_start_time
        force_print(f"Train Loss: {train_loss:.4f}")
        force_print(f"Validation Loss: {val_loss:.4f} (epoch time: {epoch_time:.1f}s)")

        # エポックごとの適合度を表示
        force_print(f"--- Validation Hit Rate @{config.TOP_K_EVAL} ---")
        for ts_len in sorted(val_metrics.keys()):
            metrics = val_metrics[ts_len]
            force_print(f"  Len {ts_len:>2} ({metrics['num_samples']:>3} samples): "
                      f"Region: {metrics['region_hit_rate_percent']:.2f}% | "
                      f"Position: {metrics['position_hit_rate_percent']:.2f}%")
        
        # Early Stopping チェック
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            force_print(f"Validation loss improved ({val_loss:.4f}). Saving model...")
            
            # ベストモデル保存
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, best_model_path)
        else:
            epochs_no_improve += 1
            force_print(f"Validation loss did not improve. Early stopping counter: {epochs_no_improve}/{patience}")
            if epochs_no_improve >= patience:
                force_print("Early stopping triggered.")
                break

    # ベストモデルから最終評価
    if os.path.exists(best_model_path):
        force_print("--- Final Evaluation with Best Model ---")
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        force_print(f"Loaded best model from epoch {checkpoint['epoch']+1}")
        
        # 最終評価
        final_val_loss, final_val_metrics = evaluate(model, val_loader, loss_fn)
        force_print(f"Final Validation Loss: {final_val_loss:.4f}")
        
        # テストデータでの評価
        if test_loader is not None and len(test_loader) > 0:
            test_loss, test_metrics = evaluate(model, test_loader, loss_fn)
            force_print(f"Test Loss: {test_loss:.4f}")
            force_print("--- Test Results ---")
            for ts_len in sorted(test_metrics.keys()):
                metrics = test_metrics[ts_len]
                force_print(f"  Len {ts_len:>2}: Region: {metrics['region_hit_rate_percent']:.2f}% | "
                          f"Position: {metrics['position_hit_rate_percent']:.2f}%")
    else:
        force_print("[WARNING] No best model found.")
        save_model(model, run_output_dir)
    print("Training complete.")

    # 4. 最終評価 (ループの外で実行)
    
    # 4a. 評価データ (ex.25-30) での最終的な「適合度」を報告
    force_print(f"Starting final evaluation on Validation Set ({val_min_len}-{val_max_len})...")
    final_val_loss, final_val_metrics = evaluate(model, val_loader, loss_fn)

    print(f"--- Final Validation Results (Goodness of Fit on {val_min_len}-{val_max_len}) ---")
    print(f"  Validation Loss: {final_val_loss:.4f}")
    print(f"  --- Validation Hit Rate @{config.TOP_K_EVAL} (on {val_min_len}-{val_max_len}) ---")
    for ts_len in sorted(final_val_metrics.keys()):
        metrics = final_val_metrics[ts_len]
        print(f"    Len {ts_len: >2} ({metrics['num_samples']: >3} samples): "
              f"Region Hit: {metrics['region_hit_rate_percent']:.2f}% | "
              f"Position Hit: {metrics['position_hit_rate_percent']:.2f}%")
    print("---------------------------------")

    # 4b. テストデータ (ex.31-40) での最終的な「予測性能」を報告
    force_print(f"Starting final evaluation on Test Set ({test_min_len}-{test_max_len})...")
    test_loss, test_metrics = evaluate(model, test_loader, loss_fn)

    print(f"--- Final Test Results (Prediction Performance on {test_min_len}-{test_max_len}) ---")
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  --- Test Hit Rate @{config.TOP_K_EVAL} (on {test_min_len}-{test_max_len}) ---")
    for ts_len in sorted(test_metrics.keys()):
        metrics = test_metrics[ts_len]
        print(f"    Len {ts_len: >2} ({metrics['num_samples']: >3} samples): "
              f"Region Hit: {metrics['region_hit_rate_percent']:.2f}% | "
              f"Position Hit: {metrics['position_hit_rate_percent']:.2f}%")
    print("---------------------------------")

    # 4c. 評価・テストデータでの領域一致での総合適合度・予測性能を報告
    print(f"--- Summary of Region Hit Rates @{config.TOP_K_EVAL} ---")
    for ts_len in sorted(final_val_metrics.keys()):
        metrics = final_val_metrics[ts_len]
        print(f"    Len {ts_len: >2} ({metrics['num_samples']: >3} samples): "
              f"Region Hit: {metrics['region_hit_rate_percent']:.2f}%")

    for ts_len in sorted(test_metrics.keys()):
        metrics = test_metrics[ts_len]
        print(f"    Len {ts_len: >2} ({metrics['num_samples']: >3} samples): "
              f"Region Hit: {metrics['region_hit_rate_percent']:.2f}%")
    print("---------------------------------")

    # 4d. 評価・テストデータでの位置一致での総合適合度・予測性能を報告
    print(f"--- Summary of Position Hit Rates @{config.TOP_K_EVAL} ---")
    for ts_len in sorted(final_val_metrics.keys()):
        metrics = final_val_metrics[ts_len]
        print(f"    Len {ts_len: >2} ({metrics['num_samples']: >3} samples): "
              f"Position Hit: {metrics['position_hit_rate_percent']:.2f}%")

    for ts_len in sorted(test_metrics.keys()):
        metrics = test_metrics[ts_len]
        print(f"    Len {ts_len: >2} ({metrics['num_samples']: >3} samples): "
              f"Position Hit: {metrics['position_hit_rate_percent']:.2f}%")
    print("---------------------------------")

if __name__ == "__main__":
    main()