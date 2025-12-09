# --- main.py ---
# 予測結果の保存・グラフ化、再現率etcの計算は未実装
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import pandas as pd
import os
import time
from datetime import datetime
import wandb
import types

# ローカルモジュール
from . import config
from .model import HierarchicalTransformer, MultiTaskLoss
from .dataset import (
    create_dataloader, import_strains, filter_unique, 
    sort_strain_by_num_and_filter_strain, filter_num_per_strain,
    split_data_by_length, get_mutation_data, separate_XY
)
from .train import train_one_epoch
from .evaluate import evaluate
from .utils import (
    load_cache, save_cache, get_config_hash, 
    force_print, save_model, print_config,
    save_training_log, save_prediction_results, save_strain_info
)

def set_seed(seed):
    """乱数シードの固定"""
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

    if config.USE_WANDB:
        # configの内容を辞書化してwandbに保存
        # モジュールオブジェクトを除外するように修正
        config_dict = {
            k: v for k, v in vars(config).items() 
            if not k.startswith('__') and not isinstance(v, types.ModuleType)
        }
        wandb.init(
            project=config.WANDB_PROJECT_NAME, 
            name=config.WANDB_RUN_NAME if config.WANDB_RUN_NAME else f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config=config_dict
        )

    # 1. キャッシュ設定
    config_hash = get_config_hash()
    # メインキャッシュファイル（訓練/評価/テスト分割後のデータ用）
    cache_path = os.path.join(config.CACHE_DIR, f"data_cache_{config_hash}.pkl")

    cached_data = None
    if not config.FORCE_REPROCESS:
        cached_data = load_cache(cache_path)
    
    # 表示用変数の初期化 (NameError防止)
    train_min_len = train_max_len = 0
    val_min_len = val_max_len = 0
    test_min_len = test_max_len = 0

    # --- データロード & 前処理 ---
    if cached_data:
        train, valid, test = cached_data
        force_print("[INFO] Loaded split datasets from cache.")
        
        # キャッシュ使用時: データの長さを取得して表示用変数を更新
        # item構造: (x, y, original_len, raw_path, strain)
        if len(train) > 0:
             train_lens = [item[2] for item in train]
             train_min_len, train_max_len = min(train_lens), max(train_lens)
        if len(valid) > 0:
             val_lens = [item[2] for item in valid]
             val_min_len, val_max_len = min(val_lens), max(val_lens)
        if len(test) > 0:
             test_lens = [item[2] for item in test]
             test_min_len, test_max_len = min(test_lens), max(test_lens)

    else:
        # キャッシュなし: フルプロセス実行
        force_print("--- 1. Loading and Preprocessing Data ---")
        
        # 1a. 参照データのロード
        force_print("[INFO] Loading base dataframes...")
        df_freq = pd.read_csv(config.Freq_csv)
        df_dissimilarity = pd.read_csv(config.Disimilarity_csv)
        df_codon = pd.read_csv(config.Codon_csv)
        df_pam250 = pd.read_csv(config.PAM250_csv, index_col=0)

        # 1b. 生データのインポート
        names, lengths, mutation_paths, strains = import_strains(
            usher_dir=config.DATA_BASE_DIR, 
            max_num=config.MAX_NUM, 
            max_cooccur=config.MAX_CO_OCCURRENCE
        )
        
        # 1c. フィルタリング
        df_unique = filter_unique(names, lengths, mutation_paths, strains)

        if config.MAX_STRAIN_NUM is not None:
            df_unique = sort_strain_by_num_and_filter_strain(df_unique, config.MAX_STRAIN_NUM)
        if config.MAX_NUM_PER_STRAIN is not None:
            df_unique = filter_num_per_strain(df_unique, config.MAX_NUM_PER_STRAIN)

        # 1d. 訓練/評価/テストへの分割
        train_df, valid_df, test_df = split_data_by_length(
            df_unique, config.TRAIN_MAX, config.VALID_NUM, config.VALID_RATIO, config.SEED
        )
        
        # 表示用変数の設定
        if not train_df.empty:
            train_min_len = int(train_df['original_len'].min())
            train_max_len = int(train_df['original_len'].max())
        if not valid_df.empty:
            val_min_len = int(valid_df['original_len'].min())
            val_max_len = int(valid_df['original_len'].max())
        if not test_df.empty:
            test_min_len = int(test_df['original_len'].min())
            test_max_len = int(test_df['original_len'].max())

        # 1e. 特徴量生成 (Pandas排除・インクリメンタルキャッシュ版)
        force_print("[INFO] Processing Train features...")
        _, _, train_feats = get_mutation_data(
            train_df['name'].tolist(), train_df['original_len'].tolist(), train_df['path'].tolist(),
            df_codon, df_freq, df_dissimilarity, df_pam250
        )
        force_print("[INFO] Processing Validation features...")
        _, _, valid_feats = get_mutation_data(
            valid_df['name'].tolist(), valid_df['original_len'].tolist(), valid_df['path'].tolist(),
            df_codon, df_freq, df_dissimilarity, df_pam250
        )
        force_print("[INFO] Processing Test features...")
        _, _, test_feats = get_mutation_data(
            test_df['name'].tolist(), test_df['original_len'].tolist(), test_df['path'].tolist(),
            df_codon, df_freq, df_dissimilarity, df_pam250
        )

        # 1f. 入力(X)と正解(Y)への分割 (Strain情報も渡す)
        force_print("[INFO] Separating X and Y...")
        train = separate_XY(train_feats, train_df['original_len'].tolist(), train_df['path'].tolist(), 
                            train_df['strain'].tolist(), # Strain追加
                            config.MAX_SEQ_LEN, config.TARGET_LEN)
        valid = separate_XY(valid_feats, valid_df['original_len'].tolist(), valid_df['path'].tolist(), 
                            valid_df['strain'].tolist(), # Strain追加
                            config.MAX_SEQ_LEN, config.TARGET_LEN)
        test = separate_XY(test_feats, test_df['original_len'].tolist(), test_df['path'].tolist(), 
                           test_df['strain'].tolist(), # Strain追加
                           config.MAX_SEQ_LEN, config.TARGET_LEN)

        # 1g. メインキャッシュ保存
        save_cache([train, valid, test], cache_path)   

    # --- 3. データローダー作成 ---
    train_loader = create_dataloader(train, config.BATCH_SIZE, shuffle=True)
    val_loader = create_dataloader(valid, config.BATCH_SIZE, shuffle=False)
    test_loader = create_dataloader(test, config.BATCH_SIZE, shuffle=False)
    
    force_print(f"Data loaded: {len(train)} train, {len(valid)} validation, {len(test)} test samples.")
    force_print(f"Training lengths: {train_min_len} to {train_max_len}")
    force_print(f"Validation lengths: {val_min_len} to {val_max_len}")
    force_print(f"Test lengths: {test_min_len} to {test_max_len}")
    
    # --- 4. モデル・学習準備 ---
    model = HierarchicalTransformer().to(config.DEVICE)

    loss_wrapper = None
    if config.USE_MULTITASK_LOSS:
        force_print("[INFO] Using MultiTaskLoss (Automatic Weighting)")
        # 2つのタスク (Region, Position) 用に初期化し、デバイスへ転送
        loss_wrapper = MultiTaskLoss(num_tasks=2).to(config.DEVICE)
    else:
        force_print(f"[INFO] Using Fixed Loss Weights: Region={config.LOSS_WEIGHT_REGION}, Position={config.LOSS_WEIGHT_POSITION}")

    if config.USE_LABEL_SMOOTHING:
        force_print(f"[INFO] Using Label Smoothing (factor={config.LABEL_SMOOTHING_FACTOR})")
        loss_fn = nn.CrossEntropyLoss(reduction='none', label_smoothing=config.LABEL_SMOOTHING_FACTOR)
    else:
        loss_fn = nn.CrossEntropyLoss(reduction='none')

    if loss_wrapper is not None:
        params_to_optimize = list(model.parameters()) + list(loss_wrapper.parameters())
    else:
        params_to_optimize = model.parameters()

    optimizer = optim.AdamW(
        params_to_optimize,
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    scheduler = None
    if config.USE_SCHEDULER:
        force_print("[INFO] Using CosineAnnealingLR Scheduler")
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.EPOCHS, eta_min=config.SCHEDULER_ETA_MIN
        )
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience = config.EARLY_STOPPING_PATIENCE
    
    # 出力ディレクトリ設定
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # config.RESULT_SAVE_DIR (ex: ./results/) の下にタイムスタンプフォルダを作成
    run_output_dir = os.path.join(config.RESULT_SAVE_DIR, timestamp)
    os.makedirs(run_output_dir, exist_ok=True)
    best_model_path = os.path.join(run_output_dir, f"best_model_{config_hash[:8]}.pth")
    
    force_print(f"Starting training... Output dir: {run_output_dir}")
    
    # 株情報の保存 (Datasetの要素 idx=4 がstrain)
    if config.SAVE_STRAIN_INFO:
        save_strain_info([t[4] for t in train], run_output_dir, prefix="train")
        save_strain_info([t[4] for t in valid], run_output_dir, prefix="valid")
        save_strain_info([t[4] for t in test],  run_output_dir, prefix="test")

    # 学習ログ
    training_log = []
    
    # --- 5. 学習ループ ---
    for epoch in range(config.EPOCHS):
        epoch_start_time = time.time()
        force_print(f"--- Epoch {epoch+1}/{config.EPOCHS} ---")
        
        # 訓練
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, loss_wrapper)

        current_lr = optimizer.param_groups[0]['lr']
        if scheduler:
            scheduler.step()
            # 更新後のLRを取得
            current_lr = optimizer.param_groups[0]['lr']
        
        # 評価 (詳細結果はループ内では保存しないので _ で受ける)
        val_loss, val_metrics, _ = evaluate(model, val_loader, loss_fn)
        
        epoch_time = time.time() - epoch_start_time
        force_print(f"Train Loss: {train_loss:.4f}")
        force_print(f"Validation Loss: {val_loss:.4f} (epoch time: {epoch_time:.1f}s)")
        force_print(f"Learning Rate: {current_lr:.6f}")

        if config.USE_WANDB:
            # 最初のタイムステップのメトリクスを代表としてログ (必要に応じて平均化も可)
            # ここでは各タイムステップごとの詳細ではなく、代表値(例えば最後のタイムステップ)やLossを記録
            wandb_log_dict = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": current_lr,
            }
            # タイムステップごとの精度もログに追加 (Hit Rateのみ)
            for ts_len, m in val_metrics.items():
                wandb_log_dict[f"val_reg_hit_rate_len{ts_len}"] = m['region_hit_rate']
                wandb_log_dict[f"val_pos_hit_rate_len{ts_len}"] = m['position_hit_rate']
            
            wandb.log(wandb_log_dict)

        # ログ記録
        log_entry = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'time_seconds': epoch_time
        }
        # 全体平均のメトリクスもログに追加 (任意のタイムステップ平均など)
        # ここではログには損失のみ保存し、詳細はコンソール出力

        training_log.append(log_entry)

        # エポックごとの結果表示 (Hit Rate, Precision等)
        force_print(f"--- Validation Metrics @{config.TOP_K_EVAL} ---")
        for ts_len in sorted(val_metrics.keys()):
            m = val_metrics[ts_len]
            force_print(f"  Len {ts_len:>2} ({m['num_samples']:>3}): "
                      f"Reg Hit: {m['region_hit_rate']:.2f}% Prec: {m['region_precision']:.2f}% | "
                      f"Pos Hit: {m['position_hit_rate']:.2f}% Prec: {m['position_precision']:.2f}%")
        
        # Early Stopping & 保存
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            force_print(f"Validation loss improved. Saving model...")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, best_model_path)
        else:
            epochs_no_improve += 1
            force_print(f"No improvement. Counter: {epochs_no_improve}/{patience}")
            if epochs_no_improve >= patience:
                force_print("Early stopping triggered.")
                break
    
    # ログ保存
    save_training_log(training_log, run_output_dir)

    # --- 6. 最終評価 ---
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        force_print(f"Loaded best model from epoch {checkpoint['epoch']+1}")
    else:
        force_print("[WARNING] Best model not found, using last epoch model.")

    # 6a. 評価データでの最終確認 (詳細結果も取得)
    force_print(f"Starting final evaluation on Validation Set ({val_min_len}-{val_max_len})...")
    final_val_loss, final_val_metrics, val_details = evaluate(model, val_loader, loss_fn)
    
    if config.SAVE_PREDICTIONS:
        save_prediction_results(val_details, run_output_dir, prefix="valid")

    print(f"--- Final Validation Results ---")
    print(f"  Validation Loss: {final_val_loss:.4f}")
    for ts_len in sorted(final_val_metrics.keys()):
        m = final_val_metrics[ts_len]
        print(f"    Len {ts_len:>2}: Reg F1: {m['region_f1']:.2f}% Hit: {m['region_hit_rate']:.2f}% | "
              f"Pos F1: {m['position_f1']:.2f}% Hit: {m['position_hit_rate']:.2f}%")
    print("---------------------------------")

    # 6b. テストデータでの性能評価
    force_print(f"Starting final evaluation on Test Set ({test_min_len}-{test_max_len})...")
    if len(test_loader) > 0:
        test_loss, test_metrics, test_details = evaluate(model, test_loader, loss_fn)
        
        if config.SAVE_PREDICTIONS:
            save_prediction_results(test_details, run_output_dir, prefix="test")

        print(f"--- Final Test Results ---")
        print(f"  Test Loss: {test_loss:.4f}")
        for ts_len in sorted(test_metrics.keys()):
            m = test_metrics[ts_len]
            print(f"    Len {ts_len:>2}: Reg F1: {m['region_f1']:.2f}% Hit: {m['region_hit_rate']:.2f}% | "
                  f"Pos F1: {m['position_f1']:.2f}% Hit: {m['position_hit_rate']:.2f}%")
        print("---------------------------------")

        # 6c. まとめ (領域 Hit Rate)
        print(f"--- Summary of Region Hit Rates @{config.TOP_K_EVAL} ---")
        for ts_len in sorted(final_val_metrics.keys()):
            m = final_val_metrics[ts_len]
            print(f"    Len {ts_len: >2} ({m['num_samples']: >3} samples): "
                  f"Region Hit: {m['region_hit_rate']:.2f}%")
        
        for ts_len in sorted(test_metrics.keys()):
            m = test_metrics[ts_len]
            print(f"    Len {ts_len: >2} ({m['num_samples']: >3} samples): "
                  f"Region Hit: {m['region_hit_rate']:.2f}%")
        print("---------------------------------")

        # 6d. まとめ (位置 Hit Rate)
        print(f"--- Summary of Position Hit Rates @{config.TOP_K_EVAL} ---")
        for ts_len in sorted(final_val_metrics.keys()):
            m = final_val_metrics[ts_len]
            print(f"    Len {ts_len: >2} ({m['num_samples']: >3} samples): "
              f"Position Hit: {m['position_hit_rate']:.2f}%")
        
        for ts_len in sorted(test_metrics.keys()):
            m = test_metrics[ts_len]
            print(f"    Len {ts_len: >2} ({m['num_samples']: >3} samples): "
                  f"Position Hit: {m['position_hit_rate']:.2f}%")
        print("---------------------------------")
    else:
        force_print("[WARNING] Test set is empty.")
    
    force_print(f"[INFO] Process completed. Results saved to {run_output_dir}")

if __name__ == "__main__":
    main()