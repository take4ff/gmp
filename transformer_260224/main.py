# python -m transformer_260224.main

# --- main.py ---
# DuckDB対応版メイン処理
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
import time
from datetime import datetime

# ローカルモジュール
from . import config
from .model import HierarchicalTransformer, MultiTaskLoss, FocalLoss
from .train import train_one_epoch
from .evaluate import evaluate
from .utils import (
    force_print, print_config, print_sample_structure,
    save_training_log, save_prediction_results, save_strain_info,
    save_config_copy, save_synonymous_distribution_csv, plot_training_curve, save_confusion_matrix_csv,
    save_metrics_csv, save_category_metrics_csv,
    save_random_baseline_csv, save_region_metrics_csv, save_prediction_distribution_csv,
    plot_metrics_by_timestep, plot_category_metrics,
    plot_combined_val_test_metrics, plot_combined_category_comparison,
    init_wandb, finish_wandb, print_combined_report
)

def set_seed(seed):
    """乱数シードの固定"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def create_dataloader_from_list(data_list, batch_size, shuffle=False):
    """リストからDataLoaderを作成"""
    from .dataset import create_dataloader
    return create_dataloader(data_list, batch_size, shuffle)

def main():
    print(f"PID: {os.getpid()}")
    set_seed(config.SEED)
    force_print(f"Using device: {config.DEVICE}")
    print_config()

    # WandB初期化
    wandb = init_wandb()

    # --- 1. データ準備 ---
    if config.USE_DB:
        # DuckDBからデータを読み込み（メモリ効率版）
        from .db_utils import (
            check_db_exists, print_db_stats, get_db_path,
            create_db_dataloader, get_db_data_info
        )
        
        db_exists, msg = check_db_exists()
        if not db_exists:
            force_print(f"[ERROR] Database not found or invalid: {msg}")
            force_print("[INFO] Run 'python -m transformer_260224.preprocess' first.")
            return
        
        db_path = get_db_path()
        force_print(f"[INFO] Loading data from DuckDB: {db_path}")
        print_db_stats()
        
        # データ情報を取得
        data_info = get_db_data_info(db_path, max_cooccurrence=config.MAX_CO_OCCURRENCE)
        
        # メモリ効率的なDataLoaderを作成
        train_loader = create_db_dataloader(
            db_path=db_path,
            split_type=0,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            max_cooccurrence=config.MAX_CO_OCCURRENCE
        )
        val_loader = create_db_dataloader(
            db_path=db_path,
            split_type=1,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            max_cooccurrence=config.MAX_CO_OCCURRENCE
        )
        test_loader = create_db_dataloader(
            db_path=db_path,
            split_type=2,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            max_cooccurrence=config.MAX_CO_OCCURRENCE
        )
        
        force_print(f"Data loaded: {data_info['train_count']} train, {data_info['val_count']} validation, {data_info['test_count']} test samples.")
    else:
        # 従来のpickleキャッシュ方式
        from .dataset import prepare_all_data
        train, valid, test, data_info = prepare_all_data()
        train_loader = create_dataloader_from_list(train, config.BATCH_SIZE, shuffle=True)
        val_loader = create_dataloader_from_list(valid, config.BATCH_SIZE, shuffle=False)
        test_loader = create_dataloader_from_list(test, config.BATCH_SIZE, shuffle=False)
        force_print(f"Data loaded: {len(train)} train, {len(valid)} validation, {len(test)} test samples.")
    
    force_print(f"Training lengths: {data_info['train_min_len']} to {data_info['train_max_len']}")
    force_print(f"Validation lengths: {data_info['val_min_len']} to {data_info['val_max_len']}")
    force_print(f"Test lengths: {data_info['test_min_len']} to {data_info['test_max_len']}")
    
    # サンプルデータの構造を表示
    if not config.USE_DB and len(train) > 0:
        force_print("[INFO] Displaying sample data structure for verification...")
        print_sample_structure(train[0], sample_idx=0)
    
    # --- 2. モデル・学習準備 ---
    model = HierarchicalTransformer().to(config.DEVICE)

    loss_wrapper = None
    if config.USE_MULTITASK_LOSS:
        force_print("[INFO] Using MultiTaskLoss (Automatic Weighting) - 6 Tasks")
        loss_wrapper = MultiTaskLoss(num_tasks=6).to(config.DEVICE)
    else:
        force_print(f"[INFO] Using Fixed Loss Weights: Region={config.LOSS_WEIGHT_REGION}, Position={config.LOSS_WEIGHT_POSITION}, AA_POS={config.LOSS_WEIGHT_AA_POS}, CODON_POS={config.LOSS_WEIGHT_CODON_POS}, SYNONYMOUS={config.LOSS_WEIGHT_SYNONYMOUS}, Strength={config.LOSS_WEIGHT_STRENGTH}")

    # 損失関数の選択: Focal Loss vs CrossEntropyLoss
    if config.USE_FOCAL_LOSS:
        label_smooth = config.LABEL_SMOOTHING_FACTOR if config.USE_LABEL_SMOOTHING else 0.0
        force_print(f"[INFO] Using Focal Loss (gamma={config.FOCAL_LOSS_GAMMA}, label_smoothing={label_smooth})")
        loss_fn = FocalLoss(
            gamma=config.FOCAL_LOSS_GAMMA,
            reduction='none',
            label_smoothing=label_smooth
        )
    elif config.USE_LABEL_SMOOTHING:
        force_print(f"[INFO] Using CrossEntropyLoss with Label Smoothing (factor={config.LABEL_SMOOTHING_FACTOR})")
        loss_fn = nn.CrossEntropyLoss(reduction='none', label_smoothing=config.LABEL_SMOOTHING_FACTOR)
    else:
        force_print(f"[INFO] Using CrossEntropyLoss")
        loss_fn = nn.CrossEntropyLoss(reduction='none')

    params_to_optimize = list(model.parameters()) + list(loss_wrapper.parameters()) if loss_wrapper else model.parameters()
    optimizer = optim.AdamW(params_to_optimize, lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    
    scheduler = None
    if config.USE_SCHEDULER:
        force_print("[INFO] Using CosineAnnealingLR Scheduler")
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS, eta_min=config.SCHEDULER_ETA_MIN)
    
    # 出力ディレクトリ設定
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = os.path.join(config.RESULT_SAVE_DIR, timestamp)
    os.makedirs(run_output_dir, exist_ok=True)
    best_model_path = os.path.join(run_output_dir, f"best_model.pth")
    
    force_print(f"Starting training... Output dir: {run_output_dir}")
    
    # 株情報の保存
    if not config.USE_DB and config.SAVE_STRAIN_INFO:
        save_strain_info([t[4] for t in train], run_output_dir, prefix="train")
        save_strain_info([t[4] for t in valid], run_output_dir, prefix="valid")
        save_strain_info([t[4] for t in test], run_output_dir, prefix="test")
    
    # 設定ファイルのコピーを保存
    save_config_copy(run_output_dir)
    
    # シノニマス分布を保存
    if not config.USE_DB:
        save_synonymous_distribution_csv(train, valid, test, run_output_dir)

    # --- 3. 学習ループ ---
    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience = config.EARLY_STOPPING_PATIENCE
    training_log = []
    
    for epoch in range(config.EPOCHS):
        epoch_start_time = time.time()
        force_print(f"--- Epoch {epoch+1}/{config.EPOCHS} ---")
        
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, loss_wrapper)
        
        current_lr = optimizer.param_groups[0]['lr']
        if scheduler:
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
        
        # 動的閾値を取得
        strength_thresholds = (data_info.get('strength_low_max', 3), data_info.get('strength_med_max', 5))
        val_loss, val_metrics, _, _ = evaluate(model, val_loader, loss_fn, strength_thresholds)
        
        epoch_time = time.time() - epoch_start_time
        force_print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.6f} | Time: {epoch_time:.1f}s")

        # WandBログ
        if wandb and config.USE_WANDB:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": current_lr,
            })

        training_log.append({'epoch': epoch + 1, 'train_loss': train_loss, 'val_loss': val_loss, 'time_seconds': epoch_time})

        # 簡易メトリクス表示
        for ts_len in sorted(val_metrics.keys()):
            m = val_metrics[ts_len]
            force_print(f"  Len {ts_len:>2}: Region={m['region_hit_rate']:.1f}% | Base Pos={m['position_hit_rate']:.1f}% | AA Pos={m['aa_pos_hit_rate']:.1f}% | Codon={m['codon_pos_hit_rate']:.1f}% | Syn={m['synonymous_hit_rate']:.1f}%")
        
        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            force_print("Validation loss improved. Saving model...")
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'val_loss': val_loss}, best_model_path)
        else:
            epochs_no_improve += 1
            force_print(f"No improvement. Counter: {epochs_no_improve}/{patience}")
            if epochs_no_improve >= patience:
                force_print("Early stopping triggered.")
                break
    
    save_training_log(training_log, run_output_dir)
    plot_training_curve(training_log, run_output_dir)

    # --- 4. 最終評価 ---
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        force_print(f"Loaded best model from epoch {checkpoint['epoch']+1}")
    
    # Validation評価
    force_print(f"Final evaluation on Validation Set...")
    # 動的閾値を使用
    strength_thresholds = (data_info.get('strength_low_max', 3), data_info.get('strength_med_max', 5))
    final_val_loss, final_val_metrics, val_details, val_cat_metrics = evaluate(model, val_loader, loss_fn, strength_thresholds)
    
    if config.SAVE_PREDICTIONS:
        save_prediction_results(val_details, run_output_dir, prefix="valid")
    save_metrics_csv(final_val_metrics, run_output_dir, prefix="valid")
    save_category_metrics_csv(val_cat_metrics, run_output_dir, prefix="valid")
    save_random_baseline_csv(val_details, run_output_dir, prefix="valid")
    save_region_metrics_csv(val_details, run_output_dir, prefix="valid")
    save_prediction_distribution_csv(val_details, run_output_dir, prefix="valid")
    save_confusion_matrix_csv(val_details, run_output_dir, prefix="valid")
    plot_metrics_by_timestep(final_val_metrics, run_output_dir, prefix="valid")
    plot_category_metrics(val_cat_metrics, run_output_dir, prefix="valid")

    # Test評価
    test_loss, test_metrics, test_cat_metrics = 0.0, None, None
    if len(test_loader) > 0:
        force_print(f"Final evaluation on Test Set...")
        test_loss, test_metrics, test_details, test_cat_metrics = evaluate(model, test_loader, loss_fn, strength_thresholds)
        
        if config.SAVE_PREDICTIONS:
            save_prediction_results(test_details, run_output_dir, prefix="test")
        save_metrics_csv(test_metrics, run_output_dir, prefix="test")
        save_category_metrics_csv(test_cat_metrics, run_output_dir, prefix="test")
        save_random_baseline_csv(test_details, run_output_dir, prefix="test")
        save_region_metrics_csv(test_details, run_output_dir, prefix="test")
        save_prediction_distribution_csv(test_details, run_output_dir, prefix="test")
        save_confusion_matrix_csv(test_details, run_output_dir, prefix="test")
        plot_metrics_by_timestep(test_metrics, run_output_dir, prefix="test")
        plot_category_metrics(test_cat_metrics, run_output_dir, prefix="test")
    
    # 統合レポート (Validation vs Test) - 動的閾値を使用
    print_combined_report(val_details, test_details if len(test_loader) > 0 else None, strength_thresholds)
    
    # Val/Test統合比較グラフ
    if len(test_loader) > 0:
        plot_combined_val_test_metrics(final_val_metrics, test_metrics, run_output_dir)
        plot_combined_category_comparison(val_details, test_details, strength_thresholds, run_output_dir)
    
    force_print(f"[INFO] Process completed. Results saved to {run_output_dir}")
    
    # WandB終了（オフラインモードの場合は自動sync）
    finish_wandb(wandb)

if __name__ == "__main__":
    main()