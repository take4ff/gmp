# python -m transformer_260224.run_eval --model_path outputs/transformer_260224/models/best_model.pth

import torch
import torch.nn as nn
import os
import argparse
from datetime import datetime

# ローカルモジュール
from . import config
from .model import HierarchicalTransformer, FocalLoss
from .evaluate import evaluate
from .utils import (
    force_print, print_config,
    save_prediction_results,
    save_config_copy, plot_metrics_by_timestep, plot_category_metrics,
    plot_combined_val_test_metrics, plot_combined_category_comparison,
    print_combined_report, save_metrics_csv, save_category_metrics_csv,
    save_random_baseline_csv, save_region_metrics_csv,
    save_prediction_distribution_csv, save_confusion_matrix_csv
)

def main():
    parser = argparse.ArgumentParser(description="Evaluate a pre-trained HierarchicalTransformer model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model checkpoint (.pth)")
    parser.add_argument("--out_dir", type=str, default=None, help="Directory to save results (default: outputs/results/eval_YYYYMMDD_HHMMSS)")
    args = parser.parse_args()

    force_print(f"Using device: {config.DEVICE}")
    print_config()

    # --- 1. データ準備 ---
    if config.USE_DB:
        from .db_utils import (
            check_db_exists, print_db_stats, get_db_path,
            create_db_dataloader, get_db_data_info
        )
        
        db_exists, msg = check_db_exists()
        if not db_exists:
            force_print(f"[ERROR] Database not found or invalid: {msg}")
            return
        
        db_path = get_db_path()
        force_print(f"[INFO] Loading data from DuckDB: {db_path}")
        print_db_stats()
        
        data_info = get_db_data_info(db_path, max_cooccurrence=config.MAX_CO_OCCURRENCE)
        
        val_loader = create_db_dataloader(
            db_path=db_path, split_type=1, batch_size=config.BATCH_SIZE, shuffle=False,
            max_cooccurrence=config.MAX_CO_OCCURRENCE
        )
        test_loader = create_db_dataloader(
            db_path=db_path, split_type=2, batch_size=config.BATCH_SIZE, shuffle=False,
            max_cooccurrence=config.MAX_CO_OCCURRENCE
        )
        force_print(f"Data loaded: {data_info['val_count']} validation, {data_info['test_count']} test samples.")
    else:
        from .dataset import prepare_all_data, create_dataloader
        _, valid, test, data_info = prepare_all_data()
        val_loader = create_dataloader(valid, config.BATCH_SIZE, shuffle=False)
        test_loader = create_dataloader(test, config.BATCH_SIZE, shuffle=False)
        force_print(f"Data loaded: {len(valid)} validation, {len(test)} test samples.")

    # --- 2. モデル準備 ---
    model = HierarchicalTransformer().to(config.DEVICE)
    
    if not os.path.exists(args.model_path):
        force_print(f"[ERROR] Model path not found: {args.model_path}")
        return
        
    checkpoint = torch.load(args.model_path, map_location=config.DEVICE)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        force_print(f"Loaded model state dict from {args.model_path}")
    else:
        model.load_state_dict(checkpoint)
        force_print(f"Loaded model from {args.model_path} (direct state dict)")

    # 損失関数の準備 (評価用)
    if config.USE_FOCAL_LOSS:
        label_smooth = config.LABEL_SMOOTHING_FACTOR if config.USE_LABEL_SMOOTHING else 0.0
        loss_fn = FocalLoss(gamma=config.FOCAL_LOSS_GAMMA, reduction='none', label_smoothing=label_smooth)
    elif config.USE_LABEL_SMOOTHING:
        loss_fn = nn.CrossEntropyLoss(reduction='none', label_smoothing=config.LABEL_SMOOTHING_FACTOR)
    else:
        loss_fn = nn.CrossEntropyLoss(reduction='none')

    # 出力ディレクトリ設定
    if args.out_dir:
        run_output_dir = args.out_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_output_dir = os.path.join(config.RESULT_SAVE_DIR, f"eval_{timestamp}")
    
    os.makedirs(run_output_dir, exist_ok=True)
    save_config_copy(run_output_dir)
    force_print(f"Evaluation output dir: {run_output_dir}")

    # --- 3. 評価実行 ---
    strength_thresholds = (data_info.get('strength_low_max', 3), data_info.get('strength_med_max', 5))
    
    # Validation評価
    force_print(f"\nFinal evaluation on Validation Set...")
    val_loss, val_metrics, val_details, val_cat_metrics = evaluate(model, val_loader, loss_fn, strength_thresholds)
    
    if config.SAVE_PREDICTIONS:
        save_prediction_results(val_details, run_output_dir, prefix="valid")
    save_metrics_csv(val_metrics, run_output_dir, prefix="valid")
    save_category_metrics_csv(val_cat_metrics, run_output_dir, prefix="valid")
    save_random_baseline_csv(val_details, run_output_dir, prefix="valid")
    save_region_metrics_csv(val_details, run_output_dir, prefix="valid")
    save_prediction_distribution_csv(val_details, run_output_dir, prefix="valid")
    save_confusion_matrix_csv(val_details, run_output_dir, prefix="valid")
    plot_metrics_by_timestep(val_metrics, run_output_dir, prefix="valid")
    plot_category_metrics(val_cat_metrics, run_output_dir, prefix="valid")

    # Test評価
    test_details = None
    test_metrics = None
    if len(test_loader) > 0:
        force_print(f"\nFinal evaluation on Test Set...")
        _, test_metrics, test_details, test_cat_metrics = evaluate(model, test_loader, loss_fn, strength_thresholds)
        
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
    
    # 統合レポート
    print_combined_report(val_details, test_details, strength_thresholds)
    
    # Val/Test統合比較グラフ
    if test_details:
        plot_combined_val_test_metrics(val_metrics, test_metrics, run_output_dir)
        plot_combined_category_comparison(val_details, test_details, strength_thresholds, run_output_dir)
    
    force_print(f"\n[INFO] Evaluation completed. Results saved to {run_output_dir}")

if __name__ == "__main__":
    main()
