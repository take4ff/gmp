# 251114より使用データ全件の特徴量を生成(バッチ化)してからデータ分割、予測精度は低め？
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import random
import os
from torch.amp import GradScaler
from datetime import datetime
from torch.utils.data import DataLoader

# 各モジュールをインポート
from . import config
from . import dataset
from . import model
from . import train
from . import evaluate
from . import utils

def set_seed(seed):
    """乱数シードを固定する関数"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def process_in_batches(df, func, batch_size, *args):
    """バッチ処理用のヘルパー関数"""
    results = []
    for i in range(0, len(df), batch_size):
        utils.force_print(f"  Processing batch {i//batch_size + 1}/{(len(df)-1)//batch_size + 1}...")
        batch_df = df.iloc[i:i+batch_size].copy()
        results.append(func(batch_df, *args))
    
    # バッチの結果を結合
    combined_results = {}
    if results:
        for key in results[0]:
            combined_results[key] = [item for res in results for item in res[key]]
    return combined_results

def main():
    # --- 1. セットアップ ---
    set_seed(config.SEED)
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # タイムスタンプ付きの出力ディレクトリを作成
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = os.path.join(config.OUTPUT_DIR, timestamp)
    os.makedirs(run_output_dir, exist_ok=True)
    utils.force_print(f"Output will be saved to: {run_output_dir}")

    # 設定に基づいたキャッシュファイルパスを生成
    utils.print_config()
    config_hash = utils.get_config_hash()
    cache_path = os.path.join(config.CACHE_DIR, f"data_optimized_{config_hash}.pkl")
    
    # --- 2. データ準備（効率化版） ---
    cached_data = utils.load_cache(cache_path)
    if cached_data:
        train_data, valid_data, test_data, metadata = cached_data
        utils.force_print("--- Data Summary (from cache) ---")
        for key, value in metadata.items():
            utils.force_print(f"{key}: {value}")
        utils.force_print("-----------------------------------\n")
    else:
        utils.force_print("--- 1. Starting Optimized Data Pre-processing ---")
        # 補助データをロード
        df_codon, df_freq, df_dissimilarity = dataset.load_aux_data()

        # 生データをロード
        df_raw = dataset.import_strains(config.USHER_DIR, config.MAX_STRAIN_NUM)
        
        # フィルタリングとサンプリング
        utils.force_print("--- 1.1. Data Filtering ---")
        df_filtered = dataset.filter_by_max_co_occurrence(df_raw, config.MAX_CO_OCCUR)
        df_unique = dataset.filter_unique_paths(df_filtered)
        df_filtered = dataset.sort_strain_by_num_and_filter_strain(df_unique, config.MAX_TOP_STRAINS)
        df_filtered = dataset.sample_by_total_count(df_filtered, config.MAX_TOTAL_SAMPLES, config.SEED)
        df_filtered = dataset.filter_num_per_strain(df_filtered, config.MAX_NUM_PER_STRAIN, config.SEED)
        
        # --- 2. 効率化: 特徴量生成を分割前に実行 ---
        utils.force_print("--- 1.2. Generating Features (Before Split) ---")
        batch_size = config.BATCH_SIZE_FEATURE_GEN
        utils.force_print(f"Processing {len(df_filtered)} samples with batch size {batch_size}...")
        
        # 一括で特徴量生成
        all_features = process_in_batches(
            df_filtered, 
            dataset.process_feature_batch_optimized, 
            batch_size, 
            df_codon, 
            df_freq, 
            df_dissimilarity
        )
        
        # 特徴量生成後にDataFrameに変換
        utils.force_print("--- 1.3. Converting to DataFrame ---")
        features_df = pd.DataFrame({
            'input_cat_seq': all_features['input_cat_seq'],
            'input_num_seq': all_features['input_num_seq'], 
            'target_protein': all_features['target_protein'],
            'target_pos': all_features['target_pos'],
            'original_len': all_features['original_len']
        })
        
        # --- 3. 特徴量生成後にtrain/valid/test分割 ---
        utils.force_print("--- 1.4. Splitting Features into Train/Valid/Test ---")
        train_data, valid_data, test_data = dataset.split_features_by_length(
            features_df, 
            config.TRAIN_MAX_LEN, 
            config.VALID_NUM, 
            config.VALID_RATIO, 
            config.SEED
        )

        # サマリー情報を作成
        metadata = {
            "Total unique samples": len(df_unique),
            "Total features generated": len(features_df),
            "Train records": len(train_data['target_protein']),
            "Validation records": len(valid_data['target_protein']),
            "Test records": len(test_data['target_protein']),
        }
        
        utils.force_print("--- Data Summary ---")
        for key, value in metadata.items():
            utils.force_print(f"{key}: {value}")
        utils.force_print("--------------------\n")

        # キャッシュに保存
        utils.save_cache((train_data, valid_data, test_data, metadata), cache_path)

    # パス長別の件数を表示
    utils.print_path_length_distribution(train_data, "Train", utils)
    utils.print_path_length_distribution(valid_data, "Validation", utils)
    utils.print_path_length_distribution(test_data, "Test", utils)

    # --- 3. データローダー作成 ---
    utils.force_print("--- 2. Creating Dataloaders ---")
    train_loader = dataset.create_dataloader(train_data, config.BATCH_SIZE, shuffle=True)
    valid_loader = dataset.create_dataloader(valid_data, config.BATCH_SIZE, shuffle=False)
    test_loader = dataset.create_dataloader(test_data, config.BATCH_SIZE, shuffle=False)
    utils.force_print("Dataloaders created successfully.\n")

    # --- 4. モデル、損失関数、オプティマイザ準備 ---
    utils.force_print("--- 3. Initializing Model, Optimizer, etc. ---")
    device = torch.device(config.DEVICE)
    current_model = model.HierarchicalTransformer().to(device)
    optimizer = optim.AdamW(current_model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss()
    scaler = GradScaler()
    utils.force_print("Initialization complete.\n")

    # デバッグ表示
    if config.DEBUG_PRINT_BATCH:
        utils.force_print("--- Debug Information ---")
        utils.force_print("Sample data structure:")
        utils.force_print(f"input_cat_seq (first sample): {train_data['input_cat_seq'][0]}")
        utils.force_print(f"input_num_seq (first sample): {train_data['input_num_seq'][0]}")
        utils.force_print(f"target_protein: {train_data['target_protein'][0]}")
        utils.force_print(f"target_pos: {train_data['target_pos'][0]}")
        utils.force_print(f"original_len: {train_data['original_len'][0]}")
        utils.force_print("-----------------------------\n")

    # --- 5. 訓練ループ ---
    utils.force_print("--- 4. Starting Training Loop ---")
    best_valid_loss = float('inf')
    epochs_no_improve = 0
    best_model_path = os.path.join(run_output_dir, f"best_model_{config_hash}.pth")

    # 履歴を保存するリスト
    history = {
        'train_loss': [],
        'valid_loss': [],
        'valid_acc_protein': [],
        'valid_acc_pos': []
    }

    for epoch in range(config.EPOCHS):
        utils.force_print(f"--- Epoch {epoch + 1}/{config.EPOCHS} ---")
        
        train_loss = train.train_one_epoch(current_model, train_loader, optimizer, loss_fn, scaler)
        valid_metrics = evaluate.evaluate(current_model, valid_loader, loss_fn)

        # 履歴に追加
        history['train_loss'].append(train_loss)
        history['valid_loss'].append(valid_metrics['loss'])
        history['valid_acc_protein'].append(valid_metrics[f'protein_top{config.TOP_K_EVAL}_acc'])
        history['valid_acc_pos'].append(valid_metrics[f'position_top{config.TOP_K_EVAL}_acc'])
        
        valid_loss = valid_metrics['loss']
        valid_acc_protein = valid_metrics[f'protein_top{config.TOP_K_EVAL}_acc']
        valid_acc_pos = valid_metrics[f'position_top{config.TOP_K_EVAL}_acc']

        utils.force_print(f"Train Loss: {train_loss:.4f}")
        utils.force_print(f"Valid Loss: {valid_loss:.4f} | Valid Acc (Protein): {valid_acc_protein:.2f}% | Valid Acc (Pos): {valid_acc_pos:.2f}%")

        # 早期終了とモデル保存
        if valid_loss < best_valid_loss:
            utils.force_print(f"Validation loss improved ({best_valid_loss:.4f} --> {valid_loss:.4f}). Saving model...")
            best_valid_loss = valid_loss
            utils.save_model(current_model, optimizer, epoch, best_valid_loss, best_model_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            utils.force_print(f"Validation loss did not improve. Early stopping counter: {epochs_no_improve}/{config.EARLY_STOPPING_PATIENCE}")
            if epochs_no_improve >= config.EARLY_STOPPING_PATIENCE:
                utils.force_print("Early stopping triggered.")
                break

    # --- 6. 最終評価 ---
    utils.force_print("--- 6. Final Evaluation on Test Set ---")

    # グラフと設定ファイルを保存
    utils.save_plots(history, run_output_dir)
    utils.save_config(run_output_dir)

    if os.path.exists(best_model_path):
        # ベストモデルをロード
        checkpoint = torch.load(best_model_path)
        current_model.load_state_dict(checkpoint['model_state_dict'])
        utils.force_print("Best model loaded for final evaluation.")
        
        # --- テストデータでの評価 ---
        test_metrics = evaluate.evaluate(current_model, test_loader, loss_fn)
        utils.force_print("--- Test Results ---")
        for key, value in test_metrics.items():
            utils.force_print(f"{key}: {value:.4f}")
        utils.force_print("--------------------\n")

        # --- タイムステップごとの評価 (Valid + Test) ---
        utils.force_print(f"--- 7. Evaluating by Timestep (Valid + Test) ---")
        if best_model_path:
            # valid_dataとtest_dataを結合
            combined_data = {
                key: np.concatenate((valid_data[key], test_data[key]), axis=0)
                if isinstance(valid_data[key], np.ndarray) else valid_data[key] + test_data[key]
                for key in valid_data
            }
            combined_loader = dataset.create_dataloader(combined_data, config.BATCH_SIZE, shuffle=False)

            # 評価
            df_preds = evaluate.evaluate_by_timestep(current_model, combined_loader)
            if not df_preds.empty:
                # 正解かどうかを判定
                df_preds['protein_correct'] = df_preds.apply(lambda row: row['true_protein'] in row['pred_protein_topk'], axis=1)
                df_preds['position_correct'] = df_preds.apply(lambda row: row['true_pos'] in row['pred_pos_topk'], axis=1)

                # タイムステップごとにグループ化して精度を計算
                df_timestep_acc = df_preds.groupby('original_len').agg(
                    protein_accuracy=('protein_correct', 'mean'),
                    position_accuracy=('position_correct', 'mean')
                ).reset_index()
                df_timestep_acc['protein_accuracy'] *= 100
                df_timestep_acc['position_accuracy'] *= 100

                # 結果を保存
                utils.save_timestep_evaluation(df_timestep_acc, run_output_dir)
            else:
                utils.force_print("[WARNING] No data for timestep evaluation.")
        else:
            utils.force_print("[WARNING] No best model found to evaluate on the test set.")

    else:
        utils.force_print("[WARNING] No best model found to evaluate on the test set.")


if __name__ == "__main__":
    main()