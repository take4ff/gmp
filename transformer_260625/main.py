# --- main.py ---
# DuckDB対応版メイン処理
# Usage: nohup python -m transformer_260625.main > nohup0.out 2>&1 & 

import torch
import torch.optim as optim
import numpy as np
import random
import os
import time
from datetime import datetime

# ローカルモジュール
from . import config
from .model import HierarchicalTransformer, MultiTaskLoss
from .train import train_one_epoch
from .evaluate import evaluate, evaluate_topk
from .utils.losses import build_loss_fn
from .utils.logging import force_print, print_config, print_sample_structure, init_wandb, finish_wandb
from .utils.io import (
    save_training_log, save_prediction_results, save_strain_info,
    save_config_copy, save_synonymous_distribution_csv,
    save_metrics_csv, save_category_metrics_csv, save_combined_metrics_csv,
    save_region_metrics_csv,
    save_per_position_recall_csv,
    save_val_test_gap_csv,
    save_error_analysis_csv,
    save_strain_metrics_csv,
    save_early_stopping_json,
    save_model_summary_txt,
    save_topk_precision_csv,
    save_date_metrics_csv,
    save_strength_fine_csv,
    save_lineage_metrics_csv,
    save_run_summary_json,
    save_r_precision_csv,
    save_random_baseline_csv,
    save_prediction_distribution_csv,
    save_calibration_csv,
)
from .utils.plotting import (
    plot_training_curve, plot_metrics_by_timestep, plot_category_metrics,
    plot_combined_val_test_metrics, plot_combined_category_comparison,
    print_combined_report,
    plot_mutation_dist,
    load_position_to_region,
    plot_base_embedding_network,
    plot_position_threshold_network,
    plot_position_tsne,
    plot_position_cluster_network,
    plot_major_mutation_network,
    plot_strength_calibration,
    plot_per_position_recall,
    plot_topk_precision,
    plot_attention_heatmap,
    plot_metrics_by_date,
    plot_strength_fine,
    plot_lineage_metrics,
    plot_calibration,
    plot_region_metrics,
    plot_r_precision,
)


# ──────────────────────────────────────────────
# 1. ユーティリティ
# ──────────────────────────────────────────────

def set_seed(seed):
    """乱数シードを固定する。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


# ──────────────────────────────────────────────
# 2. データ準備
# ──────────────────────────────────────────────

def prepare_data():
    """設定に応じてデータローダーと統計情報を返す。

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader, dict]:
            (train_loader, val_loader, test_loader, data_info)
    """
    if config.USE_DB:
        from .db.connection import check_db_exists, print_db_stats, get_db_path, run_db_migrations
        from .db.dataset import create_db_dataloader
        from .db.queries import get_db_data_info

        db_exists, msg = check_db_exists()
        if not db_exists:
            force_print(f"[ERROR] Database not found or invalid: {msg}")
            force_print(f"[INFO] Run 'python -m {__package__}.preprocess' first.")
            raise RuntimeError("Database not found.")

        db_path = get_db_path()

        # スキーマ拡張のマイグレーションを自動実行（並列実行時のロック競合を避けるためコメントアウト）
        # run_db_migrations(db_path)

        force_print(f"[INFO] Loading data from DuckDB: {db_path}")
        print_db_stats()

        data_info = get_db_data_info(db_path, max_cooccurrence=config.MAX_CO_OCCURRENCE)

        from .db.queries import get_combined_sampled_strength
        combined_strength = None
        if not getattr(config, 'STRENGTH_SCORE_FROM_CSV', True):
            combined_strength = get_combined_sampled_strength(db_path)

        def _make_loader(split_type, shuffle):
            return create_db_dataloader(
                db_path=db_path, split_type=split_type,
                batch_size=config.BATCH_SIZE, shuffle=shuffle,
                max_cooccurrence=config.MAX_CO_OCCURRENCE,
                strain_to_strength=combined_strength,
            )

        train_loader = _make_loader(0, shuffle=True)
        val_loader   = _make_loader(1, shuffle=False)
        test_loader  = _make_loader(2, shuffle=False)

        force_print(
            f"Data loaded: {data_info['train_count']} train, "
            f"{data_info['val_count']} validation, {data_info['test_count']} test samples."
        )

        # SPLIT_MODE='date'/'walk_forward' かつ FORCE_DATE_REASSIGN=True のとき split を再計算
        if (getattr(config, 'SPLIT_MODE', 'timestep') in ('date', 'walk_forward')
                and getattr(config, 'FORCE_DATE_REASSIGN', False)):
            _split_mode = getattr(config, 'SPLIT_MODE', 'timestep')
            if _split_mode == 'walk_forward':
                force_print("[INFO] FORCE_DATE_REASSIGN=True: re-assigning walk_forward splits (split_type_wf)...")
                from .db.queries import assign_wf_splits
                from .db.connection import connect_db
                _con = connect_db(db_path)
                assign_wf_splits(_con)
            else:
                force_print("[INFO] FORCE_DATE_REASSIGN=True: re-assigning date splits (split_type_date)...")
                from .db.queries import assign_date_splits
                from .db.connection import connect_db
                _con = connect_db(db_path)
                assign_date_splits(_con)
            _con.close()
            # 再分割後に data_info を更新
            data_info = get_db_data_info(db_path, max_cooccurrence=config.MAX_CO_OCCURRENCE)
            force_print(
                f"[INFO] After re-assign: {data_info['train_count']} train, "
                f"{data_info['val_count']} valid, {data_info['test_count']} test"
            )
            # 再生成した分割に基づいてデータローダーも作り直す
            train_loader = _make_loader(0, shuffle=True)
            val_loader   = _make_loader(1, shuffle=False)
            test_loader  = _make_loader(2, shuffle=False)

        # Curriculum Learning 用: 訓練ローダーを再生成するラムダ
        def make_train_loader(max_length=None):
            return create_db_dataloader(
                db_path=db_path, split_type=0,
                batch_size=config.BATCH_SIZE, shuffle=True,
                max_cooccurrence=config.MAX_CO_OCCURRENCE,
                max_length=max_length,
                strain_to_strength=combined_strength,
            )

        return train_loader, val_loader, test_loader, data_info, None, None, None, make_train_loader

    else:
        # 従来の pickle キャッシュ方式
        from .db.feature import prepare_all_data, create_dataloader

        train, valid, test, data_info = prepare_all_data()
        train_loader = create_dataloader(train, config.BATCH_SIZE, shuffle=True)
        val_loader   = create_dataloader(valid, config.BATCH_SIZE, shuffle=False)
        test_loader  = create_dataloader(test,  config.BATCH_SIZE, shuffle=False)
        force_print(
            f"Data loaded: {len(train)} train, {len(valid)} validation, {len(test)} test samples."
        )
        return train_loader, val_loader, test_loader, data_info, train, valid, test, None



# ──────────────────────────────────────────────
# 3. モデル・損失・最適化器の構築
# ──────────────────────────────────────────────

def build_components(class_counts_dict=None):
    """モデル・損失関数・最適化器・スケジューラを構築して返す。

    Returns:
        Tuple[nn.Module, nn.Module/dict, Optional[MultiTaskLoss], Optimizer, Optional[Scheduler]]
    """
    model = HierarchicalTransformer().to(config.DEVICE)

    # Walk-forward: 直前フォールドの fine-tuned 重みをロード（Fold 2 以降）
    wf_prev_ckpt = getattr(config, 'WF_PREV_FOLD_CHECKPOINT', None)
    if wf_prev_ckpt and os.path.exists(wf_prev_ckpt):
        ckpt = torch.load(wf_prev_ckpt, map_location=config.DEVICE, weights_only=True)
        missing, unexpected = model.load_state_dict(ckpt['model_state_dict'], strict=False)
        force_print(f"[INFO] Loaded previous fold weights from {wf_prev_ckpt}")
        if missing:
            force_print(f"[INFO]   Missing keys : {missing}")
        if unexpected:
            force_print(f"[INFO]   Unexpected keys: {unexpected}")
    # 事前学習済み backbone 重みのロード（Fold 1 or 非 walk_forward）
    elif getattr(config, 'USE_PRETRAINING', False):
        mode = getattr(config, 'PRETRAINING_MODE', 'mlm').lower()
        pretrain_path = os.path.join(
            config.OUTPUT_DIR, 'pretrain', f'pretrain_{mode}_final.pth'
        )
        if os.path.exists(pretrain_path):
            ckpt = torch.load(pretrain_path, map_location=config.DEVICE, weights_only=True)
            missing, unexpected = model.load_state_dict(
                ckpt['backbone_state_dict'], strict=False
            )
            force_print(f"[INFO] Loaded pretrained backbone ({mode}) from {pretrain_path}")
            if missing:
                force_print(f"[INFO]   Missing keys : {missing}")
            if unexpected:
                force_print(f"[INFO]   Unexpected keys: {unexpected}")
        else:
            force_print(f"[WARNING] USE_PRETRAINING=True but checkpoint not found: {pretrain_path}")
            force_print("[WARNING]   Falling back to random initialization.")

    loss_wrapper = None
    if config.USE_MULTITASK_LOSS:
        force_print("[INFO] Using MultiTaskLoss (Automatic Weighting) - 6 Tasks")
        loss_wrapper = MultiTaskLoss(num_tasks=6).to(config.DEVICE)
    else:
        force_print(
            f"[INFO] Using Fixed Loss Weights: "
            f"Region={config.LOSS_WEIGHT_REGION}, Position={config.LOSS_WEIGHT_POSITION}, "
            f"AA_POS={config.LOSS_WEIGHT_AA_POS}, CODON_POS={config.LOSS_WEIGHT_CODON_POS}, "
            f"SYNONYMOUS={config.LOSS_WEIGHT_SYNONYMOUS}, Strength={config.LOSS_WEIGHT_STRENGTH}"
        )

    loss_fn = build_loss_fn(class_counts_dict)

    params = (
        list(model.parameters()) + list(loss_wrapper.parameters())
        if loss_wrapper else model.parameters()
    )
    opt_type = getattr(config, 'OPTIMIZER_TYPE', 'adamw').lower()
    if opt_type == 'adamw':
        optimizer = optim.AdamW(
            params,
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
            betas=(config.OPTIMIZER_BETA1, config.OPTIMIZER_BETA2)
        )
    elif opt_type == 'adam':
        optimizer = optim.Adam(
            params,
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
            betas=(config.OPTIMIZER_BETA1, config.OPTIMIZER_BETA2)
        )
    elif opt_type == 'sgd':
        optimizer = optim.SGD(
            params,
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
            momentum=getattr(config, 'OPTIMIZER_SGD_MOMENTUM', 0.9)
        )
    else:
        raise ValueError(f"Unknown OPTIMIZER_TYPE: {opt_type}")

    scheduler = None
    if config.USE_SCHEDULER:
        use_warmup   = getattr(config, 'USE_WARMUP', False)
        warmup_epochs = getattr(config, 'WARMUP_EPOCHS', 2)
        if use_warmup and warmup_epochs > 0:
            # LinearWarmup: LR を 1/warmup_epochs → 1.0 に線形増加
            # CosineAnnealing: その後 SCHEDULER_ETA_MIN まで余弦減衰
            warmup_scheduler = optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1.0 / max(warmup_epochs, 1),
                end_factor=1.0,
                total_iters=warmup_epochs,
            )
            cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(config.EPOCHS - warmup_epochs, 1),
                eta_min=config.SCHEDULER_ETA_MIN,
            )
            scheduler = optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_epochs],
            )
            force_print(
                f"[INFO] Using LinearWarmup({warmup_epochs} epochs) + CosineAnnealingLR Scheduler"
            )
        else:
            force_print("[INFO] Using CosineAnnealingLR Scheduler")
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config.EPOCHS, eta_min=config.SCHEDULER_ETA_MIN
            )

    return model, loss_fn, loss_wrapper, optimizer, scheduler


# ──────────────────────────────────────────────
# 4. 学習ループ
# ──────────────────────────────────────────────

def run_training(model, train_loader, val_loader, loss_fn, loss_wrapper,
                 optimizer, scheduler, run_output_dir, data_info, wandb,
                 strength_thresholds, make_train_loader=None):
    """学習ループを実行し、best_model を保存する。

    Args:
        make_train_loader: Curriculum Learning 用なローダー再生成関数。
                           None の場合は Curriculum Learning を無効化。
    Returns:
        Tuple[list, str]: (training_log, best_model_path)
    """
    best_model_path = os.path.join(run_output_dir, "models", "best_model.pth")
    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
    early_stop_metric = getattr(config, "EARLY_STOPPING_METRIC", "val_loss")
    early_stop_mode   = getattr(config, "EARLY_STOPPING_MODE", "min")

    # min モードなら初期値 inf, max モードなら初期値 -inf
    best_metric_val = float('inf') if early_stop_mode == 'min' else float('-inf')
    epochs_no_improve = 0
    patience = config.EARLY_STOPPING_PATIENCE
    training_log = []

    for epoch in range(config.EPOCHS):
        epoch_start_time = time.time()
        force_print(f"--- Epoch {epoch+1}/{config.EPOCHS} ---")

        # Curriculum Learning: エポックに応じて訓練ローダーを再構築
        if getattr(config, 'CURRICULUM_LEARNING', False) and make_train_loader is not None:
            warmup = getattr(config, 'CURRICULUM_WARMUP_EPOCHS', 5)
            init_len = getattr(config, 'CURRICULUM_MAX_LEN_INIT', 5)
            if epoch < warmup:
                # 線形補間: init_len → TRAIN_MAX（短いパスのみ → 全長パスを解禁）
                frac = epoch / max(warmup - 1, 1)
                curr_max_len = int(init_len + (config.TRAIN_MAX - init_len) * frac)
                curr_max_len = min(curr_max_len, config.TRAIN_MAX)
            else:
                curr_max_len = None  # 全データ使用
            train_loader = make_train_loader(max_length=curr_max_len)
            force_print(f"[Curriculum] Epoch {epoch+1}: max_length={curr_max_len}")

        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, loss_wrapper)

        current_lr = optimizer.param_groups[0]['lr']  # このエポックに実際に使用した LR
        if scheduler:
            scheduler.step()

        val_loss, val_metrics, _, _, _, _ = evaluate(model, val_loader, loss_fn, strength_thresholds)

        epoch_time = time.time() - epoch_start_time
        force_print(
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"LR: {current_lr:.6f} | Time: {epoch_time:.1f}s"
        )

        if wandb and config.USE_WANDB:
            wandb.log({
                "epoch": epoch + 1, "train_loss": train_loss,
                "val_loss": val_loss, "learning_rate": current_lr,
            })

        training_log.append({
            'epoch': epoch + 1, 'train_loss': train_loss,
            'val_loss': val_loss, 'time_seconds': epoch_time,
        })

        # エポックごとの簡易メトリクス表示
        for ts_len in sorted(val_metrics.keys()):
            m = val_metrics[ts_len]
            force_print(
                f"  Len {ts_len:>2}: Region={m['region_hit_rate']:.1f}% | "
                f"Base Pos={m['position_hit_rate']:.1f}% | "
                f"AA Pos={m['aa_pos_hit_rate']:.1f}% | "
                f"Codon={m['codon_pos_hit_rate']:.1f}% | "
                f"Syn={m['synonymous_hit_rate']:.1f}%"
            )

        # Early Stopping 判定用のメトリクス抽出
        if early_stop_metric == "val_loss":
            current_metric = val_loss
        else:
            total_samples = sum(m['num_samples'] for m in val_metrics.values())
            if total_samples > 0:
                current_metric = sum(m.get(early_stop_metric.replace('val_', ''), 0) * m['num_samples']
                                     for m in val_metrics.values()) / total_samples
            else:
                current_metric = val_loss

        # 改善したかの判定
        is_improved = False
        if early_stop_mode == 'min' and current_metric < best_metric_val:
            is_improved = True
        elif early_stop_mode == 'max' and current_metric > best_metric_val:
            is_improved = True

        # Early Stopping 処理
        if is_improved:
            best_metric_val = current_metric
            epochs_no_improve = 0
            force_print(f"Validation metric '{early_stop_metric}' improved. Saving model...")
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

    return training_log, best_model_path


# ──────────────────────────────────────────────
# 5. 最終評価・結果保存
# ──────────────────────────────────────────────

def run_final_evaluation(model, val_loader, test_loader, loss_fn,
                         run_output_dir, data_info, strength_thresholds):
    """Validation・Test セットで最終評価を行い、全CSVとグラフを保存する。

    Returns:
        Tuple[dict, dict, list, dict, list]: (val_metrics, test_metrics, val_details, test_details, strength_thresholds)
    """

    def _save_results(metrics, cat_metrics, details, prefix):
        if config.SAVE_PREDICTIONS:
            save_prediction_results(details, run_output_dir, prefix=prefix)
        df_ts = save_metrics_csv(metrics, run_output_dir, prefix=prefix, save=False)
        df_cat = save_category_metrics_csv(cat_metrics, run_output_dir, prefix=prefix, save=False)
        save_strength_fine_csv(details, run_output_dir, prefix=prefix)
        save_lineage_metrics_csv(details, run_output_dir, prefix=prefix)
        save_region_metrics_csv(details, run_output_dir, prefix=prefix)
        save_per_position_recall_csv(details, run_output_dir, prefix=prefix)
        save_error_analysis_csv(details, run_output_dir, prefix=prefix)
        save_random_baseline_csv(details, run_output_dir, prefix=prefix)
        save_prediction_distribution_csv(details, run_output_dir, prefix=prefix)
        if getattr(config, 'SAVE_STRAIN_METRICS', False):
            save_strain_metrics_csv(details, run_output_dir, prefix=prefix)
        plot_metrics_by_timestep(metrics, run_output_dir, prefix=prefix)
        plot_category_metrics(cat_metrics, run_output_dir, prefix=prefix)
        plot_strength_calibration(details, run_output_dir, prefix=prefix)
        plot_per_position_recall(details, run_output_dir, prefix=prefix, top_n=getattr(config, "PLOT_TOP_N_POSITIONS", 40))
        plot_strength_fine(details, run_output_dir, prefix=prefix)
        plot_lineage_metrics(details, run_output_dir, prefix=prefix,
                             top_n=getattr(config, 'PLOT_TOP_N_LINEAGES', 30))
        plot_region_metrics(details, run_output_dir, prefix=prefix)
        return df_ts, df_cat

    # Validation
    force_print("Final evaluation on Validation Set...")
    val_loss, val_metrics, val_details, val_cat_metrics, val_metrics_ym, val_calib_bins = evaluate(
        model, val_loader, loss_fn, strength_thresholds
    )
    val_df_ts, val_df_cat = _save_results(val_metrics, val_cat_metrics, val_details, prefix="valid")
    if val_calib_bins:
        save_calibration_csv(val_calib_bins, run_output_dir, prefix="valid")
        plot_calibration(val_calib_bins, run_output_dir, prefix="valid")

    # Test
    test_loss = None
    test_metrics, test_details, test_cat_metrics = None, None, None
    test_df_ts, test_df_cat = None, None
    test_metrics_ym = {}
    if len(test_loader) > 0:
        force_print("Final evaluation on Test Set...")
        test_loss, test_metrics, test_details, test_cat_metrics, test_metrics_ym, test_calib_bins = evaluate(
            model, test_loader, loss_fn, strength_thresholds
        )
        test_df_ts, test_df_cat = _save_results(test_metrics, test_cat_metrics, test_details, prefix="test")
        if test_calib_bins:
            save_calibration_csv(test_calib_bins, run_output_dir, prefix="test")
            plot_calibration(test_calib_bins, run_output_dir, prefix="test")

    # 統合レポート
    print_combined_report(val_details, test_details, strength_thresholds)
    if test_details:
        plot_combined_val_test_metrics(val_metrics, test_metrics, run_output_dir)
        plot_combined_category_comparison(val_details, test_details, strength_thresholds, run_output_dir)
        save_val_test_gap_csv(val_metrics, test_metrics, run_output_dir)

        # 結合したメトリクスCSVを出力（timestepでソート）
        save_combined_metrics_csv(val_df_ts, test_df_ts, run_output_dir, 'combined_metrics_by_timestep.csv')
        save_combined_metrics_csv(val_df_cat, test_df_cat, run_output_dir, 'combined_metrics_by_category.csv')

    # 月別 CSV とグラフを常に出力
    save_date_metrics_csv(val_metrics_ym,  run_output_dir, prefix='valid')
    save_date_metrics_csv(test_metrics_ym, run_output_dir, prefix='test')
    plot_metrics_by_date(val_metrics_ym, test_metrics_ym, run_output_dir)

    # Top-K 評価 (Validation)
    eval_ks = getattr(config, "EVAL_TOP_KS", (1, 3, 5))
    force_print(f"[INFO] Evaluating Top-K precision (k={eval_ks}) on Validation...")
    val_topk = evaluate_topk(model, val_loader, ks=eval_ks)
    save_topk_precision_csv(val_topk, run_output_dir, prefix='valid')
    plot_topk_precision(val_topk, run_output_dir, prefix='valid')
    if getattr(config, 'USE_R_PRECISION', False):
        save_r_precision_csv(val_topk, run_output_dir, prefix='valid')
        plot_r_precision(val_topk, run_output_dir, prefix='valid')

    test_topk = None
    if len(test_loader) > 0:
        force_print(f"[INFO] Evaluating Top-K precision (k={eval_ks}) on Test...")
        test_topk = evaluate_topk(model, test_loader, ks=eval_ks)
        save_topk_precision_csv(test_topk, run_output_dir, prefix='test')
        plot_topk_precision(test_topk, run_output_dir, prefix='test')
        if getattr(config, 'USE_R_PRECISION', False):
            save_r_precision_csv(test_topk, run_output_dir, prefix='test')
            plot_r_precision(test_topk, run_output_dir, prefix='test')

    save_run_summary_json(
        val_metrics, test_metrics, val_topk, test_topk,
        val_loss, test_loss, run_output_dir,
    )

    return val_metrics, test_metrics, val_details, test_details, strength_thresholds


# ──────────────────────────────────────────────
# 6. 可視化
# ──────────────────────────────────────────────

def run_visualization(model, run_output_dir, val_loader=None, test_loader=None):
    """変異位置分布プロット、語彙ネットワーク可視化、Attention ヒートマップを実行する。"""
    # 変異位置分布プロット
    if config.SAVE_PREDICTIONS:
        for prefix in ('valid', 'test'):
            csv_path = os.path.join(run_output_dir, "csv", f'{prefix}_predictions.csv')
            if os.path.exists(csv_path):
                force_print(f"[INFO] Plotting mutation dist: {csv_path}")
                plot_mutation_dist(csv_path)

    # Attention ヒートマップ（Validation の最初のバッチを使用）
    if val_loader is not None and getattr(config, "SAVE_ATTENTION_HEATMAP", True):
        try:
            sample_batch = next(iter(val_loader))
            plot_attention_heatmap(model, sample_batch, run_output_dir, prefix='valid')
        except Exception as e:
            force_print(f"[WARNING] Attention heatmap skipped: {e}")

    # 語彙ネットワーク可視化
    best_model_path = os.path.join(run_output_dir, "models", "best_model.pth")
    if not os.path.exists(best_model_path):
        return

    force_print("[INFO] === 語彙ネットワーク可視化 ===")
    version    = os.path.basename(run_output_dir)
    codon_csv  = config.CODON_CSV
    freq_csv   = config.FREQ_CSV
    pos_to_region = load_position_to_region(codon_csv)

    force_print(f"[INFO] Vocab network output dir: {run_output_dir}")
    plot_base_embedding_network(model, config, run_output_dir, version)
    plot_position_threshold_network(model, pos_to_region, run_output_dir, version)
    plot_position_tsne(model, pos_to_region, run_output_dir, version)
    plot_position_cluster_network(model, pos_to_region, run_output_dir, version)
    if os.path.exists(freq_csv):
        plot_major_mutation_network(model, pos_to_region, freq_csv, codon_csv, run_output_dir, version)
    else:
        force_print(f"[WARNING] freq_csv not found, skipping major mutation network: {freq_csv}")


# ──────────────────────────────────────────────
# 6. Ensemble 評価
# ──────────────────────────────────────────────

def ensemble_evaluate(val_loader, test_loader, loss_fn, run_output_dir, strength_thresholds):
    """複数のチェックポイントをロードし、softmax 確率を平均したアンサンブル評価を行う。

    config.ENSEMBLE_CHECKPOINT_PATHS に記載されたチェックポイントを全てロードし、
    各バッチで全モデルの softmax 予測を平均して最終予測とする。

    Args:
        val_loader: Validation DataLoader
        test_loader: Test DataLoader
        loss_fn: 損失関数 dict
        run_output_dir: 結果保存ディレクトリ
        strength_thresholds: (low_max, med_max) の閾値タプル
    """
    import torch.nn.functional as F

    ckpt_paths = getattr(config, 'ENSEMBLE_CHECKPOINT_PATHS', [])
    if not ckpt_paths:
        force_print("[INFO] ENSEMBLE_CHECKPOINT_PATHS が空のためアンサンブルをスキップします。")
        return

    # 全チェックポイントのモデルをロード
    models = []
    for ckpt_path in ckpt_paths:
        if not os.path.exists(ckpt_path):
            force_print(f"[WARNING] Ensemble checkpoint not found, skipping: {ckpt_path}")
            continue
        m = HierarchicalTransformer().to(config.DEVICE)
        ckpt = torch.load(ckpt_path, map_location=config.DEVICE, weights_only=True)
        state_dict = ckpt.get('model_state_dict', ckpt)
        m.load_state_dict(state_dict)
        m.eval()
        models.append(m)
        force_print(f"[Ensemble] Loaded checkpoint: {ckpt_path}")

    if not models:
        force_print("[WARNING] 有効なアンサンブルチェックポイントがありません。スキップします。")
        return

    force_print(f"[Ensemble] {len(models)} モデルでアンサンブル評価を実行します。")

    def ensemble_forward(x_cat, x_num, mask):
        """全モデルの softmax 確率を平均して返す。"""
        all_probs = []
        with torch.no_grad():
            for m in models:
                out = m(x_cat, x_num, src_key_padding_mask=mask)
                # 8-tuple に対応（先頭 6 要素を使用）
                (reg, pos, aa, strength, cod, syn, *_) = out
                probs = (
                    F.softmax(reg,      dim=-1),
                    F.softmax(pos,      dim=-1),
                    F.softmax(aa,       dim=-1),
                    strength,                    # 回帰値はそのまま平均
                    F.softmax(cod,      dim=-1),
                    F.softmax(syn,      dim=-1),
                    None, None,                  # base_after, aa_after は無効
                )
                all_probs.append(probs)

        # タスクごとに平均
        n = len(all_probs)
        avg_reg     = sum(p[0] for p in all_probs) / n
        avg_pos     = sum(p[1] for p in all_probs) / n
        avg_aa      = sum(p[2] for p in all_probs) / n
        avg_str     = sum(p[3] for p in all_probs) / n
        avg_cod     = sum(p[4] for p in all_probs) / n
        avg_syn     = sum(p[5] for p in all_probs) / n

        return (avg_reg, avg_pos, avg_aa, avg_str, avg_cod, avg_syn, None, None)

    # アンサンブルモデルを関数として evaluate() に渡すためのラッパークラス
    class EnsembleWrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def eval(self):
            for m in models:
                m.eval()
            return self

        def forward(self, x_cat, x_num, src_key_padding_mask=None):
            return ensemble_forward(x_cat, x_num, src_key_padding_mask)

    ens_model = EnsembleWrapper()

    # Validation アンサンブル評価
    force_print("[Ensemble] Validation set evaluation...")
    _, ens_val_metrics, ens_val_details, ens_val_cat, ens_val_ym, _ = evaluate(
        ens_model, val_loader, loss_fn, strength_thresholds
    )
    ens_dir = os.path.join(run_output_dir, 'ensemble')
    os.makedirs(ens_dir, exist_ok=True)
    save_metrics_csv(ens_val_metrics, ens_dir, prefix='ensemble_valid')
    force_print("[Ensemble] Validation metrics saved.")

    # Test アンサンブル評価
    if len(test_loader) > 0:
        force_print("[Ensemble] Test set evaluation...")
        _, ens_test_metrics, ens_test_details, ens_test_cat, ens_test_ym, _ = evaluate(
            ens_model, test_loader, loss_fn, strength_thresholds
        )
        save_metrics_csv(ens_test_metrics, ens_dir, prefix='ensemble_test')
        force_print("[Ensemble] Test metrics saved.")

    force_print(f"[Ensemble] Results saved to {ens_dir}")


# ──────────────────────────────────────────────
# 7. エントリポイント
# ──────────────────────────────────────────────

def main():
    print(f"PID: {os.getpid()}")
    set_seed(config.SEED)
    force_print(f"Using device: {config.DEVICE}")
    print_config()

    wandb = init_wandb()

    # 1. データ準備
    train_loader, val_loader, test_loader, data_info, train, valid, test, make_train_loader = prepare_data()
    force_print(f"Training lengths:   {data_info['train_min_len']} to {data_info['train_max_len']}")
    force_print(f"Validation lengths: {data_info['val_min_len']} to {data_info['val_max_len']}")
    force_print(f"Test lengths:       {data_info['test_min_len']} to {data_info['test_max_len']}")

    if not config.USE_DB and train:
        print_sample_structure(train[0], sample_idx=0)

    # 2. モデル・損失・最適化器の構築
    # [260417] HYBRID_ALPHA 方式では CE + class_weights を使うため、
    # LOSS_FUNCTION_TYPE に依存せず USE_DB=True の場合は常にクラス頻度を計算する
    class_counts_dict = None
    if config.USE_DB:
        force_print("[INFO] Computing class counts for class-weighted loss...")
        from .db.queries import get_train_class_counts
        class_counts_dict = get_train_class_counts()
    elif getattr(config, 'LOSS_FUNCTION_TYPE', 'ce').lower() in ['wce', 'cbce']:
        from .db.queries import get_train_class_counts
        class_counts_dict = get_train_class_counts()

    model, loss_fn, loss_wrapper, optimizer, scheduler = build_components(class_counts_dict)

    # 出力ディレクトリ設定
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment = getattr(config, 'EXPERIMENT_NAME', '').strip('/')
    result_base = os.path.join(config.RESULT_SAVE_DIR, experiment) if experiment else config.RESULT_SAVE_DIR
    run_output_dir = os.path.join(result_base, timestamp)
    os.makedirs(run_output_dir, exist_ok=True)
    force_print(f"Starting training... Output dir: {run_output_dir}")

    # 前処理情報の保存
    if not config.USE_DB and config.SAVE_STRAIN_INFO:
        save_strain_info([t[4] for t in train], run_output_dir, prefix="train")
        save_strain_info([t[4] for t in valid], run_output_dir, prefix="valid")
        save_strain_info([t[4] for t in test],  run_output_dir, prefix="test")
    if not config.USE_DB:
        save_synonymous_distribution_csv(train, valid, test, run_output_dir)
    save_config_copy(run_output_dir)

    # 共通: strength_thresholds を決定して両関数に渡す (config優先、DYNAMIC_STRENGTH_CATEGORYで動的切り替え)
    use_dynamic = getattr(config, 'DYNAMIC_STRENGTH_CATEGORY', False)

    if use_dynamic:
        low_max = data_info.get('strength_low_max', 6.0)
        med_max = data_info.get('strength_med_max', 10.0)
        force_print(f"[INFO] Using dynamic thresholds from DB: low_max={low_max}, med_max={med_max}")
    else:
        low_max = getattr(config, 'STRENGTH_CATEGORY_LOW_MAX', 6.0)
        med_max = getattr(config, 'STRENGTH_CATEGORY_MED_MAX', 10.0)
        force_print(f"[INFO] Using fixed thresholds from config: low_max={low_max}, med_max={med_max}")

    strength_thresholds = (low_max, med_max)

    # 3. 学習ループ
    training_log, best_model_path = run_training(
        model, train_loader, val_loader, loss_fn, loss_wrapper,
        optimizer, scheduler, run_output_dir, data_info, wandb,
        strength_thresholds=strength_thresholds,
        make_train_loader=make_train_loader,
    )
    save_training_log(training_log, run_output_dir)
    plot_training_curve(training_log, run_output_dir)
    save_early_stopping_json(training_log, best_model_path, run_output_dir, config.EPOCHS)
    save_model_summary_txt(model, run_output_dir)

    # ベストモデルのロード
    if os.path.exists(best_model_path):
        checkpoint = torch.load(
            best_model_path,
            map_location=config.DEVICE,
            weights_only=True,
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        force_print(f"Loaded best model from epoch {checkpoint['epoch']+1}")

    # 4. 最終評価・結果保存
    run_final_evaluation(model, val_loader, test_loader, loss_fn, run_output_dir, data_info,
                         strength_thresholds=strength_thresholds)

    # 5. 可視化
    run_visualization(model, run_output_dir, val_loader=val_loader, test_loader=test_loader)

    # 6. Ensemble 評価 (USE_ENSEMBLE=True かつ ENSEMBLE_CHECKPOINT_PATHS が非空の場合)
    if getattr(config, 'USE_ENSEMBLE', False) and getattr(config, 'ENSEMBLE_CHECKPOINT_PATHS', []):
        force_print("[INFO] Ensemble evaluation starting...")
        ensemble_evaluate(val_loader, test_loader, loss_fn, run_output_dir, strength_thresholds)

    force_print(f"[INFO] Process completed. Results saved to {run_output_dir}")
    finish_wandb(wandb)
    return best_model_path


# ──────────────────────────────────────────────
# Walk-forward 検証ユーティリティ
# ──────────────────────────────────────────────

def _wf_patch_config(train_start, split_date: str, split_end, fold_id: int, wf_run_dir: str):
    """1 フォールド分の config 属性をインプロセスで書き換える。"""
    config.SPLIT_MODE                = 'walk_forward'
    config.WALK_FORWARD_TRAIN_START  = train_start
    config.TEMPORAL_SPLIT_DATE       = split_date
    config.TEMPORAL_SPLIT_TEST_END   = split_end
    config.FORCE_DATE_REASSIGN       = False  # run_walk_forward が事前に assign_wf_splits を呼ぶ
    config.EXPERIMENT_NAME           = f'walk_forward/{os.path.basename(wf_run_dir)}/fold_{fold_id}'
    config.WANDB_RUN_NAME            = f'wf_fold{fold_id}_{split_date}'


def _wf_clear_split_cache():
    """フォールド間でクラスカウントキャッシュを削除して再計算を強制する。"""
    cache_file = os.path.join(config.CACHE_DIR, 'class_counts.pkl')
    if os.path.exists(cache_file):
        os.remove(cache_file)


def _wf_summarize(folds, wf_run_dir: str):
    """全フォールドの run_summary.json を集約して CSV に出力する。

    run_summary.json のネスト構造 {"val": {...}, "test": {...}} を
    val_* / test_* のフラットな列に展開し、フォールドメタ情報も付加する。
    """
    import csv, json as _json

    fold_meta = {fold_id: (train_start, split_date, split_end, desc)
                 for fold_id, train_start, split_date, split_end, desc in folds}

    rows = []
    for fold_id, *_ in folds:
        fold_dir = os.path.join(config.RESULT_SAVE_DIR, 'walk_forward',
                                os.path.basename(wf_run_dir), f'fold_{fold_id}')
        if not os.path.isdir(fold_dir):
            continue
        subdirs = sorted(d for d in os.listdir(fold_dir)
                         if os.path.isdir(os.path.join(fold_dir, d)))
        if not subdirs:
            continue
        summary_path = os.path.join(fold_dir, subdirs[-1], 'run_summary.json')
        if not os.path.exists(summary_path):
            continue
        with open(summary_path) as f:
            summary = _json.load(f)

        # フォールドメタ情報
        train_start, split_date, split_end, desc = fold_meta.get(fold_id, (None, '', '', ''))
        row = {
            'fold': fold_id,
            'train_start': train_start or '',
            'split_date': split_date,
            'split_end': split_end or '',
            'description': desc,
            'val_loss': summary.get('val_loss'),
            'test_loss': summary.get('test_loss'),
        }
        # val / test のネストをフラット展開
        for prefix in ('val', 'test'):
            for k, v in summary.get(prefix, {}).items():
                row[f'{prefix}_{k}'] = v
        rows.append(row)

    if not rows:
        return

    force_print("\n" + "="*60)
    force_print("  Walk-forward Summary")
    force_print("="*60)
    force_print(f"  {'fold':>4}  {'split_date':>12}  {'test_top1_position_hit_rate_pct':>31}  {'test_r_precision_position':>25}  {'test_loss':>10}")
    force_print(f"  {'-'*88}")
    for r in rows:
        force_print(
            f"  {r.get('fold', '?'):>4}  {r.get('split_date', ''):>12}"
            f"  {r.get('test_top1_position_hit_rate_pct', float('nan')):>31.4f}"
            f"  {r.get('test_r_precision_position',       float('nan')):>25.4f}"
            f"  {r.get('test_loss', float('nan')):>10.6f}"
        )

    csv_path = os.path.join(wf_run_dir, 'walk_forward_summary.csv')
    keys = sorted({k for r in rows for k in r})
    # 見やすい順に並べ替え（メタ列を先頭に）
    meta_keys = ['fold', 'train_start', 'split_date', 'split_end', 'description', 'val_loss', 'test_loss']
    ordered_keys = meta_keys + [k for k in keys if k not in meta_keys]
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=ordered_keys, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    force_print(f"\n  Summary CSV → {csv_path}")


def run_walk_forward(folds, wf_run_dir: str):
    """Walk-forward 検証：フォールドを順番に学習＋評価して集計する。

    設計:
      - Fold 1: 全歴史データ (< split_date) で学習。USE_PRETRAINING=True なら事前学習も実行。
      - Fold N (N>=2): 直前フォールドの best_model.pth を起点に、直近6ヶ月のみで学習 (Method B)。
      - 事前学習は Fold 1 のみ。Fold 2 以降は WF_PREV_FOLD_CHECKPOINT で初期化。

    Args:
        folds     : [(fold_id, train_start, split_date, split_end, desc), ...]
        wf_run_dir: walk-forward 全体の出力ルートディレクトリ
    """
    import json as _json

    # メタ情報を保存
    meta = {
        'wf_run_dir': wf_run_dir,
        'folds': [{'fold': fid, 'train_start': ts, 'split_date': sd, 'split_end': se, 'desc': desc}
                  for fid, ts, sd, se, desc in folds],
    }
    with open(os.path.join(wf_run_dir, 'walk_forward_meta.json'), 'w') as f:
        _json.dump(meta, f, indent=2, ensure_ascii=False)

    prev_best_model_path = None

    for idx, (fold_id, train_start, split_date, split_end, desc) in enumerate(folds):
        force_print(f"\n{'='*60}")
        force_print(f"  Walk-forward Fold {fold_id}: {desc}")
        if train_start:
            force_print(f"  Train : {train_start} <= collection_date < {split_date}")
        else:
            force_print(f"  Train : collection_date < {split_date} (全歴史データ)")
        if split_end:
            force_print(f"  Test  : {split_date} <= collection_date < {split_end}")
        else:
            force_print(f"  Test  : {split_date} <= collection_date (no upper bound)")
        force_print(f"{'='*60}")

        # 1. このフォールドの config を設定
        _wf_patch_config(train_start, split_date, split_end, fold_id, wf_run_dir)
        _wf_clear_split_cache()

        # 2. このフォールドの split_type_wf を事前に確定（pretrain より前に実行）
        from .db.connection import connect_db, get_db_path
        from .db.queries import assign_wf_splits
        _con = connect_db(get_db_path())
        assign_wf_splits(_con)
        _con.close()

        # 3. Fold 1 のみ事前学習を実行
        if idx == 0 and getattr(config, 'USE_PRETRAINING', False):
            force_print(f"[INFO] Walk-forward Fold {fold_id}: 事前学習を開始 (split_type_wf 基準)...")
            from .scripts.train.pretrain import run_pretraining
            run_pretraining()

        # 4. 直前フォールドの重みを設定（Fold 1 = None → USE_PRETRAINING が適用される）
        config.WF_PREV_FOLD_CHECKPOINT = prev_best_model_path

        # 5. 学習・評価
        prev_best_model_path = main()

        force_print(f"[INFO] Fold {fold_id} complete. Best model: {prev_best_model_path}")

    _wf_summarize(folds, wf_run_dir)
    force_print(f"\n[INFO] Walk-forward complete. Results root: {wf_run_dir}")


def _run_entry():
    """config.SPLIT_MODE に応じて単一実行 or Walk-forward を切り替えるエントリポイント。"""
    if getattr(config, 'SPLIT_MODE', 'timestep') == 'walk_forward':
        from .scripts.eval.walk_forward import FOLDS
        wf_folds = getattr(config, 'WALK_FORWARD_FOLDS', None)
        selected = [(fid, ts, sd, se, desc) for fid, ts, sd, se, desc in FOLDS
                    if wf_folds is None or fid in set(wf_folds)]
        if not selected:
            raise ValueError(f"WALK_FORWARD_FOLDS={wf_folds} に該当するフォールドが FOLDS に存在しません")
        wf_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        wf_run_dir = os.path.join(config.RESULT_SAVE_DIR, 'walk_forward', wf_timestamp)
        os.makedirs(wf_run_dir, exist_ok=True)
        run_walk_forward(selected, wf_run_dir)
    else:
        main()


if __name__ == "__main__":
    _run_entry()
