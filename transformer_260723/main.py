# --- main.py ---
# DuckDB対応版メイン処理
# Usage: setsid nohup python -m transformer_260723.main > nohup0.out 2>&1 & 

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


def _log_rss(tag: str):
    """[調査用 2026-07-28] walk_forward の fold_3 test評価でのOOM原因切り分けのため、
    プロセスの現在RSS（/proc/self/status VmRSS）とこれまでの最大RSS（ru_maxrss）を
    要所でログ出力する。fold境界を跨いで単調にRSSが積み上がるか（同一プロセス内での
    蓄積仮説）を直接確認するための一時計測。"""
    try:
        with open('/proc/self/status') as f:
            vmrss_kb = None
            for line in f:
                if line.startswith('VmRSS:'):
                    vmrss_kb = int(line.split()[1])
                    break
        import resource
        maxrss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024
        cur_gb = (vmrss_kb / 1024 / 1024) if vmrss_kb is not None else float('nan')
        force_print(f"[MEM] {tag}: current_rss={cur_gb:.2f} GiB, peak_rss={maxrss_gb:.2f} GiB")
    except Exception as e:
        force_print(f"[MEM] {tag}: (failed to read RSS: {e})")
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
    save_date_metrics_csv,
    save_strength_fine_csv,
    save_lineage_metrics_csv,
    save_entropy_bin_csv,
    save_nll_csv,
    save_confident_recall_csv,
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
    plot_attention_heatmap,
    plot_metrics_by_date,
    plot_strength_fine,
    plot_lineage_metrics,
    plot_entropy_vs_accuracy,
    plot_labeldiversity_vs_accuracy,
    plot_simpson_vs_accuracy,
    save_lineage_diversity_csv,
    plot_entropy_bin,
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

        # split_type=0(train)はconfig.TRAIN_NUM_DATALOADER_WORKERS、
        # split_type>=1(val/test、評価用)はconfig.EVAL_NUM_DATALOADER_WORKERSを使う。
        # trainはpersistent_workers=Trueで全epochにわたり同一workerが生存するため、
        # _group_label_cacheのCopy-on-Write共有がバッチを重ねるごとに崩れてworker数分
        # 重複し、巨大共起グループを含むfoldでOOMした（walk_forward fold_4実測,
        # 2026-07-31）。worker数を絞ってこの重複の総量を抑える
        # （MAX_GROUP_MEMBERS_FOR_CACHEによるキャッシュ自体の縮小と併用）。
        def _make_loader(split_type, shuffle):
            num_workers_override = (
                getattr(config, 'TRAIN_NUM_DATALOADER_WORKERS', None) if split_type == 0
                else getattr(config, 'EVAL_NUM_DATALOADER_WORKERS', None)
            )
            return create_db_dataloader(
                db_path=db_path, split_type=split_type,
                batch_size=config.BATCH_SIZE, shuffle=shuffle,
                max_cooccurrence=config.MAX_CO_OCCURRENCE,
                strain_to_strength=combined_strength,
                num_workers_override=num_workers_override,
            )

        train_loader = _make_loader(0, shuffle=True)
        val_loader   = _make_loader(1, shuffle=False)
        # test_loader はここでは構築しない（遅延構築、下記 make_test_loader 参照）。
        # testは学習ループ中は一切使われず run_final_evaluation まで不要だが、ここで
        # 即座に構築すると DBIterableDataset.__init__ が test split 分（実測 fold により
        # 最大 66万グループ超）のラベルキャッシュを persistent_workers=8 と共に即座に
        # 確保してしまい、train+valid+testの3キャッシュ×24 workerが学習開始直後から
        # 数時間常駐してメモリを圧迫する（walk_forward fold_3 OOM実測, 2026-07-25）。
        test_loader  = None

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
            # 再生成した分割に基づいてデータローダーも作り直す（testはmake_test_loaderが
            # 呼ばれた時点でDBから読むため、ここで作り直す必要はない）
            train_loader = _make_loader(0, shuffle=True)
            val_loader   = _make_loader(1, shuffle=False)

        # Curriculum Learning 用: 訓練ローダーを再生成するラムダ
        def make_train_loader(max_length=None):
            return create_db_dataloader(
                db_path=db_path, split_type=0,
                batch_size=config.BATCH_SIZE, shuffle=True,
                max_cooccurrence=config.MAX_CO_OCCURRENCE,
                max_length=max_length,
                strain_to_strength=combined_strength,
                num_workers_override=getattr(config, 'TRAIN_NUM_DATALOADER_WORKERS', None),
            )

        # test_loader の遅延構築用ラムダ（run_final_evaluation 直前に呼ぶ）
        def make_test_loader():
            return _make_loader(2, shuffle=False)

        # val_loader の再構築用ラムダ（R-Precision評価前にval_loaderを一度破棄し、
        # 新規（かつEVAL_NUM_DATALOADER_WORKERS採用済みの）workerで作り直すために使う。
        # walk_forward fold_3 OOM調査（2026-07-28）: evaluate()一巡目の後もval/test
        # loaderのworkerが生存し続けたまま2巡目(evaluate_topk)に入ることがOOMの一因
        # だったため、巡目の境界でworkerを一度完全に終了・再生成する。
        def make_val_loader():
            return _make_loader(1, shuffle=False)

        return (train_loader, val_loader, test_loader, data_info, None, None, None,
                make_train_loader, make_test_loader, make_val_loader)

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
        # 非DBパスは元々testも即時構築済み（規模が小さく遅延の恩恵が薄いため据え置き）。
        # make_test_loaderは呼び出し側とのインターフェース統一のためのラップに過ぎない。
        def make_test_loader():
            return test_loader

        def make_val_loader():
            return val_loader

        return (train_loader, val_loader, test_loader, data_info, train, valid, test, None,
                make_test_loader, make_val_loader)



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
        split_mode = getattr(config, 'SPLIT_MODE', 'timestep')
        pretrain_path = os.path.join(
            config.OUTPUT_DIR, 'pretrain', split_mode, f'pretrain_{mode}_final.pth'
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
                 strength_thresholds, make_train_loader=None, optuna_trial=None):
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
            metric_key = early_stop_metric.replace('val_', '')
            # 指定キーが val_metrics に無いと m.get(key, 0) が常に 0 を返し、
            # MODE='max' では「一度も改善しない＝best_model が保存されない」事故になる。
            # これを防ぐため存在を検証し、無ければ WARNING を出して val_loss にフォールバックする。
            available_keys = set().union(*(m.keys() for m in val_metrics.values())) if val_metrics else set()
            total_samples = sum(m['num_samples'] for m in val_metrics.values())
            if metric_key not in available_keys:
                force_print(
                    f"[WARNING] EARLY_STOPPING_METRIC='{early_stop_metric}' (key='{metric_key}') は "
                    f"val_metrics に存在しません。利用可能: "
                    f"{sorted(k for k in available_keys if k != 'num_samples')}. "
                    f"val_loss にフォールバックします（EARLY_STOPPING_MODE='min' を確認）。"
                )
                current_metric = val_loss
            elif total_samples > 0:
                # タイムステップ長ごとの hit-rate を num_samples で重み付き平均
                current_metric = sum(m.get(metric_key, 0) * m['num_samples']
                                     for m in val_metrics.values()) / total_samples
            else:
                current_metric = val_loss

        # Optuna pruning: 各エポックの early-stop 指標を報告し、見込みの薄い trial を打ち切る。
        # study 方向と early_stop_mode が一致するときのみ報告する（不一致だと良い trial を
        # 誤って枝刈りするため）。current_metric は上で算出済み。
        if optuna_trial is not None:
            import optuna
            study_dir = getattr(config, 'OPTUNA_DIRECTION', 'maximize')
            aligned = ((early_stop_mode == 'max' and study_dir == 'maximize') or
                       (early_stop_mode == 'min' and study_dir == 'minimize'))
            if aligned:
                optuna_trial.report(current_metric, epoch)
                if optuna_trial.should_prune():
                    force_print(f"[Optuna] Trial pruned at epoch {epoch+1} "
                                f"(metric={current_metric:.4f})")
                    raise optuna.TrialPruned()

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

def _plot_on(name):
    """config.PLOT_OUTPUTS に基づき、指定プロットを出力するかを返す（キー無し/未定義なら True）。"""
    outputs = getattr(config, 'PLOT_OUTPUTS', None)
    if not isinstance(outputs, dict):
        return True
    return outputs.get(name, True)


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
        if _plot_on('metrics_by_timestep'):
            plot_metrics_by_timestep(metrics, run_output_dir, prefix=prefix)
        if _plot_on('category_metrics'):
            plot_category_metrics(cat_metrics, run_output_dir, prefix=prefix)
        if _plot_on('strength_calibration'):
            plot_strength_calibration(details, run_output_dir, prefix=prefix)
        if _plot_on('per_position_recall'):
            plot_per_position_recall(details, run_output_dir, prefix=prefix, top_n=getattr(config, "PLOT_TOP_N_POSITIONS", 40))
        if _plot_on('strength_fine'):
            plot_strength_fine(details, run_output_dir, prefix=prefix)
        if _plot_on('lineage_metrics'):
            plot_lineage_metrics(details, run_output_dir, prefix=prefix,
                                 top_n=getattr(config, 'PLOT_TOP_N_LINEAGES', 30))
        # 散布図の裏づけ系統別テーブルを CSV 保存（run 間比較・再描画の基準。プロット有無に依らず常時出力）
        save_lineage_diversity_csv(details, run_output_dir, prefix=prefix)
        if _plot_on('entropy_vs_accuracy'):
            plot_entropy_vs_accuracy(details, run_output_dir, prefix=prefix)
        if _plot_on('labeldiversity_vs_accuracy'):
            plot_labeldiversity_vs_accuracy(details, run_output_dir, prefix=prefix)
        if _plot_on('simpson_vs_accuracy'):
            plot_simpson_vs_accuracy(details, run_output_dir, prefix=prefix)
        df_ent = save_entropy_bin_csv(details, run_output_dir, prefix=prefix)
        if _plot_on('entropy_bin'):
            plot_entropy_bin(df_ent, run_output_dir, prefix=prefix)
        save_nll_csv(details, run_output_dir, prefix=prefix)
        if getattr(config, 'SAVE_CONFIDENT_SUBSET', True):
            save_confident_recall_csv(details, run_output_dir, prefix=prefix)
        if _plot_on('region_metrics'):
            plot_region_metrics(details, run_output_dir, prefix=prefix)
        return df_ts, df_cat

    # Validation
    force_print("Final evaluation on Validation Set...")
    _log_rss("before evaluate(val_loader)")
    val_loss, val_metrics, val_details, val_cat_metrics, val_metrics_ym, val_calib_bins = evaluate(
        model, val_loader, loss_fn, strength_thresholds
    )
    _log_rss("after evaluate(val_loader)")
    val_df_ts, val_df_cat = _save_results(val_metrics, val_cat_metrics, val_details, prefix="valid")
    if val_calib_bins:
        save_calibration_csv(val_calib_bins, run_output_dir, prefix="valid")
        if _plot_on('calibration'):
            plot_calibration(val_calib_bins, run_output_dir, prefix="valid")

    # Test
    test_loss = None
    test_metrics, test_details, test_cat_metrics = None, None, None
    test_df_ts, test_df_cat = None, None
    test_metrics_ym = {}
    if len(test_loader) > 0:
        force_print("Final evaluation on Test Set...")
        _log_rss("before evaluate(test_loader)")
        test_loss, test_metrics, test_details, test_cat_metrics, test_metrics_ym, test_calib_bins = evaluate(
            model, test_loader, loss_fn, strength_thresholds
        )
        _log_rss("after evaluate(test_loader)")
        test_df_ts, test_df_cat = _save_results(test_metrics, test_cat_metrics, test_details, prefix="test")
        if test_calib_bins:
            save_calibration_csv(test_calib_bins, run_output_dir, prefix="test")
            if _plot_on('calibration'):
                plot_calibration(test_calib_bins, run_output_dir, prefix="test")

    # 統合レポート
    print_combined_report(val_details, test_details, strength_thresholds)
    if test_details:
        if _plot_on('combined_val_test_metrics'):
            plot_combined_val_test_metrics(val_metrics, test_metrics, run_output_dir)
        if _plot_on('combined_category_comparison'):
            plot_combined_category_comparison(val_details, test_details, strength_thresholds, run_output_dir)
        save_val_test_gap_csv(val_metrics, test_metrics, run_output_dir)

        # 結合したメトリクスCSVを出力（timestepでソート）
        save_combined_metrics_csv(val_df_ts, test_df_ts, run_output_dir, 'combined_metrics_by_timestep.csv')
        save_combined_metrics_csv(val_df_cat, test_df_cat, run_output_dir, 'combined_metrics_by_category.csv')

    # 月別 CSV とグラフを常に出力
    save_date_metrics_csv(val_metrics_ym,  run_output_dir, prefix='valid')
    save_date_metrics_csv(test_metrics_ym, run_output_dir, prefix='test')
    if _plot_on('metrics_by_date'):
        plot_metrics_by_date(val_metrics_ym, test_metrics_ym, run_output_dir)

    return val_metrics, test_metrics, val_details, test_details, strength_thresholds, val_loss, test_loss


def run_topk_evaluation_val(model, val_loader, run_output_dir, val_metrics, val_loss):
    """R-Precision（動的K=|target|）評価をValidationに対してのみ行う。

    val/test_loaderへの追加フルパスであり、run_final_evaluationとは分離している。
    test側（run_topk_evaluation_test）と別関数にしているのは、呼び出し元（main()）が
    このval評価の完了後に自身のフレーム上でval_loaderへの参照を明示的に破棄できる
    ようにするため。単一の関数内でval→test評価を続けて行うと、val評価用の
    ローダー・workerへの参照が呼び出し元(main())のローカル変数として関数呼び出しの
    間ずっと生き続け、test評価が最も重いタイミングでval用の待機workerまで同時に
    メモリへ乗ったままになる（walk_forward fold_3 region-conditioned検証runでの
    R-Precision OOM調査, 2026-08-17）。
    """
    force_print("[INFO] Evaluating R-Precision on Validation...")
    val_topk = evaluate_topk(model, val_loader, ks=())
    if getattr(config, 'USE_R_PRECISION', False):
        save_r_precision_csv(val_topk, run_output_dir, prefix='valid')
        if _plot_on('r_precision'):
            plot_r_precision(val_topk, run_output_dir, prefix='valid')
    return val_topk


def run_topk_evaluation_test(model, test_loader, run_output_dir, test_metrics, test_loss):
    """R-Precision（動的K=|target|）評価をTestに対してのみ行う。

    呼び出し元（main()）がrun_topk_evaluation_val完了後にval_loaderを破棄してから
    このtest評価を呼ぶことを想定する（run_topk_evaluation_valのdocstring参照）。
    固定K(1,3,5)のTop-K評価は処理時間・メモリ負荷が大きい割に有用性が低いため除外し、
    R-Precisionのみ計算する。
    """
    test_topk = None
    if len(test_loader) > 0:
        force_print("[INFO] Evaluating R-Precision on Test...")
        test_topk = evaluate_topk(model, test_loader, ks=())
        if getattr(config, 'USE_R_PRECISION', False):
            save_r_precision_csv(test_topk, run_output_dir, prefix='test')
            if _plot_on('r_precision'):
                plot_r_precision(test_topk, run_output_dir, prefix='test')
    return test_topk


# ──────────────────────────────────────────────
# 6. 可視化
# ──────────────────────────────────────────────

def run_visualization(model, run_output_dir, val_loader=None, test_loader=None):
    """変異位置分布プロット、語彙ネットワーク可視化、Attention ヒートマップを実行する。"""
    # 変異位置分布プロット
    if config.SAVE_PREDICTIONS and _plot_on('mutation_dist'):
        for prefix in ('valid', 'test'):
            csv_path = os.path.join(run_output_dir, "csv", f'{prefix}_predictions.csv')
            if os.path.exists(csv_path):
                force_print(f"[INFO] Plotting mutation dist: {csv_path}")
                plot_mutation_dist(csv_path)

    # Attention ヒートマップ（Validation の最初のバッチを使用）
    if val_loader is not None and getattr(config, "SAVE_ATTENTION_HEATMAP", True) and _plot_on('attention_heatmap'):
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
    if _plot_on('base_embedding_network'):
        plot_base_embedding_network(model, config, run_output_dir, version)
    if _plot_on('position_threshold_network'):
        plot_position_threshold_network(model, pos_to_region, run_output_dir, version)
    if _plot_on('position_tsne'):
        plot_position_tsne(model, pos_to_region, run_output_dir, version)
    if _plot_on('position_cluster_network'):
        plot_position_cluster_network(model, pos_to_region, run_output_dir, version)
    if _plot_on('major_mutation_network'):
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

def main(optuna_trial=None):
    print(f"PID: {os.getpid()}")
    set_seed(config.SEED)
    force_print(f"Using device: {config.DEVICE}")
    print_config()
    _log_rss("main() start")

    wandb = init_wandb()

    # 1. データ準備
    # test_loader は None（DBパスでは遅延構築）で返る。run_final_evaluation 直前に
    # make_test_loader() で構築する（詳細は prepare_data() 内コメント参照）。
    (train_loader, val_loader, test_loader, data_info, train, valid, test,
     make_train_loader, make_test_loader, make_val_loader) = prepare_data()
    force_print(f"Training lengths:   {data_info['train_min_len']} to {data_info['train_max_len']}")
    force_print(f"Validation lengths: {data_info['val_min_len']} to {data_info['val_max_len']}")
    force_print(f"Test lengths:       {data_info['test_min_len']} to {data_info['test_max_len']}")
    _log_rss("after prepare_data (train+val loaders built)")

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
        optuna_trial=optuna_trial,
    )
    save_training_log(training_log, run_output_dir)
    if _plot_on('training_curve'):
        plot_training_curve(training_log, run_output_dir)
    save_early_stopping_json(training_log, best_model_path, run_output_dir, config.EPOCHS)
    save_model_summary_txt(model, run_output_dir)

    # 学習完了後、train_loader（DATALOADER_PERSISTENT_WORKERS=True の場合はここまで生き続ける
    # workerプロセス群）をここで明示的に解放する。run_final_evaluationはval_loader/test_loader
    # しか使わないため、train_loaderを持ち続ける意味は無い一方、trainは全splitの中で最大
    # （実測900万行超）で、そのworker群がtest評価開始時にもまだ生きていると
    # train+valid+testの全workerが同時に常駐しメモリを圧迫し、OOM-Killの原因になっていた
    # （2026-07-24 walk_forward fold_2実測）。
    _log_rss("before del train_loader")
    del train_loader, make_train_loader
    import gc
    gc.collect()
    _log_rss("after del train_loader + gc.collect()")

    # ベストモデルのロード
    if os.path.exists(best_model_path):
        checkpoint = torch.load(
            best_model_path,
            map_location=config.DEVICE,
            weights_only=True,
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        force_print(f"Loaded best model from epoch {checkpoint['epoch']+1}")

    # test_loader をここで初めて構築する（train_loader解放後・評価直前）。
    # DBパスではprepare_data()がtest_loaderをNoneのまま返しているため、train分の
    # ラベルキャッシュ×workerが完全に解放されてからtest分を確保することで、
    # train+testの2キャッシュが同時に常駐するピークを避ける。
    if test_loader is None:
        test_loader = make_test_loader()
    _log_rss("after test_loader built (val_loader + test_loader both alive)")

    # 4. 最終評価・結果保存
    (val_metrics, test_metrics, val_details, test_details, strength_thresholds,
     val_loss, test_loss) = run_final_evaluation(
        model, val_loader, test_loader, loss_fn, run_output_dir, data_info,
        strength_thresholds=strength_thresholds)

    # 5. 可視化
    run_visualization(model, run_output_dir, val_loader=val_loader, test_loader=test_loader)

    # 6. Ensemble 評価 (USE_ENSEMBLE=True かつ ENSEMBLE_CHECKPOINT_PATHS が非空の場合)
    if getattr(config, 'USE_ENSEMBLE', False) and getattr(config, 'ENSEMBLE_CHECKPOINT_PATHS', []):
        force_print("[INFO] Ensemble evaluation starting...")
        ensemble_evaluate(val_loader, test_loader, loss_fn, run_output_dir, strength_thresholds)

    # val_loader/test_loaderはここまでで用済み（run_visualization/ensemble_evaluateが最後の
    # 利用箇所）。R-Precision評価はval/test_loaderへの追加フルパスであり、学習開始からここ
    # まで生存し続けたworkerに加えてさらにworkerを常駐させることがOOMの一因だったため
    # （walk_forward fold_3 OOM調査, 2026-07-28）、一旦完全に破棄してから
    # EVAL_NUM_DATALOADER_WORKERS採用の新規ローダーを作り直す。
    _log_rss("before discarding val/test loaders for R-Precision pass")
    del val_loader, test_loader
    gc.collect()
    _log_rss("after discarding val/test loaders")
    val_loader = make_val_loader()
    test_loader = make_test_loader()
    _log_rss("after rebuilding val/test loaders (EVAL_NUM_DATALOADER_WORKERS) for R-Precision pass")

    val_topk = run_topk_evaluation_val(model, val_loader, run_output_dir, val_metrics, val_loss)

    # test評価が最も重い（巨大共起グループを含みうる）フェーズのため、val評価用の
    # ローダー・workerをここで完全に解放してから臨む。del だけでは呼び出し先
    # （run_topk_evaluation_val）のローカル参照が既に消えていても、このmain()自身の
    # フレームのval_loaderがまだ生きていれば参照が残りworkerは終了しない。ここで
    # main()自身のval_loaderを明示的に破棄することで初めて参照カウントが0になり
    # workerプロセスが実際に終了する。
    _log_rss("before discarding val_loader for test R-Precision pass")
    del val_loader
    gc.collect()
    _log_rss("after discarding val_loader")

    test_topk = run_topk_evaluation_test(model, test_loader, run_output_dir, test_metrics, test_loss)

    save_run_summary_json(
        val_metrics, test_metrics, val_topk, test_topk,
        val_loss, test_loss, run_output_dir,
    )

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
    # walk-forwardは時系列リーク検証が主目的のため、静的FREQ_CSVではなく
    # このフォールドのsplit_date時点のpoint-in-time頻度を強制的に使用する。
    config.USE_POINT_IN_TIME_FREQ    = True
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


def _wf_resolve_fold_csv_dir(fold_id, wf_run_dir):
    """フォールドの最新 run の csv ディレクトリを返す（無ければ None）。"""
    fold_dir = os.path.join(config.RESULT_SAVE_DIR, 'walk_forward',
                            os.path.basename(wf_run_dir), f'fold_{fold_id}')
    if not os.path.isdir(fold_dir):
        return None
    subdirs = sorted(d for d in os.listdir(fold_dir)
                     if os.path.isdir(os.path.join(fold_dir, d)))
    if not subdirs:
        return None
    csv_dir = os.path.join(fold_dir, subdirs[-1], 'csv')
    return csv_dir if os.path.isdir(csv_dir) else None


def _wf_plot_monthly_hitrate(monthly, wf_run_dir, fold_desc=None):
    """月次 Top-1 hit-rate を walk-forward の連続時系列としてプロットし png 保存する。

    fold_desc: {fold_id: description} 。scripts/eval/walk_forward.py の FOLDS に定義された
    優勢系統era名（例: 'Omicron early (BA.1/BA.2)'）をfold区切りラベルに併記する。
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fold_desc = fold_desc or {}
    # 'unknown' に加え、collection_date が不正な行（例: '2022/2024' 由来の '2022/20' 等）も除外する。
    # 既知issue: reference/sequences-241017_2.csv の一部レコード（Zimbabwe/Harare, release_date=
    # 2025-07-18, 217件）が Collection_Date='2022/2024' という不正値を持ち、文字列比較の split
    # ロジック（assign_fold_test_window）で偶然 fold4 のテスト窓に混入していた。DB再構築なしでの
    # 応急対応として、月ラベルが YYYY-MM 形式に一致しない行は表示から除外する。
    df = monthly[monthly['yearmonth'].astype(str).str.match(r'^\d{4}-\d{2}$')].sort_values('yearmonth')
    if df.empty:
        return
    x = df['yearmonth'].tolist()
    xi = list(range(len(x)))
    fig, ax = plt.subplots(figsize=(max(10, len(x) * 0.28), 5))
    for col, label, color in [
        ('region_hit_rate',     'Region',        '#4E79A7'),
        ('position_hit_rate',   'Base Position', '#E15759'),
        ('aa_pos_hit_rate',     'AA Position',   '#59A14F'),
        ('codon_pos_hit_rate',  'Codon Pos',     '#B07AA1'),
        ('synonymous_hit_rate', 'Synonymous',    '#EDC948'),
    ]:
        if col in df.columns:
            ax.plot(xi, df[col], marker='o', ms=3, lw=1.2, label=label, color=color)
    # フォールド境界に薄い区切り線 + fold番号/優勢系統eraラベルを入れる
    if 'fold' in df.columns:
        folds_list = df['fold'].tolist()

        def _label(fid):
            desc = fold_desc.get(fid, '')
            return f'fold{fid}: {desc}' if desc else f'fold{fid}'

        prev = None
        seg_start = 0
        for i, f in enumerate(folds_list):
            if prev is not None and f != prev:
                ax.axvline(i - 0.5, color='black', ls='--', lw=0.8)
                mid = (seg_start + i - 1) / 2
                ax.text(mid, 1.01, _label(prev), transform=ax.get_xaxis_transform(),
                        ha='center', va='bottom', fontsize=5.5, color='black', rotation=0)
                seg_start = i
            prev = f
        if folds_list:
            mid = (seg_start + len(folds_list) - 1) / 2
            ax.text(mid, 1.01, _label(prev), transform=ax.get_xaxis_transform(),
                    ha='center', va='bottom', fontsize=5.5, color='black', rotation=0)
    step = max(1, len(x) // 24)
    ax.set_xticks(xi[::step])
    ax.set_xticklabels(x[::step], rotation=90, fontsize=7)
    ax.set_ylabel('Top-1 Hit Rate (%)')
    ax.set_xlabel('Year-Month (walk-forward test timeline)')
    ax.set_title('Monthly Top-1 Hit Rate across all folds', pad=32)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=8)
    fig.tight_layout()
    ppath = os.path.join(wf_run_dir, 'monthly_hitrate_all_folds.png')
    fig.savefig(ppath, dpi=130)
    plt.close(fig)
    force_print(f"[WF-Pool] monthly hitrate plot → {ppath}")


def _wf_plot_nll_trend(nll_frames, fold_desc, wf_run_dir):
    """タスク別 held-out NLL の fold横断トレンド（5小図）を保存する。
    nll_pooled_all_folds.csv が全fold平均を1値に潰すのに対し、こちらはfold別の値を残し
    時代（era）による難易度変化を見る。
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pandas as pd

    if not nll_frames:
        return
    cat = pd.concat(nll_frames, ignore_index=True)
    folds_sorted = sorted(cat['fold'].unique())
    xtick_labels = [f"fold{f}\n{fold_desc.get(f, '')}" for f in folds_sorted]
    tasks = ['region', 'position', 'aa_pos', 'codon_pos', 'synonymous']

    fig, axes = plt.subplots(1, len(tasks), figsize=(4 * len(tasks), 5), sharex=True)
    for ax, task in zip(axes, tasks):
        sub = cat[cat['task'] == task].set_index('fold').reindex(folds_sorted)
        ax.plot(range(len(folds_sorted)), sub['nll_nats'], marker='o', color='#c0392b')
        ax.set_title(task)
        ax.set_xticks(range(len(folds_sorted)))
        ax.set_xticklabels(xtick_labels, fontsize=7, rotation=45, ha='right')
        ax.set_ylabel('NLL (nats)')
        ax.grid(alpha=0.3)
    fig.suptitle('Held-out NLL trend across folds (test split)')
    plt.tight_layout()
    path = os.path.join(wf_run_dir, 'nll_trend_by_fold.png')
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    force_print(f"[WF-Pool] NLL trend plot → {path}")


def _wf_plot_confident_subset_trend(conf_frames, fold_desc, wf_run_dir):
    """確信度ベース coverage-vs-hit_rate 曲線を fold 別に色分けして重ね描きする（タスク別5小図）。
    confident_subset_pooled_all_folds.csv が全fold加重平均に潰すのに対し、こちらはfold別の
    曲線を残し、確信度による絞り込みが時代を跨いで同じように効くかを見る。
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pandas as pd

    if not conf_frames:
        return
    cat = pd.concat(conf_frames, ignore_index=True)
    folds_sorted = sorted(cat['fold'].unique())
    cmap = plt.get_cmap('viridis', max(len(folds_sorted), 2))
    tasks = ['region', 'position', 'aa_pos', 'codon_pos', 'synonymous']

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True)
    axes = axes.flatten()
    for ax, task in zip(axes, tasks):
        for i, fid in enumerate(folds_sorted):
            sub = cat[(cat['task'] == task) & (cat['fold'] == fid)].sort_values('coverage')
            ax.plot(sub['coverage'], sub['hit_rate_pct'], marker='o', color=cmap(i),
                     label=f"fold{fid}: {fold_desc.get(fid, '')}")
        ax.invert_xaxis()
        ax.set_title(task)
        ax.set_xlabel('Coverage (1.0=all, 0.1=top10% confident)')
        ax.set_ylabel('Hit Rate (%)')
        ax.grid(alpha=0.3)
    axes[-1].axis('off')
    handles, labels = axes[0].get_legend_handles_labels()
    axes[-1].legend(handles, labels, title='Fold (era)', fontsize=7, loc='center')
    fig.suptitle('Confidence-based coverage curve across folds (test split)')
    plt.tight_layout()
    path = os.path.join(wf_run_dir, 'confident_subset_trend_by_fold.png')
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    force_print(f"[WF-Pool] confident subset trend plot → {path}")


def _wf_plot_region_trend(reg_frames, fold_desc, wf_run_dir, top_n=12):
    """遺伝子(region)別 recall の fold横断トレンドを、サンプル数上位 top_n 遺伝子に絞って描く。
    region_metrics_pooled_all_folds.csv が全fold合算の1値に潰すのに対し、こちらはfold別の
    値を残し、どの遺伝子が時代とともに悪化/改善するかを見る。
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pandas as pd

    if not reg_frames:
        return
    cat = pd.concat(reg_frames, ignore_index=True)
    folds_sorted = sorted(cat['fold'].unique())
    xtick_labels = [f"fold{f}\n{fold_desc.get(f, '')}" for f in folds_sorted]
    top_regions = (cat.groupby('region_name')['sample_count'].sum()
                   .sort_values(ascending=False).head(top_n).index.tolist())
    cmap = plt.get_cmap('tab20', max(len(top_regions), 2))

    fig, ax = plt.subplots(figsize=(max(12, len(folds_sorted) * 1.8), 7))
    for i, region in enumerate(top_regions):
        sub = cat[cat['region_name'] == region].set_index('fold').reindex(folds_sorted)
        ax.plot(range(len(folds_sorted)), sub['recall_pct'], marker='o', color=cmap(i), label=region)
    ax.set_xticks(range(len(folds_sorted)))
    ax.set_xticklabels(xtick_labels, fontsize=8, rotation=45, ha='right')
    ax.set_ylabel('Recall (%)')
    ax.set_title(f'Region-level recall trend across folds (test split)\n'
                 f'Top {len(top_regions)} genes by total sample count')
    ax.legend(fontsize=7, ncol=2, loc='best')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(wf_run_dir, 'region_recall_trend_by_fold.png')
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    force_print(f"[WF-Pool] region recall trend plot → {path}")


def _wf_plot_strength_trend(strength_frames, fold_desc, wf_run_dir):
    """流行度カテゴリ(9段階)別精度の fold横断トレンドをタスク別5小図で描く。
    category_metrics_all_folds.csv は低頻度〜高頻度をlow/medium/highの3段階に潰すのに対し、
    こちらは test_strength_fine.csv の9段階ビンをそのままfold別トレンドとして残す。
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pandas as pd

    if not strength_frames:
        return
    cat = pd.concat(strength_frames, ignore_index=True)
    folds_sorted = sorted(cat['fold'].unique())
    xtick_labels = [f"fold{f}\n{fold_desc.get(f, '')}" for f in folds_sorted]
    task_cols = ['region_hit_rate_pct', 'position_hit_rate_pct', 'aa_pos_hit_rate_pct',
                 'codon_pos_hit_rate_pct', 'synonymous_hit_rate_pct']
    category_order = ['~100', '~500', '~1K', '~5K', '~10K', '~50K', '~100K', '~500K', '>500K']
    categories = [c for c in category_order if c in set(cat['strength_category'])]
    cmap = plt.get_cmap('plasma', max(len(categories), 2))

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True)
    axes = axes.flatten()
    for ax, col in zip(axes, task_cols):
        for i, sc in enumerate(categories):
            sub = cat[cat['strength_category'] == sc].set_index('fold').reindex(folds_sorted)
            ax.plot(range(len(folds_sorted)), sub[col], marker='o', color=cmap(i), label=sc)
        ax.set_title(col.replace('_hit_rate_pct', ''))
        ax.set_xticks(range(len(folds_sorted)))
        ax.set_xticklabels(xtick_labels, fontsize=7, rotation=45, ha='right')
        ax.set_ylabel('Hit Rate (%)')
        ax.grid(alpha=0.3)
    axes[-1].axis('off')
    handles, labels = axes[0].get_legend_handles_labels()
    axes[-1].legend(handles, labels, title='Strength category\n(prevalence, low→high)',
                     fontsize=8, loc='center')
    fig.suptitle('Strength-category accuracy trend across folds (test split)')
    plt.tight_layout()
    path = os.path.join(wf_run_dir, 'strength_category_trend_by_fold.png')
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    force_print(f"[WF-Pool] strength category trend plot → {path}")


def _wf_plot_diversity_trend(diversity_frames, fold_desc, wf_run_dir):
    """系統粒度の entropy_norm / unique_ratio / simpson vs 位置正解率を、fold別に色分けして
    3枚重ね描きする（utils/plotting.py の3種散布図を fold 単位の色分けに差し替えた版）。
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    if not diversity_frames:
        return
    cat = pd.concat(diversity_frames, ignore_index=True)
    folds_sorted = sorted(cat['fold'].unique())
    cmap = plt.get_cmap('viridis', max(len(folds_sorted), 2))
    metric_specs = [
        ('entropy_norm', 'Normalized Target-Position Entropy'),
        ('unique_ratio', 'Unique Target-Position Ratio'),
        ('simpson',      'Gini-Simpson Diversity'),
    ]
    log_n_all = np.log1p(cat['n'])

    for col, label in metric_specs:
        fig, ax = plt.subplots(figsize=(9, 6))
        for i, fid in enumerate(folds_sorted):
            sub = cat[cat['fold'] == fid]
            log_n = np.log1p(sub['n'])
            sizes = 20 + (log_n - log_n_all.min()) / (log_n_all.max() - log_n_all.min() + 1e-9) * 400
            ax.scatter(sub[col], sub['hit_rate'], s=sizes, alpha=0.65, color=cmap(i),
                       edgecolors='gray', linewidths=0.3,
                       label=f"fold{fid}: {fold_desc.get(fid, '')}")
        ax.set_xlabel(label)
        ax.set_ylabel('Position Hit Rate (%)')
        ax.set_title(f'{label} vs Position Hit Rate — all folds overlaid\n'
                     f'(per-lineage, test split, point size ∝ log(n_samples))')
        ax.legend(fontsize=7, loc='best')
        ax.grid(linestyle='--', alpha=0.35)
        plt.tight_layout()
        path = os.path.join(wf_run_dir, f'diversity_{col}_trend_by_fold.png')
        fig.savefig(path, dpi=130, bbox_inches='tight')
        plt.close(fig)
        force_print(f"[WF-Pool] diversity ({col}) trend plot → {path}")


def _wf_pool_test_outputs(folds, wf_run_dir):
    """全フォールドの test 出力を串刺し集計して wf_run_dir 直下へ保存する。

    出力:
      - monthly_metrics_all_folds.csv / monthly_hitrate_all_folds.png
          各フォールドの test_metrics_by_date.csv に fold 列を付けて縦結合。
          walk-forward はテスト期間が非重複なので月次で連続した時系列になる。
      - entropy_bin_pooled_all_folds.csv
          各フォールドの test_entropy_bin.csv を bin 単位で num_samples 重み付き集計。
      - nll_pooled_all_folds.csv（推奨追加）
          各フォールドの test_nll.csv を n_samples 重み付きでプールした held-out NLL。
      - confident_subset_pooled_all_folds.csv（推奨追加・C-4）
          各フォールドの test_confident_subset.csv を (task, coverage) 単位で n 重み付き集計。
      - nll_trend_by_fold.png / confident_subset_trend_by_fold.png / region_recall_trend_by_fold.png
        / strength_category_trend_by_fold.png / diversity_{entropy_norm,unique_ratio,simpson}_trend_by_fold.png
          上記の「全fold集約1値」とは別に、fold別の値を残した時系列トレンド版
          （scripts/analysis/walk_forward/overlay_*.py と同じ可視化をこの関数内に統合したもの）。
    ソース CSV が無いフォールド/指標はスキップする（部分実行・旧フォールドでも落ちない）。
    """
    import pandas as pd
    import math

    fold_csv = {}
    for fold_id, *_ in folds:
        d = _wf_resolve_fold_csv_dir(fold_id, wf_run_dir)
        if d:
            fold_csv[fold_id] = d
    if not fold_csv:
        force_print("[WF-Pool] フォールドの csv が見つからないため集計をスキップします。")
        return
    fold_desc = {fid: desc for fid, _ts, _sd, _se, desc in folds}

    def _read_all(fname):
        frames = []
        for fold_id, d in fold_csv.items():
            p = os.path.join(d, fname)
            if os.path.exists(p):
                df = pd.read_csv(p)
                df['fold'] = fold_id
                frames.append(df)
        return frames

    def _wmean(g, col, w):
        """num_samples 等の重み w による col の加重平均（W<=0 なら 0）。"""
        W = float(w.sum())
        return (g[col] * w).sum() / W if W > 0 else 0.0

    # --- 1) monthly metrics 縦結合 + プロット ---
    monthly_frames = _read_all('test_metrics_by_date.csv')

    # フォールド別テストサンプル数（topk/r_precision の重み付けに使う）
    fold_n = {}
    for fr in monthly_frames:
        try:
            fold_n[int(fr['fold'].iloc[0])] = float(fr['num_samples'].sum())
        except Exception:
            pass

    if monthly_frames:
        monthly = pd.concat(monthly_frames, ignore_index=True)
        # 収集日不明（'unknown'）は時系列に載らないため除外（260625 と同挙動）。
        # 加えて collection_date が不正な行（既知issue: '2022/2024' → yearmonth='2022/20'、
        # reference/sequences-241017_2.csv のZimbabwe/Harare 217件、詳細は上記コメント参照）も除外する。
        monthly = monthly[monthly['yearmonth'].astype(str).str.match(r'^\d{4}-\d{2}$')]
        monthly = monthly.sort_values(['yearmonth', 'fold']).reset_index(drop=True)
        mpath = os.path.join(wf_run_dir, 'monthly_metrics_all_folds.csv')
        monthly.to_csv(mpath, index=False)
        force_print(f"[WF-Pool] monthly metrics → {mpath}")
        _wf_plot_monthly_hitrate(monthly, wf_run_dir, fold_desc=fold_desc)

    # --- 2) entropy_bin プール（num_samples 重み付き）---
    ent_frames = _read_all('test_entropy_bin.csv')
    if ent_frames:
        cat = pd.concat(ent_frames, ignore_index=True)
        hit_cols = [c for c in cat.columns if c.endswith('_hit_rate_pct')]
        rows = []
        for b, g in cat.groupby('entropy_bin'):
            n = float(g['num_samples'].sum())
            if n <= 0:
                continue  # 全フォールドで0件のビンは出さない（260625 と同挙動）
            row = {'entropy_bin': b, 'num_samples': int(n),
                   # num_strains はフォールド跨ぎの重複を保持したまま合算（旧 260625 と同挙動）
                   'num_strains': int(g['num_strains'].sum())}
            for c in hit_cols:
                row[c] = round((g[c] * g['num_samples']).sum() / n, 2) if n > 0 else 0.0
            rows.append(row)
        pooled = pd.DataFrame(rows).sort_values('entropy_bin').reset_index(drop=True)
        epath = os.path.join(wf_run_dir, 'entropy_bin_pooled_all_folds.csv')
        pooled.to_csv(epath, index=False)
        force_print(f"[WF-Pool] entropy_bin pooled → {epath}")

    # --- 3) held-out NLL プール（n_samples 重み付き）---
    nll_frames = _read_all('test_nll.csv')
    if nll_frames:
        cat = pd.concat(nll_frames, ignore_index=True)
        rows = []
        for t, g in cat.groupby('task'):
            n = float(g['n_samples'].sum())
            if n <= 0:
                continue
            mean_nll = (g['nll_nats'] * g['n_samples']).sum() / n
            rows.append({'task': t, 'n_samples': int(n),
                         'nll_nats': round(mean_nll, 4),
                         'nll_bits': round(mean_nll / math.log(2), 4),
                         'perplexity': round(math.exp(mean_nll), 2)})
        if rows:
            npath = os.path.join(wf_run_dir, 'nll_pooled_all_folds.csv')
            pd.DataFrame(rows).to_csv(npath, index=False)
            force_print(f"[WF-Pool] NLL pooled → {npath}")
    _wf_plot_nll_trend(nll_frames, fold_desc, wf_run_dir)

    # --- 4) confident-subset プール（C-4, n 重み付き）---
    conf_frames = _read_all('test_confident_subset.csv')
    if conf_frames:
        cat = pd.concat(conf_frames, ignore_index=True)
        rows = []
        for (t, cov), g in cat.groupby(['task', 'coverage']):
            n = float(g['n'].sum())
            if n <= 0:
                continue
            rows.append({'task': t, 'coverage': cov, 'n': int(n),
                         'avg_confidence': round((g['avg_confidence'] * g['n']).sum() / n, 4),
                         'hit_rate_pct':   round((g['hit_rate_pct'] * g['n']).sum() / n, 3)})
        if rows:
            cpath = os.path.join(wf_run_dir, 'confident_subset_pooled_all_folds.csv')
            (pd.DataFrame(rows)
               .sort_values(['task', 'coverage'], ascending=[True, False])
               .to_csv(cpath, index=False))
            force_print(f"[WF-Pool] confident subset pooled → {cpath}")
    _wf_plot_confident_subset_trend(conf_frames, fold_desc, wf_run_dir)

    # --- 5) Top-K precision プール（フォールドのテストサンプル数で重み付け）---
    # 注: per-fold CSV に tp/total 生カウントが無いため厳密な micro 平均ではなく、
    #     フォールドのテストサンプル数による加重平均（全指標が近似値）。厳密値が要る場合は
    #     evaluate_topk に生カウントを持たせる拡張が必要。region 別は生カウントがあるので厳密。
    topk_frames = _read_all('test_topk_precision.csv')
    if topk_frames:
        cat = pd.concat(topk_frames, ignore_index=True)
        w = cat['fold'].map(lambda f: fold_n.get(int(f), 1.0))
        cat = cat.assign(_w=w)
        rows = []
        for (t, k), g in cat.groupby(['task', 'top_k']):
            rows.append({'task': t, 'top_k': int(k),
                         'precision_pct': round(_wmean(g, 'precision_pct', g['_w']), 3),
                         'recall_pct':    round(_wmean(g, 'recall_pct',    g['_w']), 3),
                         'hit_rate_pct':  round(_wmean(g, 'hit_rate_pct',  g['_w']), 3),
                         'n_samples': int(g['_w'].sum())})
        tpath = os.path.join(wf_run_dir, 'topk_precision_pooled_all_folds.csv')
        pd.DataFrame(rows).sort_values(['task', 'top_k']).to_csv(tpath, index=False)
        force_print(f"[WF-Pool] topk precision pooled → {tpath}")

    # --- 6) R-Precision プール（同上・サンプル数重み付き）---
    rp_frames = _read_all('test_r_precision.csv')
    if rp_frames:
        cat = pd.concat(rp_frames, ignore_index=True)
        cat = cat.assign(_w=cat['fold'].map(lambda f: fold_n.get(int(f), 1.0)))
        rows = []
        for t, g in cat.groupby('task'):
            rows.append({'task': t,
                         'precision_pct': round(_wmean(g, 'precision_pct', g['_w']), 4),
                         'recall_pct':    round(_wmean(g, 'recall_pct',    g['_w']), 4),
                         'hit_rate_pct':  round(_wmean(g, 'hit_rate_pct',  g['_w']), 4),
                         'n_samples': int(g['_w'].sum())})
        rppath = os.path.join(wf_run_dir, 'r_precision_pooled_all_folds.csv')
        pd.DataFrame(rows).to_csv(rppath, index=False)
        force_print(f"[WF-Pool] r_precision pooled → {rppath}")

    # --- 7) 流行度カテゴリ別プール（combined_metrics_by_category.csv の test 行）---
    # timestep×category を category(low/medium/high) に集約し num_samples 加重平均。
    catg_frames = _read_all('combined_metrics_by_category.csv')
    if catg_frames:
        cat = pd.concat(catg_frames, ignore_index=True)
        cat = cat[cat['split'] == 'test']
        hit_cols = ['region_hit_rate', 'position_hit_rate', 'aa_pos_hit_rate',
                    'codon_pos_hit_rate', 'synonymous_hit_rate']
        rows = []
        for c in ['low', 'medium', 'high']:
            g = cat[cat['category'] == c]
            n = float(g['num_samples'].sum())
            row = {'category': c, 'num_samples': int(n),
                   'strength_mae': round(_wmean(g, 'strength_mae', g['num_samples']), 4) if n > 0 else 0.0}
            for hc in hit_cols:
                row[hc] = round(_wmean(g, hc, g['num_samples']), 3) if n > 0 else 0.0
            rows.append(row)
        cgpath = os.path.join(wf_run_dir, 'category_metrics_all_folds.csv')
        pd.DataFrame(rows).to_csv(cgpath, index=False)
        force_print(f"[WF-Pool] category metrics pooled → {cgpath}")

    # --- 8) 遺伝子領域別プール（test_region_metrics.csv, tp/fp/fn の生カウントで厳密集計）---
    reg_frames = _read_all('test_region_metrics.csv')
    if reg_frames:
        cat = pd.concat(reg_frames, ignore_index=True)
        rows = []
        for rid, g in cat.groupby('region_id'):
            sc = int(g['sample_count'].sum())
            tp = float(g['true_positives'].sum())
            fp = float(g['false_positives'].sum())
            fn = float(g['false_negatives'].sum())
            prec = 100.0 * tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec  = 100.0 * tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            rows.append({'region_id': int(rid),
                         'region_name': g['region_name'].iloc[0],
                         'sample_count': sc,
                         'true_positives': int(tp), 'false_positives': int(fp),
                         'false_negatives': int(fn),
                         'precision_pct': round(prec, 4), 'recall_pct': round(rec, 4),
                         'f1_pct': round(f1, 4)})
        rgpath = os.path.join(wf_run_dir, 'region_metrics_pooled_all_folds.csv')
        (pd.DataFrame(rows).sort_values('sample_count', ascending=False)
           .to_csv(rgpath, index=False))
        force_print(f"[WF-Pool] region metrics pooled → {rgpath}")
    _wf_plot_region_trend(reg_frames, fold_desc, wf_run_dir)

    # --- 9) 流行度カテゴリ(9段階)別トレンド（test_strength_fine.csv, fold別に残す）---
    strength_frames = _read_all('test_strength_fine.csv')
    _wf_plot_strength_trend(strength_frames, fold_desc, wf_run_dir)

    # --- 10) 系統別多様度(entropy/unique_ratio/simpson) vs 精度トレンド ---
    diversity_frames = _read_all('test_lineage_diversity.csv')
    _wf_plot_diversity_trend(diversity_frames, fold_desc, wf_run_dir)


def _wf_find_prev_fold_checkpoint(fold_id):
    """fold_id-1 の best_model.pth を既存の walk_forward 結果ディレクトリから自動探索する。

    `--folds 2` や `--folds 2 6` のような部分/非連続fold実行では、fold_id-1がこの
    実行内で直前に処理されていないため、ループ内の prev_best_model_path が使えない。
    outputs/.../results/walk_forward/*/fold_{fold_id-1}/*/models/best_model.pth を
    全て探し、最終更新時刻が最も新しいものを返す（見つからなければ None）。
    """
    import glob
    pattern = os.path.join(config.RESULT_SAVE_DIR, 'walk_forward', '*',
                           f'fold_{fold_id - 1}', '*', 'models', 'best_model.pth')
    candidates = glob.glob(pattern)
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def run_walk_forward(folds, wf_run_dir: str):
    """Walk-forward 検証：フォールドを順番に学習＋評価して集計する。

    設計:
      - Fold 1: 全歴史データ (< split_date) で学習。USE_PRETRAINING=True なら事前学習も実行。
      - Fold N (N>=2): 直前フォールドの best_model.pth を起点に、直近6ヶ月のみで学習 (Method B)。
      - 事前学習は Fold 1 のみ。Fold 2 以降は WF_PREV_FOLD_CHECKPOINT で初期化。

    `--folds` で部分/非連続実行する場合（例: fold_id==1を含まない、またはfold間に
    抜けがある選択）、直前フォールドのcheckpointはこのループ内に存在しないため、
    `WF_PREV_CHECKPOINT_OVERRIDE`（明示指定）または`_wf_find_prev_fold_checkpoint`
    （既存結果からの自動探索）で解決する。見つからなければ、事前学習にフォール
    バックしてしまう（＝Method Bの重み引き継ぎが行われない）事故を防ぐため
    RuntimeErrorを送出する。

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
    prev_fold_id = None  # このループ内で直近に完了した fold_id

    for idx, (fold_id, train_start, split_date, split_end, desc) in enumerate(folds):
        _log_rss(f"run_walk_forward: before fold_{fold_id}")
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

        # 3. 直前フォールドの重みを解決する。
        #    fold_id==1: 直前foldは存在しない（事前学習 or ランダム初期化）。
        #    このループ内で fold_id-1 を直前に処理済み: そのcheckpointをそのまま使う
        #    （フル/連続実行時の従来通りの挙動）。
        #    それ以外（部分/非連続実行でfold_id-1がこの実行内にない）:
        #    WF_PREV_CHECKPOINT_OVERRIDE か既存結果からの自動探索で解決し、
        #    見つからなければ誤って事前学習に落ちないようエラーで止める。
        if fold_id == 1:
            resolved_prev_ckpt = None
        elif prev_fold_id == fold_id - 1:
            resolved_prev_ckpt = prev_best_model_path
        else:
            resolved_prev_ckpt = getattr(config, 'WF_PREV_CHECKPOINT_OVERRIDE', None)
            if resolved_prev_ckpt is None:
                resolved_prev_ckpt = _wf_find_prev_fold_checkpoint(fold_id)
            if resolved_prev_ckpt is None:
                raise RuntimeError(
                    f"Fold {fold_id} の部分/非連続実行には直前フォールド(fold_{fold_id - 1})の"
                    f"checkpointが必要ですが、既存の walk_forward 結果からも見つかりませんでした"
                    f"（{os.path.join(config.RESULT_SAVE_DIR, 'walk_forward', '*', f'fold_{fold_id - 1}', '*', 'models', 'best_model.pth')}）。"
                    f"--prev_checkpoint で明示的に指定するか、fold_{fold_id - 1}を先に実行してください。"
                )
            force_print(f"[INFO] 部分/非連続実行検出: fold_{fold_id - 1}のcheckpointを自動使用: "
                        f"{resolved_prev_ckpt}")

        # 4. Fold 1 のみ事前学習を実行（fold_idベース。--folds でfold 1以外を単独/先頭
        #    実行しても誤って事前学習がトリガーされないようにする）
        if fold_id == 1 and getattr(config, 'USE_PRETRAINING', False):
            force_print(f"[INFO] Walk-forward Fold {fold_id}: 事前学習を開始 (split_type_wf 基準)...")
            from .scripts.train.pretrain import run_pretraining
            run_pretraining()

        config.WF_PREV_FOLD_CHECKPOINT = resolved_prev_ckpt

        # 5. 学習・評価
        prev_best_model_path = main()
        prev_fold_id = fold_id

        # fold境界での明示的な解放。main()内のtrain/val/test loader（persistent_workers=True
        # の場合、各8 workerプロセスと_group_label_cacheを抱える）はmain()のローカル変数
        # なのでリターン時に参照は切れるが、workerプロセスの終了(SIGTERM送信・join)には
        # 実時間がかかる。ここでgc.collect()を挟まずに次foldのprepare_data()が新規に
        # 24 workerを立ち上げると、前foldの旧workerがまだ完全終了しきっていない状態と
        # 一時的に重なり、メモリを圧迫しうる（walk_forward fold_3 OOM調査より、2026-07-27）。
        import gc
        gc.collect()
        _log_rss(f"run_walk_forward: after fold_{fold_id} + gc.collect()")

        force_print(f"[INFO] Fold {fold_id} complete. Best model: {prev_best_model_path}")

    _wf_summarize(folds, wf_run_dir)
    _wf_pool_test_outputs(folds, wf_run_dir)
    force_print(f"\n[INFO] Walk-forward complete. Results root: {wf_run_dir}")


def _aggregate_seed_summaries(summaries):
    """複数 run_summary の数値リーフを再帰的に mean/std/n へ集計する。

    構造（ネストした dict）を保ったまま、各数値リーフを
    {'mean': ..., 'std': ..., 'n': ..., 'values': [...]} に置き換えた dict を返す。
    """
    from statistics import mean as _mean, stdev as _stdev

    def _agg(dicts):
        keys = set()
        for d in dicts:
            if isinstance(d, dict):
                keys |= set(d.keys())
        out = {}
        for k in sorted(keys):
            vals = [d[k] for d in dicts if isinstance(d, dict) and k in d]
            if vals and all(isinstance(v, dict) for v in vals):
                out[k] = _agg(vals)
            else:
                nums = [v for v in vals
                        if isinstance(v, (int, float)) and not isinstance(v, bool)]
                if nums:
                    out[k] = {
                        'mean': _mean(nums),
                        'std': _stdev(nums) if len(nums) > 1 else 0.0,
                        'n': len(nums),
                        'values': nums,
                    }
        return out

    return _agg([s for s in summaries if isinstance(s, dict)])


def run_multi_seed(seeds, ms_run_dir):
    """複数シードで main() を回し、run_summary を mean±std に集計する。

    各シードは EXPERIMENT_NAME を multi_seed/<ts>/seed_<n> に切り替えて実行し、
    出力された run_summary.json を回収して集計結果を ms_run_dir に保存する。
    """
    import json as _json
    import glob as _glob

    orig_seed = config.SEED
    orig_exp = getattr(config, 'EXPERIMENT_NAME', '')
    ms_tag = os.path.basename(ms_run_dir)

    per_seed = {}
    for seed in seeds:
        force_print(f"\n{'='*60}")
        force_print(f"  Multi-seed run: SEED={seed}")
        force_print(f"{'='*60}")
        config.SEED = seed
        config.EXPERIMENT_NAME = f'multi_seed/{ms_tag}/seed_{seed}'
        best_model_path = main()

        summary = None
        if best_model_path:
            run_dir = os.path.dirname(os.path.dirname(best_model_path))
            hits = _glob.glob(os.path.join(run_dir, '**', 'run_summary.json'),
                              recursive=True)
            if hits:
                with open(hits[0]) as f:
                    summary = _json.load(f)
        per_seed[str(seed)] = summary
        force_print(f"[INFO] SEED={seed} complete "
                    f"(run_summary: {'found' if summary else 'MISSING'})")

    # config を元に戻す
    config.SEED = orig_seed
    config.EXPERIMENT_NAME = orig_exp

    valid_summaries = [v for v in per_seed.values() if isinstance(v, dict)]
    result = {
        'seeds': list(seeds),
        'n_success': len(valid_summaries),
        'aggregate': _aggregate_seed_summaries(valid_summaries),
        'per_seed': per_seed,
    }
    out_path = os.path.join(ms_run_dir, 'multi_seed_summary.json')
    with open(out_path, 'w') as f:
        _json.dump(result, f, indent=2, ensure_ascii=False)
    force_print(f"\n[INFO] Multi-seed complete ({len(valid_summaries)}/{len(seeds)} seeds). "
                f"Aggregate saved to {out_path}")
    return result


def _run_entry():
    """config.SPLIT_MODE / MULTI_SEED に応じて実行方式を切り替えるエントリポイント。"""
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
    elif getattr(config, 'MULTI_SEED', False):
        seeds = getattr(config, 'MULTI_SEED_LIST', None) or [config.SEED]
        ms_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ms_run_dir = os.path.join(config.RESULT_SAVE_DIR, 'multi_seed', ms_timestamp)
        os.makedirs(ms_run_dir, exist_ok=True)
        run_multi_seed(seeds, ms_run_dir)
    else:
        main()


if __name__ == "__main__":
    _run_entry()
