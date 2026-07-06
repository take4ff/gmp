# --- transformer_260707/scripts/evaluate_only.py ---
# 既存チェックポイントを使って評価のみ実行する。学習は行わない。
#
# Usage:
#   python -m transformer_260707.scripts.evaluate_only \
#       --checkpoint outputs/transformer_260707/results/<timestamp>/models/best_model.pth
#
# 結果は元の実験ディレクトリ配下に <timestamp>_eval/ として保存される。
import os
import argparse
from datetime import datetime

import torch

from transformer_260707 import config
from transformer_260707.model import HierarchicalTransformer, MultiTaskLoss
from transformer_260707.evaluate import evaluate, evaluate_topk
from transformer_260707.utils.losses import build_loss_fn
from transformer_260707.utils.logging import force_print, print_config
from transformer_260707.utils.io import (
    save_config_copy,
    save_metrics_csv, save_category_metrics_csv, save_combined_metrics_csv,
    save_lineage_metrics_csv, save_region_metrics_csv,
    save_per_position_recall_csv, save_error_analysis_csv,
    save_val_test_gap_csv, save_strength_fine_csv,
    save_topk_precision_csv, save_date_metrics_csv,
    save_strain_metrics_csv,
)
from transformer_260707.utils.plotting import (
    plot_metrics_by_timestep, plot_category_metrics,
    plot_combined_val_test_metrics, plot_combined_category_comparison,
    print_combined_report, plot_strength_calibration,
    plot_per_position_recall, plot_topk_precision, plot_metrics_by_date,
)
from transformer_260707.main import prepare_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='best_model.pth へのパス')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='結果保存先（省略時は checkpoint の親ディレクトリ/../<timestamp>_eval/）')
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    # 出力ディレクトリ決定
    if args.output_dir:
        run_output_dir = args.output_dir
    else:
        exp_dir = os.path.dirname(os.path.dirname(args.checkpoint))  # models/ の親
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_output_dir = os.path.join(exp_dir, f"{timestamp}_eval")
    os.makedirs(run_output_dir, exist_ok=True)
    force_print(f"Output dir: {run_output_dir}")

    # チェックポイントと同じディレクトリの config_snapshot.py で config を上書き
    checkpoint_dir = os.path.dirname(args.checkpoint)
    snapshot_path = os.path.join(checkpoint_dir, 'config_snapshot.py')
    if os.path.exists(snapshot_path):
        import importlib.util
        spec = importlib.util.spec_from_file_location("config_snapshot", snapshot_path)
        snap = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(snap)
        overridden = []
        for attr in dir(snap):
            if not attr.startswith('_') and hasattr(config, attr):
                setattr(config, attr, getattr(snap, attr))
                overridden.append(attr)
        force_print(f"[INFO] Loaded config_snapshot.py from {snapshot_path} ({len(overridden)} attrs overridden)")
    else:
        force_print(f"[WARNING] config_snapshot.py not found at {snapshot_path}. Using current config.")

    print_config()
    save_config_copy(run_output_dir)

    # データ準備（test_loader のみ使用）
    _, val_loader, test_loader, data_info, _, _, _, _ = prepare_data()

    # strength_thresholds
    use_dynamic = getattr(config, 'DYNAMIC_STRENGTH_CATEGORY', False)
    if use_dynamic:
        low_max = data_info.get('strength_low_max', 6.0)
        med_max = data_info.get('strength_med_max', 10.0)
    else:
        low_max = getattr(config, 'STRENGTH_CATEGORY_LOW_MAX', 6.0)
        med_max = getattr(config, 'STRENGTH_CATEGORY_MED_MAX', 10.0)
    strength_thresholds = (low_max, med_max)

    # 損失関数・モデル構築
    if config.USE_DB:
        from transformer_260707.db.queries import get_train_class_counts
        class_counts_dict = get_train_class_counts()
    else:
        class_counts_dict = None

    loss_fn = build_loss_fn(class_counts_dict)
    model = HierarchicalTransformer().to(config.DEVICE)

    # チェックポイントロード
    checkpoint = torch.load(args.checkpoint, map_location=config.DEVICE, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint.get('epoch', -1)
    force_print(f"Loaded checkpoint: epoch={epoch + 1}")
    model.eval()

    def _save_results(metrics, cat_metrics, details, prefix):
        save_metrics_csv(metrics, run_output_dir, prefix=prefix)
        save_category_metrics_csv(cat_metrics, run_output_dir, prefix=prefix)
        save_strength_fine_csv(details, run_output_dir, prefix=prefix)
        save_lineage_metrics_csv(details, run_output_dir, prefix=prefix)
        save_region_metrics_csv(details, run_output_dir, prefix=prefix)
        save_per_position_recall_csv(details, run_output_dir, prefix=prefix)
        save_error_analysis_csv(details, run_output_dir, prefix=prefix)
        if getattr(config, 'SAVE_STRAIN_METRICS', False):
            save_strain_metrics_csv(details, run_output_dir, prefix=prefix)
        plot_metrics_by_timestep(metrics, run_output_dir, prefix=prefix)
        plot_category_metrics(cat_metrics, run_output_dir, prefix=prefix)
        plot_strength_calibration(details, run_output_dir, prefix=prefix)
        plot_per_position_recall(details, run_output_dir, prefix=prefix,
                                 top_n=getattr(config, "PLOT_TOP_N_POSITIONS", 40))

    # Validation
    force_print("Evaluating Validation Set...")
    val_loss, val_metrics, val_details, val_cat_metrics, val_metrics_ym, _ = evaluate(
        model, val_loader, loss_fn, strength_thresholds
    )
    _save_results(val_metrics, val_cat_metrics, val_details, prefix="valid")

    # Test
    test_loss, test_metrics, test_details, test_cat_metrics, test_metrics_ym = None, None, None, None, {}
    if len(test_loader) > 0:
        force_print("Evaluating Test Set...")
        test_loss, test_metrics, test_details, test_cat_metrics, test_metrics_ym, _ = evaluate(
            model, test_loader, loss_fn, strength_thresholds
        )
        _save_results(test_metrics, test_cat_metrics, test_details, prefix="test")

    # 統合レポート
    print_combined_report(val_details, test_details, strength_thresholds)
    if test_details:
        plot_combined_val_test_metrics(val_metrics, test_metrics, run_output_dir)
        plot_combined_category_comparison(val_details, test_details, strength_thresholds, run_output_dir)
        save_val_test_gap_csv(val_metrics, test_metrics, run_output_dir)

    # 月別 CSV（常に保存）
    save_date_metrics_csv(val_metrics_ym,  run_output_dir, prefix='valid')
    save_date_metrics_csv(test_metrics_ym, run_output_dir, prefix='test')
    plot_metrics_by_date(val_metrics_ym, test_metrics_ym, run_output_dir)

    # Top-K
    eval_ks = getattr(config, "EVAL_TOP_KS", (1, 3, 5))
    force_print(f"Evaluating Top-K (k={eval_ks}) on Validation...")
    val_topk = evaluate_topk(model, val_loader, ks=eval_ks)
    save_topk_precision_csv(val_topk, run_output_dir, prefix='valid')
    plot_topk_precision(val_topk, run_output_dir, prefix='valid')

    if test_details:
        force_print(f"Evaluating Top-K (k={eval_ks}) on Test...")
        test_topk = evaluate_topk(model, test_loader, ks=eval_ks)
        save_topk_precision_csv(test_topk, run_output_dir, prefix='test')
        plot_topk_precision(test_topk, run_output_dir, prefix='test')

    force_print(f"Done. Results saved to {run_output_dir}")


if __name__ == '__main__':
    main()
