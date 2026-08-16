# --- transformer_260817/scripts/analysis/verify/verify_point_in_time_freq.py ---
"""codon_freq (変異頻度特徴量, x_num[...,0]) の時系列リーク検証。

各 walk-forward フォールドについて、そのフォールド専用チェックポイントをロードし、
(a) 静的全期間 FREQ_CSV 由来の baseline と
(b) config.USE_POINT_IN_TIME_FREQ=True によるフォールド時点までの point-in-time 頻度
の双方で同一テストセットを評価し、Top-1 Hit Rate を比較する。

呼び出し順序に注意 (このバグで過去に誤ったフォールドを評価してしまったことがある):
  1. X.load_fold_model(checkpoint, device)       # config_snapshot.py をロード
     -> config_snapshot.py は保存時の config.py ファイルをそのままコピーしたもので、
        walk_forward.py がループ内で動的に上書きした TEMPORAL_SPLIT_DATE 等は
        反映されていない（常に config.py のファイル上のデフォルト値になる）。
  2. X.assign_fold_test_window(db_path, train_start, split_date, split_end)
     -> 上記を明示的にフォールド固有の値で上書きし、DB の split_type_wf 列も
        該当ウィンドウに更新する。必ず load_fold_model の「後」に呼ぶこと。

Usage:
  python -m transformer_260817.scripts.analysis.verify.verify_point_in_time_freq \\
      --walk_forward_dir outputs/transformer_260702/results/walk_forward/20260706_223816 \\
      --folds 1 2
"""

import argparse

from transformer_260817 import config
from transformer_260817.utils.logging import force_print
from transformer_260817.evaluate import evaluate_topk
from transformer_260817.scripts.analysis.xai import _xai_common as X


def _run_one(model, loader, label):
    result = evaluate_topk(model, loader, ks=(1,))
    force_print(f"=== {label} ===")
    for task in ['region', 'position', 'aa_pos', 'codon_pos', 'synonymous']:
        hr = result[task][1]['hit_rate']  # 既に % 表記 (evaluate_topk の仕様)
        force_print(f"  {task} top1 hit_rate: {hr:.3f}%")
    force_print("")
    return {task: result[task][1]['hit_rate'] for task in
             ['region', 'position', 'aa_pos', 'codon_pos', 'synonymous']}


def main():
    parser = argparse.ArgumentParser(description='codon_freq point-in-time leak verification')
    parser.add_argument('--walk_forward_dir', type=str, required=True)
    parser.add_argument('--folds', type=int, nargs='+', default=None)
    parser.add_argument('--force_cpu', action='store_true')
    args = parser.parse_args()

    from transformer_260817.db.connection import get_db_path
    from transformer_260817.db.dataset import create_db_dataloader

    device = 'cpu' if args.force_cpu else config.DEVICE
    db_path = get_db_path()
    fold_windows = X.get_fold_windows()
    fold_ckpts = X.discover_fold_checkpoints(args.walk_forward_dir, args.folds)
    if not fold_ckpts:
        raise SystemExit(f"[ERROR] fold チェックポイントが見つかりません: {args.walk_forward_dir}")
    force_print(f"[INFO] 発見したフォールド: {sorted(fold_ckpts)}")

    rows = []
    for fold_id in sorted(fold_ckpts):
        if fold_id not in fold_windows:
            continue
        train_start, split_date, split_end = fold_windows[fold_id]
        force_print(f"\n===== Fold {fold_id}: test window [{split_date}, {split_end}) =====")

        # 1) モデルロード（config_snapshot 反映。ただし split window はファイルのデフォルト値のまま）
        model = X.load_fold_model(fold_ckpts[fold_id], device)
        if getattr(model, 'use_flat_coattn', False):
            force_print(f"[WARN] fold {fold_id} は USE_FLAT_COATTN のためスキップ")
            continue

        # 2) split window を明示的にこのフォールド用に上書き（load_fold_model の後で必須）
        X.assign_fold_test_window(db_path, train_start, split_date, split_end)
        force_print(f"[CHECK] config.TEMPORAL_SPLIT_DATE={config.TEMPORAL_SPLIT_DATE} "
                    f"TEMPORAL_SPLIT_TEST_END={config.TEMPORAL_SPLIT_TEST_END} "
                    f"WALK_FORWARD_TRAIN_START={config.WALK_FORWARD_TRAIN_START}")

        config.USE_POINT_IN_TIME_FREQ = False
        loader_base = create_db_dataloader(
            db_path=db_path, split_type=2,
            batch_size=config.BATCH_SIZE, shuffle=False,
            max_cooccurrence=config.MAX_CO_OCCURRENCE,
        )
        res_base = _run_one(model, loader_base, f'fold{fold_id} baseline (静的全期間FREQ_CSV)')

        config.USE_POINT_IN_TIME_FREQ = True
        loader_pit = create_db_dataloader(
            db_path=db_path, split_type=2,
            batch_size=config.BATCH_SIZE, shuffle=False,
            max_cooccurrence=config.MAX_CO_OCCURRENCE,
        )
        res_pit = _run_one(model, loader_pit, f'fold{fold_id} point-in-time ({split_date}時点)')
        config.USE_POINT_IN_TIME_FREQ = False

        for task in res_base:
            rows.append({
                'fold': fold_id, 'split_date': split_date, 'task': task,
                'baseline_hr': res_base[task], 'point_in_time_hr': res_pit[task],
                'delta': res_pit[task] - res_base[task],
            })

    import pandas as pd
    df = pd.DataFrame(rows)
    out_dir = X.make_output_dir('verify_point_in_time_freq')
    X.save_csv(df, out_dir, 'verify_point_in_time_freq.csv')
    force_print(f"\n{df.to_string(index=False)}")


if __name__ == '__main__':
    main()
