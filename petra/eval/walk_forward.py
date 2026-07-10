# petra/eval/walk_forward.py — PETRA比較モデルの半年次walk-forward学習
#
# transformer_260707/scripts/eval/walk_forward.py と全く同じ FOLDS 定義・同じ
# split_type_wf 列・同じ assign_wf_splits() を使い、本体の walk_forward と直接
# 対応するフォールドごとの Recall@K を得る。
#
# 本体の walk_forward.run_walk_forward() と同じ設計に合わせる:
#   - Fold 1: 全歴史データ（< split_date）でスクラッチから学習
#   - Fold N (N>=2): 直前フォールドの best_model.pth を初期値として継続学習
#     （本体の「直前フォールドの重みを設定」という Method B に対応）
#
# 注意: split_type_wf は本体の walk_forward と共有のDB列。本体のwalk_forwardと
# 同時に実行すると互いに split_type_wf を上書きし合い壊れるため、同時実行しないこと。
#
# Usage:
#   python -m petra.eval.walk_forward
#   python -m petra.eval.walk_forward --folds 1 2 3   # 一部のフォールドのみ
import argparse
import json
import os
import time

from transformer_260707 import config as main_config
from transformer_260707.db.connection import connect_db, get_db_path
from transformer_260707.db.queries import assign_wf_splits
from transformer_260707.scripts.eval.walk_forward import FOLDS

from .. import main as petra_main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', nargs='+', type=int, default=None,
                    help='実行するフォールド番号（省略時は全7フォールド）')
    args = ap.parse_args()

    folds = FOLDS if not args.folds else [f for f in FOLDS if f[0] in set(args.folds)]

    timestamp = time.strftime('%Y%m%d_%H%M%S')
    wf_run_dir = os.path.join('outputs', 'petra', 'walk_forward', timestamp)
    os.makedirs(wf_run_dir, exist_ok=True)

    meta = {
        'wf_run_dir': wf_run_dir,
        'folds': [{'fold': fid, 'train_start': ts, 'split_date': sd,
                  'split_end': se, 'desc': desc} for fid, ts, sd, se, desc in folds],
    }
    with open(os.path.join(wf_run_dir, 'walk_forward_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    prev_checkpoint = None
    fold_summaries = []

    for fold_id, train_start, split_date, split_end, desc in folds:
        print(f"\n{'='*60}")
        print(f"  PETRA Walk-forward Fold {fold_id}: {desc}")
        if train_start:
            print(f"  Train : {train_start} <= collection_date < {split_date}")
        else:
            print(f"  Train : collection_date < {split_date} (全歴史データ)")
        if split_end:
            print(f"  Test  : {split_date} <= collection_date < {split_end}")
        else:
            print(f"  Test  : {split_date} <= collection_date (no upper bound)")
        print('='*60)

        # このフォールドの split_type_wf を確定（本体 walk_forward と全く同じ関数を使用）
        main_config.WALK_FORWARD_TRAIN_START = train_start
        main_config.TEMPORAL_SPLIT_DATE = split_date
        main_config.TEMPORAL_SPLIT_TEST_END = split_end
        con = connect_db(get_db_path())
        assign_wf_splits(con)
        con.close()

        fold_dir = os.path.join(wf_run_dir, f'fold_{fold_id}')
        result = petra_main.run_training(
            split_col='split_type_wf',
            init_checkpoint=prev_checkpoint,
            out_dir=fold_dir,
            log_prefix=f'[fold{fold_id}] ',
        )
        prev_checkpoint = result['best_model_path']

        fold_summaries.append({
            'fold': fold_id, 'desc': desc, 'train_start': train_start,
            'split_date': split_date, 'split_end': split_end,
            'best_model_path': result['best_model_path'],
            'best_val_loss': result['best_val_loss'],
            'test_loss': result['test_loss'],
            'recall': result['recall'],
            'n_params': result['n_params'],
        })
        print(f"[INFO] Fold {fold_id} complete. Best model: {prev_checkpoint}")

    with open(os.path.join(wf_run_dir, 'walk_forward_summary.json'), 'w') as f:
        json.dump(fold_summaries, f, indent=2, ensure_ascii=False)
    print(f"\n[INFO] PETRA walk-forward complete. Results root: {wf_run_dir}")


if __name__ == '__main__':
    main()
