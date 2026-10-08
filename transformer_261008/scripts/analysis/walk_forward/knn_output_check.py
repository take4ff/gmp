# --- transformer_261008/scripts/analysis/walk_forward/knn_output_check.py ---
"""kNN検索拡張出力（USE_KNN_OUTPUT）の効果を、学習済みcheckpointに対し評価時のみで検証する。

build_knn_datastore.py で作ったデータストア（fold_Nのtrain由来）を使い、同じfoldのtest群を
無作為抽出（--max_test_groups）して、λ（KNN_LAMBDA）を振りながら position_hit_rate を比較する。
λ=0 はkNN無し（階層的予測のみ）と同値の基準。fold全体の評価は1パス約1時間かかるため、
同一の抽出サンプル上でλ間を比較する（サンプル内比較なので公平）。

Usage:
  python -m transformer_261008.scripts.analysis.walk_forward.knn_output_check \\
      --checkpoint .../fold_3/<ts>/models/best_model.pth --datastore_path cache/knn_datastore_fold3 \\
      --max_test_groups 100000 --lambdas 0 0.1 0.25 0.5 [--k 16]
"""
import argparse
import random
import re

import pandas as pd

from transformer_261008 import config
from transformer_261008.db.connection import get_db_path
from transformer_261008.scripts.analysis.xai import _xai_common as X
from transformer_261008.evaluate import evaluate
from transformer_261008.utils.losses import build_loss_fn
from transformer_261008.utils.logging import force_print


def _weighted(metrics, key):
    n = sum(m['num_samples'] for m in metrics.values())
    return (sum(m[key] * m['num_samples'] for m in metrics.values()) / n if n else 0.0), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--datastore_path', required=True)
    ap.add_argument('--max_test_groups', type=int, default=100000)
    ap.add_argument('--lambdas', type=float, nargs='+', default=[0.0, 0.1, 0.25, 0.5])
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--output_dir', default=None)
    args = ap.parse_args()

    fold_id = int(re.search(r'fold_(\d+)', args.checkpoint).group(1))
    db_path = get_db_path()
    train_start, split_date, split_end = X.get_fold_windows()[fold_id]
    X.assign_fold_test_window(db_path, train_start, split_date, split_end)
    model, loader, device = X.load_model_and_loader(args.checkpoint, split='test')
    X.assign_fold_test_window(db_path, train_start, split_date, split_end)

    ds = loader.dataset
    if len(ds.sample_ids) > args.max_test_groups:
        ds.sample_ids = sorted(random.Random(args.seed).sample(ds.sample_ids, args.max_test_groups))
        ds._length = len(ds.sample_ids)
    force_print(f"[INFO] fold_{fold_id} test群: {len(ds.sample_ids):,}（無作為抽出 seed={args.seed}）")

    # snapshotが上書きするため、現行既定に明示的に揃える
    config.USE_HIERARCHICAL_PREDICTION = True
    config.HIERARCHICAL_TOPK_REGIONS = 3
    config.KNN_DATASTORE_PATH = args.datastore_path
    config.KNN_K = args.k
    loss_fn = build_loss_fn(None)

    rows = []
    for lam in args.lambdas:
        config.KNN_LAMBDA = lam
        config.USE_KNN_OUTPUT = lam > 0
        _, metrics, _, _, _, _ = evaluate(model, loader, loss_fn, (6.0, 10.0))
        pos, n = _weighted(metrics, 'position_hit_rate')
        reg, _ = _weighted(metrics, 'region_hit_rate')
        force_print(f"  lambda={lam}: position_hit_rate={pos:.4f} region_hit_rate={reg:.4f} n={n} (k={args.k})")
        rows.append({'fold': fold_id, 'lambda': lam, 'k': args.k, 'position_hit_rate': pos,
                     'region_hit_rate': reg, 'n': n})
    df = pd.DataFrame(rows)
    out = X.make_output_dir('knn_output_check', args.output_dir)
    X.save_csv(df, out, 'knn_output_check.csv')
    base = df.loc[df['lambda'] == 0.0, 'position_hit_rate']
    if len(base):
        df['diff_vs_lambda0'] = df['position_hit_rate'] - float(base.iloc[0])
        force_print("\n[SUMMARY]\n" + df.to_string(index=False))


if __name__ == '__main__':
    main()
