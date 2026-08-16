# --- transformer_260817/scripts/analysis/majority_baseline.py ---
"""単純ベースライン（一様ランダム・多数決）を walk_forward の各フォールド窓で計算する。

「デルタ株45%/35%、オミクロン株50%/10%」等の精度が良いか悪いかを判断する基準として、
学習不要の2つの下限ベースラインを提供する。
  - 一様ランダム: 1/VOCAB_SIZE（語彙サイズの逆数、解析的に計算するだけ）
  - 多数決: 学習期間内で最頻出のregion/positionを常に予測した場合の、
    テスト期間での的中率（DBを直接集計するだけ、モデル学習不要）

Usage:
    python -m transformer_260817.scripts.analysis.majority_baseline
"""
import pickle
from collections import Counter

import pandas as pd

from transformer_260817 import config
from transformer_260817.db.connection import get_db_path, connect_db
from transformer_260817.utils.logging import force_print
from transformer_260817.scripts.analysis.xai import _xai_common as X
from transformer_260817.scripts.eval.walk_forward import FOLDS


def _target_counts(con, date_lo, date_hi):
    """[date_lo, date_hi) の collection_date を持つサンプルの targets を集計し、
    (region_counter, position_counter, n_targets, n_samples) を返す。
    date_lo=None は下限なし、date_hi=None は上限なし。
    """
    where = "collection_date IS NOT NULL AND collection_date != ''"
    params = []
    if date_lo is not None:
        where += " AND RPAD(collection_date, 10, '-01-01') >= ?"
        params.append(date_lo)
    if date_hi is not None:
        where += " AND RPAD(collection_date, 10, '-01-01') < ?"
        params.append(date_hi)

    sample_ids = [r[0] for r in con.execute(
        f"SELECT sample_id FROM samples WHERE {where}", params).fetchall()]
    if not sample_ids:
        return Counter(), Counter(), 0, 0

    region_c, pos_c, n_targets = Counter(), Counter(), 0
    CHUNK = 200000
    for i in range(0, len(sample_ids), CHUNK):
        chunk = sample_ids[i:i + CHUNK]
        placeholders = ','.join(['?'] * len(chunk))
        rows = con.execute(
            f"SELECT targets FROM labels WHERE sample_id IN ({placeholders})", chunk
        ).fetchall()
        for (blob,) in rows:
            for t in pickle.loads(blob):
                # t: (region_id, position_id, aa_pos_id, codon_pos_id, is_synonymous)
                region_c[t[0]] += 1
                pos_c[t[1]] += 1
                n_targets += 1
    return region_c, pos_c, n_targets, len(sample_ids)


def main():
    db_path = get_db_path()
    con = connect_db(db_path, read_only=True)

    uniform_region = 100.0 / config.NUM_REGIONS
    uniform_position = 100.0 / config.VOCAB_SIZE_POSITION
    force_print(f"[INFO] 一様ランダム基準: region={uniform_region:.3f}%, "
                f"position={uniform_position:.5f}%")

    rows = []
    for fold_id, train_start, split_date, split_end, desc in FOLDS:
        force_print(f"\n===== Fold {fold_id}: {desc} =====")
        train_region_c, train_pos_c, n_train_targets, n_train = _target_counts(
            con, train_start, split_date)
        if n_train_targets == 0:
            force_print("[WARN] 学習窓にターゲットが無いためスキップ")
            continue
        majority_region, majority_region_train_cnt = train_region_c.most_common(1)[0]
        majority_pos, majority_pos_train_cnt = train_pos_c.most_common(1)[0]
        force_print(f"[INFO] 学習窓: {n_train:,}サンプル / {n_train_targets:,}ターゲット, "
                    f"最頻region={majority_region}({majority_region_train_cnt/n_train_targets*100:.1f}%), "
                    f"最頻position={majority_pos}({majority_pos_train_cnt/n_train_targets*100:.1f}%)")

        test_region_c, test_pos_c, n_test_targets, n_test = _target_counts(
            con, split_date, split_end)
        if n_test_targets == 0:
            force_print("[WARN] テスト窓にターゲットが無いためスキップ")
            continue

        majority_region_acc = test_region_c.get(majority_region, 0) / n_test_targets * 100
        majority_position_acc = test_pos_c.get(majority_pos, 0) / n_test_targets * 100
        force_print(f"[INFO] テスト窓: {n_test:,}サンプル / {n_test_targets:,}ターゲット, "
                    f"多数決region的中={majority_region_acc:.2f}%, "
                    f"多数決position的中={majority_position_acc:.2f}%")

        rows.append({
            'fold_id': fold_id, 'era': desc,
            'n_train_samples': n_train, 'n_test_samples': n_test,
            'n_test_targets': n_test_targets,
            'majority_region_id': majority_region,
            'majority_region_accuracy_pct': majority_region_acc,
            'majority_position_id': majority_pos,
            'majority_position_accuracy_pct': majority_position_acc,
            'uniform_random_region_pct': uniform_region,
            'uniform_random_position_pct': uniform_position,
        })

    con.close()

    df = pd.DataFrame(rows)
    out_dir = X.make_output_dir('majority_baseline', None)
    X.save_csv(df, out_dir, 'majority_baseline_by_fold.csv')
    X.save_json({'uniform_random_region_pct': uniform_region,
                 'uniform_random_position_pct': uniform_position,
                 'num_regions': config.NUM_REGIONS,
                 'vocab_size_position': config.VOCAB_SIZE_POSITION,
                 'folds': rows}, out_dir, 'majority_baseline_meta.json')
    force_print(f"\n[INFO] 完了。{out_dir}")


if __name__ == '__main__':
    main()
