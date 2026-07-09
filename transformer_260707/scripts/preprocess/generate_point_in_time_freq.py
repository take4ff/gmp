# --- transformer_260707/scripts/preprocess/generate_point_in_time_freq.py ---
"""walk-forwardフォールドごとに「そのfoldのtrain窓カットオフ時点まで」のデータだけを使った
変異頻度テーブル（reference/table_set.csvと同じスキーマ）を生成する。

【背景・問題】
現行の `config.FREQ_CSV = "reference/table_set.csv"` は、データセット全期間
（2020年〜作成時点）から静的に1回だけ計算された変異頻度テーブルで、train/testやfoldを
問わず全サンプルに同じ値が使われている。walk-forward評価では、test窓より未来の期間に
何度も起きた変異かどうかという情報が特徴量(codon_freq, x_num[...,0])に混入しており、
時系列的なデータリークになっている。

【本スクリプトの方針】
各foldのtrain窓カットオフ日（split_date）より前のcollection_dateを持つサンプルだけを使い、
raw_pathをユニーク化した上で「その変異(position, base_before, base_after)を含む
ユニークraw_pathの数」を数える（系統樹の枝を明示的に辿らない近似だが、独立発生した
系統がそれぞれ異なるraw_pathを持つ、という前提の下では合理的な近似）。

出力: reference/point_in_time_freq/fold{N}_table_set.csv （fold_windowsで指定した各foldごと）
      → config.FREQ_CSV をこのファイルに差し替えて学習・評価すればリークフリーになる。

Usage:
  python -m transformer_260707.scripts.preprocess.generate_point_in_time_freq [--folds 1 2 3]
"""

import argparse
import os
import re
from collections import defaultdict

import duckdb
import pandas as pd

from transformer_260707.db.connection import get_db_path
from transformer_260707.scripts.eval.walk_forward import FOLDS

_MUT_RE = re.compile(r'^([ACGT])(\d+)([ACGT])$')
_BASES = ['A', 'T', 'G', 'C']
_TRANSITIONS = [f'{a}->{b}' for a in _BASES for b in _BASES if a != b]


def _parse_mutations(raw_path):
    """raw_path ('>' 区切り, ',' 共起区切り) から (base_before, position, base_after) の集合を返す。"""
    out = set()
    for node in raw_path.split('>'):
        for mut in node.split(','):
            mut = mut.strip()
            if not mut:
                continue
            m = _MUT_RE.match(mut)
            if m:
                out.add((m.group(1), int(m.group(2)), m.group(3)))
    return out


def compute_point_in_time_freq(db_path, cutoff_date, max_position=29903):
    """cutoff_date より前のcollection_dateを持つサンプルのみでvariant frequencyを計算する。

    Returns: dict[(base_before, position, base_after)] -> count（ユニークraw_pathの数）
    """
    con = duckdb.connect(db_path, read_only=True)
    where = "collection_date IS NOT NULL AND collection_date != '' "
    params = []
    if cutoff_date:
        where += "AND RPAD(collection_date, 10, '-01-01') < ? "
        params.append(cutoff_date)
    rows = con.execute(f"SELECT DISTINCT raw_path FROM samples WHERE {where}", params).fetchall()
    con.close()

    counts = defaultdict(int)
    for (raw_path,) in rows:
        for mut in _parse_mutations(raw_path):
            counts[mut] += 1
    print(f"[INFO]   cutoff={cutoff_date}: ユニークraw_path={len(rows):,}件, "
          f"ユニーク変異(位置×塩基置換)={len(counts):,}種")
    return counts


def save_table(counts, out_path, max_position=29903):
    table = {t: [0] * max_position for t in _TRANSITIONS}
    for (base_before, position, base_after), count in counts.items():
        if not (1 <= position <= max_position):
            continue
        transition = f'{base_before}->{base_after}'
        if transition in table:
            table[transition][position - 1] = count

    df = pd.DataFrame({'position': range(1, max_position + 1)})
    for t in _TRANSITIONS:
        df[t] = table[t]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[INFO] Saved {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', type=int, nargs='+', default=None,
                    help='対象フォールド番号（省略時は全7フォールド）')
    ap.add_argument('--output_dir', default='reference/point_in_time_freq')
    args = ap.parse_args()

    db_path = get_db_path()
    target_folds = [f for f in FOLDS if args.folds is None or f[0] in set(args.folds)]

    for fold_id, train_start, split_date, split_end, desc in target_folds:
        print(f"\n===== Fold {fold_id}: {desc} (cutoff={split_date}) =====")
        counts = compute_point_in_time_freq(db_path, split_date)
        # db/dataset.py 側は config.TEMPORAL_SPLIT_DATE（walk-forwardフォールドのsplit_dateと
        # 同一値）でこのファイルを直接引くため、日付をファイル名にする（fold番号名でも保存）。
        save_table(counts, os.path.join(args.output_dir, f'{split_date}.csv'))
        save_table(counts, os.path.join(args.output_dir, f'fold{fold_id}_table_set.csv'))


if __name__ == '__main__':
    main()
