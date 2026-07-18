# --- transformer_260707/scripts/analysis/branching_cooccurrence_frequency.py ---
"""raw_path から「分岐(K)」と「共起(M)」の発生頻度を測る（PETRA比較の評価方式検討用）。

- 共起 M: 1サンプルの target（raw_path 末尾の '>' 区切り最終ステップ）内で ',' 区切りされた
  同時変異の数。M>=2 なら共起あり。
- 分岐 K: 同一の input_path_str（raw_path から末尾ステップを除いた履歴）を共有する
  サンプル数。K>=2 ならその履歴から複数の異なる次ステップが観測された「分岐」。

DB読み取り専用（split_type_wf 等は一切参照・書き込みしない）。他の walk_forward プロセスと
並行実行しても安全。月別（＝fold era別）内訳も出す。

Usage:
  python -m transformer_260707.scripts.analysis.branching_cooccurrence_frequency
"""
import os
from collections import defaultdict, Counter

import pandas as pd

from transformer_260707 import config
from transformer_260707.db.connection import get_db_path, connect_db
from transformer_260707.utils.logging import force_print

BATCH_SIZE = 200000


def main():
    db_path = get_db_path()
    force_print(f"[INFO] Connecting to: {db_path} (read_only)")
    con = connect_db(db_path, read_only=True)

    total = con.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
    force_print(f"[INFO] total samples={total:,}")

    m_dist_overall = Counter()
    m_dist_by_month = defaultdict(Counter)
    parent_to_months = defaultdict(list)  # input_path_str -> [month, month, ...]（分岐検出用）

    offset = 0
    processed = 0
    while True:
        rows = con.execute(f"""
            SELECT sample_id, raw_path, substr(collection_date, 1, 7) AS month
            FROM samples
            LIMIT {BATCH_SIZE} OFFSET {offset}
        """).fetchall()
        if not rows:
            break
        for sample_id, raw_path, month in rows:
            if not raw_path:
                continue
            parts = raw_path.split('>')
            last = parts[-1]
            input_path_str = '>'.join(parts[:-1]) if len(parts) > 1 else ''
            m = len(last.split(','))
            m_dist_overall[m] += 1
            if month:
                m_dist_by_month[month][m] += 1
            parent_to_months[input_path_str].append(month or 'unknown')
        processed += len(rows)
        offset += BATCH_SIZE
        if processed % 1000000 == 0:
            force_print(f"[INFO]   {processed:,} / {total:,} processed")
    con.close()
    force_print(f"[INFO] Done. {processed:,} samples processed.")

    # --- 共起(M)の分布 ---
    total_m = sum(m_dist_overall.values())
    force_print("\n===== 共起(M) 分布（全体） =====")
    for m in sorted(m_dist_overall):
        c = m_dist_overall[m]
        force_print(f"  M={m:2d}: {c:>10,} ({100*c/total_m:.2f}%)")
    pct_co = 100 * sum(c for m, c in m_dist_overall.items() if m >= 2) / total_m
    force_print(f"  => 共起あり(M>=2)の割合: {pct_co:.2f}%")

    # --- 分岐(K)の分布 ---
    k_dist = Counter(len(v) for v in parent_to_months.values())
    total_k_groups = sum(k_dist.values())
    total_k_samples = sum(k * c for k, c in k_dist.items())
    force_print("\n===== 分岐(K) 分布（親履歴グループ単位） =====")
    for k in sorted(k_dist):
        c = k_dist[k]
        force_print(f"  K={k:2d}: {c:>10,} groups ({100*c/total_k_groups:.2f}% of groups, "
                    f"{100*k*c/total_k_samples:.2f}% of samples)")
    pct_branch_groups = 100 * sum(c for k, c in k_dist.items() if k >= 2) / total_k_groups
    pct_branch_samples = 100 * sum(k * c for k, c in k_dist.items() if k >= 2) / total_k_samples
    force_print(f"  => 分岐あり(K>=2)の割合: グループ数の{pct_branch_groups:.2f}% / "
                f"サンプル数の{pct_branch_samples:.2f}%")

    # --- 月別（era別）内訳 ---
    from transformer_260707.scripts.visualization._fold_annotate import _month_to_fold
    month_fold = {}
    branch_samples_by_month = Counter()
    total_samples_by_month = Counter()
    for input_path_str, months in parent_to_months.items():
        k = len(months)
        for mo in months:
            total_samples_by_month[mo] += 1
            if k >= 2:
                branch_samples_by_month[mo] += 1

    rows = []
    for mo in sorted(total_samples_by_month):
        if mo == 'unknown':
            continue
        fid, desc = _month_to_fold(mo)
        n = total_samples_by_month[mo]
        n_branch = branch_samples_by_month.get(mo, 0)
        n_m = sum(m_dist_by_month[mo].values())
        n_co = sum(c for m, c in m_dist_by_month[mo].items() if m >= 2)
        rows.append({
            'month': mo, 'fold': fid, 'fold_desc': desc, 'n_samples': n,
            'pct_branching_ge2': 100 * n_branch / n if n else 0.0,
            'pct_cooccurrence_ge2': 100 * n_co / n_m if n_m else 0.0,
        })
    df = pd.DataFrame(rows)

    out_dir = os.path.join(config.OUTPUT_DIR, 'scripts', 'branching_cooccurrence_frequency')
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, 'branching_cooccurrence_by_month.csv')
    df.to_csv(csv_path, index=False)
    force_print(f"\n[INFO] Saved: {csv_path}")

    force_print("\n===== fold(era)別 平均（月次値の単純平均） =====")
    if not df.empty:
        for fid, g in df.groupby('fold'):
            desc = g['fold_desc'].iloc[0]
            label = f'fold{int(fid)}: {desc}'
            force_print(f"  {label:45s} 分岐率={g['pct_branching_ge2'].mean():5.2f}%  "
                        f"共起率={g['pct_cooccurrence_ge2'].mean():5.2f}%  (n_months={len(g)})")


if __name__ == '__main__':
    main()
