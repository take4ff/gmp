# --- transformer_260817/scripts/visualization/plot_target_position_by_month_normalized.py ---
# 月別・ターゲット変異位置（NucPos）別のサンプル数ヒートマップを、月ごとのスケール差を
# 補正して表示する（plot_target_position_by_month.py の正規化版）。
#
# plot_target_position_by_month.py は「全期間通算の出現数」で上位100位置を固定するため、
# 2022年前半のようにサンプル数自体が突出して多い時期の位置に選定が偏りやすい。
# 本スクリプトでは:
#   - 値: その月の総サンプル数に対する割合(%) で正規化（月ごとのボリューム差を吸収）
#   - 行(位置)選定: 各月ごとの上位10位置（その月の生カウント基準）の和集合
#     （通算Top100の代わりに、特定の月にしか目立たない位置も拾う）
# Usage: python -m transformer_260817.scripts.visualization.plot_target_position_by_month_normalized
import os
import pickle
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

from transformer_260817 import config
from transformer_260817.db.connection import get_db_path, connect_db

TOP_N_PER_MONTH = 10
BATCH_SIZE = 50000


def main():
    db_path = get_db_path()
    print(f"[INFO] Connecting to: {db_path}")
    con = connect_db(db_path, read_only=True)

    # サンプルIDと月のマッピング
    print("[INFO] Loading sample months...")
    month_rows = con.execute("""
        SELECT sample_id, substr(collection_date, 1, 7) AS month
        FROM samples
        WHERE length(collection_date) >= 7 AND substr(collection_date, 5, 1) = '-'
    """).fetchall()
    sample_to_month = {r[0]: r[1] for r in month_rows}
    print(f"[INFO] {len(sample_to_month):,} samples with valid month")

    total_samples_per_month: dict[str, int] = defaultdict(int)
    for m in sample_to_month.values():
        total_samples_per_month[m] += 1

    # ラベルを走査して (month, position_id) をカウント
    print("[INFO] Counting (month, nuc_pos) pairs from labels...")
    counts: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    total_labels = con.execute("SELECT COUNT(*) FROM labels").fetchone()[0]
    print(f"[INFO] Total labels: {total_labels:,}")

    offset = 0
    processed = 0
    while True:
        rows = con.execute(
            f"SELECT sample_id, targets FROM labels LIMIT {BATCH_SIZE} OFFSET {offset}"
        ).fetchall()
        if not rows:
            break
        for sample_id, targets_blob in rows:
            month = sample_to_month.get(sample_id)
            if month is None:
                continue
            targets = pickle.loads(targets_blob)
            for t in targets:
                # t: (region_id, position_id, aa_pos_id, codon_pos_id, is_synonymous)
                pos_id = t[1]
                counts[month][pos_id] += 1
        processed += len(rows)
        offset += BATCH_SIZE
        if processed % 200000 == 0:
            print(f"[INFO]   {processed:,} / {total_labels:,} labels processed")

    con.close()
    print(f"[INFO] Done. {processed:,} labels processed.")

    # 各月ごとの上位 TOP_N_PER_MONTH 位置（その月の生カウント基準）の和集合を行として採用
    selected_positions: set[int] = set()
    for month, mc in counts.items():
        top_this_month = sorted(mc, key=lambda p: mc[p], reverse=True)[:TOP_N_PER_MONTH]
        selected_positions.update(top_this_month)

    positions_sorted = sorted(selected_positions)  # ゲノム位置昇順
    months = sorted(counts.keys())
    n_months = len(months)
    n_pos = len(positions_sorted)
    print(f"[INFO] {n_pos} unique positions selected "
          f"(union of top {TOP_N_PER_MONTH} per month across {n_months} months)")

    # ヒートマップ行列 [pos × month] = その月の総サンプル数に対する割合(%)
    month_idx = {m: i for i, m in enumerate(months)}
    pos_idx = {p: i for i, p in enumerate(positions_sorted)}
    matrix_pct = np.zeros((n_pos, n_months), dtype=np.float64)
    for month, mc in counts.items():
        mi = month_idx.get(month)
        total = total_samples_per_month.get(month, 0)
        if mi is None or total <= 0:
            continue
        for pos_id, cnt in mc.items():
            pi = pos_idx.get(pos_id)
            if pi is not None:
                matrix_pct[pi, mi] = cnt / total * 100.0

    print(f"[INFO] Matrix: {n_pos} positions × {n_months} months")

    # --- プロット ---
    fig_w = max(18, n_months * 0.38)
    fig_h = max(14, n_pos * 0.19)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    log_matrix = np.log1p(matrix_pct)
    im = ax.imshow(log_matrix, aspect='auto', cmap='YlOrRd', interpolation='nearest')

    # カラーバー（元の%値で目盛り表示）
    cbar = plt.colorbar(im, ax=ax, pad=0.01, fraction=0.02)
    cbar.set_label('% of month\'s samples (log scale)', fontsize=9)
    max_val = float(matrix_pct.max())
    raw_ticks = [v for v in [0, 0.1, 1, 5, 10, 25, 50, 75, 100] if v <= max_val]
    if max_val not in raw_ticks:
        raw_ticks.append(round(max_val, 1))
    cbar.set_ticks([np.log1p(v) for v in raw_ticks])
    cbar.set_ticklabels([f'{v:g}%' for v in raw_ticks], fontsize=7)

    ax.set_xticks(range(n_months))
    ax.set_xticklabels(months, rotation=45, ha='right', fontsize=7)
    ax.set_yticks(range(n_pos))
    ax.set_yticklabels([str(p) for p in positions_sorted], fontsize=6)
    ax.set_xlabel('Month')
    ax.set_ylabel('Target NucPos')
    ax.set_title(f'% of month\'s samples by target NucPos\n'
                 f'(rows = union of each month\'s top {TOP_N_PER_MONTH} positions, '
                 f'{n_pos} positions total)')

    plt.tight_layout()

    output_dir = os.path.join(config.OUTPUT_DIR, 'scripts', 'timestep')
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, 'target_position_by_month_normalized.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved: {out_path}")


if __name__ == '__main__':
    main()
