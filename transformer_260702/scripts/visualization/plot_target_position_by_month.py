# 月別・ターゲット変異位置（NucPos）別のサンプル数ヒートマップをプロット。
# 上位 TOP_N 位置のみ表示（総出現数順で選択、ゲノム位置順に並べて表示）。
# Usage: python -m transformer_260702.scripts.visualization.plot_target_position_by_month
import os
import pickle
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

from transformer_260702 import config
from transformer_260702.db.connection import get_db_path, connect_db

TOP_N = 100
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

    # 総出現数上位 TOP_N 位置を選択
    pos_total: dict[int, int] = defaultdict(int)
    for mc in counts.values():
        for pos_id, cnt in mc.items():
            pos_total[pos_id] += cnt

    top_positions = sorted(pos_total, key=lambda p: pos_total[p], reverse=True)[:TOP_N]
    top_positions_sorted = sorted(top_positions)  # ゲノム位置昇順に並び替え

    months = sorted(counts.keys())
    n_months = len(months)
    n_pos = len(top_positions_sorted)

    # ヒートマップ行列を構築 [pos × month]
    month_idx = {m: i for i, m in enumerate(months)}
    pos_idx = {p: i for i, p in enumerate(top_positions_sorted)}
    matrix = np.zeros((n_pos, n_months), dtype=np.int32)
    for month, mc in counts.items():
        mi = month_idx.get(month)
        if mi is None:
            continue
        for pos_id, cnt in mc.items():
            pi = pos_idx.get(pos_id)
            if pi is not None:
                matrix[pi, mi] = cnt

    print(f"[INFO] Matrix: {n_pos} positions × {n_months} months")
    print(f"[INFO] Top 10 positions: {[(p, pos_total[p]) for p in top_positions[:10]]}")

    # --- プロット ---
    fig_w = max(18, n_months * 0.38)
    fig_h = max(14, n_pos * 0.19)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    log_matrix = np.log1p(matrix)
    im = ax.imshow(log_matrix, aspect='auto', cmap='YlOrRd', interpolation='nearest')

    # カラーバー（元のカウント値で目盛り表示）
    cbar = plt.colorbar(im, ax=ax, pad=0.01, fraction=0.02)
    cbar.set_label('Sample count (log scale)', fontsize=9)
    max_val = int(matrix.max())
    raw_ticks = [v for v in [0, 10, 100, 1000, 10000, 100000, 1000000] if v <= max_val]
    if max_val not in raw_ticks:
        raw_ticks.append(max_val)
    cbar.set_ticks([np.log1p(v) for v in raw_ticks])
    cbar.set_ticklabels([f'{v:,}' for v in raw_ticks], fontsize=7)

    ax.set_xticks(range(n_months))
    ax.set_xticklabels(months, rotation=45, ha='right', fontsize=7)
    ax.set_yticks(range(n_pos))
    ax.set_yticklabels([str(p) for p in top_positions_sorted], fontsize=6)
    ax.set_xlabel('Month')
    ax.set_ylabel('Target NucPos')
    ax.set_title(f'Sample count by month × target NucPos (top {TOP_N} positions)')

    plt.tight_layout()

    output_dir = os.path.join(config.OUTPUT_DIR, 'scripts', 'timestep')
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, 'target_position_by_month.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved: {out_path}")


if __name__ == '__main__':
    main()
