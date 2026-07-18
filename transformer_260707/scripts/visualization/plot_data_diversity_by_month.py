# --- transformer_260707/scripts/visualization/plot_data_diversity_by_month.py ---
# 変異パス末尾（ターゲット位置）の多様度3種（entropy_norm / unique_ratio / simpson）を
# 月単位で集計し、時系列プロットする。
#
# db.queries.get_db_lineage_target_entropy() が系統(lineage)粒度でDBから直接計算するのと
# 同じ考え方（モデル評価不要・生データのみ）を月粒度に適用したもの。
# ターゲット位置の出現源は plot_target_position_by_month.py と同じ labels テーブル
# （raw_path 末尾から前処理時に確定済みの position_id）を使う。
#
# 「予測の難しさの理由づけ」という観点では、末尾（ターゲット）だけの多様度で十分
# （入力パス側の多様性とは別軸の問題）という整理のもとで作成。
#
# Usage: python -m transformer_260707.scripts.visualization.plot_data_diversity_by_month
import os
import math
import pickle
from collections import defaultdict, Counter

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from transformer_260707 import config
from transformer_260707.db.connection import get_db_path, connect_db

BATCH_SIZE = 50000


def _diversity_from_counts(counts: Counter):
    """位置ごとの出現回数 Counter から entropy_norm / unique_ratio / simpson を計算する。
    utils/plotting._lineage_diversity_rows と同じ式（系統粒度版と月粒度版で定義を揃える）。
    """
    total = sum(counts.values())
    if total == 0:
        return 0.0, 0.0, 0.0
    probs = [c / total for c in counts.values()]
    entr = -sum(p * math.log2(p) for p in probs)
    max_e = math.log2(len(counts)) if len(counts) > 1 else 1.0
    entropy_norm = entr / max_e
    unique_ratio = len(counts) / total
    simpson = 1.0 - sum(p * p for p in probs)
    return entropy_norm, unique_ratio, simpson


def main():
    db_path = get_db_path()
    print(f"[INFO] Connecting to: {db_path}")
    con = connect_db(db_path, read_only=True)

    print("[INFO] Loading sample months...")
    month_rows = con.execute("""
        SELECT sample_id, substr(collection_date, 1, 7) AS month
        FROM samples
        WHERE length(collection_date) >= 7 AND substr(collection_date, 5, 1) = '-'
    """).fetchall()
    sample_to_month = {r[0]: r[1] for r in month_rows}
    print(f"[INFO] {len(sample_to_month):,} samples with valid month")

    print("[INFO] Counting (month, nuc_pos) pairs from labels...")
    counts: dict[str, Counter] = defaultdict(Counter)
    total_labels = con.execute("SELECT COUNT(*) FROM labels").fetchone()[0]

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
                counts[month][t[1]] += 1
        processed += len(rows)
        offset += BATCH_SIZE
        if processed % 1000000 == 0:
            print(f"[INFO]   {processed:,} / {total_labels:,} labels processed")
    con.close()
    print(f"[INFO] Done. {processed:,} labels processed.")

    months = sorted(counts.keys())
    entropy_norm_l, unique_ratio_l, simpson_l, n_samples_l = [], [], [], []
    for m in months:
        e, u, s = _diversity_from_counts(counts[m])
        entropy_norm_l.append(e)
        unique_ratio_l.append(u)
        simpson_l.append(s)
        n_samples_l.append(sum(counts[m].values()))

    output_dir = os.path.join(config.OUTPUT_DIR, 'scripts', 'timestep')
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, 'data_diversity_by_month.csv')
    pd.DataFrame({
        'month': months, 'n_labels': n_samples_l,
        'entropy_norm': entropy_norm_l, 'unique_ratio': unique_ratio_l, 'simpson': simpson_l,
    }).to_csv(csv_path, index=False)
    print(f"[INFO] Saved: {csv_path}")

    # --- プロット ---
    fig, ax = plt.subplots(figsize=(20, 6))
    x = range(len(months))

    ax2 = ax.twinx()
    ax2.bar(x, n_samples_l, color='steelblue', alpha=0.15, zorder=0)
    ax2.set_ylabel('Target label count', color='steelblue', fontsize=9)
    ax2.tick_params(axis='y', labelcolor='steelblue', labelsize=8)
    ax2.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda v, _: f'{v/1000:.0f}K' if v >= 1000 else str(int(v)))
    )
    ax2.set_zorder(0)
    ax.set_zorder(1)
    ax.patch.set_visible(False)

    ax.plot(x, entropy_norm_l, marker='o', ms=4, lw=1.5, label='Entropy (normalized)', color='#c0392b')
    ax.plot(x, unique_ratio_l, marker='s', ms=4, lw=1.5, label='Unique position ratio', color='#2980b9')
    ax.plot(x, simpson_l,      marker='^', ms=4, lw=1.5, label='Gini-Simpson diversity', color='#27ae60')

    ax.set_xticks(list(x))
    ax.set_xticklabels(months, rotation=45, ha='right', fontsize=8)
    ax.set_xlabel('Month')
    ax.set_ylabel('Diversity (0-1)')
    ax.set_ylim(-0.02, 1.05)
    ax.set_title('Target-position diversity by month (tail of mutation path, model-independent)')
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.legend(loc='lower right', fontsize=9)

    plt.tight_layout()

    out_path = os.path.join(output_dir, 'data_diversity_by_month.png')
    plt.savefig(out_path, dpi=150)
    print(f"[INFO] Saved: {out_path}")

    from transformer_260707.scripts.visualization._fold_annotate import add_fold_annotations
    add_fold_annotations(ax, months)
    ax.set_title(ax.get_title(), pad=30)
    out_path_fold = os.path.join(output_dir, 'data_diversity_by_month_by_fold.png')
    plt.savefig(out_path_fold, dpi=150, bbox_inches='tight')
    print(f"[INFO] Saved: {out_path_fold}")

    plt.close()


if __name__ == '__main__':
    main()
