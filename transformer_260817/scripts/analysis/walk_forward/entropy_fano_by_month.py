# --- transformer_260817/scripts/analysis/walk_forward/entropy_fano_by_month.py ---
"""月別の実測精度と、Fano不等式による理論上限を、5タスク（Region/Base Position/AA Position/
Codon Pos/Synonymous）それぞれについて同一グラフに重ねる。

位置(Base Position)だけでなく、'labels' テーブルの pickled targets タプル
(region, position, aa_pos, codon_pos, synonymous) の全要素についてタスク別に
月別エントロピー H(bits) を計算し、各タスクの語彙数（config.NUM_REGIONS /
VOCAB_SIZE_POSITION / VOCAB_SIZE_AA_POS / VOCAB_SIZE_CODON_POS / VOCAB_SIZE_SYNONYMOUS）で
Fano理論上限を求める。scripts/visualization/plot_data_diversity_by_month.py と同じ
'labels' テーブル1回スキャンで全タスク分を同時に集計する（DB書き込みなし・read_only）。

walk-forward の各foldテスト窓は月単位で重複しないため、月ごとのエントロピーは「その月が
どのfoldのテスト対象であるか」に関わらず不変 = split_type_wf を参照・書き込みする必要がない。
そのため他の walk_forward プロセス（petra 側等、split_type_wf を共有・上書きする処理）と
並行実行しても split_type_wf を壊さず安全に実行できる。

実測精度は main.py の _wf_pool_test_outputs が生成する monthly_metrics_all_folds.csv
（{task}_hit_rate 列）を使う。

Usage:
    python -m transformer_260817.scripts.analysis.walk_forward.entropy_fano_by_month \\
        --walk_forward_dir outputs/transformer_260817/results/walk_forward/<timestamp>
    # 一部タスクだけ:
    python -m transformer_260817.scripts.analysis.walk_forward.entropy_fano_by_month \\
        --walk_forward_dir outputs/transformer_260817/results/walk_forward/<timestamp> \\
        --tasks position region
"""
import os
import math
import pickle
import argparse
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import brentq

from transformer_260817 import config
from transformer_260817.db.connection import get_db_path, connect_db
from transformer_260817.scripts.visualization._fold_annotate import add_fold_annotations

BATCH_SIZE = 50000

# task_key -> (targetsタプル内index, hit_rate列名, 語彙数, 表示ラベル, 色)
TASKS = {
    'region':      (0, 'region_hit_rate',      config.NUM_REGIONS,           'Region',        '#4E79A7'),
    'position':    (1, 'position_hit_rate',    config.VOCAB_SIZE_POSITION,   'Base Position', '#E15759'),
    'aa_pos':      (2, 'aa_pos_hit_rate',      config.VOCAB_SIZE_AA_POS,     'AA Position',   '#59A14F'),
    'codon_pos':   (3, 'codon_pos_hit_rate',   config.VOCAB_SIZE_CODON_POS,  'Codon Pos',     '#B07AA1'),
    'synonymous':  (4, 'synonymous_hit_rate',  config.VOCAB_SIZE_SYNONYMOUS, 'Synonymous',    '#EDC948'),
}


def fano_min_error_rate(h_bits: float, vocab_size: int) -> float:
    """Fano不等式 H(X|Y) <= H_b(Pe) + Pe*log2(M-1) を満たす最小のPeを二分探索で求める。
    entropy_accuracy_analysis.py と同一定義（重複はDB非依存スクリプトをtorch非依存に保つため）。

    右辺 f(pe) = H_b(pe) + pe*log2(M-1) は pe*=1-1/M で最大値log2(M)を取り、
    そこから先は減少に転じる（M=2だとlog2(M-1)=0のため特に顕著）。求めたいのは
    「達成可能な最小のPe」＝f(0)<h_bits<f(pe*)となる区間内の小さい方の解なので、
    探索区間は [0, pe*] に限定する（[0, 1]全体だと、M=2等でf(0)とf(1)が両方とも
    h_bits未満になり得て符号が反転せずbrentqが失敗し、常にpe=1-1/Mへフォール
    バックしてしまうバグがあった。Synonymous(M=2)タスクでFano上限が実測エントロピー
    に関わらず常に50%固定になっていたのはこれが原因）。
    """
    M = vocab_size
    if M <= 1 or h_bits <= 0:
        return 0.0
    max_h = np.log2(M)
    if h_bits >= max_h:
        return 1.0 - 1.0 / M

    def gap(pe):
        if pe <= 0.0 or pe >= 1.0:
            hb = 0.0
        else:
            hb = -pe * np.log2(pe) - (1 - pe) * np.log2(1 - pe)
        return hb + pe * np.log2(max(M - 1, 1)) - h_bits

    pe_star = 1.0 - 1.0 / M
    try:
        return brentq(gap, 1e-12, pe_star - 1e-12)
    except ValueError:
        return 1.0 - 1.0 / M


def _monthly_task_entropy_bits(task_keys):
    """月別・タスク別ターゲットエントロピー(bits)を'labels'テーブルから1回のスキャンで
    まとめて計算する（DB書き込みなし・read_only）。
    Returns: dict task_key -> dict month -> entropy_bits
    """
    db_path = get_db_path()
    print(f"[INFO] Connecting to: {db_path} (read_only)")
    con = connect_db(db_path, read_only=True)

    month_rows = con.execute("""
        SELECT sample_id, substr(collection_date, 1, 7) AS month
        FROM samples
        WHERE length(collection_date) >= 7 AND substr(collection_date, 5, 1) = '-'
    """).fetchall()
    sample_to_month = {r[0]: r[1] for r in month_rows}

    idx_by_task = {tk: TASKS[tk][0] for tk in task_keys}
    counts = {tk: defaultdict(Counter) for tk in task_keys}

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
                for tk, idx in idx_by_task.items():
                    if idx < len(t):
                        counts[tk][month][t[idx]] += 1
        processed += len(rows)
        offset += BATCH_SIZE
        if processed % 1000000 == 0:
            print(f"[INFO]   {processed:,} / {total_labels:,} labels processed")
    con.close()
    print(f"[INFO] Done. {processed:,} labels processed.")

    entropy_bits = {tk: {} for tk in task_keys}
    for tk in task_keys:
        for month, c in counts[tk].items():
            total = sum(c.values())
            if total == 0:
                entropy_bits[tk][month] = 0.0
                continue
            probs = [v / total for v in c.values()]
            entropy_bits[tk][month] = -sum(p * math.log2(p) for p in probs)
    return entropy_bits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--walk_forward_dir', required=True,
                     help='monthly_metrics_all_folds.csv を含む walk_forward 実行ディレクトリ')
    ap.add_argument('--tasks', nargs='+', default=list(TASKS), choices=list(TASKS),
                     help='対象タスク（省略時は全5タスク）')
    ap.add_argument('--output_dir', default=None)
    args = ap.parse_args()

    mpath = os.path.join(args.walk_forward_dir, 'monthly_metrics_all_folds.csv')
    if not os.path.exists(mpath):
        raise SystemExit(f"[ERROR] {mpath} が見つかりません（先に walk_forward 集計を実行してください）")
    monthly = pd.read_csv(mpath)
    # 'unknown' に加え、collection_date不正行（既知issue: '2022/2024' → yearmonth='2022/20'、
    # reference/sequences-241017_2.csv のZimbabwe/Harare 217件が原因、CLAUDE.md/メモリ参照）も除外する。
    monthly = monthly[monthly['yearmonth'].astype(str).str.match(r'^\d{4}-\d{2}$')]
    monthly = monthly.sort_values('yearmonth').reset_index(drop=True)
    months = monthly['yearmonth'].tolist()

    print(f"[INFO] 月別ターゲットエントロピーをDBから計算中（タスク: {', '.join(args.tasks)}、"
          f"split_type_wf非依存・読み取り専用）...")
    entropy_bits = _monthly_task_entropy_bits(args.tasks)

    out_dir = args.output_dir or os.path.join('outputs', 'transformer_260817', 'scripts',
                                               'entropy_accuracy', 'by_month')
    os.makedirs(out_dir, exist_ok=True)

    for tk in args.tasks:
        _idx, hit_col, vocab_size, label, color = TASKS[tk]
        monthly[f'{tk}_entropy_bits'] = [entropy_bits[tk].get(m, 0.0) for m in months]
        monthly[f'fano_max_{tk}_accuracy_pct'] = [
            (1 - fano_min_error_rate(h, vocab_size)) * 100 for h in monthly[f'{tk}_entropy_bits']
        ]
        if hit_col in monthly.columns:
            monthly[f'fano_gap_{tk}_pct'] = monthly[f'fano_max_{tk}_accuracy_pct'] - monthly[hit_col]

    csv_path = os.path.join(out_dir, 'entropy_fano_by_month.csv')
    monthly.to_csv(csv_path, index=False)
    print(f"[INFO] Saved: {csv_path}")

    xi = list(range(len(months)))
    n = len(args.tasks)
    fig, axes = plt.subplots(n, 1, figsize=(max(10, len(months) * 0.28), 4.2 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, tk in zip(axes, args.tasks):
        _idx, hit_col, vocab_size, label, color = TASKS[tk]
        if hit_col in monthly.columns:
            ax.plot(xi, monthly[hit_col], marker='o', ms=3, lw=1.2,
                    label=f'{label} (actual)', color=color)
            actual = monthly[hit_col].to_numpy()
            ceiling = monthly[f'fano_max_{tk}_accuracy_pct'].to_numpy()
            xi_arr = np.array(xi)
            # 同系色で塗り分け: 実測<=上限（伸びしろ区間）は薄め、実測>上限（上界を超えている区間）は暗め
            ax.fill_between(xi_arr, actual, ceiling, where=(actual <= ceiling),
                             color=color, alpha=0.10, interpolate=True,
                             label='gap (below ceiling)')
            ax.fill_between(xi_arr, actual, ceiling, where=(actual > ceiling),
                             color=color, alpha=0.45, interpolate=True,
                             label='actual exceeds ceiling')
            gap = monthly[f'fano_gap_{tk}_pct'].mean()
            gap_txt = f'  (mean gap={gap:.1f}pt)'
        else:
            gap_txt = ''
        ax.plot(xi, monthly[f'fano_max_{tk}_accuracy_pct'], marker='^', ms=3, lw=1.2,
                label='Fano theoretical ceiling', color=color, linestyle='--')
        ax.set_ylabel(f'{label}\nAccuracy (%)')
        ax.set_title(f'{label}: actual accuracy vs Fano theoretical ceiling{gap_txt}', fontsize=10, pad=22)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=7)

    # 最上段だけラベル付きでfold区切りを表示（タイトルと衝突しないよう高めに配置）。
    # 残りの段は縦線のみ引いて視線を下まで揃える。
    add_fold_annotations(axes[0], months, y=1.14)
    if len(axes) > 1:
        from transformer_260817.scripts.visualization._fold_annotate import _month_to_fold
        fold_ids = [_month_to_fold(m)[0] for m in months]
        boundaries = [i for i in range(1, len(fold_ids)) if fold_ids[i] != fold_ids[i - 1]]
        for ax in axes[1:]:
            for i in boundaries:
                ax.axvline(i - 0.5, color='black', ls='--', lw=0.8)

    step = max(1, len(months) // 24)
    axes[-1].set_xticks(xi[::step])
    axes[-1].set_xticklabels([months[i] for i in xi[::step]], rotation=90, fontsize=7)
    axes[-1].set_xlabel('Year-Month (walk-forward test timeline)')
    fig.suptitle('Actual accuracy vs Fano theoretical ceiling, by task', y=1.0, fontsize=13)
    fig.tight_layout()

    png_path = os.path.join(out_dir, 'entropy_fano_by_month.png')
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved: {png_path}")
    for tk in args.tasks:
        if f'fano_gap_{tk}_pct' in monthly.columns:
            print(f"[INFO] {TASKS[tk][3]}: 平均 Fano gap（理論上限-実測, pct pt）="
                  f"{monthly[f'fano_gap_{tk}_pct'].mean():.2f}")


if __name__ == '__main__':
    main()
