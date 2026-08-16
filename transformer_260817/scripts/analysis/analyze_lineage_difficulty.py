# --- transformer_260817/scripts/analyze_lineage_difficulty.py ---
# テスト系統ごとの「予測難易度」を2軸で定量化し可視化する。
#   1. ターゲット偏り (target_entropy) : raw_path の最終タイムステップ変異位置の Shannon entropy
#   2. 系統的新規性 (phylo_dist)       : alias_key.json による学習系統への最小系統距離
#
# Usage:
#   python -m transformer_260817.scripts.analyze_lineage_difficulty
#   python -m transformer_260817.scripts.analyze_lineage_difficulty \
#       --metrics outputs/transformer_260817/results/xxx/csv/test_lineage_metrics.csv
import os
import re
import json
import math
import argparse
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import stats

from transformer_260817 import config
from transformer_260817.db.connection import get_db_path, connect_db

ALIAS_KEY_PATH      = 'reference/phylogeny/alias_key.json'
MAX_SAMPLES_ENTROPY = 2000   # 系統ごとのエントロピー計算サンプル上限
MIN_TEST_SAMPLES    = 50     # 分析対象とする最小テストサンプル数


# ------------------------------------------------------------------ #
# alias_key による系統距離計算
# ------------------------------------------------------------------ #
def load_alias(path):
    with open(path) as f:
        alias_key = json.load(f)
    rev = {}
    for alias, full in alias_key.items():
        if isinstance(full, str) and full:
            rev[full] = alias
    return alias_key, rev


def expand(name, alias_key):
    parts = name.split('.')
    base = alias_key.get(parts[0], '')
    if isinstance(base, list):
        base = base[0].rstrip('*')
    if not base:
        return name
    return base + ('.' + '.'.join(parts[1:]) if len(parts) > 1 else '')


def lca_depth(fa, fb):
    pa, pb = fa.split('.'), fb.split('.')
    d = 0
    for a, b in zip(pa, pb):
        if a == b:
            d += 1
        else:
            break
    return d


def compute_min_phylo_dist(test_fulls, train_fulls):
    """各テスト系統から最近傍訓練系統への最小系統距離を計算する。"""
    # 訓練系統のプレフィックスマップを構築して探索を高速化
    prefix_to_train = defaultdict(list)
    for tf in train_fulls:
        parts = tf.split('.')
        for i in range(1, len(parts) + 1):
            prefix_to_train['.'.join(parts[:i])].append((tf, len(parts)))

    results = {}
    for test_full in test_fulls:
        test_parts = test_full.split('.')
        n = len(test_parts)
        min_dist = n + 10  # 上限初期値

        # テスト系統の祖先を上位から辿り、訓練系統との LCA を探す
        checked = set()
        for lca_d in range(n, 0, -1):
            ancestor = '.'.join(test_parts[:lca_d])
            if ancestor in prefix_to_train:
                for train_full, m in prefix_to_train[ancestor]:
                    if train_full in checked:
                        continue
                    checked.add(train_full)
                    actual_lca = lca_depth(test_full, train_full)
                    dist = (n - actual_lca) + (m - actual_lca)
                    if dist < min_dist:
                        min_dist = dist

        results[test_full] = min_dist if min_dist < n + 10 else n

    return results


# ------------------------------------------------------------------ #
# ターゲット変異位置エントロピーの計算（raw_path から）
# ------------------------------------------------------------------ #
MUT_POS_RE = re.compile(r'[A-Za-z](\d+)[A-Za-z]')


def parse_target_positions(raw_path: str):
    """raw_path の最終タイムステップから変異位置リストを返す。"""
    last_step = raw_path.split('>')[-1]
    muts = last_step.split(',')
    positions = []
    for m in muts:
        m = m.strip()
        match = MUT_POS_RE.search(m)
        if match:
            positions.append(int(match.group(1)))
    return positions


def calc_entropy_stats(positions):
    """変異位置リストから (H_bits, n_unique) を返す。"""
    if not positions:
        return 0.0, 0
    counts = Counter(positions)
    total = len(positions)
    h = -sum((c / total) * math.log2(c / total) for c in counts.values())
    return h, len(counts)


# ------------------------------------------------------------------ #
# データ読み込み
# ------------------------------------------------------------------ #
def load_from_db(alias_key):
    split_col = 'split_type_date' if getattr(config, 'SPLIT_MODE', 'timestep') in ('date', 'walk_forward') else 'split_type'
    sn_col = 'strain_name_usher' if getattr(config, 'STRENGTH_SOURCE', 'ncbi') == 'usher' else 'strain_name_ncbi'
    con = connect_db(get_db_path(), read_only=True)

    print("[INFO] Loading training lineages...")
    train_rows = con.execute(f"""
        SELECT DISTINCT st.{sn_col}
        FROM samples s JOIN strains st ON s.strain_id = st.strain_id
        WHERE s.{split_col} = 0 AND st.{sn_col} IS NOT NULL
          AND st.{sn_col} != 'unclassifiable'
    """).fetchall()
    train_lineages = [r[0] for r in train_rows]

    print("[INFO] Loading test lineages, raw_paths, and collection_dates...")
    test_rows = con.execute(f"""
        SELECT st.{sn_col}, s.raw_path, COUNT(*) OVER (PARTITION BY st.{sn_col}) as n,
               s.collection_date
        FROM samples s JOIN strains st ON s.strain_id = st.strain_id
        WHERE s.{split_col} = 2 AND st.{sn_col} IS NOT NULL
          AND st.{sn_col} != 'unclassifiable'
    """).fetchall()

    print("[INFO] Loading total DB sample counts per lineage...")
    db_total_rows = con.execute(f"""
        SELECT st.{sn_col}, st.sample_count
        FROM strains st
        WHERE st.{sn_col} IS NOT NULL AND st.{sn_col} != 'unclassifiable'
    """).fetchall()
    con.close()

    # 系統ごとにグループ化
    lineage_paths = defaultdict(list)
    lineage_count = {}
    lineage_dates = defaultdict(list)
    for lineage, raw_path, n, collection_date in test_rows:
        lineage_count[lineage] = n
        if len(lineage_paths[lineage]) < MAX_SAMPLES_ENTROPY:
            lineage_paths[lineage].append(raw_path)
        if collection_date:
            lineage_dates[lineage].append(str(collection_date)[:10])

    db_total_count = {row[0]: row[1] for row in db_total_rows}

    return train_lineages, lineage_paths, lineage_count, lineage_dates, db_total_count


# ------------------------------------------------------------------ #
# メイン処理
# ------------------------------------------------------------------ #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metrics', type=str, default=None,
                        help='lineage_metrics.csv のパス（recall付き分析に使用）')
    args = parser.parse_args()

    output_dir = os.path.join(config.OUTPUT_DIR, 'scripts', 'lineage')
    os.makedirs(output_dir, exist_ok=True)

    print("[INFO] Loading alias key...")
    alias_key, rev = load_alias(ALIAS_KEY_PATH)

    train_lineages, lineage_paths, lineage_count, lineage_dates, db_total_count = load_from_db(alias_key)
    print(f"[INFO] Train lineages: {len(train_lineages)}, Test lineages: {len(lineage_paths)}")

    # 最小サンプル数フィルタ
    test_lineages = [l for l, n in lineage_count.items() if n >= MIN_TEST_SAMPLES]
    print(f"[INFO] Test lineages (>= {MIN_TEST_SAMPLES} samples): {len(test_lineages)}")

    # 系統距離計算
    print("[INFO] Computing phylogenetic distances...")
    train_fulls = [expand(l, alias_key) for l in train_lineages]
    test_fulls_map = {l: expand(l, alias_key) for l in test_lineages}
    dist_map = compute_min_phylo_dist(list(test_fulls_map.values()), train_fulls)
    phylo_dist = {l: dist_map[test_fulls_map[l]] for l in test_lineages}

    # エントロピー計算
    print("[INFO] Computing target entropy...")
    entropy_map, n_unique_map = {}, {}
    for lineage in test_lineages:
        all_positions = []
        for raw_path in lineage_paths[lineage]:
            all_positions.extend(parse_target_positions(raw_path))
        h, n_unique = calc_entropy_stats(all_positions)
        entropy_map[lineage]  = h
        n_unique_map[lineage] = n_unique

    # 系統ごとの採取日中央値を計算
    print("[INFO] Computing median collection dates...")
    median_date_map = {}
    for l in test_lineages:
        dates = lineage_dates.get(l, [])
        valid = [d for d in dates if d and len(d) >= 7]
        if valid:
            valid_sorted = sorted(valid)
            median_date_map[l] = valid_sorted[len(valid_sorted) // 2]
        else:
            median_date_map[l] = None

    # DataFrame構築
    records = []
    max_dist = max(phylo_dist.values()) or 1
    for l in test_lineages:
        nd = phylo_dist[l] / max_dist
        n_uniq = n_unique_map[l]
        h_bits = entropy_map[l]
        max_h  = math.log2(n_uniq) if n_uniq > 1 else 1.0
        records.append({
            'lineage':               l,
            'n_test_sample':         lineage_count[l],
            'n_db_sample':           db_total_count.get(l, 0),
            'phylo_dist':            phylo_dist[l],
            'norm_phylo_dist':       round(nd, 4),
            'db_target_entropy_bits':   round(h_bits, 4),
            'db_target_entropy_norm':          round(h_bits / max_h, 4),
            'n_unique_positions':    n_uniq,
            'median_collection_date': median_date_map.get(l),
        })
    df = pd.DataFrame(records)

    # difficulty score: log10(n_unique+1) と phylo_dist を正規化して平均
    log_u = np.log10(df['n_unique_positions'] + 1)
    df['norm_log_unique']   = (log_u / log_u.max()).round(4)
    df['difficulty_score']  = (0.5 * df['norm_phylo_dist'] + 0.5 * df['norm_log_unique']).round(4)
    df = df.sort_values('difficulty_score', ascending=False).reset_index(drop=True)

    # CSV保存
    csv_path = os.path.join(output_dir, 'lineage_difficulty.csv')
    df.to_csv(csv_path, index=False)
    print(f"[INFO] Saved: {csv_path}")
    print(f"\n[TOP 10 難易度の高い系統]")
    print(df[['lineage', 'n_test_sample', 'n_db_sample', 'phylo_dist',
              'db_target_entropy_bits', 'db_target_entropy_norm', 'n_unique_positions', 'difficulty_score']].head(10).to_string(index=False))

    # ----------------------------------------------------------------
    # 共通ヘルパー: jitter 付き散布図（周辺ヒストグラム）
    # ----------------------------------------------------------------
    rng = np.random.default_rng(42)

    def scatter_with_marginals(source_df, x_vals, x_label, x_log, filename, top_df,
                               color_col='difficulty_score',
                               color_label='Difficulty score',
                               cmap='RdYlGn_r', vmin=None, vmax=None,
                               title_ref_col=None):
        fig = plt.figure(figsize=(11, 9))
        gs  = fig.add_gridspec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
                               hspace=0.05, wspace=0.05)
        ax_main  = fig.add_subplot(gs[1, 0])
        ax_top   = fig.add_subplot(gs[0, 0], sharex=ax_main)
        ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

        # Y: phylo_dist (integer) + jitter
        y_raw    = source_df['phylo_dist'].values.astype(float)
        y_jitter = y_raw + rng.uniform(-0.35, 0.35, len(y_raw))

        if vmin is None and color_col == 'difficulty_score':
            vmin = 0
        if vmax is None and color_col == 'difficulty_score':
            vmax = 1

        sc = ax_main.scatter(
            x_vals, y_jitter,
            s=np.log1p(source_df['n_test_samples']) * 6,
            c=source_df[color_col], cmap=cmap,
            vmin=vmin, vmax=vmax,
            alpha=0.65, edgecolors='none',
        )
        plt.colorbar(sc, ax=ax_right, label=color_label, fraction=0.8, pad=0.1)

        # 上位10系統にラベル
        top_x = x_vals[top_df.index]
        top_y = y_jitter[top_df.index]
        for xi, yi, row in zip(top_x, top_y, top_df.itertuples()):
            ax_main.annotate(row.lineage, (xi, yi), fontsize=7, alpha=0.9,
                             xytext=(4, 2), textcoords='offset points')

        ax_main.set_ylabel('Phylo distance to nearest train lineage')
        ax_main.set_xlabel(x_label)
        ax_main.set_yticks(sorted(source_df['phylo_dist'].unique()))
        ax_main.grid(linestyle='--', alpha=0.3)

        # 上部ヒストグラム
        ax_top.hist(x_vals, bins=40, color='steelblue', alpha=0.7)
        ax_top.set_ylabel('Count')
        plt.setp(ax_top.get_xticklabels(), visible=False)
        ax_top.grid(linestyle='--', alpha=0.3)

        # 右部ヒストグラム（phylo_dist の整数分布）
        dist_counts = source_df['phylo_dist'].value_counts().sort_index()
        ax_right.barh(dist_counts.index, dist_counts.values, height=0.6,
                      color='steelblue', alpha=0.7)
        ax_right.set_xlabel('Count')
        plt.setp(ax_right.get_yticklabels(), visible=False)
        ax_right.grid(linestyle='--', alpha=0.3)

        ref_col = title_ref_col if title_ref_col and title_ref_col in source_df.columns else 'phylo_dist'
        ref_label = title_ref_col if title_ref_col else 'phylo_dist'
        try:
            r_val, p_val = stats.pearsonr(x_vals, source_df[ref_col])
            title_str = f'Pearson r={r_val:.3f}, p={p_val:.3e} (x vs {ref_label}), n={len(source_df)}'
        except Exception:
            title_str = f'Pearson r=N/A (x vs {ref_label}), n={len(source_df)}'
        fig.suptitle(f'Lineage prediction difficulty  ({title_str})', y=0.98, fontsize=11)
        if x_log:
            ax_main.set_xscale('log')
            ax_top.set_xscale('log')

        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[INFO] Saved: {filename}")

    def scatter_with_date_y(source_df, x_vals, x_label, filename, top_df,
                            color_col='difficulty_score',
                            color_label='Difficulty score',
                            cmap='RdYlGn_r', vmin=None, vmax=None):
        """縦軸を採取日中央値にした散布図（周辺ヒストグラム付き）。"""
        import matplotlib.dates as mdates

        def _normalize_date(d):
            """YYYY-MM → YYYY-MM-01 に補完し、パース不可なら None を返す。"""
            if not d or not isinstance(d, str):
                return None
            d = d.strip()
            if len(d) == 7:
                d = d + '-01'
            try:
                return pd.Timestamp(d)
            except Exception:
                return None

        # source_df をリセットして numpy と 1:1 対応させる
        src = source_df.reset_index(drop=True)
        timestamps = [_normalize_date(src.at[i, 'median_collection_date']) for i in range(len(src))]
        valid_mask = np.array([t is not None for t in timestamps])

        if valid_mask.sum() < 5:
            print(f"[WARNING] scatter_with_date_y: 有効な採取日が不足 ({valid_mask.sum()} 件). スキップ.")
            return

        x_date = x_vals[valid_mask]
        src_valid = src[valid_mask].reset_index(drop=True)
        y_ts = [timestamps[i] for i, v in enumerate(valid_mask) if v]
        y_num = np.array(mdates.date2num(y_ts))

        if vmin is None and color_col == 'difficulty_score':
            vmin = 0
        if vmax is None and color_col == 'difficulty_score':
            vmax = 1

        fig = plt.figure(figsize=(11, 9))
        gs = fig.add_gridspec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
                              hspace=0.05, wspace=0.05)
        ax_main  = fig.add_subplot(gs[1, 0])
        ax_top   = fig.add_subplot(gs[0, 0], sharex=ax_main)
        ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

        sc = ax_main.scatter(
            x_date, y_num,
            s=np.log1p(src_valid['n_test_samples']) * 6,
            c=src_valid[color_col], cmap=cmap,
            vmin=vmin, vmax=vmax,
            alpha=0.65, edgecolors='none',
        )
        plt.colorbar(sc, ax=ax_right, label=color_label, fraction=0.8, pad=0.1)

        top_lineages = set(top_df['lineage'])
        for idx, row in src_valid.iterrows():
            if row['lineage'] in top_lineages:
                ax_main.annotate(
                    row['lineage'],
                    (x_date[idx], y_num[idx]),
                    fontsize=7, alpha=0.9,
                    xytext=(4, 2), textcoords='offset points',
                )

        ax_main.yaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax_main.yaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax_main.get_yticklabels(), rotation=45, ha='right')
        ax_main.set_ylabel('Median collection date')
        ax_main.set_xlabel(x_label)
        ax_main.grid(linestyle='--', alpha=0.3)

        ax_top.hist(x_date, bins=40, color='steelblue', alpha=0.7)
        ax_top.set_ylabel('Count')
        plt.setp(ax_top.get_xticklabels(), visible=False)
        ax_top.grid(linestyle='--', alpha=0.3)

        ax_right.hist(y_num, bins=20, color='steelblue', alpha=0.7, orientation='horizontal')
        ax_right.yaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax_right.set_xlabel('Count')
        plt.setp(ax_right.get_yticklabels(), visible=False)
        ax_right.grid(linestyle='--', alpha=0.3)

        try:
            r_val, p_val = stats.pearsonr(x_date, y_num)
            title_str = f'Pearson r={r_val:.3f}, p={p_val:.3e} (x vs median date), n={len(src_valid)}'
        except Exception:
            title_str = f'n={len(src_valid)}'
        fig.suptitle(f'Lineage difficulty vs Collection date  ({title_str})', y=0.98, fontsize=11)

        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[INFO] Saved: {filename}")

    top10 = df.head(10)

    # 図A: Shannon entropy H (bits)
    out_a = os.path.join(output_dir, 'lineage_difficulty_scatter_entropy.png')
    scatter_with_marginals(
        df,
        df['db_target_entropy_bits'].values,
        'Shannon entropy H  (bits)',
        x_log=False,
        filename=out_a,
        top_df=top10,
    )

    # 図B: n_unique_positions (log10 scale)
    out_b = os.path.join(output_dir, 'lineage_difficulty_scatter_unique.png')
    scatter_with_marginals(
        df,
        (df['n_unique_positions'] + 1).values.astype(float),
        'Unique target mutation positions  (log10 scale)',
        x_log=True,
        filename=out_b,
        top_df=top10,
    )

    # 図E: 採取日中央値 × Shannon entropy（difficulty_score カラー）
    out_e = os.path.join(output_dir, 'lineage_difficulty_scatter_entropy_date.png')
    scatter_with_date_y(
        df,
        df['db_target_entropy_bits'].values,
        'Shannon entropy H  (bits)',
        filename=out_e,
        top_df=top10,
    )

    # ----------------------------------------------------------------
    # 図2: lineage_metrics.csv がある場合 → difficulty vs recall
    # ----------------------------------------------------------------
    if args.metrics and os.path.exists(args.metrics):
        metrics_df = pd.read_csv(args.metrics)
        # 系統名カラムを探す
        name_col   = next((c for c in metrics_df.columns if 'lineage' in c.lower()), None)
        # 優先順: position_hit_rate → 他の hit_rate → recall
        recall_col = (
            next((c for c in metrics_df.columns if 'position_hit_rate' in c.lower()), None)
            or next((c for c in metrics_df.columns if 'hit_rate' in c.lower()), None)
            or next((c for c in metrics_df.columns if 'recall' in c.lower()), None)
        )
        if name_col and recall_col:
            merged = df.merge(metrics_df[[name_col, recall_col]].rename(
                columns={name_col: 'lineage', recall_col: 'recall'}), on='lineage', how='inner')
            if len(merged) > 5:
                fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                cols = [
                    ('db_target_entropy_bits', 'Shannon entropy H (bits)'),
                    ('n_unique_positions',  'Unique target positions'),
                    ('norm_phylo_dist',     'Norm phylo distance to train'),
                ]
                for ax, (xcol, xlabel) in zip(axes, cols):
                    xv = np.log10(merged[xcol] + 1) if xcol == 'n_unique_positions' else merged[xcol]
                    ax.scatter(xv, merged['recall'],
                               s=np.log1p(merged['n_test_samples']) * 6,
                               alpha=0.6, edgecolors='none', c='steelblue')
                    xl = f'log10({xlabel})' if xcol == 'n_unique_positions' else xlabel
                    ax.set_xlabel(xl)
                    ax.set_ylabel(f'{recall_col} (lineage-level)')
                    ax.grid(linestyle='--', alpha=0.3)
                    try:
                        r, p = stats.pearsonr(xv, merged['recall'])
                        z = np.polyfit(xv, merged['recall'], 1)
                        xline = np.linspace(xv.min(), xv.max(), 100)
                        ax.plot(xline, np.polyval(z, xline), 'r--', lw=1.2)
                        ax.set_title(f'r={r:.3f}, p={p:.3e}')
                    except Exception:
                        r, p = float('nan'), float('nan')
                        ax.set_title('r=N/A (constant input)')
                plt.suptitle('Prediction difficulty vs Recall', fontsize=12)
                plt.tight_layout()
                out2 = os.path.join(output_dir, 'lineage_difficulty_vs_recall.png')
                plt.savefig(out2, dpi=150)
                plt.close()
                print(f"[INFO] Saved: {out2}")
                print(f"\n[相関サマリ] ({recall_col})")
                for xcol, xlabel in cols:
                    xv = np.log10(merged[xcol] + 1) if xcol == 'n_unique_positions' else merged[xcol]
                    try:
                        r, p = stats.pearsonr(xv, merged['recall'])
                        print(f"  {xlabel:<35} vs {recall_col}: r={r:.3f}, p={p:.3e}")
                    except Exception:
                        print(f"  {xlabel:<35} vs {recall_col}: r=N/A (constant input)")

                # 図C/D: recall カラーリング版の scatter_with_marginals
                merged_sorted = merged.sort_values('difficulty_score', ascending=False).reset_index(drop=True)
                top10_merged = merged_sorted.head(10)

                out_c = os.path.join(output_dir, 'lineage_difficulty_scatter_entropy_recall.png')
                scatter_with_marginals(
                    merged_sorted,
                    merged_sorted['db_target_entropy_bits'].values,
                    'Shannon entropy H  (bits)',
                    x_log=False,
                    filename=out_c,
                    top_df=top10_merged,
                    color_col='recall',
                    color_label=f'{recall_col} (lineage-level)',
                    cmap='RdYlGn',
                    title_ref_col='recall',
                )

                out_d = os.path.join(output_dir, 'lineage_difficulty_scatter_unique_recall.png')
                scatter_with_marginals(
                    merged_sorted,
                    (merged_sorted['n_unique_positions'] + 1).values.astype(float),
                    'Unique target mutation positions  (log10 scale)',
                    x_log=True,
                    filename=out_d,
                    top_df=top10_merged,
                    color_col='recall',
                    color_label=f'{recall_col} (lineage-level)',
                    cmap='RdYlGn',
                    title_ref_col='recall',
                )

                # 図F: 採取日中央値 × Shannon entropy（recall カラー）
                out_f = os.path.join(output_dir, 'lineage_difficulty_scatter_entropy_date_recall.png')
                scatter_with_date_y(
                    merged_sorted,
                    merged_sorted['db_target_entropy_bits'].values,
                    'Shannon entropy H  (bits)',
                    filename=out_f,
                    top_df=top10_merged,
                    color_col='recall',
                    color_label=f'{recall_col} (lineage-level)',
                    cmap='RdYlGn',
                    vmin=merged_sorted['recall'].min(),
                    vmax=merged_sorted['recall'].max(),
                )
    else:
        if args.metrics:
            print(f"[WARNING] metrics file not found: {args.metrics}")
        print("[INFO] --metrics を指定するとrecallとの相関分析も出力されます")

    print("[INFO] Done.")


if __name__ == '__main__':
    main()
