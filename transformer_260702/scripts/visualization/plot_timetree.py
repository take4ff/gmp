# --- transformer_260702/scripts/plot_timetree.py ---
# alias_key.json を使って真の系統関係を解析し、時系列系統樹を描画する。
# X軸: ピーク月（最多サンプル月）, Y軸: DFS葉カウントによるツリー配置
# 組換え系統 (X*) は第一親系統に接続し破線で表示。
# Usage: python -m transformer_260702.scripts.plot_timetree
import os
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from transformer_260702 import config
from transformer_260702.db.connection import get_db_path, connect_db

ALIAS_KEY_PATH = 'reference/phylogeny/alias_key.json'
MIN_SAMPLES    = 5_000    # ノードとして表示する最小サンプル数
MAX_NODES      = 150      # 表示ノード上限

GROUP_COLORS = {
    'Early':       '#aaaaaa',
    'Alpha':       '#d62728',
    'Beta/Gamma':  '#ff7f0e',
    'Delta':       '#1f77b4',
    'BA.1':        '#2ca02c',
    'BA.2':        '#98df8a',
    'BA.4/5':      '#ffbb78',
    'BQ':          '#e377c2',
    'XBB':         '#9467bd',
    'JN/KP':       '#17becf',
    'Other':       '#dddddd',
}

def get_group(name):
    if not name: return 'Other'
    if name.startswith('AY.') or name == 'B.1.617.2': return 'Delta'
    if name.startswith('B.1.1.7'): return 'Alpha'
    if name.startswith('B.1.351') or name.startswith('P.'): return 'Beta/Gamma'
    if name.startswith('BA.1'): return 'BA.1'
    if name.startswith('BA.2.86') or name.startswith('JN.') or name.startswith('KP.'): return 'JN/KP'
    if name.startswith('BA.2'): return 'BA.2'
    if name.startswith('BA.4') or name.startswith('BA.5') or name.startswith('BF.') or name.startswith('BE.'): return 'BA.4/5'
    if name.startswith('BQ.'): return 'BQ'
    if name.startswith('X') or name.startswith('EG.') or name.startswith('HK.'): return 'XBB'
    if name in ('A', 'B') or name.startswith('B.1') or name.startswith('B.2'): return 'Early'
    return 'Other'


# ------------------------------------------------------------------ #
# alias_key を使った系統名解決
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
    """'JN.1' → 'B.1.1.529.2.86.1.1'"""
    parts = name.split('.')
    base = alias_key.get(parts[0], '')
    if isinstance(base, list):
        base = base[0].rstrip('*')
    if not base:
        return name
    return base + ('.' + '.'.join(parts[1:]) if len(parts) > 1 else '')

def compress(full, rev):
    """'B.1.1.529.2.86.1' → 'JN'"""
    parts = full.split('.')
    for i in range(len(parts), 0, -1):
        sub = '.'.join(parts[:i])
        if sub in rev:
            rest = parts[i:]
            return rev[sub] + ('.' + '.'.join(rest) if rest else '')
    return full

def is_recombinant(name, alias_key):
    return isinstance(alias_key.get(name.split('.')[0], ''), list)

def get_parent(name, alias_key, rev):
    """真の親系統名を返す（alias_key経由）"""
    full = expand(name, alias_key)
    parts = full.split('.')
    if len(parts) <= 1:
        return None
    return compress('.'.join(parts[:-1]), rev)


# ------------------------------------------------------------------ #
# データ読み込み
# ------------------------------------------------------------------ #
def load_lineage_data():
    con = connect_db(get_db_path(), read_only=True)
    rows = con.execute("""
        SELECT
            st.strain_name_usher             AS lineage,
            substr(s.collection_date, 1, 7)  AS month,
            COUNT(*)                         AS month_n
        FROM samples s JOIN strains st ON s.strain_id = st.strain_id
        WHERE length(s.collection_date) >= 7
          AND substr(s.collection_date, 5, 1) = '-'
          AND st.strain_name_usher IS NOT NULL
          AND st.strain_name_usher != 'unclassifiable'
        GROUP BY lineage, month
        ORDER BY lineage, month
    """).fetchall()
    con.close()

    df = pd.DataFrame(rows, columns=['lineage', 'month', 'month_n'])

    records = []
    for lineage, grp in df.groupby('lineage'):
        peak_row = grp.loc[grp['month_n'].idxmax()]
        records.append({
            'lineage':     lineage,
            'total_n':     grp['month_n'].sum(),
            'first_month': grp['month'].min(),
            'peak_month':  peak_row['month'],
        })
    summary = pd.DataFrame(records)
    return df, summary


# ------------------------------------------------------------------ #
# ノード選択 + ツリー構築
# ------------------------------------------------------------------ #
def build_tree(summary, alias_key, rev, min_samples, max_nodes):
    # 上位ノード選択
    top = summary[summary['total_n'] >= min_samples].nlargest(max_nodes, 'total_n')
    selected = set(top['lineage'].tolist())

    # 各ノードの最近共通祖先（selected内）を探す
    parent_map = {}   # child → parent
    recom_set  = set()

    for node in list(selected):
        if is_recombinant(node, alias_key):
            recom_set.add(node)
        anc = get_parent(node, alias_key, rev)
        found = False
        while anc is not None:
            if anc in selected:
                parent_map[node] = anc
                found = True
                break
            anc = get_parent(anc, alias_key, rev)
        if not found:
            parent_map[node] = '__ROOT__'

    selected.add('__ROOT__')
    parent_map['__ROOT__'] = None

    # children map
    children_map = {n: [] for n in selected}
    for child, par in parent_map.items():
        if par in children_map:
            children_map[par].append(child)

    # first_month / total_n を ROOT に付与
    info = summary.set_index('lineage').to_dict('index')
    info['__ROOT__'] = {'total_n': 0, 'first_month': '2019-11', 'peak_month': '2019-11'}

    return selected, parent_map, children_map, recom_set, info


# ------------------------------------------------------------------ #
# レイアウト計算
# ------------------------------------------------------------------ #
def compute_layout(children_map, info, all_months):
    month_idx = {m: i for i, m in enumerate(all_months)}

    # DFS で Y 位置を割り当て
    y_counter = [0]
    y_pos = {}

    def dfs(node):
        children = sorted(children_map.get(node, []),
                          key=lambda c: info[c]['first_month'])
        if not children:
            y_pos[node] = y_counter[0]
            y_counter[0] += 1
        else:
            ys = []
            for c in children:
                dfs(c)
                ys.append(y_pos[c])
            y_pos[node] = (min(ys) + max(ys)) / 2

    dfs('__ROOT__')

    # X 位置: peak_month のインデックス
    x_pos = {}
    for node, d in info.items():
        m = d.get('peak_month', d.get('first_month', all_months[0]))
        x_pos[node] = month_idx.get(m, 0)

    return x_pos, y_pos


# ------------------------------------------------------------------ #
# 描画
# ------------------------------------------------------------------ #
def draw_tree(selected, parent_map, children_map, recom_set, info,
              x_pos, y_pos, all_months, output_dir):

    month_idx = {m: i for i, m in enumerate(all_months)}
    max_n = max(d['total_n'] for d in info.values() if d['total_n'] > 0)

    fig, ax = plt.subplots(figsize=(22, max(10, len(selected) * 0.3)))

    # エッジ描画
    for child, par in parent_map.items():
        if par is None or child == '__ROOT__':
            continue
        xc, yc = x_pos[child], y_pos[child]
        xp, yp = x_pos[par],   y_pos[par]
        is_recom = child in recom_set
        ls = '--' if is_recom else '-'
        lw = 0.8
        color = '#999999'
        # 水平線（親X → 子X）at 親Y
        ax.plot([xp, xc], [yp, yp], ls=ls, lw=lw, color=color, zorder=1)
        # 垂直線（親Y → 子Y）at 子X
        ax.plot([xc, xc], [yp, yc], ls=ls, lw=lw, color=color, zorder=1)

    # ノード描画
    for node in selected:
        if node == '__ROOT__':
            continue
        x, y = x_pos[node], y_pos[node]
        n = info[node]['total_n']
        size = 30 + 180 * math.log1p(n) / math.log1p(max_n)
        color = GROUP_COLORS.get(get_group(node), '#dddddd')
        ec = 'dimgray' if node in recom_set else 'none'
        ax.scatter(x, y, s=size, color=color, alpha=0.85,
                   edgecolors=ec, linewidths=0.8, zorder=3)
        ax.text(x + 0.15, y, node, fontsize=9, va='center', zorder=4)

    # X軸 ラベル（3ヶ月ごと）
    tick_step = 3
    ticks = [i for i in range(len(all_months)) if i % tick_step == 0]
    ax.set_xticks(ticks)
    ax.set_xticklabels([all_months[i] for i in ticks], rotation=45, ha='right', fontsize=8)
    ax.set_xlim(-1, len(all_months))
    ax.set_yticks([])
    ax.set_xlabel('Peak month (highest sample count)')
    ax.set_title('SARS-CoV-2 lineage timetree  (dashed = recombinant,  node size ∝ log(samples))')
    ax.grid(axis='x', linestyle='--', alpha=0.3)

    # 凡例
    handles = [mpatches.Patch(color=c, label=g) for g, c in GROUP_COLORS.items()]
    ax.legend(handles=handles, loc='upper left', fontsize=7, ncol=2, framealpha=0.7)

    plt.tight_layout()
    out = os.path.join(output_dir, 'lineage_timetree.png')
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[INFO] Saved: {out}")


# ------------------------------------------------------------------ #
# main
# ------------------------------------------------------------------ #
def main():
    output_dir = os.path.join(config.OUTPUT_DIR, 'scripts', 'lineage')
    os.makedirs(output_dir, exist_ok=True)

    print("[INFO] Loading alias key...")
    alias_key, rev = load_alias(ALIAS_KEY_PATH)

    print("[INFO] Loading lineage data from DB...")
    df, summary = load_lineage_data()
    all_months = sorted(df['month'].unique())
    print(f"[INFO] Lineages: {len(summary)}, Months: {len(all_months)}")

    print("[INFO] Building tree...")
    selected, parent_map, children_map, recom_set, info = build_tree(
        summary, alias_key, rev, MIN_SAMPLES, MAX_NODES)
    print(f"[INFO] Selected nodes: {len(selected)-1}  Recombinants: {len(recom_set)}")

    print("[INFO] Computing layout...")
    x_pos, y_pos = compute_layout(children_map, info, all_months)

    print("[INFO] Drawing timetree...")
    draw_tree(selected, parent_map, children_map, recom_set, info,
              x_pos, y_pos, all_months, output_dir)
    print("[INFO] Done.")


if __name__ == '__main__':
    main()
