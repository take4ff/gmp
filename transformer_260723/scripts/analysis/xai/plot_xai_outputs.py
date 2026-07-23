# --- transformer_260723/scripts/visualization/plot_xai_outputs.py ---
"""既存のXAI分析出力（CSV/JSONのみで図が無いもの）を可視化する。

対象:
  local_explain      : local_explanations.json → サンプルごとの上位寄与特徴を小さい棒グラフで並べる
  constellation       : model/target_cooccurrence_pairs.csv → 共起ペアのネットワーク図
  majority_baseline    : majority_baseline_by_fold.csv → フォールド別の多数決/ランダム基準比較

Usage:
  python -m transformer_260723.scripts.visualization.plot_xai_outputs local_explain \\
      --json outputs/transformer_260723/scripts/local_explain/<ts>/local_explanations.json
  python -m transformer_260723.scripts.visualization.plot_xai_outputs constellation \\
      --dir outputs/transformer_260723/scripts/constellation_walk_forward/<ts>
  python -m transformer_260723.scripts.visualization.plot_xai_outputs majority_baseline \\
      --csv outputs/transformer_260723/scripts/majority_baseline/<ts>/majority_baseline_by_fold.csv
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Okabe-Ito系の色覚多様性対応カラー（他のvisualizationスクリプトの配色系統に合わせた寒色/暖色ペア基調）
BLUE = '#2b5c8f'
ORANGE = '#d95f02'
GREEN = '#1b9e77'
GRAY = '#888888'


def plot_local_explanations(json_path, out_dir=None):
    with open(json_path) as f:
        explanations = json.load(f)

    out_dir = out_dir or os.path.dirname(json_path)
    n = len(explanations)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows), squeeze=False)

    for i, e in enumerate(explanations):
        ax = axes[i // ncols][i % ncols]
        feats = e['top_features'][::-1]
        names = [f['feature'] for f in feats]
        vals = [f['attribution'] for f in feats]
        ax.barh(names, vals, color=BLUE)
        voc_tag = f"\n[{e['voc_mutation_name']}]" if e.get('is_known_voc_position') else ''
        ax.set_title(f"sample {e['sample_index']}: pos={e['predicted_next_position']} "
                    f"({e['predicted_gene']}){voc_tag}", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.set_xlabel('|IG| attribution', fontsize=8)

    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis('off')

    fig.suptitle('local_explain: top attributed features per sample (Integrated Gradients)', y=1.02)
    plt.tight_layout()
    path = os.path.join(out_dir, 'local_explanations_features.png')
    fig.savefig(path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'[INFO] Saved {path}')

    # 入力変異(timestep)側も同様に並べる
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows), squeeze=False)
    for i, e in enumerate(explanations):
        ax = axes[i // ncols][i % ncols]
        muts = e['top_input_mutations'][::-1]
        labels = [f"{m['position']}({m['gene']})" for m in muts]
        vals = [m['attribution'] for m in muts]
        ax.barh(labels, vals, color=ORANGE)
        ax.set_title(f"sample {e['sample_index']}: pos={e['predicted_next_position']} "
                    f"({e['predicted_gene']})", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.set_xlabel('|IG| attribution', fontsize=8)

    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis('off')

    fig.suptitle('local_explain: top attributed input mutations per sample', y=1.02)
    plt.tight_layout()
    path = os.path.join(out_dir, 'local_explanations_mutations.png')
    fig.savefig(path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'[INFO] Saved {path}')


def plot_cooccurrence_constellation(dir_path, top_n=25, out_dir=None):
    import networkx as nx

    model_df = pd.read_csv(os.path.join(dir_path, 'model_cooccurrence_pairs.csv')).head(top_n)
    target_df = pd.read_csv(os.path.join(dir_path, 'target_cooccurrence_pairs.csv')).head(top_n)
    out_dir = out_dir or dir_path

    def _pair_key(row):
        return tuple(sorted((int(row['pos_a']), int(row['pos_b']))))

    model_pairs = {_pair_key(r): r for _, r in model_df.iterrows()}
    target_pairs = {_pair_key(r): r for _, r in target_df.iterrows()}
    both = set(model_pairs) & set(target_pairs)
    model_only = set(model_pairs) - both
    target_only = set(target_pairs) - both

    G = nx.Graph()
    edge_colors = []
    edge_groups = []
    all_pairs = [(k, model_pairs.get(k, target_pairs.get(k)), 'both' if k in both
                 else ('model_only' if k in model_only else 'target_only'))
                for k in list(both) + list(model_only) + list(target_only)]

    color_map = {'both': GREEN, 'model_only': ORANGE, 'target_only': BLUE}
    node_gene = {}
    for (a, b), row, group in all_pairs:
        G.add_edge(a, b, group=group)
        node_gene[a] = row['gene_a']
        node_gene[b] = row['gene_b']

    pos = nx.spring_layout(G, seed=42, k=0.6)
    fig, ax = plt.subplots(figsize=(11, 11))
    for group, color in color_map.items():
        edges = [(u, v) for u, v, d in G.edges(data=True) if d['group'] == group]
        nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=color, width=1.6, alpha=0.7, ax=ax)

    nx.draw_networkx_nodes(G, pos, node_color='white', edgecolors=GRAY, node_size=380, ax=ax)
    labels = {n: f"{n}\n({node_gene.get(n, '?')})" for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=6.5, ax=ax)

    from matplotlib.lines import Line2D
    legend_items = [
        Line2D([0], [0], color=GREEN, lw=2, label=f'Both (model & target) — {len(both)}'),
        Line2D([0], [0], color=ORANGE, lw=2, label=f'Model-predicted only — {len(model_only)}'),
        Line2D([0], [0], color=BLUE, lw=2, label=f'Target (real) only — {len(target_only)}'),
    ]
    ax.legend(handles=legend_items, loc='upper left', fontsize=9)
    ax.set_title(f'Co-occurrence constellation: top-{top_n} predicted vs. real pairs')
    ax.axis('off')
    plt.tight_layout()
    path = os.path.join(out_dir, 'cooccurrence_constellation_network.png')
    fig.savefig(path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'[INFO] Saved {path}')


def plot_majority_baseline(csv_path, out_dir=None):
    df = pd.read_csv(csv_path)
    out_dir = out_dir or os.path.dirname(csv_path)
    labels = [f"fold{r.fold_id}\n{r.era}" for r in df.itertuples()]
    x = np.arange(len(df))
    w = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    ax1.bar(x - w / 2, df['majority_region_accuracy_pct'], width=w, color=BLUE, label='Majority vote')
    ax1.bar(x + w / 2, df['uniform_random_region_pct'], width=w, color=GRAY, label='Uniform random')
    ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=8, rotation=30, ha='right')
    ax1.set_ylabel('Region accuracy (%)')
    ax1.set_title('Region-level baseline accuracy by fold')
    ax1.legend(fontsize=9)

    ax2.bar(x - w / 2, df['majority_position_accuracy_pct'], width=w, color=ORANGE, label='Majority vote')
    ax2.bar(x + w / 2, df['uniform_random_position_pct'], width=w, color=GRAY, label='Uniform random')
    ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=8, rotation=30, ha='right')
    ax2.set_ylabel('Position accuracy (%)')
    ax2.set_yscale('symlog', linthresh=0.1)
    ax2.set_title('Position-level baseline accuracy by fold (symlog y)')
    ax2.legend(fontsize=9)

    plt.tight_layout()
    path = os.path.join(out_dir, 'majority_baseline_by_fold.png')
    fig.savefig(path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'[INFO] Saved {path}')


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)

    p1 = sub.add_parser('local_explain')
    p1.add_argument('--json', required=True)
    p1.add_argument('--output_dir', default=None)

    p2 = sub.add_parser('constellation')
    p2.add_argument('--dir', required=True)
    p2.add_argument('--top_n', type=int, default=25)
    p2.add_argument('--output_dir', default=None)

    p3 = sub.add_parser('majority_baseline')
    p3.add_argument('--csv', required=True)
    p3.add_argument('--output_dir', default=None)

    args = ap.parse_args()
    if args.cmd == 'local_explain':
        plot_local_explanations(args.json, args.output_dir)
    elif args.cmd == 'constellation':
        plot_cooccurrence_constellation(args.dir, args.top_n, args.output_dir)
    elif args.cmd == 'majority_baseline':
        plot_majority_baseline(args.csv, args.output_dir)


if __name__ == '__main__':
    main()
