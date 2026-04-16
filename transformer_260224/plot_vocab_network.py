"""
語彙ネットワーク描画スクリプト
学習済みモデルの埋め込み層からコサイン類似度ネットワーク等を描画する。

対応する埋め込み層:
  - base_embed: 塩基 (A, T, C, G, N, n) — ネットワーク + ヒートマップ
  - pos_embed: 塩基位置 (30K語彙) — 3パターン可視化
    1. 閾値フィルタ付きネットワークグラフ
    2. t-SNE 散布図 (遺伝子領域カラーリング)
    3. クラスタ間ネットワーク (k-means)

使用例:
 python -m transformer_260224.plot_vocab_network --model_path outputs/transformer_260112/results/20260118_003806/best_model.pth
"""
import sys
import os
import argparse
import importlib
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans


# ===== ユーティリティ =====

def load_config_from_model_dir(model_path: str):
    """モデルパスから対応するconfigを自動検出してロードする。"""
    from transformer_260224 import config as default_config
    abs_path = os.path.abspath(model_path)
    parts = abs_path.split(os.sep)
    for i, part in enumerate(parts):
        if part.startswith('transformer_') and i > 0 and parts[i - 1] == 'outputs':
            pkg_name = part
            try:
                cfg = importlib.import_module(f'{pkg_name}.config')
                print(f"Config: {pkg_name}.config を使用")
                return cfg
            except ImportError:
                pass
    print(f"Config: transformer_260224.config を使用 (デフォルト)")
    return default_config


def load_model(model_path: str, cfg):
    """学習済みモデルをロードする"""
    abs_path = os.path.abspath(model_path)
    parts = abs_path.split(os.sep)
    model_module = None
    for i, part in enumerate(parts):
        if part.startswith('transformer_') and i > 0 and parts[i - 1] == 'outputs':
            pkg_name = part
            try:
                model_module = importlib.import_module(f'{pkg_name}.model')
                print(f"Model: {pkg_name}.model を使用")
                break
            except ImportError:
                pass
    if model_module is None:
        from transformer_260224 import model as model_module
        print(f"Model: transformer_260224.model を使用 (デフォルト)")

    model = model_module.HierarchicalTransformer()
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model


def extract_embedding_weights(model, layer_name: str) -> np.ndarray:
    """指定した埋め込み層の重み行列を取得する"""
    embed_layer = getattr(model.input_embed, layer_name)
    return embed_layer.weight.detach().cpu().numpy()


def compute_cosine_similarity_matrix(weights: np.ndarray) -> np.ndarray:
    """コサイン類似度行列を計算する"""
    return cosine_similarity(weights)


def extract_model_version(model_path: str) -> str:
    """モデルパスからバージョン（タイムスタンプ）を抽出する。"""
    parts = os.path.normpath(model_path).split(os.sep)
    for i, part in enumerate(parts):
        if part == 'results' and i + 1 < len(parts):
            return parts[i + 1]
    return os.path.basename(os.path.dirname(model_path))


def load_position_to_region(codon_csv_path: str) -> dict:
    """codon_mutation CSVから 塩基位置→遺伝子領域 のマッピングを読み込む"""
    df = pd.read_csv(codon_csv_path, encoding='utf-8-sig')
    pos_to_region = {}
    for _, row in df.iterrows():
        pos_to_region[int(row['base_pos'])] = str(row['protein'])
    return pos_to_region


# ===== 遺伝子領域カラーマップ =====

REGION_COLOR_MAP = {
    # 構造タンパク質 (暖色系)
    'S': '#e94560',      # Spike — 赤
    'N': '#ff6b35',      # Nucleocapsid — オレンジ
    'M': '#ffc300',      # Membrane — 黄
    'E': '#ff85a1',      # Envelope — ピンク
    # ORFタンパク質 (寒色系)
    'ORF3a': '#00b4d8', 'ORF6': '#0096c7', 'ORF7a': '#0077b6',
    'ORF7b': '#023e8a', 'ORF8': '#48cae4', 'ORF10': '#90e0ef',
    # NSPタンパク質 (緑系)
    'nsp1': '#2d6a4f', 'nsp2': '#40916c', 'nsp3': '#52b788',
    'nsp4': '#74c69d', 'nsp5': '#95d5b2', 'nsp6': '#b7e4c7',
    'nsp7': '#d8f3dc', 'nsp8': '#1b4332', 'nsp9': '#2d6a4f',
    'nsp10': '#40916c', 'nsp12': '#52b788', 'nsp13': '#74c69d',
    'nsp14': '#95d5b2', 'nsp15': '#b7e4c7', 'nsp16': '#d8f3dc',
}
DEFAULT_REGION_COLOR = '#555555'  # non_coding

def get_region_color(region: str) -> str:
    return REGION_COLOR_MAP.get(region, DEFAULT_REGION_COLOR)


# ===== 1. 塩基埋め込み可視化 =====

def build_network(sim_matrix, labels, threshold=-999):
    G = nx.Graph()
    n = len(labels)
    for i in range(n):
        G.add_node(labels[i])
    for i in range(n):
        for j in range(i + 1, n):
            sim = sim_matrix[i, j]
            if sim > threshold:
                G.add_edge(labels[i], labels[j], weight=float(sim))
    return G


def plot_base_network(G, title, output_path, figsize=(10, 8)):
    fig, ax = plt.subplots(1, 1, figsize=figsize, facecolor='#1a1a2e')
    ax.set_facecolor('#1a1a2e')
    pos = nx.spring_layout(G, k=2.5, iterations=100, seed=42)
    edges = G.edges(data=True)
    edge_weights = [d['weight'] for _, _, d in edges]
    if not edge_weights:
        print("エッジがありません。")
        return
    min_w, max_w = min(edge_weights), max(edge_weights)
    cmap = plt.get_cmap('RdYlBu_r')
    norm = Normalize(vmin=min_w, vmax=max_w)
    edge_colors = [cmap(norm(w)) for w in edge_weights]
    edge_widths = [1.0 + 4.0 * (w - min_w) / (max_w - min_w + 1e-8) for w in edge_weights]
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_colors, width=edge_widths, alpha=0.7)
    node_colors = ['#e94560' if n in ('A', 'G') else '#0f3460' if n in ('T', 'C') else '#533483'
                   for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=2000, node_color=node_colors,
                           edgecolors='white', linewidths=2.0, alpha=0.95)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=16, font_weight='bold',
                            font_color='white', font_family='monospace')
    edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels, ax=ax, font_size=8,
                                 font_color='#aaaacc', font_family='monospace',
                                 bbox=dict(boxstyle='round,pad=0.1', facecolor='#1a1a2e',
                                           edgecolor='none', alpha=0.8))
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label('Cosine Similarity', color='white', fontsize=12)
    cbar.ax.tick_params(colors='white')
    legend_patches = [
        mpatches.Patch(color='#e94560', label='Purine (A, G)'),
        mpatches.Patch(color='#0f3460', label='Pyrimidine (T, C)'),
        mpatches.Patch(color='#533483', label='Other (N, n)'),
    ]
    ax.legend(handles=legend_patches, loc='upper left', fontsize=10,
              facecolor='#16213e', edgecolor='#e94560', labelcolor='white')
    ax.set_title(title, fontsize=16, fontweight='bold', color='white', pad=20)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    print(f"ネットワークグラフを保存: {output_path}")


def plot_similarity_heatmap(sim_matrix, labels, title, output_path, figsize=(8, 6)):
    fig, ax = plt.subplots(1, 1, figsize=figsize, facecolor='#1a1a2e')
    ax.set_facecolor('#1a1a2e')
    im = ax.imshow(sim_matrix, cmap='RdYlBu_r', aspect='auto', vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=12, color='white', fontweight='bold')
    ax.set_yticklabels(labels, fontsize=12, color='white', fontweight='bold')
    for i in range(len(labels)):
        for j in range(len(labels)):
            tc = 'white' if abs(sim_matrix[i, j]) > 0.5 else 'black'
            ax.text(j, i, f"{sim_matrix[i, j]:.2f}", ha='center', va='center',
                    fontsize=10, color=tc, fontweight='bold')
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Cosine Similarity', color='white', fontsize=12)
    cbar.ax.tick_params(colors='white')
    ax.set_title(title, fontsize=14, fontweight='bold', color='white', pad=15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    print(f"ヒートマップを保存: {output_path}")


# ===== 2. 位置埋め込み可視化: パターン1 — 閾値ネットワーク =====

def plot_position_threshold_network(weights, positions, pos_to_region,
                                    title, output_path, threshold=None,
                                    max_nodes=500, figsize=(16, 12)):
    """高類似度ペアのみのネットワークグラフ。thresholdがNoneなら自動設定。"""
    # サンプリングで類似度分布を表示
    n = len(weights)
    np.random.seed(42)
    sample_size = min(2000, n)
    sample_idx = np.random.choice(n, sample_size, replace=False)
    sample_sim = cosine_similarity(weights[sample_idx])
    upper_vals = sample_sim[np.triu_indices(sample_size, k=1)]
    print(f"  類似度分布 (サンプル{sample_size}点):")
    print(f"    Min={upper_vals.min():.4f}, Max={upper_vals.max():.4f}, "
          f"Mean={upper_vals.mean():.4f}, Std={upper_vals.std():.4f}")
    for pct in [99.0, 99.5, 99.9]:
        val = np.percentile(upper_vals, pct)
        print(f"    {pct}%ile: {val:.4f}")

    # 閾値の自動設定: 上位0.1%パーセンタイル
    if threshold is None:
        threshold = float(np.percentile(upper_vals, 99.9))
        print(f"  → 自動閾値 (99.9%ile): {threshold:.4f}")
    else:
        print(f"  → 指定閾値: {threshold:.4f}")

    sim_matrix = compute_cosine_similarity_matrix(weights)
    np.fill_diagonal(sim_matrix, 0)

    # 各ノードの最大類似度を取得し、上位ノードに絞る
    max_sims = sim_matrix.max(axis=1)
    top_indices = np.argsort(max_sims)[-max_nodes:]

    G = nx.Graph()
    for idx in top_indices:
        p = positions[idx]
        region = pos_to_region.get(p, 'unknown')
        G.add_node(p, region=region)

    top_set = set(top_indices)
    for i in top_indices:
        for j in top_indices:
            if i < j and sim_matrix[i, j] > threshold:
                G.add_edge(positions[i], positions[j], weight=float(sim_matrix[i, j]))

    # 孤立ノード除去
    isolates = list(nx.isolates(G))
    G.remove_nodes_from(isolates)

    if len(G.nodes()) == 0:
        print(f"  閾値 {threshold} ではエッジが見つかりません。閾値を下げてください。")
        return

    print(f"  ノード数: {len(G.nodes())}, エッジ数: {len(G.edges())}")

    fig, ax = plt.subplots(1, 1, figsize=figsize, facecolor='#1a1a2e')
    ax.set_facecolor('#1a1a2e')

    pos_layout = nx.spring_layout(G, k=1.5, iterations=80, seed=42)

    # ノードカラー
    node_colors = [get_region_color(G.nodes[n].get('region', ''))
                   for n in G.nodes()]

    edges = G.edges(data=True)
    edge_weights = [d['weight'] for _, _, d in edges]
    if edge_weights:
        cmap = plt.get_cmap('RdYlBu_r')
        norm = Normalize(vmin=min(edge_weights), vmax=max(edge_weights))
        edge_colors = [cmap(norm(w)) for w in edge_weights]
    else:
        edge_colors = ['gray']

    nx.draw_networkx_edges(G, pos_layout, ax=ax, edge_color=edge_colors, width=0.5, alpha=0.4)
    nx.draw_networkx_nodes(G, pos_layout, ax=ax, node_size=40, node_color=node_colors,
                           edgecolors='none', alpha=0.8)

    # 凡例（主要領域のみ）
    shown_regions = set()
    for n in G.nodes():
        shown_regions.add(G.nodes[n].get('region', 'unknown'))
    legend_patches = []
    for region in sorted(shown_regions):
        if region.startswith('non_coding'):
            continue
        legend_patches.append(mpatches.Patch(color=get_region_color(region), label=region))
    if any(r.startswith('non_coding') for r in shown_regions):
        legend_patches.append(mpatches.Patch(color=DEFAULT_REGION_COLOR, label='non_coding'))
    ax.legend(handles=legend_patches, loc='upper left', fontsize=8, ncol=2,
              facecolor='#16213e', edgecolor='#444', labelcolor='white')

    ax.set_title(title, fontsize=14, fontweight='bold', color='white', pad=20)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    print(f"  閾値ネットワークを保存: {output_path}")


# ===== 3. 位置埋め込み可視化: パターン2 — t-SNE 散布図 =====

def plot_position_tsne(weights, positions, pos_to_region,
                       title, output_path, perplexity=50, figsize=(16, 10)):
    """t-SNEで2D散布図を描画。遺伝子領域で色分け。"""
    print(f"  t-SNE 実行中 (perplexity={perplexity}, {len(weights)} points)...")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42,
                n_iter=1000, learning_rate='auto', init='pca')
    coords = tsne.fit_transform(weights)

    regions = [pos_to_region.get(p, 'unknown') for p in positions]

    # 領域をグループ化
    region_groups = {}
    for i, region in enumerate(regions):
        # non_coding をまとめる
        key = 'non_coding' if region.startswith('non_coding') else region
        if key not in region_groups:
            region_groups[key] = []
        region_groups[key].append(i)

    fig, ax = plt.subplots(1, 1, figsize=figsize, facecolor='#1a1a2e')
    ax.set_facecolor('#1a1a2e')

    # 背景（non_coding）を先に描画
    if 'non_coding' in region_groups:
        idx = region_groups['non_coding']
        ax.scatter(coords[idx, 0], coords[idx, 1], c=DEFAULT_REGION_COLOR,
                   s=3, alpha=0.15, label='non_coding', rasterized=True)

    # 主要領域を描画
    for region in sorted(region_groups.keys()):
        if region == 'non_coding':
            continue
        idx = region_groups[region]
        color = get_region_color(region)
        ax.scatter(coords[idx, 0], coords[idx, 1], c=color,
                   s=8, alpha=0.6, label=region, rasterized=True)

    ax.legend(fontsize=7, ncol=3, loc='upper right',
              facecolor='#16213e', edgecolor='#444', labelcolor='white',
              markerscale=3)

    ax.set_title(title, fontsize=14, fontweight='bold', color='white', pad=15)
    ax.tick_params(colors='#888888')
    for spine in ax.spines.values():
        spine.set_color('#333333')
    ax.set_xlabel('t-SNE 1', color='#aaaaaa', fontsize=10)
    ax.set_ylabel('t-SNE 2', color='#aaaaaa', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    print(f"  t-SNE散布図を保存: {output_path}")


# ===== 4. 位置埋め込み可視化: パターン3 — クラスタ間ネットワーク =====

def plot_position_cluster_network(weights, positions, pos_to_region,
                                  title, output_path, n_clusters=20, figsize=(14, 10)):
    """k-meansクラスタリング→クラスタ重心間のコサイン類似度ネットワーク"""
    print(f"  k-means クラスタリング中 (k={n_clusters})...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(weights)
    centroids = kmeans.cluster_centers_

    # クラスタごとの遺伝子領域の構成を解析
    cluster_info = {}
    for c in range(n_clusters):
        mask = cluster_labels == c
        cluster_positions = [positions[i] for i in range(len(positions)) if mask[i]]
        cluster_regions = [pos_to_region.get(p, 'unknown') for p in cluster_positions]

        # 最頻の遺伝子領域
        region_counts = {}
        for r in cluster_regions:
            key = 'non_coding' if r.startswith('non_coding') else r
            region_counts[key] = region_counts.get(key, 0) + 1
        dominant = max(region_counts, key=region_counts.get)
        dominant_pct = region_counts[dominant] / len(cluster_regions) * 100

        cluster_info[c] = {
            'size': int(mask.sum()),
            'dominant_region': dominant,
            'dominant_pct': dominant_pct,
            'region_counts': region_counts,
        }

    # クラスタ間コサイン類似度
    sim_matrix = compute_cosine_similarity_matrix(centroids)

    G = nx.Graph()
    for c in range(n_clusters):
        info = cluster_info[c]
        label = f"C{c}\n{info['dominant_region']}\n({info['size']})"
        G.add_node(c, label=label, region=info['dominant_region'], size=info['size'])

    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            sim = sim_matrix[i, j]
            if sim > 0.3:  # 低すぎるエッジは非表示
                G.add_edge(i, j, weight=float(sim))

    fig, ax = plt.subplots(1, 1, figsize=figsize, facecolor='#1a1a2e')
    ax.set_facecolor('#1a1a2e')

    pos_layout = nx.spring_layout(G, k=3.0, iterations=100, seed=42)

    # ノードサイズはクラスタサイズに比例
    sizes = [cluster_info[n]['size'] for n in G.nodes()]
    max_size = max(sizes) if sizes else 1
    node_sizes = [300 + 3000 * (s / max_size) for s in sizes]
    node_colors = [get_region_color(cluster_info[n]['dominant_region']) for n in G.nodes()]

    # エッジ
    edges = G.edges(data=True)
    edge_weights = [d['weight'] for _, _, d in edges]
    if edge_weights:
        cmap = plt.get_cmap('RdYlBu_r')
        norm = Normalize(vmin=min(edge_weights), vmax=max(edge_weights))
        edge_colors = [cmap(norm(w)) for w in edge_weights]
        edge_widths = [0.5 + 3.0 * (w - min(edge_weights)) / (max(edge_weights) - min(edge_weights) + 1e-8)
                       for w in edge_weights]
    else:
        edge_colors, edge_widths = ['gray'], [1.0]

    nx.draw_networkx_edges(G, pos_layout, ax=ax, edge_color=edge_colors,
                           width=edge_widths, alpha=0.5)
    nx.draw_networkx_nodes(G, pos_layout, ax=ax, node_size=node_sizes,
                           node_color=node_colors, edgecolors='white',
                           linewidths=1.5, alpha=0.9)

    labels_dict = {n: G.nodes[n]['label'] for n in G.nodes()}
    nx.draw_networkx_labels(G, pos_layout, labels_dict, ax=ax,
                            font_size=7, font_color='white', font_weight='bold')

    # エッジラベル
    edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos_layout, edge_labels, ax=ax,
                                 font_size=6, font_color='#aaaacc',
                                 bbox=dict(boxstyle='round,pad=0.1',
                                           facecolor='#1a1a2e', edgecolor='none', alpha=0.8))

    # カラーバー
    if edge_weights:
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.5, pad=0.02)
        cbar.set_label('Cosine Similarity (centroids)', color='white', fontsize=10)
        cbar.ax.tick_params(colors='white')

    ax.set_title(title, fontsize=14, fontweight='bold', color='white', pad=20)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    print(f"  クラスタ間ネットワークを保存: {output_path}")

    # クラスタ情報をテキスト出力
    print(f"\n  === クラスタ一覧 ===")
    for c in range(n_clusters):
        info = cluster_info[c]
        top3 = sorted(info['region_counts'].items(), key=lambda x: -x[1])[:3]
        top3_str = ", ".join(f"{r}:{cnt}" for r, cnt in top3)
        print(f"  C{c}: {info['size']}pos, 主要={info['dominant_region']}({info['dominant_pct']:.0f}%), 上位3={top3_str}")


# ===== 5. 主要変異位置ネットワーク =====

def load_mutation_frequencies(freq_csv: str) -> pd.DataFrame:
    """変異頻度CSVを読み込み、合計頻度列を追加して返す"""
    df = pd.read_csv(freq_csv)
    mut_cols = [c for c in df.columns if c != 'position']
    df['total_freq'] = df[mut_cols].sum(axis=1)
    return df


def plot_major_mutation_network(pos_weights_all, pos_to_region, freq_df, codon_df,
                                title, output_path, top_n=50, figsize=(16, 14)):
    """
    高頻度変異位置のみを抽出し、位置埋め込みのコサイン類似度ネットワークを描画。
    ノードラベル: 位置番号 + 遺伝子名
    ノードサイズ: 変異頻度に比例
    ノードカラー: 遺伝子領域
    エッジ: コサイン類似度
    """
    # 上位N位置を取得
    top_positions = freq_df.nlargest(top_n, 'total_freq')
    positions = top_positions['position'].tolist()
    freqs = top_positions['total_freq'].tolist()

    print(f"  上位{top_n}位置の変異頻度:")
    print(f"    最大: pos={positions[0]}, freq={freqs[0]:.0f}")
    print(f"    最小: pos={positions[-1]}, freq={freqs[-1]:.0f}")

    # 変異タイプ情報（最も頻度の高い変異パターンを取得）
    mut_cols = [c for c in freq_df.columns if c not in ('position', 'total_freq')]

    # 埋め込みを取得 (pos_weights_all はindex=positionで直接アクセス可)
    embeddings = []
    valid_positions = []
    valid_freqs = []
    for p, f in zip(positions, freqs):
        if p < pos_weights_all.shape[0]:
            embeddings.append(pos_weights_all[p])
            valid_positions.append(p)
            valid_freqs.append(f)
    embeddings = np.array(embeddings)

    # コサイン類似度
    sim_matrix = compute_cosine_similarity_matrix(embeddings)

    # ラベル構築
    labels = []
    for p in valid_positions:
        region = pos_to_region.get(p, '?')
        # codon_dfからタンパク質位置も取得
        row = codon_df[codon_df['base_pos'] == p]
        if len(row) > 0:
            protein = str(row.iloc[0]['protein'])
            protein_pos = str(row.iloc[0]['protein_pos'])
            if protein.startswith('non_coding'):
                label = f"{p}\n(nc)"
            else:
                label = f"{p}\n{protein}:{protein_pos}"
        else:
            label = str(p)
        labels.append(label)

    # ネットワーク構築
    # 閾値: 上位10%のエッジのみ表示
    upper_tri = sim_matrix[np.triu_indices(len(valid_positions), k=1)]
    if len(upper_tri) > 0:
        edge_threshold = float(np.percentile(upper_tri, 80))
    else:
        edge_threshold = 0.0
    print(f"  エッジ閾値 (80%ile): {edge_threshold:.4f}")

    G = nx.Graph()
    for i, (p, lab) in enumerate(zip(valid_positions, labels)):
        region = pos_to_region.get(p, 'unknown')
        region_key = 'non_coding' if region.startswith('non_coding') else region
        # 最頻変異パターンを取得
        freq_row = freq_df[freq_df['position'] == p]
        if len(freq_row) > 0:
            mut_values = freq_row[mut_cols].values[0]
            dominant_mut = mut_cols[np.argmax(mut_values)]
        else:
            dominant_mut = ''
        G.add_node(i, label=lab, region=region_key, freq=valid_freqs[i],
                   position=p, dominant_mut=dominant_mut)

    for i in range(len(valid_positions)):
        for j in range(i + 1, len(valid_positions)):
            sim = sim_matrix[i, j]
            if sim > edge_threshold:
                G.add_edge(i, j, weight=float(sim))

    print(f"  ノード数: {len(G.nodes())}, エッジ数: {len(G.edges())}")

    # 描画
    fig, ax = plt.subplots(1, 1, figsize=figsize, facecolor='#1a1a2e')
    ax.set_facecolor('#1a1a2e')

    pos_layout = nx.spring_layout(G, k=2.5, iterations=120, seed=42)

    # ノードサイズ: 頻度に比例
    max_freq = max(valid_freqs)
    node_sizes = [200 + 2500 * (G.nodes[n]['freq'] / max_freq) for n in G.nodes()]
    node_colors = [get_region_color(G.nodes[n]['region']) for n in G.nodes()]

    # エッジ
    edges = G.edges(data=True)
    edge_weights = [d['weight'] for _, _, d in edges]
    if edge_weights:
        cmap = plt.get_cmap('RdYlBu_r')
        norm = Normalize(vmin=min(edge_weights), vmax=max(edge_weights))
        edge_colors = [cmap(norm(w)) for w in edge_weights]
        edge_widths = [0.3 + 2.5 * (w - min(edge_weights)) / (max(edge_weights) - min(edge_weights) + 1e-8)
                       for w in edge_weights]
    else:
        edge_colors, edge_widths = ['gray'], [1.0]

    nx.draw_networkx_edges(G, pos_layout, ax=ax, edge_color=edge_colors,
                           width=edge_widths, alpha=0.4)
    nx.draw_networkx_nodes(G, pos_layout, ax=ax, node_size=node_sizes,
                           node_color=node_colors, edgecolors='white',
                           linewidths=1.0, alpha=0.9)

    # ラベル
    labels_dict = {n: G.nodes[n]['label'] for n in G.nodes()}
    nx.draw_networkx_labels(G, pos_layout, labels_dict, ax=ax,
                            font_size=6, font_color='white', font_weight='bold')

    # カラーバー
    if edge_weights:
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.4, pad=0.02)
        cbar.set_label('Cosine Similarity', color='white', fontsize=10)
        cbar.ax.tick_params(colors='white')

    # 凡例
    shown_regions = set(G.nodes[n]['region'] for n in G.nodes())
    legend_patches = []
    for region in sorted(shown_regions):
        if region == 'non_coding':
            legend_patches.append(mpatches.Patch(color=DEFAULT_REGION_COLOR, label='non_coding'))
        else:
            legend_patches.append(mpatches.Patch(color=get_region_color(region), label=region))
    ax.legend(handles=legend_patches, loc='upper left', fontsize=8, ncol=2,
              facecolor='#16213e', edgecolor='#444', labelcolor='white')

    ax.set_title(title, fontsize=14, fontweight='bold', color='white', pad=20)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    print(f"  主要変異ネットワークを保存: {output_path}")

    # 上位変異のテーブル出力
    print(f"\n  === 上位{top_n}変異位置 ===")
    print(f"  {'Pos':>6}  {'Region':<12} {'Prot.Pos':>8}  {'Freq':>8}  {'DominantMut':<8}")
    for n in sorted(G.nodes(), key=lambda x: -G.nodes[x]['freq']):
        nd = G.nodes[n]
        print(f"  {nd['position']:>6}  {nd['region']:<12} "
              f"{nd['label'].split(chr(10))[-1] if chr(10) in nd['label'] else '':>8}  "
              f"{nd['freq']:>8.0f}  {nd['dominant_mut']:<8}")


# ===== メイン =====

def main():
    parser = argparse.ArgumentParser(description='語彙ネットワーク描画')
    parser.add_argument('--model_path', type=str, required=True,
                        help='学習済みモデルのパス (.pth)')
    parser.add_argument('--output_dir', type=str, default='outputs/vocab_network',
                        help='出力先ディレクトリ (デフォルト: outputs/vocab_network)')
    parser.add_argument('--codon_csv', type=str, default='meta_data/codon_mutation4.csv',
                        help='コドン変異CSV (位置→遺伝子領域マッピング用)')
    parser.add_argument('--freq_csv', type=str,
                        default='outputs/table_heatmap/251031/table_set/table_set.csv',
                        help='変異頻度CSV')
    parser.add_argument('--threshold', type=float, default=-999,
                        help='塩基ネットワークのエッジ閾値')
    parser.add_argument('--pos_threshold', type=float, default=None,
                        help='位置ネットワークのエッジ閾値 (デフォルト: 自動=上位0.1%%)')
    parser.add_argument('--n_clusters', type=int, default=20,
                        help='クラスタ数 (デフォルト: 20)')
    parser.add_argument('--top_n', type=int, default=50,
                        help='主要変異ネットワークの表示位置数 (デフォルト: 50)')
    parser.add_argument('--skip_base', action='store_true',
                        help='塩基埋め込み可視化をスキップ')
    parser.add_argument('--skip_position', action='store_true',
                        help='位置埋め込み (全30K) 可視化をスキップ')
    parser.add_argument('--skip_major', action='store_true',
                        help='主要変異ネットワークをスキップ')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    version = extract_model_version(args.model_path)
    print(f"モデルバージョン: {version}")

    cfg = load_config_from_model_dir(args.model_path)
    print(f"モデルをロード中: {args.model_path}")
    model = load_model(args.model_path, cfg)

    # セクション間で共有する変数
    pos_weights = None
    pos_to_region = None

    # ========================================
    # A. 塩基埋め込み (base_embed)
    # ========================================
    if not args.skip_base:
        print(f"\n{'='*60}")
        print(f"塩基埋め込み (base_embed) の可視化")
        print(f"{'='*60}")

        base_labels_map = {v: k for k, v in cfg.BASE_VOCABS.items() if k != 'PAD'}
        base_indices = sorted(base_labels_map.keys())
        base_labels = [base_labels_map[i] for i in base_indices]

        base_weights = extract_embedding_weights(model, 'base_embed')
        base_weights_no_pad = base_weights[base_indices]
        sim_matrix = compute_cosine_similarity_matrix(base_weights_no_pad)

        print(f"語彙: {base_labels}, 埋め込み次元: {base_weights.shape[1]}")
        header = "\t".join(f"{l:>6}" for l in base_labels)
        print(f"       {header}")
        for i, label_i in enumerate(base_labels):
            row = "\t".join(f"{sim_matrix[i, j]:+.3f}" for j in range(len(base_labels)))
            print(f"  {label_i:>4}: {row}")

        G = build_network(sim_matrix, base_labels, threshold=args.threshold)
        plot_base_network(G, f'Base Nucleotide Embedding — Cosine Similarity Network\n({version})',
                          os.path.join(args.output_dir, f'vocab_network_base_{version}.png'))
        plot_similarity_heatmap(sim_matrix, base_labels,
                                f'Base Nucleotide Embedding — Cosine Similarity ({version})',
                                os.path.join(args.output_dir, f'vocab_heatmap_base_{version}.png'))

    # ========================================
    # B. 位置埋め込み (pos_embed)
    # ========================================
    if not args.skip_position:
        print(f"\n{'='*60}")
        print(f"位置埋め込み (pos_embed) の可視化")
        print(f"{'='*60}")

        pos_weights = extract_embedding_weights(model, 'pos_embed')
        print(f"語彙サイズ: {pos_weights.shape[0]}, 埋め込み次元: {pos_weights.shape[1]}")

        # PAD(index=0)を除外、実際のゲノム位置(1~30000)のみ
        max_genome_pos = min(pos_weights.shape[0] - 1, 30000)
        pos_indices = list(range(1, max_genome_pos + 1))
        pos_weights_no_pad = pos_weights[pos_indices]
        print(f"有効な位置数: {len(pos_indices)} (PAD除外)")

        # 遺伝子領域マッピング
        pos_to_region = load_position_to_region(args.codon_csv)
        print(f"領域マッピング: {len(pos_to_region)} 位置をロード")

        # パターン1: 閾値フィルタ付きネットワーク
        thresh_label = f'{args.pos_threshold}' if args.pos_threshold is not None else 'auto'
        print(f"\n--- パターン1: 閾値フィルタ付きネットワーク (threshold={thresh_label}) ---")
        plot_position_threshold_network(
            pos_weights_no_pad, pos_indices, pos_to_region,
            f'Position Embedding — Threshold Network\n({version})',
            os.path.join(args.output_dir, f'pos_network_threshold_{version}.png'),
            threshold=args.pos_threshold,
        )

        # パターン2: t-SNE 散布図
        print(f"\n--- パターン2: t-SNE 散布図 ---")
        plot_position_tsne(
            pos_weights_no_pad, pos_indices, pos_to_region,
            f'Position Embedding — t-SNE (colored by gene region)\n({version})',
            os.path.join(args.output_dir, f'pos_tsne_{version}.png'),
        )

        # パターン3: クラスタ間ネットワーク
        print(f"\n--- パターン3: クラスタ間ネットワーク (k={args.n_clusters}) ---")
        plot_position_cluster_network(
            pos_weights_no_pad, pos_indices, pos_to_region,
            f'Position Embedding — Cluster Network (k={args.n_clusters})\n({version})',
            os.path.join(args.output_dir, f'pos_cluster_network_{version}.png'),
            n_clusters=args.n_clusters,
        )

    # ========================================
    # C. 主要変異位置ネットワーク
    # ========================================
    if not args.skip_major:
        print(f"\n{'='*60}")
        print(f"主要変異位置ネットワーク (top {args.top_n})")
        print(f"{'='*60}")

        # 位置埋め込み（まだロードしていない場合）
        if pos_weights is None:
            pos_weights = extract_embedding_weights(model, 'pos_embed')
        if pos_to_region is None:
            pos_to_region = load_position_to_region(args.codon_csv)

        freq_df = load_mutation_frequencies(args.freq_csv)
        codon_df = pd.read_csv(args.codon_csv, encoding='utf-8-sig')
        print(f"変異頻度CSV: {args.freq_csv}")
        print(f"全位置の合計頻度: {freq_df['total_freq'].sum():.0f}")

        plot_major_mutation_network(
            pos_weights, pos_to_region, freq_df, codon_df,
            f'Major Mutation Positions — Embedding Similarity Network (top {args.top_n})\n({version})',
            os.path.join(args.output_dir, f'pos_major_mutations_{version}.png'),
            top_n=args.top_n,
        )

    print(f"\n完了！ 出力先: {args.output_dir}")


if __name__ == '__main__':
    main()
