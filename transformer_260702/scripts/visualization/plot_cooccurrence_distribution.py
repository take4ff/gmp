# --- transformer_260702/scripts/plot_cooccurrence_distribution.py ---
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm

# プロジェクトルートを sys.path に追加
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from transformer_260702 import config
from transformer_260702.db.connection import get_db_path, connect_db
from transformer_260702.db.feature import import_mutation_paths


def plot_cooccurrence_distribution(output_dir=None):
    """入力パス内の最大共起数分布をターゲット共起数で色分けした積み上げ棒グラフを作成・保存する。"""
    usher_dir = "/mnt/ssd1/home3/aiba/usher_output"
    csv_path = os.path.join(project_root, config.SEQUENCES_CSV)
    
    if output_dir is None:
        output_dir = os.path.join(project_root, "outputs/co-occur")
        
    os.makedirs(output_dir, exist_ok=True)
    output_png = os.path.join(output_dir, "cooccurrence_distribution.png")

    print("[INFO] Loading Collection Dates from CSV...")
    if not os.path.exists(csv_path):
        print(f"[ERROR] CSV file not found: {csv_path}")
        return
        
    df_csv = pd.read_csv(csv_path, usecols=['Accession', 'Collection_Date'])
    df_csv['ym'] = df_csv['Collection_Date'].astype(str).str[:7]
    date_map = dict(zip(df_csv['Accession'], df_csv['ym']))
    del df_csv

    strains = sorted([f for f in os.listdir(usher_dir) if not f.startswith('.')])
    print(f"[INFO] Found {len(strains)} strains. Sampling paths...")

    # 集計用辞書: key = input_max_co, value = {'le5': count, 'gt5': count}
    dist_pre = defaultdict(lambda: {'le5': 0, 'gt5': 0})
    dist_post = defaultdict(lambda: {'le5': 0, 'gt5': 0})

    # 全株から最大50サンプルずつロード
    for strain in tqdm(strains, desc="Analyzing strains"):
        file_paths = import_mutation_paths(usher_dir, strain)
        for file_path in file_paths:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    f.readline()
                    lines = []
                    for _ in range(50):
                        l = f.readline()
                        if not l:
                            break
                        lines.append(l)
            except Exception:
                continue

            for line in lines:
                parts = line.strip().split('\t')
                if len(parts) < 3:
                    continue
                
                acc = parts[0]
                path_steps = parts[2].split('>')
                if len(path_steps) <= 1:
                    continue
                
                ym = date_map.get(acc, '')
                is_post_2022 = (ym >= '2022-01')

                # 入力パス最大共起数（最後を除く）
                input_steps = path_steps[:-1]
                input_max_co = max(len(step.split(',')) for step in input_steps) if input_steps else 0

                # ターゲット共起数（最後のタイムステップ）
                target_step = path_steps[-1]
                target_co = len(target_step.split(','))

                target_cat = 'le5' if target_co <= 5 else 'gt5'

                if is_post_2022:
                    dist_post[input_max_co][target_cat] += 1
                else:
                    dist_pre[input_max_co][target_cat] += 1

    # プロットデータの整形
    def make_plot_data(dist_dict, max_key):
        if max_key == 0:
            return [], [], []
        x = list(range(1, max_key + 1))
        y_le5 = [dist_dict[k]['le5'] for k in x]
        y_gt5 = [dist_dict[k]['gt5'] for k in x]
        return x, y_le5, y_gt5

    all_keys = list(dist_pre.keys()) + list(dist_post.keys())
    global_max_key = max(all_keys) if all_keys else 0

    x_pre, y_pre_le5, y_pre_gt5 = make_plot_data(dist_pre, global_max_key)
    x_post, y_post_le5, y_post_gt5 = make_plot_data(dist_post, global_max_key)

    # プロット作成
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # 2022年前
    ax = axes[0]
    ax.bar(x_pre, y_pre_le5, label=r'Target Co-occur $\leq$ 5', color='#2b5c8f', alpha=0.85)
    ax.bar(x_pre, y_pre_gt5, bottom=y_pre_le5, label=r'Target Co-occur $\geq$ 6', color='#d95f02', alpha=0.85)
    ax.set_title('Max Co-occurrence Distribution in Input Path (Before 2022)', fontsize=14)
    ax.set_xlabel('Max Co-occurrence', fontsize=11)
    ax.set_ylabel('Sample Count', fontsize=11)
    ax.set_xticks(x_pre)
    if global_max_key > 0:
        ax.set_xlim(0.5, global_max_key + 0.5)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=10)

    # 2022年以降
    ax = axes[1]
    ax.bar(x_post, y_post_le5, label=r'Target Co-occur $\leq$ 5', color='#2b5c8f', alpha=0.85)
    ax.bar(x_post, y_post_gt5, bottom=y_post_le5, label=r'Target Co-occur $\geq$ 6', color='#d95f02', alpha=0.85)
    ax.set_title('Max Co-occurrence Distribution in Input Path (2022 and After)', fontsize=14)
    ax.set_xlabel('Max Co-occurrence', fontsize=11)
    ax.set_ylabel('Sample Count', fontsize=11)
    ax.set_xticks(x_post)
    if global_max_key > 0:
        ax.set_xlim(0.5, global_max_key + 0.5)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    print(f"[INFO] Stacked bar plot saved to {output_png}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default=None, help='Directory to save output distribution plot')
    args = parser.parse_args()
    plot_cooccurrence_distribution(args.output_dir)
