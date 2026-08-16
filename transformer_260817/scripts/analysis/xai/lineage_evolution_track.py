# --- transformer_260817/scripts/analysis/lineage_evolution_track.py ---
"""上位系統の代表サンプル1件ずつについて、変異獲得の「タイムステップ×ゲノム位置」を可視化する。

各系統（Pango lineage, strain_name_usher）から代表サンプルを1件選び（その系統内で
path_lengthが中央値に最も近いもの＝典型例）、raw_pathを timestep 単位で展開して
(timestep, genome position, gene) の点をプロットする。同じ祖先を共有する系統は
序盤のtimestepで近い位置に点が重なり、その後どこで進化が分岐するかが視覚化できる。

`attention_by_month.py` とは異なり、モデルの attention 重みではなく実際の変異位置
（ground truth）を使う点に注意。

Usage:
  python -m transformer_260817.scripts.analysis.xai.lineage_evolution_track \\
      --top_n_lineages 5
"""

import argparse
import re

import pandas as pd
import matplotlib.pyplot as plt

from transformer_260817.db.connection import connect_db, get_db_path
from transformer_260817.scripts.analysis.xai import _xai_common as X

_MUT_RE = re.compile(r'^[ACGT-](\d+)[ACGT-]$')


def _parse_path_to_positions(raw_path):
    """raw_path ('>'=timestep区切り, ','=共起区切り) を [(timestep, position), ...] に展開する。"""
    out = []
    for t, node in enumerate(raw_path.split('>')):
        for mut in node.split(','):
            mut = mut.strip()
            if not mut:
                continue
            m = _MUT_RE.match(mut)
            if m:
                out.append((t, int(m.group(1))))
    return out


def pick_representative_samples(db_path, top_n):
    con = connect_db(db_path, read_only=True)

    lineages = con.execute("""
        SELECT st.strain_name_usher, COUNT(*) as n
        FROM samples s JOIN strains st ON s.strain_id = st.strain_id
        WHERE st.strain_name_usher IS NOT NULL AND st.strain_name_usher != ''
        GROUP BY st.strain_name_usher ORDER BY n DESC LIMIT ?
    """, [top_n]).fetchall()

    reps = []
    for lineage, n in lineages:
        median_len = con.execute("""
            SELECT median(s.path_length)
            FROM samples s JOIN strains st ON s.strain_id = st.strain_id
            WHERE st.strain_name_usher = ?
        """, [lineage]).fetchone()[0]

        row = con.execute("""
            SELECT s.sample_id, s.raw_path, s.path_length, s.collection_date
            FROM samples s JOIN strains st ON s.strain_id = st.strain_id
            WHERE st.strain_name_usher = ?
            ORDER BY ABS(s.path_length - ?) ASC
            LIMIT 1
        """, [lineage, median_len]).fetchone()

        sample_id, raw_path, path_length, collection_date = row
        reps.append({
            'lineage': lineage, 'n_samples': n, 'sample_id': sample_id,
            'raw_path': raw_path, 'path_length': path_length,
            'collection_date': collection_date, 'median_path_length_in_lineage': median_len,
        })

    con.close()
    return reps


def main():
    ap = argparse.ArgumentParser(description='Top-N lineage evolution: timestep x genome position')
    ap.add_argument('--top_n_lineages', type=int, default=5)
    ap.add_argument('--output_dir', type=str, default=None)
    args = ap.parse_args()

    db_path = get_db_path()
    pos2gene = X.load_position_gene_map()

    reps = pick_representative_samples(db_path, args.top_n_lineages)

    rows = []
    for rep in reps:
        for t, pos in _parse_path_to_positions(rep['raw_path']):
            gene = pos2gene.get(pos, ('unknown', '0'))[0]
            rows.append({'lineage': rep['lineage'], 'sample_id': rep['sample_id'],
                        'collection_date': rep['collection_date'], 'timestep': t,
                        'position': pos, 'gene': gene})
    df = pd.DataFrame(rows)

    out_dir = X.make_output_dir('lineage_evolution_track', args.output_dir)
    X.save_csv(df, out_dir, 'lineage_evolution_points.csv')
    X.save_csv(pd.DataFrame(reps).drop(columns=['raw_path']), out_dir, 'representative_samples.csv')

    fig, ax = plt.subplots(figsize=(12, 8))
    colors = plt.cm.tab10.colors
    for i, rep in enumerate(reps):
        sub = df[df['lineage'] == rep['lineage']]
        ax.scatter(sub['timestep'], sub['position'], s=14, color=colors[i % len(colors)],
                  label=f"{rep['lineage']} (n={rep['n_samples']:,}, path_len={rep['path_length']}, "
                        f"{rep['collection_date']})", alpha=0.8)

    # 主要遺伝子境界の目安線（Sのみ強調、他は軽く）
    gene_starts = {'S': 21563, 'N': 28274, 'M': 26523, 'E': 26245, 'ORF8': 27894}
    for gene, pos in gene_starts.items():
        ax.axhline(pos, color='gray', linewidth=0.5, linestyle=':')
        ax.text(ax.get_xlim()[1] if ax.get_xlim()[1] > 0 else 40, pos, f' {gene}',
              fontsize=8, color='gray', va='center')

    ax.set_xlabel('Timestep (order along this sample\'s own mutation path)')
    ax.set_ylabel('Genome position (nt)')
    ax.set_title(f'Top-{args.top_n_lineages} lineages: representative-sample evolution track')
    ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.01, 1.0))
    plt.tight_layout()

    X.save_fig(fig, out_dir, 'lineage_evolution_track.png')
    print(f"[INFO] Saved: {out_dir}/lineage_evolution_track.png")
    print(f"[INFO] Saved: {out_dir}/lineage_evolution_points.csv")
    for rep in reps:
        print(f"  {rep['lineage']}: sample_id={rep['sample_id']}, n_samples={rep['n_samples']:,}, "
              f"path_length={rep['path_length']} (lineage median={rep['median_path_length_in_lineage']}), "
              f"collection_date={rep['collection_date']}")


if __name__ == '__main__':
    main()
