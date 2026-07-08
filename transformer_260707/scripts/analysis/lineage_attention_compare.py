# --- transformer_260707/scripts/analysis/lineage_attention_compare.py ---
"""上位系統の代表サンプルについて、実際の変異位置(ground truth)とモデルの
Co-occurrence Attention重みを同一の入力スロット単位で比較する。

lineage_evolution_track.py で選んだ代表サンプル（系統ごとに path_length が中央値に
最も近い1件）を、その採取日が属する walk-forward フォールドのチェックポイントで
1件ずつ評価し、x_cat 由来のゲノム位置と Co-occurrence Attention の重みを対にして
出力する。timestepのズレを避けるため、raw_pathの再パースはせず、モデルに実際に
入力されたテンソル(x_cat)からゲノム位置を直接読み出す（ground truthとモデル入力が
常に同一スロットで一致することを保証するため）。

Usage:
  python -m transformer_260707.scripts.analysis.lineage_attention_compare \\
      --walk_forward_dir outputs/transformer_260702/results/walk_forward/20260706_223816 \\
      --top_n_lineages 5 [--force_cpu]
"""

import argparse

import pandas as pd
import torch
import matplotlib.pyplot as plt

from transformer_260707 import config
from transformer_260707.db.connection import get_db_path
from transformer_260707.db.dataset import DBIterableDataset, _NUM_MASK_KEYS_BASE
from transformer_260707.scripts.analysis import _xai_common as X
from transformer_260707.scripts.analysis.attention_by_month import (
    _attach_attention_capture, POS_COL, BASE_BEFORE_COL)
from transformer_260707.scripts.analysis.lineage_evolution_track import pick_representative_samples


def _build_single_sample_batch(db_path, sample_id):
    """DBIterableDataset.__init__ の高コストな全件クエリを避け、指定した1 sample_id だけを
    読み込む軽量インスタンスを作る（__iter__ が使う属性のみ手動設定）。
    """
    ds = object.__new__(DBIterableDataset)
    ds.db_path = db_path
    ds.sample_ids = [sample_id]
    ds._length = 1
    ds.chunk_size = 1
    ds.shuffle = False
    ds.strain_to_strength = None
    ds._num_mask_keys = list(_NUM_MASK_KEYS_BASE)
    if getattr(config, 'USE_LINEAGE_GROWTH_FEATURES', False):
        from transformer_260707.db.dataset import _NUM_MASK_KEYS_GROWTH
        ds._num_mask_keys = ds._num_mask_keys + _NUM_MASK_KEYS_GROWTH

    items = list(ds)
    if not items:
        return None
    item = items[0]
    return item['x_cat'].unsqueeze(0), item['x_num'].unsqueeze(0), item['mask'].unsqueeze(0)


def _find_fold_for_date(fold_windows, date_str):
    """test窓 [split_date, split_end) に date_str が含まれるフォールドIDを返す。
    どの test 窓にも含まれない（fold1 の test 開始日より前）場合は、最も古い
    フォールド(fold1、その日付を学習データとして含む唯一のモデル)にフォールバックする。
    """
    for fold_id, (_train_start, split_date, split_end) in sorted(fold_windows.items()):
        if date_str >= split_date and (split_end is None or date_str < split_end):
            return fold_id, True
    earliest_fold = min(fold_windows.keys())
    return earliest_fold, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--walk_forward_dir', required=True)
    ap.add_argument('--top_n_lineages', type=int, default=5)
    ap.add_argument('--force_cpu', action='store_true')
    ap.add_argument('--output_dir', default=None)
    args = ap.parse_args()

    device = torch.device('cpu' if args.force_cpu or not torch.cuda.is_available() else config.DEVICE)
    db_path = get_db_path()
    pos2gene = X.load_position_gene_map()
    fold_windows = X.get_fold_windows()
    fold_ckpts = X.discover_fold_checkpoints(args.walk_forward_dir)

    reps = pick_representative_samples(db_path, args.top_n_lineages)

    rows = []
    for rep in reps:
        fold_id, is_test_window = _find_fold_for_date(fold_windows, rep['collection_date'])
        if fold_id not in fold_ckpts:
            print(f"[SKIP] {rep['lineage']}: fold{fold_id} のチェックポイントが見つかりません")
            continue
        note = '' if is_test_window else '（test窓より前のためfold1にフォールバック、学習データでの評価）'
        print(f"[INFO] {rep['lineage']} ({rep['collection_date']}) -> fold{fold_id}{note}")

        model = X.load_fold_model(fold_ckpts[fold_id], device)  # config_snapshot反映込み

        batch = _build_single_sample_batch(db_path, rep['sample_id'])
        if batch is None:
            print(f"[SKIP] {rep['lineage']}: sample_id={rep['sample_id']} の特徴量が見つかりません")
            continue
        x_cat, x_num, mask = [t.to(device) for t in batch]

        captured = _attach_attention_capture(model)
        with torch.no_grad():
            model(x_cat, x_num, src_key_padding_mask=mask)
        w = captured['w'][0].float().cpu().numpy()          # [T, C]
        pos = x_cat[0, ..., POS_COL].cpu().numpy()           # [T, C]
        is_pad = (x_cat[0, ..., BASE_BEFORE_COL] == 0).cpu().numpy()

        T, C = w.shape
        n_valid = 0
        for t in range(T):
            for c in range(C):
                if is_pad[t, c]:
                    continue
                n_valid += 1
                p = int(pos[t, c])
                rows.append({
                    'lineage': rep['lineage'], 'sample_id': rep['sample_id'], 'fold': fold_id,
                    'is_test_window': is_test_window, 'timestep_slot': t, 'position': p,
                    'gene': pos2gene.get(p, ('unknown', '0'))[0], 'attn_weight': float(w[t, c]),
                })
        print(f"  -> {n_valid} valid slots")

    if not rows:
        raise SystemExit("[ERROR] 有効なサンプルがありませんでした。")

    df = pd.DataFrame(rows)
    out_dir = X.make_output_dir('lineage_attention_compare', args.output_dir)
    X.save_csv(df, out_dir, 'lineage_attention_compare.csv')

    fig, axes = plt.subplots(1, 2, figsize=(18, 8), sharey=True)
    colors = plt.cm.tab10.colors
    lineages = list(df['lineage'].unique())

    ax = axes[0]
    for i, lin in enumerate(lineages):
        sub = df[df['lineage'] == lin]
        ax.scatter(sub['timestep_slot'], sub['position'], s=16, color=colors[i % len(colors)],
                  label=lin, alpha=0.85)
    ax.set_title('Ground truth mutation positions (from model input x_cat)')
    ax.set_xlabel('Input slot (timestep, right-aligned/padded)')
    ax.set_ylabel('Genome position (nt)')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    ax = axes[1]
    sc = ax.scatter(df['timestep_slot'], df['position'], c=df['attn_weight'], cmap='viridis', s=36)
    ax.set_title('Co-occurrence Attention weight (same slots)')
    ax.set_xlabel('Input slot (timestep, right-aligned/padded)')
    ax.grid(alpha=0.2)
    plt.colorbar(sc, ax=ax, label='attention weight')

    plt.tight_layout()
    X.save_fig(fig, out_dir, 'lineage_attention_compare.png')


if __name__ == '__main__':
    main()
