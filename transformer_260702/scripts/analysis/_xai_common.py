# --- transformer_260702/scripts/analysis/_xai_common.py ---
"""生物学的 XAI 分析スクリプトの共通ローダ・ユーティリティ。

- checkpoint 同梱の config_snapshot.py で config を再現
- 学習済みモデルと評価用 DataLoader を構築（決定性のため num_workers=0）
- ゲノム位置別の「モデルが次変異と予測する確率質量」と「実ターゲット出現数」を集計
- 位置→遺伝子(protein) の annotation を codon_mutation4.csv から取得

genome_track.py / homoplasy_alignment.py など各分析はこれを土台にする。
"""

import os
from datetime import datetime

import numpy as np
import torch

from transformer_260702 import config
from transformer_260702.model import HierarchicalTransformer
from transformer_260702.utils.logging import force_print


def load_config_snapshot(checkpoint_dir):
    """checkpoint と同じディレクトリの config_snapshot.py で config を上書きする。"""
    snapshot_path = os.path.join(checkpoint_dir, 'config_snapshot.py')
    if not os.path.exists(snapshot_path):
        force_print(f"[WARNING] config_snapshot.py が見つかりません: {snapshot_path}")
        return
    import importlib.util
    spec = importlib.util.spec_from_file_location('config_snapshot', snapshot_path)
    snap = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(snap)
    overridden = [a for a in dir(snap) if not a.startswith('_') and hasattr(config, a)]
    for a in overridden:
        setattr(config, a, getattr(snap, a))
    force_print(f"[INFO] Loaded config_snapshot.py ({len(overridden)} attrs overridden)")


def make_output_dir(subdir, output_dir=None):
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        return output_dir
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out = os.path.join('outputs', 'transformer_260702', 'scripts', subdir, ts)
    os.makedirs(out, exist_ok=True)
    return out


def save_csv(df, out_dir, filename):
    """DataFrame を out_dir/filename に保存しログする。パスを返す。"""
    path = os.path.join(out_dir, filename)
    df.to_csv(path, index=False)
    force_print(f"[INFO] Saved {path}")
    return path


def save_json(obj, out_dir, filename):
    """dict/list を out_dir/filename に JSON 保存しログする。パスを返す。"""
    import json
    path = os.path.join(out_dir, filename)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    force_print(f"[INFO] Saved {path}")
    return path


def save_fig(fig, out_dir, filename, dpi=140):
    """matplotlib Figure を out_dir/filename に保存・close しログする。パスを返す。"""
    import matplotlib.pyplot as plt
    path = os.path.join(out_dir, filename)
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    force_print(f"[INFO] Saved {path}")
    return path


def load_model_and_loader(checkpoint, split='test', batch_size=None, force_cpu=False):
    """config_snapshot 反映済みのモデルと評価用 DataLoader を返す。

    Returns: (model, loader, device)
    """
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    load_config_snapshot(os.path.dirname(checkpoint))

    # 決定性のため単一プロセス・非シャッフル
    config.NUM_DATALOADER_WORKERS = 0
    if force_cpu:
        config.DEVICE = 'cpu'
    device = config.DEVICE

    model = HierarchicalTransformer().to(device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    force_print(f"[INFO] Loaded checkpoint: epoch={ckpt.get('epoch', -1) + 1}")

    from transformer_260702.db.connection import get_db_path
    from transformer_260702.db.dataset import create_db_dataloader
    split_map = {'train': 0, 'val': 1, 'test': 2}
    loader = create_db_dataloader(
        db_path=get_db_path(),
        split_type=split_map[split],
        batch_size=batch_size or config.BATCH_SIZE,
        shuffle=False,
        max_cooccurrence=config.MAX_CO_OCCURRENCE,
    )
    return model, loader, device


def load_position_gene_map(codon_csv=None):
    """base_pos -> (protein, protein_pos) の辞書を codon_mutation4.csv から作る。"""
    import csv as _csv
    path = codon_csv or config.CODON_CSV
    pos2gene = {}
    with open(path, newline='', encoding='utf-8-sig') as f:
        for row in _csv.DictReader(f):
            try:
                p = int(row['base_pos'])
            except (KeyError, ValueError, TypeError):
                continue
            pos2gene[p] = (row.get('protein', 'unknown'), row.get('protein_pos', '0'))
    return pos2gene


def compute_position_importance(model, loader, n_batches, device, pos_head_idx=1):
    """テストバッチを流し、ゲノム位置ごとに集計する。

    - pred_mass[p]    : モデルが位置 p を次変異と予測した softmax 確率の総和（モデルの注目度）
    - target_count[p] : 実際に位置 p がターゲットだった回数（データの真の分布）

    Returns: (pred_mass ndarray[V], target_count ndarray[V], n_samples)
    """
    V = config.VOCAB_SIZE_POSITION
    pred_mass = np.zeros(V, dtype=np.float64)
    target_count = np.zeros(V, dtype=np.float64)
    n_samples = 0

    model.eval()
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if bi >= n_batches:
                break
            x_cat, x_num, mask = batch[0]
            x_cat = x_cat.to(device)
            x_num = x_num.to(device)
            mask = mask.to(device)
            out = model(x_cat, x_num, src_key_padding_mask=mask)
            logits = out[pos_head_idx]                      # [B, V]
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            # 予測質量を加算（安全のため V に切り詰め）
            w = min(probs.shape[1], V)
            pred_mass[:w] += probs[:, :w].sum(axis=0)
            n_samples += probs.shape[0]

            # 実ターゲット位置（batch[8] = 評価用 Hard Target のフラットリスト）
            raw_y = batch[8] if len(batch) > 8 else None
            if raw_y is not None:
                for targets in raw_y:
                    for t in targets:
                        p = int(t[1])
                        if 0 <= p < V:
                            target_count[p] += 1
    return pred_mass, target_count, n_samples
