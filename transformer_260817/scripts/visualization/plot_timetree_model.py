# --- transformer_260817/scripts/visualization/plot_timetree_model.py ---
# plot_timetree.py と同一レイアウトの系統樹に、モデルの学習・予測信号を重ねる。
#
# 視覚エンコード:
#   ノードサイズ ∝ log(サンプル数)                          … plot_timetree.py と同じ
#   ノード色     ∝ 系統別 position_hit_rate_pct（または fano_gap_pct）
#                  … 既存の walk_forward 集計CSV（entropy_accuracy_pooled_lineages.csv）から取得。GPU不要。
#   エッジ太さ   ∝ 系統別の内部信号（推論で新規抽出、3種を個別図で出力）:
#       ① Co-occurrence Attention の注目の偏り（一様分布 1/co_count からの絶対乖離の平均。0=一様）
#       ② Grad×Input 勾配強度（position head ロジットの x_num に対する勾配×入力の絶対値平均）
#       ③ 予測位置の確信度（position head softmax の最大値）
#
# 系統樹のノードキー（strain_name_usher の full Pango名）は per-lineage 評価CSVおよび推論時の
# batch.strains と同一キー空間のため、両者をそのまま join できる。
#
# 内部信号は各系統から収集日で均等抽出した最大 K サンプルを、採取日に対応する walk_forward
# フォールドのチェックポイントで評価して平均する（fold単位でモデルを1回だけロード）。
#
# Usage:
#   python -m transformer_260817.scripts.visualization.plot_timetree_model \
#       --walk_forward_dir outputs/transformer_260817/results/walk_forward/20260719_001947 \
#       [--pooled_csv outputs/transformer_260817/scripts/entropy_accuracy/<ts>/entropy_accuracy_pooled_lineages.csv] \
#       [--max_samples_per_lineage 30] [--color_metric position_hit_rate_pct] [--force_cpu]
import os
import math
import glob
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

from transformer_260817 import config
from transformer_260817.utils.logging import force_print
from transformer_260817.db.connection import get_db_path, connect_db
from transformer_260817.scripts.analysis.xai import _xai_common as X
from transformer_260817.scripts.analysis.xai.attention_by_month import (
    _attach_attention_capture, POS_COL, BASE_BEFORE_COL)
from transformer_260817.scripts.analysis.xai.lineage_attention_compare import (
    _build_single_sample_batch, _find_fold_for_date)
from transformer_260817.scripts.visualization.plot_timetree import (
    ALIAS_KEY_PATH, MIN_SAMPLES, MAX_NODES,
    load_alias, load_lineage_data, build_tree, compute_layout)


# ------------------------------------------------------------------ #
# per-lineage 精度（ノード色）
# ------------------------------------------------------------------ #
def _latest_pooled_csv():
    """entropy_accuracy/<ts>/entropy_accuracy_pooled_lineages.csv のうち最新timestampを返す。"""
    candidates = sorted(glob.glob(os.path.join(
        'outputs', 'transformer_260817', 'scripts', 'entropy_accuracy',
        '*', 'entropy_accuracy_pooled_lineages.csv')))
    if not candidates:
        raise SystemExit(
            "[ERROR] entropy_accuracy_pooled_lineages.csv が見つかりません。"
            "先に scripts.analysis.walk_forward.entropy_accuracy_analysis を実行するか "
            "--pooled_csv を指定してください。")
    return candidates[-1]


def _weighted_lineage_metric(df, metric):
    """foldをまたいで存在する lineage 行を num_samples 加重平均し、lineage -> 値 の dict を返す。"""
    def _wmean(g):
        w = g['num_samples'].astype(float)
        if w.sum() <= 0:
            return float(g[metric].mean())
        return float((g[metric] * w).sum() / w.sum())
    return df.groupby('lineage').apply(_wmean).to_dict()


# ------------------------------------------------------------------ #
# 内部信号（エッジ太さ）用サンプル収集
# ------------------------------------------------------------------ #
def _collect_lineage_samples(db_path, strain_name_col, lineages, max_samples, fold_windows):
    """系統ごとに収集日で均等抽出した最大 max_samples 件の (sample_id, date) を集め、
    採取日から担当foldを決めて fold_id -> [(lineage, sample_id), ...] にまとめる。

    Returns: (fold_to_items, lineage_folds, n_found)
    """
    con = connect_db(db_path, read_only=True)
    fold_to_items = defaultdict(list)
    lineage_folds = defaultdict(set)
    n_found = {}
    for lineage in lineages:
        rows = con.execute(f"""
            SELECT s.sample_id, s.collection_date
            FROM samples s JOIN strains st ON s.strain_id = st.strain_id
            WHERE st.{strain_name_col} = ?
              AND s.collection_date IS NOT NULL AND s.collection_date != ''
            ORDER BY s.collection_date
        """, [lineage]).fetchall()
        n_found[lineage] = len(rows)
        if not rows:
            continue
        if len(rows) > max_samples:
            idx = sorted(set(np.linspace(0, len(rows) - 1, max_samples).round().astype(int).tolist()))
            rows = [rows[i] for i in idx]
        for sample_id, date in rows:
            fold_id, _is_test_window = _find_fold_for_date(fold_windows, date)
            fold_to_items[fold_id].append((lineage, sample_id))
            lineage_folds[lineage].add(fold_id)
    con.close()
    return fold_to_items, lineage_folds, n_found


def _run_inference(db_path, fold_to_items, fold_ckpts, device):
    """fold単位でモデルを1回だけロードし、サンプルごとに3信号（co-attn/grad/confidence）を集計する。

    信号ごとに2パスへ分離する（同一passに混ぜるとNaNが混入するため。下記Note参照）:
      Pass 1 (no_grad, attentionフック有効): confidence（position head softmaxの最大値）と
              co_attn（Co-occurrence Attentionの一様分布からの乖離）を同時取得。
      Pass 2 (勾配有効, フック無効=モデル素の forward): grad（Grad×Input, position headロジット
              の x_num に対する勾配×入力の絶対値平均）を取得。

    Note（実装上の既知の罠）: _attach_attention_capture はCo-occurrence Attentionの
    need_weights を常に True に強制する。これは nn.MultiheadAttention の明示的softmax経路を
    使わせるため、全共起スロットがPADな timestep で 0/0=NaN を生みうる（forward側は
    CoOccurrenceAttention 内の torch.nan_to_num で 0 埋めされ隠れるが、実測では同一 forward で
    backward まで行うと x_num.grad に NaN が混入することを確認済み）。そのため grad 計算は
    フックを外した素の forward で行い、フック有効時は no_grad に限定する。

    Returns: (signals: {lineage: {'coattn': [...], 'grad': [...], 'confidence': [...]}}, skip_counts: dict)
    """
    signals = defaultdict(lambda: {'coattn': [], 'grad': [], 'confidence': []})
    skip_counts = defaultdict(int)

    for fold_id in sorted(fold_to_items):
        items = fold_to_items[fold_id]
        if fold_id not in fold_ckpts:
            skip_counts['no_checkpoint'] += len(items)
            force_print(f"[WARN] fold {fold_id}: チェックポイントが見つからないため {len(items)} 件をスキップ")
            continue

        force_print(f"[INFO] Fold {fold_id}: {len(items)} サンプルを推論")
        model = X.load_fold_model(fold_ckpts[fold_id], device)
        for p in model.parameters():
            p.requires_grad_(False)  # backward はx_numのみに使う（モデル重みへの勾配計算を省く）

        has_coattn = not getattr(model, 'use_flat_coattn', False)
        orig_co_attn_forward = model.co_attn.forward if has_coattn else None
        captured = _attach_attention_capture(model) if has_coattn else None
        wrapped_co_attn_forward = model.co_attn.forward if has_coattn else None
        if not has_coattn:
            skip_counts['flat_coattn_model'] += len(items)
            force_print(f"[WARN] fold {fold_id}: USE_FLAT_COATTN構成のためco-attention信号は取得不可")

        for lineage, sample_id in items:
            batch = _build_single_sample_batch(db_path, sample_id)
            if batch is None:
                skip_counts['missing_features'] += 1
                continue
            x_cat_cpu, x_num_cpu, mask_cpu = batch
            x_cat = x_cat_cpu.to(device)
            mask = mask_cpu.to(device)

            # --- Pass 1 (no_grad): 確信度 + Co-occurrence Attention ---
            if has_coattn:
                model.co_attn.forward = wrapped_co_attn_forward
            x_num_ng = x_num_cpu.to(device, dtype=torch.float32)
            with torch.no_grad():
                out = model(x_cat, x_num_ng, src_key_padding_mask=mask)
                confidence = float(torch.softmax(out[1], dim=-1).max().item())

            co_signal = None
            if has_coattn and 'w' in captured:
                w = captured['w'][0].cpu().numpy()                               # [T, C]
                is_pad = (x_cat[0, ..., BASE_BEFORE_COL] == 0).cpu().numpy()      # [T, C]
                co_count_per_t = (~is_pad).sum(axis=1)                           # [T]
                # relative_attn = attn_weight * co_count は timestep内で合計=co_count
                # （softmaxの正規化から）なので単純平均は常に1.0に潰れて情報を持たない
                # （数学的恒等式）。一様分布からの絶対乖離の平均を「注目の偏り度合い」
                # （0=一様, 大きいほど特定の共起変異に注目が集中）として使う。
                vals = [abs(float(w[t, c]) * int(co_count_per_t[t]) - 1.0)
                        for t in range(w.shape[0]) for c in range(w.shape[1])
                        if not is_pad[t, c] and co_count_per_t[t] > 0]
                if vals:
                    co_signal = float(np.mean(vals))

            # --- Pass 2 (勾配有効・フック無効): Grad×Input ---
            if has_coattn:
                model.co_attn.forward = orig_co_attn_forward
            x_num_g = x_num_cpu.to(device, dtype=torch.float32).detach().requires_grad_(True)
            out_g = model(x_cat, x_num_g, src_key_padding_mask=mask)
            score = out_g[1].max(dim=-1).values.sum()
            score.backward()
            grad = x_num_g.grad
            grad_signal = (float((grad * x_num_g).abs().mean().item())
                          if grad is not None and torch.isfinite(grad).all() else float('nan'))

            signals[lineage]['confidence'].append(confidence)
            signals[lineage]['grad'].append(grad_signal)
            if co_signal is not None:
                signals[lineage]['coattn'].append(co_signal)

        if has_coattn:
            model.co_attn.forward = orig_co_attn_forward  # 後始末（model自体はこの後delするが念のため）
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    return signals, skip_counts


# ------------------------------------------------------------------ #
# 描画（plot_timetree.draw_tree の骨格を踏襲、色/太さを可変に）
# ------------------------------------------------------------------ #
def draw_tree_overlay(selected, parent_map, recom_set, info, x_pos, y_pos, all_months,
                       node_color, node_color_label, edge_signal, edge_signal_label,
                       output_dir, filename, title):
    max_n = max(d['total_n'] for d in info.values() if d['total_n'] > 0)

    valid_edge_vals = [v for v in edge_signal.values() if v is not None and not np.isnan(v)]
    if valid_edge_vals:
        lo, hi = np.percentile(valid_edge_vals, [5, 95])
        if hi <= lo:
            lo, hi = min(valid_edge_vals), max(valid_edge_vals) + 1e-9
    else:
        lo, hi = 0.0, 1.0

    valid_color_vals = [v for v in node_color.values() if v is not None and not np.isnan(v)]
    cmap = plt.cm.viridis
    if valid_color_vals:
        cmin, cmax = min(valid_color_vals), max(valid_color_vals)
        if cmax <= cmin:
            cmax = cmin + 1e-9
    else:
        cmin, cmax = 0.0, 1.0
    norm_color = Normalize(vmin=cmin, vmax=cmax)

    fig, ax = plt.subplots(figsize=(22, max(10, len(selected) * 0.3)))

    # エッジ描画（水平: 親X→子X at 親Y, 垂直: 親Y→子Y at 子X）。太さ=内部信号、破線=組換え系統。
    for child, par in parent_map.items():
        if par is None or child == '__ROOT__':
            continue
        xc, yc = x_pos[child], y_pos[child]
        xp, yp = x_pos[par], y_pos[par]
        sig = edge_signal.get(child)
        if sig is None or np.isnan(sig):
            lw, ls, color = 0.6, ':', '#cccccc'
        else:
            frac = float(np.clip((sig - lo) / (hi - lo), 0.0, 1.0))
            lw = 0.5 + frac * 4.5
            ls = '--' if child in recom_set else '-'
            color = '#555555'
        ax.plot([xp, xc], [yp, yp], ls=ls, lw=lw, color=color, zorder=1)
        ax.plot([xc, xc], [yp, yc], ls=ls, lw=lw, color=color, zorder=1)

    # ノード描画。サイズ=log(サンプル数)、色=精度指標。
    for node in selected:
        if node == '__ROOT__':
            continue
        x, y = x_pos[node], y_pos[node]
        n = info[node]['total_n']
        size = 30 + 180 * math.log1p(n) / math.log1p(max_n)
        cval = node_color.get(node)
        fill = '#dddddd' if cval is None or np.isnan(cval) else cmap(norm_color(cval))
        ec = 'dimgray' if node in recom_set else 'none'
        ax.scatter(x, y, s=size, color=fill, alpha=0.9, edgecolors=ec, linewidths=0.8, zorder=3)
        ax.text(x + 0.15, y, node, fontsize=9, va='center', zorder=4)

    tick_step = 3
    ticks = [i for i in range(len(all_months)) if i % tick_step == 0]
    ax.set_xticks(ticks)
    ax.set_xticklabels([all_months[i] for i in ticks], rotation=45, ha='right', fontsize=8)
    ax.set_xlim(-1, len(all_months))
    ax.set_yticks([])
    ax.set_xlabel('Peak month (highest sample count)')
    ax.set_title(title)
    ax.grid(axis='x', linestyle='--', alpha=0.3)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_color)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.01, fraction=0.02)
    cbar.set_label(node_color_label, fontsize=9)

    edge_legend = [
        Line2D([0], [0], color='#555555', lw=0.6, label=f'low {edge_signal_label}'),
        Line2D([0], [0], color='#555555', lw=3.2, label=f'high {edge_signal_label}'),
        Line2D([0], [0], color='#cccccc', lw=0.6, ls=':', label='no signal (skipped)'),
        Line2D([0], [0], color='#555555', lw=1.5, ls='--', label='recombinant lineage'),
    ]
    ax.legend(handles=edge_legend, loc='upper left', fontsize=7, framealpha=0.7)

    plt.tight_layout()
    return X.save_fig(fig, output_dir, filename)


# ------------------------------------------------------------------ #
# main
# ------------------------------------------------------------------ #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--walk_forward_dir', required=True,
                    help='fold_N/*/models/best_model.pth を含む walk_forward 出力ディレクトリ')
    ap.add_argument('--pooled_csv', default=None,
                    help='entropy_accuracy_pooled_lineages.csv のパス（省略時は最新runを自動探索）')
    ap.add_argument('--max_samples_per_lineage', type=int, default=30,
                    help='内部信号推論に使う系統あたりの最大サンプル数（収集日で均等抽出）')
    ap.add_argument('--color_metric', default='position_hit_rate_pct',
                    choices=['position_hit_rate_pct', 'fano_gap_pct'],
                    help='ノード色に使う精度指標')
    ap.add_argument('--min_samples', type=int, default=MIN_SAMPLES)
    ap.add_argument('--max_nodes', type=int, default=MAX_NODES)
    ap.add_argument('--force_cpu', action='store_true')
    ap.add_argument('--output_dir', default=None)
    args = ap.parse_args()

    device = torch.device('cpu' if args.force_cpu or not torch.cuda.is_available() else config.DEVICE)
    pooled_csv = args.pooled_csv or _latest_pooled_csv()
    force_print(f"[INFO] Using pooled lineage CSV: {pooled_csv}")

    force_print("[INFO] Loading alias key...")
    alias_key, rev = load_alias(ALIAS_KEY_PATH)

    force_print("[INFO] Loading lineage data from DB...")
    df, summary = load_lineage_data()
    all_months = sorted(df['month'].unique())
    force_print(f"[INFO] Lineages: {len(summary)}, Months: {len(all_months)}")

    force_print("[INFO] Building tree...")
    selected, parent_map, children_map, recom_set, info = build_tree(
        summary, alias_key, rev, args.min_samples, args.max_nodes)
    lineages = [n for n in selected if n != '__ROOT__']
    force_print(f"[INFO] Selected nodes: {len(lineages)}  Recombinants: {len(recom_set)}")

    force_print("[INFO] Computing layout...")
    x_pos, y_pos = compute_layout(children_map, info, all_months)

    force_print("[INFO] Loading per-lineage accuracy metrics...")
    pooled_df = pd.read_csv(pooled_csv)
    acc_lookup = {}
    for metric in ('position_hit_rate_pct', 'fano_gap_pct'):
        acc_lookup[metric] = (_weighted_lineage_metric(pooled_df, metric)
                              if metric in pooled_df.columns else {})

    force_print("[INFO] Discovering walk_forward fold checkpoints...")
    fold_windows = X.get_fold_windows()
    fold_ckpts = X.discover_fold_checkpoints(args.walk_forward_dir)
    if not fold_ckpts:
        raise SystemExit(f"[ERROR] fold チェックポイントが見つかりません: {args.walk_forward_dir}")
    force_print(f"[INFO] Found fold checkpoints: {sorted(fold_ckpts)}")

    strain_name_col = ('strain_name_usher' if getattr(config, 'STRENGTH_SOURCE', 'ncbi') == 'usher'
                       else 'strain_name_ncbi')
    db_path = get_db_path()

    force_print(f"[INFO] Collecting up to {args.max_samples_per_lineage} samples per lineage...")
    fold_to_items, lineage_folds, n_found = _collect_lineage_samples(
        db_path, strain_name_col, lineages, args.max_samples_per_lineage, fold_windows)
    n_no_samples = sum(1 for l in lineages if n_found.get(l, 0) == 0)
    force_print(f"[INFO] {n_no_samples}/{len(lineages)} lineages had no DB samples with a collection_date.")

    force_print("[INFO] Running per-sample inference for internal signals...")
    signals, skip_counts = _run_inference(db_path, fold_to_items, fold_ckpts, device)
    force_print(f"[INFO] Skip counts: {dict(skip_counts)}")

    rows = []
    for lineage in lineages:
        sig = signals.get(lineage, {'coattn': [], 'grad': [], 'confidence': []})
        rows.append({
            'lineage': lineage,
            'n_samples': info[lineage]['total_n'],
            'position_hit_rate_pct': acc_lookup['position_hit_rate_pct'].get(lineage, float('nan')),
            'fano_gap_pct': acc_lookup['fano_gap_pct'].get(lineage, float('nan')),
            'coattn_mean': float(np.mean(sig['coattn'])) if sig['coattn'] else float('nan'),
            'grad_mean': float(np.mean(sig['grad'])) if sig['grad'] else float('nan'),
            'confidence_mean': float(np.mean(sig['confidence'])) if sig['confidence'] else float('nan'),
            'fold_assigned': ','.join(str(f) for f in sorted(lineage_folds.get(lineage, []))),
            'n_inference_samples': len(sig['confidence']),
        })
    df_signals = pd.DataFrame(rows)

    out_dir = X.make_output_dir('timetree_model', args.output_dir)
    X.save_csv(df_signals, out_dir, 'timetree_node_signals.csv')

    node_color = dict(zip(df_signals['lineage'], df_signals[args.color_metric]))
    color_label = {'position_hit_rate_pct': 'Position Hit Rate (%, higher=better)',
                   'fano_gap_pct': 'Fano Gap (%, ceiling − actual, lower=better)'}[args.color_metric]

    signal_specs = [
        ('coattn_mean', 'timetree_coattn.png', 'Co-occurrence Attention skew (0=uniform, higher=concentrated)',
         'Model learning signal on lineage timetree — edge width = Co-occurrence Attention skew'),
        ('grad_mean', 'timetree_gradient.png', 'Gradient x Input magnitude (position head)',
         'Model learning signal on lineage timetree — edge width = Gradient x Input'),
        ('confidence_mean', 'timetree_confidence.png', 'Predicted-position confidence (softmax max)',
         'Model learning signal on lineage timetree — edge width = prediction confidence'),
    ]
    for col, filename, edge_label, title_base in signal_specs:
        edge_signal = dict(zip(df_signals['lineage'], df_signals[col]))
        title = f'{title_base}\nnode color = {color_label}, node size ∝ log(samples)'
        draw_tree_overlay(selected, parent_map, recom_set, info, x_pos, y_pos, all_months,
                          node_color, color_label, edge_signal, edge_label, out_dir, filename, title)

    meta = {
        'walk_forward_dir': args.walk_forward_dir,
        'pooled_csv': pooled_csv,
        'max_samples_per_lineage': args.max_samples_per_lineage,
        'color_metric': args.color_metric,
        'n_tree_nodes': len(lineages),
        'n_recombinants': len(recom_set),
        'fold_checkpoints_found': sorted(fold_ckpts),
        'skip_counts': dict(skip_counts),
        'n_lineages_no_db_samples': n_no_samples,
    }
    X.save_json(meta, out_dir, 'timetree_model_meta.json')
    force_print("[INFO] Done.")


if __name__ == '__main__':
    main()
