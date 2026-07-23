# --- transformer_260723/scripts/analysis/verify/verify_coattn_frequency_confound.py ---
"""Co-occurrence Attentionの「偏り」(coattn skew)と精度の逆相関が、頻度交絡
（グローバルに目立つ"定番"変異への注目集中による見かけの偏りで、系統固有の推論では
ないという疑い）で説明できるかを検証する。

背景: plot_timetree_model.py の系統別集計で、Co-occurrence Attentionの一様分布
からの乖離（coattn_mean, 0=一様・大きいほど特定の共起変異に集中）が
position_hit_rate_pct と負相関（オミクロン系統群で偏りが強いのに精度は低い）という
直感に反する傾向が見つかった。attention_by_month.py の check_frequency_confound は
月×位置プールでの頻度交絡を検定するが、「偏りが強い系統 vs 弱い系統」で層別した
検証は行っていなかったため、本スクリプトで補う。

手法:
  1. plot_timetree_model.py の出力(timetree_node_signals.csv)から coattn_mean の
     上位/下位グループ（系統）を選ぶ。
  2. 各グループの系統について lineage_attention_compare.py と同じ要領で
     スロット単位 (position, attn_weight, co_count, relative_attn) を再抽出する。
  3. 各スロットの position を reference/homoplasy/site_recurrence.csv の
     recurrence_count（独立系統間での再発回数＝ホモプラシー。値が大きいほど
     「多くの系統で繰り返し出現する定番変異」）と突き合わせ、
     relative_attn との Spearman相関をグループ別に比較する。相関が「偏り大」
     グループで有意に強ければ、attentionが系統固有の情報ではなく、グローバルに
     目立つ定番変異へ収束している（頻度交絡）疑いが強まる。
  4. 各グループの argmax（timestepごとに最も注目されたスロット）が、ホモプラシー
     再発回数の上位5%（＝定番変異）に該当する割合も比較する。

注意: coattn_mean上位/下位はDelta期とオミクロン期でおおむね系統群が分かれるため
（era交絡）、本検証だけで「偏り→精度低下」の因果は確定できない。ただし
「attentionが実際にどの位置に集中しているか」を直接見ることで、era交絡とは別に
機構的な手がかり（定番変異への収束か否か）を与える。

DB読み取り専用（split_type_wf 等は変更しない）。

Usage:
  python -m transformer_260723.scripts.analysis.verify.verify_coattn_frequency_confound \\
      --walk_forward_dir outputs/transformer_260723/results/walk_forward/20260719_001947 \\
      --node_signals_csv outputs/transformer_260723/scripts/timetree_model/<ts>/timetree_node_signals.csv \\
      [--top_n_lineages 15] [--max_samples_per_lineage 30] [--force_cpu]
"""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import torch

from transformer_260723 import config
from transformer_260723.db.connection import get_db_path
from transformer_260723.utils.logging import force_print
from transformer_260723.scripts.analysis.xai import _xai_common as X
from transformer_260723.scripts.analysis.xai.attention_by_month import (
    _attach_attention_capture, POS_COL, BASE_BEFORE_COL)
from transformer_260723.scripts.analysis.xai.lineage_attention_compare import _build_single_sample_batch
from transformer_260723.scripts.visualization.plot_timetree_model import _collect_lineage_samples
from transformer_260723.scripts.visualization.plot_timetree import get_group


def _load_recurrence_map():
    df = pd.read_csv(config.HOMOPLASY_CSV)
    return dict(zip(df['position_id'].astype(int), df['recurrence_count'].astype(float)))


def _select_groups(node_signals_csv, top_n):
    df = pd.read_csv(node_signals_csv).dropna(subset=['coattn_mean'])
    concentrated = df.sort_values('coattn_mean', ascending=False).head(top_n)
    uniform = df.sort_values('coattn_mean', ascending=True).head(top_n)

    for name, sub in (('偏り大(concentrated)', concentrated), ('偏り小(uniform)', uniform)):
        groups = sub['lineage'].apply(get_group).value_counts().to_dict()
        force_print(f"[INFO] {name} 上位{len(sub)}系統: "
                    f"coattn_mean {sub['coattn_mean'].min():.3f}-{sub['coattn_mean'].max():.3f}, "
                    f"mean hit_rate={sub['position_hit_rate_pct'].mean():.2f}%, "
                    f"era内訳={groups}")
    return concentrated['lineage'].tolist(), uniform['lineage'].tolist()


def _extract_slots(db_path, lineages, fold_ckpts, fold_windows, max_samples, device):
    """スロット単位 (lineage, position, attn_weight, co_count, relative_attn) を抽出する
    （lineage_attention_compare.py の1サンプル抽出を多系統・多サンプルへ一般化）。
    """
    fold_to_items, _lineage_folds, _n_found = _collect_lineage_samples(
        db_path, 'strain_name_usher', lineages, max_samples, fold_windows)

    rows = []
    for fold_id in sorted(fold_to_items):
        items = fold_to_items[fold_id]
        if fold_id not in fold_ckpts:
            force_print(f"[WARN] fold {fold_id}: チェックポイントが見つからないため {len(items)} 件をスキップ")
            continue
        model = X.load_fold_model(fold_ckpts[fold_id], device)
        if getattr(model, 'use_flat_coattn', False):
            force_print(f"[WARN] fold {fold_id}: USE_FLAT_COATTN構成のためスキップ")
            continue
        captured = _attach_attention_capture(model)

        for lineage, sample_id in items:
            batch = _build_single_sample_batch(db_path, sample_id)
            if batch is None:
                continue
            x_cat, x_num, mask = [t.to(device) for t in batch]
            with torch.no_grad():
                model(x_cat, x_num, src_key_padding_mask=mask)
            w = captured['w'][0].cpu().numpy()                              # [T, C]
            pos = x_cat[0, ..., POS_COL].cpu().numpy()                      # [T, C]
            is_pad = (x_cat[0, ..., BASE_BEFORE_COL] == 0).cpu().numpy()    # [T, C]
            co_count_per_t = (~is_pad).sum(axis=1)                         # [T]

            for t in range(w.shape[0]):
                cc = int(co_count_per_t[t])
                if cc == 0:
                    continue
                for c in range(w.shape[1]):
                    if is_pad[t, c]:
                        continue
                    rows.append({
                        'lineage': lineage, 'fold': fold_id, 'sample_id': sample_id,
                        'timestep': t, 'position': int(pos[t, c]),
                        'attn_weight': float(w[t, c]), 'co_count': cc,
                        'relative_attn': float(w[t, c]) * cc,
                    })
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    return pd.DataFrame(rows)


def _argmax_slots(df):
    """(lineage, sample_id, timestep) ごとに最大attn_weightのスロットだけ残す。
    co_count==1（選択の余地がない）は除外する。
    """
    df = df[df['co_count'] >= 2]
    if df.empty:
        return df
    idx = df.groupby(['lineage', 'sample_id', 'timestep'])['attn_weight'].idxmax()
    return df.loc[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--walk_forward_dir', required=True)
    ap.add_argument('--node_signals_csv', required=True,
                    help='plot_timetree_model.py が出力した timetree_node_signals.csv')
    ap.add_argument('--top_n_lineages', type=int, default=15,
                    help='coattn_mean 上位/下位それぞれ何系統を比較対象にするか')
    ap.add_argument('--max_samples_per_lineage', type=int, default=30)
    ap.add_argument('--force_cpu', action='store_true')
    ap.add_argument('--output_dir', default=None)
    args = ap.parse_args()

    device = torch.device('cpu' if args.force_cpu or not torch.cuda.is_available() else config.DEVICE)
    db_path = get_db_path()
    recurrence = _load_recurrence_map()
    force_print(f"[INFO] Homoplasy recurrence map: {len(recurrence)} positions")

    concentrated_lineages, uniform_lineages = _select_groups(args.node_signals_csv, args.top_n_lineages)

    fold_windows = X.get_fold_windows()
    fold_ckpts = X.discover_fold_checkpoints(args.walk_forward_dir)
    if not fold_ckpts:
        raise SystemExit(f"[ERROR] fold チェックポイントが見つかりません: {args.walk_forward_dir}")

    force_print("\n[INFO] === 偏り大(concentrated)グループ 推論中 ===")
    df_c = _extract_slots(db_path, concentrated_lineages, fold_ckpts, fold_windows,
                          args.max_samples_per_lineage, device)
    force_print("\n[INFO] === 偏り小(uniform)グループ 推論中 ===")
    df_u = _extract_slots(db_path, uniform_lineages, fold_ckpts, fold_windows,
                          args.max_samples_per_lineage, device)

    results = {}
    for name, df in (('concentrated', df_c), ('uniform', df_u)):
        df = df.copy()
        df['recurrence'] = df['position'].map(recurrence).fillna(0.0)
        rho, pval = spearmanr(df['recurrence'], df['relative_attn'])
        force_print(f"\n[{name}] 全スロット(n={len(df)}): "
                    f"Spearman(homoplasy再発回数, relative_attn) r={rho:.4f}, p={pval:.2e}")
        arg = _argmax_slots(df)
        results[name] = {'df': df, 'argmax': arg, 'rho': rho, 'pval': pval}

    all_recurrence = pd.concat([results['concentrated']['df']['recurrence'],
                                results['uniform']['df']['recurrence']])
    positive = all_recurrence[all_recurrence > 0]
    hotspot_thresh = float(np.percentile(positive, 95)) if len(positive) else 0.0
    force_print(f"\n[INFO] ホモプラシー再発回数 上位5%閾値（両グループ込み）: {hotspot_thresh:.1f}")

    summary_rows = []
    for name in ('concentrated', 'uniform'):
        arg = results[name]['argmax']
        frac_hotspot = float((arg['recurrence'] >= hotspot_thresh).mean()) if len(arg) else float('nan')
        mean_rec = float(arg['recurrence'].mean()) if len(arg) else float('nan')
        force_print(f"[{name}] argmaxスロット(n={len(arg)})のうち再発回数上位5%該当率: "
                    f"{frac_hotspot * 100:.1f}%  (平均recurrence={mean_rec:.1f})")
        summary_rows.append({
            'group': name, 'n_slots': len(results[name]['df']), 'n_argmax_slots': len(arg),
            'spearman_r_recurrence_vs_relative_attn': results[name]['rho'],
            'spearman_p': results[name]['pval'],
            'frac_argmax_in_top5pct_recurrence': frac_hotspot,
            'mean_argmax_recurrence': mean_rec,
        })

    out_dir = X.make_output_dir('verify_coattn_frequency_confound', args.output_dir)
    summary_df = pd.DataFrame(summary_rows)
    X.save_csv(summary_df, out_dir, 'confound_summary.csv')
    X.save_csv(results['concentrated']['df'], out_dir, 'concentrated_slots.csv')
    X.save_csv(results['uniform']['df'], out_dir, 'uniform_slots.csv')

    force_print(f"\n{summary_df.to_string(index=False)}")
    force_print("\n[INFO] Done.")


if __name__ == '__main__':
    main()
