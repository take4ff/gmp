# 月別・ゲノム位置(NucPos)別の「モデルの Co-occurrence Attention 重み」ヒートマップ。
#
# target_position_by_month.png（= 月×NucPos のサンプル数ヒートマップ）と同じレイアウトで、
# 色をデータのサンプル数ではなく「共起集約 Cross-Attention がその位置の変異に置いた注目度」に
# 置き換えたもの。学習済みモデルが、どの月・どのゲノム位置の変異に注目しているかを可視化する。
#
# 仕組み:
#   CoOccurrenceAttention の Cross-Attention は各タイムステップの共起変異集合（C 個）に対する
#   注目重み（合計 1）を持つ。need_weights=True で attn_weights[B*T,1,C] を取り出し、各共起変異の
#   NucPos（x_cat[...,1]）とサンプルの収集月（collection_date）で集計する。
#     value='mean'  : 位置 p が共起集合に現れたときの平均注目重み（頻度から切り離した「単位注目度」）
#     value='mass'  : 注目重みの総和（頻度 × 注目度。データ分布に近い見え方）
#
# 2 つのモード:
#   (A) 単一チェックポイント:
#     python -m transformer_260702.scripts.analysis.attention_by_month \
#         --checkpoint outputs/.../models/best_model.pth [--split test] [--include_duplicates]
#   (B) walk_forward 日付別（各フォールドを自分の test ウィンドウで評価し全期間を連結・外挿なし）:
#     python -m transformer_260702.scripts.analysis.attention_by_month \
#         --walk_forward_dir outputs/transformer_260702/results/walk_forward/20260706_223816
import argparse
import os
import re
from collections import defaultdict

import numpy as np
import torch
import matplotlib.pyplot as plt

from transformer_260702 import config
from transformer_260702.utils.logging import force_print
from transformer_260702.scripts.analysis import _xai_common as X

# x_cat の列インデックス（pretrain.py: position=1, region=7 と一致）
POS_COL = 1
BASE_BEFORE_COL = 0  # ==0 は PAD 共起スロット

# walk_forward の半年次フォールド定義（scripts/eval/walk_forward.py の FOLDS と同一）。
# (fold_id, train_start, split_date, split_end): test ウィンドウ = [split_date, split_end)
FOLDS = [
    (1, None,         '2021-01-01', '2021-07-01'),
    (2, '2021-01-01', '2021-07-01', '2022-01-01'),
    (3, '2021-07-01', '2022-01-01', '2022-07-01'),
    (4, '2022-01-01', '2022-07-01', '2023-01-01'),
    (5, '2022-07-01', '2023-01-01', '2023-07-01'),
    (6, '2023-01-01', '2023-07-01', '2024-01-01'),
    (7, '2023-07-01', '2024-01-01', None),
]
FOLD_WINDOW = {f[0]: (f[1], f[2], f[3]) for f in FOLDS}


def _attach_attention_capture(model):
    """model.co_attn.forward をラップし、直近 forward の attn 重みを captured に格納する。

    返り値 captured['w'] は [B, T, C]（average_attn_weights 済み）。外側の forward は
    重みなし単一テンソルを期待するため、ラッパは output のみ返す。
    """
    captured = {}
    co_attn = getattr(model, 'co_attn', None)
    if co_attn is None:
        raise RuntimeError("model に co_attn がありません（USE_FLAT_COATTN=True 構成は非対応）。")
    orig_forward = co_attn.forward

    def wrapped(x, co_occur_mask=None, need_weights=False):
        out, w = orig_forward(x, co_occur_mask=co_occur_mask, need_weights=True)
        B, T, C, _ = x.shape
        captured['w'] = w.reshape(B, T, C).detach()
        return out

    co_attn.forward = wrapped
    return captured


def aggregate_attention(model, loader, device, n_batches, attn_sum=None, occ_count=None):
    """loader を流し、(month, nuc_pos) ごとに注目重みの総和 attn_sum と出現回数 occ_count を集計する。

    attn_sum / occ_count を渡すと追記する（フォールド跨ぎのマージ用）。処理サンプル数を返す。
    """
    if attn_sum is None:
        attn_sum = defaultdict(lambda: defaultdict(float))
    if occ_count is None:
        occ_count = defaultdict(lambda: defaultdict(int))

    captured = _attach_attention_capture(model)
    n_processed = 0
    bi = -1
    model.eval()
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if n_batches >= 0 and bi >= n_batches:
                break
            x_cat, x_num, mask = batch[0]
            collection_dates = batch[9]
            x_cat = x_cat.to(device)
            x_num = x_num.to(device)
            mask = mask.to(device)

            model(x_cat, x_num, src_key_padding_mask=mask)  # forward → captured['w']
            w = captured['w'].float().cpu().numpy()          # [B, T, C]
            pos = x_cat[..., POS_COL].cpu().numpy()          # [B, T, C]
            is_pad = (x_cat[..., BASE_BEFORE_COL] == 0).cpu().numpy()  # True=PAD

            for b in range(x_cat.shape[0]):
                date = collection_dates[b] if b < len(collection_dates) else ''
                if not date or len(date) < 7 or date[4] != '-':
                    continue
                month = date[:7]
                valid = ~is_pad[b]
                if not valid.any():
                    continue
                pw = np.nan_to_num(w[b], nan=0.0)
                pp = pos[b]
                am, oc = attn_sum[month], occ_count[month]
                for t_i, c_i in zip(*np.where(valid)):
                    p = int(pp[t_i, c_i])
                    am[p] += float(pw[t_i, c_i])
                    oc[p] += 1
            n_processed += x_cat.shape[0]
            if (bi + 1) % 50 == 0:
                force_print(f"[INFO]   {bi + 1} batches, {n_processed:,} samples")
    force_print(f"[INFO] Aggregated {n_processed:,} samples over {bi + 1} batches.")
    return attn_sum, occ_count, n_processed


def build_matrix_and_plot(attn_sum, occ_count, top_n, value, out_dir, title, meta_extra=None):
    """集計辞書から 月×NucPos 行列を作り、YlOrRd ヒートマップを保存する。"""
    pos_total_occ = defaultdict(int)
    for oc in occ_count.values():
        for p, c in oc.items():
            pos_total_occ[p] += c
    if not pos_total_occ:
        raise SystemExit("[ERROR] 有効な (月, 位置) が集計されませんでした。split/データを確認してください。")

    top_positions = sorted(pos_total_occ, key=lambda p: pos_total_occ[p], reverse=True)[:top_n]
    top_positions_sorted = sorted(top_positions)   # ゲノム位置昇順
    months = sorted(occ_count.keys())              # YYYY-MM は辞書順＝時系列順
    month_idx = {m: i for i, m in enumerate(months)}
    pos_idx = {p: i for i, p in enumerate(top_positions_sorted)}
    n_pos, n_months = len(top_positions_sorted), len(months)

    matrix = np.zeros((n_pos, n_months), dtype=np.float64)
    for month in months:
        mi = month_idx[month]
        am, oc = attn_sum[month], occ_count[month]
        for p in top_positions_sorted:
            c = oc.get(p, 0)
            if c:
                matrix[pos_idx[p], mi] = (am[p] / c) if value == 'mean' else am[p]

    force_print(f"[INFO] Matrix: {n_pos} positions × {n_months} months (value={value})")

    fig_w = max(18, n_months * 0.38)
    fig_h = max(14, n_pos * 0.19)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    if value == 'mass':
        disp = np.log1p(matrix)
        cbar_label = 'Attention mass (Σ weight, log scale)'
    else:
        disp = matrix
        cbar_label = 'Mean attention weight'
    im = ax.imshow(disp, aspect='auto', cmap='YlOrRd', interpolation='nearest')

    cbar = plt.colorbar(im, ax=ax, pad=0.01, fraction=0.02)
    cbar.set_label(cbar_label, fontsize=9)
    if value == 'mass':
        max_val = float(matrix.max())
        raw_ticks = [v for v in [0, 1, 10, 100, 1000, 10000, 100000] if v <= max_val]
        if max_val not in raw_ticks:
            raw_ticks.append(max_val)
        cbar.set_ticks([np.log1p(v) for v in raw_ticks])
        cbar.set_ticklabels([f'{v:,.0f}' for v in raw_ticks], fontsize=7)

    ax.set_xticks(range(n_months))
    ax.set_xticklabels(months, rotation=45, ha='right', fontsize=7)
    ax.set_yticks(range(n_pos))
    ax.set_yticklabels([str(p) for p in top_positions_sorted], fontsize=6)
    ax.set_xlabel('Month')
    # y 軸は入力変異パス全体（系列全体）の共起変異位置。ターゲット（次変異）ではない点に注意。
    ax.set_ylabel('Attended mutation NucPos (whole input series)')
    ax.set_title(title)
    plt.tight_layout()

    X.save_fig(fig, out_dir, 'attention_by_month.png')

    import pandas as pd
    df = pd.DataFrame(matrix, index=top_positions_sorted, columns=months)
    df.index.name = 'nuc_pos'
    X.save_csv(df.reset_index(), out_dir, 'attention_by_month.csv')
    meta = {
        'value': value, 'n_months': n_months, 'n_positions': n_pos,
        'months': [months[0], months[-1]] if months else [],
        'top_positions_by_occurrence': [[int(p), int(pos_total_occ[p])] for p in top_positions[:20]],
    }
    if meta_extra:
        meta.update(meta_extra)
    X.save_json(meta, out_dir, 'attention_by_month_meta.json')
    return out_dir


def _load_fold_model(checkpoint, device):
    """config_snapshot を反映してフォールドのモデルを構築・ロードする（loader は作らない）。"""
    from transformer_260702.model import HierarchicalTransformer
    X.load_config_snapshot(os.path.dirname(checkpoint))
    config.NUM_DATALOADER_WORKERS = 0
    model = HierarchicalTransformer().to(device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    force_print(f"[INFO] Loaded fold checkpoint: {checkpoint} (epoch={ckpt.get('epoch', -1) + 1})")
    return model


def run_walk_forward(args):
    """各フォールドを自分の test ウィンドウで評価し、日付軸で連結した全期間ヒートマップを作る。"""
    from transformer_260702.db.connection import get_db_path, connect_db
    from transformer_260702.db.dataset import create_db_dataloader

    device = 'cpu' if args.force_cpu else config.DEVICE
    db_path = get_db_path()

    # フォールドチェックポイントを探索（fold_N/*/models/best_model.pth）
    fold_ckpts = {}
    for root, _dirs, files in os.walk(args.walk_forward_dir):
        if 'best_model.pth' in files:
            m = re.search(r'fold_(\d+)', root)
            if m:
                fold_ckpts[int(m.group(1))] = os.path.join(root, 'best_model.pth')
    if not fold_ckpts:
        raise SystemExit(f"[ERROR] fold チェックポイントが見つかりません: {args.walk_forward_dir}")
    if args.folds:
        fold_ckpts = {f: p for f, p in fold_ckpts.items() if f in set(args.folds)}
    force_print(f"[INFO] 発見したフォールド: {sorted(fold_ckpts)}")

    attn_sum = defaultdict(lambda: defaultdict(float))
    occ_count = defaultdict(lambda: defaultdict(int))
    total_samples = 0
    used_folds = []

    for fold_id in sorted(fold_ckpts):
        if fold_id not in FOLD_WINDOW:
            force_print(f"[WARN] fold {fold_id} は FOLDS 定義に無いためスキップ")
            continue
        train_start, split_date, split_end = FOLD_WINDOW[fold_id]
        force_print(f"\n===== Fold {fold_id}: test window [{split_date}, {split_end}) =====")

        model = _load_fold_model(fold_ckpts[fold_id], device)
        if getattr(model, 'use_flat_coattn', False):
            force_print(f"[WARN] fold {fold_id} は USE_FLAT_COATTN のためスキップ")
            continue

        # このフォールドの test ウィンドウで split_type_wf を割り当て直す
        config.WALK_FORWARD_TRAIN_START = train_start
        config.TEMPORAL_SPLIT_DATE = split_date
        config.TEMPORAL_SPLIT_TEST_END = split_end
        config.USE_UNIQUE_FILTER = False        # 参照と同じ全件基準（重複を潰さない）
        config.USE_TRAIN_ENTROPY_FILTER = False
        # test ウィンドウ [split_date, split_end) だけを split_type_wf=2 に立てる軽量割り当て。
        # assign_wf_splits() は train/valid 分割（巨大な IN 句 UPDATE）まで行い遅いので、
        # test しか読まない本用途では自前の 2 本の UPDATE で済ませる。
        con = connect_db(db_path, read_only=False)
        cols = [r[1] for r in con.execute("PRAGMA table_info('samples')").fetchall()]
        if 'split_type_wf' not in cols:
            con.execute("ALTER TABLE samples ADD COLUMN split_type_wf INTEGER DEFAULT -1")
        con.execute("UPDATE samples SET split_type_wf = -1")
        base = ("UPDATE samples SET split_type_wf = 2 WHERE collection_date IS NOT NULL "
                "AND collection_date != '' AND RPAD(collection_date, 10, '-01-01') >= ?")
        if split_end is None:
            con.execute(base, [split_date])
        else:
            con.execute(base + " AND RPAD(collection_date, 10, '-01-01') < ?", [split_date, split_end])
        n_test = con.execute("SELECT COUNT(*) FROM samples WHERE split_type_wf = 2").fetchone()[0]
        con.close()
        force_print(f"[INFO] Fold {fold_id}: test split_type_wf=2 に {n_test:,} 件を割り当て")

        loader = create_db_dataloader(
            db_path=db_path, split_type=2,       # test（split_type_wf=2）
            batch_size=config.BATCH_SIZE, shuffle=True,
            max_cooccurrence=config.MAX_CO_OCCURRENCE,
        )
        _, _, n = aggregate_attention(model, loader, device, args.n_batches, attn_sum, occ_count)
        total_samples += n
        used_folds.append(fold_id)
        del model

    out_dir = X.make_output_dir('attention_walk_forward', args.output_dir)
    title = (f'Walk-forward co-occurrence attention by month × NucPos '
             f'(top {args.top_n}, value={args.value}, folds={used_folds})')
    build_matrix_and_plot(
        attn_sum, occ_count, args.top_n, args.value, out_dir, title,
        meta_extra={'mode': 'walk_forward', 'folds': used_folds,
                    'n_samples': total_samples, 'n_batches_per_fold': args.n_batches,
                    'walk_forward_dir': args.walk_forward_dir})


def run_single(args):
    model, loader, device = X.load_model_and_loader(
        args.checkpoint, split=args.split, force_cpu=args.force_cpu)

    if args.include_duplicates or args.full_timeline:
        if args.include_duplicates:
            config.USE_UNIQUE_FILTER = False
            config.USE_TRAIN_ENTROPY_FILTER = False
            force_print("[INFO] include_duplicates: USE_UNIQUE_FILTER / USE_TRAIN_ENTROPY_FILTER を無効化")
        split_col_override = 'split_type' if args.full_timeline else None
        if args.full_timeline:
            force_print("[INFO] full_timeline: split_type(timestep分割列)を強制し全期間を流す")
        from transformer_260702.db.connection import get_db_path
        from transformer_260702.db.dataset import create_db_dataloader
        split_map = {'train': 0, 'val': 1, 'test': 2}
        loader = create_db_dataloader(
            db_path=get_db_path(), split_type=split_map[args.split],
            batch_size=config.BATCH_SIZE, shuffle=bool(args.full_timeline),
            max_cooccurrence=config.MAX_CO_OCCURRENCE, split_col_override=split_col_override,
        )

    if getattr(model, 'use_flat_coattn', False):
        raise SystemExit("[ERROR] このチェックポイントは USE_FLAT_COATTN=True で Co-occurrence "
                         "Attention を持たないため attention 分布を出力できません。")

    attn_sum, occ_count, n = aggregate_attention(model, loader, device, args.n_batches)
    out_dir = X.make_output_dir('attention_by_month', args.output_dir)
    title = (f'Co-occurrence attention by month × NucPos '
             f'(top {args.top_n} positions, value={args.value})')
    build_matrix_and_plot(
        attn_sum, occ_count, args.top_n, args.value, out_dir, title,
        meta_extra={'checkpoint': args.checkpoint, 'split': args.split,
                    'n_samples': n, 'n_batches': args.n_batches})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', help='単一チェックポイントモード')
    ap.add_argument('--walk_forward_dir',
                    help='walk_forward 日付別モード。fold_N/*/models/best_model.pth を含む親ディレクトリ')
    ap.add_argument('--folds', nargs='+', type=int, default=None,
                    help='walk_forward モードで対象フォールドを限定（省略時は全フォールド）')
    ap.add_argument('--split', default='test', choices=['train', 'val', 'test'])
    ap.add_argument('--n_batches', type=int, default=200, help='流すバッチ数（-1 で全件、wf は各フォールド毎）')
    ap.add_argument('--top_n', type=int, default=100, help='表示する上位位置数（出現回数順で選択）')
    ap.add_argument('--value', default='mean', choices=['mean', 'mass'],
                    help="mean=平均注目重み / mass=注目重み総和")
    ap.add_argument('--force_cpu', action='store_true')
    ap.add_argument('--include_duplicates', action='store_true',
                    help="USE_UNIQUE_FILTER/エントロピーフィルタを無効化し全サンプルを流す")
    ap.add_argument('--full_timeline', action='store_true',
                    help="単一モードで timestep 分割列を強制し全期間を流す（fold モデルは外挿）")
    ap.add_argument('--output_dir', default=None)
    args = ap.parse_args()

    if args.walk_forward_dir:
        run_walk_forward(args)
    elif args.checkpoint:
        run_single(args)
    else:
        raise SystemExit("[ERROR] --checkpoint か --walk_forward_dir のどちらかを指定してください。")


if __name__ == '__main__':
    main()
