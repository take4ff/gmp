# --- transformer_260707/scripts/analysis/verify/verify_batch_local_grouping.py ---
"""本体 evaluate.py の any-of-set 判定が、実際のバッチ境界内でどこまで「真の分岐グループ」を
捕捉できているかを検証する（読み取り専用、DB書き込みなし）。

【既知バグは修正済み】本スクリプトが発見した以下の問題（バッチ局所グルーピング）は
db/dataset.py 側で修正済み（グローバルグルーピングに変更、collate_fn の group_map
ロジックは削除）。本スクリプトは修正前の挙動の記録・回帰検証用として残している。

背景（修正前の状態）:
  db/dataset.py の collate_fn は同一 input_path_str（分岐親履歴）を持つサンプルを
  「同一ミニバッチ内でのみ」グループ化し、any-of-set の正解集合（raw_y）を作っていた
  （group_map はバッチ単位で構築されていた）。一方 petra 側の比較用グルーピング
  （petra/eval/plot_monthly_position_tolerance.py の resolve_test_groups_by_daterange）は
  テスト期間全体を1回のSQLでスキャンし、真にグローバルな分岐グループを作る。

  本体の実際のテストDataLoaderは shuffle=False・ORDER BY sample_id・batch_size=512
  (main.py: test_loader = _make_loader(2, shuffle=False)) で連続サンプルを切り出すだけなので、
  真のグローバル分岐グループの兄弟サンプルが512件のバッチ窓を跨いで散らばっていれば、
  評価時に部分的にしか（最悪 K=1 として、分岐なしと同じ扱いで）合流しない可能性がある。

  本スクリプトはこれを実際のDBデータで直接測定する: 各walk-forward foldのテスト期間について
  (1) 真のグローバル分岐グループ（input_path_str単位、全メンバー）
  (2) 本体と全く同じ順序・バッチサイズでの「バッチ局所グループ」
  を両方計算し、真に分岐している(K_true>=2)サンプルのうち何%が
    - fully:     バッチ局所でも全メンバーが揃った(K_obs == K_true)
    - partial:   一部だけ揃った(1 < K_obs < K_true)
    - fragmented: バッチが割れて単独評価になった(K_obs == 1、分岐なしと区別不可能)
  になるかを集計する。

Usage:
  python -m transformer_260707.scripts.analysis.verify.verify_batch_local_grouping
  python -m transformer_260707.scripts.analysis.verify.verify_batch_local_grouping --folds 2 6
"""
import argparse
import random
import re
from collections import defaultdict

from transformer_260707 import config
from transformer_260707.db.connection import get_db_path, connect_db
from transformer_260707.scripts.eval.walk_forward import FOLDS

_MUT_POS_RE = re.compile(r'[A-Za-z](\d+)[A-Za-z]')
_YEARMONTH_RE = re.compile(r'^\d{4}-\d{2}$')


def _target_positions_from_raw_path(raw_path):
    last_step = raw_path.split('>')[-1]
    positions = set()
    for mut in last_step.split(','):
        m = _MUT_POS_RE.search(mut.strip())
        if m:
            positions.add(int(m.group(1)))
    return positions


def fetch_test_rows(con, split_date, split_end):
    """本体 db/dataset.py._get_sample_ids と同じフィルタ(max_cooccurrence,
    EVAL_MAX_Y_CO_OCCURRENCE)・同じ ORDER BY sample_id で、日付範囲直接指定によりテスト行を取得する
    （split_type_wf 列には一切触れない・書き込まない。他プロセスと並行実行安全）。
    """
    where = "collection_date IS NOT NULL AND collection_date != '' "
    where += "AND RPAD(collection_date, 10, '-01-01') >= ? "
    params = [split_date]
    if split_end:
        where += "AND RPAD(collection_date, 10, '-01-01') < ? "
        params.append(split_end)
    where += "AND s.max_cooccurrence <= ?"
    params.append(config.MAX_CO_OCCURRENCE)

    rows = con.execute(
        f"""
        SELECT s.sample_id, s.raw_path, l.targets
        FROM samples s JOIN labels l ON s.sample_id = l.sample_id
        WHERE {where}
        ORDER BY s.sample_id
        """,
        params,
    ).fetchall()

    eval_max_y_co = getattr(config, 'EVAL_MAX_Y_CO_OCCURRENCE', None)
    if eval_max_y_co is None:
        return [(sid, rp) for sid, rp, _t in rows]

    import pickle
    out = []
    excluded = 0
    for sid, rp, targets_blob in rows:
        targets = pickle.loads(targets_blob)
        if len(targets) > eval_max_y_co:
            excluded += 1
            continue
        out.append((sid, rp))
    return out, excluded


def fetch_train_rows(con, train_start, split_date):
    """queries.py assign_wf_splits と同じ訓練ウィンドウ定義（EVAL_MAX_Y_CO_OCCURRENCEフィルタなし、
    train split には適用されないため）で訓練データを取得する。"""
    if train_start is None:
        where = ("(collection_date IS NULL OR collection_date = '' "
                 "OR RPAD(collection_date, 10, '-01-01') < ?)")
        params = [split_date]
    else:
        where = ("collection_date IS NOT NULL AND collection_date != '' "
                 "AND RPAD(collection_date, 10, '-01-01') >= ? "
                 "AND RPAD(collection_date, 10, '-01-01') < ?")
        params = [train_start, split_date]
    where += " AND s.max_cooccurrence <= ?"
    params.append(config.MAX_CO_OCCURRENCE)

    rows = con.execute(
        f"SELECT s.sample_id, s.raw_path FROM samples s WHERE {where} ORDER BY s.sample_id",
        params,
    ).fetchall()
    return rows


def compute_fragmentation(rows, batch_size, shuffle=False, seed=None):
    """rows: list of (sample_id, raw_path)。shuffle=True なら本体の学習時
    (dataset.py __iter__: random.shuffle(sample_ids) してから chunk_size/batch_size で
    連続バッチ化）と同じ手順を再現する。戻り値: (fully, partial, fragmented, target_size_ratios,
    n_branch_groups, n_branch_samples)。
    """
    parts_cache = {sid: raw_path.split('>') for sid, raw_path in rows}

    def input_path_str_of(sid):
        parts = parts_cache[sid]
        return '>'.join(parts[:-1]) if len(parts) > 1 else ''

    global_groups = defaultdict(list)
    for sid, _rp in rows:
        global_groups[input_path_str_of(sid)].append(sid)

    raw_path_by_id = dict(rows)
    global_target_positions = {}
    for ips, members in global_groups.items():
        pos = set()
        for sid in members:
            pos |= _target_positions_from_raw_path(raw_path_by_id[sid])
        global_target_positions[ips] = pos

    n_branch_groups = sum(1 for m in global_groups.values() if len(m) >= 2)
    n_branch_samples = sum(len(m) for m in global_groups.values() if len(m) >= 2)

    sample_ids_ordered = [sid for sid, _rp in rows]
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(sample_ids_ordered)

    fully = partial = fragmented = 0
    target_size_ratios = []

    for i in range(0, len(sample_ids_ordered), batch_size):
        batch_ids = sample_ids_ordered[i:i + batch_size]
        local_groups = defaultdict(list)
        for sid in batch_ids:
            local_groups[input_path_str_of(sid)].append(sid)

        for ips, local_members in local_groups.items():
            k_true = len(global_groups[ips])
            if k_true < 2:
                continue
            k_obs = len(local_members)

            local_pos = set()
            for sid in local_members:
                local_pos |= _target_positions_from_raw_path(raw_path_by_id[sid])
            true_pos = global_target_positions[ips]

            n_members_here = len(local_members)
            if k_obs == k_true:
                fully += n_members_here
            elif k_obs == 1:
                fragmented += n_members_here
            else:
                partial += n_members_here

            if true_pos:
                target_size_ratios.append(len(local_pos) / len(true_pos))

    return fully, partial, fragmented, target_size_ratios, n_branch_groups, n_branch_samples


def _print_fragmentation_result(label, n_total, batch_size, fully, partial, fragmented,
                                 target_size_ratios, n_branch_groups, n_branch_samples):
    total_branch_evals = fully + partial + fragmented
    print(f"  真のグローバル分岐グループ(K>=2): {n_branch_groups:,} groups "
          f"/ {n_branch_samples:,} samples ({100*n_branch_samples/n_total:.2f}% of {label})")
    if total_branch_evals == 0:
        print("  (分岐グループに属するサンプルなし)")
        return None
    avg_ratio = sum(target_size_ratios) / len(target_size_ratios) if target_size_ratios else float('nan')
    print(f"  [分岐サンプル(K_true>=2)の内訳、batch_size={batch_size}]")
    print(f"    fully captured  (K_obs==K_true) : {fully:>8,} ({100*fully/total_branch_evals:5.2f}%)")
    print(f"    partial         (1<K_obs<K_true): {partial:>8,} ({100*partial/total_branch_evals:5.2f}%)")
    print(f"    fragmented      (K_obs==1)      : {fragmented:>8,} ({100*fragmented/total_branch_evals:5.2f}%)")
    print(f"    平均 target集合サイズ比 (batch局所/真のグローバル): {avg_ratio:.3f}")
    return {
        'fully': fully, 'partial': partial, 'fragmented': fragmented,
        'avg_target_size_ratio': avg_ratio, 'n_branch_samples': n_branch_samples,
    }


def analyze_fold(con, fold_id, split_date, split_end, desc, batch_size):
    rows, excluded = fetch_test_rows(con, split_date, split_end)
    n = len(rows)
    if n == 0:
        print(f"=== Fold {fold_id} ({desc}): テストサンプルなし、スキップ ===")
        return None
    print(f"\n=== Fold {fold_id}: {desc}  [{split_date}, {split_end}) ===")
    print(f"  test n={n:,}（Y共起>{config.EVAL_MAX_Y_CO_OCCURRENCE}除外={excluded:,}）")

    fully, partial, fragmented, ratios, n_bg, n_bs = compute_fragmentation(
        rows, batch_size, shuffle=False)
    r = _print_fragmentation_result('test', n, batch_size, fully, partial, fragmented,
                                     ratios, n_bg, n_bs)
    if r is None:
        return None
    r.update({'fold': fold_id, 'desc': desc, 'n_test': n})
    return r


def analyze_fold_train_shuffle(con, fold_id, train_start, split_date, desc, batch_size, seed):
    rows = fetch_train_rows(con, train_start, split_date)
    n = len(rows)
    if n == 0:
        print(f"=== Fold {fold_id} ({desc}): 訓練サンプルなし、スキップ ===")
        return None
    print(f"\n=== Fold {fold_id}: {desc}  train window="
          f"[{train_start}, {split_date})  n={n:,} ===")

    print("  -- 順序通り(ORDER BY sample_id、参考) --")
    f_o, p_o, fr_o, r_o, nbg, nbs = compute_fragmentation(rows, batch_size, shuffle=False)
    _print_fragmentation_result('train', n, batch_size, f_o, p_o, fr_o, r_o, nbg, nbs)

    print("  -- 実際の学習時と同じ shuffle=True (dataset.py __iter__ の random.shuffle) --")
    f_s, p_s, fr_s, r_s, nbg2, nbs2 = compute_fragmentation(
        rows, batch_size, shuffle=True, seed=seed)
    result = _print_fragmentation_result('train', n, batch_size, f_s, p_s, fr_s, r_s, nbg2, nbs2)
    if result:
        result.update({'fold': fold_id, 'desc': desc, 'n_train': n})
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', type=int, nargs='+', default=None)
    ap.add_argument('--batch_size', type=int, default=config.BATCH_SIZE)
    ap.add_argument('--train_shuffle_check', action='store_true',
                    help='評価(test)側だけでなく、学習(train)側の実際のshuffle=Trueバッチ化も検証する')
    args = ap.parse_args()

    db_path = get_db_path()
    print(f"[INFO] DB: {db_path} (read_only), batch_size={args.batch_size}, "
          f"MAX_CO_OCCURRENCE={config.MAX_CO_OCCURRENCE}, "
          f"EVAL_MAX_Y_CO_OCCURRENCE={config.EVAL_MAX_Y_CO_OCCURRENCE}")
    con = connect_db(db_path, read_only=True)

    target_folds = set(args.folds) if args.folds else None
    summary = []
    train_summary = []
    for fold_id, train_start, split_date, split_end, desc in FOLDS:
        if target_folds and fold_id not in target_folds:
            continue
        r = analyze_fold(con, fold_id, split_date, split_end, desc, args.batch_size)
        if r:
            summary.append(r)
        if args.train_shuffle_check:
            rt = analyze_fold_train_shuffle(con, fold_id, train_start, split_date, desc,
                                             args.batch_size, seed=config.SEED)
            if rt:
                train_summary.append(rt)
    con.close()

    if summary:
        print("\n===== 全fold横断サマリ（評価=test, shuffle=False） =====")
        for r in summary:
            tot = r['fully'] + r['partial'] + r['fragmented']
            print(f"  fold{r['fold']} ({r['desc']:38s}): "
                  f"分岐サンプル={r['n_branch_samples']:>8,} / test={r['n_test']:>8,}  "
                  f"fragmented率={100*r['fragmented']/tot:5.2f}%  "
                  f"target集合サイズ比={r['avg_target_size_ratio']:.3f}")

    if train_summary:
        print("\n===== 全fold横断サマリ（学習=train, shuffle=True） =====")
        for r in train_summary:
            tot = r['fully'] + r['partial'] + r['fragmented']
            print(f"  fold{r['fold']} ({r['desc']:38s}): "
                  f"分岐サンプル={r['n_branch_samples']:>8,} / train={r['n_train']:>8,}  "
                  f"fragmented率={100*r['fragmented']/tot:5.2f}%  "
                  f"target集合サイズ比={r['avg_target_size_ratio']:.3f}")


if __name__ == '__main__':
    main()
