# --- transformer_261008/scripts/inspect/preflight_check.py ---
"""学習・walk_forward 起動前のプリフライト検査（DB上で数秒〜数分、読み取り専用）。

過去の「クラッシュせずもっともらしい値を出し続ける静かなバグ」（raw_path切り詰め、split境界の誤り、
train/test間の重複リーク、collection_dateの不正値）を、実行前に機械的に検知する。
結果は stage_verification.json に書き出せる（各チェックは pass / warn / fail）。

各 check_* は duckdb 接続を受け取る純粋な読み取り関数で、合成DBでもテストできる
（tests/test_preflight_checks.py）。実DB向けの低速テストは tests/real_data/（pytest -m slow）。

Usage:
  python -m transformer_261008.scripts.inspect.preflight_check [--fold N] [--output stage_verification.json]
注意: DuckDBは別プロセスが書き込みで開いていると接続できない。学習/評価の実行中は使えない。
"""
import argparse
import json
import re

# 日付は 'YYYY' / 'YYYY-MM' / 'YYYY-MM-DD' のいずれか（assign_wf_splits は RPAD で '-01-01' 補完する前提）。
# 判定は split割当と同一の db/queries.py::VALID_DATE_REGEX を使う（ずれると検査が意味を失う）。
from transformer_261008.db.queries import VALID_DATE_REGEX as _DATE_RE
PASS, WARN, FAIL = 'pass', 'warn', 'fail'


def rpad_date(s):
    """assign_wf_splits の SQL `RPAD(collection_date, 10, '-01-01')` と同じ補完。"""
    return (s + '-01-01' * 3)[:10] if len(s) < 10 else s[:10]


def expected_wf_split(collection_date, train_start, split_date, split_end):
    """assign_wf_splits の日付規則の参照実装（独立実装による差分テスト用）。

    戻り値: 0(train or valid) / 2(test) / -1(除外)。valid(1)は0の部分集合として扱う。
    NULL/空日付は Fold1(train_start=None)のみ train、Fold N は除外。
    不正な日付形式（'2022/2024' 等）はどのfoldでも除外(-1)。
    """
    if collection_date is None or collection_date == '':
        return 0 if train_start is None else -1
    if not re.match(_DATE_RE, collection_date):
        return -1
    d = rpad_date(collection_date)
    if d >= split_date and (split_end is None or d < split_end):
        return 2
    if train_start is None:
        return 0 if d < split_date else -1
    return 0 if (train_start <= d < split_date) else -1


def check_collection_dates(con):
    """collection_date の NULL/空/不正形式の件数（'2022/2024' 型。data_quality_collection_date_bug）。

    不正形式はsplit割当から除外される（db/queries.py::valid_date_sql）。データ自体の存在を知らせるwarn。"""
    total, n_null, n_bad = con.execute(f"""
        SELECT COUNT(*),
               SUM(CASE WHEN collection_date IS NULL OR collection_date = '' THEN 1 ELSE 0 END),
               SUM(CASE WHEN collection_date IS NOT NULL AND collection_date != ''
                         AND NOT regexp_matches(collection_date, '{_DATE_RE}') THEN 1 ELSE 0 END)
        FROM samples""").fetchone()
    n_null, n_bad = int(n_null or 0), int(n_bad or 0)
    examples = [r[0] for r in con.execute(f"""
        SELECT DISTINCT collection_date FROM samples
        WHERE collection_date IS NOT NULL AND collection_date != ''
          AND NOT regexp_matches(collection_date, '{_DATE_RE}') LIMIT 5""").fetchall()]
    # 不正形式の行は assign_wf_splits 等のsplit割当から除外される（valid_date_sql）ため warn。
    # 割当側のガードが外れていないかは check_wf_assignment が検知する。
    return {'status': WARN if n_bad else PASS, 'total': total, 'null_or_empty': n_null,
            'malformed': n_bad, 'malformed_examples': examples,
            'note': 'split割当からは除外済み' if n_bad else ''}


def check_wf_assignment(con, train_start, split_date, split_end):
    """DB上の split_type_wf が、日付規則の参照実装と全サンプルで一致するか（split割当の検証）。"""
    rows = con.execute("SELECT sample_id, collection_date, split_type_wf FROM samples").fetchall()
    mismatches = []
    for sid, cd, got in rows:
        exp = expected_wf_split(cd, train_start, split_date, split_end)
        got_norm = 0 if got == 1 else got           # valid(1) は train(0) の部分集合
        if got_norm != exp:
            mismatches.append((sid, cd, got, exp))
    return {'status': FAIL if mismatches else PASS, 'n_samples': len(rows),
            'n_mismatch': len(mismatches), 'examples': mismatches[:5]}


def check_cross_split_duplicates(con, split_col='split_type_wf'):
    """train/valid と test に**完全一致の raw_path** が跨っている件数（リーク検知の土台）。

    PETRA型（CLM設計）ではこの一致が機械的な暗記リークになる。本体側も要確認。
    情報提供が目的のため status は pass/warn（重複率>0で warn）。
    """
    n_test, n_dup = con.execute(f"""
        WITH tr AS (SELECT DISTINCT raw_path FROM samples WHERE {split_col} IN (0, 1)),
             te AS (SELECT raw_path FROM samples WHERE {split_col} = 2)
        SELECT (SELECT COUNT(*) FROM te),
               (SELECT COUNT(*) FROM te WHERE raw_path IN (SELECT raw_path FROM tr))""").fetchone()
    ratio = (n_dup / n_test) if n_test else 0.0
    n_train = con.execute(f"SELECT COUNT(*) FROM samples WHERE {split_col} IN (0, 1)").fetchone()[0]
    if n_train == 0 or n_test == 0:
        # splitが割り当てられていない（例: 軽量版 assign_fold_test_window の直後はtestのみ）。
        # 重複0件が「問題なし」ではなく「評価できていない」ことを明示する。
        return {'status': WARN, 'n_train': n_train, 'n_test': n_test, 'n_test_with_train_duplicate': n_dup,
                'duplicate_ratio': ratio,
                'note': 'train/test のいずれかが空のため評価不能。assign_wf_splits 後に実行すること'}
    return {'status': WARN if n_dup else PASS, 'n_train': n_train, 'n_test': n_test,
            'n_test_with_train_duplicate': n_dup, 'duplicate_ratio': ratio}


def check_raw_path_truncation(con, truncate_len=None, historical_len=500, warn_ratio=0.01):
    """raw_path が切り詰められていないか（2026-07-10 のバグ: 先頭500文字に切り詰め）。

    truncate_len(config.RAW_PATH_TRUNCATE_LEN) が None（切り詰め無し）なのに、長さがちょうど
    historical_len(500) の行が warn_ratio を超えていれば fail。設定値がある場合は情報のみ。
    """
    total, at_hist = con.execute(
        f"SELECT COUNT(*), SUM(CASE WHEN LENGTH(raw_path) = {int(historical_len)} THEN 1 ELSE 0 END) FROM samples"
    ).fetchone()
    at_hist = int(at_hist or 0)
    ratio = (at_hist / total) if total else 0.0
    status = FAIL if (truncate_len is None and ratio > warn_ratio) else PASS
    return {'status': status, 'total': total, 'rows_at_historical_len': at_hist,
            'ratio': ratio, 'config_truncate_len': truncate_len}


def check_labels_and_features(con):
    """全サンプルにラベル行・特徴量行が存在するか。"""
    n_no_label = con.execute("""SELECT COUNT(*) FROM samples s
        WHERE NOT EXISTS (SELECT 1 FROM labels l WHERE l.sample_id = s.sample_id)""").fetchone()[0]
    n_no_feat = con.execute("""SELECT COUNT(*) FROM samples s
        WHERE NOT EXISTS (SELECT 1 FROM features f WHERE f.sample_id = s.sample_id)""").fetchone()[0]
    return {'status': FAIL if (n_no_label or n_no_feat) else PASS,
            'samples_without_labels': n_no_label, 'samples_without_features': n_no_feat}


def run_preflight(con, fold_window=None, truncate_len=None):
    """全チェックを実行して {stage名: 結果} を返す。fold_window=(train_start, split_date, split_end)。"""
    out = {
        'stage_1_raw_path_integrity': check_raw_path_truncation(con, truncate_len),
        'stage_1_collection_dates': check_collection_dates(con),
        'stage_1_labels_features': check_labels_and_features(con),
        'stage_2_cross_split_duplicates': check_cross_split_duplicates(con),
    }
    if fold_window is not None:
        out['stage_2_wf_assignment'] = check_wf_assignment(con, *fold_window)
    order = {PASS: 0, WARN: 1, FAIL: 2}
    out['overall'] = max((v['status'] for v in out.values()), key=lambda s: order[s])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fold', type=int, default=None,
                    help='指定するとそのfoldの日付規則と現在の split_type_wf を突合する（割当済みであること）')
    ap.add_argument('--output', default='stage_verification.json')
    args = ap.parse_args()

    from transformer_261008 import config
    from transformer_261008.db.connection import connect_db, get_db_path
    window = None
    if args.fold is not None:
        from transformer_261008.scripts.analysis.xai import _xai_common as X
        window = X.get_fold_windows()[args.fold]
    con = connect_db(get_db_path(), read_only=True)
    result = run_preflight(con, window, getattr(config, 'RAW_PATH_TRUNCATE_LEN', None))
    con.close()
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    for k, v in result.items():
        print(f"{k}: {v['status'] if isinstance(v, dict) else v}")
    print(f"saved: {args.output}")
    raise SystemExit(1 if result['overall'] == FAIL else 0)


if __name__ == '__main__':
    main()
