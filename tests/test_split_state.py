"""C4: 共有DBの split 割当の「状態」を記録し、読む側が矛盾を検知する。

再現した問題（2026-10-08）: 軽量割当(assign_fold_test_window)はtest以外を-1にする。その直後にtrainを読むと0件になり、
学習・データストア構築が黙って空のまま進んだ。別foldの割当が残っていても同様に気付けなかった。
"""
import duckdb
import pytest

from transformer_261008 import config
from transformer_261008.db.dataset import create_db_dataloader
from transformer_261008.db.queries import (StaleSplitError, assign_wf_splits, check_split_state,
                                           read_split_state)
from transformer_261008.scripts.analysis.xai import _xai_common as X
from transformer_261008.scripts.inspect.preflight_check import check_split_state_info

FOLD_A = (None, '2021-07-01', '2021-12-01')         # train: 2021-03〜05(ids1-6) / test: 2021-08,09(ids7,8)
FOLD_B = ('2021-04-01', '2021-07-01', '2021-12-01')


def _set_window(cfg, w):
    cfg(WALK_FORWARD_TRAIN_START=w[0], TEMPORAL_SPLIT_DATE=w[1], TEMPORAL_SPLIT_TEST_END=w[2], DATE_VALID_RATIO=0.0)


def _full(db, cfg, w):
    _set_window(cfg, w)
    con = duckdb.connect(db)
    assign_wf_splits(con)
    con.close()


def _loader(db, split):
    return create_db_dataloader(db, split, 4, max_cooccurrence=20, num_workers_override=0)


def _state(db):
    con = duckdb.connect(db)
    try:
        return read_split_state(con)
    finally:
        con.close()


def test_full_assignment_records_state(synthetic_db, cfg):
    _full(synthetic_db, cfg, FOLD_A)
    st = _state(synthetic_db)
    assert st['kind'] == 'full' and (st['train_start'], st['split_date'], st['split_end']) == FOLD_A
    assert (st['n_train'], st['n_valid'], st['n_test']) == (6, 0, 2)


def test_light_assignment_records_test_only(synthetic_db, cfg):
    _set_window(cfg, FOLD_A)
    X.assign_fold_test_window(synthetic_db, *FOLD_A)
    st = _state(synthetic_db)
    assert st['kind'] == 'test_only' and st['n_test'] == 2 and st['n_train'] == 0


def test_reading_train_after_light_assignment_is_rejected_with_a_clear_message(synthetic_db, cfg):
    _set_window(cfg, FOLD_A)
    X.assign_fold_test_window(synthetic_db, *FOLD_A)
    with pytest.raises(StaleSplitError, match='test のみ'):
        _loader(synthetic_db, 0)
    with pytest.raises(StaleSplitError):
        _loader(synthetic_db, 1)
    assert len(_loader(synthetic_db, 2).dataset) == 2           # testは読める


def test_full_assignment_after_light_makes_train_readable_again(synthetic_db, cfg):
    _set_window(cfg, FOLD_A)
    X.assign_fold_test_window(synthetic_db, *FOLD_A)
    _full(synthetic_db, cfg, FOLD_A)
    assert len(_loader(synthetic_db, 0).dataset) == 3           # 6サンプル→3グループ
    assert _state(synthetic_db)['kind'] == 'full'


def test_window_mismatch_between_config_and_assignment_is_rejected(synthetic_db, cfg):
    _full(synthetic_db, cfg, FOLD_A)
    _set_window(cfg, FOLD_B)                                    # 設定だけ別foldへ（DBは割当し直していない）
    with pytest.raises(StaleSplitError, match='一致しません'):
        _loader(synthetic_db, 0)
    with pytest.raises(StaleSplitError):
        _loader(synthetic_db, 2)
    _full(synthetic_db, cfg, FOLD_B)                            # 割り当て直せば読める
    assert len(_loader(synthetic_db, 2).dataset) == 2


def test_window_mismatch_is_not_checked_for_test_only_assignments(synthetic_db, cfg):
    """分析用の軽量割当では窓の一致は見ない（割当→checkpoint読込で窓が上書き→再割当、の既存パターンのため）。"""
    _set_window(cfg, FOLD_A)
    X.assign_fold_test_window(synthetic_db, *FOLD_A)
    _set_window(cfg, FOLD_B)
    assert len(_loader(synthetic_db, 2).dataset) == 2


def test_no_recorded_state_is_not_checked(synthetic_db):
    """旧DB・手組みの合成DB（記録なし）は検査しない（不明）。"""
    assert _state(synthetic_db) is None
    assert len(_loader(synthetic_db, 0).dataset) == 3


def test_only_walk_forward_column_is_checked(synthetic_db, cfg):
    _full(synthetic_db, cfg, FOLD_A)
    _set_window(cfg, FOLD_B)
    check_split_state(synthetic_db, 0, 'split_type_date')       # 他の列は対象外


def test_state_has_a_single_row_per_column(synthetic_db, cfg):
    _full(synthetic_db, cfg, FOLD_A)
    _full(synthetic_db, cfg, FOLD_B)
    con = duckdb.connect(synthetic_db)
    assert con.execute("SELECT COUNT(*) FROM split_state").fetchone()[0] == 1
    con.close()
    assert _state(synthetic_db)['train_start'] == FOLD_B[0]


def test_preflight_reports_split_state(synthetic_db, cfg):
    con = duckdb.connect(synthetic_db)
    assert check_split_state_info(con)['status'] == 'warn'      # 記録なし
    con.close()
    _full(synthetic_db, cfg, FOLD_A)
    con = duckdb.connect(synthetic_db)
    r = check_split_state_info(con)
    con.close()
    assert r['status'] == 'pass' and r['kind'] == 'full' and r['n_test'] == 2
    _set_window(cfg, FOLD_A)
    X.assign_fold_test_window(synthetic_db, *FOLD_A)
    con = duckdb.connect(synthetic_db)
    r = check_split_state_info(con)
    con.close()
    assert r['status'] == 'warn' and 'test専用' in r['note']
