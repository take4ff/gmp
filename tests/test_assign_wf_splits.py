"""B1: assign_wf_splits（walk_forwardのsplit割当）を合成DuckDBで検証。

検証: 日付境界、RPADによる不完全日付、NULL/空日付（Fold1はtrain・Fold Nは除外）、
train/test非重複、valid比率、参照実装(expected_wf_split)との全件一致。
"""
import duckdb
import pytest

from transformer_261008 import config
from transformer_261008.db.queries import assign_wf_splits
from transformer_261008.scripts.inspect.preflight_check import expected_wf_split, check_wf_assignment
from conftest import build_synthetic_db

DATES = ['2020-12-31', '2021-01-01', '2021-06-30', '2021-07-01', '2021-12-31', '2022-01-01',
         '2021', '2021-07', '2022', '', None, '2022/2024', '2021/07', '2021-7-1', 'abc']


def _db(tmp_path, dates):
    path = str(tmp_path / 'wf.duckdb')
    samples = [(i + 1, 1, f'A{i + 1}T>C2G>T3A', d, 0, [(1, 100, 10, 1, 0)]) for i, d in enumerate(dates)]
    build_synthetic_db(path, samples)
    return duckdb.connect(path)


def _assign(con, cfg, train_start, split_date, split_end, valid_ratio=0.0):
    cfg(WALK_FORWARD_TRAIN_START=train_start, TEMPORAL_SPLIT_DATE=split_date,
        TEMPORAL_SPLIT_TEST_END=split_end, DATE_VALID_RATIO=valid_ratio)
    assign_wf_splits(con)
    return dict(con.execute("SELECT sample_id, split_type_wf FROM samples").fetchall())


@pytest.mark.parametrize('fold', [
    (None, '2021-07-01', '2022-01-01'),          # Fold1相当（train_start なし）
    ('2021-01-01', '2021-07-01', '2022-01-01'),  # FoldN
    ('2021-07-01', '2022-01-01', None),          # 最終fold（split_end なし）
])
def test_matches_reference_implementation_for_every_sample(tmp_path, cfg, fold):
    con = _db(tmp_path, DATES)
    _assign(con, cfg, *fold)
    res = check_wf_assignment(con, *fold)
    assert res['status'] == 'pass', res


def test_boundaries_are_half_open(tmp_path, cfg):
    con = _db(tmp_path, ['2021-06-30', '2021-07-01', '2021-12-31', '2022-01-01'])
    got = _assign(con, cfg, '2021-01-01', '2021-07-01', '2022-01-01')
    assert [got[i] for i in (1, 2, 3, 4)] == [0, 2, 2, -1]   # train<split_date<=test<split_end


def test_partial_dates_are_padded_with_first_of_year_month(tmp_path, cfg):
    con = _db(tmp_path, ['2021', '2021-07', '2021-06'])
    got = _assign(con, cfg, '2021-01-01', '2021-07-01', '2022-01-01')
    # '2021'→2021-01-01(train), '2021-07'→2021-07-01(test), '2021-06'→2021-06-01(train)
    assert [got[1], got[2], got[3]] == [0, 2, 0]


def test_null_and_empty_dates_train_in_fold1_only(tmp_path, cfg):
    con = _db(tmp_path, [None, '', '2021-03-01'])
    got1 = _assign(con, cfg, None, '2021-07-01', '2022-01-01')
    assert [got1[1], got1[2], got1[3]] == [0, 0, 0]
    got2 = _assign(con, cfg, '2021-01-01', '2021-07-01', '2022-01-01')
    assert [got2[1], got2[2], got2[3]] == [-1, -1, 0]


def test_train_and_test_never_overlap_and_valid_is_subset_of_train(tmp_path, cfg):
    con = _db(tmp_path, [f'2021-{m:02d}-15' for m in range(1, 13)] * 5)
    _assign(con, cfg, '2021-01-01', '2021-07-01', '2022-01-01', valid_ratio=0.2)
    rows = con.execute("SELECT split_type_wf, MIN(collection_date), MAX(collection_date), COUNT(*)"
                       " FROM samples GROUP BY 1").fetchall()
    by = {r[0]: r for r in rows}
    assert max(by[0][2], by[1][2]) < min(by[2][1], '9999')        # train/valid の最大日 < test の最小日
    n_train_valid = by[0][3] + by[1][3]
    assert by[1][3] == int(n_train_valid * 0.2)                    # valid比率
    assert sum(r[3] for r in rows) == 60


def test_reassignment_overwrites_previous_fold(tmp_path, cfg):
    con = _db(tmp_path, ['2021-03-01', '2021-09-01'])
    a = _assign(con, cfg, None, '2021-07-01', '2022-01-01')
    b = _assign(con, cfg, '2021-07-01', '2022-01-01', None)
    assert (a[1], a[2]) == (0, 2)
    assert (b[1], b[2]) == (-1, 0)           # 前foldの割当が残らない


@pytest.mark.parametrize('fold', [
    (None, '2021-01-01', '2021-07-01'),          # Fold1相当
    ('2022-07-01', '2023-01-01', '2023-07-01'),  # '2022/2024' が文字列比較で入り込んでいたfold(train)
    ('2022-01-01', '2022-07-01', '2023-01-01'),  # 同(test, fold4相当)
    ('2021-07-01', '2022-01-01', None),
])
# 文字列比較で日付範囲より「後ろ」に並ぶ値と「前」に並ぶ値（'2020/12'等。Fold1のtrain側ガードの検証に必要）の両方
@pytest.mark.parametrize('bad', ['2022/2024', '2021/07', '2021-7-1', 'abc', '2022-07-1x',
                                 '2020/12', '2019/2020', '2020-1-1'])
def test_malformed_date_is_excluded_in_every_fold(tmp_path, cfg, fold, bad):
    """不正な日付形式は、文字列比較でどのfoldの日付範囲に入ってもtrain/test/validに割り当てない。
    過去の問題: '2022/2024'（217件）が '/' > '-' の文字コード差でfold4のtestに混入（2026-07）。"""
    con = _db(tmp_path, [bad])
    got = _assign(con, cfg, *fold)
    assert got[1] == -1


def test_date_mode_excludes_malformed_dates(tmp_path, cfg):
    """SPLIT_MODE='date' の assign_date_splits でも不正形式は -1（既定のtrain扱いにしない）。"""
    from transformer_261008.db.queries import assign_date_splits
    con = _db(tmp_path, ['2021-03-01', '2022/2024', '2023-05-01', None])
    cfg(TEMPORAL_SPLIT_DATE='2022-01-01', TEMPORAL_SPLIT_TEST_END=None, DATE_VALID_RATIO=0.0)
    assign_date_splits(con)
    got = dict(con.execute("SELECT sample_id, split_type_date FROM samples").fetchall())
    assert got == {1: 0, 2: -1, 3: 2, 4: 0}      # 不正→-1、NULL→train（従来仕様）


def test_light_assign_fold_test_window_matches_full_assignment(tmp_path, cfg):
    """分析用の軽量割当（assign_fold_test_window）も、testの集合が assign_wf_splits と一致する
    （不正日付を含めて）。ずれると分析のtestが学習時のtestと食い違う。"""
    from transformer_261008.scripts.analysis.xai import _xai_common as X
    path = str(tmp_path / 'light.duckdb')
    samples = [(i + 1, 1, f'A{i + 1}T>C2G>T3A', d, 0, [(1, 100, 10, 1, 0)]) for i, d in enumerate(DATES)]
    build_synthetic_db(path, samples)
    fold = ('2021-01-01', '2021-07-01', '2022-01-01')
    con = duckdb.connect(path)
    full = _assign(con, cfg, *fold)
    con.close()
    X.assign_fold_test_window(path, *fold)
    con = duckdb.connect(path)
    light = dict(con.execute("SELECT sample_id, split_type_wf FROM samples").fetchall())
    test_full = {k for k, v in full.items() if v == 2}
    test_light = {k for k, v in light.items() if v == 2}
    assert test_light == test_full
    assert 12 not in test_light                  # sample_id=12 は '2022/2024'


def test_valid_date_sql_accepts_only_expected_formats():
    from transformer_261008.db.queries import valid_date_sql
    con = duckdb.connect()
    ok = ['2021', '2021-07', '2021-07-01']
    ng = ['2022/2024', '2021/07', '2021-7-1', 'abc', '', '2021-07-01T00', ' 2021']
    for v in ok:
        assert con.execute(f"SELECT {valid_date_sql(repr(v))}").fetchone()[0] is True, v
    for v in ng:
        assert con.execute(f"SELECT {valid_date_sql(repr(v))}").fetchone()[0] is False, v
    assert con.execute(f"SELECT {valid_date_sql('NULL::VARCHAR')}").fetchone()[0] is None   # NULLは偽扱い(WHEREで除外)
