"""B2 + 層C基礎: プリフライト検査（preflight_check.py）を合成DBで検証する。"""
import json

import duckdb
import pytest

from transformer.scripts.inspect import preflight_check as pf
from conftest import SYNTH_SAMPLES, build_synthetic_db


@pytest.fixture
def con(tmp_path):
    path = str(tmp_path / 'p.duckdb')
    build_synthetic_db(path)
    c = duckdb.connect(path)
    yield c
    c.close()


def test_cross_split_duplicate_detected(con):
    r = pf.check_cross_split_duplicates(con)       # sample7(test) は sample4(train) と raw_path 一致
    assert r['n_test'] == 2 and r['n_test_with_train_duplicate'] == 1
    assert r['duplicate_ratio'] == pytest.approx(0.5) and r['status'] == 'warn'


def test_no_cross_split_duplicate_is_pass(tmp_path):
    path = str(tmp_path / 'nd.duckdb')
    samples = [s for s in SYNTH_SAMPLES if s[0] != 7]
    build_synthetic_db(path, samples)
    r = pf.check_cross_split_duplicates(duckdb.connect(path))
    assert r['n_test_with_train_duplicate'] == 0 and r['status'] == 'pass'


def test_collection_date_check_flags_malformed(tmp_path):
    path = str(tmp_path / 'd.duckdb')
    samples = [(1, 1, 'A1T>C2G>T3A', '2021-03-01', 0, [(1, 1, 1, 1, 0)]),
               (2, 1, 'A2T>C2G>T3A', '2022/2024', 0, [(1, 1, 1, 1, 0)]),
               (3, 1, 'A3T>C2G>T3A', '2021', 0, [(1, 1, 1, 1, 0)]),
               (4, 1, 'A4T>C2G>T3A', None, 0, [(1, 1, 1, 1, 0)])]
    build_synthetic_db(path, samples)
    r = pf.check_collection_dates(duckdb.connect(path))
    assert r['status'] == 'warn' and r['malformed'] == 1 and r['null_or_empty'] == 1   # 割当側で除外済みのためwarn
    assert r['malformed_examples'] == ['2022/2024']


def test_collection_date_check_passes_for_valid_formats(con):
    assert pf.check_collection_dates(con)['status'] == 'pass'


def test_raw_path_truncation_detected_when_config_says_untruncated(tmp_path):
    path = str(tmp_path / 't.duckdb')
    long_path = ('A1T,' * 125)[:500]                 # ちょうど500文字
    samples = [(i, 1, long_path + str(i), '2021-03-01', 0, [(1, 1, 1, 1, 0)]) for i in range(1, 4)]
    samples = [(i, 1, long_path, '2021-03-01', 0, [(1, 1, 1, 1, 0)]) for i in range(1, 4)]
    build_synthetic_db(path, samples)
    c = duckdb.connect(path)
    assert pf.check_raw_path_truncation(c, truncate_len=None)['status'] == 'fail'
    assert pf.check_raw_path_truncation(c, truncate_len=500)['status'] == 'pass'   # 設定で意図した切り詰め


def test_raw_path_truncation_passes_for_normal_paths(con):
    assert pf.check_raw_path_truncation(con, None)['status'] == 'pass'


def test_missing_labels_and_features_detected(con):
    con.execute("DELETE FROM labels WHERE sample_id = 3")
    con.execute("DELETE FROM features WHERE sample_id = 4")
    r = pf.check_labels_and_features(con)
    assert r['status'] == 'fail' and r['samples_without_labels'] == 1 and r['samples_without_features'] == 1


def test_run_preflight_overall_and_json_roundtrip(con, tmp_path):
    res = pf.run_preflight(con, None, None)
    assert res['overall'] == 'warn'                  # 重複(warn)のみ、failなし
    out = tmp_path / 'stage_verification.json'
    out.write_text(json.dumps(res, default=str))
    assert json.loads(out.read_text())['overall'] == 'warn'


def test_expected_wf_split_reference_cases():
    f = pf.expected_wf_split
    assert f('2021-03-01', None, '2021-07-01', '2022-01-01') == 0
    assert f('2021-07-01', None, '2021-07-01', '2022-01-01') == 2
    assert f('2022-01-01', None, '2021-07-01', '2022-01-01') == -1
    assert f('2021-03-01', '2021-01-01', '2021-07-01', None) == 0
    assert f('2020-12-31', '2021-01-01', '2021-07-01', None) == -1
    assert f(None, None, '2021-07-01', None) == 0 and f('', '2021-01-01', '2021-07-01', None) == -1


def test_cross_split_duplicates_not_evaluable_when_a_split_is_empty(con):
    """軽量版 assign_fold_test_window 直後（trainが空）に「重複0件=問題なし」と誤認しない。"""
    con.execute("UPDATE samples SET split_type_wf = -1 WHERE split_type_wf IN (0, 1)")
    r = pf.check_cross_split_duplicates(con)
    assert r['status'] == 'warn' and r['n_train'] == 0 and 'note' in r


def test_expected_wf_split_excludes_malformed_in_every_fold():
    f = pf.expected_wf_split
    for fold in [(None, '2021-07-01', '2022-01-01'), ('2022-07-01', '2023-01-01', '2023-07-01'),
                 ('2022-01-01', '2022-07-01', '2023-01-01')]:
        assert f('2022/2024', *fold) == -1
