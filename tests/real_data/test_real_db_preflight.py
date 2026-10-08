"""層C: 実DBに対するプリフライト検査（低速・読み取り専用）。実行: python -m pytest -m slow

学習/評価プロセスが同じDBを開いている間は接続できないため、その場合は skip する。
"""
import duckdb
import pytest

from transformer_260817 import config
from transformer_260817.db.connection import connect_db, get_db_path
from transformer_260817.scripts.inspect import preflight_check as pf

pytestmark = pytest.mark.slow


@pytest.fixture(scope='module')
def real_con():
    import os
    path = get_db_path()
    if not os.path.exists(path):
        pytest.skip(f"実DBが無い: {path}")
    try:
        con = connect_db(path, read_only=True)
    except duckdb.Error as e:                       # 別プロセスが書き込みで開いている等
        pytest.skip(f"実DBに接続できない: {e}")
    yield con
    con.close()


def test_real_raw_path_is_not_truncated(real_con):
    r = pf.check_raw_path_truncation(real_con, getattr(config, 'RAW_PATH_TRUNCATE_LEN', None))
    assert r['status'] == 'pass', r


def test_real_labels_and_features_complete(real_con):
    r = pf.check_labels_and_features(real_con)
    assert r['status'] == 'pass', r


@pytest.mark.xfail(strict=True, reason="既知のデータ品質問題: collection_date='2022/2024' が217件残っている"
                   "（data_quality_collection_date_bug）。DB側を修正/除外したらこのxfailを外すこと。")
def test_real_collection_dates_are_wellformed(real_con):
    r = pf.check_collection_dates(real_con)
    assert r['status'] == 'pass', r


def test_real_cross_split_duplicates_evaluable_after_assignment(real_con):
    r = pf.check_cross_split_duplicates(real_con)
    if 'note' in r:
        pytest.skip("split未割当（assign_wf_splits 後に実行すること）: " + r['note'])
    assert r['status'] in ('pass', 'warn')           # 件数は情報提供（リーク率の把握）。値は検査ログに出る
