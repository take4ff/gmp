"""層C: 実DBに対するプリフライト検査（低速・読み取り専用）。実行: python -m pytest -m slow

学習/評価プロセスが同じDBを開いている間は接続できないため、その場合は skip する。
"""
import duckdb
import pytest

from transformer import config
from transformer.db.connection import connect_db, get_db_path
from transformer.db.queries import valid_date_sql
from transformer.scripts.inspect import preflight_check as pf

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


def test_real_malformed_dates_exist_but_are_excluded_from_splits(real_con):
    """実DBには不正形式（'2022/2024' 217件）が残るが、割当側のガード(valid_date_sql)で除外される。
    データ自体は未修正のためwarn。割当に使われる述語で拾われる行が0であることを確認する。"""
    r = pf.check_collection_dates(real_con)
    assert r['status'] in ('pass', 'warn'), r
    n_guarded = real_con.execute(
        f"SELECT COUNT(*) FROM samples WHERE collection_date IS NOT NULL AND collection_date != '' "
        f"AND NOT {valid_date_sql()}").fetchone()[0]
    assert n_guarded == r['malformed']


def test_real_cross_split_duplicates_evaluable_after_assignment(real_con):
    r = pf.check_cross_split_duplicates(real_con)
    if 'note' in r:
        pytest.skip("split未割当（assign_wf_splits 後に実行すること）: " + r['note'])
    assert r['status'] in ('pass', 'warn')           # 件数は情報提供（リーク率の把握）。値は検査ログに出る
