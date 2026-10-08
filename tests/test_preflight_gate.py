"""A3: 学習起動前ゲート（失敗したら学習を始めない）。"""
import duckdb
import pytest

from transformer_261008.scripts.inspect import preflight_check as pf


def test_evaluate_gate_passes_on_warn_and_raises_on_fail():
    ok = {'a': {'status': 'pass'}, 'b': {'status': 'warn'}, 'overall': 'warn'}
    assert pf.evaluate_gate(ok) is ok
    bad = {'a': {'status': 'pass'}, 'b': {'status': 'fail', 'x': 1}, 'overall': 'fail'}
    with pytest.raises(pf.PreflightFailed, match='b='):
        pf.evaluate_gate(bad)


def test_gate_passes_on_healthy_synthetic_db(synthetic_db):
    res = pf.run_gate(run_tests=False, db_path=synthetic_db)
    assert res['overall'] in ('pass', 'warn')


def test_gate_fails_when_labels_are_missing(synthetic_db):
    con = duckdb.connect(synthetic_db)
    con.execute("DELETE FROM labels WHERE sample_id = 1")
    con.close()
    with pytest.raises(pf.PreflightFailed, match='stage_1_labels_features'):
        pf.run_gate(run_tests=False, db_path=synthetic_db)


def test_gate_fails_when_db_is_missing(tmp_path):
    with pytest.raises(pf.PreflightFailed, match='DBが見つかりません'):
        pf.run_gate(run_tests=False, db_path=str(tmp_path / 'nope.duckdb'))


def test_gate_fails_when_tests_fail(tmp_path):
    """pytestが失敗するリポジトリでは起動を中止する（空のリポジトリ＝テスト収集0件は exit 5）。"""
    with pytest.raises(pf.PreflightFailed, match='pytest が失敗'):
        pf.run_gate(run_tests=True, repo_dir=str(tmp_path), db_path=str(tmp_path / 'x'))


def test_walk_forward_entry_has_skip_gate_option_and_calls_gate():
    import inspect
    from transformer_261008.scripts.eval import walk_forward as wf
    src = inspect.getsource(wf.main)
    assert '--skip_gate' in src and 'run_gate(' in src
