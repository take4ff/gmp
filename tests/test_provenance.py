"""A2: 実行の来歴（git commit・dirty・差分・環境）の保存。"""
import json
import os
import subprocess

import pytest

from transformer_261008.utils import provenance as pv


def _git(cwd, *args):
    env = dict(os.environ, GIT_AUTHOR_NAME='t', GIT_AUTHOR_EMAIL='t@t', GIT_COMMITTER_NAME='t',
               GIT_COMMITTER_EMAIL='t@t')
    return subprocess.run(['git'] + list(args), cwd=cwd, capture_output=True, text=True, check=True,
                          env=env).stdout.strip()


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / 'repo'
    r.mkdir()
    _git(r, 'init', '-q')
    (r / 'a.py').write_text('x = 1\n')
    _git(r, 'add', 'a.py')
    _git(r, 'commit', '-q', '-m', 'init')
    return r


def test_clean_repo_records_commit_without_patch(repo, tmp_path):
    out = tmp_path / 'out'
    info = pv.save_run_provenance(str(out), repo_dir=str(repo), db_path='/nonexistent')
    saved = json.loads((out / 'provenance.json').read_text())
    assert saved['git']['commit'] == _git(repo, 'rev-parse', 'HEAD')
    assert saved['git']['dirty'] is False and not (out / 'code_changes.patch').exists()
    assert info['git']['branch'] == _git(repo, 'rev-parse', '--abbrev-ref', 'HEAD')


def test_dirty_repo_saves_patch_and_lists_untracked(repo, tmp_path):
    (repo / 'a.py').write_text('x = 2\n')            # 追跡ファイルの変更
    (repo / 'new.py').write_text('y = 1\n')          # 未追跡
    out = tmp_path / 'out'
    pv.save_run_provenance(str(out), repo_dir=str(repo), db_path='/nonexistent')
    saved = json.loads((out / 'provenance.json').read_text())
    assert saved['git']['dirty'] is True
    assert saved['git']['modified'] == ['a.py'] and saved['git']['untracked'] == ['new.py']
    patch = (out / 'code_changes.patch').read_text()
    assert '-x = 1' in patch and '+x = 2' in patch


def test_large_patch_is_skipped_but_recorded(repo, tmp_path):
    (repo / 'a.py').write_text('z = 1\n' * 1000)
    out = tmp_path / 'out'
    pv.save_run_provenance(str(out), repo_dir=str(repo), db_path='/nonexistent', max_patch_bytes=100)
    saved = json.loads((out / 'provenance.json').read_text())
    assert saved['git']['patch_file'].startswith('(skipped') and not (out / 'code_changes.patch').exists()


def test_non_git_directory_does_not_raise(tmp_path):
    out = tmp_path / 'out'
    pv.save_run_provenance(str(out), repo_dir=str(tmp_path), db_path='/nonexistent')
    saved = json.loads((out / 'provenance.json').read_text())
    assert 'error' in saved['git'] and (out / 'provenance.json').exists()


def test_records_environment_and_db_fingerprint(repo, tmp_path):
    db = tmp_path / 'f.duckdb'
    db.write_bytes(b'x' * 10)
    info = pv.collect_provenance(str(repo), str(db))
    assert info['torch'] and info['numpy'] and info['python']
    assert info['db']['size_bytes'] == 10 and info['db']['path'] == str(db)


def test_failure_is_swallowed(monkeypatch, tmp_path):
    monkeypatch.setattr(pv, 'collect_provenance', lambda *a, **k: (_ for _ in ()).throw(RuntimeError('boom')))
    assert pv.save_run_provenance(str(tmp_path / 'o')) is None      # 学習を止めない
