"""A6: _wf_find_prev_fold_checkpoint（部分/非連続fold実行での直前foldのcheckpoint探索）。
過去バグ: fold間checkpoint引き継ぎの取り違え（8d3301f）。"""
import os
import time

from transformer.main import _wf_find_prev_fold_checkpoint


def _make(root, run, fold, mtime):
    d = root / 'walk_forward' / run / f'fold_{fold}' / '20260101_000000' / 'models'
    d.mkdir(parents=True)
    p = d / 'best_model.pth'
    p.write_bytes(b'x')
    os.utime(p, (mtime, mtime))
    return str(p)


def test_returns_none_when_not_found(tmp_path, cfg):
    cfg(RESULT_SAVE_DIR=str(tmp_path))
    assert _wf_find_prev_fold_checkpoint(3) is None


def test_looks_up_fold_minus_one(tmp_path, cfg):
    cfg(RESULT_SAVE_DIR=str(tmp_path))
    now = time.time()
    p2 = _make(tmp_path, 'runA', 2, now)
    _make(tmp_path, 'runA', 3, now)      # 自分自身(fold3)は対象外
    assert _wf_find_prev_fold_checkpoint(3) == p2
    assert _wf_find_prev_fold_checkpoint(4) is not None   # fold3 が直前
    assert _wf_find_prev_fold_checkpoint(2) is None       # fold1 は無い


def test_picks_most_recently_modified_among_runs(tmp_path, cfg):
    cfg(RESULT_SAVE_DIR=str(tmp_path))
    now = time.time()
    _make(tmp_path, 'old', 2, now - 1000)
    newest = _make(tmp_path, 'new', 2, now)
    _make(tmp_path, 'mid', 2, now - 500)
    assert _wf_find_prev_fold_checkpoint(3) == newest
