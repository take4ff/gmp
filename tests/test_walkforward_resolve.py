"""A5: _wf_resolve_prev_checkpoint（walk_forwardで引き継ぐ直前foldのcheckpoint解決）。

過去バグ: 事前学習トリガーの判定がループ内位置(idx==0)ベースで、--folds の組み合わせによっては
誤ったcheckpointを引き継いだ・誤って事前学習に落ちた（8d3301f）。
"""
import pytest

from transformer_260817 import main as m


@pytest.fixture
def no_search(monkeypatch):
    calls = []
    def fake(fold_id):
        calls.append(fold_id)
        return None
    monkeypatch.setattr(m, '_wf_find_prev_fold_checkpoint', fake)
    return calls


def test_fold1_has_no_predecessor_even_if_previous_given(no_search, cfg):
    cfg(WF_PREV_CHECKPOINT_OVERRIDE=None)
    assert m._wf_resolve_prev_checkpoint(1, None, None) == (None, False)
    assert m._wf_resolve_prev_checkpoint(1, 7, '/x/best.pth') == (None, False)
    assert no_search == []


def test_consecutive_run_uses_in_loop_checkpoint(no_search, cfg):
    cfg(WF_PREV_CHECKPOINT_OVERRIDE=None)
    assert m._wf_resolve_prev_checkpoint(3, 2, '/runs/fold2/best.pth') == ('/runs/fold2/best.pth', False)
    assert no_search == []        # 探索しない


def test_noncontiguous_prefers_override_over_search(monkeypatch, cfg):
    cfg(WF_PREV_CHECKPOINT_OVERRIDE='/explicit/best.pth')
    monkeypatch.setattr(m, '_wf_find_prev_fold_checkpoint', lambda f: '/searched/best.pth')
    assert m._wf_resolve_prev_checkpoint(6, 2, '/runs/fold2/best.pth') == ('/explicit/best.pth', True)


def test_noncontiguous_falls_back_to_search(monkeypatch, cfg):
    cfg(WF_PREV_CHECKPOINT_OVERRIDE=None)
    seen = []
    monkeypatch.setattr(m, '_wf_find_prev_fold_checkpoint',
                        lambda f: seen.append(f) or '/searched/best.pth')
    assert m._wf_resolve_prev_checkpoint(6, 2, '/runs/fold2/best.pth') == ('/searched/best.pth', True)
    assert seen == [6]


def test_single_fold_run_without_previous_uses_search(monkeypatch, cfg):
    """--folds 2 のように先頭が fold 1 でない単独実行（prev_fold_id=None）。"""
    cfg(WF_PREV_CHECKPOINT_OVERRIDE=None)
    monkeypatch.setattr(m, '_wf_find_prev_fold_checkpoint', lambda f: '/searched/best.pth')
    assert m._wf_resolve_prev_checkpoint(2, None, None) == ('/searched/best.pth', True)


def test_raises_when_unresolvable_rather_than_silently_pretraining(no_search, cfg):
    cfg(WF_PREV_CHECKPOINT_OVERRIDE=None)
    with pytest.raises(RuntimeError, match='fold_1'):
        m._wf_resolve_prev_checkpoint(2, None, None)
    with pytest.raises(RuntimeError, match='fold_5'):
        m._wf_resolve_prev_checkpoint(6, 2, '/runs/fold2/best.pth')
