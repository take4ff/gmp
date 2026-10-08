"""A2: MAX_GROUP_MEMBERS_FOR_CACHE による巨大グループの間引き（OOM対策、2026-08-03）。"""
from transformer_261008.db.dataset import DBIterableDataset


def _cap(repr_id, members):
    ds = DBIterableDataset.__new__(DBIterableDataset)
    return ds._capped_member_ids(repr_id, members)


def test_identity_when_below_or_equal_cap(cfg):
    cfg(MAX_GROUP_MEMBERS_FOR_CACHE=5)
    members = [3, 4, 5, 6, 7]
    assert _cap(3, members) == members


def test_cap_none_disables(cfg):
    cfg(MAX_GROUP_MEMBERS_FOR_CACHE=None)
    members = list(range(1000))
    assert _cap(0, members) == members


def test_oversized_group_truncated_and_keeps_representative(cfg):
    cfg(MAX_GROUP_MEMBERS_FOR_CACHE=10)
    members = list(range(100, 300))
    out = _cap(100, members)
    assert len(out) == 10
    assert out[0] == 100                       # 代表が必ず先頭に残る
    assert len(set(out)) == 10                 # 重複なし
    assert set(out) <= set(members)


def test_deterministic_for_same_representative(cfg):
    cfg(MAX_GROUP_MEMBERS_FOR_CACHE=10)
    members = list(range(100, 300))
    assert _cap(100, members) == _cap(100, members)
    assert _cap(100, members) == _cap(100, list(members))


def test_representative_not_required_to_be_min(cfg):
    cfg(MAX_GROUP_MEMBERS_FOR_CACHE=4)
    members = list(range(50))
    out = _cap(7, members)
    assert out[0] == 7 and len(out) == 4 and out.count(7) == 1
