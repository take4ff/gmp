"""A1: DBIterableDataset._build_groups（分岐グルーピング）。

過去バグ: グルーピングがミニバッチ内でしか機能せず、split全体では分断されていた（7c87286）。
_build_groups は split 全体の (sample_id, raw_path) を一括で受けるため、入力順序に
依存せず「同一input_path_strは必ず同一グループ」でなければならない。
"""
import random

from transformer_261008.db.dataset import DBIterableDataset


def _groups(pairs):
    ds = DBIterableDataset.__new__(DBIterableDataset)   # __init__（DBアクセス）を迂回
    return ds._build_groups(pairs)


PAIRS = [
    (1, "A1T>C2G>G3A"),      # 入力履歴 "A1T>C2G"
    (2, "A1T>C2G>T4C"),      # 同上（兄弟）
    (3, "A1T>C2G>T4C,G9A"),  # 同上（兄弟、共起あり）
    (4, "A1T>G5A>T6C"),      # 入力履歴 "A1T>G5A"
    (5, "A1T>G5A>C7T"),
    (6, "X9Z>Y8W"),          # 単独グループ
]


def test_siblings_share_group_and_representative_is_min_id():
    reps, members = _groups(PAIRS)
    assert reps == [1, 4, 6]
    assert sorted(members[1]) == [1, 2, 3]
    assert sorted(members[4]) == [4, 5]
    assert members[6] == [6]


def test_every_sample_in_exactly_one_group():
    reps, members = _groups(PAIRS)
    all_ids = [i for m in members.values() for i in m]
    assert sorted(all_ids) == sorted(i for i, _ in PAIRS)
    assert len(all_ids) == len(set(all_ids))


def test_result_is_independent_of_input_order():
    base_reps, base_members = _groups(PAIRS)
    for seed in range(5):
        shuffled = PAIRS[:]
        random.Random(seed).shuffle(shuffled)
        reps, members = _groups(shuffled)
        assert reps == base_reps
        assert {k: sorted(v) for k, v in members.items()} == \
               {k: sorted(v) for k, v in base_members.items()}


def test_representative_ids_ascending():
    reps, _ = _groups(PAIRS[::-1])
    assert reps == sorted(reps)


def test_single_step_paths_share_empty_input_group():
    """履歴が1ステップのみ（'>'なし）のサンプルは input_path_str='' で1グループになる
    （現行仕様の記録。変更するなら意図的な仕様変更としてこのテストを更新すること）。"""
    reps, members = _groups([(10, "A1T"), (11, "C2G"), (12, "A1T>C2G")])
    assert sorted(members[10]) == [10, 11]
    assert members[12] == [12]
