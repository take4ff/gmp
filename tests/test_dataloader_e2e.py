"""B3: 合成DBでの DataLoader エンドツーエンド（実際のSQL・グルーピング・collateを通す）。

検証: 全グループがちょうど1回出る（取りこぼし・重複なし）、num_workers=0/2で同一内容、
group_targets_list が代表だけに偏らず全メンバーのラベルを含む、split間で混ざらない。
"""
import pytest

from transformer_260817.db.dataset import create_db_dataloader


def _collect(db, split, workers):
    loader = create_db_dataloader(db, split, batch_size=2, shuffle=False,
                                  max_cooccurrence=20, num_workers_override=workers)
    items = []
    for batch in loader:
        for path, raw_y in zip(batch.full_paths, batch.raw_y):
            items.append((path, sorted(t[1] for t in raw_y)))
    return items


def test_train_each_group_exactly_once_with_all_member_targets(synthetic_db):
    items = _collect(synthetic_db, 0, 0)
    paths = [p for p, _ in items]
    # 代表 = 各グループの min(sample_id): G1→sample1, G2→sample4, G3→sample6
    assert sorted(paths) == sorted(['A10T>C20G>T30A', 'G1T>A2C>C3G', 'C5A>G6T>A7C'])
    assert len(paths) == len(set(paths))                     # 重複なし
    targets = dict(items)
    # any-of-set: 代表だけでなくグループ全メンバーのターゲット位置を含む
    assert targets['A10T>C20G>T30A'] == [100, 200, 210, 300]   # sample1,2,3
    assert targets['G1T>A2C>C3G'] == [400, 500]                # sample4,5
    assert targets['C5A>G6T>A7C'] == [600]


def test_test_split_contains_only_test_samples(synthetic_db):
    items = _collect(synthetic_db, 2, 0)
    assert sorted(p for p, _ in items) == sorted(['G1T>A2C>C3G', 'T9C>A8G>C7T'])


def test_train_and_test_splits_are_disjoint_by_sample(synthetic_db):
    tr = {p for p, _ in _collect(synthetic_db, 0, 0)}
    te = {p for p, _ in _collect(synthetic_db, 2, 0)}
    assert 'T9C>A8G>C7T' not in tr and 'C5A>G6T>A7C' not in te
    # 'G1T>A2C>C3G' は train(4)・test(7) の完全一致（B2が数えるリーク候補）で両方に現れる
    assert 'G1T>A2C>C3G' in tr and 'G1T>A2C>C3G' in te


def test_workers_do_not_change_content(synthetic_db):
    base = sorted(_collect(synthetic_db, 0, 0))
    assert sorted(_collect(synthetic_db, 0, 2)) == base      # worker分割で取りこぼし/重複が出ない


def test_batch_tensor_shapes_and_padding_are_left_aligned(synthetic_db):
    from transformer_260817 import config
    loader = create_db_dataloader(synthetic_db, 0, batch_size=3, shuffle=False,
                                  max_cooccurrence=20, num_workers_override=0)
    batch = next(iter(loader))
    x_cat, x_num, mask = batch.inputs
    assert x_cat.shape == (3, config.TRAIN_MAX, config.MAX_CO_OCCURRENCE, config.NUM_FEATURE_STRING)
    assert x_num.shape[-1] == config.NUM_CHEM_FEATURES
    # 左パディング: 末尾タイムステップは必ず有効(mask=False)、先頭はPAD(mask=True)
    assert not bool(mask[:, -1].any())
    assert bool(mask[:, 0].all())
    for i, ln in enumerate(batch.lens):
        assert int((~mask[i]).sum()) == ln - 1               # 入力ステップ数 = path_length - TARGET_LEN


def test_soft_target_sums_to_one_per_item(synthetic_db, cfg):
    cfg(HYBRID_ALPHA=1.0)
    loader = create_db_dataloader(synthetic_db, 0, batch_size=3, shuffle=False,
                                  max_cooccurrence=20, num_workers_override=0)
    for y in next(iter(loader)).y:
        for task in ('region', 'position', 'aa_pos', 'codon_pos', 'synonymous'):
            assert float(y[task].sum()) == pytest.approx(1.0)
