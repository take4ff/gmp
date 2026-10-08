"""A11: kNN検索拡張出力（utils/knn_output.py）。

過去の問題: build_datastore が Soft Target(dict) のラベルを読めず常に空になっていた（2026-10-06修正）。
"""
import numpy as np
import pytest
import torch

from transformer_261008 import config
from transformer_261008.utils import knn_output as ko
from transformer_261008.utils.knn_output import KNNOutput, build_datastore, labels_to_sparse

V = 10


def _store(keys, labels, use_faiss=True):
    """labels: list of {pos: weight}"""
    indptr, pos, w = [0], [], []
    for lab in labels:
        for p, x in lab.items():
            pos.append(p); w.append(x)
        indptr.append(len(pos))
    ds = KNNOutput(np.array(keys, dtype=np.float32), np.array(indptr, dtype=np.int64),
                   np.array(pos, dtype=np.int64), np.array(w, dtype=np.float32), V)
    if not use_faiss:
        ds.index = None
    return ds


# ---- labels_to_sparse -------------------------------------------------------
def test_labels_to_sparse_soft_target_dict():
    vec = torch.zeros(V); vec[2] = 0.5; vec[7] = 0.25; vec[8] = 0.25
    pos, w = labels_to_sparse({'position': vec}, V)
    assert pos.tolist() == [2, 7, 8]
    assert w.sum() == pytest.approx(1.0)
    assert w[0] == pytest.approx(0.5)


def test_labels_to_sparse_hard_tuples_uniform_over_unique_and_counts():
    tuples = [(0, 3, 0, 0, 0), (0, 3, 0, 0, 0), (0, 5, 0, 0, 0)]
    pos, w = labels_to_sparse(tuples, V)
    assert pos.tolist() == [3, 5]
    assert w.tolist() == pytest.approx([2 / 3, 1 / 3])


def test_labels_to_sparse_ignores_out_of_range_and_handles_empty():
    pos, w = labels_to_sparse([(0, V + 5, 0, 0, 0), (0, -1, 0, 0, 0)], V)
    assert len(pos) == 0 and len(w) == 0
    pos, w = labels_to_sparse([], V)
    assert len(pos) == 0
    pos, w = labels_to_sparse({'position': torch.zeros(V)}, V)
    assert len(pos) == 0


# ---- KNNOutput --------------------------------------------------------------
@pytest.mark.parametrize('use_faiss', [True, False])
def test_neighbors_and_distribution(cfg, use_faiss):
    if use_faiss and not ko._HAS_FAISS:
        pytest.skip('faiss not installed')
    cfg(KNN_K=2)
    keys = [[0, 0], [0, 1], [10, 10], [10, 11]]
    labels = [{1: 1.0}, {1: 0.5, 2: 0.5}, {7: 1.0}, {8: 1.0}]
    ds = _store(keys, labels, use_faiss)
    p = ds.knn_distribution(np.array([[0.0, 0.2]], dtype=np.float32))   # 近傍 = key0, key1
    assert p.shape == (1, V)
    assert float(p[0, 1]) == pytest.approx((1.0 + 0.5) / 2)
    assert float(p[0, 2]) == pytest.approx(0.25)
    assert float(p.sum()) == pytest.approx(1.0)
    p2 = ds.knn_distribution(np.array([[10.0, 10.4]], dtype=np.float32))  # 近傍 = key2, key3
    assert float(p2[0, 7]) == pytest.approx(0.5) and float(p2[0, 8]) == pytest.approx(0.5)


def test_faiss_and_torch_fallback_agree(cfg):
    if not ko._HAS_FAISS:
        pytest.skip('faiss not installed')
    cfg(KNN_K=3)
    rng = np.random.default_rng(0)
    keys = rng.normal(size=(50, 8)).astype(np.float32)
    labels = [{int(rng.integers(0, V)): 1.0} for _ in range(50)]
    a = _store(keys, labels, True)
    b = _store(keys, labels, False)
    q = rng.normal(size=(7, 8)).astype(np.float32)
    assert torch.allclose(a.knn_distribution(q), b.knn_distribution(q))


def test_blend_lambda_zero_equals_model_softmax_and_one_equals_knn(cfg):
    ds = _store([[0, 0], [1, 1]], [{3: 1.0}, {4: 1.0}])
    scores = torch.randn(2, V)
    q = np.array([[0, 0], [1, 1]], dtype=np.float32)
    cfg(KNN_K=1, KNN_LAMBDA=0.0)
    assert torch.allclose(ds.blend(q, scores), torch.softmax(scores, -1), atol=1e-6)
    cfg(KNN_LAMBDA=1.0)
    p = ds.blend(q, scores)
    assert float(p[0, 3]) == pytest.approx(1.0) and float(p[1, 4]) == pytest.approx(1.0)


def test_blend_respects_allowed_mask(cfg):
    cfg(KNN_K=1, KNN_LAMBDA=0.5)
    ds = _store([[0, 0]], [{3: 1.0}])
    scores = torch.zeros(1, V)
    allowed = torch.ones(1, V, dtype=torch.bool); allowed[0, 3] = False   # kNN が指す位置を禁止
    p = ds.blend(np.array([[0, 0]], dtype=np.float32), scores, allowed)
    assert float(p[0, 3]) == pytest.approx(0.5 * 0.0 + 0.5 * (1 / V))    # kNN成分は0、モデル成分のみ


def test_blend_preserves_probability_mass_without_mask(cfg):
    cfg(KNN_K=2, KNN_LAMBDA=0.3)
    ds = _store([[0, 0], [0, 1]], [{1: 1.0}, {2: 1.0}])
    p = ds.blend(np.zeros((1, 2), dtype=np.float32), torch.randn(1, V))
    assert float(p.sum()) == pytest.approx(1.0, abs=1e-5)


def test_save_load_roundtrip(tmp_path, cfg):
    cfg(KNN_DATASTORE_PATH=str(tmp_path / 'ds'), KNN_K=1)
    ds = _store([[0, 0], [5, 5]], [{1: 1.0}, {2: 0.5, 3: 0.5}])
    ds.save()
    loaded = KNNOutput.load(V)
    assert loaded is not None and len(loaded) == 2
    assert np.array_equal(loaded.indptr, ds.indptr)
    assert np.array_equal(loaded.positions, ds.positions)
    p = loaded.knn_distribution(np.array([[5.1, 5.0]], dtype=np.float32))
    assert float(p[0, 2]) == pytest.approx(0.5)


def test_load_returns_none_when_missing_or_empty(tmp_path, cfg):
    cfg(KNN_DATASTORE_PATH=str(tmp_path / 'nope'))
    assert KNNOutput.load(V) is None
    cfg(KNN_DATASTORE_PATH=str(tmp_path / 'empty'))
    KNNOutput(np.zeros((0, 2), np.float32), np.zeros(1, np.int64), np.zeros(0, np.int64),
              np.zeros(0, np.float32), V).save()
    assert KNNOutput.load(V) is None


# ---- build_datastore --------------------------------------------------------
class _FakeModel(torch.nn.Module):
    def forward(self, x_cat, x_num, src_key_padding_mask=None):
        self._last_context = x_num[:, 0, 0, :2].clone()      # [B, 2] をキーとして公開
        return None


def _batch(keys, ys):
    B = len(keys)
    x_num = torch.zeros(B, 1, 1, 2)
    x_num[:, 0, 0, :] = torch.tensor(keys, dtype=torch.float)
    return ((torch.zeros(B, 1, 1, 1, dtype=torch.long), x_num, torch.zeros(B, 1, dtype=torch.bool)), ys)


def test_build_datastore_handles_soft_and_hard_labels_and_skips_empty(cfg):
    cfg(VOCAB_SIZE_POSITION=V, FEATURE_DIM=2)
    soft = torch.zeros(V); soft[4] = 1.0
    ys = [{'position': soft}, [(0, 6, 0, 0, 0)], {'position': torch.zeros(V)}]   # 3つ目はラベル無し
    store = build_datastore(_FakeModel(), [_batch([[0, 0], [1, 1], [2, 2]], ys)], 'cpu')
    assert len(store) == 2                       # 空ラベルは捨てる
    assert store.positions.tolist() == [4, 6]
    assert store.indptr.tolist() == [0, 1, 2]


def test_build_datastore_max_groups_and_empty_result(cfg):
    cfg(VOCAB_SIZE_POSITION=V, FEATURE_DIM=2)
    soft = torch.zeros(V); soft[1] = 1.0
    batches = [_batch([[i, i] for i in range(4)], [{'position': soft}] * 4) for _ in range(3)]
    assert len(build_datastore(_FakeModel(), batches, 'cpu', max_groups=5)) == 5
    empty = build_datastore(_FakeModel(), [_batch([[0, 0]], [{'position': torch.zeros(V)}])], 'cpu')
    assert len(empty) == 0
