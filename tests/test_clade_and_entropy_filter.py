"""未検証だった2フラグのテスト: USE_CLADE_EMBEDDING（クレード埋め込み）と USE_TRAIN_ENTROPY_FILTER（学習サンプルの除外）。

クレード: 系統名→ID写像、OFF時にパラメータが増えない、ID=0/None で現行挙動と一致、IDが違えば出力が変わる、勾配が流れる。
フィルタ: 高エントロピー系統が train だけから除外され、test は影響を受けない。
"""
import pytest
import torch

from transformer import config
from transformer.db.dataset import create_db_dataloader
from transformer.model import HierarchicalTransformer
from transformer.utils.clade import CLADE_NAMES, NUM_CLADES, clade_ids_tensor, lineage_to_clade_id
from tests.conftest import build_synthetic_db
from tests.test_training_smoke import _train_one


@pytest.mark.parametrize('lineage,expected', [
    ('B.1', 'Wuhan'), ('A', 'Wuhan'), ('B.1.1.7', 'Alpha'), ('Q.3', 'Alpha'), ('B.1.351', 'Beta'), ('P.1.2', 'Gamma'),
    ('B.1.617.2', 'Delta'), ('AY.44', 'Delta'), ('BA.1.1', 'BA.1'), ('BA.2.12.1', 'BA.2'),
    ('BA.2.86', 'JN'), ('BA.2.86.1', 'JN'),            # BA.2 より長い規則(BA.2.86)が優先される
    ('BA.5.2', 'BA.4-5'), ('BQ.1', 'BA.4-5'), ('XBB.1.5', 'XBB'), ('JN.1', 'JN'),
])
def test_lineage_to_clade_name(lineage, expected):
    assert CLADE_NAMES[lineage_to_clade_id(lineage)] == expected


@pytest.mark.parametrize('bad', [None, '', '   ', 'ZZ.9', 123])
def test_unknown_lineage_maps_to_zero(bad):
    assert lineage_to_clade_id(bad) == 0


def test_all_ids_in_range_and_tensor_helper():
    ids = clade_ids_tensor(['B.1', 'BA.5.2', 'zzz', None])
    assert ids.dtype == torch.long and ids.tolist() == [lineage_to_clade_id(s) for s in ['B.1', 'BA.5.2', 'zzz', None]]
    assert ids.min() >= 0 and ids.max() < NUM_CLADES


def _batch(cfg):
    T, C, F, N = config.TRAIN_MAX, config.MAX_CO_OCCURRENCE, config.NUM_FEATURE_STRING, config.NUM_CHEM_FEATURES
    g = torch.Generator().manual_seed(3)
    x_cat = torch.zeros(2, T, C, F, dtype=torch.long); x_num = torch.zeros(2, T, C, N)
    mask = torch.ones(2, T, dtype=torch.bool); mask[:, T - 5:] = False
    for b in range(2):
        for t in range(T - 5, T):
            v = torch.randint(0, 2, (F,), generator=g); v[0] = 1; v[1] = 100 + t
            x_cat[b, t, 0] = v; x_num[b, t, 0] = torch.rand(N, generator=g)
    return x_cat, x_num, mask


def _model(cfg, **flags):
    cfg(DEVICE='cpu', USE_SUBSTITUTION_HEAD=False, **flags)
    torch.manual_seed(0)
    return HierarchicalTransformer().eval()


def test_flag_off_adds_no_parameters_and_ignores_clade_ids(cfg):
    m = _model(cfg, USE_CLADE_EMBEDDING=False)
    assert m.clade_embed is None and not any('clade' in k for k in m.state_dict())
    x_cat, x_num, mask = _batch(cfg)
    with torch.no_grad():
        a = m(x_cat, x_num, src_key_padding_mask=mask)
        b = m(x_cat, x_num, src_key_padding_mask=mask, clade_ids=torch.tensor([5, 9]))
    assert all(torch.equal(p, q) for p, q in zip(a, b) if p is not None)


def test_flag_on_unknown_or_none_equals_off_but_known_clade_changes_output(cfg):
    m = _model(cfg, USE_CLADE_EMBEDDING=True)
    assert m.clade_embed is not None and m.clade_embed.weight.shape[0] == NUM_CLADES
    assert torch.count_nonzero(m.clade_embed.weight[0]) == 0                # padding_idx=0 はゼロベクトル
    x_cat, x_num, mask = _batch(cfg)
    with torch.no_grad():
        ref = m(x_cat, x_num, src_key_padding_mask=mask)                    # clade_ids=None
        zero = m(x_cat, x_num, src_key_padding_mask=mask, clade_ids=torch.zeros(2, dtype=torch.long))
        known = m(x_cat, x_num, src_key_padding_mask=mask, clade_ids=torch.tensor([5, 9]))
    assert all(torch.equal(p, q) for p, q in zip(ref, zero) if p is not None)
    assert any(float((p - q).abs().max()) > 1e-6 for p, q in zip(ref, known) if p is not None)
    # サンプルごとに別のクレードを与えると、そのサンプルだけ変わる
    with torch.no_grad():
        mixed = m(x_cat, x_num, src_key_padding_mask=mask, clade_ids=torch.tensor([0, 9]))
    assert torch.allclose(ref[1][0], mixed[1][0], atol=1e-6) and float((ref[1][1] - mixed[1][1]).abs().max()) > 1e-6


def test_clade_embedding_receives_gradient_in_training(synthetic_db, cfg):
    model, *_ = _train_one(synthetic_db, cfg, USE_CLADE_EMBEDDING=True)     # 合成strain 'A','B' は Wuhan(1)
    g = model.clade_embed.weight.grad
    assert g is not None and torch.count_nonzero(g[1]) > 0 and torch.count_nonzero(g[0]) == 0


# ---- エントロピーフィルタ ----------------------------------------------------------
def _train_paths(db, split):
    loader = create_db_dataloader(db, split, batch_size=2, shuffle=False, max_cooccurrence=20, num_workers_override=0)
    return sorted(p for b in loader for p in b.full_paths)


def test_entropy_filter_drops_high_entropy_lineages_from_train_only(synthetic_db, cfg, monkeypatch):
    all_train = _train_paths(synthetic_db, 0)
    all_test = _train_paths(synthetic_db, 2)
    assert len(all_train) == 3
    monkeypatch.setattr('transformer.db.queries.compute_lineage_entropy_map', lambda *a, **k: {'A': 0.9, 'B': 0.1})
    cfg(USE_TRAIN_ENTROPY_FILTER=True, TRAIN_ENTROPY_MAX=0.5)
    # strain 'A'(sample1-3 → G1) が除外され、'B'(G2, G3) だけ残る
    assert _train_paths(synthetic_db, 0) == sorted(['G1T>A2C>C3G', 'C5A>G6T>A7C'])
    assert _train_paths(synthetic_db, 2) == all_test                      # test は不変


def test_entropy_filter_off_or_loose_threshold_keeps_everything(synthetic_db, cfg, monkeypatch):
    base = _train_paths(synthetic_db, 0)
    monkeypatch.setattr('transformer.db.queries.compute_lineage_entropy_map', lambda *a, **k: {'A': 0.9, 'B': 0.1})
    cfg(USE_TRAIN_ENTROPY_FILTER=False, TRAIN_ENTROPY_MAX=0.5)
    assert _train_paths(synthetic_db, 0) == base
    cfg(USE_TRAIN_ENTROPY_FILTER=True, TRAIN_ENTROPY_MAX=0.95)
    assert _train_paths(synthetic_db, 0) == base
    monkeypatch.setattr('transformer.db.queries.compute_lineage_entropy_map', lambda *a, **k: {})   # 未知系統は保持
    cfg(TRAIN_ENTROPY_MAX=0.0)
    assert _train_paths(synthetic_db, 0) == base


def test_lineage_entropy_map_values(tmp_path, cfg):
    from transformer.db.queries import compute_lineage_entropy_map
    # A: 全サンプルが同一位置（エントロピー0）、B: 3位置が等頻度（素のエントロピーは log2(3)=1.585、正規化後は1）
    samples = [
        (1, 1, 'A10T>C20G>T30A', '2021-03-01', 0, [(1, 100, 10, 1, 0)]),
        (2, 1, 'A10T>C20G>T31A', '2021-03-02', 0, [(1, 100, 10, 1, 0)]),
        (3, 2, 'G1T>A2C>C3G',    '2021-04-10', 0, [(2, 400, 40, 2, 0)]),
        (4, 2, 'G1T>A2C>T4A',    '2021-05-01', 0, [(3, 500, 50, 3, 1)]),
        (5, 2, 'G1T>A2C>T5A',    '2021-05-02', 0, [(3, 600, 60, 3, 1)]),
    ]
    db = build_synthetic_db(str(tmp_path / 'e.duckdb'), samples)
    cfg(DB_FILE=db, SPLIT_MODE='walk_forward', STRENGTH_SOURCE='usher', CACHE_DIR=str(tmp_path / 'cache'),
        FORCE_REPROCESS=True)
    emap = compute_lineage_entropy_map(db, split_type=0)
    assert emap['A'] == pytest.approx(0.0) and emap['B'] == pytest.approx(1.0)
