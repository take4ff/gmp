"""A4: 主貢献（Co-occurrence Attention）の性質テスト。

同一タイムステップ内の共起変異は順序なしの集合として扱われる（スロット順・配置に依らない）。
一方でタイムステップ間の順序には敏感でなければならない。PAD領域の中身は出力に影響しない。
"""
import pytest
import torch

from transformer import config
from transformer.model import HierarchicalTransformer

B, ACTIVE = 3, (6, 7, 8)             # サンプルごとの有効タイムステップ数


@pytest.fixture
def model_and_batch(cfg):
    cfg(DEVICE='cpu', USE_SUBSTITUTION_HEAD=False)
    torch.manual_seed(0)
    model = HierarchicalTransformer().eval()
    T, C, F, N = config.TRAIN_MAX, config.MAX_CO_OCCURRENCE, config.NUM_FEATURE_STRING, config.NUM_CHEM_FEATURES
    g = torch.Generator().manual_seed(1)
    x_cat = torch.zeros(B, T, C, F, dtype=torch.long)
    x_num = torch.zeros(B, T, C, N)
    mask = torch.ones(B, T, dtype=torch.bool)
    for b, L in enumerate(ACTIVE):
        mask[b, T - L:] = False
        for t in range(T - L, T):
            for c in range(int(torch.randint(1, 6, (1,), generator=g))):
                v = torch.randint(0, 2, (F,), generator=g)
                v[0] = 1 + int(torch.randint(0, 4, (1,), generator=g))       # base_before != 0（=非PADスロット）
                v[1] = int(torch.randint(1, 30000, (1,), generator=g))       # 位置
                x_cat[b, t, c] = v
                x_num[b, t, c] = torch.rand(N, generator=g)
    return model, x_cat, x_num, mask, g


def _run(model, x_cat, x_num, mask):
    with torch.no_grad():
        return model(x_cat, x_num, src_key_padding_mask=mask)


def _scatter_slots(x_cat, x_num, g):
    """各タイムステップの有効スロットを、任意の順序・任意のスロット位置へ並べ替える。"""
    xc, xn = torch.zeros_like(x_cat), torch.zeros_like(x_num)
    C = x_cat.shape[2]
    for b in range(x_cat.shape[0]):
        for t in range(x_cat.shape[1]):
            valid = (x_cat[b, t, :, 0] != 0).nonzero().flatten()
            if len(valid) == 0:
                continue
            dest = torch.randperm(C, generator=g)[:len(valid)]
            perm = torch.randperm(len(valid), generator=g)
            xc[b, t, dest] = x_cat[b, t, valid[perm]]
            xn[b, t, dest] = x_num[b, t, valid[perm]]
    return xc, xn


def _assert_close(ref, out, atol=1e-5):
    for a, b in zip(ref, out):
        if a is None:
            assert b is None
        else:
            assert torch.allclose(a, b, atol=atol), float((a - b).abs().max())


def test_output_is_invariant_to_cooccurring_mutation_order_and_slot_placement(model_and_batch):
    model, x_cat, x_num, mask, g = model_and_batch
    xc2, xn2 = _scatter_slots(x_cat, x_num, g)
    assert not torch.equal(xc2, x_cat)                          # 実際に並べ替わっている
    _assert_close(_run(model, x_cat, x_num, mask), _run(model, xc2, xn2, mask))


@pytest.mark.parametrize('flag', ['USE_BROADCAST_BACK_ATTENTION', 'USE_REGION_CONDITIONED_POSITION'])
def test_invariance_also_holds_with_optional_modules_enabled(cfg, flag):
    cfg(DEVICE='cpu', USE_SUBSTITUTION_HEAD=False, **{flag: True})
    torch.manual_seed(0)
    model = HierarchicalTransformer().eval()
    T, C, F, N = config.TRAIN_MAX, config.MAX_CO_OCCURRENCE, config.NUM_FEATURE_STRING, config.NUM_CHEM_FEATURES
    g = torch.Generator().manual_seed(2)
    x_cat = torch.zeros(2, T, C, F, dtype=torch.long); x_num = torch.zeros(2, T, C, N)
    mask = torch.ones(2, T, dtype=torch.bool); mask[:, T - 5:] = False
    for b in range(2):
        for t in range(T - 5, T):
            for c in range(3):
                v = torch.randint(0, 2, (F,), generator=g); v[0] = 1 + c; v[1] = 100 + 10 * c + t
                x_cat[b, t, c] = v; x_num[b, t, c] = torch.rand(N, generator=g)
    xc2, xn2 = _scatter_slots(x_cat, x_num, g)
    _assert_close(_run(model, x_cat, x_num, mask), _run(model, xc2, xn2, mask))


def test_output_depends_on_timestep_order(model_and_batch):
    model, x_cat, x_num, mask, _ = model_and_batch
    ref = _run(model, x_cat, x_num, mask)
    T, L = config.TRAIN_MAX, ACTIVE[2]
    idx = list(range(T - L, T))
    xc2, xn2 = x_cat.clone(), x_num.clone()
    xc2[2, idx], xn2[2, idx] = x_cat[2, idx[::-1]], x_num[2, idx[::-1]]
    out = _run(model, xc2, xn2, mask)
    assert float((ref[1][2] - out[1][2]).abs().max()) > 1e-3     # 時系列は順序依存
    assert torch.allclose(ref[1][0], out[1][0], atol=1e-5)         # 他サンプルは影響を受けない


def test_pad_timestep_contents_do_not_affect_output(model_and_batch):
    model, x_cat, x_num, mask, g = model_and_batch
    ref = _run(model, x_cat, x_num, mask)
    xc2, xn2 = x_cat.clone(), x_num.clone()
    pad = mask.nonzero(as_tuple=True)                          # (b, t) of PAD timesteps
    xn2[pad[0], pad[1]] = torch.rand_like(xn2[pad[0], pad[1]])  # PAD領域にノイズを入れる
    _assert_close(ref, _run(model, xc2, xn2, mask))


def test_output_is_unchanged_when_unused_pad_slots_are_filled_with_noise_numerics(model_and_batch):
    """PADスロット(base_before==0)の数値特徴は集約から除外される（co_occur_mask）。"""
    model, x_cat, x_num, mask, g = model_and_batch
    ref = _run(model, x_cat, x_num, mask)
    xn2 = x_num.clone()
    is_pad_slot = (x_cat[..., 0] == 0)
    xn2[is_pad_slot] = torch.rand_like(xn2[is_pad_slot])
    _assert_close(ref, _run(model, x_cat, xn2, mask))
