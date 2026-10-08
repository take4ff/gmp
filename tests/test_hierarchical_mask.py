"""A8: build_allowed_position_mask（階層的予測のRegion→Positionマスキング）。

evaluate()内のインライン処理を切り出した関数。元のインライン実装と同じ結果を返すこと
（リファクタの回帰）と、期待される性質を検証する。
"""
import pytest
import torch

from transformer_260817.evaluate import build_allowed_position_mask


def _inline_original(pred_pos, pred_region, pos_region_map, hier_topk_regions):
    """切り出し前の evaluate() 内の実装（そのまま転記した参照実装）。"""
    R = min(hier_topk_regions, pred_region.size(1))
    topR_regions = torch.topk(pred_region, R, dim=1).indices
    pr = pos_region_map.unsqueeze(0)
    allowed_position_mask = torch.zeros_like(pred_pos, dtype=torch.bool)
    for r in range(R):
        allowed_position_mask |= (pr == topR_regions[:, r:r + 1])
    no_allowed = ~allowed_position_mask.any(dim=1, keepdim=True)
    return allowed_position_mask | no_allowed


def test_matches_original_inline_implementation_on_random_inputs():
    g = torch.Generator().manual_seed(0)
    V, NR, B = 200, 12, 16
    pos_region = torch.randint(0, NR, (V,), generator=g)
    for topk in (1, 3, 5, NR, NR + 10):
        pred_pos = torch.randn(B, V, generator=g)
        pred_reg = torch.randn(B, NR, generator=g)
        got = build_allowed_position_mask(pred_pos, pred_reg, pos_region, topk)
        ref = _inline_original(pred_pos, pred_reg, pos_region, topk)
        assert torch.equal(got, ref)


def test_only_positions_in_top_regions_are_allowed():
    pos_region = torch.tensor([0, 0, 1, 1, 2, 2])
    pred_pos = torch.zeros(1, 6)
    pred_reg = torch.tensor([[0.1, 5.0, 3.0]])           # 上位2 = region 1, 2
    allowed = build_allowed_position_mask(pred_pos, pred_reg, pos_region, 2)
    assert allowed.tolist() == [[False, False, True, True, True, True]]


def test_top1_region_only():
    pos_region = torch.tensor([0, 0, 1, 1, 2, 2])
    pred_reg = torch.tensor([[0.1, 5.0, 3.0]])
    allowed = build_allowed_position_mask(torch.zeros(1, 6), pred_reg, pos_region, 1)
    assert allowed.tolist() == [[False, False, True, True, False, False]]


def test_row_with_no_mapped_position_falls_back_to_all_allowed():
    pos_region = torch.tensor([0, 0, 1, 1])               # region 2 に写像位置なし
    pred_reg = torch.tensor([[0.0, 0.0, 9.0]])           # top1 = region 2
    allowed = build_allowed_position_mask(torch.zeros(1, 4), pred_reg, pos_region, 1)
    assert allowed.all()                                   # NaN回避のためマスク解除


def test_each_row_is_independent():
    pos_region = torch.tensor([0, 1])
    pred_reg = torch.tensor([[9.0, 0.0], [0.0, 9.0]])
    allowed = build_allowed_position_mask(torch.zeros(2, 2), pred_reg, pos_region, 1)
    assert allowed.tolist() == [[True, False], [False, True]]


def test_topk_larger_than_num_regions_is_clamped():
    pos_region = torch.tensor([0, 1, 2])
    allowed = build_allowed_position_mask(torch.zeros(1, 3), torch.randn(1, 3), pos_region, 99)
    assert allowed.all()


def test_masking_does_not_change_region_prediction_and_preserves_allowed_scores():
    """マスクを適用してもRegion予測は無関係で、許可位置のスコアは不変・他は最小値になる。"""
    pos_region = torch.tensor([0, 0, 1, 1])
    pred_pos = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    pred_reg = torch.tensor([[5.0, 1.0]])
    allowed = build_allowed_position_mask(pred_pos, pred_reg, pos_region, 1)
    masked = pred_pos.masked_fill(~allowed, torch.finfo(pred_pos.dtype).min)
    assert masked[0, 0] == 1.0 and masked[0, 1] == 2.0
    assert masked[0, 2] == torch.finfo(pred_pos.dtype).min
    assert int(masked.argmax()) == 1      # 許可位置内の最大へ（元のargmax=3は除外）
