"""A9: ホモプラシーバイアス（log(1+再発回数)）の構築。"""
import math

import pytest

from transformer import config
from transformer.model import HierarchicalTransformer


def _load(csv_path, cfg):
    cfg(HOMOPLASY_CSV=str(csv_path))
    # _load_homoplasy_bias は self を使わないため、モデルを構築せず呼べる
    return HierarchicalTransformer._load_homoplasy_bias(None, 'USE_HOMOPLASY_PRIOR')


def test_log1p_of_recurrence(tmp_path, cfg):
    p = tmp_path / 'h.csv'
    p.write_text('position_id,recurrence_count\n5,0\n10,9\n20,99\n')
    bias = _load(p, cfg)
    assert bias.shape[0] == config.VOCAB_SIZE_POSITION
    assert float(bias[5]) == pytest.approx(0.0)
    assert float(bias[10]) == pytest.approx(math.log(10))
    assert float(bias[20]) == pytest.approx(math.log(100))
    assert float(bias[0]) == 0.0      # CSVに無い位置は0


def test_out_of_range_position_ignored(tmp_path, cfg):
    p = tmp_path / 'h.csv'
    p.write_text(f'position_id,recurrence_count\n-1,5\n{config.VOCAB_SIZE_POSITION},5\n3,5\n')
    bias = _load(p, cfg)
    assert float(bias[3]) == pytest.approx(math.log(6))
    assert int((bias != 0).sum()) == 1


def test_negative_recurrence_clamped_to_zero(tmp_path, cfg):
    p = tmp_path / 'h.csv'
    p.write_text('position_id,recurrence_count\n7,-5\n')
    assert float(_load(p, cfg)[7]) == 0.0


def test_missing_csv_returns_none_when_lenient(tmp_path, cfg):
    cfg(STRICT_FALLBACKS=False)                      # 厳格時の挙動は tests/test_strict_fallbacks.py
    assert _load(tmp_path / 'nope.csv', cfg) is None
