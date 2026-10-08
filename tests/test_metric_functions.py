"""B1: 指標の集計関数（utils/logging.py）を手計算で検証する。報告される Hit Rate / Precision / Recall / F1 の定義。"""
import pytest

from transformer_261008.utils.logging import calculate_metrics, calculate_weighted_macro_recall


def test_calculate_metrics_by_hand():
    pred = [{1}, {2, 3}, {4}]
    target = [{1, 9}, {3}, {5}]
    hit, prec, rec, f1 = calculate_metrics(pred, target)
    # sample0: tp=1 P=1 R=1/2 F1=2/3 / sample1: tp=1 P=1/2 R=1 F1=2/3 / sample2: tp=0
    assert hit == pytest.approx(200 / 3)                 # any-of-set: 3サンプル中2サンプルが当たり
    assert prec == pytest.approx(50.0)                   # (1 + 0.5 + 0) / 3
    assert rec == pytest.approx(50.0)                    # (0.5 + 1 + 0) / 3
    assert f1 == pytest.approx(400 / 9)                  # (2/3 + 2/3 + 0) / 3


def test_samples_without_targets_are_excluded_from_the_denominator():
    hit, prec, rec, f1 = calculate_metrics([{1}, {2}], [set(), {2}])
    assert hit == pytest.approx(100.0) and prec == pytest.approx(100.0)


def test_empty_inputs_return_zeros():
    assert calculate_metrics([], []) == (0.0, 0.0, 0.0, 0.0)
    assert calculate_metrics([{1}], [set()]) == (0.0, 0.0, 0.0, 0.0)


def test_perfect_and_zero_predictions():
    assert calculate_metrics([{1}, {2}], [{1}, {2}]) == (100.0, 100.0, 100.0, 100.0)
    assert calculate_metrics([{3}, {4}], [{1}, {2}]) == (0.0, 0.0, 0.0, 0.0)


def test_weighted_and_macro_recall_by_hand():
    # クラス1: 出現1・当たり1、クラス2: 出現1・当たり0、クラス3: 出現1・当たり0
    w, m = calculate_weighted_macro_recall([{1}, {5}], [{1, 2}, {3}])
    assert w == pytest.approx(100 / 3) and m == pytest.approx(100 / 3)


def test_weighted_differs_from_macro_when_classes_are_imbalanced():
    # クラス1: 出現2・当たり2、クラス2: 出現1・当たり0
    w, m = calculate_weighted_macro_recall([{1}, {1}, {1}], [{1}, {1}, {2}])
    assert w == pytest.approx(200 / 3)                   # 頻度加重: 2/3
    assert m == pytest.approx(50.0)                      # クラス平均: (1 + 0) / 2


def test_weighted_macro_recall_empty():
    assert calculate_weighted_macro_recall([], []) == (0.0, 0.0)
