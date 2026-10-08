"""B1: 報告される指標（evaluate / evaluate_topk）を、手計算できる小さな例で検証する。

合成DB(split 0)の3グループ（any-of-set）:
  item0 = G1 代表sample1: 位置 {100,200,210,300}(4種)  region {1,2,3,4}  codon_pos {1,2,3}
  item1 = G2 代表sample4: 位置 {400,500}              region {5,6}
  item2 = G3 代表sample6: 位置 {600}                  region {7}
偽モデルが「各サンプルの予測ランキング」を返すので、期待値は手計算で決まる。
"""
import pytest
import torch

from transformer_261008 import config
from transformer_261008.db.dataset import create_db_dataloader
from transformer_261008.evaluate import evaluate, evaluate_topk
from transformer_261008.utils.losses import build_loss_fn

V = {'region': 37, 'position': 30006, 'aa_pos': 10001, 'codon_pos': 6, 'synonymous': 2}


class FakeModel(torch.nn.Module):
    """ranks[task][i] = サンプルiのスコア降順の予測インデックス列。未指定のtaskは [0,1,2,...]。"""
    def __init__(self, ranks):
        super().__init__()
        self.ranks = ranks
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def _logits(self, task, B):
        out = torch.full((B, V[task]), -50.0)
        for i in range(B):
            order = self.ranks[task][i] if task in self.ranks else list(range(min(6, V[task])))
            for j, idx in enumerate(order):
                out[i, idx] = 100.0 - j
        return out

    def forward(self, x_cat, x_num, src_key_padding_mask=None, clade_ids=None):
        B = x_cat.shape[0]
        self._last_context = torch.zeros(B, 4)
        return (self._logits('region', B), self._logits('position', B), self._logits('aa_pos', B),
                torch.zeros(B), self._logits('codon_pos', B), self._logits('synonymous', B), None, None)


@pytest.fixture
def loader(synthetic_db, cfg):
    cfg(USE_HIERARCHICAL_PREDICTION=False, USE_KNN_OUTPUT=False, USE_TTA=False, USE_R_PRECISION=True,
        TOP_K_EVAL=1, USE_EVAL_STRENGTH_FILTER=False, MAX_R_PRECISION_K=2000)
    return create_db_dataloader(synthetic_db, 0, batch_size=3, shuffle=False, max_cooccurrence=20,
                                num_workers_override=0)


POS_RANKS = [[210, 999, 100, 5, 6], [999, 500, 400, 7, 8], [601, 600, 3, 4, 5]]


def _topk(loader, **kw):
    return evaluate_topk(FakeModel({'position': POS_RANKS}), loader, ks=(1, 3), **kw)


def test_topk_hit_precision_recall_by_hand(loader):
    r = _topk(loader)['position']
    # K=1: 予測{210},{999},{601} → 当たりは item0 のみ
    assert r[1]['hit_rate'] == pytest.approx(100 / 3)
    assert r[1]['precision'] == pytest.approx(100 / 3)          # tp=1 / 予測3
    assert r[1]['recall'] == pytest.approx(100 / 7)             # tp=1 / 正解総数 4+2+1
    # K=3: tp = 2 + 2 + 1 = 5、予測 9、全サンプルがany-of-setで当たり
    assert r[3]['hit_rate'] == pytest.approx(100.0)
    assert r[3]['precision'] == pytest.approx(500 / 9)
    assert r[3]['recall'] == pytest.approx(500 / 7)


def test_r_precision_uses_k_equal_to_number_of_unique_targets(loader):
    r = _topk(loader)['position']['r_precision']
    # item0 K=4: {210,999,100,5} tp=2 / item1 K=2: {999,500} tp=1 / item2 K=1: {601} tp=0
    assert r['precision'] == pytest.approx(300 / 7)             # tp=3 / 予測 4+2+1
    assert r['recall'] == pytest.approx(300 / 7)                # tp=3 / 正解総数7
    assert r['hit_rate'] == pytest.approx(200 / 3)              # item0, item1 が当たり


def test_r_precision_absent_when_disabled(loader, cfg):
    cfg(USE_R_PRECISION=False)
    assert 'r_precision' not in _topk(loader)['position']


def test_position_tolerance_by_hand(loader):
    tol = _topk(loader, position_tolerances=(0, 1, 498, 499))['position_tolerance']
    # top1位置と最も近い正解の距離: item0=0(210が正解), item1=min(|999-400|,|999-500|)=499, item2=|601-600|=1
    assert tol[0]['hit_rate'] == pytest.approx(100 / 3)
    assert tol[1]['hit_rate'] == pytest.approx(200 / 3)
    assert tol[498]['hit_rate'] == pytest.approx(200 / 3)
    assert tol[499]['hit_rate'] == pytest.approx(100.0)
    assert tol[0]['n'] == 3


def test_tolerance_zero_equals_top1_hit_rate(loader):
    out = _topk(loader, position_tolerances=(0,))
    assert out['position_tolerance'][0]['hit_rate'] == pytest.approx(out['position'][1]['hit_rate'])


def _eval_metrics(loader, ranks):
    _, metrics, *_ = evaluate(FakeModel(ranks), loader, build_loss_fn(None), (6.0, 10.0))
    assert set(metrics) == {3}                                   # 3ステップの履歴 → ts_len=3 のみ
    return metrics[3]


def test_evaluate_hit_rates_by_hand_for_top1(loader):
    m = _eval_metrics(loader, {'position': POS_RANKS, 'region': [[3, 1], [9, 1], [7, 1]]})
    assert m['position_hit_rate'] == pytest.approx(100 / 3)      # item0のみ
    assert m['region_hit_rate'] == pytest.approx(200 / 3)        # item0(3∈{1,2,3,4}), item2(7) が当たり
    assert m['num_samples'] == 3


def test_evaluate_top3_is_any_of_set(loader, cfg):
    cfg(TOP_K_EVAL=3)
    m = _eval_metrics(loader, {'position': POS_RANKS})
    assert m['position_hit_rate'] == pytest.approx(100.0)        # 各サンプルのtop3に正解が1つ以上


def test_evaluate_and_topk_agree_on_top1_hit_rate(loader):
    ranks = {'position': POS_RANKS}
    ev = _eval_metrics(loader, ranks)['position_hit_rate']
    tk = evaluate_topk(FakeModel(ranks), loader, ks=(1,))['position'][1]['hit_rate']
    assert ev == pytest.approx(tk)


def test_hierarchical_masking_changes_hit_rate_through_evaluate(loader, cfg, monkeypatch):
    """予測Region上位R個に属さない位置を除外 → top1が変わりhit_rateが上がる（統合確認）。"""
    pr = torch.zeros(config.VOCAB_SIZE_POSITION, dtype=torch.long)
    pr[999] = 9; pr[210] = 3; pr[600] = 7
    monkeypatch.setattr('transformer_261008.db.queries.get_position_region_map', lambda *a, **k: pr)
    ranks = {'position': [[999, 210, 100, 5, 6], [999, 500, 400, 7, 8], [601, 600, 3, 4, 5]],
             'region': [[3, 1], [9, 1], [7, 1]]}
    cfg(USE_HIERARCHICAL_PREDICTION=False)
    off = _eval_metrics(loader, ranks)['position_hit_rate']
    cfg(USE_HIERARCHICAL_PREDICTION=True, HIERARCHICAL_TOPK_REGIONS=1)
    on = _eval_metrics(loader, ranks)['position_hit_rate']
    assert off == pytest.approx(0.0)
    assert on == pytest.approx(200 / 3)           # item0: 999(region9)を除外→210が当たり、item2: 601除外→600が当たり


def test_evaluate_returns_finite_loss(loader):
    loss, *_ = evaluate(FakeModel({'position': POS_RANKS}), loader, build_loss_fn(None), (6.0, 10.0))
    assert loss == loss and loss > 0               # NaNでなく正


def test_per_sample_hits_in_detailed_results_match_aggregated_hit_rate(loader):
    """サンプル別の hit フラグ（detailed_results）と、集計された hit_rate が食い違わない。"""
    _, metrics, detailed, *_ = evaluate(FakeModel({'position': POS_RANKS}), loader, build_loss_fn(None), (6.0, 10.0))
    hits = [d['hit_position'] for d in detailed]
    assert len(hits) == 3 and sum(hits) == 1
    assert 100 * sum(hits) / len(hits) == pytest.approx(metrics[3]['position_hit_rate'])
    assert detailed[0]['targets_position'] == {100, 200, 210, 300}      # any-of-set のターゲット集合
